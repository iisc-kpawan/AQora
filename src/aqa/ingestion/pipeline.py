from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from ..config import load_config
from ..errors import IngestionError
from .connectors import fetch_cnemc, fetch_cpcb_year
from .impute import impute_new_data
from .parsers import parse_cnemc, parse_cpcb
from .schema import begin_run, connect_control
from .storage import write_partitions
from ..retention_refresh import append_and_refresh_lakehouse


PROCESSING_VERSION = "aqa-v2-1"


class IngestionPipeline:
    def __init__(self, config: dict | None = None):
        self.config = config or load_config()

    def ingest_files(self, dataset: str, source: str, paths: list[str | Path], *, start: datetime, end: datetime, dry_run: bool = False) -> dict:
        control_path = Path(self.config["project"]["control_database"])
        control_path.parent.mkdir(parents=True, exist_ok=True)
        connection = connect_control(str(control_path))
        run_id = begin_run(connection, dataset, source, start, end, PROCESSING_VERSION)
        try:
            parser = parse_cpcb if dataset == "CPCB" else parse_cnemc
            parsed = [parser(path, source=source, run_id=run_id) for path in paths]
            if not parsed:
                raise IngestionError("No source artifacts were available")
            frame = pd.concat(parsed, ignore_index=True)
            frame = frame[(frame["timestamp_local"] >= start) & (frame["timestamp_local"] <= end)]
            policy = self.config["ingestion"]
            canonical = impute_new_data(frame, max_missing_fraction=policy["max_missing_fraction"], max_gap_hours=policy["max_imputation_gap_hours"], retain_missing=True)
            artifact_digest = hashlib.sha256()
            for path in sorted(Path(item) for item in paths):
                artifact_digest.update(path.name.encode())
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        artifact_digest.update(chunk)
            stats = {"rows_inserted": 0, "rows_updated": 0, "conflicting_candidates": 0, "conflict_artifact": None}
            if not dry_run:
                data_root = Path(self.config["project"]["data_root"])
                stats = write_partitions(canonical, data_root / "curated", source_priority=self.config["datasets"][dataset]["source_priority"], conflict_root=data_root / "control" / "source_conflicts", run_id=run_id)
                if self.config.get("storage", {}).get("refresh_after_ingestion", True):
                    refresh = append_and_refresh_lakehouse(
                        canonical, data_root,
                        country_code=self.config["datasets"][dataset]["country_code"],
                        country_name=self.config["datasets"][dataset]["country_name"],
                        hot_months=int(self.config.get("storage", {}).get("hourly_query_months", 13)),
                    )
                    stats["lakehouse_refresh"] = refresh
            counts = canonical["value_status"].value_counts()
            connection.execute(
                "UPDATE ingestion_runs SET actual_start=?,actual_end=?,completed_at=?,rows_downloaded=?,rows_parsed=?,rows_valid=?,rows_rejected=?,rows_observed=?,rows_imputed=?,rows_inserted=?,rows_updated_within_grace=?,artifact_sha256=?,status='success' WHERE ingestion_run_id=?",
                [frame["timestamp_local"].min(), frame["timestamp_local"].max(), datetime.now(timezone.utc), len(frame), len(frame), len(canonical), max(0, len(frame) - len(canonical)), int(counts.get("observed", 0)), int(counts.get("imputed", 0)), stats["rows_inserted"], stats["rows_updated"], artifact_digest.hexdigest(), run_id],
            )
            if not dry_run:
                grace = timedelta(hours=self.config["datasets"][dataset]["grace_hours"])
                finalized = canonical["timestamp_local"].max() - grace
                connection.execute("INSERT INTO source_checkpoints VALUES (?,?,?,?,?) ON CONFLICT(dataset,source) DO UPDATE SET last_successful_ingestion=excluded.last_successful_ingestion,last_finalized_timestamp=excluded.last_finalized_timestamp,latest_source_timestamp_seen=excluded.latest_source_timestamp_seen", [dataset, source, datetime.now(timezone.utc), finalized, canonical["timestamp_local"].max()])
            return {"run_id": run_id, "dry_run": dry_run, "rows": len(canonical), **stats}
        except Exception as exc:
            connection.execute("UPDATE ingestion_runs SET completed_at=?,status='failed',error=? WHERE ingestion_run_id=?", [datetime.now(timezone.utc), str(exc), run_id])
            raise
        finally:
            connection.close()

    def latest_window(self, dataset: str, source: str, latest: datetime | None = None) -> tuple[datetime, datetime]:
        cfg = self.config["datasets"][dataset]
        end = latest or datetime.now()
        control_path = Path(self.config["project"]["control_database"])
        checkpoint = None
        if control_path.exists():
            connection = connect_control(str(control_path))
            row = connection.execute("SELECT latest_source_timestamp_seen FROM source_checkpoints WHERE dataset=? AND source=?", [dataset, source]).fetchone()
            connection.close()
            checkpoint = row[0] if row else None
        latest_seen = checkpoint or datetime.fromisoformat(cfg["historical_end"])
        return latest_seen - timedelta(hours=cfg["grace_hours"]), end

    def fetch_and_ingest(self, dataset: str, *, start: datetime | None = None, end: datetime | None = None, dry_run: bool = False) -> dict:
        dataset_cfg = self.config["datasets"][dataset]
        source = dataset_cfg["source_priority"][0]
        source_cfg = self.config["ingestion"]["sources"][source]
        if start is None or end is None:
            start, end = self.latest_window(dataset, source, end)
        raw_root = Path(self.config["project"]["data_root"]) / "raw" / dataset / f"ingestion_date={datetime.now():%Y-%m-%d}"
        if dataset == "CNEMC":
            paths = fetch_cnemc(start.date(), end.date(), raw_root, source_cfg["url_template"], dry_run=dry_run)
        else:
            artifacts = self._cpcb_artifacts()
            paths = []
            for year in range(start.year, end.year + 1):
                paths.extend(fetch_cpcb_year(artifacts, year, raw_root, source_cfg["base_url"], dry_run=dry_run))
        if dry_run:
            return {"dataset": dataset, "source": source, "start": start.isoformat(), "end": end.isoformat(), "planned_artifacts": [str(path) for path in paths], "dry_run": True}
        return self.ingest_files(dataset, source, paths, start=start, end=end, dry_run=False)

    def _cpcb_artifacts(self) -> list[str]:
        root = Path(self.config["datasets"]["CPCB"]["historical_path"])
        suffixes = ("_PM2.5.csv", "_PM10.csv", "_NO2.csv", "_SO2.csv", "_CO.csv", "_Ozone.csv")
        stems = set()
        for path in root.glob("*.csv"):
            for suffix in suffixes:
                if path.name.endswith(suffix):
                    stems.add(path.name[:-len(suffix)] + "_15Min.csv")
                    break
        return sorted(stems)
