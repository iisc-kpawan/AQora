from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import duckdb
import numpy as np
import pandas as pd

from .retention_build import _write_rollup
from .retention_availability import refresh_value_availability
from .retention_quality import refresh_area_daily, refresh_quality_cube, refresh_station_dimension


KEY = ["source_station_id", "pollutant", "timestamp_local"]


def append_and_refresh_lakehouse(frame: pd.DataFrame, data_root: str | Path, *,
                                 country_code: str, country_name: str,
                                 hot_months: int = 13) -> dict:
    """Append a successful ingestion and atomically refresh affected query tiers."""
    if frame.empty:
        return {"archive_rows": 0, "archive_partitions": 0, "query_refreshed": False}
    root = Path(data_root).resolve()
    lakehouse = root / "lakehouse"
    if not lakehouse.is_dir():
        raise FileNotFoundError(f"Lakehouse is not built: {lakehouse}")
    active_builds = [
        path for path in (root / "staging").glob("lakehouse-*")
        if len(path.name.removeprefix("lakehouse-")) == 32
        and all(character in "0123456789abcdef" for character in path.name.removeprefix("lakehouse-"))
    ]
    if active_builds:
        raise RuntimeError("Full lakehouse rebuild is in progress; retry ingestion after it publishes")
    canonical = _lakehouse_frame(frame, country_code, country_name)
    dataset = str(canonical["dataset"].iloc[0])
    affected: set[tuple[int, int]] = set()
    connection = duckdb.connect()
    connection.execute("SET threads=8")
    connection.execute("SET memory_limit='12GB'")
    try:
        for (year, month), incoming in canonical.groupby([
            canonical["timestamp_local"].dt.year,
            canonical["timestamp_local"].dt.month,
        ]):
            _upsert_archive_partition(connection, lakehouse, dataset, int(year), int(month), incoming)
            affected.add((int(year), int(month)))
        bounds = _dataset_bounds(connection, lakehouse, dataset, hot_months)
        _refresh_hot_links(root, lakehouse, dataset, bounds["hourly_start"])
        _refresh_rollups(connection, lakehouse, dataset, bounds["hourly_start"], affected)
        _upsert_availability(lakehouse, bounds)
        refresh_value_availability(lakehouse, [dataset])
        metadata_root = Path(__file__).resolve().parents[2] / "reference" / "UpstreamDataset" / "Metadata"
        refresh_station_dimension(lakehouse, metadata_root, [dataset])
        refresh_quality_cube(lakehouse, [dataset])
        refresh_area_daily(lakehouse, [dataset])
    finally:
        connection.close()
    return {
        "archive_rows": len(canonical), "archive_partitions": len(affected),
        "query_refreshed": True, "dataset": dataset,
        "available_end": bounds["available_end"].isoformat(),
        "hourly_start": bounds["hourly_start"].isoformat(),
    }


def _lakehouse_frame(frame: pd.DataFrame, country_code: str, country_name: str) -> pd.DataFrame:
    output = frame.copy()
    output["timestamp_local"] = pd.to_datetime(output["timestamp_local"])
    output["network"] = output["dataset"]
    output["country_code"] = country_code
    output["country_name"] = country_name
    output["legacy_imputed_value"] = None
    for _, indexes in output.groupby(["dataset", "source_station_id", "pollutant"]).groups.items():
        indexes = list(indexes)
        observed = output.loc[indexes].sort_values("timestamp_local")["observed_value"].astype(float)
        missing = observed.isna()
        fraction = float(observed.notna().mean())
        max_gap = int(_max_internal_gap(missing.reset_index(drop=True)))
        eligible = fraction >= 0.70 and int(observed.notna().sum()) >= 336 and max_gap <= 336
        output.loc[indexes, "series_observed_fraction"] = fraction
        output.loc[indexes, "series_max_internal_gap_hours"] = max_gap
        output.loc[indexes, "guarded_repair_eligible"] = eligible
    output["value_status"] = output["value_status"].replace({
        "imputed": "imputed_guarded",
        "missing": "missing_unresolved_gap",
    })
    columns = [
        "network", "country_code", "country_name", "source_station_id", "pollutant",
        "timestamp_local", "observed_value", "legacy_imputed_value", "analysis_value", "unit",
        "value_status", "imputation_method", "imputation_policy_version",
        "series_observed_fraction", "series_max_internal_gap_hours", "guarded_repair_eligible", "source",
    ]
    return output[["dataset", *columns]]


def _upsert_archive_partition(connection, lakehouse: Path, dataset: str, year: int, month: int,
                              incoming: pd.DataFrame) -> None:
    directory = lakehouse / "archive" / "hourly" / f"dataset={dataset}" / f"year={year}" / f"month={month}"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "observations.parquet"
    temporary = directory / f".observations.{uuid4().hex}.parquet"
    incoming = incoming.drop(columns=["dataset"])
    connection.register("retention_incoming", incoming)
    if target.exists():
        source = f"SELECT *, 0 AS arrival FROM read_parquet('{_literal(target)}') UNION ALL BY NAME SELECT *, 1 AS arrival FROM retention_incoming"
    else:
        source = "SELECT *, 1 AS arrival FROM retention_incoming"
    connection.execute(f"""
        COPY (
            SELECT * EXCLUDE (arrival)
            FROM ({source})
            QUALIFY row_number() OVER (
                PARTITION BY source_station_id, pollutant, timestamp_local
                ORDER BY arrival DESC
            ) = 1
            ORDER BY pollutant, source_station_id, timestamp_local
        ) TO '{_literal(temporary)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)
    """)
    connection.unregister("retention_incoming")
    os.replace(temporary, target)


def _dataset_bounds(connection, lakehouse: Path, dataset: str, hot_months: int) -> dict:
    paths = sorted((lakehouse / "archive" / "hourly" / f"dataset={dataset}").glob("year=*/month=*/*.parquet"))
    row = connection.execute("""
        SELECT any_value(network), any_value(country_code), any_value(country_name),
               min(timestamp_local), max(timestamp_local),
               date_trunc('month', max(timestamp_local) - ? * INTERVAL '1 month')
        FROM read_parquet(?, hive_partitioning=true, union_by_name=true)
    """, [hot_months, [str(path) for path in paths]]).fetchone()
    return {
        "dataset": dataset, "network": row[0], "country_code": row[1], "country_name": row[2],
        "available_start": row[3], "available_end": row[4], "hourly_start": row[5],
    }


def _refresh_hot_links(root: Path, lakehouse: Path, dataset: str, hourly_start: datetime) -> None:
    staging = root / "staging" / f"query-hot-{dataset}-{uuid4().hex}" / f"dataset={dataset}"
    archive_dataset = lakehouse / "archive" / "hourly" / f"dataset={dataset}"
    for partition in sorted(archive_dataset.glob("year=*/month=*")):
        year = int(partition.parent.name.split("=", 1)[1])
        month = int(partition.name.split("=", 1)[1])
        if datetime(year, month, 1) < hourly_start:
            continue
        destination = staging / partition.relative_to(archive_dataset)
        destination.mkdir(parents=True, exist_ok=True)
        for source in partition.glob("*.parquet"):
            os.link(source, destination / source.name)
    target = lakehouse / "query" / "hourly" / f"dataset={dataset}"
    backup = staging.parent / "previous"
    if target.exists():
        os.replace(target, backup)
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, target)
    shutil.rmtree(staging.parent, ignore_errors=True)


def _refresh_rollups(connection, lakehouse: Path, dataset: str, hourly_start: datetime,
                     affected: set[tuple[int, int]]) -> None:
    archive_dataset = lakehouse / "archive" / "hourly" / f"dataset={dataset}"
    cold_by_year: dict[int, list[str]] = {}
    for partition in sorted(archive_dataset.glob("year=*/month=*")):
        year = int(partition.parent.name.split("=", 1)[1])
        month = int(partition.name.split("=", 1)[1])
        if datetime(year, month, 1) >= hourly_start:
            continue
        paths = [str(path) for path in partition.glob("*.parquet")]
        cold_by_year.setdefault(year, []).extend(paths)
        for resolution, truncation in (("daily", "day"), ("monthly", "month")):
            target = lakehouse / "query" / resolution / f"dataset={dataset}" / f"year={year}" / f"month={month}" / "observations.parquet"
            if (year, month) in affected or not target.exists():
                _replace_rollup(connection, paths, target, truncation)
    for year, paths in cold_by_year.items():
        target = lakehouse / "query" / "yearly" / f"dataset={dataset}" / f"year={year}" / "month=1" / "observations.parquet"
        if any(item[0] == year for item in affected) or not target.exists():
            _replace_rollup(connection, paths, target, "year")


def _replace_rollup(connection, paths: list[str], target: Path, truncation: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{uuid4().hex}.tmp"
    _write_rollup(connection, paths, temporary, truncation)
    os.replace(temporary, target)


def _upsert_availability(lakehouse: Path, bounds: dict) -> None:
    path = lakehouse / "control" / "retention.duckdb"
    connection = duckdb.connect(str(path))
    try:
        connection.execute("DELETE FROM dataset_availability WHERE dataset=?", [bounds["dataset"]])
        connection.execute("INSERT INTO dataset_availability VALUES (?,?,?,?,?,?,?)", list(bounds.values()))
    finally:
        connection.close()


def _literal(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _max_internal_gap(mask: pd.Series) -> int:
    valid_positions = np.flatnonzero(~mask.to_numpy())
    if len(valid_positions) < 2:
        return len(mask)
    internal = mask.iloc[valid_positions[0]:valid_positions[-1] + 1].to_numpy()
    longest = current = 0
    for value in internal:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest
