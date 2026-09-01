from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import duckdb

from .catalog import Catalog
from .config import load_config


def build_historical(*, datasets: list[str] | None = None, replace: bool = False) -> dict:
    config = load_config()
    catalog = Catalog(config)
    datasets = datasets or list(config["datasets"])
    data_root = Path(config["project"]["data_root"])
    destination = data_root / "curated" / "historical"
    staging = data_root / "staging" / f"historical-{uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    measurements = []
    connection = duckdb.connect()
    connection.execute("SET threads TO 8")
    connection.execute("SET preserve_insertion_order TO false")
    connection.execute("SET memory_limit='8GB'")
    try:
        for dataset in datasets:
            for pollutant in ("PM25", "PM10", "NO2", "SO2", "CO", "O3"):
                selected = [item for item in catalog.series if item.dataset == dataset and item.pollutant == pollutant]
                if not selected:
                    continue
                paths = [item.path for item in selected]
                value_column = _quote(selected[0].header)
                unit = selected[0].unit.replace("'", "''")
                station = "'site_' || regexp_extract(filename, 'site_([^_/]+)_', 1)" if dataset == "CPCB" else "regexp_extract(filename, 'site_([^_/]+)_', 1)"
                pattern = f"{dataset.lower()}-{pollutant.lower()}-{{uuid}}"
                sql = f"""
                    COPY (
                        SELECT '{dataset}'::VARCHAR AS dataset,
                               {station} AS source_station_id,
                               '{pollutant}'::VARCHAR AS pollutant,
                               try_cast(Timestamp AS TIMESTAMP) AS timestamp_local,
                               NULL::DOUBLE AS observed_value,
                               try_cast({value_column} AS DOUBLE) AS analysis_value,
                               '{unit}'::VARCHAR AS unit,
                               'historical_unknown'::VARCHAR AS value_status,
                               NULL::VARCHAR AS imputation_method,
                               NULL::VARCHAR AS imputation_policy_version,
                               'historical_import'::VARCHAR AS source,
                               'historical-import'::VARCHAR AS ingestion_run_id,
                               year(try_cast(Timestamp AS TIMESTAMP))::INTEGER AS year,
                               month(try_cast(Timestamp AS TIMESTAMP))::INTEGER AS month
                        FROM read_csv(?, header=true, filename=true, union_by_name=true)
                        WHERE try_cast(Timestamp AS TIMESTAMP) IS NOT NULL
                    ) TO '{_literal(staging)}'
                    (FORMAT PARQUET, PARTITION_BY (dataset, year, month), COMPRESSION ZSTD,
                     ROW_GROUP_SIZE 122880, APPEND true, FILENAME_PATTERN '{pattern}')
                """
                step = time.perf_counter()
                connection.execute(sql, [paths])
                measurements.append({"dataset": dataset, "pollutant": pollutant, "series": len(paths), "elapsed_seconds": round(time.perf_counter() - step, 3)})
                print(json.dumps(measurements[-1]), flush=True)
    finally:
        connection.close()
    if destination.exists():
        if not replace:
            raise FileExistsError(f"Destination exists: {destination}; pass replace=True")
        backup = data_root / "staging" / f"historical-replaced-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
        os.replace(destination, backup)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, destination)
    report = {
        "built_at": datetime.now(timezone.utc).isoformat(), "destination": str(destination),
        "elapsed_seconds": round(time.perf_counter() - started, 3), "steps": measurements,
    }
    report_path = data_root / "control" / "historical_build.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def compact_historical() -> dict:
    config = load_config()
    data_root = Path(config["project"]["data_root"])
    source = data_root / "curated" / "historical"
    if not source.exists():
        raise FileNotFoundError(source)
    staging = data_root / "staging" / f"historical-compact-{uuid4().hex}"
    started = time.perf_counter()
    partitions = sorted(path for path in source.glob("dataset=*/year=*/month=*") if path.is_dir())
    connection = duckdb.connect()
    connection.execute("SET threads TO 8")
    connection.execute("SET preserve_insertion_order TO false")
    connection.execute("SET memory_limit='8GB'")
    total_rows = 0
    try:
        for index, partition in enumerate(partitions, 1):
            relative = partition.relative_to(source)
            destination = staging / relative
            destination.mkdir(parents=True, exist_ok=True)
            target = destination / "observations.parquet"
            input_glob = _literal(partition / "*.parquet")
            connection.execute(f"COPY (SELECT * EXCLUDE (dataset, year, month) FROM read_parquet('{input_glob}', hive_partitioning=true) ORDER BY pollutant, source_station_id, timestamp_local) TO '{_literal(target)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)")
            rows = connection.execute("SELECT num_rows FROM parquet_file_metadata(?)", [str(target)]).fetchone()[0]
            total_rows += rows
            if index % 12 == 0 or index == len(partitions):
                print(json.dumps({"partitions": index, "of": len(partitions), "rows": total_rows}), flush=True)
    finally:
        connection.close()
    backup = data_root / "staging" / f"historical-uncompacted-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    os.replace(source, backup)
    os.replace(staging, source)
    report = {"partitions": len(partitions), "rows": total_rows, "elapsed_seconds": round(time.perf_counter() - started, 3), "backup": str(backup)}
    (data_root / "control" / "historical_compaction.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _quote(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _literal(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")
