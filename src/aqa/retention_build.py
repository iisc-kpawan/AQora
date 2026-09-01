from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import duckdb

from .config import load_config
from .retention_availability import refresh_value_availability
from .retention_quality import refresh_area_daily, refresh_quality_cube, refresh_station_dimension


POLLUTANTS = {
    "PM2.5": ("PM25", "PM2.5 (µg/m³)", "µg/m³"),
    "PM10": ("PM10", "PM10 (µg/m³)", "µg/m³"),
    "NO2": ("NO2", "NO2 (µg/m³)", "µg/m³"),
    "SO2": ("SO2", "SO2 (µg/m³)", "µg/m³"),
    "CO": ("CO", "CO (mg/m³)", "mg/m³"),
    "Ozone": ("O3", "Ozone (µg/m³)", "µg/m³"),
}

EEA_COUNTRIES = {
    "AD": "Andorra", "AL": "Albania", "AT": "Austria", "BA": "Bosnia and Herzegovina",
    "BE": "Belgium", "BG": "Bulgaria", "CH": "Switzerland", "CY": "Cyprus",
    "CZ": "Czechia", "DE": "Germany", "DK": "Denmark", "EE": "Estonia", "ES": "Spain",
    "FI": "Finland", "FR": "France", "GR": "Greece", "HR": "Croatia", "HU": "Hungary",
    "IE": "Ireland", "IS": "Iceland", "IT": "Italy", "LT": "Lithuania", "LU": "Luxembourg",
    "LV": "Latvia", "ME": "Montenegro", "MK": "North Macedonia", "MT": "Malta",
    "NL": "Netherlands", "NO": "Norway", "PL": "Poland", "PT": "Portugal", "RO": "Romania",
    "RS": "Serbia", "SE": "Sweden", "SI": "Slovenia", "SK": "Slovakia", "XK": "Kosovo",
}

OPENAQ_COUNTRIES = {
    "AE": "United Arab Emirates", "AR": "Argentina", "AU": "Australia", "BR": "Brazil",
    "CL": "Chile", "JP": "Japan", "KR": "South Korea", "MN": "Mongolia",
    "NZ": "New Zealand", "PE": "Peru", "TH": "Thailand", "TR": "Turkey",
    "ZA": "South Africa",
}

STANDALONE = {
    "AURN": ("GB", "United Kingdom"),
    "CNEMC": ("CN", "China"),
    "CPCB": ("IN", "India"),
    "EPA": ("US", "United States"),
    "SINAICA": ("MX", "Mexico"),
}


@dataclass(frozen=True)
class SeriesPair:
    dataset: str
    network: str
    country_code: str
    country_name: str
    source_station_id: str
    pollutant: str
    value_column: str
    unit: str
    imputed_path: str
    processed_path: str
    source: str
    source_rank: int


def discover_series(data_root: str | Path) -> list[SeriesPair]:
    """Discover authoritative imputed series and their processed counterparts."""
    root = Path(data_root).resolve()
    pairs: list[SeriesPair] = []
    for network, (country_code, country_name) in STANDALONE.items():
        pairs.extend(_pairs_in(network, network, country_code, country_name,
                               root / network / "imputed_import",
                               root / network / "processed_import", "historical", 0))
    for country_code, country_name in EEA_COUNTRIES.items():
        pairs.extend(_pairs_in(f"EEA_{country_code}", "EEA", country_code, country_name,
                               root / "EEA" / country_code / "imputed_import",
                               root / "EEA" / country_code / "processed_import", "historical", 0))
    for country_code, country_name in OPENAQ_COUNTRIES.items():
        pairs.extend(_pairs_in(f"OPENAQ_{country_code}", "OpenAQ", country_code, country_name,
                               root / "OpenAQ" / country_code / "imputed_import",
                               root / "OpenAQ" / country_code / "processed_import", "historical", 0))

    recent = root / "source_import" / "AtmoBench" / "new_data"
    recent_sources = [
        ("AURN", "AURN", "GB", "United Kingdom", recent / "aurn"),
        ("CNEMC", "CNEMC", "CN", "China", recent / "CNEMC"),
        ("EPA", "EPA", "US", "United States", recent / "epa"),
        ("SINAICA", "SINAICA", "MX", "Mexico", recent / "SINAICA"),
        ("EEA_DE", "EEA", "DE", "Germany", recent / "eea" / "DE"),
        ("EEA_FR", "EEA", "FR", "France", recent / "eea" / "FR"),
    ]
    for dataset, network, code, name, source_root in recent_sources:
        pairs.extend(_pairs_in(dataset, network, code, name, source_root / "imputed",
                               source_root / "processed", "recent_through_2026-08-28", 1))
    return pairs


def build_retention_store(
    *,
    data_root: str | Path | None = None,
    hot_months: int | None = None,
    replace: bool = False,
    datasets: list[str] | None = None,
) -> dict:
    """Build the provenance-aware archive and resolution-specific query tiers."""
    config = load_config()
    root = Path(data_root or config["project"]["data_root"]).resolve()
    storage = config.get("storage", {})
    hot_months = int(hot_months or storage.get("hourly_query_months", 13))
    final = root / "lakehouse"
    if final.exists() and not replace:
        raise FileExistsError(f"Destination exists: {final}; pass replace=True")
    staging = root / "staging" / f"lakehouse-{uuid4().hex}"
    archive = staging / "archive" / "hourly"
    query = staging / "query"
    staging.mkdir(parents=True, exist_ok=True)
    archive.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    pairs = discover_series(root)
    if datasets:
        wanted = set(datasets)
        pairs = [item for item in pairs if item.dataset in wanted]
    if not pairs:
        raise FileNotFoundError("No paired imputed/processed station series were discovered")
    missing_processed = [item.processed_path for item in pairs if not Path(item.processed_path).is_file()]
    if missing_processed:
        raise FileNotFoundError(f"Missing {len(missing_processed)} processed counterparts; first: {missing_processed[0]}")

    connection = duckdb.connect()
    connection.execute(f"SET threads TO {int(storage.get('build_threads', 12))}")
    connection.execute(f"SET memory_limit='{storage.get('build_memory_limit', '16GB')}'")
    connection.execute("SET preserve_insertion_order TO false")
    steps: list[dict] = []
    try:
        groups: dict[tuple[str, str], list[SeriesPair]] = {}
        for pair in pairs:
            groups.setdefault((pair.dataset, pair.pollutant), []).append(pair)
        batch_size = int(storage.get("series_batch_size", 32))
        batches = [
            (dataset, pollutant, group[offset:offset + batch_size])
            for (dataset, pollutant), group in sorted(groups.items())
            for offset in range(0, len(group), batch_size)
        ]
        for index, (dataset, pollutant, group) in enumerate(batches, 1):
            step_started = time.perf_counter()
            _write_archive_group(connection, archive, group)
            detail = {
                "dataset": dataset, "pollutant": pollutant,
                "series_files": len(group), "batch": index, "of": len(batches),
                "elapsed_seconds": round(time.perf_counter() - step_started, 3),
            }
            steps.append(detail)
            print(json.dumps(detail), flush=True)
        _compact_archive(connection, archive, staging / "compact")
        bounds = _materialize_query_tiers(connection, archive, query, hot_months)
        _write_control_database(connection, staging / "control" / "retention.duckdb", archive, query, bounds)
        refresh_value_availability(staging)
        refresh_station_dimension(staging, Path(config["_project_root"]) / "reference" / "AtmoBench" / "Metadata")
        refresh_quality_cube(staging)
        refresh_area_daily(staging)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        connection.close()

    backup = None
    if final.exists():
        backup = root / "staging" / f"lakehouse-replaced-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
        os.replace(final, backup)
    os.replace(staging, final)
    report = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "destination": str(final), "backup": str(backup) if backup else None,
        "hourly_query_months": hot_months, "series_files": len(pairs),
        "datasets": len({item.dataset for item in pairs}), "countries": len({item.country_code for item in pairs}),
        "elapsed_seconds": round(time.perf_counter() - started, 3), "steps": steps,
    }
    control = root / "control"
    control.mkdir(parents=True, exist_ok=True)
    (control / "retention_build.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _pairs_in(dataset: str, network: str, country_code: str, country_name: str,
              imputed: Path, processed: Path, source: str, source_rank: int) -> list[SeriesPair]:
    if not imputed.is_dir():
        return []
    output = []
    for path in sorted(imputed.glob("*.csv")):
        parsed = _parse_series_filename(path.name, dataset)
        if parsed is None:
            continue
        station_id, pollutant, value_column, unit = parsed
        output.append(SeriesPair(
            dataset, network, country_code, country_name, station_id, pollutant,
            value_column, unit, str(path.resolve()), str((processed / path.name).resolve()),
            source, source_rank,
        ))
    return output


def _parse_series_filename(filename: str, dataset: str) -> tuple[str, str, str, str] | None:
    stem = Path(filename).stem
    for suffix, (pollutant, column, unit) in POLLUTANTS.items():
        marker = "_" + suffix
        if not stem.endswith(marker):
            continue
        prefix = stem[:-len(marker)]
        if dataset == "CPCB":
            station_id = "site_" + prefix.split("_", 2)[1]
        elif dataset == "SINAICA":
            station_id = prefix.removeprefix("station_")
        else:
            station_id = prefix.removeprefix("site_")
        return station_id, pollutant, column, unit
    return None


def _write_archive_group(connection: duckdb.DuckDBPyConnection, archive: Path, group: list[SeriesPair]) -> None:
    import pandas as pd

    frame = pd.DataFrame([asdict(item) for item in group])
    connection.register("series_manifest", frame)
    imputed_paths = frame["imputed_path"].tolist()
    processed_paths = frame["processed_path"].tolist()
    column = _quote(group[0].value_column)
    output = _literal(archive)
    pattern = f"{group[0].dataset.lower()}-{group[0].pollutant.lower()}-{{uuid}}"
    sql = f"""
        COPY (
            WITH imputed AS (
                SELECT filename AS file_path,
                       try_cast(Timestamp AS TIMESTAMP) AS timestamp_local,
                       try_cast({column} AS DOUBLE) AS legacy_imputed_raw
                FROM read_csv(?, header=true, filename=true, union_by_name=true)
            ), processed AS (
                SELECT filename AS file_path,
                       try_cast(Timestamp AS TIMESTAMP) AS timestamp_local,
                       try_cast({column} AS DOUBLE) AS observed_raw
                FROM read_csv(?, header=true, filename=true, union_by_name=true)
            ), processed_stats AS (
                SELECT file_path, count(*) AS total_count,
                       count(*) FILTER (WHERE observed_raw >= 0) AS valid_count,
                       min(timestamp_local) FILTER (WHERE observed_raw >= 0) AS first_valid,
                       max(timestamp_local) FILTER (WHERE observed_raw >= 0) AS last_valid
                FROM processed GROUP BY file_path
            ), valid_intervals AS (
                SELECT file_path, timestamp_local AS previous_valid,
                       observed_raw AS previous_value,
                       lead(timestamp_local) OVER (PARTITION BY file_path ORDER BY timestamp_local) AS next_valid,
                       lead(observed_raw) OVER (PARTITION BY file_path ORDER BY timestamp_local) AS next_value
                FROM processed WHERE observed_raw >= 0
            ), gap_stats AS (
                SELECT file_path,
                       coalesce(max(date_diff('hour', previous_valid, next_valid) - 1), 0) AS max_internal_gap
                FROM valid_intervals GROUP BY file_path
            ), joined AS (
                SELECT i.*, m0.processed_path AS processed_file_path, p.observed_raw,
                       s.total_count, s.valid_count, s.first_valid, s.last_valid,
                       coalesce(g.max_internal_gap, s.total_count) AS max_internal_gap
                FROM imputed i
                JOIN series_manifest m0 ON i.file_path = m0.imputed_path
                LEFT JOIN processed p ON p.file_path = m0.processed_path
                                     AND p.timestamp_local = i.timestamp_local
                JOIN processed_stats s ON s.file_path = m0.processed_path
                LEFT JOIN gap_stats g ON g.file_path = m0.processed_path
            ), guarded_repairs AS (
                SELECT j.file_path, j.timestamp_local,
                       greatest(
                           v.previous_value +
                           (epoch(j.timestamp_local) - epoch(v.previous_valid)) /
                           nullif(epoch(v.next_valid) - epoch(v.previous_valid), 0) *
                           (v.next_value - v.previous_value),
                           0
                       ) AS guarded_linear_value
                FROM joined j
                JOIN valid_intervals v
                  ON v.file_path = j.processed_file_path
                 AND j.timestamp_local > v.previous_valid
                 AND j.timestamp_local < v.next_valid
                WHERE (j.observed_raw IS NULL OR j.observed_raw < 0)
                  AND (j.legacy_imputed_raw IS NULL OR j.legacy_imputed_raw < 0)
                  AND j.valid_count::DOUBLE / nullif(j.total_count, 0) >= 0.70
                  AND j.valid_count >= 336 AND j.max_internal_gap <= 336
            ), candidates AS (
                SELECT m.dataset, m.network, m.country_code, m.country_name,
                       m.source_station_id, m.pollutant, j.timestamp_local,
                       CASE WHEN j.observed_raw >= 0 THEN j.observed_raw END AS observed_value,
                       j.legacy_imputed_raw AS legacy_imputed_value,
                       CASE WHEN j.observed_raw >= 0 THEN j.observed_raw
                            WHEN j.legacy_imputed_raw >= 0 THEN j.legacy_imputed_raw
                            WHEN j.valid_count::DOUBLE / nullif(j.total_count, 0) >= 0.70
                                 AND j.valid_count >= 336 AND j.max_internal_gap <= 336
                                 AND r.guarded_linear_value IS NOT NULL
                            THEN r.guarded_linear_value
                            ELSE NULL END AS analysis_value,
                       m.unit,
                       CASE WHEN j.observed_raw >= 0 THEN 'observed'
                            WHEN j.legacy_imputed_raw >= 0 THEN 'imputed_legacy_unrestricted'
                            WHEN j.valid_count::DOUBLE / nullif(j.total_count, 0) >= 0.70
                                 AND j.valid_count >= 336 AND j.max_internal_gap <= 336
                                 AND r.guarded_linear_value IS NOT NULL THEN 'imputed_guarded'
                            WHEN j.legacy_imputed_raw < 0 THEN 'invalid_negative_imputation'
                            WHEN j.valid_count::DOUBLE / nullif(j.total_count, 0) < 0.05 THEN 'missing_sparse_series'
                            ELSE 'missing_unresolved_gap' END AS value_status,
                       CASE WHEN (j.observed_raw < 0 OR j.observed_raw IS NULL) AND j.legacy_imputed_raw >= 0
                            THEN 'atmobench_mstl'
                            WHEN j.legacy_imputed_raw < 0 OR j.legacy_imputed_raw IS NULL
                            THEN CASE WHEN j.valid_count::DOUBLE / nullif(j.total_count, 0) >= 0.70
                                           AND j.valid_count >= 336 AND j.max_internal_gap <= 336
                                           AND r.guarded_linear_value IS NOT NULL
                                      THEN 'guarded_linear_internal' END
                            ELSE NULL END AS imputation_method,
                       CASE WHEN j.legacy_imputed_raw >= 0 AND (j.observed_raw < 0 OR j.observed_raw IS NULL)
                            THEN 'legacy_reference_unverified'
                            WHEN (j.legacy_imputed_raw < 0 OR j.legacy_imputed_raw IS NULL)
                                 AND j.valid_count::DOUBLE / nullif(j.total_count, 0) >= 0.70
                                 AND j.valid_count >= 336 AND j.max_internal_gap <= 336
                                 AND r.guarded_linear_value IS NOT NULL
                            THEN 'guarded-30pct-336h-v1' ELSE NULL END AS imputation_policy_version,
                       j.valid_count::DOUBLE / nullif(j.total_count, 0) AS series_observed_fraction,
                       j.max_internal_gap::INTEGER AS series_max_internal_gap_hours,
                       (j.valid_count::DOUBLE / nullif(j.total_count, 0) >= 0.70
                        AND j.valid_count >= 336 AND j.max_internal_gap <= 336) AS guarded_repair_eligible,
                       m.source, m.source_rank,
                       year(j.timestamp_local)::INTEGER AS year,
                       month(j.timestamp_local)::INTEGER AS month
                FROM joined j
                JOIN series_manifest m ON j.file_path = m.imputed_path
                LEFT JOIN guarded_repairs r USING (file_path, timestamp_local)
                WHERE j.timestamp_local IS NOT NULL
            )
            SELECT * EXCLUDE (source_rank)
            FROM candidates
            QUALIFY row_number() OVER (
                PARTITION BY dataset, source_station_id, pollutant, timestamp_local
                ORDER BY source_rank DESC
            ) = 1
        ) TO '{output}' (FORMAT PARQUET, PARTITION_BY (dataset, year, month),
             COMPRESSION ZSTD, ROW_GROUP_SIZE 122880, APPEND true, FILENAME_PATTERN '{pattern}')
    """
    connection.execute(sql, [imputed_paths, processed_paths])
    connection.unregister("series_manifest")


def _compact_archive(connection: duckdb.DuckDBPyConnection, archive: Path, temporary_root: Path) -> None:
    partitions = sorted(path for path in archive.glob("dataset=*/year=*/month=*") if path.is_dir())
    for index, partition in enumerate(partitions, 1):
        relative = partition.relative_to(archive)
        target_dir = temporary_root / relative
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / "observations.parquet"
        connection.execute(f"""
            COPY (
                SELECT * EXCLUDE (dataset, year, month)
                FROM read_parquet('{_literal(partition / '*.parquet')}', hive_partitioning=true, union_by_name=true)
                QUALIFY row_number() OVER (
                    PARTITION BY source_station_id, pollutant, timestamp_local
                    ORDER BY CASE source WHEN 'recent_through_2026-08-28' THEN 0 ELSE 1 END
                ) = 1
                ORDER BY pollutant, source_station_id, timestamp_local
            ) TO '{_literal(target)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)
        """)
        if index % 100 == 0 or index == len(partitions):
            print(json.dumps({"archive_partitions_compacted": index, "of": len(partitions)}), flush=True)
    shutil.rmtree(archive)
    os.replace(temporary_root, archive)


def _materialize_query_tiers(connection: duckdb.DuckDBPyConnection, archive: Path,
                             query: Path, hot_months: int) -> list[dict]:
    source = _literal(archive / "dataset=*" / "year=*" / "month=*" / "*.parquet")
    bounds = connection.execute(f"""
        SELECT dataset, any_value(network) AS network, any_value(country_code) AS country_code,
               any_value(country_name) AS country_name, min(timestamp_local) AS available_start,
               max(timestamp_local) AS available_end,
               date_trunc('month', max(timestamp_local) - INTERVAL '{hot_months} months') AS hourly_start
        FROM read_parquet('{source}', hive_partitioning=true, union_by_name=true)
        GROUP BY dataset ORDER BY dataset
    """).fetchdf()
    for resolution in ("hourly", "daily", "monthly", "yearly"):
        (query / resolution).mkdir(parents=True, exist_ok=True)
    completed = 0
    total_partitions = len(list(archive.glob("dataset=*/year=*/month=*")))
    for row in bounds.to_dict(orient="records"):
        dataset = row["dataset"]
        hourly_start = row["hourly_start"].to_pydatetime() if hasattr(row["hourly_start"], "to_pydatetime") else row["hourly_start"]
        old_by_year: dict[int, list[str]] = {}
        dataset_root = archive / f"dataset={dataset}"
        for partition in sorted(dataset_root.glob("year=*/month=*")):
            year = int(partition.parent.name.split("=", 1)[1])
            month = int(partition.name.split("=", 1)[1])
            files = sorted(partition.glob("*.parquet"))
            if datetime(year, month, 1) >= hourly_start:
                destination = query / "hourly" / partition.relative_to(archive)
                destination.mkdir(parents=True, exist_ok=True)
                for path in files:
                    os.link(path, destination / path.name)
            else:
                paths = [str(path.resolve()) for path in files]
                old_by_year.setdefault(year, []).extend(paths)
                for resolution, truncation in (("daily", "day"), ("monthly", "month")):
                    destination = query / resolution / partition.relative_to(archive) / "observations.parquet"
                    _write_rollup(connection, paths, destination, truncation)
            completed += 1
            if completed % 100 == 0 or completed == total_partitions:
                print(json.dumps({"query_partitions_materialized": completed, "of": total_partitions}), flush=True)
        for year, paths in old_by_year.items():
            destination = query / "yearly" / f"dataset={dataset}" / f"year={year}" / "month=1" / "observations.parquet"
            _write_rollup(connection, paths, destination, "year")
    return [
        {key: (value.isoformat() if hasattr(value, "isoformat") else value) for key, value in row.items()}
        for row in bounds.to_dict(orient="records")
    ]


def _write_rollup(connection: duckdb.DuckDBPyConnection, paths: list[str], destination: Path,
                  truncation: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    connection.execute(f"""
        COPY (
            SELECT any_value(a.network) AS network,
                   any_value(a.country_code) AS country_code, any_value(a.country_name) AS country_name,
                   a.source_station_id, a.pollutant,
                   date_trunc('{truncation}', a.timestamp_local) AS bucket_start,
                   avg(a.analysis_value) AS mean, min(a.analysis_value) AS min,
                   max(a.analysis_value) AS max, median(a.analysis_value) AS median,
                   stddev_samp(a.analysis_value) AS std,
                   quantile_cont(a.analysis_value, 0.95) AS p95,
                   sum(a.analysis_value) AS value_sum,
                   sum(a.analysis_value * a.analysis_value) AS value_sum_squares,
                   count(*) AS expected_hour_count,
                   count(a.analysis_value) AS valid_hour_count,
                   count(*) FILTER (WHERE a.value_status='observed') AS observed_count,
                   count(*) FILTER (WHERE a.value_status LIKE 'imputed%') AS imputed_count,
                   count(*) FILTER (WHERE a.value_status='imputed_legacy_unrestricted') AS legacy_imputed_count,
                   count(*) FILTER (WHERE a.value_status='imputed_guarded') AS guarded_imputed_count,
                   count(*) FILTER (WHERE a.value_status LIKE 'missing%' OR a.value_status LIKE 'invalid%') AS missing_count,
                   count(*) FILTER (WHERE a.value_status='invalid_negative_imputation') AS invalid_negative_count,
                   count(*) FILTER (WHERE a.value_status='missing_sparse_series') AS sparse_missing_count,
                   count(*) FILTER (WHERE a.value_status='missing_unresolved_gap') AS unresolved_missing_count,
                   count(a.analysis_value)::DOUBLE / nullif(count(*), 0) AS coverage,
                   min(a.timestamp_local) FILTER (WHERE a.analysis_value IS NOT NULL) AS first_valid_timestamp,
                   max(a.timestamp_local) FILTER (WHERE a.analysis_value IS NOT NULL) AS last_valid_timestamp,
                   any_value(a.unit) AS unit
            FROM read_parquet(?, hive_partitioning=true, union_by_name=true) a
            GROUP BY a.source_station_id, a.pollutant, bucket_start
            ORDER BY a.pollutant, a.source_station_id, bucket_start
        ) TO '{_literal(destination)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)
    """, [paths])


def _write_control_database(connection: duckdb.DuckDBPyConnection, target: Path, archive: Path,
                            query: Path, bounds: list[dict]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    control = duckdb.connect(str(target))
    try:
        import pandas as pd
        frame = pd.DataFrame(bounds)
        control.register("bounds_frame", frame)
        control.execute("""
            CREATE TABLE dataset_availability AS
            SELECT dataset, network, country_code, country_name,
                   try_cast(available_start AS TIMESTAMP) AS available_start,
                   try_cast(available_end AS TIMESTAMP) AS available_end,
                   try_cast(hourly_start AS TIMESTAMP) AS hourly_start
            FROM bounds_frame
        """)
        control.execute("CREATE TABLE storage_policy(key VARCHAR PRIMARY KEY, value VARCHAR)")
        control.executemany("INSERT INTO storage_policy VALUES (?, ?)", [
            ("archive_hourly", str(archive)), ("query_root", str(query)),
            ("last_available_semantics", "per_dataset"),
            ("comparison_period_semantics", "common_overlap"),
        ])
    finally:
        control.close()


def _quote(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _literal(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")
