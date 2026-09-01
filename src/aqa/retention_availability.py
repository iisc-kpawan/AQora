from __future__ import annotations

from pathlib import Path
from typing import Iterable

import duckdb


MONTHLY_COLUMNS = """
    dataset VARCHAR, network VARCHAR, country_code VARCHAR, country_name VARCHAR,
    pollutant VARCHAR, unit VARCHAR, month_start TIMESTAMP,
    expected_count BIGINT, usable_count BIGINT, observed_count BIGINT,
    legacy_imputed_count BIGINT, guarded_imputed_count BIGINT,
    unresolved_count BIGINT, invalid_count BIGINT,
    active_station_count BIGINT, usable_station_count BIGINT, observed_station_count BIGINT,
    usable_coverage DOUBLE, observed_coverage DOUBLE,
    first_usable_timestamp TIMESTAMP, last_usable_timestamp TIMESTAMP
"""


def refresh_value_availability(lakehouse: str | Path, datasets: Iterable[str] | None = None) -> dict:
    """Rebuild value-aware monthly and lifetime availability from published query tiers."""
    root = Path(lakehouse).resolve()
    selected = sorted(set(datasets or ()))
    monthly_paths = _paths(root / "query" / "monthly", selected)
    hourly_paths = _paths(root / "query" / "hourly", selected)
    if not monthly_paths and not hourly_paths:
        raise FileNotFoundError("No query-tier Parquet files found for value-aware availability")
    control = duckdb.connect(str(root / "control" / "retention.duckdb"))
    try:
        control.execute(f"CREATE TABLE IF NOT EXISTS pollutant_monthly_availability({MONTHLY_COLUMNS})")
        control.execute("""
            CREATE TABLE IF NOT EXISTS pollutant_availability(
                dataset VARCHAR, network VARCHAR, country_code VARCHAR, country_name VARCHAR,
                pollutant VARCHAR, unit VARCHAR,
                row_month_start TIMESTAMP, row_month_end TIMESTAMP,
                observed_month_start TIMESTAMP, observed_month_end TIMESTAMP,
                first_usable_timestamp TIMESTAMP, last_usable_timestamp TIMESTAMP,
                expected_count BIGINT, usable_count BIGINT, observed_count BIGINT,
                legacy_imputed_count BIGINT, guarded_imputed_count BIGINT,
                unresolved_count BIGINT, invalid_count BIGINT,
                active_station_count BIGINT, usable_station_count BIGINT, observed_station_count BIGINT,
                usable_coverage DOUBLE, observed_coverage DOUBLE
            )
        """)
        unions = []
        parameters: list = []
        if monthly_paths:
            unions.append("""
                SELECT dataset_id dataset,
                       any_value(network) network, any_value(country_code) country_code,
                       any_value(country_name) country_name, pollutant, any_value(unit) unit,
                       date_trunc('month', bucket_start) month_start,
                       sum(expected_hour_count)::BIGINT expected_count,
                       sum(valid_hour_count)::BIGINT usable_count,
                       sum(observed_count)::BIGINT observed_count,
                       sum(legacy_imputed_count)::BIGINT legacy_imputed_count,
                       sum(guarded_imputed_count)::BIGINT guarded_imputed_count,
                       sum(unresolved_missing_count + sparse_missing_count)::BIGINT unresolved_count,
                       sum(invalid_negative_count)::BIGINT invalid_count,
                       count(DISTINCT source_station_id)::BIGINT active_station_count,
                       count(DISTINCT source_station_id) FILTER (WHERE valid_hour_count>0)::BIGINT usable_station_count,
                       count(DISTINCT source_station_id) FILTER (WHERE observed_count>0)::BIGINT observed_station_count,
                       sum(valid_hour_count)::DOUBLE/nullif(sum(expected_hour_count),0) usable_coverage,
                       sum(observed_count)::DOUBLE/nullif(sum(expected_hour_count),0) observed_coverage,
                       min(first_valid_timestamp) first_usable_timestamp,
                       max(last_valid_timestamp) last_usable_timestamp
                FROM (
                    SELECT *, regexp_extract(filename, 'dataset=([^/]+)', 1) dataset_id
                    FROM read_parquet(?, filename=true, hive_partitioning=true, union_by_name=true)
                )
                GROUP BY dataset_id, pollutant, month_start
            """)
            parameters.append(monthly_paths)
        if hourly_paths:
            unions.append("""
                SELECT dataset_id dataset,
                       any_value(network) network, any_value(country_code) country_code,
                       any_value(country_name) country_name, pollutant, any_value(unit) unit,
                       date_trunc('month', timestamp_local) month_start,
                       count(*)::BIGINT expected_count,
                       count(analysis_value)::BIGINT usable_count,
                       count(*) FILTER (WHERE value_status='observed' AND observed_value>=0)::BIGINT observed_count,
                       count(*) FILTER (WHERE value_status='imputed_legacy_unrestricted')::BIGINT legacy_imputed_count,
                       count(*) FILTER (WHERE value_status='imputed_guarded')::BIGINT guarded_imputed_count,
                       count(*) FILTER (WHERE value_status LIKE 'missing%')::BIGINT unresolved_count,
                       count(*) FILTER (WHERE value_status LIKE 'invalid%')::BIGINT invalid_count,
                       count(DISTINCT source_station_id)::BIGINT active_station_count,
                       count(DISTINCT source_station_id) FILTER (WHERE analysis_value IS NOT NULL)::BIGINT usable_station_count,
                       count(DISTINCT source_station_id) FILTER (WHERE value_status='observed' AND observed_value>=0)::BIGINT observed_station_count,
                       count(analysis_value)::DOUBLE/nullif(count(*),0) usable_coverage,
                       count(*) FILTER (WHERE value_status='observed' AND observed_value>=0)::DOUBLE/nullif(count(*),0) observed_coverage,
                       min(timestamp_local) FILTER (WHERE analysis_value IS NOT NULL) first_usable_timestamp,
                       max(timestamp_local) FILTER (WHERE analysis_value IS NOT NULL) last_usable_timestamp
                FROM (
                    SELECT *, regexp_extract(filename, 'dataset=([^/]+)', 1) dataset_id
                    FROM read_parquet(?, filename=true, hive_partitioning=true, union_by_name=true)
                )
                GROUP BY dataset_id, pollutant, month_start
            """)
            parameters.append(hourly_paths)
        control.execute("CREATE OR REPLACE TEMP TABLE refreshed_value_availability AS " + " UNION ALL BY NAME ".join(unions), parameters)
        if selected:
            placeholders = ",".join("?" for _ in selected)
            control.execute(f"DELETE FROM pollutant_monthly_availability WHERE dataset IN ({placeholders})", selected)
        else:
            control.execute("DELETE FROM pollutant_monthly_availability")
        control.execute("INSERT INTO pollutant_monthly_availability SELECT * FROM refreshed_value_availability")
        control.execute("DELETE FROM pollutant_availability")
        control.execute("""
            INSERT INTO pollutant_availability
            SELECT dataset, any_value(network), any_value(country_code), any_value(country_name),
                   pollutant, any_value(unit), min(month_start), max(month_start),
                   min(month_start) FILTER (WHERE observed_count>0),
                   max(month_start + INTERVAL '1 month' - INTERVAL '1 second') FILTER (WHERE observed_count>0),
                   min(first_usable_timestamp), max(last_usable_timestamp),
                   sum(expected_count), sum(usable_count), sum(observed_count),
                   sum(legacy_imputed_count), sum(guarded_imputed_count),
                   sum(unresolved_count), sum(invalid_count),
                   max(active_station_count), max(usable_station_count), max(observed_station_count),
                   sum(usable_count)::DOUBLE/nullif(sum(expected_count),0),
                   sum(observed_count)::DOUBLE/nullif(sum(expected_count),0)
            FROM pollutant_monthly_availability
            GROUP BY dataset, pollutant
        """)
        monthly_rows = control.execute("SELECT count(*) FROM pollutant_monthly_availability").fetchone()[0]
        pollutant_rows = control.execute("SELECT count(*) FROM pollutant_availability").fetchone()[0]
    finally:
        control.close()
    return {"monthly_rows": monthly_rows, "pollutant_rows": pollutant_rows, "datasets_refreshed": selected or "all"}


def _paths(root: Path, datasets: list[str]) -> list[str]:
    if datasets:
        return [str(path) for dataset in datasets for path in (root / f"dataset={dataset}").glob("year=*/month=*/*.parquet")]
    return [str(path) for path in root.glob("dataset=*/year=*/month=*/*.parquet")]
