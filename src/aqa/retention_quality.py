from __future__ import annotations

import csv
import json
import os
import re
import shutil
from pathlib import Path
from typing import Iterable
from uuid import uuid4

import duckdb
import pandas as pd

from .catalog import _cpcb_geography


OPENAQ_METADATA = {
    "AE": "openaq_uae_site_metadata.csv", "AR": "openaq_ar_site_metadata.csv",
    "AU": "openaq_au_site_metadata.csv", "BR": "openaq_br_site_metadata.csv",
    "CL": "openaq_cl_site_metadata.csv", "JP": "openaq_japan_site_metadata.csv",
    "KR": "openaq_south_korea_site_metadata.csv", "MN": "openaq_mn_site_metadata.csv",
    "NZ": "openaq_nz_site_metadata.csv", "PE": "openaq_pe_site_metadata.csv",
    "TH": "openaq_thailand_site_metadata.csv", "TR": "openaq_turkey_site_metadata.csv",
    "ZA": "openaq_za_site_metadata.csv",
}


def refresh_station_dimension(
    lakehouse: str | Path, metadata_root: str | Path, datasets: Iterable[str] | None = None,
) -> dict:
    """Build a canonical station dimension for every station present in query tiers."""
    root = Path(lakehouse).resolve()
    metadata_root = Path(metadata_root).resolve()
    selected = sorted(set(datasets or ()))
    paths = (
        [path for dataset in selected for path in _dataset_paths(root / "query" / "monthly", dataset)]
        + [path for dataset in selected for path in _dataset_paths(root / "query" / "hourly", dataset)]
        if selected else _all_paths(root / "query" / "monthly") + _all_paths(root / "query" / "hourly")
    )
    if not paths:
        raise FileNotFoundError("No query-tier files are available for station dimension discovery")
    connection = duckdb.connect()
    try:
        actual = connection.execute("""
            SELECT dataset_id dataset, any_value(network) network,
                   any_value(country_code) country_code, any_value(country_name) country_name,
                   source_station_id
            FROM (
                SELECT *, regexp_extract(filename, 'dataset=([^/]+)', 1) dataset_id
                FROM read_parquet(?, filename=true, hive_partitioning=true, union_by_name=true)
            )
            GROUP BY dataset_id, source_station_id
            ORDER BY dataset_id, source_station_id
        """, [paths]).fetchall()
    finally:
        connection.close()
    candidates = _metadata_candidates(metadata_root)
    rows = []
    matched = 0
    for dataset, network, code, country, station_id in actual:
        item = candidates.get((dataset, str(station_id)))
        if item:
            matched += 1
        rows.append({
            "dataset": dataset, "network": network, "country_code": code,
            "country_name": country, "source_station_id": str(station_id),
            "station_name": item.get("station_name") if item else None,
            "city": item.get("city") if item else None,
            "admin1": item.get("admin1") if item else None,
            "latitude": item.get("latitude") if item else None,
            "longitude": item.get("longitude") if item else None,
            "geography_status": item.get("geography_status") if item else "unresolved",
            "metadata_source": item.get("metadata_source") if item else None,
        })
    control = duckdb.connect(str(root / "control" / "retention.duckdb"))
    try:
        frame = pd.DataFrame(rows)
        control.register("station_dimension_frame", frame)
        exists = control.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name='station_dimension'"
        ).fetchone()[0]
        if exists and selected:
            placeholders = ",".join("?" for _ in selected)
            control.execute(f"DELETE FROM station_dimension WHERE dataset IN ({placeholders})", selected)
            control.execute("INSERT INTO station_dimension SELECT * FROM station_dimension_frame")
        else:
            control.execute("CREATE OR REPLACE TABLE station_dimension AS SELECT * FROM station_dimension_frame")
        control.execute("CREATE UNIQUE INDEX IF NOT EXISTS station_dimension_key ON station_dimension(dataset, source_station_id)")
    finally:
        control.close()
    return {"stations": len(rows), "matched": matched, "unresolved": len(rows) - matched}


def refresh_quality_cube(lakehouse: str | Path, datasets: Iterable[str] | None = None) -> dict:
    """Materialize monthly multi-metric area quality from canonical station-day values."""
    root = Path(lakehouse).resolve()
    selected = sorted(set(datasets or ()))
    control = duckdb.connect(str(root / "control" / "retention.duckdb"), read_only=True)
    try:
        dimensions = control.execute("SELECT * FROM station_dimension").fetchdf()
        available = [row[0] for row in control.execute("SELECT dataset FROM dataset_availability ORDER BY dataset").fetchall()]
    finally:
        control.close()
    selected = selected or available
    built_rows = 0
    for dataset in selected:
        daily = _dataset_paths(root / "query" / "daily", dataset)
        hourly = _dataset_paths(root / "query" / "hourly", dataset)
        if not daily and not hourly:
            continue
        subset = dimensions[dimensions["dataset"] == dataset].copy()
        dataset_rows = _build_dataset_cube(root, dataset, daily, hourly, subset)
        built_rows += dataset_rows
        print(json.dumps({"quality_cube_dataset": dataset, "rows": dataset_rows}), flush=True)
    return {"datasets_refreshed": selected, "rows": built_rows}


def refresh_area_daily(lakehouse: str | Path, datasets: Iterable[str] | None = None) -> dict:
    """Materialize recent area-day values used by thresholds, episodes, and profiles."""
    root = Path(lakehouse).resolve()
    selected = sorted(set(datasets or ()))
    control = duckdb.connect(str(root / "control" / "retention.duckdb"), read_only=True)
    try:
        dimensions = control.execute("SELECT * FROM station_dimension").fetchdf()
        available = [row[0] for row in control.execute("SELECT dataset FROM dataset_availability ORDER BY dataset").fetchall()]
    finally:
        control.close()
    selected = selected or available
    total = 0
    for dataset in selected:
        hourly = _dataset_paths(root / "query" / "hourly", dataset)
        if not hourly:
            continue
        subset = dimensions[dimensions["dataset"] == dataset].copy()
        rows = _build_dataset_area_daily(root, dataset, hourly, subset)
        total += rows
        print(json.dumps({"area_daily_dataset": dataset, "rows": rows}), flush=True)
    return {"datasets_refreshed": selected, "rows": total}


def _build_dataset_area_daily(root: Path, dataset: str, hourly: list[str], dimensions: pd.DataFrame) -> int:
    connection = duckdb.connect()
    connection.execute("SET threads=8")
    connection.execute("SET memory_limit='12GB'")
    connection.register("station_dimension", dimensions)
    sql = """
        WITH station_day AS (
          SELECT dataset_id dataset, any_value(network) network, any_value(country_code) country_code,
                 any_value(country_name) country_name, source_station_id, pollutant,
                 date_trunc('day', timestamp_local) bucket_day,
                 avg(analysis_value) FILTER (WHERE analysis_value>=0) station_value,
                 count(*)::BIGINT expected_hour_count,
                 count(analysis_value) FILTER (WHERE analysis_value>=0)::BIGINT valid_hour_count,
                 count(*) FILTER (WHERE value_status='observed' AND observed_value>=0)::BIGINT observed_count,
                 count(*) FILTER (WHERE value_status LIKE 'imputed%' AND analysis_value>=0)::BIGINT imputed_count,
                 count(*) FILTER (WHERE value_status LIKE 'missing%')::BIGINT missing_count,
                 count(*) FILTER (WHERE value_status LIKE 'invalid%')::BIGINT invalid_count,
                 any_value(unit) unit
          FROM (
            SELECT *, regexp_extract(filename, 'dataset=([^/]+)', 1) dataset_id
            FROM read_parquet(?, filename=true, hive_partitioning=true, union_by_name=true)
          )
          GROUP BY dataset_id, source_station_id, pollutant, bucket_day
        ), enriched AS (
          SELECT s.*, d.city,
                 CASE WHEN d.city IS NULL OR trim(d.city)='' THEN NULL ELSE lower(trim(d.city)) END city_id
          FROM station_day s LEFT JOIN station_dimension d USING(dataset, source_station_id)
        ), city_day AS (
          SELECT dataset, any_value(network) network, any_value(country_code) country_code,
                 any_value(country_name) country_name, 'city' geography_level, city_id area_id,
                 any_value(city) area_name, pollutant, bucket_day,
                 avg(station_value) mean, median(station_value) median, quantile_cont(station_value,.95) p95,
                 min(station_value) min, max(station_value) max,
                 sum(expected_hour_count) expected_hour_count, sum(valid_hour_count) valid_hour_count,
                 sum(observed_count) observed_count, sum(imputed_count) imputed_count,
                 sum(missing_count) missing_count, sum(invalid_count) invalid_count,
                 count(station_value)::BIGINT active_station_count, 1::BIGINT contributing_city_count,
                 any_value(unit) unit, 'equal_station_within_city' aggregation_method
          FROM enriched WHERE city_id IS NOT NULL
          GROUP BY dataset, city_id, pollutant, bucket_day
        ), country_station_day AS (
          SELECT dataset, any_value(network) network, any_value(country_code) country_code,
                 any_value(country_name) country_name, 'country' geography_level,
                 any_value(country_code) area_id, any_value(country_name) area_name,
                 pollutant, bucket_day, avg(station_value) mean, median(station_value) median,
                 quantile_cont(station_value,.95) p95, min(station_value) min, max(station_value) max,
                 sum(expected_hour_count) expected_hour_count, sum(valid_hour_count) valid_hour_count,
                 sum(observed_count) observed_count, sum(imputed_count) imputed_count,
                 sum(missing_count) missing_count, sum(invalid_count) invalid_count,
                 count(station_value)::BIGINT active_station_count,
                 count(DISTINCT city_id) FILTER (WHERE city_id IS NOT NULL)::BIGINT contributing_city_count,
                 any_value(unit) unit, 'equal_station' aggregation_method
          FROM enriched GROUP BY dataset, pollutant, bucket_day
        ), country_city_day AS (
          SELECT dataset, any_value(network) network, any_value(country_code) country_code,
                 any_value(country_name) country_name, 'country_city_balanced' geography_level,
                 any_value(country_code) area_id, any_value(country_name) area_name,
                 pollutant, bucket_day, avg(mean) mean, median(mean) median, quantile_cont(mean,.95) p95,
                 min(mean) min, max(mean) max,
                 sum(expected_hour_count) expected_hour_count, sum(valid_hour_count) valid_hour_count,
                 sum(observed_count) observed_count, sum(imputed_count) imputed_count,
                 sum(missing_count) missing_count, sum(invalid_count) invalid_count,
                 sum(active_station_count)::BIGINT active_station_count,
                 count(*)::BIGINT contributing_city_count, any_value(unit) unit,
                 'equal_city_after_equal_station' aggregation_method
          FROM city_day GROUP BY dataset, pollutant, bucket_day
        ), output AS (
          SELECT * FROM country_station_day UNION ALL BY NAME
          SELECT * FROM city_day UNION ALL BY NAME
          SELECT * FROM country_city_day
        )
        SELECT *, valid_hour_count::DOUBLE/nullif(expected_hour_count,0) coverage,
               observed_count::DOUBLE/nullif(expected_hour_count,0) observed_fraction,
               imputed_count::DOUBLE/nullif(expected_hour_count,0) imputed_fraction,
               year(bucket_day)::INTEGER AS "year", month(bucket_day)::INTEGER AS "month"
        FROM output WHERE mean IS NOT NULL
    """
    temporary = root / "staging" / f"area-daily-{dataset}-{uuid4().hex}"
    target = root / "query" / "area_daily" / f"dataset={dataset}"
    temporary.mkdir(parents=True, exist_ok=True)
    connection.execute(
        f"COPY ({sql}) TO '{_literal(temporary)}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (\"year\", \"month\"), ROW_GROUP_SIZE 122880)",
        [hourly],
    )
    rows = connection.execute(f"SELECT count(*) FROM read_parquet('{_literal(temporary / '**/*.parquet')}')").fetchone()[0]
    connection.close()
    backup = temporary.parent / f"area-daily-previous-{dataset}-{uuid4().hex}"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists(): os.replace(target, backup)
    os.replace(temporary, target)
    shutil.rmtree(backup, ignore_errors=True)
    return int(rows)


def _build_dataset_cube(root: Path, dataset: str, daily: list[str], hourly: list[str], dimensions: pd.DataFrame) -> int:
    connection = duckdb.connect()
    connection.execute("SET threads=8")
    connection.execute("SET memory_limit='12GB'")
    connection.register("station_dimension", dimensions)
    station_day_parts = []
    parameters: list = []
    if daily:
        station_day_parts.append("""
            SELECT dataset_id dataset, any_value(network) network, any_value(country_code) country_code,
                   any_value(country_name) country_name, source_station_id, pollutant,
                   bucket_start bucket_day, mean station_value,
                   expected_hour_count, valid_hour_count, observed_count, imputed_count,
                   missing_count, invalid_negative_count invalid_count, unit
            FROM (
                SELECT *, regexp_extract(filename, 'dataset=([^/]+)', 1) dataset_id
                FROM read_parquet(?, filename=true, hive_partitioning=true, union_by_name=true)
            )
            GROUP BY dataset_id, source_station_id, pollutant, bucket_start, mean,
                     expected_hour_count, valid_hour_count, observed_count, imputed_count,
                     missing_count, invalid_negative_count, unit
        """)
        parameters.append(daily)
    if hourly:
        station_day_parts.append("""
            SELECT dataset_id dataset, any_value(network) network, any_value(country_code) country_code,
                   any_value(country_name) country_name, source_station_id, pollutant,
                   date_trunc('day', timestamp_local) bucket_day,
                   avg(analysis_value) FILTER (WHERE analysis_value>=0) station_value,
                   count(*)::BIGINT expected_hour_count,
                   count(analysis_value) FILTER (WHERE analysis_value>=0)::BIGINT valid_hour_count,
                   count(*) FILTER (WHERE value_status='observed' AND observed_value>=0)::BIGINT observed_count,
                   count(*) FILTER (WHERE value_status LIKE 'imputed%' AND analysis_value>=0)::BIGINT imputed_count,
                   count(*) FILTER (WHERE value_status LIKE 'missing%')::BIGINT missing_count,
                   count(*) FILTER (WHERE value_status LIKE 'invalid%')::BIGINT invalid_count,
                   any_value(unit) unit
            FROM (
                SELECT *, regexp_extract(filename, 'dataset=([^/]+)', 1) dataset_id
                FROM read_parquet(?, filename=true, hive_partitioning=true, union_by_name=true)
            )
            GROUP BY dataset_id, source_station_id, pollutant, bucket_day
        """)
        parameters.append(hourly)
    station_days = " UNION ALL BY NAME ".join(station_day_parts)
    sql = f"""
        WITH station_day AS ({station_days}),
        enriched AS (
          SELECT s.*, d.city,
                 CASE WHEN d.city IS NULL OR trim(d.city)='' THEN NULL ELSE lower(trim(d.city)) END city_id
          FROM station_day s LEFT JOIN station_dimension d USING(dataset, source_station_id)
        ),
        city_day AS (
          SELECT dataset, any_value(network) network, any_value(country_code) country_code,
                 any_value(country_name) country_name, 'city' geography_level,
                 city_id area_id, any_value(city) area_name, pollutant, bucket_day,
                 avg(station_value) area_value, sum(expected_hour_count) expected_hour_count,
                 sum(valid_hour_count) valid_hour_count, sum(observed_count) observed_count,
                 sum(imputed_count) imputed_count, sum(missing_count) missing_count,
                 sum(invalid_count) invalid_count, count(station_value) active_station_count,
                 1::BIGINT contributing_city_count, any_value(unit) unit,
                 'equal_station_within_city' aggregation_method
          FROM enriched WHERE city_id IS NOT NULL
          GROUP BY dataset, city_id, pollutant, bucket_day
        ),
        country_station_day AS (
          SELECT dataset, any_value(network) network, any_value(country_code) country_code,
                 any_value(country_name) country_name, 'country' geography_level,
                 any_value(country_code) area_id, any_value(country_name) area_name,
                 pollutant, bucket_day, avg(station_value) area_value,
                 sum(expected_hour_count) expected_hour_count, sum(valid_hour_count) valid_hour_count,
                 sum(observed_count) observed_count, sum(imputed_count) imputed_count,
                 sum(missing_count) missing_count, sum(invalid_count) invalid_count,
                 count(station_value) active_station_count,
                 count(DISTINCT city_id) FILTER (WHERE city_id IS NOT NULL)::BIGINT contributing_city_count,
                 any_value(unit) unit, 'equal_station' aggregation_method
          FROM enriched GROUP BY dataset, pollutant, bucket_day
        ),
        country_city_day AS (
          SELECT dataset, any_value(network) network, any_value(country_code) country_code,
                 any_value(country_name) country_name, 'country_city_balanced' geography_level,
                 any_value(country_code) area_id, any_value(country_name) area_name,
                 pollutant, bucket_day, avg(area_value) area_value,
                 sum(expected_hour_count) expected_hour_count, sum(valid_hour_count) valid_hour_count,
                 sum(observed_count) observed_count, sum(imputed_count) imputed_count,
                 sum(missing_count) missing_count, sum(invalid_count) invalid_count,
                 sum(active_station_count) active_station_count,
                 count(*)::BIGINT contributing_city_count, any_value(unit) unit,
                 'equal_city_after_equal_station' aggregation_method
          FROM city_day GROUP BY dataset, pollutant, bucket_day
        ),
        area_day AS (
          SELECT * FROM country_station_day UNION ALL BY NAME
          SELECT * FROM city_day UNION ALL BY NAME
          SELECT * FROM country_city_day
        )
        SELECT dataset, any_value(network) network, any_value(country_code) country_code,
               any_value(country_name) country_name, geography_level, area_id, any_value(area_name) area_name,
               pollutant, date_trunc('month', bucket_day) month_start,
               avg(area_value) mean, median(area_value) median,
               quantile_cont(area_value, .90) p90, quantile_cont(area_value, .95) p95,
               stddev_samp(area_value) std, min(area_value) min, max(area_value) max,
               count(area_value)::BIGINT valid_day_count,
               date_diff('day', date_trunc('month', bucket_day), date_trunc('month', bucket_day)+INTERVAL '1 month')::BIGINT expected_day_count,
               count(*) FILTER (WHERE valid_hour_count::DOUBLE/nullif(expected_hour_count,0)>=.50)::BIGINT days_coverage_50,
               count(*) FILTER (WHERE valid_hour_count::DOUBLE/nullif(expected_hour_count,0)>=.75)::BIGINT days_coverage_75,
               count(*) FILTER (WHERE valid_hour_count::DOUBLE/nullif(expected_hour_count,0)>=.90)::BIGINT days_coverage_90,
               sum(expected_hour_count)::BIGINT expected_hour_count,
               sum(valid_hour_count)::BIGINT valid_hour_count,
               sum(observed_count)::BIGINT observed_count, sum(imputed_count)::BIGINT imputed_count,
               sum(missing_count)::BIGINT missing_count, sum(invalid_count)::BIGINT invalid_count,
               sum(valid_hour_count)::DOUBLE/nullif(sum(expected_hour_count),0) coverage,
               sum(observed_count)::DOUBLE/nullif(sum(valid_hour_count),0) observed_fraction,
               sum(imputed_count)::DOUBLE/nullif(sum(valid_hour_count),0) imputed_fraction,
               max(active_station_count)::BIGINT active_station_count,
               max(contributing_city_count)::BIGINT contributing_city_count,
               any_value(unit) unit, any_value(aggregation_method) aggregation_method,
               year(date_trunc('month', bucket_day))::INTEGER AS "year",
               month(date_trunc('month', bucket_day))::INTEGER AS "month"
        FROM area_day
        GROUP BY dataset, geography_level, area_id, pollutant, month_start
    """
    temporary = root / "staging" / f"quality-{dataset}-{uuid4().hex}"
    target = root / "query" / "quality_monthly" / f"dataset={dataset}"
    temporary.mkdir(parents=True, exist_ok=True)
    connection.execute(
        f"COPY ({sql}) TO '{_literal(temporary)}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (\"year\", \"month\"), ROW_GROUP_SIZE 122880)",
        parameters,
    )
    rows = connection.execute(f"SELECT count(*) FROM read_parquet('{_literal(temporary / '**/*.parquet')}')").fetchone()[0]
    connection.close()
    backup = temporary.parent / f"quality-previous-{dataset}-{uuid4().hex}"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        os.replace(target, backup)
    os.replace(temporary, target)
    shutil.rmtree(backup, ignore_errors=True)
    return int(rows)


def _metadata_candidates(root: Path) -> dict[tuple[str, str], dict]:
    output: dict[tuple[str, str], dict] = {}
    for path in sorted(root.glob("*.csv")):
        dataset = _dataset_for_metadata(path.name)
        if not dataset:
            continue
        with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
            for row in csv.DictReader(handle):
                parsed = _parse_metadata_row(dataset, path.name, row)
                if not parsed:
                    continue
                ids, values = parsed
                values["metadata_source"] = path.name
                for station_id in ids:
                    if station_id:
                        output.setdefault((dataset, station_id), values)
    return output


def _dataset_for_metadata(filename: str) -> str | None:
    if filename == "cpcb_sites_with_coordinates.csv": return "CPCB"
    if filename == "cnemc_station_new.csv": return "CNEMC"
    if filename == "epa_stations.csv": return "EPA"
    if filename == "aurn_stations.csv": return "AURN"
    if filename == "sinaica_stations_with_coordinates.csv": return "SINAICA"
    match = re.match(r"eea_([a-z]{2})_stations\.csv", filename)
    if match: return "EEA_" + match.group(1).upper()
    for code, name in OPENAQ_METADATA.items():
        if filename == name: return "OPENAQ_" + code
    return None


def _parse_metadata_row(dataset: str, filename: str, row: dict) -> tuple[list[str], dict] | None:
    ids: list[str]
    city = admin1 = None
    status = "coordinates_only"
    if dataset == "CPCB":
        ids = [str(row.get("site_ids", ""))]
        station_name = row.get("site_names")
        city, admin1, status = _cpcb_geography(station_name or "")
    elif dataset == "CNEMC":
        ids = [str(row.get("Site_code", ""))]
        station_name, city, status = row.get("Site_name"), row.get("City"), "authoritative_city"
    elif dataset == "EPA":
        ids = ["_".join(_integer_text(row.get(key)) for key in ("State Code", "County Code", "Site Number"))]
        station_name, city, admin1 = row.get("Local Site Name"), row.get("City Name"), row.get("State Name")
        status = "authoritative_city"
    elif dataset == "AURN":
        ids = [str(row.get("site_id", ""))]
        station_name = row.get("site_name")
        city = row.get("local_authority") or row.get("agglomeration")
        admin1, status = row.get("zone"), "authoritative_local_authority"
    elif dataset == "SINAICA":
        ids = [str(row.get("station_id", ""))]
        station_name = row.get("station_name")
        tokens = str(row.get("raw_tokens") or "").split()
        admin1 = tokens[0] if tokens else None
        city = tokens[3] if len(tokens) > 3 else None
        status = "derived_from_source_tokens" if city else "coordinates_only"
    elif dataset.startswith("EEA_"):
        ids = [
            str(row.get("Air Quality Station Nat Code", "")),
            str(row.get("Air Quality Station EoI Code", "")),
            str(row.get("Sampling Point Id", "")),
        ]
        station_name = row.get("Air Quality Station Name")
        city = row.get("Municipality")
        status = "authoritative_municipality" if city else "coordinates_only"
    else:
        ids = [str(row.get("site_id", ""))]
        station_name = row.get("site_name")
    ids = [value.strip() for value in ids if value and value.strip()]
    if not ids:
        return None
    return ids, {
        "station_name": _clean(station_name), "city": _clean(city), "admin1": _clean(admin1),
        "latitude": _float(row.get("Latitude") or row.get("latitude")),
        "longitude": _float(row.get("Longitude") or row.get("longitude")),
        "geography_status": status,
    }


def _integer_text(value) -> str:
    try: return str(int(float(value)))
    except (TypeError, ValueError): return str(value or "").strip()


def _float(value) -> float | None:
    try: return float(value) if value not in (None, "") else None
    except (TypeError, ValueError): return None


def _clean(value) -> str | None:
    value = str(value).strip() if value is not None else ""
    return value or None


def _all_paths(root: Path) -> list[str]:
    return [str(path) for path in root.glob("dataset=*/year=*/month=*/*.parquet")]


def _dataset_paths(root: Path, dataset: str) -> list[str]:
    return [str(path) for path in (root / f"dataset={dataset}").glob("year=*/month=*/*.parquet")]


def _literal(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")
