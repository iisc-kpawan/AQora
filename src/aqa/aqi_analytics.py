from __future__ import annotations

import re
import time
from datetime import datetime
from typing import Any

import duckdb
import pandas as pd

from .aqi import calculate_aqi, get_standard, standard_for_date
from .capabilities import CapabilityAssessment
from .config import load_config
from .errors import DataUnavailableError
from .global_analytics import _period
from .lakehouse_catalog import LakehouseCatalog
from .quality_analytics import _city_alias, _requested_cities


class AQIAnalytics:
    """Daily national AQI from station-balanced hourly lakehouse concentrations."""

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.catalog = LakehouseCatalog(self.config)
        self.root = self.catalog.root

    @staticmethod
    def should_handle(question: str, assessment: CapabilityAssessment) -> bool:
        return "aqi" in question.casefold() or "air quality index" in question.casefold()

    def execute(self, question: str, assessment: CapabilityAssessment) -> dict[str, Any]:
        started = time.perf_counter()
        text = question.casefold()
        if is_aqi_provenance_policy_question(text):
            return self._provenance_policy_aqi(question, assessment, started)
        selected = list(self.catalog.selected(assessment.supported_datasets))
        if len(selected) != 1 or selected[0].dataset not in {"CPCB", "CNEMC"}:
            raise DataUnavailableError("National AQI requires one India/CPCB or China/CNEMC dataset; cross-standard comparisons must be requested as separate outputs")
        item = selected[0]
        start, end = _period(text, self.catalog, assessment.supported_datasets)
        paths = [str(path) for path in (self.root / "query" / "hourly" / f"dataset={item.dataset}").glob("year=*/month=*/*.parquet")]
        if not paths:
            raise DataUnavailableError("No recent hourly partitions are available for AQI calculation")
        cities = _requested_cities(text)
        control = duckdb.connect(str(self.catalog.control_path), read_only=True)
        try:
            dimension = control.execute(
                "SELECT source_station_id, city FROM station_dimension WHERE dataset=?", [item.dataset]
            ).fetchdf()
        finally:
            control.close()
        if cities:
            wanted = {_city_alias(name) for name in cities}
            dimension = dimension[dimension["city"].fillna("").map(_city_alias).isin(wanted)].copy()
            dimension["area_name"] = dimension["city"]
            found = {_city_alias(value) for value in dimension["city"].dropna().unique()}
            missing = [name for name in cities if _city_alias(name) not in found]
            if missing:
                raise DataUnavailableError("AQI station metadata did not resolve: " + ", ".join(missing))
        else:
            dimension["area_name"] = item.country_name
        if dimension.empty:
            raise DataUnavailableError("No stations resolved for AQI calculation")
        connection = duckdb.connect()
        connection.register("aqi_stations", dimension[["source_station_id", "area_name"]])
        try:
            result = connection.execute("""
                WITH station_hour AS (
                  SELECT s.area_name, h.source_station_id, h.pollutant, h.timestamp_local,
                         avg(h.analysis_value) FILTER (WHERE h.analysis_value>=0) station_value
                  FROM read_parquet(?, hive_partitioning=true, union_by_name=true) h
                  JOIN aqi_stations s USING(source_station_id)
                  WHERE h.timestamp_local BETWEEN ? AND ?
                  GROUP BY s.area_name, h.source_station_id, h.pollutant, h.timestamp_local
                )
                SELECT area_name, pollutant, timestamp_local, avg(station_value) concentration,
                       count(station_value) active_stations
                FROM station_hour GROUP BY area_name, pollutant, timestamp_local
                ORDER BY area_name, pollutant, timestamp_local
            """, [paths, start, end])
            names = [column[0] for column in result.description]
            hourly = pd.DataFrame(result.fetchall(), columns=names)
        finally:
            connection.close()
        if hourly.empty:
            raise DataUnavailableError("No valid hourly concentrations matched the AQI request")
        forced = _forced_standard(text, item.dataset, start)
        rows = _daily_aqi(hourly, item.dataset, forced)
        if not rows:
            raise DataUnavailableError("At least three valid pollutants including PM2.5 or PM10 are required for daily AQI")
        category_request = "category" in text or any(label in text for label in ("good", "poor", "severe", "satisfactory"))
        if category_request and ("how many" in text or "percentage" in text or "distribution" in text):
            rows = _category_summary(rows)
            visualization = "bar"
        elif "maximum" in text or "highest" in text:
            rows = sorted(rows, key=lambda row: row["aqi"], reverse=True)[:1]
            visualization = "table"
        else:
            visualization = "line" if len(rows) > 1 else "table"
        answer = _answer(rows, item.country_name, category_request)
        standards = sorted({row["standard_id"] for row in rows if "standard_id" in row})
        return {
            "kind": "analytics",
            "query_spec": {
                "dataset": item.dataset, "operation": "aqi", "geography_level": "city" if cities else "country",
                "locations": cities or [item.country_name], "pollutants": ["PM25", "PM10", "NO2", "SO2", "CO", "O3"],
                "time_range": {"start": start.isoformat(), "end": end.isoformat()}, "resolution": "daily",
                "metrics": ["aqi", "category"], "filters": {"standard": forced.id if forced else "effective-date default"},
            },
            "resolved_period": {"start": start.isoformat(), "end": end.isoformat(), "basis": "explicit period"},
            "data": rows,
            "metadata": {
                "dataset": item.dataset, "geography_level": "city" if cities else "country",
                "station_count": int(dimension["source_station_id"].nunique()), "value_status": "observed_and_policy_approved_filled",
                "station_balanced": True, "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "warnings": [], "standards": standards,
            },
            "visualization": {"type": visualization},
        "analysis": {
                "operator": "national_aqi_hourly_v1",
                "method": "pollutant averaging windows come from the effective national standard; daily AQI is the maximum pollutant sub-index",
                "standard_selection": "explicit" if forced else "effective_date",
            },
            "answer": answer, "display_title": f"{item.country_name} · daily AQI",
            "model_trace": {"provider": "deterministic", "model": "national_aqi_hourly_v1", "latency_ms": 0,
                            "prompt_tokens": None, "completion_tokens": None, "fallback_used": False},
        }

    def _provenance_policy_aqi(self, question: str, assessment: CapabilityAssessment, started: float) -> dict[str, Any]:
        text = question.casefold()
        selected = list(self.catalog.selected(assessment.supported_datasets))
        if len(selected) != 1 or selected[0].dataset not in {"CPCB", "CNEMC"}:
            raise DataUnavailableError("National AQI requires one India/CPCB or China/CNEMC dataset; cross-standard comparisons must be requested as separate outputs")
        item = selected[0]
        start, end = _period(text, self.catalog, assessment.supported_datasets)
        paths = [str(path) for path in (self.root / "query" / "hourly" / f"dataset={item.dataset}").glob("year=*/month=*/*.parquet")]
        if not paths:
            raise DataUnavailableError("No recent hourly partitions are available for AQI calculation")
        cities = _requested_cities(text)
        control = duckdb.connect(str(self.catalog.control_path), read_only=True)
        try:
            dimension = control.execute(
                "SELECT source_station_id, city FROM station_dimension WHERE dataset=?", [item.dataset]
            ).fetchdf()
        finally:
            control.close()
        if cities:
            wanted = {_city_alias(name) for name in cities}
            dimension = dimension[dimension["city"].fillna("").map(_city_alias).isin(wanted)].copy()
            dimension["area_name"] = dimension["city"]
            found = {_city_alias(value) for value in dimension["city"].dropna().unique()}
            missing = [name for name in cities if _city_alias(name) not in found]
            if missing:
                raise DataUnavailableError("AQI station metadata did not resolve: " + ", ".join(missing))
        else:
            dimension["area_name"] = item.country_name
        if dimension.empty:
            raise DataUnavailableError("No stations resolved for AQI calculation")
        connection = duckdb.connect()
        connection.register("aqi_stations", dimension[["source_station_id", "area_name"]])
        try:
            result = connection.execute("""
                WITH station_hour AS (
                  SELECT s.area_name, h.source_station_id, h.pollutant, h.timestamp_local,
                         avg(h.observed_value) FILTER (WHERE h.value_status='observed' AND h.observed_value>=0) observed_value,
                         avg(h.analysis_value) FILTER (WHERE h.analysis_value>=0) usable_value
                  FROM read_parquet(?, hive_partitioning=true, union_by_name=true) h
                  JOIN aqi_stations s USING(source_station_id)
                  WHERE h.timestamp_local BETWEEN ? AND ?
                  GROUP BY s.area_name, h.source_station_id, h.pollutant, h.timestamp_local
                )
                SELECT area_name, pollutant, timestamp_local,
                       avg(observed_value) observed_concentration,
                       avg(usable_value) usable_concentration
                FROM station_hour GROUP BY area_name, pollutant, timestamp_local
                ORDER BY area_name, pollutant, timestamp_local
            """, [paths, start, end])
            names = [column[0] for column in result.description]
            hourly = pd.DataFrame(result.fetchall(), columns=names)
        finally:
            connection.close()
        if hourly.empty:
            raise DataUnavailableError("No valid hourly concentrations matched the AQI request")
        forced = _forced_standard(text, item.dataset, start)
        observed_hourly = hourly.rename(columns={"observed_concentration": "concentration"}).drop(columns=["usable_concentration"])
        usable_hourly = hourly.rename(columns={"usable_concentration": "concentration"}).drop(columns=["observed_concentration"])
        observed_rows = {(row["area_name"], row["timestamp"]): row for row in _daily_aqi(observed_hourly, item.dataset, forced)}
        usable_rows = {(row["area_name"], row["timestamp"]): row for row in _daily_aqi(usable_hourly, item.dataset, forced)}
        shared_days = sorted(set(observed_rows) & set(usable_rows))
        if not shared_days:
            raise DataUnavailableError("No day has a valid daily AQI under both the observed-only and all-usable policies")
        rows = []
        for key in shared_days:
            observed_row, usable_row = observed_rows[key], usable_rows[key]
            rows.append({
                "area_name": key[0], "timestamp": key[1],
                "aqi_observed_only": observed_row["aqi"], "category_observed_only": observed_row["category"],
                "aqi_all_usable": usable_row["aqi"], "category_all_usable": usable_row["category"],
                "aqi_difference": usable_row["aqi"] - observed_row["aqi"],
                "category_changed": observed_row["category"] != usable_row["category"],
                "standard_id": observed_row["standard_id"], "standard_name": observed_row["standard_name"],
            })
        changed = [row for row in rows if row["category_changed"]]
        area_label = cities[0].title() if cities else item.country_name
        answer = (
            f"I calculated {area_label}'s daily AQI for {start:%d %b %Y} to {end:%d %b %Y} twice: once using "
            f"only directly measured (observed-only) hourly values, and once using all usable (policy-approved filled) "
            f"values. Of {len(rows)} day(s) with a valid AQI under both policies, {len(changed)} changed AQI category "
            "because of imputation."
        )
        if changed:
            answer += " Changed day(s): " + ", ".join(
                f"{row['timestamp'][:10]} ({row['category_observed_only']} → {row['category_all_usable']})"
                for row in changed[:8]
            ) + "."
        return {
            "kind": "analytics",
            "query_spec": {
                "dataset": item.dataset, "operation": "aqi_provenance_policy_comparison",
                "geography_level": "city" if cities else "country", "locations": cities or [item.country_name],
                "pollutants": ["PM25", "PM10", "NO2", "SO2", "CO", "O3"],
                "time_range": {"start": start.isoformat(), "end": end.isoformat()}, "resolution": "daily",
                "metrics": ["aqi", "category"], "filters": {"policies": ["observed_only", "all_usable"]},
            },
            "resolved_period": {"start": start.isoformat(), "end": end.isoformat(), "basis": "explicit period"},
            "data": rows,
            "metadata": {
                "dataset": item.dataset, "geography_level": "city" if cities else "country",
                "station_count": int(dimension["source_station_id"].nunique()), "value_status": "policy_comparison",
                "station_balanced": True, "latency_ms": round((time.perf_counter() - started) * 1000, 2), "warnings": [],
            },
            "visualization": {"type": "table"},
            "analysis": {
                "operator": "aqi_provenance_policy_comparison_v1",
                "method": "daily AQI computed independently under each value policy, then compared by day",
                "category_changed_day_count": len(changed), "total_day_count": len(rows),
            },
            "answer": answer, "display_title": f"{area_label} · AQI category sensitivity to imputation",
            "model_trace": {"provider": "deterministic", "model": "aqi_provenance_policy_comparison_v1",
                            "latency_ms": 0, "prompt_tokens": None, "completion_tokens": None, "fallback_used": False},
        }


def is_aqi_provenance_policy_question(text: str) -> bool:
    if not ("aqi" in text and "observed" in text and any(term in text for term in ("filled", "imputed"))):
        return False
    return not any(term in text for term in (
        "coverage threshold", "aggregation rule", "high-burden cities", "severe episodes",
    ))


def _daily_aqi(hourly: pd.DataFrame, dataset: str, forced) -> list[dict]:
    output = []
    hourly["timestamp_local"] = pd.to_datetime(hourly["timestamp_local"])
    for area, area_frame in hourly.groupby("area_name"):
        pieces = []
        for pollutant, group in area_frame.groupby("pollutant"):
            series = group.set_index("timestamp_local")["concentration"].sort_index().astype(float).asfreq("h")
            daily = series.resample("D").agg(["mean", "count"])
            value_24 = daily["mean"].where(daily["count"] >= 18)
            value_8 = series.rolling(8, min_periods=6).mean().resample("D").max()
            pieces.append(pd.DataFrame({
                "day": value_24.index, "pollutant": pollutant,
                "value_24": value_24.values, "value_8": value_8.reindex(value_24.index).values,
            }))
        concentrations = pd.concat(pieces)
        for day, group in concentrations.groupby("day"):
            standard = forced or standard_for_date(dataset, day.to_pydatetime())
            values = {}
            for record in group.to_dict("records"):
                window = standard.averaging_hours.get(record["pollutant"], 24)
                value = record["value_8"] if window == 8 else record["value_24"]
                if pd.notna(value):
                    values[record["pollutant"]] = float(value)
            if len(values) < 3 or not ({"PM25", "PM10"} & values.keys()):
                continue
            computed = calculate_aqi(values, standard.id)
            output.append({
                "timestamp": day.isoformat(), "area_name": area, "aqi": computed["aqi"],
                "category": computed["category"], "dominant_pollutants": computed["dominant_pollutants"],
                "standard_id": standard.id, "standard_name": standard.name,
                "valid_pollutant_count": len(computed["sub_indices"]), "unit": "AQI",
            })
    return sorted(output, key=lambda row: (row["timestamp"], row["area_name"]))


def _category_summary(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows: grouped.setdefault((row["area_name"], row["standard_id"], row["category"]), []).append(row)
    totals: dict[str, int] = {}
    for row in rows: totals[row["area_name"]] = totals.get(row["area_name"], 0) + 1
    return [{
        "area_name": area, "standard_id": standard, "category": category,
        "day_count": len(values), "percentage": len(values) / totals[area], "valid_day_count": totals[area],
        "unit": "days",
    } for (area, standard, category), values in grouped.items()]


def _forced_standard(text: str, dataset: str, start: datetime):
    if "cpcb" in text or "india standard" in text: return get_standard("IN_CPCB_NAQI_2014")
    if "hj 633-2012" in text or "2012 standard" in text: return get_standard("CN_HJ633_2012")
    if "hj 633-2026" in text or "2026 standard" in text: return get_standard("CN_HJ633_2026")
    return None


def _answer(rows: list[dict], country: str, category_request: bool) -> str:
    if category_request and rows and "day_count" in rows[0]:
        total = sum(row["day_count"] for row in rows)
        return f"I calculated the AQI category distribution for {total} valid area-days. Counts and percentages use the versioned national standard shown in the table."
    if len(rows) == 1:
        row = rows[0]
        return f"{row['area_name']} had AQI {row['aqi']} ({row['category']}) on {row['timestamp'][:10]} under {row['standard_name']}."
    return f"I calculated {len(rows)} valid daily AQI results for {country}, using the national standard effective on each date."
