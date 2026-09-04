from __future__ import annotations

import math
import re
import time
from datetime import datetime
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm

from .capabilities import CapabilityAssessment, is_flexible_country_cohort
from .config import load_config
from .errors import DataUnavailableError
from .global_analytics import POLLUTANTS, _period
from .lakehouse_catalog import LakehouseCatalog
from .quality_analytics import _city_alias, _requested_cities


class TemporalAnalytics:
    """Reusable daily/monthly trends and calendar profiles over area-balanced data."""

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.catalog = LakehouseCatalog(self.config)
        self.root = self.catalog.root

    @staticmethod
    def should_handle(question: str, assessment: CapabilityAssessment) -> bool:
        text = question.casefold()
        return assessment.intent == "temporal" or bool(re.search(r"change[- ]?point", text)) or any(
            term in text for term in (
                "moving average", "day-of-week", "day of week", "hour-of-day", "hour of day", "increase or decrease",
                "month-over-month", "trend slope", "highest mean", "weekday", "weekend", "daytime", "nighttime",
            )
        )

    def execute(self, question: str, assessment: CapabilityAssessment) -> dict[str, Any]:
        started = time.perf_counter()
        text = question.casefold()
        pollutants = [name for pattern, name in POLLUTANTS if re.search(pattern, text)]
        if len(pollutants) > 1:
            raise DataUnavailableError("The temporal operator currently accepts one pollutant per request")
        pollutant = pollutants[0] if pollutants else "PM25"
        cities = _requested_cities(text)
        geography = "city" if cities else "country"
        candidates = list(self.catalog.selected(assessment.supported_datasets))
        operation = _operation(text)
        if operation == "joint_weekday_diurnal" and not _has_period(text):
            start, end = self._latest_common_observed_month(candidates, pollutant)
        else:
            start, end = _period(text, self.catalog, assessment.supported_datasets)
        flexible = is_flexible_country_cohort(question, assessment.intent)
        cohort = self.catalog.resolve_cohort(
            [item.dataset for item in candidates], pollutant, start, end,
            value_status="any", require_all_months=True,
        )
        exclusions = [item for item in cohort if not item.included]
        if exclusions and not flexible and not cities:
            detail = "; ".join(f"{item.country_name}: {item.reason}" for item in exclusions)
            raise DataUnavailableError("The explicitly requested temporal cohort is incomplete. " + detail)
        included = {item.dataset for item in cohort if item.included}
        selected = [item for item in candidates if item.dataset in included]
        if operation == "joint_weekday_diurnal":
            rows = self._joint_diurnal(selected, cities, pollutant, start, end)
            visualization = "line"
        elif operation == "day_night":
            rows = self._day_night(selected, cities, pollutant, start, end)
            visualization = "bar"
        elif operation == "hour_of_day":
            rows = self._hour_profile(selected, cities, pollutant, start, end)
            visualization = "line"
        elif operation in {"day_of_week", "weekday_weekend", "moving_average", "daily_series"}:
            rows = self._daily(selected, cities, pollutant, start, end)
            if operation == "weekday_weekend":
                minimum_coverage = _coverage_threshold(text, float(self.config["analytics"].get("minimum_daily_coverage", .75)))
                rows = [row for row in rows if (row.get("coverage") or 0) >= minimum_coverage]
            if operation == "day_of_week": rows = _day_profile(rows)
            elif operation == "weekday_weekend": rows = _weekday_weekend(rows)
            elif operation == "moving_average": rows = _moving_average(rows, _window(text))
            visualization = "line"
        elif operation == "changepoint":
            minimum_segment_days = int(self.config.get("analytics", {}).get("changepoint_minimum_segment_days", 30))
            rows = _changepoints(self._daily(selected, cities, pollutant, start, end), minimum_segment_days)
            visualization = "line"
        elif operation == "hour_by_weekday":
            minimum_cell_samples = int(self.config.get("analytics", {}).get("heatmap_minimum_cell_samples", 3))
            rows = self._hour_by_weekday_profile(selected, cities, pollutant, start, end, minimum_cell_samples)
            visualization = "heatmap_grid"
        else:
            rows = self._monthly(selected, cities, pollutant, start, end)
            if operation == "slope": rows = _slopes(rows)
            elif operation == "percent_change": rows = _percent_changes(rows)
            elif operation == "two_period_change": rows = _two_period_change(rows)
            elif operation == "highest_months": rows = _highest_months(rows, _number_word(text, 3))
            visualization = "bar" if operation in {"slope", "two_period_change", "highest_months"} else "line"
        if not rows:
            raise DataUnavailableError("No valid temporal values match the requested area and period")
        if cities:
            found = {_city_alias(row.get("area_name") or "") for row in rows}
            missing_cities = [city for city in cities if _city_alias(city) not in found]
            if missing_cities:
                raise DataUnavailableError("The explicitly requested temporal city cohort is incomplete. Missing: " + ", ".join(missing_cities))
        answer = _answer(rows, operation, pollutant, start, end)
        if operation == "joint_weekday_diurnal":
            answer += " Only directly measured values in UTC-aligned hourly bins present for both cities are included; profile hours are shown in each city's local clock."
        if operation == "weekday_weekend":
            answer += f" Days below {_coverage_threshold(text, float(self.config['analytics'].get('minimum_daily_coverage', .75))):.0%} coverage were excluded."
            if any((row.get("imputed_fraction") or 0) > .5 for row in rows):
                answer += " At least one comparison is mostly filled rather than directly measured; use the imputed-fraction column when interpreting it."
        exclusion_rows = [{"country": item.country_name, "reason": item.reason, "reason_code": item.reason_code} for item in exclusions]
        if exclusion_rows:
            answer += " Excluded: " + "; ".join(f"{item['country']} ({item['reason']})" for item in exclusion_rows[:8]) + "."
        return {
            "kind": "analytics",
            "query_spec": {
                "dataset": "GLOBAL_TEMPORAL" if len(candidates) > 1 else candidates[0].dataset,
                "operation": operation, "geography_level": geography, "locations": cities or [item.country_name for item in candidates],
                "pollutants": [pollutant], "time_range": {"start": start.isoformat(), "end": end.isoformat()},
                "resolution": "hourly" if operation in {"hour_of_day", "joint_weekday_diurnal", "day_night", "hour_by_weekday"} else "daily" if operation in {"day_of_week", "weekday_weekend", "moving_average", "daily_series", "changepoint"} else "monthly",
                "metrics": (["before_mean", "after_mean", "mean_shift"] if operation == "changepoint" else
                            ["mean", "sample_count"] if operation == "hour_by_weekday" else ["mean"]),
                "filters": {},
            },
            "resolved_period": {"start": start.isoformat(), "end": end.isoformat(),
                                "basis": "latest common observed month with UTC-hour intersection" if operation == "joint_weekday_diurnal" and not _has_period(text) else "explicit period"},
            "data": rows,
            "metadata": {
                "dataset": "GLOBAL_TEMPORAL", "geography_level": geography,
                "station_count": max((row.get("active_station_count") or 0 for row in rows), default=0),
                "value_status": "observed" if operation in {"joint_weekday_diurnal", "hour_by_weekday"} else "observed_and_policy_approved_filled", "station_balanced": True,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2), "warnings": [],
            },
            "visualization": (
                {"type": "heatmap_grid", "x": "hour_of_day", "y": "day_of_week", "z": "mean", "mask": "masked",
                 "hover_fields": ["sample_count", "mask_reason"], "color_title": f"{pollutant} µg/m³"}
                if operation == "hour_by_weekday" else {"type": visualization}
            ),
            "analysis": {"operator": "area_temporal_v1", "method": operation, "exclusions": exclusion_rows,
                         "joint_timestamp_basis": "UTC hour floor" if operation == "joint_weekday_diurnal" else None,
                         **({"change_point_method": "minimum two-segment squared error",
                             "minimum_segment_days": rows[0]["minimum_segment_days"]} if operation == "changepoint" else {}),
                         **({"minimum_cell_samples": rows[0]["minimum_cell_samples"],
                             "masked_cell_count": sum(row["masked"] for row in rows),
                             "total_cell_count": len(rows)} if operation == "hour_by_weekday" else {})},
            "answer": answer, "display_title": f"{pollutant} · temporal analysis",
            "model_trace": {"provider": "deterministic", "model": "area_temporal_v1", "latency_ms": 0,
                            "prompt_tokens": None, "completion_tokens": None, "fallback_used": False},
        }

    def _latest_common_observed_month(self, selected, pollutant):
        connection = duckdb.connect(str(self.catalog.control_path), read_only=True)
        try:
            rows = connection.execute("""
                SELECT dataset,observed_month_end FROM pollutant_availability
                WHERE pollutant=? AND dataset IN (SELECT unnest(?))
            """, [pollutant, [item.dataset for item in selected]]).fetchall()
        finally:
            connection.close()
        found = {row[0]: row[1] for row in rows if row[1] is not None}
        missing = [item.country_name for item in selected if item.dataset not in found]
        if missing:
            raise DataUnavailableError("No observed-period bound is available for: " + ", ".join(missing))
        end = min(found.values())
        start = end.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return start, end

    def _day_night(self, selected, cities, pollutant, start, end) -> list[dict]:
        stations = self._location_stations(selected, cities)
        paths = [str(path) for item in selected for path in (self.root / "query" / "hourly" / f"dataset={item.dataset}").glob("year=*/month=*/*.parquet")]
        if not paths or stations.empty: return []
        day_start = int(self.config["analytics"].get("daytime_start_hour", 6))
        day_end = int(self.config["analytics"].get("daytime_end_hour", 18))
        connection = duckdb.connect(); connection.register("temporal_locations", stations)
        try:
            result = connection.execute(f"""
                WITH station_hour AS (
                  SELECT l.area_name,h.source_station_id,h.timestamp_local,
                         avg(h.analysis_value) FILTER(WHERE h.analysis_value>=0) station_value,
                         count(*) FILTER(WHERE h.value_status='observed' AND h.observed_value>=0) observed_count,
                         count(*) FILTER(WHERE h.value_status LIKE 'imputed%' AND h.analysis_value>=0) filled_count,
                         any_value(h.unit) unit
                  FROM read_parquet(?,hive_partitioning=true,union_by_name=true) h
                  JOIN temporal_locations l ON h.dataset=l.dataset AND h.source_station_id=l.source_station_id
                  WHERE h.pollutant=? AND h.timestamp_local BETWEEN ? AND ?
                  GROUP BY l.area_name,h.source_station_id,h.timestamp_local
                ), area_hour AS (
                  SELECT area_name,timestamp_local,avg(station_value) mean,
                         sum(observed_count) observed_count,sum(filled_count) filled_count,
                         count(station_value) active_station_count,any_value(unit) unit
                  FROM station_hour GROUP BY area_name,timestamp_local
                )
                SELECT area_name,CASE WHEN hour(timestamp_local)>={day_start} AND hour(timestamp_local)<{day_end}
                                      THEN 'daytime' ELSE 'nighttime' END period_of_day,
                       avg(mean) mean,median(mean) median,quantile_cont(mean,.95) p95,
                       count(mean) valid_hour_count,sum(observed_count)::DOUBLE/nullif(sum(observed_count+filled_count),0) observed_fraction,
                       sum(filled_count)::DOUBLE/nullif(sum(observed_count+filled_count),0) imputed_fraction,
                       max(active_station_count) active_station_count,any_value(unit) unit
                FROM area_hour GROUP BY area_name,period_of_day ORDER BY area_name,period_of_day
            """, [paths,pollutant,start,end])
            names=[x[0] for x in result.description]; rows=[dict(zip(names,row)) for row in result.fetchall()]
        finally: connection.close()
        return _add_day_night_difference(rows)

    def _joint_diurnal(self, selected, cities, pollutant, start, end) -> list[dict]:
        if len(cities) != 2:
            raise DataUnavailableError("Jointly observed diurnal comparison requires exactly two named cities")
        stations = self._location_stations(selected, cities)
        found = set(stations["area_name"]) if not stations.empty else set()
        missing = [city for city in cities if city.title() not in found]
        if missing:
            raise DataUnavailableError("No station mapping is available for: " + ", ".join(missing))
        paths = [str(path) for item in selected for path in (self.root / "query" / "hourly" / f"dataset={item.dataset}").glob("year=*/month=*/*.parquet")]
        if not paths: return []
        connection = duckdb.connect(); connection.register("temporal_locations", stations)
        try:
            result = connection.execute("""
                WITH station_hour AS (
                  SELECT l.area_name,h.source_station_id,h.timestamp_local,
                         date_trunc('hour',CASE h.dataset
                           WHEN 'CPCB' THEN timezone('Asia/Kolkata',h.timestamp_local)
                           WHEN 'CNEMC' THEN timezone('Asia/Shanghai',h.timestamp_local)
                           WHEN 'AURN' THEN timezone('Europe/London',h.timestamp_local)
                           WHEN 'SINAICA' THEN timezone('America/Mexico_City',h.timestamp_local)
                           WHEN 'EPA' THEN timezone('America/New_York',h.timestamp_local)
                           WHEN 'EEA_DE' THEN timezone('Europe/Berlin',h.timestamp_local)
                           WHEN 'EEA_FR' THEN timezone('Europe/Paris',h.timestamp_local)
                         END) timestamp_utc,
                         avg(h.observed_value) FILTER(WHERE h.value_status='observed' AND h.observed_value>=0) station_value,
                         any_value(h.unit) unit
                  FROM read_parquet(?,hive_partitioning=true,union_by_name=true) h
                  JOIN temporal_locations l ON h.dataset=l.dataset AND h.source_station_id=l.source_station_id
                  WHERE h.pollutant=? AND h.timestamp_local BETWEEN ? AND ?
                  GROUP BY l.area_name,h.source_station_id,h.timestamp_local,h.dataset
                ), area_hour AS (
                  SELECT area_name,timestamp_local,timestamp_utc,avg(station_value) mean,
                         count(station_value) active_station_count,any_value(unit) unit
                  FROM station_hour WHERE station_value IS NOT NULL
                  GROUP BY area_name,timestamp_local,timestamp_utc
                ), matched AS (
                  SELECT * FROM area_hour
                  QUALIFY count(DISTINCT area_name) OVER(PARTITION BY timestamp_utc)=?
                )
                SELECT area_name,hour(timestamp_local) hour_of_day,avg(mean) mean,median(mean) median,
                       quantile_cont(mean,.95) p95,count(*) matched_hour_count,
                       max(active_station_count) active_station_count,any_value(unit) unit
                FROM matched WHERE isodow(timestamp_local) BETWEEN 1 AND 5
                GROUP BY area_name,hour_of_day ORDER BY area_name,hour_of_day
            """, [paths,pollutant,start,end,len(cities)])
            names=[x[0] for x in result.description]; return [dict(zip(names,row)) for row in result.fetchall()]
        finally: connection.close()

    def _location_stations(self, selected, cities):
        connection = duckdb.connect(str(self.catalog.control_path), read_only=True)
        try:
            frame = connection.execute("SELECT dataset,source_station_id,city,admin1,country_name FROM station_dimension").fetchdf()
        finally:
            connection.close()
        frame = frame[frame["dataset"].isin({item.dataset for item in selected})].copy()
        if not cities:
            frame["area_name"] = frame["country_name"]
            return frame[["dataset","source_station_id","area_name"]]
        parts = []
        for city in cities:
            clean = frame["city"].fillna("").str.strip().str.casefold()
            admin = frame["admin1"].fillna("").str.strip().str.casefold()
            aliases = {city.casefold()}
            if city == "bangalore": aliases.add("bengaluru")
            if city == "beijing": aliases.add("北京")
            if city == "mexico city": aliases.add("cdmx")
            mask = clean.isin(aliases)
            if city == "london": mask |= admin.eq("greater london")
            part = frame[mask].copy(); part["area_name"] = city.title(); parts.append(part)
        if not parts: return frame.iloc[0:0][["dataset","source_station_id"]].assign(area_name=[])
        return pd.concat(parts,ignore_index=True)[["dataset","source_station_id","area_name"]].drop_duplicates()

    def _monthly(self, selected, cities, pollutant, start, end) -> list[dict]:
        paths = [str(path) for item in selected for path in (self.root / "query" / "quality_monthly" / f"dataset={item.dataset}").glob("year=*/month=*/*.parquet")]
        return _query_area(paths, cities, pollutant, start, end, "month_start", "country" if not cities else "city")

    def _daily(self, selected, cities, pollutant, start, end) -> list[dict]:
        paths = [str(path) for item in selected for path in (self.root / "query" / "area_daily" / f"dataset={item.dataset}").glob("year=*/month=*/*.parquet")]
        return _query_area(paths, cities, pollutant, start, end, "bucket_day", "country" if not cities else "city")

    def _hour_profile(self, selected, cities, pollutant, start, end) -> list[dict]:
        paths = [str(path) for item in selected for path in (self.root / "query" / "hourly" / f"dataset={item.dataset}").glob("year=*/month=*/*.parquet")]
        if not paths: return []
        control = duckdb.connect(str(self.catalog.control_path), read_only=True)
        try: dimension = control.execute("SELECT dataset,source_station_id,city,country_name FROM station_dimension").fetchdf()
        finally: control.close()
        ids = {item.dataset for item in selected}; dimension = dimension[dimension["dataset"].isin(ids)].copy()
        if cities:
            wanted = {_city_alias(item) for item in cities}
            dimension = dimension[dimension["city"].fillna("").map(_city_alias).isin(wanted)]
            dimension["area_name"] = dimension["city"]
        else: dimension["area_name"] = dimension["country_name"]
        connection = duckdb.connect(); connection.register("temporal_stations", dimension[["dataset","source_station_id","area_name"]])
        try:
            result = connection.execute("""
                WITH station_hour AS (
                  SELECT t.area_name,h.source_station_id,h.timestamp_local,
                         avg(h.analysis_value) FILTER(WHERE h.analysis_value>=0) station_value,
                         any_value(h.unit) unit
                  FROM read_parquet(?,hive_partitioning=true,union_by_name=true) h
                  JOIN temporal_stations t ON h.dataset=t.dataset AND h.source_station_id=t.source_station_id
                  WHERE h.pollutant=? AND h.timestamp_local BETWEEN ? AND ?
                  GROUP BY t.area_name,h.source_station_id,h.timestamp_local
                ), area_hour AS (
                  SELECT area_name,timestamp_local,avg(station_value) mean,count(station_value) active_station_count,any_value(unit) unit
                  FROM station_hour GROUP BY area_name,timestamp_local
                )
                SELECT area_name,hour(timestamp_local) hour_of_day,avg(mean) mean,median(mean) median,
                       quantile_cont(mean,.95) p95,count(*) valid_hour_count,max(active_station_count) active_station_count,any_value(unit) unit
                FROM area_hour GROUP BY area_name,hour_of_day ORDER BY area_name,hour_of_day
            """, [paths,pollutant,start,end])
            names=[x[0] for x in result.description]; return [dict(zip(names,row)) for row in result.fetchall()]
        finally: connection.close()

    def _hour_by_weekday_profile(self, selected, cities, pollutant, start, end, minimum_cell_samples) -> list[dict]:
        paths = [str(path) for item in selected for path in (self.root / "query" / "hourly" / f"dataset={item.dataset}").glob("year=*/month=*/*.parquet")]
        if not paths: return []
        control = duckdb.connect(str(self.catalog.control_path), read_only=True)
        try: dimension = control.execute("SELECT dataset,source_station_id,city,country_name FROM station_dimension").fetchdf()
        finally: control.close()
        ids = {item.dataset for item in selected}; dimension = dimension[dimension["dataset"].isin(ids)].copy()
        if cities:
            wanted = {_city_alias(item) for item in cities}
            dimension = dimension[dimension["city"].fillna("").map(_city_alias).isin(wanted)]
            dimension["area_name"] = dimension["city"]
        else: dimension["area_name"] = dimension["country_name"]
        connection = duckdb.connect(); connection.register("temporal_stations", dimension[["dataset","source_station_id","area_name"]])
        try:
            # Observed-only, unlike _hour_profile (which blends in policy-approved
            # fills via analysis_value) -- this operation's whole point is a
            # directly-measured diurnal-by-weekday picture, not a filled one.
            result = connection.execute("""
                WITH station_hour AS (
                  SELECT t.area_name,h.source_station_id,h.timestamp_local,
                         avg(h.observed_value) FILTER(WHERE h.value_status='observed' AND h.observed_value>=0) station_value,
                         any_value(h.unit) unit
                  FROM read_parquet(?,hive_partitioning=true,union_by_name=true) h
                  JOIN temporal_stations t ON h.dataset=t.dataset AND h.source_station_id=t.source_station_id
                  WHERE h.pollutant=? AND h.timestamp_local BETWEEN ? AND ?
                  GROUP BY t.area_name,h.source_station_id,h.timestamp_local
                ), area_hour AS (
                  SELECT area_name,timestamp_local,avg(station_value) mean,count(station_value) active_station_count,any_value(unit) unit
                  FROM station_hour WHERE station_value IS NOT NULL GROUP BY area_name,timestamp_local
                )
                SELECT area_name,isodow(timestamp_local) day_number,hour(timestamp_local) hour_of_day,
                       avg(mean) mean,count(*) sample_count,
                       max(active_station_count) active_station_count,any_value(unit) unit
                FROM area_hour GROUP BY area_name,day_number,hour_of_day
                ORDER BY area_name,day_number,hour_of_day
            """, [paths,pollutant,start,end])
            names=[x[0] for x in result.description]; rows=[dict(zip(names,row)) for row in result.fetchall()]
        finally: connection.close()
        day_names = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 5: "Friday", 6: "Saturday", 7: "Sunday"}
        for row in rows:
            row["day_of_week"] = day_names[row["day_number"]]
            row["day_number"] -= 1  # 0 (Monday) - 6 (Sunday), matching _day_profile's convention
            row["unmasked_mean"] = row["mean"]
            row["masked"] = row["sample_count"] < minimum_cell_samples
            row["mask_reason"] = f"fewer than {minimum_cell_samples} observed hours" if row["masked"] else None
            if row["masked"]:
                row["mean"] = None
            row["minimum_cell_samples"] = minimum_cell_samples
        return rows


def _query_area(paths, cities, pollutant, start, end, timestamp_column, geography) -> list[dict]:
    if not paths: return []
    connection=duckdb.connect()
    try:
        result=connection.execute(f"""
            SELECT dataset,country_name country,area_name,{timestamp_column} AS "timestamp",mean,coverage,
                   observed_fraction,imputed_fraction,active_station_count,unit
            FROM read_parquet(?,hive_partitioning=true,union_by_name=true)
            WHERE pollutant=? AND geography_level=? AND {timestamp_column} BETWEEN ? AND ?
            ORDER BY area_name,"timestamp"
        """,[paths,pollutant,geography,start,end])
        names=[x[0] for x in result.description]; rows=[dict(zip(names,row)) for row in result.fetchall()]
    finally: connection.close()
    if cities:
        wanted={_city_alias(item) for item in cities}; rows=[row for row in rows if _city_alias(row["area_name"] or "") in wanted]
    for row in rows: row["timestamp"]=row["timestamp"].isoformat()
    return rows


def _operation(text: str) -> str:
    if "jointly observed" in text and ("diurnal" in text or "hour" in text): return "joint_weekday_diurnal"
    if ("daytime" in text and "nighttime" in text) or "day versus night" in text: return "day_night"
    if re.search(r"hour-of-day\s+by\s+day-of-week|hour\s+of\s+day\s+by\s+day\s+of\s+week", text): return "hour_by_weekday"
    if "weekday" in text and "weekend" in text: return "weekday_weekend"
    if "hour-of-day" in text or "hour of day" in text or "diurnal" in text: return "hour_of_day"
    if "day-of-week" in text or "day of week" in text: return "day_of_week"
    if re.search(r"change[- ]?point", text): return "changepoint"
    if "moving average" in text: return "moving_average"
    if "month-over-month" in text or "percentage change" in text: return "percent_change"
    if "increase or decrease" in text or re.search(r"\bfrom\s+\w+\s+to\s+\w+",text): return "two_period_change"
    if "slope" in text: return "slope"
    if re.search(r"\b(?:three|top \d+) months? with the highest",text): return "highest_months"
    if "daily" in text: return "daily_series"
    return "monthly_series"


def _day_profile(rows):
    frame=pd.DataFrame(rows); frame["timestamp"]=pd.to_datetime(frame["timestamp"]); frame["day_number"]=frame["timestamp"].dt.weekday
    frame["day_of_week"]=frame["timestamp"].dt.day_name()
    output=[]
    for (area,number,label),g in frame.groupby(["area_name","day_number","day_of_week"]):
        output.append({"area_name":area,"day_of_week":label,"day_number":int(number),"mean":float(g["mean"].mean()),"valid_day_count":len(g),"unit":g["unit"].iloc[0]})
    return sorted(output,key=lambda row:(row["area_name"],row["day_number"]))


def _weekday_weekend(rows):
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["day_type"] = np.where(frame["timestamp"].dt.weekday < 5, "weekday", "weekend")
    output = []
    for (dataset, country, area, day_type), group in frame.groupby(["dataset", "country", "area_name", "day_type"]):
        output.append({
            "dataset": dataset, "country": country, "area_name": area, "day_type": day_type,
            "mean": float(group["mean"].mean()), "median": float(group["mean"].median()),
            "p95": float(group["mean"].quantile(.95)), "valid_day_count": int(len(group)),
            "mean_coverage": float(group["coverage"].mean()),
            "observed_fraction": float(group["observed_fraction"].mean()),
            "imputed_fraction": float(group["imputed_fraction"].mean()),
            "active_station_count": int(group["active_station_count"].max()), "unit": group["unit"].iloc[0],
        })
    by_area = {}
    for row in output: by_area.setdefault((row["dataset"], row["area_name"]), {})[row["day_type"]] = row
    for values in by_area.values():
        if "weekday" in values and "weekend" in values:
            difference = values["weekday"]["mean"] - values["weekend"]["mean"]
            for row in values.values(): row["weekday_minus_weekend"] = difference
    return sorted(output, key=lambda row: (row["area_name"], row["day_type"]))


def _moving_average(rows, window):
    frame=pd.DataFrame(rows); frame["timestamp"]=pd.to_datetime(frame["timestamp"]); output=[]
    for area,g in frame.groupby("area_name"):
        g=g.sort_values("timestamp").copy(); g["moving_average"]=g["mean"].rolling(window,min_periods=window).mean()
        for row in g.dropna(subset=["moving_average"]).to_dict("records"):
            output.append({"area_name":area,"timestamp":row["timestamp"].isoformat(),"mean":row["mean"],"moving_average":row["moving_average"],"window_days":window,"unit":row["unit"]})
    return output


def _slopes(rows):
    frame=pd.DataFrame(rows); output=[]
    for area,g in frame.groupby("area_name"):
        g=g.sort_values("timestamp"); y=g["mean"].astype(float).to_numpy(); x=np.arange(len(y),dtype=float)
        if len(y)<2: continue
        fit=sm.OLS(y,sm.add_constant(x)).fit()
        output.append({"area_name":area,"slope_per_month":float(fit.params[1]),"intercept":float(fit.params[0]),"r_squared":float(fit.rsquared),"p_value":float(fit.pvalues[1]) if len(y)>2 else None,"month_count":len(y),"direction":"increasing" if fit.params[1]>0 else "decreasing" if fit.params[1]<0 else "flat","unit":g["unit"].iloc[0]})
    return output


def _percent_changes(rows):
    frame=pd.DataFrame(rows); frame["timestamp"]=pd.to_datetime(frame["timestamp"]); output=[]
    for area,g in frame.groupby("area_name"):
        g=g.sort_values("timestamp"); previous=None
        for row in g.to_dict("records"):
            change=(row["mean"]-previous)/abs(previous) if previous not in (None,0) else None
            output.append({"area_name":area,"timestamp":row["timestamp"].isoformat(),"mean":row["mean"],"previous_mean":previous,"percentage_change":change,"unit":row["unit"]}); previous=row["mean"]
    return output


def _two_period_change(rows):
    frame=pd.DataFrame(rows); frame["timestamp"]=pd.to_datetime(frame["timestamp"]); output=[]
    for area,g in frame.groupby("area_name"):
        g=g.sort_values("timestamp"); first,last=g.iloc[0],g.iloc[-1]; diff=float(last["mean"]-first["mean"])
        output.append({"area_name":area,"start":first["timestamp"].isoformat(),"end":last["timestamp"].isoformat(),"start_mean":float(first["mean"]),"end_mean":float(last["mean"]),"absolute_change":diff,"percentage_change":float(diff/abs(first["mean"])) if first["mean"] else None,"direction":"increased" if diff>0 else "decreased" if diff<0 else "unchanged","unit":first["unit"]})
    return output


def _changepoints(rows: list[dict], minimum_segment_days: int) -> list[dict]:
    """Single strongest breakpoint per area via minimum two-segment squared error.

    Uses prefix sums so the search over every candidate split is O(n) rather
    than O(n^2); n is small (roughly a year of days) but there is no reason
    to pay the quadratic cost.
    """
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["area_name"], []).append(row)
    output = []
    for area, values in grouped.items():
        values.sort(key=lambda row: row["timestamp"])
        series = np.asarray([row["mean"] for row in values], dtype=float)
        n = len(series)
        base = {
            "dataset": values[0]["dataset"], "country": values[0]["country"], "area_name": area,
            "day_count": n, "minimum_segment_days": minimum_segment_days,
            "active_station_count": max((row.get("active_station_count") or 0) for row in values),
            "unit": values[0]["unit"],
        }
        if n < 2 * minimum_segment_days:
            output.append({
                **base, "insufficient_days": True, "breakpoint_date": None,
                "before_period_start": None, "before_period_end": None,
                "after_period_start": None, "after_period_end": None,
                "before_day_count": None, "after_day_count": None,
                "before_mean": None, "after_mean": None, "mean_shift": None,
            })
            continue
        prefix_sum = np.concatenate([[0.0], np.cumsum(series)])
        prefix_sq = np.concatenate([[0.0], np.cumsum(series ** 2)])
        total_sum, total_sq = prefix_sum[-1], prefix_sq[-1]
        best = None
        for split in range(minimum_segment_days, n - minimum_segment_days + 1):
            before_sum, before_sq = prefix_sum[split], prefix_sq[split]
            after_sum, after_sq = total_sum - before_sum, total_sq - before_sq
            before_n, after_n = split, n - split
            loss = (before_sq - before_sum ** 2 / before_n) + (after_sq - after_sum ** 2 / after_n)
            if best is None or loss < best[0]:
                best = (loss, split, before_sum / before_n, after_sum / after_n)
        _, split, before_mean, after_mean = best
        output.append({
            **base, "insufficient_days": False, "breakpoint_date": values[split]["timestamp"],
            "before_period_start": values[0]["timestamp"], "before_period_end": values[split - 1]["timestamp"],
            "after_period_start": values[split]["timestamp"], "after_period_end": values[-1]["timestamp"],
            "before_day_count": split, "after_day_count": n - split,
            "before_mean": float(before_mean), "after_mean": float(after_mean),
            "mean_shift": float(after_mean - before_mean),
        })
    return output


def _highest_months(rows,n):
    grouped={}
    for row in rows: grouped.setdefault(row["area_name"],[]).append(row)
    output=[]
    for area,values in grouped.items():
        for rank,row in enumerate(sorted(values,key=lambda x:x["mean"],reverse=True)[:n],1): output.append({**row,"rank":rank})
    return output


def _answer(rows,operation,pollutant,start,end):
    period=f"{start:%d %b %Y} to {end:%d %b %Y}"
    if operation=="slope":
        values="; ".join(f"{row['area_name']}: {row['slope_per_month']:.2f} {row['unit']} per month ({row['direction']}, R²={row['r_squared']:.2f})" for row in rows)
        return f"For {period}, the equal-station monthly {pollutant} trends are {values}."
    if operation=="two_period_change":
        row=rows[0]; return f"{row['area_name']} {pollutant} {row['direction']} from {row['start_mean']:.2f} to {row['end_mean']:.2f} {row['unit']} ({row['percentage_change']:.1%}) between the first and last requested months."
    if operation=="changepoint":
        parts=[]
        for row in rows:
            if row.get("insufficient_days"):
                parts.append(f"{row['area_name']} has only {row['day_count']} valid days, fewer than the {row['minimum_segment_days']} required on each side of a candidate breakpoint, so no changepoint could be estimated.")
                continue
            parts.append(
                f"{row['area_name']}'s largest daily mean shift is on {row['breakpoint_date'][:10]}: "
                f"{row['before_mean']:.2f} {row['unit']} over the preceding {row['before_day_count']} days versus "
                f"{row['after_mean']:.2f} {row['unit']} over the following {row['after_day_count']} days "
                f"({row['mean_shift']:+.2f} {row['unit']})."
            )
        return (
            f"For {period}, I searched for the single daily-mean breakpoint that minimizes combined before/after "
            f"squared error, requiring at least {rows[0]['minimum_segment_days']} days on each side. " + " ".join(parts)
        )
    if operation=="hour_by_weekday":
        areas=sorted({row["area_name"] for row in rows})
        masked_count=sum(row["masked"] for row in rows)
        return (
            f"For {period}, I built a directly-measured {pollutant} hour-of-day by day-of-week grid "
            f"(24 hours × 7 weekdays = {len(rows)} cells) for {', '.join(areas)}. "
            f"{masked_count} of {len(rows)} cells have fewer than {rows[0]['minimum_cell_samples']} observed hours "
            "and are marked as masked rather than shown as a computed value."
        )
    labels={"hour_of_day":"hour-of-day profile","joint_weekday_diurnal":"jointly observed weekday diurnal profile","day_night":"daytime/nighttime comparison","day_of_week":"day-of-week profile","weekday_weekend":"weekday/weekend comparison","moving_average":"moving-average series","percent_change":"month-over-month changes","highest_months":"highest-month ranking","daily_series":"daily series","monthly_series":"monthly series"}
    return f"I calculated {len(rows)} {pollutant} {labels.get(operation,operation)} results for {period} using equal-station area values."


def _window(text):
    match=re.search(r"(\d+)[- ]day moving average",text); return int(match.group(1)) if match else 7


def _coverage_threshold(text, default):
    match = re.search(r"(?:at least|minimum|below)\s+(\d+(?:\.\d+)?)\s*%\s*coverage", text)
    return float(match.group(1)) / 100 if match else default


def _add_day_night_difference(rows):
    grouped = {}
    for row in rows: grouped.setdefault(row["area_name"], {})[row["period_of_day"]] = row
    for values in grouped.values():
        if "daytime" in values and "nighttime" in values:
            difference = values["daytime"]["mean"] - values["nighttime"]["mean"]
            for row in values.values(): row["daytime_minus_nighttime"] = difference
    return rows


def _has_period(text):
    return bool(re.search(r"\b20\d{2}\b|\b(?:january|february|march|april|may|june|july|august|september|october|november|december|last|latest|previous)\b", text))


def _number_word(text,default):
    match=re.search(r"top\s+(\d+)",text)
    if match:return int(match.group(1))
    return {"three":3,"five":5,"ten":10}.get(next((w for w in ("three","five","ten") if w in text),""),default)
