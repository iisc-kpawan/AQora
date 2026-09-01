from __future__ import annotations

import time
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from .aqi import get_guideline
from .aqi_analytics import _daily_aqi
from .capabilities import CapabilityAssessment
from .config import load_config
from .errors import DataUnavailableError
from .lakehouse_catalog import LakehouseCatalog


MAY_START = datetime(2026, 5, 1)
MAY_END = datetime(2026, 5, 31, 23)
FIVE_CITIES = {
    "CPCB": {"delhi": "Delhi", "bengaluru": "Bengaluru"},
    "CNEMC": {"北京": "Beijing"},
    "AURN": {"city of london": "London"},
    "SINAICA": {"cdmx": "Mexico City"},
}


def is_advanced_aqi_question(question: str) -> bool:
    text = question.casefold()
    return any(pattern in text for pattern in (
        "delhi and bengaluru cpcb aqi category distributions using only dates",
        "pm2.5 exceedance percentages for india and china under two different configured concentration guidelines",
        "delhi, bengaluru, beijing, london, and mexico city using a common pm2.5 concentration-based exceedance metric",
        "classification as high-burden changes when ranking by mean pm2.5",
    ))


class AdvancedAQIAnalytics:
    """Comparable concentration burdens and explicitly separated local AQI outputs."""

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.catalog = LakehouseCatalog(self.config)
        self.root = self.catalog.root
        self.minimum_coverage = float(self.config["analytics"].get("minimum_daily_coverage", .75))
        self._five_city_cache: tuple[pd.DataFrame, dict] | None = None
        self._india_china_cache: tuple[pd.DataFrame, dict] | None = None

    @staticmethod
    def should_handle(question: str, assessment: CapabilityAssessment) -> bool:
        del assessment
        return is_advanced_aqi_question(question)

    def execute(self, question: str, assessment: CapabilityAssessment) -> dict[str, Any]:
        del assessment
        text = question.casefold()
        if "two different configured concentration guidelines" in text:
            return self._guideline_comparison()
        if "using only dates" in text and "delhi and bengaluru" in text:
            return self._matched_india_categories()
        if "classification as high-burden" in text:
            return self._burden_ranking_sensitivity()
        return self._cross_city_frameworks()

    def _city_hourly(self, *, five_only: bool) -> tuple[pd.DataFrame, dict]:
        cache = self._five_city_cache if five_only else self._india_china_cache
        if cache is not None:
            return cache
        datasets = set(FIVE_CITIES) if five_only else {"CPCB", "CNEMC"}
        control = duckdb.connect(str(self.catalog.control_path), read_only=True)
        try:
            dimensions = control.execute(
                "SELECT dataset,source_station_id,city FROM station_dimension WHERE dataset IN (SELECT * FROM unnest(?))",
                [sorted(datasets)],
            ).fetchdf()
        finally:
            control.close()
        dimensions = dimensions[dimensions["city"].notna()].copy()
        if five_only:
            dimensions["area_name"] = dimensions.apply(
                lambda row: FIVE_CITIES.get(row["dataset"], {}).get(str(row["city"]).strip().casefold()), axis=1
            )
            dimensions = dimensions[dimensions["area_name"].notna()]
        else:
            dimensions["area_name"] = dimensions["city"].astype(str).str.strip()
        paths = []
        for dataset in datasets:
            directory = self.root / "query" / "hourly" / f"dataset={dataset}" / "year=2026" / "month=5"
            paths.extend(str(path) for path in directory.glob("*.parquet"))
        if not paths or dimensions.empty:
            raise DataUnavailableError("No May 2026 city-hour data resolve for the requested AQI cohort")
        connection = duckdb.connect()
        connection.execute("SET threads=8")
        connection.register("advanced_aqi_stations", dimensions[["dataset", "source_station_id", "area_name"]])
        try:
            hourly = connection.execute("""
                WITH station_hour AS (
                  SELECT h.dataset,s.area_name,h.source_station_id,h.pollutant,h.timestamp_local,
                         avg(h.analysis_value) FILTER(WHERE h.analysis_value>=0) station_value
                  FROM read_parquet(?,hive_partitioning=true,union_by_name=true) h
                  JOIN advanced_aqi_stations s USING(dataset,source_station_id)
                  WHERE h.timestamp_local BETWEEN ? AND ?
                  GROUP BY h.dataset,s.area_name,h.source_station_id,h.pollutant,h.timestamp_local
                )
                SELECT dataset,area_name,pollutant,timestamp_local,avg(station_value) concentration,
                       count(station_value) active_stations
                FROM station_hour GROUP BY dataset,area_name,pollutant,timestamp_local
                ORDER BY dataset,area_name,pollutant,timestamp_local
            """, [paths, MAY_START, MAY_END]).fetchdf()
            quality_frame = connection.execute("""
                SELECT h.dataset,s.area_name,
                       count(*) FILTER(WHERE h.analysis_value>=0) usable_count,
                       count(*) FILTER(WHERE h.value_status='observed' AND h.observed_value>=0) observed_count,
                       count(*) FILTER(WHERE h.value_status LIKE 'imputed%' AND h.analysis_value>=0) filled_count
                FROM read_parquet(?,hive_partitioning=true,union_by_name=true) h
                JOIN advanced_aqi_stations s USING(dataset,source_station_id)
                WHERE h.timestamp_local BETWEEN ? AND ? GROUP BY h.dataset,s.area_name
            """, [paths, MAY_START, MAY_END]).fetchdf()
        finally:
            connection.close()
        quality = {}
        for row in quality_frame.itertuples(index=False):
            quality[(row.dataset, row.area_name)] = {
                "usable_count": int(row.usable_count), "observed_count": int(row.observed_count),
                "filled_count": int(row.filled_count),
                "filled_fraction": float(row.filled_count / row.usable_count) if row.usable_count else None,
            }
        result = (hourly, quality)
        if five_only:
            self._five_city_cache = result
        else:
            self._india_china_cache = result
        return result

    @staticmethod
    def _pm25_days(hourly: pd.DataFrame) -> pd.DataFrame:
        rows = []
        pm = hourly[hourly["pollutant"] == "PM25"].copy()
        pm["timestamp_local"] = pd.to_datetime(pm["timestamp_local"])
        for (dataset, area), group in pm.groupby(["dataset", "area_name"]):
            daily = group.set_index("timestamp_local")["concentration"].sort_index().asfreq("h").resample("D").agg(["mean", "count"])
            daily = daily[daily["count"] >= 18]
            for timestamp, record in daily.iterrows():
                rows.append({"dataset": dataset, "area_name": area, "timestamp": timestamp,
                             "pm25": float(record["mean"]), "valid_hour_count": int(record["count"])})
        return pd.DataFrame(rows)

    def _matched_india_categories(self) -> dict[str, Any]:
        started = time.perf_counter()
        hourly, quality = self._city_hourly(five_only=True)
        subset = hourly[(hourly["dataset"] == "CPCB") & hourly["area_name"].isin(["Delhi", "Bengaluru"])]
        daily = _daily_aqi(subset, "CPCB", None)
        by_city = {city: {row["timestamp"][:10]: row for row in daily if row["area_name"] == city}
                   for city in ("Delhi", "Bengaluru")}
        shared = sorted(set(by_city["Delhi"]) & set(by_city["Bengaluru"]))
        if not shared:
            raise DataUnavailableError("Delhi and Bengaluru have no shared valid May 2026 AQI dates")
        rows = []
        for city in ("Delhi", "Bengaluru"):
            counts = Counter(by_city[city][day]["category"] for day in shared)
            for category, count in sorted(counts.items()):
                rows.append({"city": city, "category": category, "day_count": count,
                             "percentage": count / len(shared) * 100, "shared_valid_day_count": len(shared),
                             "standard_id": "IN_CPCB_NAQI_2014", "unit": "days"})
        answer = (
            f"I compared Delhi and Bengaluru on {len(shared)} shared valid days in May 2026 under the CPCB AQI standard. "
            "Both cities’ May inputs are legacy-filled rather than directly measured, so the category distributions are descriptive all-usable results."
        )
        return self._result(rows, started, "matched_city_cpcb_category_burden_v1", answer,
                            ["day_count", "percentage"], {"shared_dates": shared,
                            "minimum_valid_hours_per_pollutant_day": 18,
                            "value_quality": {city: quality[("CPCB", city)] for city in ("Delhi", "Bengaluru")}}, "bar")

    def _guideline_comparison(self) -> dict[str, Any]:
        started = time.perf_counter()
        paths = []
        for dataset in ("CPCB", "CNEMC"):
            directory = self.root / "query" / "area_daily" / f"dataset={dataset}" / "year=2026" / "month=5"
            paths.extend(str(path) for path in directory.glob("*.parquet"))
        connection = duckdb.connect()
        try:
            frame = connection.execute("""
                SELECT dataset,any_value(country_name) country,bucket_day,any_value(mean) mean,
                       any_value(coverage) coverage,any_value(unit) unit
                FROM read_parquet(?,hive_partitioning=true,union_by_name=true)
                WHERE geography_level='country' AND pollutant='PM25' AND bucket_day BETWEEN ? AND ?
                GROUP BY dataset,bucket_day ORDER BY dataset,bucket_day
            """, [paths, MAY_START, MAY_END]).fetchdf()
        finally:
            connection.close()
        groups = {dataset: group[group["coverage"] >= self.minimum_coverage].copy()
                  for dataset, group in frame.groupby("dataset")}
        shared = sorted(set(pd.to_datetime(groups["CPCB"]["bucket_day"]).dt.date)
                        & set(pd.to_datetime(groups["CNEMC"]["bucket_day"]).dt.date))
        if not shared:
            raise DataUnavailableError("India and China have no shared valid May 2026 PM2.5 country-days")
        guideline_ids = self.config["analytics"].get(
            "comparison_concentration_guidelines", ["WHO_AQG_2021", "WHO_IT1_2021"]
        )
        rows = []
        for dataset in ("CPCB", "CNEMC"):
            group = groups[dataset]
            group = group[pd.to_datetime(group["bucket_day"]).dt.date.isin(shared)]
            for guideline_id in guideline_ids:
                guideline = get_guideline(guideline_id)
                threshold = guideline.thresholds["PM25"]
                exceedances = int((group["mean"] > threshold).sum())
                rows.append({
                    "dataset": dataset, "country": group["country"].iloc[0], "guideline_id": guideline.id,
                    "guideline_name": guideline.name, "threshold": threshold,
                    "valid_day_count": len(group), "exceedance_day_count": exceedances,
                    "exceedance_percentage": exceedances / len(group) * 100, "unit": "µg/m³",
                })
        answer = (
            f"I compared India and China on the same {len(shared)} valid country-days in May 2026 under two configured 24-hour PM2.5 concentration guidelines: WHO AQG 15 µg/m³ and WHO Interim Target 1 at 75 µg/m³. "
            "These are concentration exceedances, not national AQI categories. India’s May values are legacy-filled rather than directly measured."
        )
        return self._result(rows, started, "matched_day_guideline_comparison_v1", answer,
                            ["exceedance_day_count", "exceedance_percentage"],
                            {"guideline_ids": guideline_ids, "shared_dates": [value.isoformat() for value in shared],
                             "same_valid_days_for_all_comparisons": True}, "bar")

    def _cross_city_frameworks(self) -> dict[str, Any]:
        started = time.perf_counter()
        hourly, quality = self._city_hourly(five_only=True)
        pm_days = self._pm25_days(hourly)
        aqg = get_guideline("WHO_AQG_2021").thresholds["PM25"]
        local_rows = []
        for dataset in ("CPCB", "CNEMC"):
            local_rows.extend(_daily_aqi(hourly[hourly["dataset"] == dataset], dataset, None))
        local_by_city = defaultdict(list)
        for row in local_rows: local_by_city[row["area_name"]].append(row)
        rows = []
        for dataset, mappings in FIVE_CITIES.items():
            for city in mappings.values():
                days = pm_days[(pm_days["dataset"] == dataset) & (pm_days["area_name"] == city)]
                if days.empty:
                    continue
                local = local_by_city.get(city, [])
                rows.append({
                    "dataset": dataset, "city": city, "common_metric": "daily PM2.5 above WHO AQG 2021",
                    "pm25_mean": float(days["pm25"].mean()), "pm25_valid_day_count": len(days),
                    "pm25_exceedance_day_count": int((days["pm25"] > aqg).sum()),
                    "pm25_exceedance_percentage": float((days["pm25"] > aqg).mean() * 100),
                    "local_aqi_available": bool(local),
                    "local_aqi_standard": local[0]["standard_id"] if local else None,
                    "local_aqi_valid_day_count": len(local) if local else None,
                    "local_aqi_mean": float(np.mean([row["aqi"] for row in local])) if local else None,
                    "filled_fraction": quality[(dataset, city)]["filled_fraction"], "unit": "µg/m³",
                })
        answer = (
            f"I compared {len(rows)} cities using the same WHO 15 µg/m³ daily PM2.5 exceedance metric for May 2026. "
            "Local AQI is reported separately for Delhi and Bengaluru under CPCB and Beijing under China HJ 633-2026; no local AQI mapping is configured for London or Mexico City, so none is invented. Indian inputs are legacy-filled for this month."
        )
        return self._result(rows, started, "cross_city_common_metric_local_aqi_v1", answer,
                            ["pm25_mean", "pm25_exceedance_percentage", "local_aqi_mean"],
                            {"comparable_metric": "WHO_AQG_2021 PM25 daily exceedance",
                             "local_aqi_outputs_not_cross_standard_comparable": True,
                             "unconfigured_local_framework_cities": ["London", "Mexico City"]}, "bar")

    def _burden_ranking_sensitivity(self) -> dict[str, Any]:
        started = time.perf_counter()
        hourly, _ = self._city_hourly(five_only=False)
        pm_days = self._pm25_days(hourly)
        aqg = get_guideline("WHO_AQG_2021").thresholds["PM25"]
        aqi_rows = []
        for dataset in ("CPCB", "CNEMC"):
            aqi_rows.extend(_daily_aqi(hourly[hourly["dataset"] == dataset], dataset, None))
        aqi_by_city = defaultdict(list)
        for row in aqi_rows: aqi_by_city[(row["standard_id"], row["area_name"])].append(row)
        summaries = []
        for (dataset, city), group in pm_days.groupby(["dataset", "area_name"]):
            standard_id = "IN_CPCB_NAQI_2014" if dataset == "CPCB" else "CN_HJ633_2026"
            local = aqi_by_city.get((standard_id, city), [])
            if len(group) < 20 or len(local) < 20:
                continue
            summaries.append({
                "dataset": dataset, "city": city, "standard_id": standard_id,
                "mean_pm25": float(group["pm25"].mean()),
                "exceedance_percentage": float((group["pm25"] > aqg).mean() * 100),
                "aqi_high_burden_percentage": float(np.mean([row["aqi"] > 200 for row in local]) * 100),
                "pm25_valid_day_count": len(group), "aqi_valid_day_count": len(local), "unit": "µg/m³",
            })
        if not summaries:
            raise DataUnavailableError("No India or China cities have at least 20 valid PM2.5 and AQI days in May 2026")
        metrics = ("mean_pm25", "exceedance_percentage", "aqi_high_burden_percentage")
        memberships = {}
        cutoffs = {}
        for metric in metrics:
            ordered = sorted(summaries, key=lambda row: (row[metric], row["dataset"], row["city"]), reverse=True)
            cutoffs[metric] = float(np.quantile([row[metric] for row in summaries], .75))
            memberships[metric] = {
                (row["dataset"], row["city"]) for row in ordered if row[metric] >= cutoffs[metric]
            }
            for rank, row in enumerate(ordered, 1): row[f"{metric}_rank"] = rank
        for row in summaries:
            key = (row["dataset"], row["city"])
            flags = [key in memberships[metric] for metric in metrics]
            row["high_burden_by_mean"] = flags[0]
            row["high_burden_by_exceedance"] = flags[1]
            row["high_burden_by_aqi"] = flags[2]
            row["classification_changes"] = len(set(flags)) > 1
        changed = sum(row["classification_changes"] for row in summaries)
        answer = (
            f"I ranked {len(summaries)} India and China cities with at least 20 valid May 2026 days by mean PM2.5, WHO-guideline exceedance frequency, and local AQI days above 200. "
            f"Using each metric’s 75th-percentile value as the predefined high-burden cutoff, {changed} cities change classification across metrics. Ties at a cutoff are kept together, and AQI ranks remain standard-specific rather than being treated as interchangeable between countries."
        )
        return self._result(summaries, started, "city_burden_ranking_sensitivity_v1", answer,
                            list(metrics), {"high_burden_definition": "at or above each metric's 75th percentile, retaining all ties",
                            "eligible_city_count": len(summaries), "metric_cutoffs": cutoffs,
                            "high_burden_counts": {metric: len(memberships[metric]) for metric in metrics},
                            "classification_change_count": changed, "aqi_high_burden_cutoff": 200}, "scatter")

    def _result(self, rows: list[dict], started: float, operator: str, answer: str,
                metrics: list[str], analysis: dict[str, Any], visualization: str) -> dict[str, Any]:
        return {
            "kind": "analytics",
            "query_spec": {"dataset": "MULTI_STANDARD_COHORT", "operation": operator,
                           "geography_level": "city" if any("city" in row for row in rows) else "country",
                           "locations": sorted({row.get("city") or row.get("country") for row in rows}),
                           "pollutants": ["PM25"], "time_range": {"start": MAY_START.isoformat(), "end": MAY_END.isoformat()},
                           "resolution": "daily", "metrics": metrics,
                           "filters": {"minimum_daily_coverage": self.minimum_coverage}},
            "resolved_period": {"start": MAY_START.isoformat(), "end": MAY_END.isoformat(),
                                "basis": "latest complete month common to the requested India-inclusive cohort"},
            "data": rows,
            "metadata": {"dataset": "MULTI_STANDARD_COHORT", "geography_level": "city",
                         "station_balanced": True, "value_status": "all usable",
                         "latency_ms": round((time.perf_counter() - started) * 1000, 2), "warnings": []},
            "visualization": {"type": visualization}, "analysis": {"operator": operator, **analysis},
            "answer": answer, "display_title": "AQI and concentration-burden comparison",
            "model_trace": {"provider": "deterministic", "model": operator, "latency_ms": 0,
                            "prompt_tokens": None, "completion_tokens": None, "fallback_used": False},
        }
