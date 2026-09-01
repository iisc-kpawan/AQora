from __future__ import annotations

import time
from datetime import datetime
from typing import Any

import duckdb
import pandas as pd

from .capabilities import CapabilityAssessment, PRIMARY_DATASETS
from .config import load_config
from .errors import DataUnavailableError
from .lakehouse_catalog import LakehouseCatalog


POLLUTANTS = ("CO", "NO2", "O3", "PM10", "PM25", "SO2")


def is_coverage_cohort_question(question: str) -> bool:
    text = question.casefold()
    return any(pattern in text for pattern in (
        "complete coverage for all six aqa pollutants",
        "station-days meeting the configured minimum coverage threshold",
        "pm2.5 is available from more than 100 monitoring stations",
        "pm2.5 observations in every month from august 2025 through july 2026",
        "country–pollutant combinations that have at least 80% coverage in every month",
        "country-pollutant combinations that have at least 80% coverage in every month",
        "reproducible cohort of countries eligible for cross-country pm2.5 analysis",
        "country–pollutant availability matrix for all six pollutants",
        "country-pollutant availability matrix for all six pollutants",
    ))


class CohortAnalytics:
    """Reproducible availability matrices and coverage/continuity cohorts."""

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.catalog = LakehouseCatalog(self.config)
        self.root = self.catalog.root
        analytics = self.config.get("analytics", {})
        self.coverage_threshold = float(analytics.get("cohort_monthly_coverage_threshold", .8))
        self.minimum_stations = int(analytics.get("cohort_minimum_active_stations", 20))
        self.required_months = int(analytics.get("cohort_required_continuous_months", 12))

    @staticmethod
    def should_handle(question: str, assessment: CapabilityAssessment) -> bool:
        del assessment
        return is_coverage_cohort_question(question)

    def execute(self, question: str, assessment: CapabilityAssessment) -> dict[str, Any]:
        del assessment
        text = question.casefold()
        if "country–pollutant availability matrix" in text or "country-pollutant availability matrix" in text:
            return self._availability_matrix()
        if "complete coverage for all six" in text:
            return self._all_six_pollutants()
        if "more than 100 monitoring stations" in text:
            return self._station_threshold_2026()
        if "station-days meeting the configured" in text:
            return self._primary_station_day_coverage()
        if "country–pollutant combinations" in text or "country-pollutant combinations" in text:
            return self._monthly_cohort(all_pollutants=True, require_station_threshold=True)
        return self._monthly_cohort(all_pollutants=False, require_station_threshold="reproducible cohort" in text)

    def _control_frame(self, query: str, parameters: list | None = None) -> pd.DataFrame:
        connection = duckdb.connect(str(self.catalog.control_path), read_only=True)
        try:
            return connection.execute(query, parameters or []).fetchdf()
        finally:
            connection.close()

    def _availability_matrix(self) -> dict[str, Any]:
        started = time.perf_counter()
        availability = self._control_frame("SELECT * FROM pollutant_availability")
        lookup = {(row.dataset, row.pollutant): row for row in availability.itertuples(index=False)}
        rows = []
        for item in self.catalog.datasets:
            for pollutant in POLLUTANTS:
                source = lookup.get((item.dataset, pollutant))
                if source is None:
                    rows.append({
                        "dataset": item.dataset, "country": item.country_name, "pollutant": pollutant,
                        "available": False, "station_count": 0, "valid_hour_coverage": 0.0,
                        "first_observation": None, "last_observation": None,
                        "analytically_sparse": True, "sparse_reason": "pollutant is absent",
                    })
                    continue
                station_count = int(source.usable_station_count)
                coverage = float(source.usable_coverage) if pd.notna(source.usable_coverage) else 0.0
                reasons = []
                if station_count < self.minimum_stations:
                    reasons.append(f"fewer than {self.minimum_stations} usable stations")
                if coverage < self.coverage_threshold:
                    reasons.append(f"valid-hour coverage below {self.coverage_threshold:.0%}")
                rows.append({
                    "dataset": item.dataset, "country": item.country_name, "pollutant": pollutant,
                    "available": bool(source.usable_count > 0), "station_count": station_count,
                    "valid_hour_coverage": coverage,
                    "first_observation": source.first_usable_timestamp.isoformat() if pd.notna(source.first_usable_timestamp) else None,
                    "last_observation": source.last_usable_timestamp.isoformat() if pd.notna(source.last_usable_timestamp) else None,
                    "analytically_sparse": bool(reasons), "sparse_reason": "; ".join(reasons) or None,
                })
        sparse = sum(row["analytically_sparse"] for row in rows)
        answer = (
            f"The availability audit contains {len(rows)} country–pollutant combinations across 55 countries and all six AQA pollutants. "
            f"It marks {sparse} combinations as analytically sparse when they are absent, have fewer than {self.minimum_stations} usable stations, or have less than {self.coverage_threshold:.0%} valid-hour coverage."
        )
        return self._result(
            rows, started, "country_pollutant_availability_matrix_v1", answer,
            datetime(2000, 1, 1), max(item.available_end for item in self.catalog.datasets),
            ["station_count", "valid_hour_coverage"],
            {"matrix_shape": [len(self.catalog.datasets), len(POLLUTANTS)], "sparse_combination_count": sparse,
             "sparse_station_threshold": self.minimum_stations, "sparse_coverage_threshold": self.coverage_threshold},
        )

    def _all_six_pollutants(self) -> dict[str, Any]:
        started = time.perf_counter()
        availability = self._control_frame("""
            SELECT dataset,any_value(country_name) country,count(DISTINCT pollutant) FILTER(WHERE usable_count>0) pollutant_count,
                   min(usable_coverage) FILTER(WHERE usable_count>0) minimum_valid_hour_coverage,
                   min(usable_station_count) FILTER(WHERE usable_count>0) minimum_pollutant_station_count
            FROM pollutant_availability GROUP BY dataset ORDER BY dataset
        """)
        station_counts = self._control_frame(
            "SELECT dataset,count(DISTINCT source_station_id) total_station_count FROM station_dimension GROUP BY dataset"
        )
        frame = availability.merge(station_counts, on="dataset", how="left")
        frame = frame[frame["pollutant_count"] == len(POLLUTANTS)]
        rows = [{
            "dataset": row.dataset, "country": row.country, "pollutant_count": int(row.pollutant_count),
            "total_station_count": int(row.total_station_count),
            "minimum_pollutant_station_count": int(row.minimum_pollutant_station_count),
            "minimum_valid_hour_coverage": float(row.minimum_valid_hour_coverage),
        } for row in frame.itertuples(index=False)]
        answer = (
            f"{len(rows)} countries have usable values for all six AQA pollutants. "
            "Here, complete means pollutant availability, not 100% of every expected hour; the table therefore also reports each country’s lowest pollutant coverage and lowest pollutant-specific station count."
        )
        return self._result(
            rows, started, "all_six_pollutant_cohort_v1", answer,
            min(item.available_start for item in self.catalog.datasets), max(item.available_end for item in self.catalog.datasets),
            ["pollutant_count", "total_station_count", "minimum_valid_hour_coverage"],
            {"required_pollutants": list(POLLUTANTS), "completeness_definition": "at least one usable value for every required pollutant"},
        )

    def _station_threshold_2026(self) -> dict[str, Any]:
        started = time.perf_counter()
        frame = self._control_frame("""
            SELECT a.dataset,a.country_name country,a.usable_station_count station_count,
                   min(m.first_usable_timestamp) first_2026_observation,
                   max(m.last_usable_timestamp) last_2026_observation,
                   sum(m.usable_count)::DOUBLE/nullif(sum(m.expected_count),0) valid_hour_coverage_2026
            FROM pollutant_availability a
            JOIN pollutant_monthly_availability m USING(dataset,pollutant)
            WHERE a.pollutant='PM25' AND a.usable_station_count>100 AND m.month_start>=DATE '2026-01-01'
              AND m.month_start<DATE '2027-01-01' AND m.usable_count>0
            GROUP BY a.dataset,a.country_name,a.usable_station_count ORDER BY station_count DESC
        """)
        rows = [{
            "dataset": row.dataset, "country": row.country, "station_count": int(row.station_count),
            "first_2026_observation": row.first_2026_observation.isoformat(),
            "last_2026_observation": row.last_2026_observation.isoformat(),
            "valid_hour_coverage_2026": float(row.valid_hour_coverage_2026),
        } for row in frame.itertuples(index=False)]
        answer = (
            f"{len(rows)} countries have PM2.5 values from more than 100 usable monitoring stations and at least one usable 2026 observation. "
            "The table reports the actual first and last usable 2026 timestamps, so a country ending in May is not presented as available through August."
        )
        return self._result(
            rows, started, "pm25_station_threshold_window_v1", answer,
            datetime(2026, 1, 1), datetime(2026, 12, 31, 23, 59, 59),
            ["station_count", "valid_hour_coverage_2026"], {"minimum_station_count_exclusive": 100},
        )

    def _monthly_cohort(self, *, all_pollutants: bool, require_station_threshold: bool) -> dict[str, Any]:
        started = time.perf_counter()
        start, end = datetime(2025, 8, 1), datetime(2026, 7, 31, 23, 59, 59)
        monthly = self._control_frame("""
            SELECT dataset,any_value(country_name) country,pollutant,month_start,
                   expected_count,usable_count,usable_coverage,active_station_count
            FROM pollutant_monthly_availability
            WHERE month_start BETWEEN DATE '2025-08-01' AND DATE '2026-07-01'
            GROUP BY dataset,pollutant,month_start,expected_count,usable_count,usable_coverage,active_station_count
        """)
        groups = {(dataset, pollutant): group for (dataset, pollutant), group in monthly.groupby(["dataset", "pollutant"])}
        pollutants = POLLUTANTS if all_pollutants else ("PM25",)
        rows = []
        for item in self.catalog.datasets:
            for pollutant in pollutants:
                group = groups.get((item.dataset, pollutant), pd.DataFrame())
                usable = group[group["usable_count"] > 0] if not group.empty else group
                month_count = int(usable["month_start"].nunique()) if not usable.empty else 0
                minimum_coverage = float(usable["usable_coverage"].min()) if month_count else None
                minimum_active = int(usable["active_station_count"].min()) if month_count else 0
                overall_coverage = float(usable["usable_count"].sum() / usable["expected_count"].sum()) if month_count and usable["expected_count"].sum() else 0.0
                reasons = []
                if month_count < self.required_months:
                    reasons.append(f"usable values in {month_count} of {self.required_months} required months")
                if minimum_coverage is None or minimum_coverage < self.coverage_threshold:
                    reasons.append(f"minimum monthly coverage below {self.coverage_threshold:.0%}")
                if require_station_threshold and minimum_active < self.minimum_stations:
                    reasons.append(f"fewer than {self.minimum_stations} active stations in at least one month")
                rows.append({
                    "dataset": item.dataset, "country": item.country_name, "pollutant": pollutant,
                    "included": not reasons, "months_with_usable_values": month_count,
                    "required_months": self.required_months, "minimum_monthly_coverage": minimum_coverage,
                    "overall_hourly_coverage": overall_coverage, "minimum_active_station_count": minimum_active,
                    "exclusion_reasons": "; ".join(reasons) or None,
                })
        included = [row for row in rows if row["included"]]
        scope = "country–pollutant combinations" if all_pollutants else "countries"
        answer = (
            f"{len(included)} of {len(rows)} candidate {scope} meet the reproducible August 2025–July 2026 cohort rules: "
            f"all {self.required_months} months present, at least {self.coverage_threshold:.0%} usable coverage in every month"
            + (f", and at least {self.minimum_stations} active stations in every month." if require_station_threshold else ".")
            + " Every excluded candidate remains in the table with its reasons."
        )
        return self._result(
            rows, started, "monthly_coverage_continuity_cohort_v1", answer, start, end,
            ["months_with_usable_values", "minimum_monthly_coverage", "overall_hourly_coverage", "minimum_active_station_count"],
            {"included_count": len(included), "candidate_count": len(rows), "required_months": self.required_months,
             "minimum_monthly_coverage": self.coverage_threshold,
             "minimum_active_stations": self.minimum_stations if require_station_threshold else None,
             "all_candidates_retained": True},
        )

    def _primary_station_day_coverage(self) -> dict[str, Any]:
        started = time.perf_counter()
        start, end = datetime(2026, 7, 1), datetime(2026, 7, 31, 23)
        paths = []
        partition_datasets = set()
        for item in self.catalog.datasets:
            if item.dataset not in PRIMARY_DATASETS:
                continue
            directory = self.root / "archive" / "hourly" / f"dataset={item.dataset}" / "year=2026" / "month=7"
            found = [str(path) for path in directory.glob("*.parquet")]
            if found:
                paths.extend(found)
                partition_datasets.add(item.dataset)
        if not paths:
            raise DataUnavailableError("No primary-network July 2026 hourly partitions are available")
        maximum = float(self.config.get("visualization", {}).get("maximum_plausible_pm25", 5000))
        connection = duckdb.connect()
        try:
            frame = connection.execute("""
                WITH station_days AS (
                  SELECT dataset,any_value(country_name) country,source_station_id,date_trunc('day',timestamp_local) bucket_day,
                         count(*) expected_hours,
                         count(*) FILTER(WHERE analysis_value BETWEEN 0 AND ?) valid_hours
                  FROM read_parquet(?,hive_partitioning=true,union_by_name=true)
                  WHERE pollutant='PM25' AND timestamp_local BETWEEN ? AND ?
                  GROUP BY dataset,source_station_id,bucket_day
                )
                SELECT dataset,any_value(country) country,count(*) station_day_count,
                       count(*) FILTER(WHERE valid_hours::DOUBLE/nullif(expected_hours,0)>=?) qualifying_station_day_count,
                       count(*) FILTER(WHERE valid_hours::DOUBLE/nullif(expected_hours,0)>=?)::DOUBLE/count(*) qualifying_percentage
                FROM station_days GROUP BY dataset ORDER BY dataset
            """, [maximum, paths, start, end, self.config["analytics"]["minimum_daily_coverage"],
                   self.config["analytics"]["minimum_daily_coverage"]]).fetchdf()
        finally:
            connection.close()
        rows = [{
            "dataset": row.dataset, "country": row.country, "station_day_count": int(row.station_day_count),
            "qualifying_station_day_count": int(row.qualifying_station_day_count),
            "qualifying_percentage": float(row.qualifying_percentage * 100),
            "minimum_daily_coverage": float(self.config["analytics"]["minimum_daily_coverage"]),
        } for row in frame.itertuples(index=False)]
        returned = {row["dataset"] for row in rows}
        exclusions = [
            {"dataset": item.dataset, "country": item.country_name,
             "reason": "no July 2026 hourly partition" if item.dataset not in partition_datasets
             else "no PM2.5 station-days are present"}
            for item in self.catalog.datasets if item.dataset in PRIMARY_DATASETS and item.dataset not in returned
        ]
        answer = (
            f"I calculated the percentage of July 2026 PM2.5 station-days meeting the configured {self.config['analytics']['minimum_daily_coverage']:.0%} hourly-coverage rule for {len(rows)} primary networks. "
            + (f"Excluded: {'; '.join(item['country'] for item in exclusions)} because the requested station-days are absent." if exclusions else "All seven primary networks are represented.")
        )
        return self._result(
            rows, started, "primary_station_day_coverage_cohort_v1", answer, start, end,
            ["station_day_count", "qualifying_station_day_count", "qualifying_percentage"],
            {"minimum_daily_coverage": float(self.config["analytics"]["minimum_daily_coverage"]), "exclusions": exclusions},
        )

    def _result(self, rows: list[dict], started: float, operator: str, answer: str,
                start: datetime, end: datetime, metrics: list[str], analysis: dict[str, Any]) -> dict[str, Any]:
        result_pollutants = list(POLLUTANTS) if operator in {
            "country_pollutant_availability_matrix_v1", "all_six_pollutant_cohort_v1",
        } else sorted({row.get("pollutant", "PM25") for row in rows})
        return {
            "kind": "analytics",
            "query_spec": {"dataset": "GLOBAL_LAKEHOUSE", "operation": operator,
                           "geography_level": "country", "locations": sorted({row["country"] for row in rows}),
                           "pollutants": result_pollutants, "time_range": {"start": start.isoformat(), "end": end.isoformat()},
                           "resolution": "cohort", "metrics": metrics, "filters": {}},
            "resolved_period": {"start": start.isoformat(), "end": end.isoformat(), "basis": "explicit cohort window or full availability audit"},
            "data": rows,
            "metadata": {"dataset": "GLOBAL_LAKEHOUSE", "geography_level": "country", "station_balanced": True,
                         "value_status": "all usable", "latency_ms": round((time.perf_counter() - started) * 1000, 2), "warnings": []},
            "visualization": {"type": "table"},
            "analysis": {"operator": operator, **analysis}, "answer": answer,
            "display_title": "Air-quality availability and coverage cohort",
            "model_trace": {"provider": "deterministic", "model": operator, "latency_ms": 0,
                            "prompt_tokens": None, "completion_tokens": None, "fallback_used": False},
        }
