from __future__ import annotations

import math
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

from .capabilities import CapabilityAssessment, _coverage_sensitivity_supported, is_flexible_country_cohort
from .config import load_config
from .errors import DataUnavailableError
from .global_analytics import POLLUTANTS, _period, _top_n
from .lakehouse_catalog import LakehouseCatalog


METRIC_PATTERNS = (
    (r"\bmeans?\b|\baverages?\b", "mean"), (r"\bmedian\b|middle daily value", "median"),
    (r"\bp\s*90\b|90th[- ]percentile", "p90"),
    (r"\bp\s*95\b|95th[- ]percentile", "p95"),
    (r"standard deviation|\bstd\b", "std"),
    (r"coefficient of variation|\bcv\b", "coefficient_of_variation"),
    (r"\bmaximum\b|\bmax\b", "max"), (r"\bminimum\b|\bmin\b", "min"),
    (r"data coverage|\bcoverage\b", "coverage"),
    (r"missing percentage|missingness|missing fraction", "missing_percentage"),
    (r"imputed percentage|imputation rate|imputed fraction", "imputed_percentage"),
    (r"observed percentage|observed fraction", "observed_percentage"),
    (r"valid[- ]day count|number of valid days|valid days", "valid_day_count"),
    (r"active stations?|station count|number of stations", "active_station_count"),
    (r"contributing cities|number of cities", "contributing_city_count"),
)


def requested_quality_metrics(question: str) -> list[str]:
    text = question.casefold()
    metrics = [name for pattern, name in METRIC_PATTERNS if re.search(pattern, text)]
    if "regardless of station count" in text and "active_station_count" in metrics:
        metrics.remove("active_station_count")
    return metrics


def is_quality_cube_question(question: str) -> bool:
    text = question.casefold()
    metrics = requested_quality_metrics(question)
    return len(metrics) >= 2 or any(term in text for term in (
        "quality summary", "city-balanced", "city weighted", "city-weighted",
        "population-unweighted city", "valid-day coverage",
    ))


class QualityAnalytics:
    """General multi-metric analytics over the materialized monthly area-quality cube."""

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.catalog = LakehouseCatalog(self.config)
        self.root = self.catalog.root

    def should_handle(self, question: str, assessment: CapabilityAssessment) -> bool:
        if assessment.intent in {"forecast", "aqi_or_exceedance", "ranking"}:
            return False
        if _coverage_sensitivity_supported(question.casefold()):
            # A dedicated coverage-threshold sensitivity request (ThresholdAnalytics)
            # rather than a generic multi-metric quality-cube lookup, even though it
            # also mentions "mean" and "coverage".
            return False
        return is_quality_cube_question(question)

    def execute(self, question: str, assessment: CapabilityAssessment) -> dict[str, Any]:
        started = time.perf_counter()
        text = question.casefold()
        metrics = requested_quality_metrics(question)
        if not metrics:
            metrics = ["mean", "coverage"]
        pollutants = [name for pattern, name in POLLUTANTS if re.search(pattern, text)]
        if len(pollutants) > 1:
            raise DataUnavailableError("The exact multi-pollutant derived comparison operator is not enabled yet")
        pollutant = pollutants[0] if pollutants else "PM25"
        start, end = _period(text, self.catalog, assessment.supported_datasets)
        if re.search(r"\blast\s+\w+\s+complete months?\b", text):
            raise DataUnavailableError(
                "Exact daily-distribution statistics over a relative multi-month window are not enabled yet"
            )
        if start.day != 1 or end.day < 28:
            raise DataUnavailableError("The multi-metric quality cube currently supports complete calendar months")
        month_count = (end.year - start.year) * 12 + end.month - start.month + 1
        monthly_output = month_count == 1 or any(term in text for term in (
            "each month", "monthly", "by month", "month by month", "trajectory", "trend", "line chart",
        ))
        if month_count > 1 and not monthly_output:
            raise DataUnavailableError(
                "This request combines daily-distribution metrics across multiple months; the exact multi-month distribution operator is not enabled yet"
            )

        selected = list(self.catalog.selected(assessment.supported_datasets))
        candidates = list(selected)
        generic_countries = is_flexible_country_cohort(question, assessment.intent)
        geography_level = "country"
        aggregation = "equal_station"
        if any(term in text for term in ("city-balanced", "city weighted", "city-weighted", "population-unweighted city")):
            geography_level, aggregation = "country_city_balanced", "equal_city_after_equal_station"
        city_names = _requested_cities(text)
        if city_names:
            geography_level, aggregation = "city", "equal_station_within_city"

        cohort = self.catalog.resolve_cohort(
            [item.dataset for item in selected], pollutant, start, end,
            value_status="any", require_all_months=True,
        )
        cohort_exclusions = [item for item in cohort if not item.included]
        if cohort_exclusions and not generic_countries and not city_names:
            detail = "; ".join(f"{item.country_name}: {item.reason}" for item in cohort_exclusions)
            raise DataUnavailableError("The explicitly requested quality cohort is incomplete. " + detail)
        included_ids = {item.dataset for item in cohort if item.included}
        selected = [item for item in selected if item.dataset in included_ids]

        paths = [
            str(path) for item in selected
            for path in (self.root / "query" / "quality_monthly" / f"dataset={item.dataset}").glob("year=*/month=*/*.parquet")
        ]
        if not paths:
            raise DataUnavailableError("The requested datasets are not yet published in the multi-metric quality cube")
        connection = duckdb.connect()
        try:
            result = connection.execute("""
                SELECT dataset, country_code, country_name, geography_level, area_id, area_name,
                       pollutant, month_start, mean, median, p90, p95, std, min, max,
                       valid_day_count, expected_day_count, days_coverage_50, days_coverage_75,
                       days_coverage_90, expected_hour_count, valid_hour_count, observed_count,
                       imputed_count, missing_count, invalid_count, coverage, observed_fraction,
                       imputed_fraction, active_station_count, contributing_city_count, unit,
                       aggregation_method
                FROM read_parquet(?, hive_partitioning=true, union_by_name=true)
                WHERE pollutant=? AND geography_level=?
                  AND month_start BETWEEN date_trunc('month', ?) AND date_trunc('month', ?)
                  AND valid_day_count > 0
                ORDER BY month_start, country_name, area_name
            """, [paths, pollutant, geography_level, start, end])
            names = [item[0] for item in result.description]
            raw = [dict(zip(names, row)) for row in result.fetchall()]
        finally:
            connection.close()
        if city_names:
            wanted = {_city_alias(name) for name in city_names}
            raw = [row for row in raw if _city_alias(row["area_name"] or "") in wanted]

        selected_ids = {item.dataset for item in selected}
        present_ids = {row["dataset"] for row in raw}
        missing = [item.country_name for item in selected if item.dataset not in present_ids]
        if missing and not generic_countries and not city_names:
            raise DataUnavailableError("The explicitly requested quality cohort is incomplete. Missing: " + ", ".join(missing))
        if city_names:
            found = {_city_alias(row["area_name"] or "") for row in raw}
            absent = [name for name in city_names if _city_alias(name) not in found]
            if absent:
                raise DataUnavailableError("No complete-month quality result is available for: " + ", ".join(absent))
        if not raw:
            raise DataUnavailableError("No multi-metric quality rows match the requested pollutant, geography, and period")

        rows = [_project(row, metrics) for row in raw]
        if assessment.intent == "ranking":
            primary = next((name for name in metrics if name in rows[0] and isinstance(rows[0][name], (int, float))), "mean")
            rows.sort(key=lambda row: (row.get(primary) is not None, row.get(primary) or -math.inf, row["area_name"]), reverse=True)
            rows = rows[:_top_n(text)]
            for index, row in enumerate(rows, 1): row["rank"] = index
        cube_exclusions = [
            {"country": item.country_name, "reason": f"no {geography_level} quality row is available"}
            for item in selected if item.dataset not in present_ids
        ]
        exclusions = [
            {"country": item.country_name, "reason": item.reason, "reason_code": item.reason_code}
            for item in cohort_exclusions
        ] + cube_exclusions
        answer = _answer(rows, metrics, pollutant, start, end, aggregation, exclusions)
        visualization = "line" if month_count > 1 else "bar" if assessment.intent == "ranking" else "table"
        return {
            "kind": "analytics",
            "query_spec": {
                "dataset": "GLOBAL_QUALITY_CUBE" if len(selected) > 1 else selected[0].dataset,
                "operation": assessment.intent, "geography_level": geography_level,
                "locations": city_names or [item.country_name for item in selected],
                "pollutants": [pollutant], "time_range": {"start": start.isoformat(), "end": end.isoformat()},
                "resolution": "monthly", "metrics": metrics,
                "filters": {"aggregation_method": aggregation},
            },
            "resolved_period": {"start": start.isoformat(), "end": end.isoformat(), "basis": "explicit complete calendar months"},
            "data": rows,
            "metadata": {
                "dataset": "GLOBAL_QUALITY_CUBE", "geography_level": geography_level,
                "station_count": max((row.get("active_station_count") or 0 for row in rows), default=0),
                "value_status": "observed_and_policy_approved_filled", "station_balanced": True,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "warnings": [], "country_count": len({row["dataset"] for row in rows}),
            },
            "visualization": {"type": visualization, "metrics": metrics},
            "analysis": {
                "operator": "area_monthly_quality_v1", "aggregation_method": aggregation,
                "metrics": metrics, "candidate_country_count": len(candidates),
                "included_country_count": len(present_ids), "excluded_country_count": len(exclusions),
                "exclusions": exclusions,
            },
            "answer": answer, "display_title": f"{pollutant} · multi-metric quality",
            "model_trace": {"provider": "deterministic", "model": "area_monthly_quality_v1", "latency_ms": 0,
                            "prompt_tokens": None, "completion_tokens": None, "fallback_used": False},
        }


def _project(row: dict, metrics: list[str]) -> dict:
    output = {
        "dataset": row["dataset"], "country": row["country_name"], "area_name": row["area_name"],
        "geography_level": row["geography_level"], "timestamp": row["month_start"].isoformat(),
        "pollutant": row["pollutant"], "unit": row["unit"], "aggregation_method": row["aggregation_method"],
    }
    values = dict(row)
    expected = row["expected_hour_count"] or 0
    values["coefficient_of_variation"] = row["std"] / abs(row["mean"]) if row["std"] is not None and row["mean"] else None
    values["missing_percentage"] = (row["missing_count"] + row["invalid_count"]) / expected if expected else None
    values["imputed_percentage"] = row["imputed_count"] / expected if expected else None
    values["observed_percentage"] = row["observed_count"] / expected if expected else None
    for metric in metrics:
        output[metric] = values.get(metric)
    for name in ("active_station_count", "contributing_city_count", "valid_day_count", "expected_day_count"):
        if name not in output and (name in metrics or name in ("active_station_count", "valid_day_count")):
            output[name] = values.get(name)
    return output


def _answer(rows: list[dict], metrics: list[str], pollutant: str, start: datetime, end: datetime,
            aggregation: str, exclusions: list[dict]) -> str:
    period = f"{start:%b %Y}" if start.year == end.year and start.month == end.month else f"{start:%b %Y}–{end:%b %Y}"
    labels = {
        "mean": "mean", "median": "median", "p90": "P90", "p95": "P95",
        "std": "standard deviation", "coefficient_of_variation": "coefficient of variation",
        "coverage": "coverage", "missing_percentage": "missing percentage",
        "imputed_percentage": "imputed percentage", "observed_percentage": "observed percentage",
        "valid_day_count": "valid-day count", "active_station_count": "active-station count",
        "contributing_city_count": "contributing-city count", "max": "maximum", "min": "minimum",
    }
    method = "cities receive equal weight after stations are balanced within each city" if aggregation == "equal_city_after_equal_station" else "stations receive equal weight within each area"
    answer = f"I calculated {', '.join(labels[item] for item in metrics)} for {len(rows)} {pollutant} area-month result{'s' if len(rows) != 1 else ''} in {period}; {method}."
    if len(rows) == 1:
        row = rows[0]
        pieces = []
        for metric in metrics:
            value = row.get(metric)
            if value is None: continue
            if "percentage" in metric or metric == "coverage": pieces.append(f"{labels[metric]} {value:.1%}")
            elif metric.endswith("count"): pieces.append(f"{labels[metric]} {int(value):,}")
            else: pieces.append(f"{labels[metric]} {value:.2f}" + (f" {row['unit']}" if metric not in {"coefficient_of_variation"} else ""))
        answer += f" {row['area_name']}: " + "; ".join(pieces) + "."
    if exclusions:
        detail = "; ".join(f"{item['country']}: {item['reason']}" for item in exclusions[:8])
        answer += f" {len(exclusions)} candidate countr{'y was' if len(exclusions) == 1 else 'ies were'} excluded ({detail})."
    answer += " The table contains the requested metrics and provenance-derived quality percentages."
    return answer


def _requested_cities(text: str) -> list[str]:
    aliases = ("delhi", "bengaluru", "bangalore", "beijing", "london", "paris", "berlin", "mexico city", "chennai", "mumbai")
    return [name for name in aliases if re.search(rf"\b{re.escape(name)}\b", text)]


def _city_alias(value: str) -> str:
    aliases = {"bangalore": "bengaluru", "北京": "beijing"}
    clean = value.strip().casefold()
    return aliases.get(clean, clean)
