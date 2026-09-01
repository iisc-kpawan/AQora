from __future__ import annotations

import calendar
import math
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb

from .aqi import DEFAULT_GUIDELINE, get_guideline
from .capabilities import CapabilityAssessment, is_flexible_country_cohort
from .config import load_config
from .errors import DataUnavailableError
from .global_analytics import POLLUTANTS, _period, _percentage_constraint, _top_n
from .lakehouse_catalog import LakehouseCatalog
from .quality_analytics import _city_alias, _requested_cities


class ThresholdAnalytics:
    """Deterministic concentration-threshold burden and episode analytics."""

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.catalog = LakehouseCatalog(self.config)
        self.root = self.catalog.root

    @staticmethod
    def should_handle(question: str, assessment: CapabilityAssessment) -> bool:
        text = question.casefold()
        if "aqi" in text and not any(term in text for term in ("guideline", "concentration threshold", "exceedance")):
            return False
        return any(term in text for term in ("exceed", "exceedance", "guideline", "episode"))

    def execute(self, question: str, assessment: CapabilityAssessment) -> dict[str, Any]:
        started = time.perf_counter()
        text = question.casefold()
        if "station-day" in text or "station day" in text:
            raise DataUnavailableError("Station-day threshold analytics are not enabled yet; the current calculation supports city or country daily averages with equal weight for each monitoring station")
        pollutants = [name for pattern, name in POLLUTANTS if re.search(pattern, text)]
        if len(pollutants) > 1:
            raise DataUnavailableError("A threshold request must resolve to one pollutant")
        pollutant = pollutants[0] if pollutants else "PM25"
        guideline = get_guideline(DEFAULT_GUIDELINE)
        threshold = _explicit_threshold(text)
        if threshold is None:
            threshold = guideline.thresholds.get(pollutant)
        if threshold is None:
            raise DataUnavailableError(f"{guideline.name} does not define a supported daily threshold for {pollutant}")
        threshold_source = "explicit_question" if _explicit_threshold(text) is not None else guideline.id
        start, end = _period(text, self.catalog, assessment.supported_datasets)
        candidates = list(self.catalog.selected(assessment.supported_datasets))
        flexible = is_flexible_country_cohort(question, assessment.intent) or "city-level" in text
        cities = _requested_cities(text)
        geography = "city" if cities or "city-level" in text else "country"
        cohort = self.catalog.resolve_cohort(
            [item.dataset for item in candidates], pollutant, start, end,
            value_status="any", require_all_months=not cities,
        )
        exclusions = [item for item in cohort if not item.included]
        if exclusions and not flexible and not cities:
            detail = "; ".join(f"{item.country_name}: {item.reason}" for item in exclusions)
            raise DataUnavailableError("The explicitly requested threshold cohort is incomplete. " + detail)
        included = {item.dataset for item in cohort if item.included}
        selected = [item for item in candidates if item.dataset in included]
        paths = [
            str(path) for item in selected
            for path in (self.root / "query" / "area_daily" / f"dataset={item.dataset}").glob("year=*/month=*/*.parquet")
        ]
        if not paths:
            raise DataUnavailableError("No recent daily area layer overlaps the requested scope")
        minimum_coverage = _percentage_constraint(text)
        if minimum_coverage is None:
            minimum_coverage = float(self.config.get("analytics", {}).get("minimum_daily_coverage", .75))
        connection = duckdb.connect()
        try:
            result = connection.execute("""
                SELECT dataset, country_name country, area_name, bucket_day, mean, coverage,
                       observed_fraction, imputed_fraction, active_station_count, unit
                FROM read_parquet(?, hive_partitioning=true, union_by_name=true)
                WHERE pollutant=? AND geography_level=? AND bucket_day BETWEEN ? AND ?
                  AND coverage>=?
                ORDER BY dataset, area_name, bucket_day
            """, [paths, pollutant, geography, start, end, minimum_coverage])
            names = [item[0] for item in result.description]
            daily = [dict(zip(names, row)) for row in result.fetchall()]
        finally:
            connection.close()
        if cities:
            wanted = {_city_alias(item) for item in cities}
            daily = [row for row in daily if _city_alias(row["area_name"] or "") in wanted]
            found = {_city_alias(row["area_name"] or "") for row in daily}
            absent = [item for item in cities if _city_alias(item) not in found]
            if absent:
                raise DataUnavailableError("No valid daily threshold series is available for: " + ", ".join(absent))
        if not daily:
            raise DataUnavailableError("No daily values satisfy the requested period and coverage rule")

        episode_request = "episode" in text
        if episode_request:
            rows = _episodes(daily, threshold, pollutant)
            if not rows:
                raise DataUnavailableError(f"No {pollutant} episode exceeded {threshold:g} {daily[0]['unit']} in the valid daily series")
            if "longest" in text:
                rows.sort(key=lambda row: (row["duration_days"], row["severity"]), reverse=True)
            else:
                rows.sort(key=lambda row: (row["severity"], row["duration_days"]), reverse=True)
            requested = _top_n(text)
            if "longest" in text and not re.search(r"top\s+\d+|three|five|ten", text): requested = 1
            rows = rows[:requested]
            answer = _episode_answer(rows, pollutant, threshold, guideline, minimum_coverage)
            operation, visualization = "episodes", "bar" if len(rows) > 1 else "table"
        else:
            rows = _burden(daily, threshold, pollutant)
            if assessment.intent == "ranking" or any(term in text for term in ("rank", "top ", "most ")):
                ranking_metric = "exceedance_count" if re.search(r"most .*exceedance days", text) else "exceedance_percentage"
                rows.sort(key=lambda row: (row[ranking_metric], row["mean_exceedance"]), reverse=True)
                rows = rows[:_top_n(text)]
                for index, row in enumerate(rows, 1): row["rank"] = index
                visualization = "bar"
            else:
                visualization = "table"
            answer = _burden_answer(rows, pollutant, threshold, guideline, minimum_coverage)
            operation = "exceedance_burden"
        exclusion_rows = [{"country": item.country_name, "reason": item.reason, "reason_code": item.reason_code} for item in exclusions]
        if exclusion_rows:
            answer += " Excluded: " + "; ".join(f"{item['country']} ({item['reason']})" for item in exclusion_rows[:8]) + "."
        warnings = []
        if any((row.get("imputed_fraction") or 0) > .5 for row in rows):
            warnings.append("At least one result is based mostly on policy-approved filled values; use the provenance columns when interpreting it.")
        return {
            "kind": "analytics",
            "query_spec": {
                "dataset": "GLOBAL_AREA_DAILY" if len(candidates) > 1 else candidates[0].dataset,
                "operation": operation, "geography_level": geography,
                "locations": cities or [item.country_name for item in candidates], "pollutants": [pollutant],
                "time_range": {"start": start.isoformat(), "end": end.isoformat()}, "resolution": "daily",
                "metrics": (["duration_days", "peak", "mean_exceedance", "severity", "coverage", "observed_fraction", "imputed_fraction"]
                            if operation == "episodes" else ["exceedance_count", "exceedance_percentage", "mean_exceedance", "coverage"]),
                "filters": {"threshold": threshold, "minimum_coverage": minimum_coverage},
            },
            "resolved_period": {"start": start.isoformat(), "end": end.isoformat(), "basis": "explicit period and daily coverage rule"},
            "data": rows,
            "metadata": {
                "dataset": "GLOBAL_AREA_DAILY", "geography_level": geography,
                "station_count": max((row.get("active_station_count") or 0 for row in rows), default=0),
                "value_status": "observed_and_policy_approved_filled", "station_balanced": True,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2), "warnings": warnings,
                "guideline": {"id": guideline.id, "name": guideline.name, "source_url": guideline.source_url},
            },
            "visualization": {"type": visualization},
            "analysis": {
                "operator": "daily_threshold_v1", "threshold": threshold,
                "threshold_source": threshold_source, "minimum_daily_coverage": minimum_coverage,
                "candidate_country_count": len(candidates), "included_country_count": len(included),
                "exclusions": exclusion_rows,
            },
            "answer": answer, "display_title": f"{pollutant} · threshold analysis",
            "model_trace": {"provider": "deterministic", "model": "daily_threshold_v1", "latency_ms": 0,
                            "prompt_tokens": None, "completion_tokens": None, "fallback_used": False},
        }


def _burden(daily: list[dict], threshold: float, pollutant: str) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in daily: grouped.setdefault((row["dataset"], row["country"], row["area_name"]), []).append(row)
    output = []
    for (dataset, country, area), values in grouped.items():
        exceed = [row for row in values if row["mean"] > threshold]
        excesses = [row["mean"] - threshold for row in exceed]
        output.append({
            "dataset": dataset, "country": country, "area_name": area, "pollutant": pollutant,
            "threshold": threshold, "unit": values[0]["unit"], "valid_day_count": len(values),
            "exceedance_count": len(exceed), "exceedance_percentage": len(exceed) / len(values),
            "mean_exceedance": sum(excesses) / len(excesses) if excesses else 0.0,
            "peak": max(row["mean"] for row in values),
            "coverage": sum(row["coverage"] for row in values) / len(values),
            "observed_fraction": sum(row["observed_fraction"] for row in values) / len(values),
            "imputed_fraction": sum(row["imputed_fraction"] for row in values) / len(values),
            "active_station_count": max(row["active_station_count"] for row in values),
        })
    return output


def _episodes(daily: list[dict], threshold: float, pollutant: str) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in daily: grouped.setdefault((row["dataset"], row["country"], row["area_name"]), []).append(row)
    output = []
    for (dataset, country, area), values in grouped.items():
        values.sort(key=lambda row: row["bucket_day"])
        current: list[dict] = []
        groups: list[list[dict]] = []
        for row in values:
            if row["mean"] <= threshold:
                if current: groups.append(current); current = []
                continue
            if current and row["bucket_day"].date() != (current[-1]["bucket_day"] + timedelta(days=1)).date():
                groups.append(current); current = []
            current.append(row)
        if current: groups.append(current)
        for group in groups:
            excesses = [row["mean"] - threshold for row in group]
            mean_excess = sum(excesses) / len(excesses)
            output.append({
                "dataset": dataset, "country": country, "area_name": area, "pollutant": pollutant,
                "start": group[0]["bucket_day"].isoformat(), "end": group[-1]["bucket_day"].isoformat(),
                "duration_days": len(group), "threshold": threshold, "unit": group[0]["unit"],
                "peak": max(row["mean"] for row in group), "mean_exceedance": mean_excess,
                "severity": len(group) * mean_excess,
                "coverage": sum(row["coverage"] for row in group) / len(group),
                "observed_fraction": sum(row["observed_fraction"] for row in group) / len(group),
                "imputed_fraction": sum(row["imputed_fraction"] for row in group) / len(group),
                "value_provenance": _provenance_label(group),
                "active_station_count": max(row["active_station_count"] for row in group),
            })
    return output


def _provenance_label(rows: list[dict]) -> str:
    observed = sum(row["observed_fraction"] for row in rows) / len(rows)
    imputed = sum(row["imputed_fraction"] for row in rows) / len(rows)
    if imputed == 0:
        return "observed"
    if observed == 0:
        return "policy_approved_filled"
    return "observed_and_policy_approved_filled"


def _burden_answer(rows: list[dict], pollutant: str, threshold: float, guideline, coverage: float) -> str:
    if len(rows) == 1:
        row = rows[0]
        return (f"{row['area_name']} exceeded the {threshold:g} {row['unit']} daily {pollutant} threshold on "
                f"{row['exceedance_count']} of {row['valid_day_count']} valid days ({row['exceedance_percentage']:.1%}). "
                f"Mean exceedance magnitude was {row['mean_exceedance']:.2f} {row['unit']}. "
                f"Days required at least {coverage:.0%} coverage; the threshold is from {guideline.name}.")
    return (f"I calculated daily {pollutant} exceedance count, valid-day percentage, and mean magnitude for {len(rows)} areas "
            f"using {threshold:g} {rows[0]['unit']} and at least {coverage:.0%} daily coverage. The table includes the denominator for every area.")


def _episode_answer(rows: list[dict], pollutant: str, threshold: float, guideline, coverage: float) -> str:
    row = rows[0]
    return (f"The leading {pollutant} exceedance episode was in {row['area_name']} from {row['start'][:10]} through "
            f"{row['end'][:10]}: {row['duration_days']} consecutive days, peak {row['peak']:.2f} {row['unit']}, "
            f"mean exceedance {row['mean_exceedance']:.2f} {row['unit']}, and severity {row['severity']:.2f}. "
            f"It uses the {threshold:g} {row['unit']} {guideline.name} threshold and a {coverage:.0%} daily coverage rule. "
            f"Its contributing values were {row['observed_fraction']:.1%} measured and {row['imputed_fraction']:.1%} policy-approved filled values.")


def _explicit_threshold(text: str) -> float | None:
    patterns = (
        r"(?:threshold|guideline)(?:\s+of|\s*=|\s+at)?\s*(\d+(?:\.\d+)?)",
        r"(?:above|over|exceeding)\s+(\d+(?:\.\d+)?)\s*(?:µg|ug|mg)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match: return float(match.group(1))
    return None
