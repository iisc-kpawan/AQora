from __future__ import annotations

import calendar
from collections import Counter
import math
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

from .capabilities import CapabilityAssessment, is_flexible_country_cohort
from .config import load_config
from .errors import DataUnavailableError
from .lakehouse_catalog import LakehouseCatalog


POLLUTANTS = (
    (r"pm\s*2(?:\.|\s*)5|\bpm25\b", "PM25"), (r"\bpm\s*10\b", "PM10"),
    (r"\bno\s*2\b", "NO2"), (r"\bso\s*2\b", "SO2"),
    (r"\bo\s*3\b|\bozone\b", "O3"), (r"\bco\b", "CO"),
)
MONTHS = {name.casefold(): number for number, name in enumerate(calendar.month_name) if name}
MONTHS.update({name.casefold(): number for number, name in enumerate(calendar.month_abbr) if name})
CITY_TERMS = ("delhi", "mumbai", "chennai", "kolkata", "bengaluru", "bangalore", "beijing", "shanghai", "guangzhou", "shenzhen", "london", "seoul")


class GlobalAnalytics:
    """Deterministic country analytics over refined lakehouse Parquet layers."""

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.catalog = LakehouseCatalog(self.config)
        self.root = self.catalog.root

    def should_handle(self, question: str, assessment: CapabilityAssessment) -> bool:
        text = question.casefold()
        if assessment.intent in {"metadata", "forecast", "aqi_or_exceedance"}:
            return False
        if any(city in text for city in CITY_TERMS):
            return False
        if any(term in text for term in ("countries", "country", "national", "india", "china", "united states", "france", "germany", "mexico", "united kingdom")):
            return True
        return any(dataset not in {"CPCB", "CNEMC"} for dataset in assessment.supported_datasets)

    def execute(self, question: str, assessment: CapabilityAssessment) -> dict[str, Any]:
        started = time.perf_counter()
        text = question.casefold()
        pollutant = next((name for pattern, name in POLLUTANTS if re.search(pattern, text)), "PM25")
        start, end = _period(text, self.catalog, assessment.supported_datasets)
        selected = list(self.catalog.selected(assessment.supported_datasets))
        rank_countries = assessment.intent == "ranking" and "countr" in text
        if rank_countries and len(selected) == 1 and selected[0].dataset == "CPCB":
            selected = list(self.catalog.datasets)

        observed_only = "imputed" not in text and ("observed" in text or assessment.intent in {"ranking", "comparison"})
        flexible_cohort = is_flexible_country_cohort(question, assessment.intent)
        cohort = self.catalog.resolve_cohort(
            [item.dataset for item in selected], pollutant, start, end,
            value_status="observed" if observed_only else "any",
            minimum_coverage=_percentage_constraint(text),
            minimum_stations=_station_constraint(text),
            require_all_months=not flexible_cohort,
        )
        exclusions = [item for item in cohort if not item.included]
        exclusion_counts = dict(Counter(item.reason_code for item in exclusions if item.reason_code))
        if exclusions and not flexible_cohort:
            detail = "; ".join(f"{item.country_name}: {item.reason}" for item in exclusions[:8])
            raise DataUnavailableError(f"The explicitly requested country cohort is incomplete. {detail}")
        included_ids = {item.dataset for item in cohort if item.included}
        selected = [item for item in selected if item.dataset in included_ids]
        if not selected:
            raise DataUnavailableError("No loaded country satisfies the requested period, provenance, coverage, and station constraints")
        paths = []
        for item in selected:
            layer = "query/hourly" if start >= item.hourly_start else "archive/hourly"
            current = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            while current <= end:
                directory = self.root / layer / f"dataset={item.dataset}" / f"year={current.year}" / f"month={current.month}"
                paths.extend(str(path) for path in directory.glob("*.parquet"))
                current = current.replace(year=current.year + 1, month=1) if current.month == 12 else current.replace(month=current.month + 1)
        if not paths:
            raise DataUnavailableError("No hourly lakehouse partitions overlap the requested country and period")

        resolution = _resolution(text)
        bucket = {"hourly": "hour", "daily": "day", "monthly": "month", "yearly": "year"}[resolution]
        value = "observed_value" if observed_only else "analysis_value"
        asks_series = any(term in text for term in ("daily", "by day", "hourly", "by hour", "month by month", "by month", "trend", "over time"))
        daily_extreme_ranking = assessment.intent == "ranking" and bool(re.search(r"\bdaily\s+(?:maximum|max)\b", text))
        whole_period = (assessment.intent in {"ranking", "comparison"} and not daily_extreme_ranking) or (assessment.intent == "statistics" and not asks_series) or resolution == "yearly" and start.year == end.year
        bucket_expression = "TIMESTAMP '1970-01-01'" if whole_period else f"date_trunc('{bucket}', timestamp_local)"
        expected_hours = max(1, int((end - start).total_seconds() // 3600) + 1) if whole_period else {
            "hourly": 1, "daily": 24, "monthly": 744, "yearly": 8760,
        }[resolution]
        minimum_coverage = _percentage_constraint(text)
        minimum_stations = _station_constraint(text)
        metric = _metric(text)
        metric_sql = {
            "mean": "avg(station_value)", "median": "median(station_value)",
            "min": "min(station_value)", "max": "max(station_value)",
            "p95": "quantile_cont(station_value, .95)", "std": "stddev_samp(station_value)",
        }[metric]

        connection = duckdb.connect()
        connection.execute("SET threads=8")
        final_select = "SELECT * FROM country_bucket"
        if daily_extreme_ranking:
            final_select = """
                SELECT dataset, any_value(country_code) country_code, any_value(country) country,
                       arg_max("timestamp", metric_value) AS "timestamp", max(metric_value) metric_value,
                       arg_max(active_stations, metric_value) active_stations,
                       arg_max(valid_hours, metric_value) valid_hours,
                       arg_max(coverage, metric_value) coverage,
                       sum(observed_count) observed_count, sum(imputed_count) imputed_count,
                       any_value(unit) unit
                FROM country_bucket GROUP BY dataset
            """
        sql = f"""
            WITH station_bucket AS (
              SELECT dataset, any_value(country_code) AS country_code,
                     any_value(country_name) AS country,
                     source_station_id, {bucket_expression} AS timestamp,
                     avg({value}) FILTER (WHERE {value} >= 0) AS station_value,
                     count(*) FILTER (WHERE {value} >= 0) AS valid_hours,
                     count(*) FILTER (WHERE value_status='observed' AND observed_value >= 0) AS observed_count,
                     {"0" if observed_only else "count(*) FILTER (WHERE value_status LIKE 'imputed_%' AND analysis_value >= 0)"} AS imputed_count,
                     any_value(unit) AS unit
              FROM read_parquet(?, hive_partitioning=true, union_by_name=true)
              WHERE pollutant=? AND timestamp_local BETWEEN ? AND ?
              GROUP BY dataset, source_station_id, timestamp
            ), country_bucket AS (
              SELECT dataset, any_value(country_code) AS country_code, any_value(country) AS country,
                     timestamp, {metric_sql} AS metric_value,
                     count(*) FILTER (WHERE station_value IS NOT NULL) AS active_stations,
                     sum(valid_hours) AS valid_hours,
                     avg(least(valid_hours::DOUBLE/{expected_hours}, 1.0)) AS coverage,
                     sum(observed_count) AS observed_count,
                     sum(imputed_count) AS imputed_count,
                     any_value(unit) AS unit
              FROM station_bucket GROUP BY dataset, timestamp
              HAVING count(*) FILTER (WHERE station_value IS NOT NULL) > 0
            )
            {final_select} ORDER BY metric_value DESC NULLS LAST, country_code
        """
        try:
            result = connection.execute(sql, [paths, pollutant, start, end])
            names = [item[0] for item in result.description]
            rows = [dict(zip(names, row)) for row in result.fetchall()]
        finally:
            connection.close()
        total_country_count = len(rows)
        rows = [row for row in rows if (minimum_coverage is None or row["coverage"] >= minimum_coverage)
                and (minimum_stations is None or row["active_stations"] >= minimum_stations)]
        excluded_country_count = len(exclusions) + total_country_count - len(rows)
        if not rows:
            qualifier = "observed" if observed_only else "valid"
            raise DataUnavailableError(f"No {qualifier} {pollutant} values satisfy the requested country, period, and coverage constraints")

        for row in rows:
            row[metric] = row.pop("metric_value")
            row["timestamp"] = start.isoformat() if whole_period else row["timestamp"].isoformat()
            row["pollutant"] = pollutant
            for key, value_item in tuple(row.items()):
                if isinstance(value_item, float) and (math.isnan(value_item) or math.isinf(value_item)):
                    row[key] = None

        if assessment.intent == "ranking":
            top_n = _top_n(text)
            rows = rows[:top_n]
            for index, row in enumerate(rows, 1):
                row["rank"] = index
        answer = _answer(
            question, assessment.intent, rows, metric, pollutant, start, end, observed_only,
            minimum_coverage, minimum_stations, excluded_country_count, exclusion_counts,
        )
        visualization = "bar" if assessment.intent in {"ranking", "comparison"} else "line" if len(rows) > 1 else "table"
        status = "observed" if observed_only else "mixed" if any(row["imputed_count"] for row in rows) else "observed"
        return {
            "kind": "analytics",
            "query_spec": {
                "dataset": "GLOBAL_LAKEHOUSE" if len(selected) > 1 else selected[0].dataset,
                "operation": assessment.intent, "geography_level": "country",
                "locations": [item.country_name for item in selected], "pollutants": [pollutant],
                "time_range": {"start": start.isoformat(), "end": end.isoformat()},
                "resolution": resolution, "metrics": [metric],
                "filters": {"value_status": "observed" if observed_only else "any", "minimum_coverage": minimum_coverage},
            },
            "resolved_period": {"start": start.isoformat(), "end": end.isoformat(), "basis": "explicit period or latest available lakehouse data"},
            "data": rows,
            "metadata": {
                "dataset": "GLOBAL_LAKEHOUSE", "geography_level": "country",
                "station_count": sum(row["active_stations"] for row in rows),
                "value_status": status, "station_balanced": True,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "warnings": [], "country_count": len(rows),
            },
            "visualization": {"type": visualization},
            "analysis": {
                "operation": assessment.intent, "metric": metric, "observed_only": observed_only,
                "minimum_coverage": minimum_coverage, "minimum_active_stations": minimum_stations,
                "excluded_country_count": excluded_country_count,
                "cohort": {
                    "candidate_count": len(cohort), "included_count": len(selected),
                    "flexible": flexible_cohort,
                    "exclusion_counts": exclusion_counts,
                    "exclusions": [{"dataset": item.dataset, "country": item.country_name,
                                    "reason_code": item.reason_code, "reason": item.reason} for item in exclusions],
                },
            },
            "answer": answer, "display_title": f"{pollutant} · country analysis",
            "model_trace": {"provider": "deterministic", "model": "global_lakehouse_sql_v1", "latency_ms": 0, "prompt_tokens": None, "completion_tokens": None, "fallback_used": False},
        }


def _period(text: str, catalog: LakehouseCatalog, datasets: list[str]) -> tuple[datetime, datetime]:
    relative = re.search(r"\blast\s+(\d+)\s+complete months?\b", text)
    if relative:
        selected = catalog.selected(datasets)
        available_end = max(item.available_end for item in selected)
        end_month = available_end.month - 1 or 12
        end_year = available_end.year if available_end.month > 1 else available_end.year - 1
        count = int(relative.group(1))
        start_index = end_year * 12 + (end_month - 1) - (count - 1)
        start_year, start_zero_month = divmod(start_index, 12)
        return (
            datetime(start_year, start_zero_month + 1, 1),
            datetime(end_year, end_month, calendar.monthrange(end_year, end_month)[1], 23, 59, 59),
        )
    names = "|".join(sorted(MONTHS, key=len, reverse=True))
    shared_year = re.search(rf"\b({names})\s*(?:through|to|and|[–-])\s*({names})\s+(20\d{{2}})\b", text)
    if shared_year:
        year = int(shared_year.group(3))
        first, last = MONTHS[shared_year.group(1)], MONTHS[shared_year.group(2)]
        start_year, end_year = (year, year) if first <= last else (year - 1, year)
        return datetime(start_year, first, 1), datetime(end_year, last, calendar.monthrange(end_year, last)[1], 23, 59, 59)
    days = list(re.finditer(rf"\b(\d{{1,2}})\s+({names})\s+(20\d{{2}})\b", text))
    if days:
        values = [datetime(int(item.group(3)), MONTHS[item.group(2)], int(item.group(1))) for item in days]
        return min(values), max(values).replace(hour=23, minute=59, second=59)
    months = list(re.finditer(rf"\b({names})\s+(20\d{{2}})\b", text))
    if months:
        values = [(int(item.group(2)), MONTHS[item.group(1)]) for item in months]
        first_year, first_month = min(values)
        last_year, last_month = max(values)
        if len(values) > 1:
            return datetime(first_year, first_month, 1), datetime(
                last_year, last_month, calendar.monthrange(last_year, last_month)[1], 23, 59, 59,
            )
        year, number = values[0]
        return datetime(year, number, 1), datetime(year, number, calendar.monthrange(year, number)[1], 23, 59, 59)
    years = [int(value) for value in re.findall(r"\b(20\d{2})\b", text)]
    if years:
        return datetime(min(years), 1, 1), datetime(max(years), 12, 31, 23, 59, 59)
    selected = catalog.selected(datasets)
    end = max(item.available_end for item in selected)
    return datetime(end.year, end.month, 1), end


def _resolution(text: str) -> str:
    if "hour" in text or "time of day" in text:
        return "hourly"
    if "year" in text or "annual" in text:
        return "yearly"
    if "month" in text:
        return "monthly"
    return "daily"


def _metric(text: str) -> str:
    for term, metric in (("median", "median"), ("maximum", "max"), ("highest", "max"), ("minimum", "min"), ("lowest", "min"), ("95th", "p95"), ("variability", "std")):
        if term in text:
            return metric
    return "mean"


def _percentage_constraint(text: str) -> float | None:
    match = re.search(r"(?:at least|>=|minimum)\s*(\d+(?:\.\d+)?)\s*%", text)
    return float(match.group(1)) / 100 if match else None


def _station_constraint(text: str) -> int | None:
    match = re.search(r"(?:at least|>=|minimum)\s*(\d+)\s*(?:active\s+)?(?:pm2\.5\s+)?stations", text)
    return int(match.group(1)) if match else None


def _top_n(text: str) -> int:
    match = re.search(r"top\s+(\d+)", text)
    if match:
        return min(int(match.group(1)), 100)
    words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    match = re.search(r"\b(?:top\s+)?(" + "|".join(words) + r")\b", text)
    return words[match.group(1)] if match else 10


def _answer(question: str, intent: str, rows: list[dict], metric: str, pollutant: str,
            start: datetime, end: datetime, observed_only: bool,
            minimum_coverage: float | None, minimum_stations: int | None,
            excluded_country_count: int, exclusion_counts: dict[str, int]) -> str:
    unit = rows[0]["unit"]
    period = start.strftime("%d %b %Y") if start.date() == end.date() else f"{start:%d %b %Y} to {end:%d %b %Y}"
    provenance = "directly observed values" if observed_only else "valid observed and policy-approved filled values"
    if intent == "ranking":
        leader = rows[0]
        answer = f"{leader['country']} ranks first for {pollutant}, with an equal-weight station {metric} of {leader[metric]:.2f} {unit} during {period}. The ranking contains {len(rows)} qualifying countries and uses {provenance}."
        if re.search(r"\bdaily\s+(?:maximum|max)\b", question.casefold()):
            peak = datetime.fromisoformat(leader["timestamp"])
            answer = f"{leader['country']} had the highest station-day {pollutant} maximum: {leader[metric]:.2f} {unit} on {peak:%d %b %Y}. The ranking contains {len(rows)} qualifying countries and uses {provenance}."
    elif intent == "comparison":
        values = ", ".join(f"{row['country']}: {row[metric]:.2f} {unit}" for row in rows)
        answer = f"During {period}, the equal-weight station {pollutant} results are {values}. The calculation uses {provenance}."
        if len(rows) <= 6:
            differences = []
            for index, left in enumerate(rows):
                for right in rows[index + 1:]:
                    differences.append(f"{left['country']}–{right['country']}: {abs(left[metric] - right[metric]):.2f} {unit}")
            if differences:
                answer += " Pairwise absolute differences: " + "; ".join(differences) + "."
    elif len(rows) == 1:
        row = rows[0]
        answer = f"The equal-weight station {metric} {pollutant} concentration in {row['country']} was {row[metric]:.2f} {unit} during {period}, using {provenance}."
    else:
        answer = f"I calculated {len(rows)} country-period {pollutant} results for {period} using {provenance}."
    constraints = []
    if minimum_coverage is not None:
        constraints.append(f"at least {minimum_coverage:.0%} coverage")
    if minimum_stations is not None:
        constraints.append(f"at least {minimum_stations} active stations")
    if constraints:
        noun = "country was" if excluded_country_count == 1 else "countries were"
        answer += " Eligibility required " + " and ".join(constraints) + f"; {excluded_country_count} {noun} excluded."
    if exclusion_counts:
        unavailable = sum(exclusion_counts.get(code, 0) for code in ("no_rows", "partial_period", "no_observed_values", "no_usable_values"))
        quality = sum(exclusion_counts.get(code, 0) for code in ("coverage_below_threshold", "stations_below_threshold"))
        details = []
        if unavailable:
            details.append(f"{unavailable} lacked complete requested data")
        if quality:
            details.append(f"{quality} failed the requested quality threshold")
        if details:
            answer += " Exclusion summary: " + "; ".join(details) + "."
    if metric in {"mean", "median", "p95", "std"}:
        answer += " Each monitoring station receives equal weight in its country result."
    else:
        answer += " The calculation first forms station-level values, so stations with more raw readings do not receive extra weight."
    return answer
