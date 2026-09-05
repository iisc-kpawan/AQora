from __future__ import annotations

import math
import re
import time
from datetime import datetime
from typing import Any

import duckdb
import numpy as np

from .aqi import DEFAULT_GUIDELINE, get_guideline
from .capabilities import CapabilityAssessment, is_flexible_country_cohort
from .config import load_config
from .errors import DataUnavailableError
from .global_analytics import POLLUTANTS, _period, _top_n
from .lakehouse_catalog import LakehouseCatalog
from .periods import resolve_period
from .quality_analytics import _requested_cities


POLICIES = (
    ("observed_only", "measured only"),
    ("observed_guarded", "measured plus guarded fills"),
    ("all_usable", "all usable values"),
)

CITY_ALIASES = {
    "bangalore": "bengaluru", "bengaluru": "bengaluru", "beijing": "北京",
    "mexico city": "cdmx",
}


def is_provenance_policy_question(question: str, intent: str | None = None) -> bool:
    text = question.casefold()
    if intent == "forecast" or "aqi" in text:
        return False
    mentions_policy = (
        ("observed" in text and any(term in text for term in ("imputed", "filled")))
        or "imputed fraction" in text or "imputation changes" in text
        or "imputation rate" in text or "imputation dependence" in text
    )
    return mentions_policy


class ProvenanceAnalytics:
    """Compare deterministic results under explicit value-provenance policies."""

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.catalog = LakehouseCatalog(self.config)
        self.root = self.catalog.root

    @staticmethod
    def should_handle(question: str, assessment: CapabilityAssessment) -> bool:
        return is_provenance_policy_question(question, assessment.intent)

    def execute(self, question: str, assessment: CapabilityAssessment) -> dict[str, Any]:
        started = time.perf_counter()
        text = question.casefold()
        pollutant = next((name for pattern, name in POLLUTANTS if re.search(pattern, text)), "PM25")
        selected = list(self.catalog.selected(assessment.supported_datasets))
        flexible_cohort = is_flexible_country_cohort(question, assessment.intent)
        if flexible_cohort and not any(term in text for term in ("seven primary", "all seven", "primary countries")):
            selected = list(self.catalog.datasets)
        trend_slope_requested = "trend slope" in text
        if trend_slope_requested and not _has_period(text):
            # A slope needs multiple points; the default single-latest-month
            # window below (used when nothing else is requested) can't
            # produce one, so fall back to a real multi-month span instead --
            # anchored to the dataset's own current data, not a fixed
            # calendar date, matching how the rest of the codebase avoids
            # baking in a specific "now".
            anchor = min(item.available_end for item in selected)
            start, end = resolve_period("last 12 complete months", anchor, anchor)
        elif len(selected) > 1 and not flexible_cohort and not _has_period(text):
            end = min(item.available_end for item in selected)
            start = end.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            start, end = _period(text, self.catalog, [item.dataset for item in selected])
        paths = self._paths(selected, start, end)
        if not paths:
            raise DataUnavailableError("No hourly lakehouse partitions overlap the requested provenance comparison")

        cities = _requested_cities(text)
        geography = "city" if cities else "country_city_balanced" if any(
            term in text for term in ("city-balanced", "city weighted", "city-weighted")
        ) else "country"
        monthly = trend_slope_requested or any(
            term in text for term in ("monthly", "each month", "by month", "trajector", "line chart")
        )
        bucket_expression = "date_trunc('month', timestamp_local)" if monthly else "TIMESTAMP '1970-01-01'"
        minimum_coverage = _percentage_after(text, ("coverage",))
        minimum_filled = _percentage_after(text, ("imputed", "filled"), comparison="more")
        metrics = _metrics(text)
        sensitivity_requested = any(term in text for term in (
            "sensitivity", "combinations of coverage", "different coverage thresholds",
            "across coverage thresholds", "vary across", "robust to coverage",
        ))
        coverage_thresholds = _coverage_thresholds(text, self.config, sensitivity_requested, minimum_coverage)
        guideline = get_guideline(DEFAULT_GUIDELINE)
        exceedance_threshold = _explicit_threshold(text)
        if exceedance_threshold is None:
            exceedance_threshold = guideline.thresholds.get(pollutant)

        control = duckdb.connect(str(self.catalog.control_path), read_only=True)
        try:
            dimensions = control.execute(
                "SELECT dataset,source_station_id,city FROM station_dimension"
            ).fetchdf()
        finally:
            control.close()
        connection = duckdb.connect()
        connection.execute("SET threads=8")
        connection.register("provenance_station_dimension", dimensions)
        sql = f"""
            WITH raw AS (
              SELECT dataset,network,country_code,country_name,source_station_id,pollutant,
                     timestamp_local,observed_value,analysis_value,value_status,unit
              FROM read_parquet(?,hive_partitioning=true,union_by_name=true)
              WHERE pollutant=? AND timestamp_local BETWEEN ? AND ?
            ), policy_values AS (
              SELECT r.*,p.policy,
                     CASE p.policy
                       WHEN 'observed_only' THEN CASE WHEN value_status='observed' THEN observed_value END
                       WHEN 'observed_guarded' THEN CASE WHEN value_status='observed' THEN observed_value
                                                        WHEN value_status='imputed_guarded' THEN analysis_value END
                       ELSE analysis_value END selected_value
              FROM raw r CROSS JOIN (VALUES ('observed_only'),('observed_guarded'),('all_usable')) p(policy)
            ), station_day AS (
              SELECT dataset,any_value(network) network,any_value(country_code) country_code,
                     any_value(country_name) country_name,source_station_id,policy,
                     date_trunc('day',timestamp_local) bucket_day,
                     avg(selected_value) FILTER(WHERE selected_value>=0) station_value,
                     count(*) expected_count,count(selected_value) FILTER(WHERE selected_value>=0) selected_count,
                     count(*) FILTER(WHERE selected_value>=0 AND value_status LIKE 'imputed%') filled_count,
                     any_value(unit) unit
              FROM policy_values GROUP BY dataset,source_station_id,policy,bucket_day
            ), eligible_station_day AS (
              SELECT s.*,t.coverage_threshold::DOUBLE coverage_threshold
              FROM station_day s CROSS JOIN (VALUES {','.join(f'({value:.12g})' for value in coverage_thresholds)}) t(coverage_threshold)
              WHERE s.selected_count::DOUBLE/nullif(s.expected_count,0)>=t.coverage_threshold
            ), enriched AS (
              SELECT s.*,d.city,lower(trim(d.city)) city_id
              FROM eligible_station_day s LEFT JOIN provenance_station_dimension d USING(dataset,source_station_id)
            ), city_day AS (
              SELECT dataset,any_value(network) network,any_value(country_code) country_code,
                     any_value(country_name) country_name,policy,coverage_threshold,bucket_day,city_id area_id,
                     any_value(city) area_name,avg(station_value) area_value,
                     sum(expected_count) expected_count,sum(selected_count) selected_count,
                     sum(filled_count) filled_count,count(station_value) active_station_count,
                     any_value(unit) unit
              FROM enriched WHERE city_id IS NOT NULL AND city_id<>''
              GROUP BY dataset,policy,coverage_threshold,bucket_day,city_id
            ), country_day AS (
              SELECT dataset,any_value(network) network,any_value(country_code) country_code,
                     any_value(country_name) country_name,policy,coverage_threshold,bucket_day,
                     any_value(country_code) area_id,any_value(country_name) area_name,
                     avg(station_value) area_value,sum(expected_count) expected_count,
                     sum(selected_count) selected_count,sum(filled_count) filled_count,
                     count(station_value) active_station_count,any_value(unit) unit
              FROM enriched GROUP BY dataset,policy,coverage_threshold,bucket_day
            ), country_city_day AS (
              SELECT dataset,any_value(network) network,any_value(country_code) country_code,
                     any_value(country_name) country_name,policy,coverage_threshold,bucket_day,
                     any_value(country_code) area_id,any_value(country_name) area_name,
                     avg(area_value) area_value,sum(expected_count) expected_count,
                     sum(selected_count) selected_count,sum(filled_count) filled_count,
                     sum(active_station_count) active_station_count,any_value(unit) unit
              FROM city_day GROUP BY dataset,policy,coverage_threshold,bucket_day
            ), chosen AS (
              {"SELECT * FROM city_day" if geography == "city" else "SELECT * FROM country_city_day" if geography == "country_city_balanced" else "SELECT * FROM country_day"}
            ), result AS (
              SELECT dataset,any_value(network) network,any_value(country_code) country_code,
                     any_value(country_name) country,policy,coverage_threshold,{bucket_expression.replace('timestamp_local','bucket_day')} bucket,
                     area_id,any_value(area_name) area_name,
                     avg(area_value) mean,median(area_value) median,
                     quantile_cont(area_value,.95) p95,max(area_value) max,
                     count(*) FILTER(WHERE area_value>{float(exceedance_threshold) if exceedance_threshold is not None else 'NULL'})::DOUBLE/nullif(count(area_value),0) exceedance_percentage,
                     sum(selected_count)::DOUBLE/nullif(sum(expected_count),0) coverage,
                     sum(filled_count)::DOUBLE/nullif(sum(selected_count),0) filled_fraction,
                     sum(selected_count)::BIGINT selected_count,sum(filled_count)::BIGINT filled_count,
                     max(active_station_count)::BIGINT active_station_count,any_value(unit) unit
              FROM chosen GROUP BY dataset,area_id,policy,coverage_threshold,bucket
            )
            SELECT * FROM result ORDER BY bucket,country,policy
        """
        try:
            result = connection.execute(sql, [paths, pollutant, start, end])
            names = [item[0] for item in result.description]
            rows = [dict(zip(names, row)) for row in result.fetchall()]
        finally:
            connection.close()
        if cities:
            wanted = {_city_key(item) for item in cities}
            rows = [row for row in rows if _city_key(row["area_name"] or "") in wanted]
        rows = [row for row in rows if row["selected_count"]]
        policies_by_entity: dict[tuple, set[str]] = {}
        for row in rows:
            policies_by_entity.setdefault((row["dataset"], row["area_id"], row["bucket"], row["coverage_threshold"]), set()).add(row["policy"])
        comparable = {
            key for key, policies in policies_by_entity.items()
            if {"observed_only", "all_usable"} <= policies
        }
        rows = [row for row in rows if (row["dataset"], row["area_id"], row["bucket"], row["coverage_threshold"]) in comparable]
        if not flexible_cohort and not cities:
            present = {row["dataset"] for row in rows}
            missing = [item.country_name for item in selected if item.dataset not in present]
            if missing:
                raise DataUnavailableError(
                    "The explicitly requested provenance cohort is incomplete. Missing measured-versus-filled results for: "
                    + ", ".join(missing)
                )
        if not rows:
            raise DataUnavailableError("No area has both measured-only and all-usable values under the requested coverage constraints")

        _add_differences_and_ranks(rows)
        if minimum_filled is not None:
            qualifying = {
                (row["dataset"], row["bucket"], row["coverage_threshold"]) for row in rows
                if row["policy"] == "all_usable" and (row["filled_fraction"] or 0) > minimum_filled
            }
            rows = [row for row in rows if (row["dataset"], row["bucket"], row["coverage_threshold"]) in qualifying]
        rank_change = _rank_change_threshold(text)
        if rank_change is not None:
            qualifying_datasets = {
                row["dataset"] for row in rows
                if row["policy"] == "all_usable" and abs(row.get("rank_change_vs_observed") or 0) >= rank_change
            }
            rows = [row for row in rows if row["dataset"] in qualifying_datasets]
        if not rows:
            condition = (
                f"a rank change of at least {rank_change} positions" if rank_change is not None
                else f"more than {minimum_filled:.0%} filled values" if minimum_filled is not None
                else "the requested provenance sensitivity conditions"
            )
            return self._empty_result(
                pollutant=pollutant, start=start, end=end, geography=geography,
                monthly=monthly, metrics=metrics, minimum_coverage=minimum_coverage,
                minimum_filled=minimum_filled, rank_change=rank_change,
                condition=condition, started=started,
            )
        if assessment.intent == "ranking" or any(term in text for term in ("five most", "top ")):
            keep = _top_n(text)
            leaders = [row for row in rows if row["policy"] == "all_usable"]
            leaders.sort(key=lambda row: row.get("mean") or -math.inf, reverse=True)
            wanted = {row["dataset"] for row in leaders[:keep]}
            rows = [row for row in rows if row["dataset"] in wanted]
        if trend_slope_requested:
            _add_trend_slopes(rows)
        for row in rows:
            row["timestamp"] = (start if row["bucket"].year == 1970 else row["bucket"]).isoformat()
            row["pollutant"] = pollutant
            row["policy_label"] = dict(POLICIES)[row["policy"]]
            del row["bucket"]
        answer = _answer(rows, pollutant, start, end, minimum_coverage, minimum_filled, rank_change)
        if len(coverage_thresholds) > 1:
            answer += " Coverage sensitivity was evaluated at " + ", ".join(f"{value:.0%}" for value in coverage_thresholds) + "."
        reversals = _rank_reversals(rows)
        if "revers" in text:
            answer += (
                " Rank-order reversals between measured-only and all-usable results: "
                + ("; ".join(reversals) if reversals else "none") + "."
            )
        return {
            "kind": "analytics",
            "query_spec": {
                "dataset": "GLOBAL_LAKEHOUSE" if len({row['dataset'] for row in rows}) > 1 else rows[0]["dataset"],
                "operation": "provenance_policy_comparison", "geography_level": geography,
                "locations": cities or sorted({row["country"] for row in rows}), "pollutants": [pollutant],
                "time_range": {"start": start.isoformat(), "end": end.isoformat()},
                "resolution": "monthly" if monthly else "period", "metrics": metrics,
                "filters": {"policies": [item[0] for item in POLICIES], "minimum_coverage": minimum_coverage,
                            "minimum_filled_fraction": minimum_filled},
            },
            "resolved_period": {"start": start.isoformat(), "end": end.isoformat(), "basis": "explicit period or latest available common scope"},
            "data": rows,
            "metadata": {
                "dataset": "GLOBAL_LAKEHOUSE", "geography_level": geography,
                "station_count": max(row["active_station_count"] for row in rows),
                "value_status": "policy_comparison", "station_balanced": True,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2), "warnings": [],
            },
            "visualization": {"type": "line" if monthly else "bar", "series": "policy"},
            "analysis": {
                "operator": "provenance_policy_comparison_v1", "policies": [item[0] for item in POLICIES],
                "aggregation": "station-day then equal-weight area-day", "minimum_coverage": minimum_coverage,
                "minimum_filled_fraction": minimum_filled, "rank_change_threshold": rank_change,
                "rank_order_reversals": reversals,
                "coverage_thresholds": coverage_thresholds,
                "exceedance_threshold": exceedance_threshold,
                "exceedance_threshold_source": guideline.id if exceedance_threshold is not None else None,
            },
            "answer": answer, "display_title": f"{pollutant} · measured versus filled sensitivity",
            "model_trace": {"provider": "deterministic", "model": "provenance_policy_comparison_v1",
                            "latency_ms": 0, "prompt_tokens": None, "completion_tokens": None, "fallback_used": False},
        }

    def _empty_result(self, *, pollutant: str, start: datetime, end: datetime,
                      geography: str, monthly: bool, metrics: list[str],
                      minimum_coverage: float | None, minimum_filled: float | None,
                      rank_change: int | None, condition: str, started: float) -> dict[str, Any]:
        return {
            "kind": "analytics",
            "query_spec": {
                "dataset": "GLOBAL_LAKEHOUSE", "operation": "provenance_policy_comparison",
                "geography_level": geography, "locations": [], "pollutants": [pollutant],
                "time_range": {"start": start.isoformat(), "end": end.isoformat()},
                "resolution": "monthly" if monthly else "period", "metrics": metrics,
                "filters": {"minimum_coverage": minimum_coverage,
                            "minimum_filled_fraction": minimum_filled},
            },
            "resolved_period": {"start": start.isoformat(), "end": end.isoformat(), "basis": "explicit requested period"},
            "data": [],
            "metadata": {
                "dataset": "GLOBAL_LAKEHOUSE", "geography_level": geography, "station_count": 0,
                "value_status": "policy_comparison", "station_balanced": True,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2), "warnings": [],
            },
            "visualization": {"type": "none"},
            "analysis": {
                "operator": "provenance_policy_comparison_v1", "policies": [item[0] for item in POLICIES],
                "minimum_coverage": minimum_coverage, "minimum_filled_fraction": minimum_filled,
                "rank_change_threshold": rank_change, "matching_area_count": 0,
            },
            "answer": f"No country or city had {condition} for {pollutant} during {start:%d %b %Y} to {end:%d %b %Y}. The result is an empty analytical match, not a missing-data substitution.",
            "display_title": f"{pollutant} · measured versus filled sensitivity",
            "model_trace": {"provider": "deterministic", "model": "provenance_policy_comparison_v1",
                            "latency_ms": 0, "prompt_tokens": None, "completion_tokens": None, "fallback_used": False},
        }

    def _paths(self, selected, start: datetime, end: datetime) -> list[str]:
        paths = []
        for item in selected:
            current = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            while current <= end:
                directory = self.root / "archive" / "hourly" / f"dataset={item.dataset}" / f"year={current.year}" / f"month={current.month}"
                paths.extend(str(path) for path in directory.glob("*.parquet"))
                current = current.replace(year=current.year + 1, month=1) if current.month == 12 else current.replace(month=current.month + 1)
        return paths


def _metrics(text: str) -> list[str]:
    values = [name for term, name in (
        ("mean", "mean"), ("average", "mean"), ("median", "median"), ("p95", "p95"),
        ("95th", "p95"), ("maximum", "max"), ("coverage", "coverage"),
        ("completeness", "coverage"),
        ("imputed", "filled_fraction"), ("filled", "filled_fraction"),
        ("exceedance", "exceedance_percentage"), ("trend slope", "trend_slope"),
    ) if term in text]
    return list(dict.fromkeys(values)) or ["mean", "filled_fraction"]


def _add_trend_slopes(rows: list[dict]) -> None:
    """Attach a monthly OLS trend slope per (dataset, policy, coverage_threshold).

    Mirrors the plain slope calculation TemporalAnalytics already uses
    elsewhere (index-position regression against the ordered monthly means),
    just scoped per value-provenance policy so an observed-only trend and an
    all-usable trend can be compared side by side.
    """
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["dataset"], row["policy"], row["coverage_threshold"]), []).append(row)
    for group in groups.values():
        group.sort(key=lambda row: row["bucket"])
        points = [(index, row["mean"]) for index, row in enumerate(group) if row.get("mean") is not None]
        slope = float(np.polyfit([p[0] for p in points], [p[1] for p in points], 1)[0]) if len(points) >= 2 else None
        for row in group:
            row["trend_slope"] = slope


def _percentage_after(text: str, terms: tuple[str, ...], comparison: str | None = None) -> float | None:
    joined = "|".join(terms)
    if comparison == "more":
        match = re.search(rf"more than\s+(\d+(?:\.\d+)?)\s*%.*?(?:{joined})|(?:{joined}).*?more than\s+(\d+(?:\.\d+)?)\s*%", text)
    else:
        match = re.search(rf"(?:at least|minimum|requiring)\s+(\d+(?:\.\d+)?)\s*%\s*(?:{joined})", text)
    if not match:
        return None
    value = next(group for group in match.groups() if group is not None)
    return float(value) / 100


def _rank_change_threshold(text: str) -> int | None:
    number = r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
    match = re.search(rf"rank changes? by at least\s+{number}|changes? by at least\s+{number}\s+positions", text)
    if not match:
        return None
    value = next(group for group in match.groups() if group is not None)
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
             "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
    return int(value) if value.isdigit() else words[value]


def _coverage_thresholds(text: str, config: dict, sensitivity: bool,
                         minimum: float | None) -> list[float]:
    if not sensitivity:
        return [minimum if minimum is not None else 0.0]
    explicit = [float(value) / 100 for value in re.findall(r"(\d+(?:\.\d+)?)\s*%", text)]
    explicit = [value for value in explicit if 0 <= value <= 1]
    if len(explicit) >= 2:
        return sorted(set(explicit))
    configured = config.get("analytics", {}).get("coverage_sensitivity_thresholds", [.50, .75, .90])
    return sorted({float(value) for value in configured})


def _explicit_threshold(text: str) -> float | None:
    match = re.search(r"(?:threshold|guideline)(?:\s+of|\s*=|\s+at)?\s*(\d+(?:\.\d+)?)", text)
    return float(match.group(1)) if match else None


def _has_period(text: str) -> bool:
    months = "|".join(
        ("january", "february", "march", "april", "may", "june", "july", "august",
         "september", "october", "november", "december", "jan", "feb", "mar", "apr",
         "jun", "jul", "aug", "sep", "oct", "nov", "dec")
    )
    return bool(
        re.search(r"\b20\d{2}\b", text) or re.search(rf"\b(?:{months})\b", text)
        or re.search(r"\b(?:last|latest|previous)\b", text)
    )


def _city_key(value: str) -> str:
    key = value.casefold().strip()
    return CITY_ALIASES.get(key, key)


def _add_differences_and_ranks(rows: list[dict]) -> None:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["bucket"], row["policy"], row["coverage_threshold"]), []).append(row)
    for group in groups.values():
        group.sort(key=lambda row: row.get("mean") if row.get("mean") is not None else -math.inf, reverse=True)
        for rank, row in enumerate(group, 1):
            row["rank"] = rank
    by_key = {(row["dataset"], row["area_id"], row["bucket"], row["coverage_threshold"], row["policy"]): row for row in rows}
    for row in rows:
        observed = by_key.get((row["dataset"], row["area_id"], row["bucket"], row["coverage_threshold"], "observed_only"))
        if observed and observed.get("mean") is not None and row.get("mean") is not None:
            row["mean_difference_vs_observed"] = row["mean"] - observed["mean"]
            row["mean_percentage_difference_vs_observed"] = (
                (row["mean"] - observed["mean"]) / abs(observed["mean"]) if observed["mean"] else None
            )
            row["rank_change_vs_observed"] = row["rank"] - observed["rank"]
        else:
            row["mean_difference_vs_observed"] = None
            row["mean_percentage_difference_vs_observed"] = None
            row["rank_change_vs_observed"] = None


def _answer(rows: list[dict], pollutant: str, start: datetime, end: datetime,
            minimum_coverage: float | None, minimum_filled: float | None,
            rank_change: int | None) -> str:
    entities = {(row["dataset"], row["area_id"]) for row in rows}
    observed = [row for row in rows if row["policy"] == "observed_only"]
    usable = [row for row in rows if row["policy"] == "all_usable"]
    max_gap = max((abs(row.get("mean_difference_vs_observed") or 0) for row in usable), default=0)
    period = f"{start:%d %b %Y} to {end:%d %b %Y}"
    answer = (
        f"I compared {pollutant} under three explicit value policies for {len(entities)} area(s) during {period}: "
        f"measured only, measured plus guarded fills, and all usable values. The largest absolute all-usable versus measured-only mean difference was {max_gap:.2f} {rows[0]['unit']}."
    )
    if minimum_filled is not None:
        answer += f" Results are restricted to areas with more than {minimum_filled:.0%} filled values."
    if minimum_coverage is not None:
        answer += f" Every returned policy result meets the {minimum_coverage:.0%} coverage requirement."
    if rank_change is not None:
        answer += f" Returned areas changed rank by at least {rank_change} positions."
    if any(row.get("trend_slope") is not None for row in rows):
        parts = []
        for area in sorted({row["area_name"] for row in rows}):
            obs = next((row for row in observed if row["area_name"] == area and row.get("trend_slope") is not None), None)
            used = next((row for row in usable if row["area_name"] == area and row.get("trend_slope") is not None), None)
            if obs and used:
                parts.append(
                    f"{area}: {obs['trend_slope']:+.3f} {obs['unit']}/month measured-only vs. "
                    f"{used['trend_slope']:+.3f} {used['unit']}/month all-usable"
                )
        if parts:
            answer += " Monthly trend slopes -- " + "; ".join(parts) + "."
    answer += " Each day is formed at station level first, so stations with more raw rows do not receive extra weight. Filled values are never described as measurements."
    return answer


def _rank_reversals(rows: list[dict]) -> list[str]:
    output = []
    thresholds = sorted({row["coverage_threshold"] for row in rows})
    for threshold in thresholds:
        observed = {row["dataset"]: row for row in rows if row["policy"] == "observed_only" and row["coverage_threshold"] == threshold}
        usable = {row["dataset"]: row for row in rows if row["policy"] == "all_usable" and row["coverage_threshold"] == threshold}
        datasets = sorted(observed.keys() & usable.keys())
        for index, left in enumerate(datasets):
            for right in datasets[index + 1:]:
                observed_order = observed[left]["mean"] - observed[right]["mean"]
                usable_order = usable[left]["mean"] - usable[right]["mean"]
                if observed_order * usable_order < 0:
                    label = f"{observed[left]['country']} versus {observed[right]['country']}"
                    if len(thresholds) > 1:
                        label += f" at {threshold:.0%} coverage"
                    output.append(label)
    return output
