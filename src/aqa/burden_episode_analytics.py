from __future__ import annotations

import calendar
import math
import re
import time
from datetime import datetime, timedelta
from typing import Any

import duckdb

from .aqi import DEFAULT_GUIDELINE, get_guideline
from .capabilities import CapabilityAssessment
from .config import load_config
from .errors import DataUnavailableError
from .global_analytics import _top_n
from .lakehouse_catalog import LakehouseCatalog


def burden_episode_operation(question: str) -> str | None:
    """Resolve advanced exceedance operations from their analytical semantics."""
    text = question.casefold()
    annualized = (
        bool(re.search(r"annuali[sz]ed", text))
        and "uncertainty" in text
        and "exceedance" in text
        and bool(re.search(r"\bcit(?:y|ies)\b", text))
        and any(term in text for term in ("incomplete", "missing"))
    )
    if annualized:
        return "annualized_exceedance_uncertainty"
    episode_ranking = (
        bool(re.search(r"multi[ -]?day", text))
        and "episode" in text
        and bool(re.search(r"\bcit(?:y|ies)\b", text))
        and any(term in text for term in ("rank", "most severe", "severity"))
        and "duration" in text
        and "mean exceedance" in text
    )
    if episode_ranking:
        return "severe_multiday_episode_ranking"
    return None


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return a deterministic two-sided Wilson interval for a binomial proportion."""
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("Wilson interval requires 0 <= successes <= trials and trials > 0")
    proportion = successes / trials
    denominator = 1 + z * z / trials
    centre = (proportion + z * z / (2 * trials)) / denominator
    half_width = z * math.sqrt(
        proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)
    ) / denominator
    lower = 0.0 if successes == 0 else max(0.0, centre - half_width)
    upper = 1.0 if successes == trials else min(1.0, centre + half_width)
    return lower, upper


def build_multiday_episodes(daily: list[dict], threshold: float, minimum_duration: int = 2) -> list[dict]:
    """Build contiguous calendar-day episodes; missing or non-exceeding days break a run."""
    grouped: dict[tuple, list[dict]] = {}
    for row in daily:
        key = (row["dataset"], row["country"], row.get("area_id") or row["area_name"], row["area_name"])
        grouped.setdefault(key, []).append(row)
    output: list[dict] = []
    for (dataset, country, area_id, area_name), values in grouped.items():
        values.sort(key=lambda row: row["bucket_day"])
        runs: list[list[dict]] = []
        current: list[dict] = []
        for row in values:
            consecutive = not current or row["bucket_day"].date() == (
                current[-1]["bucket_day"] + timedelta(days=1)
            ).date()
            if row["mean"] <= threshold or not consecutive:
                if len(current) >= minimum_duration:
                    runs.append(current)
                current = []
            if row["mean"] > threshold:
                current.append(row)
        if len(current) >= minimum_duration:
            runs.append(current)
        for run in runs:
            excesses = [row["mean"] - threshold for row in run]
            mean_exceedance = sum(excesses) / len(excesses)
            observed_fraction = sum(row["observed_fraction"] for row in run) / len(run)
            imputed_fraction = sum(row["imputed_fraction"] for row in run) / len(run)
            output.append({
                "dataset": dataset,
                "country": country,
                "area_id": area_id,
                "area_name": area_name,
                "pollutant": "PM25",
                "start": run[0]["bucket_day"].isoformat(),
                "end": run[-1]["bucket_day"].isoformat(),
                "duration_days": len(run),
                "threshold": threshold,
                "mean_exceedance": mean_exceedance,
                "peak": max(row["mean"] for row in run),
                "severity": len(run) * mean_exceedance,
                "mean_daily_coverage": sum(row["coverage"] for row in run) / len(run),
                "observed_fraction": observed_fraction,
                "imputed_fraction": imputed_fraction,
                "value_source": _plain_value_source(observed_fraction, imputed_fraction),
                "active_station_count": max(row["active_station_count"] for row in run),
                "unit": run[0]["unit"],
            })
    return output


class BurdenEpisodeAnalytics:
    """Annualized missing-data uncertainty and guarded multi-day episode ranking."""

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.catalog = LakehouseCatalog(self.config)
        self.root = self.catalog.root
        analytics = self.config.get("analytics", {})
        self.minimum_coverage = float(analytics.get("minimum_daily_coverage", 0.75))
        self.minimum_annualized_days = int(analytics.get("annualized_minimum_valid_days", 90))
        self.minimum_annualized_months = int(analytics.get("annualized_minimum_valid_months", 9))
        self.minimum_episode_fraction = float(analytics.get("episode_minimum_valid_day_fraction", 0.80))
        self.minimum_episode_months = int(analytics.get("episode_minimum_valid_months", 10))
        self.maximum_episode_imputed_fraction = float(
            analytics.get("episode_maximum_imputed_fraction", 0.50)
        )
        self.maximum_valid_pm25 = float(
            analytics.get(
                "maximum_valid_pm25",
                self.config.get("visualization", {}).get("maximum_plausible_pm25", 5000),
            )
        )

    @staticmethod
    def should_handle(question: str, assessment: CapabilityAssessment) -> bool:
        del assessment
        return burden_episode_operation(question) is not None

    def execute(self, question: str, assessment: CapabilityAssessment) -> dict[str, Any]:
        del assessment
        operation = burden_episode_operation(question)
        paths = self._paths()
        if not paths:
            raise DataUnavailableError("No city-day layer is available for exceedance analysis")
        start, end = self._latest_complete_year(paths)
        if operation == "annualized_exceedance_uncertainty":
            return self._annualized_uncertainty(paths, start, end)
        if operation == "severe_multiday_episode_ranking":
            return self._episode_ranking(question, paths, start, end)
        raise DataUnavailableError("The requested burden or episode operation is not available")

    def _annualized_uncertainty(self, paths: list[str], start: datetime, end: datetime) -> dict[str, Any]:
        started = time.perf_counter()
        expected_days = (end.date() - start.date()).days + 1
        threshold, guideline = self._threshold()
        frame = self._city_summaries(paths, start, end, threshold)
        rows: list[dict] = []
        complete_count = 0
        insufficient_count = 0
        for item in frame.itertuples(index=False):
            valid_days = int(item.valid_day_count)
            valid_months = int(item.valid_month_count)
            if valid_days >= expected_days:
                complete_count += 1
                continue
            if valid_days < self.minimum_annualized_days or valid_months < self.minimum_annualized_months:
                insufficient_count += 1
                continue
            exceedance_days = int(item.exceedance_day_count)
            missing_days = expected_days - valid_days
            lower, upper = wilson_interval(exceedance_days, valid_days)
            annualized = 365 * exceedance_days / valid_days
            row = {
                "dataset": item.dataset,
                "country": item.country,
                "area_id": item.area_id,
                "area_name": item.area_name,
                "pollutant": "PM25",
                "threshold": threshold,
                "valid_day_count": valid_days,
                "missing_day_count": missing_days,
                "valid_month_count": valid_months,
                "data_completeness": valid_days / expected_days,
                "exceedance_day_count": exceedance_days,
                "valid_day_exceedance_fraction": exceedance_days / valid_days,
                "annualized_exceedance_days": annualized,
                "wilson_95_lower_days": 365 * lower,
                "wilson_95_upper_days": 365 * upper,
                "worst_case_lower_days": 365 * exceedance_days / expected_days,
                "worst_case_upper_days": 365 * (exceedance_days + missing_days) / expected_days,
                "mean_daily_coverage": float(item.mean_daily_coverage),
                "observed_fraction": float(item.observed_fraction),
                "imputed_fraction": float(item.imputed_fraction),
                "evidence_tier": _evidence_tier(float(item.observed_fraction), float(item.imputed_fraction)),
                "active_station_count": int(item.active_station_count),
                "unit": item.unit,
            }
            row["wilson_interval_width_days"] = row["wilson_95_upper_days"] - row["wilson_95_lower_days"]
            row["worst_case_interval_width_days"] = row["worst_case_upper_days"] - row["worst_case_lower_days"]
            rows.append(row)
        if not rows:
            raise DataUnavailableError("No incomplete city has enough valid days and calendar-month representation to annualize")
        rows.sort(key=lambda row: (-row["annualized_exceedance_days"], row["dataset"], row["area_name"]))
        mostly_filled_count = sum(row["imputed_fraction"] > 0.5 for row in rows)
        answer = (
            f"I quantified annualized PM2.5 exceedance burden for {len(rows)} incomplete city series during "
            f"{start:%b %Y}–{end:%b %Y}. Each estimate scales the valid-day exceedance fraction to 365 days. "
            "The Wilson 95% interval is a sampling interval that assumes valid days are representative and approximately independent; "
            "the wider worst-case bounds instead treat every missing day as either non-exceeding or exceeding. "
            "The Wilson interval does not include uncertainty introduced by filled values, so every row reports measured and filled fractions. "
            f"Eligibility requires at least {self.minimum_annualized_days} valid days across {self.minimum_annualized_months} months."
        )
        warnings = []
        if mostly_filled_count:
            warnings.append(
                f"{mostly_filled_count} eligible city series are based mostly on policy-approved filled values; their measured and filled fractions are reported."
            )
        return self._result(
            rows=rows,
            operation="annualized_exceedance_uncertainty",
            operator="annualized_exceedance_uncertainty_v1",
            answer=answer,
            start=start,
            end=end,
            metrics=[
                "annualized_exceedance_days", "wilson_95_lower_days", "wilson_95_upper_days",
                "worst_case_lower_days", "worst_case_upper_days", "data_completeness",
            ],
            filters={
                "threshold": threshold,
                "minimum_daily_coverage": self.minimum_coverage,
                "minimum_valid_days": self.minimum_annualized_days,
                "minimum_valid_months": self.minimum_annualized_months,
                "incomplete_series_only": True,
            },
            analysis={
                "threshold_source": guideline.id,
                "expected_day_count": expected_days,
                "eligible_incomplete_city_count": len(rows),
                "complete_city_count": complete_count,
                "insufficient_incomplete_city_count": insufficient_count,
                "mostly_filled_city_count": mostly_filled_count,
                "annualization_assumption": "valid days are representative of all days in the analysis year",
                "sampling_interval_assumption": "Wilson binomial approximation treats valid daily exceedance indicators as approximately independent",
                "sampling_interval_scope": "valid-day sampling only; it excludes imputation-model uncertainty and nonrandom-missingness bias",
                "worst_case_bounds_assumption": "all missing days are assigned first to non-exceedance and then to exceedance",
            },
            warnings=warnings,
            visualization={
                "type": "scatter", "x": "data_completeness", "y": "annualized_exceedance_days",
                "size": "valid_day_count", "label": "area_name",
                "x_title": "Valid-day completeness", "y_title": "Annualized exceedance days",
                "x_format": "percent",
            },
            started=started,
        )

    def _episode_ranking(self, question: str, paths: list[str], start: datetime, end: datetime) -> dict[str, Any]:
        started = time.perf_counter()
        expected_days = (end.date() - start.date()).days + 1
        threshold, guideline = self._threshold()
        summaries = self._city_summaries(paths, start, end, threshold)
        minimum_days = math.ceil(expected_days * self.minimum_episode_fraction)
        eligible = summaries[
            (summaries["valid_day_count"] >= minimum_days)
            & (summaries["valid_month_count"] >= self.minimum_episode_months)
        ][["dataset", "area_id"]]
        if eligible.empty:
            raise DataUnavailableError("No city meets the episode coverage and calendar-span requirements")
        daily = self._eligible_daily(paths, eligible, start, end)
        episodes = build_multiday_episodes(daily, threshold)
        guarded = [
            row for row in episodes
            if row["imputed_fraction"] <= self.maximum_episode_imputed_fraction
        ]
        excluded_episode_count = len(episodes) - len(guarded)
        if not guarded:
            raise DataUnavailableError("All multi-day episodes exceed the configured filled-value dependence limit")
        guarded.sort(key=lambda row: (-row["severity"], -row["duration_days"], row["dataset"], row["area_name"]))
        requested = max(_top_n(question.casefold()), 1)
        rows = guarded[:requested]
        for rank, row in enumerate(rows, 1):
            row["rank"] = rank
            row["score_formula"] = "duration_days × mean_exceedance"
        leader = rows[0]
        answer = (
            f"The highest-severity eligible multi-day PM2.5 episode was {leader['area_name']}, {leader['country']}, "
            f"from {leader['start'][:10]} through {leader['end'][:10]}. It lasted {leader['duration_days']} days, "
            f"averaged {leader['mean_exceedance']:.2f} {leader['unit']} above the {threshold:g} {leader['unit']} threshold, "
            f"and scored {leader['severity']:.2f} as duration × mean exceedance. The ranking reports the top {len(rows)} episodes. "
            f"Cities require at least {minimum_days} valid days across {self.minimum_episode_months} months, missing days break episodes, "
            f"and episodes with more than {self.maximum_episode_imputed_fraction:.0%} filled contribution are excluded."
        )
        warnings = []
        if excluded_episode_count:
            warnings.append(
                f"{excluded_episode_count} otherwise valid episodes were excluded because filled values contributed more than the configured limit."
            )
        return self._result(
            rows=rows,
            operation="severe_multiday_episode_ranking",
            operator="guarded_multiday_episode_ranking_v1",
            answer=answer,
            start=start,
            end=end,
            metrics=["rank", "duration_days", "mean_exceedance", "severity", "peak", "mean_daily_coverage"],
            filters={
                "threshold": threshold,
                "minimum_duration_days": 2,
                "minimum_daily_coverage": self.minimum_coverage,
                "minimum_valid_days": minimum_days,
                "minimum_valid_months": self.minimum_episode_months,
                "maximum_episode_imputed_fraction": self.maximum_episode_imputed_fraction,
            },
            analysis={
                "threshold_source": guideline.id,
                "expected_day_count": expected_days,
                "candidate_city_count": len(summaries),
                "eligible_city_count": len(eligible),
                "detected_multiday_episode_count": len(episodes),
                "filled_guard_excluded_episode_count": excluded_episode_count,
                "ranked_episode_count": len(guarded),
                "ranking_formula": "duration_days * mean_exceedance",
                "continuity_rule": "every calendar day must have a coverage-qualified exceedance; missing and non-exceeding days break an episode",
            },
            warnings=warnings,
            visualization={"type": "bar", "x": "area_name", "y": "severity"},
            started=started,
        )

    def _city_summaries(self, paths: list[str], start: datetime, end: datetime, threshold: float):
        connection = duckdb.connect()
        try:
            return connection.execute("""
                SELECT dataset,any_value(country_name) country,area_id,any_value(area_name) area_name,
                       count(DISTINCT bucket_day) FILTER(
                         WHERE coverage>=? AND mean BETWEEN 0 AND ?
                       ) valid_day_count,
                       count(DISTINCT date_trunc('month',bucket_day)) FILTER(
                         WHERE coverage>=? AND mean BETWEEN 0 AND ?
                       ) valid_month_count,
                       count(DISTINCT bucket_day) FILTER(
                         WHERE coverage>=? AND mean BETWEEN 0 AND ? AND mean>?
                       ) exceedance_day_count,
                       avg(coverage) FILTER(WHERE coverage>=? AND mean BETWEEN 0 AND ?) mean_daily_coverage,
                       avg(observed_fraction) FILTER(WHERE coverage>=? AND mean BETWEEN 0 AND ?) observed_fraction,
                       avg(imputed_fraction) FILTER(WHERE coverage>=? AND mean BETWEEN 0 AND ?) imputed_fraction,
                       max(active_station_count) FILTER(WHERE coverage>=? AND mean BETWEEN 0 AND ?) active_station_count,
                       any_value(unit) unit
                FROM read_parquet(?,hive_partitioning=true,union_by_name=true)
                WHERE geography_level='city' AND pollutant='PM25' AND area_id IS NOT NULL
                  AND bucket_day BETWEEN ? AND ?
                GROUP BY dataset,area_id
                ORDER BY dataset,area_id
            """, [
                self.minimum_coverage, self.maximum_valid_pm25,
                self.minimum_coverage, self.maximum_valid_pm25,
                self.minimum_coverage, self.maximum_valid_pm25, threshold,
                self.minimum_coverage, self.maximum_valid_pm25,
                self.minimum_coverage, self.maximum_valid_pm25,
                self.minimum_coverage, self.maximum_valid_pm25,
                self.minimum_coverage, self.maximum_valid_pm25,
                paths, start, end,
            ]).fetchdf()
        finally:
            connection.close()

    def _eligible_daily(self, paths: list[str], eligible, start: datetime, end: datetime) -> list[dict]:
        connection = duckdb.connect()
        connection.register("eligible_episode_cities", eligible)
        try:
            result = connection.execute("""
                SELECT d.dataset,d.country_name country,d.area_id,d.area_name,d.bucket_day,d.mean,d.coverage,
                       d.observed_fraction,d.imputed_fraction,d.active_station_count,d.unit
                FROM read_parquet(?,hive_partitioning=true,union_by_name=true) d
                JOIN eligible_episode_cities e USING(dataset,area_id)
                WHERE d.geography_level='city' AND d.pollutant='PM25'
                  AND d.bucket_day BETWEEN ? AND ? AND d.coverage>=? AND d.mean BETWEEN 0 AND ?
                ORDER BY d.dataset,d.area_id,d.bucket_day
            """, [paths, start, end, self.minimum_coverage, self.maximum_valid_pm25])
            names = [item[0] for item in result.description]
            return [dict(zip(names, row)) for row in result.fetchall()]
        finally:
            connection.close()

    def _latest_complete_year(self, paths: list[str]) -> tuple[datetime, datetime]:
        connection = duckdb.connect()
        try:
            latest = connection.execute("""
                SELECT max(bucket_day) FROM read_parquet(?,hive_partitioning=true,union_by_name=true)
                WHERE geography_level='city' AND pollutant='PM25'
            """, [paths]).fetchone()[0]
        finally:
            connection.close()
        if latest is None:
            raise DataUnavailableError("No PM2.5 city-day date is available")
        end_month_index = latest.year * 12 + latest.month - 2
        end_year, end_zero_month = divmod(end_month_index, 12)
        end_month = end_zero_month + 1
        start_index = end_month_index - 11
        start_year, start_zero_month = divmod(start_index, 12)
        return (
            datetime(start_year, start_zero_month + 1, 1),
            datetime(end_year, end_month, calendar.monthrange(end_year, end_month)[1], 23, 59, 59),
        )

    def _paths(self) -> list[str]:
        return [
            str(path)
            for path in (self.root / "query" / "area_daily").glob("dataset=*/year=*/month=*/*.parquet")
        ]

    @staticmethod
    def _threshold():
        guideline = get_guideline(DEFAULT_GUIDELINE)
        threshold = guideline.thresholds.get("PM25")
        if threshold is None:
            raise DataUnavailableError(f"{guideline.name} does not define a PM2.5 daily threshold")
        return float(threshold), guideline

    def _result(
        self, *, rows: list[dict], operation: str, operator: str, answer: str,
        start: datetime, end: datetime, metrics: list[str], filters: dict,
        analysis: dict, warnings: list[str], visualization: dict, started: float,
    ) -> dict[str, Any]:
        return {
            "kind": "analytics",
            "query_spec": {
                "dataset": "GLOBAL_AREA_DAILY",
                "operation": operation,
                "geography_level": "city",
                "locations": sorted({row["area_name"] for row in rows}),
                "pollutants": ["PM25"],
                "time_range": {"start": start.isoformat(), "end": end.isoformat()},
                "resolution": "daily",
                "metrics": metrics,
                "filters": filters,
            },
            "resolved_period": {
                "start": start.isoformat(),
                "end": end.isoformat(),
                "basis": "latest 12 complete calendar months before the newest city-day value",
            },
            "data": rows,
            "metadata": {
                "dataset": "GLOBAL_AREA_DAILY",
                "geography_level": "city",
                "station_count": max((row.get("active_station_count", 0) for row in rows), default=0),
                "value_status": "observed_and_policy_approved_filled",
                "station_balanced": True,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "warnings": warnings,
            },
            "visualization": visualization,
            "analysis": {"operator": operator, **analysis},
            "answer": answer,
            "display_title": "PM2.5 · annualized burden and severe episodes",
            "model_trace": {
                "provider": "deterministic", "model": operator, "latency_ms": 0,
                "prompt_tokens": None, "completion_tokens": None, "fallback_used": False,
            },
        }


def _plain_value_source(observed_fraction: float, imputed_fraction: float) -> str:
    if imputed_fraction == 0:
        return "measured values only"
    if observed_fraction == 0:
        return "policy-approved filled values only"
    return "measured and policy-approved filled values"


def _evidence_tier(observed_fraction: float, imputed_fraction: float) -> str:
    if observed_fraction == 0:
        return "filled values only"
    if imputed_fraction > 0.5:
        return "mostly filled values"
    return "measurement supported"
