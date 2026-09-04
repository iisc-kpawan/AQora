from __future__ import annotations

import calendar
import re
import time
from datetime import datetime
from typing import Any, Iterable

import duckdb
import numpy as np
import pandas as pd

from .capabilities import CapabilityAssessment
from .config import load_config
from .errors import DataUnavailableError
from .lakehouse_catalog import LakehouseCatalog


def compound_operation(question: str) -> str | None:
    """Resolve reusable compound intents without relying on an exact sentence."""
    text = question.casefold()
    if "relative variability" in text and len(_pollutants(text)) >= 2:
        return "cross_pollutant_relative_variability"
    if all(term in text for term in ("winter", "summer")) and "pm2.5" in text:
        return "seasonal_contrast"
    if re.search(r"year[- ]over[- ]year", text) and "month" in text and any(
        term in text for term in ("contribut", "driv", "explain")
    ):
        return "year_over_year_month_attribution"
    if "active-station count" in text and any(term in text for term in ("change", "changed", "direction")):
        return "active_station_change"
    if "own 95th-percentile" in text or "own 95th percentile" in text:
        return "historical_percentile_extreme_burden"
    return None


def _pollutants(text: str) -> list[str]:
    patterns = (
        (r"pm\s*2(?:\.|\s*)5|\bpm25\b", "PM25"),
        (r"\bpm\s*10\b", "PM10"),
        (r"\bno\s*2\b", "NO2"),
        (r"\bso\s*2\b", "SO2"),
        (r"\bo\s*3\b|\bozone\b", "O3"),
        (r"\bco\b", "CO"),
    )
    return [name for pattern, name in patterns if re.search(pattern, text)]


class CompoundAnalytics:
    """Derived comparisons composed from published monthly and daily layers."""

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.catalog = LakehouseCatalog(self.config)
        self.root = self.catalog.root
        self.coverage_threshold = float(self.config.get("analytics", {}).get("minimum_daily_coverage", .75))
        self.maximum_value = float(self.config.get("visualization", {}).get("maximum_plausible_pm25", 5000))

    @staticmethod
    def should_handle(question: str, assessment: CapabilityAssessment) -> bool:
        del assessment
        return compound_operation(question) is not None

    def execute(self, question: str, assessment: CapabilityAssessment) -> dict[str, Any]:
        operation = compound_operation(question)
        if operation == "cross_pollutant_relative_variability":
            return self._relative_variability(question, assessment)
        if operation == "seasonal_contrast":
            return self._seasonal_contrast(assessment)
        if operation == "year_over_year_month_attribution":
            return self._year_over_year_attribution(assessment)
        if operation == "active_station_change":
            return self._active_station_change(question)
        if operation == "historical_percentile_extreme_burden":
            return self._historical_extreme_burden()
        raise DataUnavailableError("The requested compound analytical operation is not available")

    def _relative_variability(self, question: str, assessment: CapabilityAssessment) -> dict[str, Any]:
        started = time.perf_counter()
        text = question.casefold()
        pollutants = _pollutants(text)
        if len(pollutants) < 2:
            raise DataUnavailableError("Relative-variability comparison requires at least two pollutants")
        month = _explicit_month(text)
        if month is None:
            raise DataUnavailableError("Relative-variability comparison requires an explicit month")
        datasets = assessment.supported_datasets
        frame = self._quality_months(datasets, pollutants, [month])
        rows: list[dict] = []
        exclusions: list[dict] = []
        for item in self.catalog.selected(datasets):
            group = frame[frame["dataset"] == item.dataset]
            valid: list[dict] = []
            for pollutant in pollutants:
                selected = group[group["pollutant"] == pollutant]
                if selected.empty or float(selected.iloc[0]["coverage"]) < self.coverage_threshold:
                    exclusions.append({"dataset": item.dataset, "country": item.country_name,
                                       "pollutant": pollutant, "reason": "missing or below-coverage monthly values"})
                    continue
                record = selected.iloc[0]
                mean = float(record["mean"])
                coefficient = float(record["std"] / mean) if mean else None
                valid.append({
                    "dataset": item.dataset, "country": item.country_name, "pollutant": pollutant,
                    "mean": mean, "p95": float(record["p95"]), "standard_deviation": float(record["std"]),
                    "coefficient_of_variation": coefficient, "coverage": float(record["coverage"]),
                    "active_station_count": int(record["active_station_count"]), "unit": record["unit"],
                })
            if len(valid) == len(pollutants):
                greatest = max(valid, key=lambda row: row["coefficient_of_variation"] or -np.inf)["pollutant"]
                for row in valid:
                    row["greater_relative_variability"] = row["pollutant"] == greatest
                    rows.append(row)
        if not rows:
            raise DataUnavailableError("No requested country has complete, coverage-qualified values for every pollutant")
        winners = ", ".join(
            f"{country}: {next(row['pollutant'] for row in rows if row['country'] == country and row['greater_relative_variability'])}"
            for country in sorted({row["country"] for row in rows})
        )
        answer = (
            f"I compared {', '.join(pollutants)} for {len({row['country'] for row in rows})} countries in {month:%B %Y}. "
            "Relative variability is the standard deviation of daily equal-station country means divided by their mean. "
            f"The larger relative-variability pollutant is {winners}. Mean, P95, coverage, and station counts are included."
        )
        return self._result(rows, started, "cross_pollutant_relative_variability_v1", answer, month,
                            month + pd.offsets.MonthEnd(0), ["mean", "p95", "coefficient_of_variation"],
                            exclusions, {"relative_variability_definition": "daily standard deviation divided by daily mean"}, "bar")

    def _seasonal_contrast(self, assessment: CapabilityAssessment) -> dict[str, Any]:
        started = time.perf_counter()
        selected = list(self.catalog.selected(assessment.supported_datasets))
        if not selected:
            raise DataUnavailableError("No requested countries resolve for the seasonal comparison")
        common_end = min(pd.Timestamp(item.available_end) for item in selected)
        winter_year = common_end.year if common_end.month >= 2 else common_end.year - 1
        winter = [pd.Timestamp(winter_year - 1, 12, 1), pd.Timestamp(winter_year, 1, 1), pd.Timestamp(winter_year, 2, 1)]
        summer = [pd.Timestamp(winter_year - 1, month, 1) for month in (6, 7, 8)]
        months = summer + winter
        frame = self._quality_months([item.dataset for item in selected], ["PM25"], months)
        rows: list[dict] = []
        exclusions: list[dict] = []
        for item in selected:
            group = frame[frame["dataset"] == item.dataset]
            summaries = {}
            for label, expected in (("summer", summer), ("winter", winter)):
                season = group[group["month_start"].isin(expected)].sort_values("month_start")
                if len(season) != 3 or float(season["coverage"].min()) < self.coverage_threshold:
                    exclusions.append({"dataset": item.dataset, "country": item.country_name,
                                       "season": label, "reason": "three complete coverage-qualified months are required"})
                    continue
                weights = season["valid_day_count"].to_numpy(dtype=float)
                summaries[label] = {
                    "mean": float(np.average(season["mean"], weights=weights)),
                    "p95": float(np.average(season["p95"], weights=weights)),
                    "coverage": float(season["coverage"].min()),
                    "valid_days": int(season["valid_day_count"].sum()),
                    "unit": season["unit"].iloc[0],
                }
            if len(summaries) == 2:
                difference = summaries["winter"]["mean"] - summaries["summer"]["mean"]
                rows.append({
                    "dataset": item.dataset, "country": item.country_name, "pollutant": "PM25",
                    "summer_mean": summaries["summer"]["mean"], "winter_mean": summaries["winter"]["mean"],
                    "winter_minus_summer": difference,
                    "winter_to_summer_ratio": summaries["winter"]["mean"] / summaries["summer"]["mean"] if summaries["summer"]["mean"] else None,
                    "summer_p95": summaries["summer"]["p95"], "winter_p95": summaries["winter"]["p95"],
                    "minimum_coverage": min(summaries["summer"]["coverage"], summaries["winter"]["coverage"]),
                    "summer_valid_days": summaries["summer"]["valid_days"], "winter_valid_days": summaries["winter"]["valid_days"],
                    "unit": summaries["summer"]["unit"],
                })
        if not rows:
            raise DataUnavailableError("No requested country has complete summer and winter PM2.5 seasons")
        answer = (
            f"I compared the latest complete matched seasons before the common data cutoff: June–August {winter_year - 1} and "
            f"December {winter_year - 1}–February {winter_year}. "
            + "; ".join(
                f"{row['country']} winter was {abs(row['winter_minus_summer']):.2f} {row['unit']} "
                f"{'higher' if row['winter_minus_summer'] >= 0 else 'lower'} than summer"
                for row in rows
            ) + ". Each monitoring station has equal influence on the monthly country value, and months are weighted by their valid-day counts within each season."
        )
        return self._result(rows, started, "matched_season_contrast_v1", answer, summer[0], winter[-1] + pd.offsets.MonthEnd(0),
                            ["summer_mean", "winter_mean", "winter_minus_summer"], exclusions,
                            {"summer_months": [value.isoformat() for value in summer],
                             "winter_months": [value.isoformat() for value in winter]}, "bar")

    def _year_over_year_attribution(self, assessment: CapabilityAssessment) -> dict[str, Any]:
        started = time.perf_counter()
        selected = list(self.catalog.selected(assessment.supported_datasets))
        if len(selected) != 1:
            raise DataUnavailableError("Month-level year-over-year attribution requires exactly one country")
        item = selected[0]
        current_end = pd.Timestamp(item.available_end).to_period("M").to_timestamp()
        current = list(pd.date_range(current_end - pd.offsets.MonthBegin(11), current_end, freq="MS"))
        reference = [value - pd.DateOffset(years=1) for value in current]
        frame = self._quality_months([item.dataset], ["PM25"], reference + current)
        frame = frame[(frame["dataset"] == item.dataset) & (frame["pollutant"] == "PM25")]
        lookup = {pd.Timestamp(row.month_start): row for row in frame.itertuples(index=False)}
        rows: list[dict] = []
        missing = []
        for offset, (old_month, new_month) in enumerate(zip(reference, current), 1):
            old = lookup.get(old_month); new = lookup.get(new_month)
            if old is None or new is None:
                missing.append(new_month.strftime("%B %Y")); continue
            difference = float(new.mean - old.mean)
            rows.append({
                "dataset": item.dataset, "country": item.country_name, "pollutant": "PM25",
                "calendar_month": calendar.month_name[new_month.month], "reference_month": old_month.isoformat(),
                "current_month": new_month.isoformat(), "reference_mean": float(old.mean), "current_mean": float(new.mean),
                "difference": difference, "percentage_change": difference / float(old.mean) * 100 if old.mean else None,
                "contribution_to_annual_change": difference / 12,
                "reference_coverage": float(old.coverage), "current_coverage": float(new.coverage), "unit": new.unit,
            })
        if len(rows) != 12:
            raise DataUnavailableError("Year-over-year attribution requires 12 matched months; missing: " + ", ".join(missing))
        ranked = sorted(rows, key=lambda row: abs(row["contribution_to_annual_change"]), reverse=True)
        for rank, row in enumerate(ranked, 1):
            row["absolute_contribution_rank"] = rank
            row["top_three_contributor"] = rank <= 3
        rows.sort(key=lambda row: row["current_month"])
        overall_change = float(np.mean([row["difference"] for row in rows]))
        reference_mean = float(np.mean([row["reference_mean"] for row in rows]))
        top = [row["calendar_month"] for row in ranked[:3]]
        direction = "improved" if overall_change < 0 else "worsened" if overall_change > 0 else "was unchanged"
        answer = (
            f"{item.country_name} PM2.5 {direction} by {abs(overall_change):.2f} {rows[0]['unit']} "
            f"({abs(overall_change / reference_mean * 100):.1f}%) across 12 matched calendar months ending {current_end:%B %Y}. "
            f"The three largest absolute month contributions are {', '.join(top)}. The table retains both annual profiles, coverage, and every monthly contribution."
        )
        return self._result(rows, started, "paired_month_yoy_attribution_v1", answer, reference[0], current[-1] + pd.offsets.MonthEnd(0),
                            ["reference_mean", "current_mean", "difference", "contribution_to_annual_change"], [],
                            {"overall_mean_change": overall_change,
                             "overall_percentage_change": overall_change / reference_mean * 100 if reference_mean else None,
                             "top_contributing_months": top, "matched_month_count": 12}, "line")

    def _active_station_change(self, question: str) -> dict[str, Any]:
        started = time.perf_counter()
        months = _all_explicit_months(question.casefold())
        if len(months) != 2:
            raise DataUnavailableError("Active-station comparison requires exactly two explicit months")
        first, second = sorted(months)
        threshold_match = re.search(r"more than\s+(\d+(?:\.\d+)?)%", question.casefold())
        threshold = float(threshold_match.group(1)) if threshold_match else 20.0
        connection = duckdb.connect(str(self.catalog.control_path), read_only=True)
        try:
            frame = connection.execute("""
                SELECT dataset,any_value(country_name) country,month_start,
                       max(active_station_count) active_station_count,
                       max(usable_count) usable_count,max(usable_coverage) coverage
                FROM pollutant_monthly_availability
                WHERE pollutant='PM25' AND month_start IN (?,?)
                GROUP BY dataset,month_start ORDER BY dataset,month_start
            """, [first.to_pydatetime(), second.to_pydatetime()]).fetchdf()
        finally:
            connection.close()
        rows = []; exclusions = []; nonqualifying = []
        for item in self.catalog.datasets:
            group = frame[frame["dataset"] == item.dataset]
            lookup = {pd.Timestamp(row.month_start): row for row in group.itertuples(index=False)}
            old = lookup.get(first); new = lookup.get(second)
            if old is None or new is None or old.usable_count <= 0 or new.usable_count <= 0 or old.active_station_count <= 0:
                exclusions.append({"dataset": item.dataset, "country": item.country_name,
                                   "reason": "both months require usable PM2.5 values and a nonzero initial station count"})
                continue
            change = int(new.active_station_count - old.active_station_count)
            percentage = change / int(old.active_station_count) * 100
            if abs(percentage) > threshold:
                rows.append({
                    "dataset": item.dataset, "country": item.country_name,
                    "first_month": first.isoformat(), "second_month": second.isoformat(),
                    "first_active_station_count": int(old.active_station_count),
                    "second_active_station_count": int(new.active_station_count), "station_count_change": change,
                    "percentage_change": percentage, "direction": "increase" if change > 0 else "decrease",
                    "first_month_coverage": float(old.coverage), "second_month_coverage": float(new.coverage),
                })
            else:
                nonqualifying.append({
                    "dataset": item.dataset, "country": item.country_name,
                    "percentage_change": percentage,
                    "reason": f"absolute change did not exceed {threshold:g}%",
                })
        rows.sort(key=lambda row: (-abs(row["percentage_change"]), row["dataset"]))
        evaluated_count = len(rows) + len(nonqualifying)
        answer = (
            f"Of {evaluated_count} countries with usable PM2.5 values in both {first:%B %Y} and {second:%B %Y}, "
            f"{len(rows)} changed active-station count by more than {threshold:g}%. "
            f"The direction and both monthly counts are reported; {len(exclusions)} countries are excluded rather than treated as zero-station changes."
        )
        return self._result(rows, started, "active_station_month_change_v1", answer, first, second + pd.offsets.MonthEnd(0),
                            ["first_active_station_count", "second_active_station_count", "percentage_change"], exclusions,
                            {"absolute_percentage_threshold_exclusive": threshold, "pollutant": "PM25",
                             "evaluated_country_count": evaluated_count,
                             "nonqualifying_count": len(nonqualifying),
                             "nonqualifying": nonqualifying}, "bar")

    def _historical_extreme_burden(self) -> dict[str, Any]:
        started = time.perf_counter()
        baseline_start = pd.Timestamp("2024-08-01"); baseline_end = pd.Timestamp("2025-07-31")
        current_start = pd.Timestamp("2025-08-01"); current_end = pd.Timestamp("2026-07-31")
        months = list(pd.date_range(baseline_start, current_end, freq="MS"))
        datasets = [item.dataset for item in self.catalog.datasets]
        area_paths = self._paths("area_daily", datasets, months)
        station_paths = self._paths("daily", datasets, months)
        if not area_paths and not station_paths:
            raise DataUnavailableError("No country-day values overlap the extreme-burden periods")
        connection = duckdb.connect()
        try:
            pieces = []
            if station_paths:
                pieces.append(connection.execute("""
                    SELECT dataset,any_value(country_name) country,bucket_start bucket_day,
                           avg(mean) mean,sum(valid_hour_count)::DOUBLE/nullif(sum(expected_hour_count),0) coverage,
                           count(mean)::BIGINT active_station_count,any_value(unit) unit,0 source_priority
                    FROM read_parquet(?,hive_partitioning=true,union_by_name=true)
                    WHERE pollutant='PM25' AND bucket_start BETWEEN ? AND ? AND mean BETWEEN 0 AND ?
                    GROUP BY dataset,bucket_start
                """, [station_paths, baseline_start.to_pydatetime(), current_end.to_pydatetime(), self.maximum_value]).fetchdf())
            if area_paths:
                pieces.append(connection.execute("""
                    SELECT dataset,any_value(country_name) country,bucket_day,any_value(mean) mean,
                           any_value(coverage) coverage,any_value(active_station_count) active_station_count,
                           any_value(unit) unit,1 source_priority
                    FROM read_parquet(?,hive_partitioning=true,union_by_name=true)
                    WHERE geography_level='country' AND pollutant='PM25' AND aggregation_method='equal_station'
                      AND bucket_day BETWEEN ? AND ? AND mean BETWEEN 0 AND ?
                    GROUP BY dataset,bucket_day
                """, [area_paths, baseline_start.to_pydatetime(), current_end.to_pydatetime(), self.maximum_value]).fetchdf())
            frame = pd.concat(pieces, ignore_index=True)
        finally:
            connection.close()
        frame["bucket_day"] = pd.to_datetime(frame["bucket_day"])
        frame = frame.sort_values("source_priority").drop_duplicates(["dataset", "bucket_day"], keep="last")
        frame = frame[frame["coverage"] >= self.coverage_threshold]
        rows = []; exclusions = []
        required_days = int(np.ceil(365 * .8))
        for item in self.catalog.datasets:
            group = frame[frame["dataset"] == item.dataset]
            baseline = group[(group["bucket_day"] >= baseline_start) & (group["bucket_day"] <= baseline_end)]
            current = group[(group["bucket_day"] >= current_start) & (group["bucket_day"] <= current_end)]
            if len(baseline) < required_days or len(current) < required_days:
                exclusions.append({"dataset": item.dataset, "country": item.country_name,
                                   "reason": f"requires at least {required_days} coverage-qualified days in both periods",
                                   "baseline_valid_days": len(baseline), "evaluation_valid_days": len(current)})
                continue
            threshold = float(np.quantile(baseline["mean"].to_numpy(dtype=float), .95))
            extremes = current[current["mean"] > threshold]
            rows.append({
                "dataset": item.dataset, "country": item.country_name, "pollutant": "PM25",
                "historical_p95_threshold": threshold, "extreme_day_count": int(len(extremes)),
                "evaluation_valid_day_count": int(len(current)),
                "extreme_day_percentage": float(len(extremes) / len(current) * 100),
                "mean_extreme_concentration": float(extremes["mean"].mean()) if len(extremes) else None,
                "maximum_extreme_concentration": float(extremes["mean"].max()) if len(extremes) else None,
                "minimum_daily_coverage": float(current["coverage"].min()),
                "median_active_station_count": float(current["active_station_count"].median()),
                "unit": current["unit"].iloc[0],
            })
        rows.sort(key=lambda row: (-row["extreme_day_count"], row["dataset"]))
        for rank, row in enumerate(rows, 1): row["extreme_burden_rank"] = rank
        answer = (
            f"I ranked {len(rows)} eligible countries by the number of August 2025–July 2026 PM2.5 days above each country’s own August 2024–July 2025 daily 95th percentile. "
            f"Eligibility requires at least {required_days} days meeting the {self.coverage_threshold:.0%} daily coverage rule in both periods. "
            f"{len(exclusions)} countries remain explicit exclusions."
        )
        return self._result(rows, started, "historical_percentile_extreme_burden_v1", answer, baseline_start, current_end,
                            ["historical_p95_threshold", "extreme_day_count", "extreme_day_percentage", "extreme_burden_rank"], exclusions,
                            {"baseline_period": [baseline_start.isoformat(), baseline_end.isoformat()],
                             "evaluation_period": [current_start.isoformat(), current_end.isoformat()],
                             "percentile": .95, "minimum_valid_days_per_period": required_days}, "bar")

    def _quality_months(self, datasets: Iterable[str], pollutants: list[str], months: list[pd.Timestamp]) -> pd.DataFrame:
        paths = self._paths("quality_monthly", datasets, months)
        if not paths:
            raise DataUnavailableError("No monthly quality values overlap the requested analysis")
        connection = duckdb.connect()
        try:
            frame = connection.execute("""
                SELECT dataset,any_value(country_name) country,pollutant,month_start,
                       any_value(mean) mean,any_value(p95) p95,any_value(std) std,
                       any_value(valid_day_count) valid_day_count,any_value(coverage) coverage,
                       any_value(active_station_count) active_station_count,any_value(unit) unit
                FROM read_parquet(?,hive_partitioning=true,union_by_name=true)
                WHERE geography_level='country' AND pollutant IN (SELECT unnest(?))
                GROUP BY dataset,pollutant,month_start ORDER BY dataset,pollutant,month_start
            """, [paths, pollutants]).fetchdf()
        finally:
            connection.close()
        frame["month_start"] = pd.to_datetime(frame["month_start"])
        return frame

    def _paths(self, layer: str, datasets: Iterable[str], months: Iterable[pd.Timestamp]) -> list[str]:
        paths: list[str] = []
        for dataset in datasets:
            for month in sorted({pd.Timestamp(value).to_period("M").to_timestamp() for value in months}):
                directory = self.root / "query" / layer / f"dataset={dataset}" / f"year={month.year}" / f"month={month.month}"
                paths.extend(str(path) for path in directory.glob("*.parquet"))
        return paths

    def _result(self, rows: list[dict], started: float, operator: str, answer: str,
                start: pd.Timestamp, end: pd.Timestamp, metrics: list[str], exclusions: list[dict],
                analysis: dict[str, Any], visualization: str) -> dict[str, Any]:
        return {
            "kind": "analytics",
            "query_spec": {"dataset": "GLOBAL_LAKEHOUSE", "operation": operator, "geography_level": "country",
                           "locations": sorted({row.get("country", "") for row in rows if row.get("country")}),
                           "pollutants": sorted({row.get("pollutant", "PM25") for row in rows}),
                           "time_range": {"start": pd.Timestamp(start).isoformat(), "end": pd.Timestamp(end).isoformat()},
                           "resolution": "derived", "metrics": metrics,
                           "filters": {"minimum_coverage": self.coverage_threshold}},
            "resolved_period": {"start": pd.Timestamp(start).isoformat(), "end": pd.Timestamp(end).isoformat(),
                                "basis": "explicit request or latest complete data-derived comparison window"},
            "data": rows,
            "metadata": {"dataset": "GLOBAL_LAKEHOUSE", "geography_level": "country", "station_balanced": True,
                         "value_status": "all usable", "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                         "warnings": []},
            "visualization": {"type": visualization},
            "analysis": {"operator": operator, "exclusions": exclusions, "all_candidates_audited": True, **analysis},
            "answer": answer, "display_title": "Compound air-quality analysis",
            "model_trace": {"provider": "deterministic", "model": operator, "latency_ms": 0,
                            "prompt_tokens": None, "completion_tokens": None, "fallback_used": False},
        }


def _explicit_month(text: str) -> pd.Timestamp | None:
    months = _all_explicit_months(text)
    return months[0] if len(months) == 1 else None


def _all_explicit_months(text: str) -> list[pd.Timestamp]:
    names = {name.casefold(): index for index, name in enumerate(calendar.month_name) if name}
    names.update({name.casefold(): index for index, name in enumerate(calendar.month_abbr) if name})
    pattern = "|".join(sorted(names, key=len, reverse=True))
    values = [pd.Timestamp(int(match.group(2)), names[match.group(1)], 1)
              for match in re.finditer(rf"\b({pattern})\s+(20\d{{2}})\b", text)]
    return list(dict.fromkeys(values))
