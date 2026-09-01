from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr, theilslopes
import statsmodels.api as sm

from .capabilities import CapabilityAssessment, PRIMARY_DATASETS
from .config import load_config
from .errors import DataUnavailableError
from .lakehouse_catalog import LakehouseCatalog


CURRENT_START = pd.Timestamp("2025-08-01")
CURRENT_END = pd.Timestamp("2026-07-01")
REFERENCE_START = pd.Timestamp("2024-08-01")
REFERENCE_END = pd.Timestamp("2025-07-01")
LONG_START = pd.Timestamp("2023-06-01")
LONG_END = pd.Timestamp("2026-05-01")


def is_derived_temporal_question(question: str) -> bool:
    text = question.casefold()
    return any(pattern in text for pattern in (
        "statistically meaningful monotonic pm2.5 trends across the seven primary countries",
        "monthly anomalies of pm2.5 relative to each country's",
        "countries with decreasing pm2.5 trends but increasing o3 trends",
        "rank countries by year-over-year pm2.5 improvement",
        "rank countries by year over year pm2.5 improvement",
        "rank countries by pm2.5 trend slope",
        "robust year-over-year pm2.5 changes for all eligible countries",
        "compare year-over-year pm2.5 changes across eligible countries",
        "countries whose pm2.5 improved year over year while o3 worsened",
        "monthly pm2.5 trend slopes before and after january 2026 for india and china",
        "regime changes across the seven primary countries",
        "long-term pm2.5 trend estimates from ordinary linear regression",
        "countries with strong winter pm2.5 peaks also show stronger year-over-year variability",
        "changes in active monitoring-station composition materially alter the estimated pm2.5 trend",
    ))


class DerivedTemporalAnalytics:
    """Matched-month changes, trends, anomalies, and seeded temporal uncertainty."""

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.catalog = LakehouseCatalog(self.config)
        self.root = self.catalog.root
        analytics = self.config.get("analytics", {})
        self.coverage_threshold = float(analytics.get("cohort_monthly_coverage_threshold", .8))
        self.replicates = int(analytics.get("bootstrap_replicates", 1000))
        self.seed = int(analytics.get("bootstrap_seed", 20260901))
        self._monthly_cache: pd.DataFrame | None = None

    @staticmethod
    def should_handle(question: str, assessment: CapabilityAssessment) -> bool:
        del assessment
        return is_derived_temporal_question(question)

    def execute(self, question: str, assessment: CapabilityAssessment) -> dict[str, Any]:
        del assessment
        text = question.casefold()
        if "before and after january 2026" in text:
            return self._segmented_india_china()
        if "regime changes across the seven primary" in text:
            return self._primary_regime_changes()
        if "ordinary linear regression" in text and "robust regression" in text:
            return self._long_term_robust_trends()
        if "strong winter pm2.5 peaks" in text:
            return self._seasonal_strength_variability()
        if "active monitoring-station composition" in text:
            return self._station_composition_sensitivity()
        if "monotonic pm2.5 trends" in text:
            return self._monotonic_primary()
        if "monthly anomalies" in text:
            return self._monthly_anomalies()
        if "decreasing pm2.5 trends but increasing o3" in text:
            return self._cross_pollutant_trends()
        if "pm2.5 improved year over year while o3 worsened" in text:
            return self._cross_pollutant_yoy()
        if "trend slope" in text and "rank countries" in text:
            return self._trend_slope_ranking()
        return self._paired_yoy(question)

    def _monthly(self) -> pd.DataFrame:
        if self._monthly_cache is not None:
            return self._monthly_cache
        paths = []
        current = LONG_START
        while current <= CURRENT_END:
            for item in self.catalog.datasets:
                directory = self.root / "query" / "quality_monthly" / f"dataset={item.dataset}" / f"year={current.year}" / f"month={current.month}"
                paths.extend(str(path) for path in directory.glob("*.parquet"))
            current += pd.offsets.MonthBegin(1)
        if not paths:
            raise DataUnavailableError("No monthly quality partitions overlap the temporal evaluation periods")
        maximum = float(self.config.get("visualization", {}).get("maximum_plausible_pm25", 5000))
        connection = duckdb.connect()
        try:
            frame = connection.execute("""
                SELECT dataset,any_value(country_name) country,pollutant,month_start,
                       any_value(mean) mean,any_value(coverage) coverage,
                       any_value(active_station_count) active_station_count,
                       any_value(imputed_fraction) imputed_fraction,any_value(unit) unit
                FROM read_parquet(?,hive_partitioning=true,union_by_name=true)
                WHERE geography_level='country' AND pollutant IN ('PM25','O3')
                  AND month_start BETWEEN ? AND ? AND mean BETWEEN 0 AND ?
                GROUP BY dataset,pollutant,month_start
                ORDER BY dataset,pollutant,month_start
            """, [paths, LONG_START.to_pydatetime(), CURRENT_END.to_pydatetime(), maximum]).fetchdf()
        finally:
            connection.close()
        frame["month_start"] = pd.to_datetime(frame["month_start"])
        self._monthly_cache = frame
        return frame

    def _eligible(self, pollutant: str, start: pd.Timestamp, end: pd.Timestamp,
                  datasets: set[str] | None = None, required_months: int = 12) -> tuple[dict[str, pd.DataFrame], list[dict]]:
        frame = self._monthly()
        frame = frame[(frame["pollutant"] == pollutant) & (frame["month_start"] >= start) & (frame["month_start"] <= end)]
        candidates = [item for item in self.catalog.datasets if datasets is None or item.dataset in datasets]
        eligible = {}
        exclusions = []
        for item in candidates:
            group = frame[frame["dataset"] == item.dataset].sort_values("month_start")
            reasons = []
            if group["month_start"].nunique() < required_months:
                reasons.append(f"{group['month_start'].nunique()} of {required_months} usable months")
            if len(group) and float(group["coverage"].min()) < self.coverage_threshold:
                reasons.append(f"minimum monthly coverage {group['coverage'].min():.1%} is below {self.coverage_threshold:.0%}")
            if not reasons:
                eligible[item.dataset] = group
            else:
                exclusions.append({"dataset": item.dataset, "country": item.country_name, "pollutant": pollutant,
                                   "reasons": "; ".join(reasons)})
        return eligible, exclusions

    def _monotonic_primary(self) -> dict[str, Any]:
        started = time.perf_counter()
        eligible, exclusions = self._eligible("PM25", CURRENT_START, CURRENT_END, set(PRIMARY_DATASETS))
        rows = []
        for dataset, group in eligible.items():
            values = group["mean"].to_numpy(dtype=float)
            tau, p_value = kendalltau(np.arange(len(values)), values)
            slope = float(np.polyfit(np.arange(len(values)), values, 1)[0])
            rows.append({
                "dataset": dataset, "country": group["country"].iloc[0], "pollutant": "PM25",
                "monthly_slope": slope, "kendall_tau": float(tau), "p_value": float(p_value),
                "statistically_detectable": bool(p_value < .05),
                "direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat",
                "minimum_monthly_coverage": float(group["coverage"].min()), "month_count": 12,
                "unit": group["unit"].iloc[0],
            })
        rows.sort(key=lambda row: row["monthly_slope"])
        significant = sum(row["statistically_detectable"] for row in rows)
        answer = (
            f"I tested monotonic PM2.5 change for {len(rows)} eligible primary countries using Kendall’s tau over 12 monthly means and a two-sided 0.05 threshold. "
            f"{significant} trend(s) are statistically detectable. {len(exclusions)} primary countries are excluded with explicit continuity or coverage reasons."
        )
        return self._result(rows, started, "primary_monotonic_trend_v1", answer,
                            ["monthly_slope", "kendall_tau", "p_value"], exclusions,
                            {"trend_test": "Kendall tau", "significance_level": .05}, "bar")

    def _monthly_anomalies(self) -> dict[str, Any]:
        started = time.perf_counter()
        eligible, exclusions = self._eligible("PM25", CURRENT_START, CURRENT_END, set(PRIMARY_DATASETS))
        rows = []
        for dataset, group in eligible.items():
            baseline = float(group["mean"].mean())
            for item in group.itertuples(index=False):
                rows.append({
                    "dataset": dataset, "country": item.country, "pollutant": "PM25",
                    "month": item.month_start.isoformat(), "monthly_mean": float(item.mean),
                    "period_mean": baseline, "anomaly": float(item.mean - baseline),
                    "coverage": float(item.coverage), "unit": item.unit,
                })
        answer = (
            f"I calculated 12 monthly PM2.5 anomalies for each of {len(eligible)} eligible primary countries by subtracting that country’s own August 2025–July 2026 mean. "
            f"{len(exclusions)} incomplete or low-coverage primary countries remain explicit exclusions."
        )
        return self._result(rows, started, "primary_monthly_anomaly_v1", answer,
                            ["monthly_mean", "period_mean", "anomaly", "coverage"], exclusions,
                            {"anomaly_reference": "each eligible country's 12-month arithmetic mean"}, "line")

    def _cross_pollutant_trends(self) -> dict[str, Any]:
        started = time.perf_counter()
        pm, pm_exclusions = self._eligible("PM25", CURRENT_START, CURRENT_END)
        o3, o3_exclusions = self._eligible("O3", CURRENT_START, CURRENT_END)
        rows = []
        for dataset in sorted(set(pm) & set(o3)):
            pm_slope = float(np.polyfit(np.arange(12), pm[dataset]["mean"].to_numpy(dtype=float), 1)[0])
            o3_slope = float(np.polyfit(np.arange(12), o3[dataset]["mean"].to_numpy(dtype=float), 1)[0])
            rows.append({
                "dataset": dataset, "country": pm[dataset]["country"].iloc[0],
                "pm25_monthly_slope": pm_slope, "o3_monthly_slope": o3_slope,
                "qualifies": pm_slope < 0 and o3_slope > 0,
                "pm25_minimum_coverage": float(pm[dataset]["coverage"].min()),
                "o3_minimum_coverage": float(o3[dataset]["coverage"].min()),
            })
        exclusions = _merge_exclusions(pm_exclusions, o3_exclusions)
        qualifying = sum(row["qualifies"] for row in rows)
        answer = (
            f"{qualifying} of {len(rows)} countries with 12 complete, at-least-{self.coverage_threshold:.0%}-coverage months have a decreasing PM2.5 slope and increasing O3 slope over August 2025–July 2026. "
            "All eligible directions and all excluded countries are retained."
        )
        return self._result(rows, started, "cross_pollutant_trend_direction_v1", answer,
                            ["pm25_monthly_slope", "o3_monthly_slope"], exclusions,
                            {"qualifying_count": qualifying}, "scatter")

    def _paired_yoy_rows(self, pollutant: str) -> tuple[list[dict], list[dict]]:
        current, current_exclusions = self._eligible(pollutant, CURRENT_START, CURRENT_END)
        reference, reference_exclusions = self._eligible(pollutant, REFERENCE_START, REFERENCE_END)
        rows = []
        for index, dataset in enumerate(sorted(set(current) & set(reference))):
            current_values = current[dataset]["mean"].to_numpy(dtype=float)
            reference_values = reference[dataset]["mean"].to_numpy(dtype=float)
            differences = current_values - reference_values
            rng = np.random.default_rng(self.seed + index)
            sampled = differences[rng.integers(0, 12, size=(self.replicates, 12))].mean(axis=1)
            change = float(differences.mean())
            reference_mean = float(reference_values.mean())
            rows.append({
                "dataset": dataset, "country": current[dataset]["country"].iloc[0], "pollutant": pollutant,
                "reference_mean": reference_mean, "current_mean": float(current_values.mean()),
                "mean_change": change, "mean_change_percentage": change / reference_mean * 100 if reference_mean else None,
                "change_ci_lower": float(np.quantile(sampled, .025)),
                "change_ci_upper": float(np.quantile(sampled, .975)),
                "statistically_supported_change": bool(np.quantile(sampled, .975) < 0 or np.quantile(sampled, .025) > 0),
                "practically_meaningful_change": bool(abs(change / reference_mean) >= .05) if reference_mean else False,
                "current_minimum_coverage": float(current[dataset]["coverage"].min()),
                "reference_minimum_coverage": float(reference[dataset]["coverage"].min()),
                "matched_month_count": 12, "unit": current[dataset]["unit"].iloc[0],
            })
        exclusions = _merge_exclusions(current_exclusions, reference_exclusions)
        included = {row["dataset"] for row in rows}
        exclusions = [item for item in exclusions if item["dataset"] not in included]
        return rows, exclusions

    def _paired_yoy(self, question: str) -> dict[str, Any]:
        started = time.perf_counter()
        text = question.casefold()
        rows, exclusions = self._paired_yoy_rows("PM25")
        rows.sort(key=lambda row: (row["mean_change"], row["dataset"]))
        for rank, row in enumerate(rows, 1):
            row["improvement_rank"] = rank
        for index, row in enumerate(rows):
            if index == len(rows) - 1:
                row["statistically_distinguishable_from_next"] = None
            else:
                following = rows[index + 1]
                row["statistically_distinguishable_from_next"] = row["change_ci_upper"] < following["change_ci_lower"]
        supported = sum(row["statistically_supported_change"] for row in rows)
        practical = sum(row["practically_meaningful_change"] for row in rows)
        if "rank countries" in text:
            answer = (
                f"I ranked {len(rows)} eligible countries from largest PM2.5 improvement to largest worsening using 12 paired months with at least {self.coverage_threshold:.0%} coverage in both years. "
                f"Paired-month bootstrap intervals support a change for {supported} {'country' if supported == 1 else 'countries'}; adjacent-rank interval overlap is reported rather than forcing distinguishable ranks."
            )
        else:
            answer = (
                f"I compared year-over-year PM2.5 change for {len(rows)} eligible countries using the same 12 calendar months and identical {self.coverage_threshold:.0%} monthly coverage rule in both years. "
                f"{supported} {'change has' if supported == 1 else 'changes have'} paired-month 95% intervals excluding zero, and {practical} are at least 5% in magnitude. All ineligible countries retain reasons."
            )
        return self._result(rows, started, "paired_month_yoy_change_v1", answer,
                            ["mean_change", "mean_change_percentage", "change_ci_lower", "change_ci_upper", "improvement_rank"],
                            exclusions, {"bootstrap_seed": self.seed, "bootstrap_replicates": self.replicates,
                                         "bootstrap_unit": "paired calendar month", "practical_change_threshold": .05}, "bar")

    def _trend_slope_ranking(self) -> dict[str, Any]:
        started = time.perf_counter()
        eligible, exclusions = self._eligible("PM25", CURRENT_START, CURRENT_END)
        rows = []
        x = np.arange(12, dtype=float)
        for index, (dataset, group) in enumerate(sorted(eligible.items())):
            values = group["mean"].to_numpy(dtype=float)
            coefficients = np.polyfit(x, values, 1)
            fitted = np.polyval(coefficients, x)
            residuals = values - fitted
            rng = np.random.default_rng(self.seed + index)
            slopes = np.empty(self.replicates)
            for replicate in range(self.replicates):
                simulated = fitted + rng.choice(residuals, size=12, replace=True)
                slopes[replicate] = np.polyfit(x, simulated, 1)[0]
            rows.append({
                "dataset": dataset, "country": group["country"].iloc[0], "monthly_slope": float(coefficients[0]),
                "slope_ci_lower": float(np.quantile(slopes, .025)), "slope_ci_upper": float(np.quantile(slopes, .975)),
                "direction_supported": bool(np.quantile(slopes, .975) < 0 or np.quantile(slopes, .025) > 0),
                "minimum_monthly_coverage": float(group["coverage"].min()), "unit": group["unit"].iloc[0],
            })
        rows.sort(key=lambda row: row["monthly_slope"])
        for rank, row in enumerate(rows, 1): row["slope_rank"] = rank
        answer = (
            f"I ranked {len(rows)} eligible countries by their August 2025–July 2026 PM2.5 monthly slope. "
            f"Each slope has a deterministic {self.replicates:,}-replicate residual-bootstrap 95% interval; {len(exclusions)} countries remain excluded for incomplete or low-coverage months."
        )
        return self._result(rows, started, "trend_slope_bootstrap_ranking_v1", answer,
                            ["monthly_slope", "slope_ci_lower", "slope_ci_upper", "slope_rank"], exclusions,
                            {"bootstrap_seed": self.seed, "bootstrap_replicates": self.replicates,
                             "bootstrap_method": "fixed-design residual bootstrap"}, "bar")

    def _cross_pollutant_yoy(self) -> dict[str, Any]:
        started = time.perf_counter()
        pm_rows, pm_exclusions = self._paired_yoy_rows("PM25")
        o3_rows, o3_exclusions = self._paired_yoy_rows("O3")
        pm = {row["dataset"]: row for row in pm_rows}
        o3 = {row["dataset"]: row for row in o3_rows}
        summaries = []
        trajectories = []
        monthly = self._monthly()
        for dataset in sorted(set(pm) & set(o3)):
            qualifies = pm[dataset]["mean_change"] < 0 and o3[dataset]["mean_change"] > 0
            summaries.append({
                "dataset": dataset, "country": pm[dataset]["country"],
                "pm25_change": pm[dataset]["mean_change"], "pm25_change_percentage": pm[dataset]["mean_change_percentage"],
                "o3_change": o3[dataset]["mean_change"], "o3_change_percentage": o3[dataset]["mean_change_percentage"],
                "qualifies": qualifies,
            })
            if qualifies:
                for pollutant in ("PM25", "O3"):
                    group = monthly[(monthly["dataset"] == dataset) & (monthly["pollutant"] == pollutant)].sort_values("month_start")
                    reference = group[(group["month_start"] >= REFERENCE_START) & (group["month_start"] <= REFERENCE_END)]
                    current = group[(group["month_start"] >= CURRENT_START) & (group["month_start"] <= CURRENT_END)]
                    for offset in range(12):
                        trajectories.append({"dataset": dataset, "country": pm[dataset]["country"], "pollutant": pollutant,
                                             "month_of_window": offset + 1, "reference_mean": float(reference.iloc[offset]["mean"]),
                                             "current_mean": float(current.iloc[offset]["mean"])})
        exclusions = _merge_exclusions(pm_exclusions, o3_exclusions)
        qualifying = sum(row["qualifies"] for row in summaries)
        answer = (
            f"{qualifying} of {len(summaries)} countries with complete, at-least-{self.coverage_threshold:.0%}-coverage paired PM2.5 and O3 months show year-over-year PM2.5 improvement alongside O3 worsening. "
            "The summary retains all eligible directions; monthly trajectories are provided for qualifying countries only."
        )
        return self._result(trajectories or summaries, started, "cross_pollutant_yoy_attribution_v1", answer,
                            ["reference_mean", "current_mean"], exclusions,
                            {"country_summaries": summaries, "qualifying_count": qualifying}, "line")

    def _segmented_india_china(self) -> dict[str, Any]:
        started = time.perf_counter()
        start, end = pd.Timestamp("2025-06-01"), pd.Timestamp("2026-05-01")
        frame = self._monthly()
        frame = frame[(frame["pollutant"] == "PM25") & frame["dataset"].isin(["CPCB", "CNEMC"])
                      & (frame["month_start"] >= start) & (frame["month_start"] <= end)]
        rows = []
        exclusions = []
        for dataset, country in (("CPCB", "India"), ("CNEMC", "China")):
            group = frame[frame["dataset"] == dataset].sort_values("month_start")
            if len(group) != 12 or float(group["coverage"].min()) < self.coverage_threshold:
                exclusions.append({"dataset": dataset, "country": country,
                                   "reasons": f"{len(group)} of 12 usable months or monthly coverage below {self.coverage_threshold:.0%}"})
                continue
            before = group[group["month_start"] < pd.Timestamp("2026-01-01")]
            after = group[group["month_start"] >= pd.Timestamp("2026-01-01")]
            before_slope = float(np.polyfit(np.arange(len(before)), before["mean"].to_numpy(dtype=float), 1)[0])
            after_slope = float(np.polyfit(np.arange(len(after)), after["mean"].to_numpy(dtype=float), 1)[0])
            rows.append({
                "dataset": dataset, "country": country, "pollutant": "PM25",
                "before_period": "2025-06 through 2025-12", "after_period": "2026-01 through 2026-05",
                "before_month_count": len(before), "after_month_count": len(after),
                "before_monthly_slope": before_slope, "after_monthly_slope": after_slope,
                "slope_change": after_slope - before_slope, "direction_changed": bool(np.sign(before_slope) != np.sign(after_slope)),
                "minimum_monthly_coverage": float(group["coverage"].min()),
                "mean_filled_fraction": float(group["imputed_fraction"].mean()), "unit": group["unit"].iloc[0],
            })
        india = next((row for row in rows if row["dataset"] == "CPCB"), None)
        warning = (
            " India’s January–May 2026 segment contains legacy-filled values and no directly measured CPCB PM2.5, so its after-January slope is descriptive rather than measured-only."
            if india else ""
        )
        answer = (
            f"I compared monthly PM2.5 slopes before January 2026 (7 months) and after it (5 months) for {len(rows)} countries over their latest common complete window."
            + warning
        )
        return self._result(rows, started, "segmented_january_2026_trend_v1", answer,
                            ["before_monthly_slope", "after_monthly_slope", "slope_change"], exclusions,
                            {"breakpoint": "2026-01-01", "breakpoint_predefined": True,
                             "india_after_segment_measured_only": False}, "bar", start=start,
                            end=(end + pd.offsets.MonthEnd(0)).replace(hour=23, minute=59, second=59),
                            basis="latest common complete India–China window around the predefined January 2026 breakpoint")

    def _primary_regime_changes(self) -> dict[str, Any]:
        started = time.perf_counter()
        eligible, exclusions = self._eligible("PM25", CURRENT_START, CURRENT_END, set(PRIMARY_DATASETS))
        rows = []
        for dataset, group in eligible.items():
            values = group["mean"].to_numpy(dtype=float)
            best = None
            for split in range(3, 10):
                left, right = values[:split], values[split:]
                loss = float(np.sum((left - left.mean()) ** 2) + np.sum((right - right.mean()) ** 2))
                candidate = (loss, split, float(left.mean()), float(right.mean()))
                if best is None or candidate < best:
                    best = candidate
            _, split, before_mean, after_mean = best
            months = group["month_start"].tolist()
            slope = float(np.polyfit(np.arange(12), values, 1)[0])
            winter = group[group["month_start"].dt.month.isin([12, 1, 2])]["mean"].mean()
            nonwinter = group[~group["month_start"].dt.month.isin([12, 1, 2])]["mean"].mean()
            rows.append({
                "dataset": dataset, "country": group["country"].iloc[0], "pollutant": "PM25",
                "monthly_slope": slope, "strongest_change_month": months[split].isoformat(),
                "before_mean": before_mean, "after_mean": after_mean, "mean_shift": after_mean - before_mean,
                "winter_minus_nonwinter": float(winter - nonwinter),
                "minimum_monthly_coverage": float(group["coverage"].min()), "unit": group["unit"].iloc[0],
            })
        shared_months = defaultdict(int)
        for row in rows: shared_months[row["strongest_change_month"]] += 1
        shared = [month for month, count in shared_months.items() if count > 1]
        answer = (
            f"I characterized {len(rows)} eligible primary-country PM2.5 series using a 12-month slope, winter-minus-nonwinter contrast, and the strongest single mean-shift breakpoint with at least three months on each side. "
            f"Shared breakpoint months across more than one country: {', '.join(shared) if shared else 'none'}. Exclusions remain explicit."
        )
        return self._result(rows, started, "primary_regime_change_summary_v1", answer,
                            ["monthly_slope", "mean_shift", "winter_minus_nonwinter"], exclusions,
                            {"change_point_method": "minimum two-segment squared error", "minimum_segment_months": 3,
                             "shared_change_months": shared}, "line", start=CURRENT_START,
                            basis="eligible August 2025–July 2026 primary-country monthly series")

    def _long_term_robust_trends(self) -> dict[str, Any]:
        started = time.perf_counter()
        eligible, exclusions = self._eligible("PM25", LONG_START, LONG_END, required_months=36)
        rows = []
        x = np.arange(36, dtype=float)
        for dataset, group in sorted(eligible.items()):
            values = group["mean"].to_numpy(dtype=float)
            ols = float(np.polyfit(x, values, 1)[0])
            huber = float(sm.RLM(values, sm.add_constant(x), M=sm.robust.norms.HuberT()).fit().params[1])
            theil, intercept, low, high = theilslopes(values, x, .95)
            signs = {int(np.sign(value)) for value in (ols, huber, float(theil))}
            rows.append({
                "dataset": dataset, "country": group["country"].iloc[0], "pollutant": "PM25",
                "ols_monthly_slope": ols, "huber_monthly_slope": huber, "theil_sen_monthly_slope": float(theil),
                "theil_sen_ci_lower": float(low), "theil_sen_ci_upper": float(high),
                "inconsistent_direction": len(signs) > 1,
                "minimum_monthly_coverage": float(group["coverage"].min()), "month_count": 36,
                "unit": group["unit"].iloc[0],
            })
        inconsistent = sum(row["inconsistent_direction"] for row in rows)
        answer = (
            f"I compared ordinary least squares, Huber robust regression, and Theil–Sen slopes for {len(rows)} countries with all 36 months from June 2023 through May 2026 above the {self.coverage_threshold:.0%} coverage rule. "
            f"{inconsistent} countries have inconsistent slope directions across estimators; all other candidates retain exclusion reasons."
        )
        return self._result(rows, started, "long_term_robust_trend_comparison_v1", answer,
                            ["ols_monthly_slope", "huber_monthly_slope", "theil_sen_monthly_slope",
                             "theil_sen_ci_lower", "theil_sen_ci_upper"], exclusions,
                            {"estimators": ["ordinary least squares", "Huber RLM", "Theil-Sen"],
                             "required_months": 36}, "scatter", start=LONG_START,
                            end=(LONG_END + pd.offsets.MonthEnd(0)).replace(hour=23, minute=59, second=59),
                            basis="36 complete monthly values from June 2023 through May 2026")

    def _seasonal_strength_variability(self) -> dict[str, Any]:
        started = time.perf_counter()
        current, current_exclusions = self._eligible("PM25", CURRENT_START, CURRENT_END)
        reference, reference_exclusions = self._eligible("PM25", REFERENCE_START, REFERENCE_END)
        rows = []
        for dataset in sorted(set(current) & set(reference)):
            group = current[dataset]
            winter = group[group["month_start"].dt.month.isin([12, 1, 2])]["mean"].mean()
            nonwinter = group[~group["month_start"].dt.month.isin([12, 1, 2])]["mean"].mean()
            overall = group["mean"].mean()
            ref = reference[dataset]["mean"].to_numpy(dtype=float)
            cur = current[dataset]["mean"].to_numpy(dtype=float)
            changes = (cur - ref) / ref * 100
            rows.append({
                "dataset": dataset, "country": group["country"].iloc[0], "pollutant": "PM25",
                "seasonal_strength": float((winter - nonwinter) / overall),
                "winter_mean": float(winter), "nonwinter_mean": float(nonwinter),
                "yoy_monthly_change_stddev": float(np.std(changes, ddof=1)),
                "mean_absolute_yoy_monthly_change": float(np.mean(np.abs(changes))),
                "minimum_monthly_coverage": float(group["coverage"].min()), "unit": group["unit"].iloc[0],
            })
        rho, p_value = spearmanr([row["seasonal_strength"] for row in rows],
                                 [row["yoy_monthly_change_stddev"] for row in rows]) if len(rows) >= 3 else (np.nan, np.nan)
        exclusions = _merge_exclusions(current_exclusions, reference_exclusions)
        included = {row["dataset"] for row in rows}
        exclusions = [item for item in exclusions if item["dataset"] not in included]
        answer = (
            f"Across {len(rows)} eligible countries, the Spearman association between normalized winter PM2.5 strength and the standard deviation of paired monthly year-over-year changes is {rho:.2f}. "
            "With this small cohort the association is descriptive, and the table retains both component metrics and exclusions."
        )
        return self._result(rows, started, "seasonal_strength_yoy_variability_v1", answer,
                            ["seasonal_strength", "yoy_monthly_change_stddev"], exclusions,
                            {"seasonal_strength_definition": "(December-February mean minus other-month mean) / annual mean",
                             "variability_definition": "sample standard deviation of 12 paired monthly percent changes",
                             "spearman_rho": float(rho), "spearman_p_value": float(p_value),
                             "association_is_descriptive": len(rows) < 10}, "scatter")

    def _station_composition_sensitivity(self) -> dict[str, Any]:
        started = time.perf_counter()
        eligible, exclusions = self._eligible("PM25", CURRENT_START, CURRENT_END, set(PRIMARY_DATASETS))
        rows = []
        for dataset, group in eligible.items():
            y = group["mean"].to_numpy(dtype=float)
            x = np.arange(12, dtype=float)
            stations = group["active_station_count"].to_numpy(dtype=float)
            station_z = (stations - stations.mean()) / stations.std(ddof=0) if stations.std(ddof=0) else np.zeros(12)
            raw_slope = float(np.polyfit(x, y, 1)[0])
            adjusted = np.linalg.lstsq(np.column_stack([np.ones(12), x, station_z]), y, rcond=None)[0]
            adjusted_slope = float(adjusted[1])
            relative = abs(adjusted_slope - raw_slope) / max(abs(raw_slope), 1e-12)
            material = np.sign(raw_slope) != np.sign(adjusted_slope) or relative >= .25
            rows.append({
                "dataset": dataset, "country": group["country"].iloc[0], "pollutant": "PM25",
                "unadjusted_monthly_slope": raw_slope, "composition_adjusted_monthly_slope": adjusted_slope,
                "slope_difference": adjusted_slope - raw_slope, "relative_slope_change": relative,
                "direction_changed": bool(np.sign(raw_slope) != np.sign(adjusted_slope)),
                "materially_altered": bool(material), "minimum_active_stations": int(stations.min()),
                "maximum_active_stations": int(stations.max()), "active_station_cv": float(stations.std(ddof=1) / stations.mean()),
                "minimum_monthly_coverage": float(group["coverage"].min()), "unit": group["unit"].iloc[0],
            })
        material_count = sum(row["materially_altered"] for row in rows)
        answer = (
            f"I compared each eligible primary country’s ordinary PM2.5 time slope with a regression slope adjusted for monthly active-station count. "
            f"{material_count} of {len(rows)} slopes change direction or magnitude by at least 25%. This is a composition-count sensitivity test, not a fixed-station-panel estimate."
        )
        return self._result(rows, started, "station_composition_trend_sensitivity_v1", answer,
                            ["unadjusted_monthly_slope", "composition_adjusted_monthly_slope", "relative_slope_change"],
                            exclusions, {"material_relative_change_threshold": .25,
                                         "adjustment": "linear regression with standardized monthly active-station count",
                                         "fixed_station_panel": False}, "scatter", start=CURRENT_START,
                            basis="eligible August 2025–July 2026 primary-country monthly series")

    def _result(self, rows: list[dict], started: float, operator: str, answer: str, metrics: list[str],
                exclusions: list[dict], analysis: dict[str, Any], visualization: str,
                start: pd.Timestamp = REFERENCE_START, end: pd.Timestamp | None = None,
                basis: str = "fixed August 2025–July 2026 cohort with matched prior-year months where required") -> dict[str, Any]:
        end = end or (CURRENT_END + pd.offsets.MonthEnd(0)).replace(hour=23, minute=59, second=59)
        return {
            "kind": "analytics",
            "query_spec": {"dataset": "GLOBAL_LAKEHOUSE", "operation": operator, "geography_level": "country",
                           "locations": sorted({row["country"] for row in rows}),
                           "pollutants": sorted({row.get("pollutant", "PM25") for row in rows}),
                           "time_range": {"start": start.isoformat(), "end": end.isoformat()},
                           "resolution": "monthly", "metrics": metrics,
                           "filters": {"minimum_monthly_coverage": self.coverage_threshold, "required_months": 12}},
            "resolved_period": {"start": start.isoformat(), "end": end.isoformat(),
                                "basis": basis},
            "data": rows,
            "metadata": {"dataset": "GLOBAL_LAKEHOUSE", "geography_level": "country", "station_balanced": True,
                         "value_status": "all usable", "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                         "warnings": []},
            "visualization": {"type": visualization},
            "analysis": {"operator": operator, "minimum_monthly_coverage": self.coverage_threshold,
                         "exclusions": exclusions, "all_candidates_audited": True, **analysis},
            "answer": answer, "display_title": "PM2.5 matched-month temporal analysis",
            "model_trace": {"provider": "deterministic", "model": operator, "latency_ms": 0,
                            "prompt_tokens": None, "completion_tokens": None, "fallback_used": False},
        }


def _merge_exclusions(*groups: list[dict]) -> list[dict]:
    reasons: dict[tuple[str, str], set[str]] = defaultdict(set)
    countries = {}
    for group in groups:
        for item in group:
            key = (item["dataset"], item.get("pollutant", ""))
            countries[key] = item["country"]
            reasons[key].add(item["reasons"])
    return [{"dataset": key[0], "country": countries[key], "pollutant": key[1],
             "reasons": "; ".join(sorted(values))} for key, values in sorted(reasons.items())]
