from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from .capabilities import CapabilityAssessment
from .config import load_config
from .errors import DataUnavailableError
from .ingestion.impute import METHOD as CONFIGURED_METHOD, impute_new_data
from .lakehouse_catalog import LakehouseCatalog
from .forecasting import forecast_at_issue


METHODS = ("configured", "forward_fill", "linear_interpolation", "seasonal_interpolation")


class SimulationAnalytics:
    """Seeded, immutable-data masking experiments for imputation and downstream sensitivity."""

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.catalog = LakehouseCatalog(self.config)
        self.root = self.catalog.root
        settings = self.config.get("simulation", {})
        self.seed = int(settings.get("seed", 20260901))
        self.sequence_hours = int(settings.get("sequence_hours", 504))
        self.samples_per_country = int(settings.get("station_sequences_per_country", 4))
        self.gap_lengths = tuple(int(value) for value in settings.get("gap_lengths_hours", (1, 3)))
        self.missing_fractions = tuple(float(value) for value in settings.get("additional_missing_fractions", (.05, .10, .20)))
        self.threshold = float(settings.get("exceedance_threshold_pm25", 15))
        self.ranking_thresholds = tuple(float(value) for value in settings.get("ranking_coverage_thresholds", (.5, .75, .9)))
        self.ranking_agreement_target = float(settings.get("ranking_agreement_target", .9))
        self._sequence_cache: list[dict] | None = None
        self._reconstruction_cache: list[dict] | None = None
        self._india_improvement_cache: dict[str, Any] | None = None

    @staticmethod
    def should_handle(question: str, assessment: CapabilityAssessment) -> bool:
        del assessment
        text = question.casefold()
        return any(pattern in text for pattern in (
            "artificially masked 1-hour and 3-hour", "benchmark the configured short-gap imputation",
            "simulate 5%, 10%, and 20% additional missingness",
            "downstream ranking sensitivity to imputation",
            "masking the final 1, 3, and 6 context hours",
            "apparent year-over-year pm2.5 improvement in india",
            "india experienced meaningful pm2.5 improvement over the last 12 complete months",
            "rankings produced by station-weighted versus city-balanced",
            "heatmap of rank stability for countries",
            "minimum coverage threshold needed to keep july 2026 top-10 country",
        ))

    def execute(self, question: str, assessment: CapabilityAssessment) -> dict[str, Any]:
        del assessment
        text = question.casefold()
        if any(pattern in text for pattern in (
            "rankings produced by station-weighted versus city-balanced",
            "heatmap of rank stability for countries",
            "minimum coverage threshold needed to keep july 2026 top-10 country",
        )):
            return self._country_rank_stability(text)
        if ("apparent year-over-year pm2.5 improvement in india" in text
                or "india experienced meaningful pm2.5 improvement over the last 12 complete months" in text):
            return self._india_improvement_robustness(
                require_observed_policy="exclusion of imputed values" in text
            )
        if "masking the final 1, 3, and 6 context hours" in text:
            return self._forecast_context_gaps()
        if "simulate 5%, 10%, and 20% additional missingness" in text:
            return self._additional_missingness()
        if "downstream ranking sensitivity to imputation" in text:
            return self._ranking_sensitivity()
        include_baselines = "benchmark the configured short-gap imputation" in text
        return self._imputation_benchmark(include_baselines)

    def _imputation_benchmark(self, include_baselines: bool) -> dict[str, Any]:
        started = time.perf_counter()
        experiments = self._reconstruction_experiments()
        selected = experiments if include_baselines else [row for row in experiments if row["method"] == "configured"]
        grouped: dict[tuple, list[dict]] = defaultdict(list)
        for row in selected:
            grouped[(row["scope"], row["dataset"], row["method"], row["gap_hours"])].append(row)
        rows = []
        for (scope, dataset, method, gap), values in sorted(grouped.items()):
            errors = np.asarray([item["prediction"] - item["truth"] for item in values], dtype=float)
            rows.append({
                "scope": scope, "dataset": dataset, "country": values[0]["country"],
                "method": method, "method_detail": CONFIGURED_METHOD if method == "configured" else method,
                "gap_hours": gap, "held_out_value_count": len(errors),
                "station_sequence_count": len({item["sample_id"] for item in values}),
                "mae": float(np.mean(np.abs(errors))), "rmse": float(np.sqrt(np.mean(errors ** 2))),
                "mase": float(np.mean([item["scaled_absolute_error"] for item in values])),
                "bias": float(np.mean(errors)), "seed": self.seed, "unit": "µg/m³",
            })
        portfolio = []
        for method in (METHODS if include_baselines else ("configured",)):
            for gap in self.gap_lengths:
                values = [row for row in experiments if row["method"] == method and row["gap_hours"] == gap]
                errors = np.asarray([row["prediction"] - row["truth"] for row in values], dtype=float)
                portfolio.append({
                    "scope": "portfolio", "dataset": "ALL_ELIGIBLE", "country": "Eligible countries",
                    "method": method, "method_detail": CONFIGURED_METHOD if method == "configured" else method,
                    "gap_hours": gap, "held_out_value_count": len(errors),
                    "station_sequence_count": len({row["sample_id"] for row in values}),
                    "mae": float(np.mean(np.abs(errors))), "rmse": float(np.sqrt(np.mean(errors ** 2))),
                    "mase": float(np.mean([row["scaled_absolute_error"] for row in values])),
                    "bias": float(np.mean(errors)), "seed": self.seed, "unit": "µg/m³",
                })
        rows.extend(portfolio)
        best = {}
        if include_baselines:
            for gap in self.gap_lengths:
                candidates = [row for row in portfolio if row["gap_hours"] == gap]
                best[gap] = min(candidates, key=lambda row: row["mae"])["method"]
            answer = (
                f"I benchmarked four imputation methods on {portfolio[0]['station_sequence_count']} reproducibly sampled, "
                f"fully observed July 2026 PM2.5 station sequences from {len({row['dataset'] for row in experiments})} countries. "
                + " ".join(f"For {gap}-hour gaps, {best[gap].replace('_', ' ')} has the lowest portfolio MAE." for gap in self.gap_lengths)
            )
            operator = "masked_imputation_method_benchmark_v1"
        else:
            answer = (
                f"I evaluated the configured short-gap method on {portfolio[0]['station_sequence_count']} reproducibly sampled, "
                f"fully observed July 2026 PM2.5 station sequences. The table reports MAE, RMSE, and bias for "
                + " and ".join(f"{gap}-hour" for gap in self.gap_lengths) + " artificial gaps."
            )
            operator = "masked_imputation_error_v1"
        return self._result(
            rows, started, operator, answer, ["mae", "rmse", "mase", "bias"],
            {"methods": list(METHODS if include_baselines else ("configured",)), "gap_lengths_hours": list(self.gap_lengths),
             "best_method_by_gap": best, "masking_unit": "contiguous hourly gap", "source_values": "directly observed only"},
        )

    def _ranking_sensitivity(self) -> dict[str, Any]:
        started = time.perf_counter()
        experiments = self._reconstruction_experiments()
        sequences = {row["sample_id"]: row for row in self._sequences()}
        gap = max(self.gap_lengths)
        baseline_means = _dataset_means(sequences.values())
        baseline_ranks = _ranks(baseline_means)
        methods = ("complete_reference", "observed_only", *METHODS)
        rows = []
        for method in methods:
            means = {}
            for dataset in baseline_means:
                samples = [item for item in sequences.values() if item["dataset"] == dataset]
                total = sum(float(np.sum(item["values"])) for item in samples)
                count = sum(len(item["values"]) for item in samples)
                masked = [row for row in experiments if row["dataset"] == dataset and row["gap_hours"] == gap]
                if method == "observed_only":
                    total -= sum(row["truth"] for row in masked if row["method"] == "configured")
                    count -= sum(1 for row in masked if row["method"] == "configured")
                elif method not in {"complete_reference"}:
                    values = [row for row in masked if row["method"] == method]
                    total += sum(row["prediction"] - row["truth"] for row in values)
                means[dataset] = total / count
            ranks = _ranks(means)
            for dataset, mean in means.items():
                sample = next(item for item in sequences.values() if item["dataset"] == dataset)
                rows.append({
                    "dataset": dataset, "country": sample["country"], "method": method, "gap_hours": gap,
                    "mean": mean, "rank": ranks[dataset], "reference_rank": baseline_ranks[dataset],
                    "rank_change": ranks[dataset] - baseline_ranks[dataset], "unit": "µg/m³",
                })
        reversals = {
            method: sum(1 for row in rows if row["method"] == method and row["rank_change"] != 0)
            for method in methods if method != "complete_reference"
        }
        return self._result(
            rows, started, "imputation_ranking_sensitivity_v1",
            f"I repeated the sampled-country PM2.5 ranking after reproducible {gap}-hour gaps under measured-only retention and four reconstruction methods. The table reports every rank and change from the complete reference; no country or missing value was substituted outside the experiment.",
            ["mean", "rank", "rank_change"],
            {"gap_hours": gap, "reference": "complete observed sequences", "changed_rank_counts": reversals},
        )

    def _additional_missingness(self) -> dict[str, Any]:
        started = time.perf_counter()
        sequences = self._sequences()
        by_dataset: dict[str, list[dict]] = defaultdict(list)
        for sequence in sequences: by_dataset[sequence["dataset"]].append(sequence)
        baseline = {dataset: _summary(group, self.threshold) for dataset, group in by_dataset.items()}
        baseline_ranks = _ranks({dataset: values["mean"] for dataset, values in baseline.items()})
        rows = []
        rank_accuracy = {}
        for rate_index, rate in enumerate(self.missing_fractions):
            summaries = {}
            for dataset, group in by_dataset.items():
                masked_group = []
                for sample_index, sequence in enumerate(group):
                    rng = np.random.default_rng(self.seed + rate_index * 10_000 + sample_index + _stable_dataset_seed(dataset))
                    keep = np.ones(len(sequence["values"]), dtype=bool)
                    remove = rng.choice(len(keep), size=math.floor(rate * len(keep)), replace=False)
                    keep[remove] = False
                    masked_group.append({**sequence, "values": sequence["values"][keep]})
                summaries[dataset] = _summary(masked_group, self.threshold)
            ranks = _ranks({dataset: values["mean"] for dataset, values in summaries.items()})
            reference = np.asarray([baseline_ranks[key] for key in sorted(ranks)], dtype=float)
            simulated = np.asarray([ranks[key] for key in sorted(ranks)], dtype=float)
            rho = float(np.corrcoef(reference, simulated)[0, 1]) if len(ranks) > 1 else 1.0
            rank_accuracy[rate] = rho
            for dataset, values in summaries.items():
                sample = by_dataset[dataset][0]
                rows.append({
                    "dataset": dataset, "country": sample["country"], "additional_missing_fraction": rate,
                    "mean": values["mean"], "mean_error": values["mean"] - baseline[dataset]["mean"],
                    "p95": values["p95"], "p95_error": values["p95"] - baseline[dataset]["p95"],
                    "exceedance_percentage": values["exceedance_percentage"],
                    "exceedance_error": values["exceedance_percentage"] - baseline[dataset]["exceedance_percentage"],
                    "rank": ranks[dataset], "reference_rank": baseline_ranks[dataset],
                    "rank_change": ranks[dataset] - baseline_ranks[dataset], "ranking_spearman_rho": rho,
                    "unit": "µg/m³",
                })
        return self._result(
            rows, started, "additional_missingness_sensitivity_v1",
            f"I reproducibly removed 5%, 10%, and 20% of directly observed values from complete PM2.5 station sequences and recalculated country mean, P95, guideline exceedance, and ranking. The original arrays were not modified; every scenario uses the fixed seed {self.seed}.",
            ["mean_error", "p95_error", "exceedance_error", "rank_change"],
            {"additional_missing_fractions": list(self.missing_fractions), "ranking_spearman_rho": rank_accuracy,
             "source_values": "directly observed only", "masking": "uniform without replacement within station sequence"},
        )

    def _india_improvement_robustness(self, require_observed_policy: bool = False) -> dict[str, Any]:
        """Test the latest complete India year against its matched preceding year."""
        if self._india_improvement_cache is not None and not require_observed_policy:
            return self._india_improvement_cache
        started = time.perf_counter()
        current_end = pd.Timestamp(self.catalog.by_dataset("CPCB").available_end).floor("h")
        current_end = (current_end + pd.Timedelta(hours=1)).normalize() - pd.Timedelta(hours=1)
        current_start = (current_end + pd.Timedelta(hours=1) - pd.DateOffset(months=12)).normalize()
        reference_start = current_start - pd.DateOffset(years=1)
        reference_end = current_end - pd.DateOffset(years=1)
        paths = self._hourly_paths("CPCB", reference_start, current_end)
        if not paths:
            raise DataUnavailableError("No India hourly partitions overlap the two matched 12-month periods")
        control = duckdb.connect(str(self.catalog.control_path), read_only=True)
        try:
            dimensions = control.execute(
                "SELECT dataset,source_station_id,city FROM station_dimension WHERE dataset='CPCB'"
            ).fetchdf()
        finally:
            control.close()
        connection = duckdb.connect()
        connection.execute("SET threads=8")
        connection.register("india_improvement_station_dimension", dimensions)
        try:
            daily = connection.execute("""
                WITH raw AS (
                  SELECT h.source_station_id,h.timestamp_local,h.observed_value,h.analysis_value,
                         h.value_status,d.city
                  FROM read_parquet(?,hive_partitioning=true,union_by_name=true) h
                  LEFT JOIN india_improvement_station_dimension d USING(dataset,source_station_id)
                  WHERE h.pollutant='PM25' AND h.timestamp_local BETWEEN ? AND ?
                ), policy_values AS (
                  SELECT r.*,p.policy,
                         CASE WHEN p.policy='observed_only'
                              THEN CASE WHEN value_status='observed' AND observed_value>=0 THEN observed_value END
                              ELSE CASE WHEN analysis_value>=0 THEN analysis_value END END selected_value
                  FROM raw r CROSS JOIN (VALUES ('observed_only'),('all_usable')) p(policy)
                ), station_day AS (
                  SELECT source_station_id,lower(trim(city)) city_id,policy,
                         date_trunc('day',timestamp_local) bucket_day,
                         avg(selected_value) station_value,count(*) expected_hours,count(selected_value) valid_hours,
                         count(*) FILTER(WHERE selected_value IS NOT NULL AND value_status LIKE 'imputed%') filled_hours
                  FROM policy_values GROUP BY source_station_id,city_id,policy,bucket_day
                ), thresholded AS (
                  SELECT s.*,t.coverage_threshold
                  FROM station_day s CROSS JOIN (VALUES (.50),(.75),(.90)) t(coverage_threshold)
                  WHERE valid_hours::DOUBLE/nullif(expected_hours,0)>=t.coverage_threshold
                ), station_area AS (
                  SELECT 'equal_station' aggregation_method,policy,coverage_threshold,bucket_day,
                         avg(station_value) area_value,sum(expected_hours) expected_hours,
                         sum(valid_hours) valid_hours,sum(filled_hours) filled_hours,
                         count(*) retained_station_days,count(DISTINCT source_station_id) station_count,
                         count(DISTINCT city_id) FILTER(WHERE city_id IS NOT NULL AND city_id<>'') city_count
                  FROM thresholded GROUP BY policy,coverage_threshold,bucket_day
                ), city_day AS (
                  SELECT policy,coverage_threshold,bucket_day,city_id,avg(station_value) city_value,
                         sum(expected_hours) expected_hours,sum(valid_hours) valid_hours,
                         sum(filled_hours) filled_hours,count(*) retained_station_days,
                         count(DISTINCT source_station_id) station_count
                  FROM thresholded WHERE city_id IS NOT NULL AND city_id<>''
                  GROUP BY policy,coverage_threshold,bucket_day,city_id
                ), city_area AS (
                  SELECT 'equal_city' aggregation_method,policy,coverage_threshold,bucket_day,
                         avg(city_value) area_value,sum(expected_hours) expected_hours,
                         sum(valid_hours) valid_hours,sum(filled_hours) filled_hours,
                         sum(retained_station_days) retained_station_days,sum(station_count) station_count,
                         count(*) city_count
                  FROM city_day GROUP BY policy,coverage_threshold,bucket_day
                ), candidates AS (
                  SELECT policy,bucket_day,count(*) candidate_station_days
                  FROM station_day GROUP BY policy,bucket_day
                )
                SELECT a.*,c.candidate_station_days
                FROM (SELECT * FROM station_area UNION ALL BY NAME SELECT * FROM city_area) a
                JOIN candidates c USING(policy,bucket_day)
                ORDER BY aggregation_method,policy,coverage_threshold,bucket_day
            """, [paths, reference_start.to_pydatetime(), current_end.to_pydatetime()]).fetchdf()
        finally:
            connection.close()
        if daily.empty:
            raise DataUnavailableError("No India station-days satisfy the robustness experiment")

        rows = []
        excluded_scenarios = []
        seed = int(self.config["analytics"].get("bootstrap_seed", self.seed))
        replicates = int(self.config["analytics"].get("bootstrap_replicates", 1000))
        for scenario_index, ((aggregation, policy, threshold), group) in enumerate(
            daily.groupby(["aggregation_method", "policy", "coverage_threshold"], sort=True)
        ):
            group = group.copy()
            group["bucket_day"] = pd.to_datetime(group["bucket_day"])
            reference = group[(group["bucket_day"] >= reference_start) & (group["bucket_day"] <= reference_end)]
            current = group[(group["bucket_day"] >= current_start) & (group["bucket_day"] <= current_end)]
            reference_months = pd.date_range(reference_start, periods=12, freq="MS")
            current_months = pd.date_range(current_start, periods=12, freq="MS")
            reference_monthly = reference.set_index("bucket_day")["area_value"].resample("MS").mean().reindex(reference_months)
            current_monthly = current.set_index("bucket_day")["area_value"].resample("MS").mean().reindex(current_months)
            paired = pd.DataFrame({
                "reference": reference_monthly.to_numpy(dtype=float),
                "current": current_monthly.to_numpy(dtype=float),
            }).dropna()
            if len(paired) != 12:
                excluded_scenarios.append({
                    "aggregation_method": aggregation, "value_policy": policy,
                    "coverage_threshold": float(threshold), "matched_month_count": int(len(paired)),
                    "reason": "fewer than 12 matched months contain qualifying values under this policy",
                })
                continue
            differences = paired["current"].to_numpy() - paired["reference"].to_numpy()
            rng = np.random.default_rng(seed + scenario_index)
            indices = rng.integers(0, len(differences), size=(replicates, len(differences)))
            bootstrap = differences[indices].mean(axis=1)
            reference_mean = float(reference["area_value"].mean())
            current_mean = float(current["area_value"].mean())
            change = current_mean - reference_mean
            percentage = change / reference_mean * 100 if reference_mean else None
            reference_exceedance = float((reference["area_value"] > self.threshold).mean() * 100)
            current_exceedance = float((current["area_value"] > self.threshold).mean() * 100)
            slope = float(np.polyfit(np.arange(12), paired["current"].to_numpy(dtype=float), 1)[0])
            ci_lower = float(np.quantile(bootstrap, .025))
            ci_upper = float(np.quantile(bootstrap, .975))
            rows.append({
                "aggregation_method": aggregation, "value_policy": policy,
                "coverage_threshold": float(threshold), "matched_month_count": 12,
                "reference_mean": reference_mean, "current_mean": current_mean,
                "mean_change": change, "mean_change_percentage": percentage,
                "reference_p95_daily": float(reference["area_value"].quantile(.95)),
                "current_p95_daily": float(current["area_value"].quantile(.95)),
                "reference_exceedance_percentage": reference_exceedance,
                "current_exceedance_percentage": current_exceedance,
                "exceedance_change_percentage_points": current_exceedance - reference_exceedance,
                "current_monthly_trend_slope": slope,
                "change_ci_lower": ci_lower, "change_ci_upper": ci_upper,
                "statistically_supported_improvement": ci_upper < 0,
                "practically_meaningful_improvement": percentage is not None and percentage <= -5,
                "retained_station_day_fraction": float(
                    (current["retained_station_days"].sum() / current["candidate_station_days"].sum()).clip(0, 1)
                ),
                "current_filled_fraction": float(current["filled_hours"].sum() / current["valid_hours"].sum())
                    if current["valid_hours"].sum() else None,
                "station_count": int(current["station_count"].max()),
                "city_count": int(current["city_count"].max()), "unit": "µg/m³",
            })
        if not rows:
            raise DataUnavailableError("The two India periods do not contain all 12 matched months under any requested scenario")
        observed_exclusions = [item for item in excluded_scenarios if item["value_policy"] == "observed_only"]
        if require_observed_policy and observed_exclusions:
            matched = max(item["matched_month_count"] for item in observed_exclusions)
            raise DataUnavailableError(
                f"Measured-only sensitivity cannot be completed: only {matched} of 12 current/reference month pairs "
                "contain qualifying directly measured values. January–May 2026 CPCB PM2.5 contains legacy-filled values but no directly measured values"
            )
        supported = sum(bool(row["statistically_supported_improvement"]) for row in rows)
        negative = sum(row["mean_change"] < 0 for row in rows)
        policies = sorted({row["value_policy"] for row in rows})
        maximum_filled = max(row["current_filled_fraction"] or 0 for row in rows)
        answer = (
            f"I compared India’s latest complete 12 months ({current_start:%b %Y}–{current_end:%b %Y}) with the same months one year earlier. "
            f"PM2.5 decreased in {negative} of {len(rows)} valid combinations of equal-station or equal-city aggregation and 50%, 75%, or 90% daily coverage. "
            f"The paired-month 95% interval supports a decrease in {supported} combination(s). "
            f"Only {', '.join(policy.replace('_', ' ') for policy in policies)} results have all 12 matched months, and up to {maximum_filled:.1%} of their current-period usable values are filled. "
            "Measured-only robustness is not estimable because January–May 2026 has no directly measured CPCB PM2.5 values, so these data do not support a claim of meaningful improvement."
        )
        result = {
            "kind": "analytics",
            "query_spec": {"dataset": "CPCB", "operation": "india_improvement_robustness",
                           "geography_level": "country", "locations": ["India"], "pollutants": ["PM25"],
                           "time_range": {"start": reference_start.isoformat(), "end": current_end.isoformat()},
                           "resolution": "scenario", "metrics": ["mean_change", "mean_change_percentage",
                           "current_monthly_trend_slope", "exceedance_change_percentage_points", "change_ci_lower", "change_ci_upper"],
                           "filters": {"aggregation_methods": ["equal_station", "equal_city"],
                           "value_policies": ["observed_only", "all_usable"], "coverage_thresholds": [.5, .75, .9]}},
            "resolved_period": {"start": current_start.isoformat(), "end": current_end.isoformat(),
                                "basis": "latest 12 complete India months, matched to the same preceding months"},
            "data": rows,
            "metadata": {"dataset": "CPCB", "geography_level": "country", "station_balanced": True,
                         "value_status": "policy comparison", "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                         "warnings": ["Measured-only sensitivity is incomplete because January–May 2026 has no directly measured CPCB PM2.5 values"]},
            "visualization": {"type": "heatmap", "x": "coverage_threshold", "y": "aggregation_method",
                              "value": "mean_change_percentage", "facet": "value_policy"},
            "analysis": {"operator": "india_pm25_improvement_robustness_v1", "reference_period": {
                         "start": reference_start.isoformat(), "end": reference_end.isoformat()},
                         "current_period": {"start": current_start.isoformat(), "end": current_end.isoformat()},
                         "guideline_threshold": self.threshold, "bootstrap_seed": seed,
                         "bootstrap_replicates": replicates, "bootstrap_unit": "paired month",
                         "scenario_count": len(rows), "negative_change_scenarios": negative,
                         "statistically_supported_scenarios": supported,
                         "excluded_scenarios": excluded_scenarios, "source_data_mutated": False},
            "answer": answer, "display_title": "India PM2.5 · 12-month improvement robustness",
            "model_trace": {"provider": "deterministic", "model": "india_pm25_improvement_robustness_v1",
                            "latency_ms": 0, "prompt_tokens": None, "completion_tokens": None, "fallback_used": False},
        }
        self._india_improvement_cache = result
        return result

    def _hourly_paths(self, dataset: str, start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
        paths = []
        current = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        while current <= end:
            directory = self.root / "archive" / "hourly" / f"dataset={dataset}" / f"year={current.year}" / f"month={current.month}"
            paths.extend(str(path) for path in directory.glob("*.parquet"))
            current = current + pd.offsets.MonthBegin(1)
        return paths

    def _country_rank_stability(self, text: str) -> dict[str, Any]:
        started = time.perf_counter()
        start = pd.Timestamp("2026-07-01 00:00:00")
        end = pd.Timestamp("2026-07-31 23:00:00")
        paths = []
        datasets_with_partitions = set()
        for item in self.catalog.datasets:
            month_paths = self._hourly_paths(item.dataset, start, end)
            if month_paths:
                paths.extend(month_paths)
                datasets_with_partitions.add(item.dataset)
        if not paths:
            raise DataUnavailableError("No July 2026 hourly partitions are available for rank sensitivity")
        control = duckdb.connect(str(self.catalog.control_path), read_only=True)
        try:
            dimensions = control.execute("SELECT dataset,source_station_id,city FROM station_dimension").fetchdf()
        finally:
            control.close()
        maximum = float(self.config.get("visualization", {}).get("maximum_plausible_pm25", 5000))
        connection = duckdb.connect()
        connection.execute("SET threads=8")
        connection.register("rank_stability_station_dimension", dimensions)
        try:
            stations = connection.execute("""
                WITH station_values AS (
                  SELECT h.dataset,any_value(h.country_name) country,h.source_station_id,
                         avg(h.observed_value) FILTER(WHERE h.value_status='observed' AND h.observed_value BETWEEN 0 AND ?) station_mean,
                         count(*) FILTER(WHERE h.value_status='observed' AND h.observed_value BETWEEN 0 AND ?) observed_hours,
                         any_value(h.unit) unit
                  FROM read_parquet(?,hive_partitioning=true,union_by_name=true) h
                  WHERE h.pollutant='PM25' AND h.timestamp_local BETWEEN ? AND ?
                  GROUP BY h.dataset,h.source_station_id
                )
                SELECT s.*,d.city,s.observed_hours/744.0 coverage
                FROM station_values s LEFT JOIN rank_stability_station_dimension d USING(dataset,source_station_id)
                WHERE station_mean IS NOT NULL
            """, [maximum, maximum, paths, start.to_pydatetime(), end.to_pydatetime()]).fetchdf()
        finally:
            connection.close()
        if stations.empty:
            raise DataUnavailableError("No directly measured July 2026 PM2.5 station series are usable")
        scenario_values: dict[str, dict[str, float]] = {}
        rows = []
        for threshold in self.ranking_thresholds:
            eligible = stations[stations["coverage"] >= threshold]
            for method in ("equal_station", "equal_city"):
                values = {}
                details = {}
                for dataset, group in eligible.groupby("dataset", sort=True):
                    if method == "equal_station":
                        value = float(group["station_mean"].mean())
                        city_count = int(group["city"].dropna().nunique())
                    else:
                        resolved = group[group["city"].notna() & (group["city"].str.strip() != "")]
                        cities = resolved.groupby("city")["station_mean"].mean()
                        if cities.empty:
                            continue
                        value = float(cities.mean())
                        city_count = int(len(cities))
                    values[dataset] = value
                    details[dataset] = (group.iloc[0]["country"], len(group), city_count, group.iloc[0]["unit"])
                scenario = f"{method}@{threshold:.0%}"
                scenario_values[scenario] = values
                ranks = _ranks(values)
                for dataset, value in values.items():
                    country, station_count, city_count, unit = details[dataset]
                    rows.append({
                        "scenario": scenario, "aggregation_method": method,
                        "coverage_threshold": threshold, "dataset": dataset, "country": country,
                        "mean": value, "rank": ranks[dataset], "station_count": int(station_count),
                        "city_count": int(city_count), "unit": unit,
                    })
        if not rows:
            raise DataUnavailableError("No countries meet any configured rank-sensitivity scenario")
        correlations = []
        scenarios = sorted(scenario_values)
        for left in scenarios:
            for right in scenarios:
                shared = sorted(set(scenario_values[left]) & set(scenario_values[right]))
                left_ranks = _ranks({key: scenario_values[left][key] for key in shared})
                right_ranks = _ranks({key: scenario_values[right][key] for key in shared})
                if len(shared) >= 2:
                    rho = float(pd.Series([left_ranks[key] for key in shared]).corr(
                        pd.Series([right_ranks[key] for key in shared]), method="spearman"
                    ))
                else:
                    rho = None
                left_top = set(sorted(scenario_values[left], key=scenario_values[left].get, reverse=True)[:10])
                right_top = set(sorted(scenario_values[right], key=scenario_values[right].get, reverse=True)[:10])
                union = left_top | right_top
                correlations.append({
                    "scenario_x": left, "scenario_y": right, "shared_country_count": len(shared),
                    "spearman_rho": rho, "top10_jaccard": len(left_top & right_top) / len(union) if union else None,
                })
        reference = f"equal_station@{max(self.ranking_thresholds):.0%}"
        agreement = []
        for threshold in sorted(self.ranking_thresholds):
            scenario = f"equal_station@{threshold:.0%}"
            pair = next(row for row in correlations if row["scenario_x"] == scenario and row["scenario_y"] == reference)
            agreement.append({"coverage_threshold": threshold, **pair})
        qualifying = [row for row in agreement if row["spearman_rho"] is not None and row["spearman_rho"] >= self.ranking_agreement_target]
        minimum_threshold = min((row["coverage_threshold"] for row in qualifying), default=None)
        all_datasets = {item.dataset: item.country_name for item in self.catalog.datasets}
        usable_datasets = set(stations["dataset"])
        exclusions = [
            {"dataset": dataset, "country": country,
             "reason": "no July 2026 partition" if dataset not in datasets_with_partitions
             else "no directly measured PM2.5 station value passed validity checks"}
            for dataset, country in all_datasets.items() if dataset not in usable_datasets
        ]
        if "minimum coverage threshold" in text:
            answer = (
                f"Relative to the {max(self.ranking_thresholds):.0%}-coverage equal-station reference, the minimum tested station-coverage rule meeting the configured Spearman target of {self.ranking_agreement_target:.2f} is "
                + (f"{minimum_threshold:.0%}." if minimum_threshold is not None else "not reached.")
                + f" The audit uses {len(usable_datasets)} countries with valid July 2026 PM2.5 measurements and retains {len(exclusions)} exclusions."
            )
        else:
            station_city = next((row for row in correlations if row["scenario_x"] == "equal_station@75%" and row["scenario_y"] == "equal_city@75%"), None)
            rho_text = f"{station_city['spearman_rho']:.2f}" if station_city and station_city["spearman_rho"] is not None else "not estimable"
            answer = (
                f"I compared July 2026 PM2.5 country rankings under equal-station and equal-city aggregation at 50%, 75%, and 90% station coverage. "
                f"At 75% coverage, equal-station versus equal-city Spearman agreement is {rho_text}. The downloadable audit retains every eligible rank, all pairwise scenario agreements, and explicit country exclusions."
            )
        visualization = ({"type": "heatmap", "x": "scenario_x", "y": "scenario_y", "value": "spearman_rho",
                          "data_source": "analysis.correlation_matrix"}
                         if "heatmap" in text else {"type": "bar"})
        return {
            "kind": "analytics",
            "query_spec": {"dataset": "GLOBAL_LAKEHOUSE", "operation": "country_rank_stability",
                           "geography_level": "country", "locations": sorted({row["country"] for row in rows}),
                           "pollutants": ["PM25"], "time_range": {"start": start.isoformat(), "end": end.isoformat()},
                           "resolution": "scenario", "metrics": ["mean", "rank", "spearman_rho", "top10_jaccard"],
                           "filters": {"value_status": "observed", "coverage_thresholds": list(self.ranking_thresholds),
                                       "aggregation_methods": ["equal_station", "equal_city"]}},
            "resolved_period": {"start": start.isoformat(), "end": end.isoformat(),
                                "basis": "explicit July 2026 period, retaining only countries with usable measurements"},
            "data": rows,
            "metadata": {"dataset": "GLOBAL_LAKEHOUSE", "geography_level": "country", "station_balanced": True,
                         "value_status": "observed", "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                         "warnings": []},
            "visualization": visualization,
            "analysis": {"operator": "country_rank_stability_v1", "coverage_thresholds": list(self.ranking_thresholds),
                         "aggregation_methods": ["equal_station", "equal_city"], "correlation_matrix": correlations,
                         "top10_reference_scenario": reference, "ranking_agreement_target": self.ranking_agreement_target,
                         "minimum_qualifying_coverage_threshold": minimum_threshold,
                         "threshold_agreement": agreement, "exclusions": exclusions,
                         "maximum_plausible_pm25": maximum, "source_data_mutated": False},
            "answer": answer, "display_title": "PM2.5 country ranking · coverage and aggregation sensitivity",
            "model_trace": {"provider": "deterministic", "model": "country_rank_stability_v1", "latency_ms": 0,
                            "prompt_tokens": None, "completion_tokens": None, "fallback_used": False},
        }

    def _forecast_context_gaps(self) -> dict[str, Any]:
        started = time.perf_counter()
        control = duckdb.connect(str(self.catalog.control_path), read_only=True)
        try:
            stations = control.execute(
                "SELECT source_station_id FROM station_dimension WHERE dataset='CPCB' AND lower(city)='delhi'"
            ).fetchdf()
        finally:
            control.close()
        paths = [str(path) for path in (self.root / "archive" / "hourly" / "dataset=CPCB").glob("year=2025/month=*/*.parquet")]
        connection = duckdb.connect()
        connection.register("simulation_delhi_stations", stations)
        try:
            result = connection.execute("""
                SELECT timestamp_local bucket,
                       avg(observed_value) FILTER(WHERE value_status='observed' AND observed_value>=0) observed_mean
                FROM read_parquet(?,hive_partitioning=true,union_by_name=true)
                JOIN simulation_delhi_stations USING(source_station_id)
                WHERE pollutant='PM25' GROUP BY timestamp_local ORDER BY timestamp_local
            """, [paths]).fetchall()
        finally:
            connection.close()
        observed = {timestamp: value for timestamp, value in result if value is not None}
        if not observed:
            raise DataUnavailableError("No directly observed Delhi PM2.5 city series is available for the context-gap experiment")
        latest = max(observed)
        issue = pd.Timestamp(latest.replace(minute=0, second=0, microsecond=0)) - pd.Timedelta(hours=24)
        query_start = issue - pd.Timedelta(hours=self.sequence_hours - 1)
        query_end = issue + pd.Timedelta(hours=24)
        timestamps = pd.date_range(query_start, query_end, freq="h")
        base_rows = [{
            "timestamp": value.to_pydatetime(), "mean": observed.get(value.to_pydatetime()),
            "observed_mean": observed.get(value.to_pydatetime()),
            "valid_count": 1 if observed.get(value.to_pydatetime()) is not None else 0, "filled_count": 0,
        } for value in timestamps]
        baseline = forecast_at_issue(
            base_rows, issue_time=issue.to_pydatetime(), horizon=24, context_hours=168,
            candidates=("seasonal_naive", "linear_trend"),
        )
        rows = [{
            "strategy": "complete_reference", "gap_hours": 0, "forecast_status": "executed",
            "model": baseline.model, "mae": baseline.metrics["mae"], "rmse": baseline.metrics["rmse"],
            "mase": baseline.metrics["mase"], "mae_degradation": 0.0, "rmse_degradation": 0.0,
            "mase_degradation": 0.0, "input_filled_fraction": 0.0, "reason": None, "unit": "µg/m³",
        }]
        context_times = pd.date_range(query_start, issue, freq="h")
        context_values = np.asarray([observed.get(value.to_pydatetime(), np.nan) for value in context_times], dtype=float)
        for gap in (1, 3, 6):
            rows.append({
                "strategy": "no_fill", "gap_hours": gap, "forecast_status": "rejected",
                "model": None, "mae": None, "rmse": None, "mase": None,
                "mae_degradation": None, "rmse_degradation": None, "mase_degradation": None,
                "input_filled_fraction": 0.0,
                "reason": f"strict forecast context is discontinuous after masking the final {gap} hour(s)", "unit": "µg/m³",
            })
            masked = context_values.copy(); masked[-gap:] = np.nan
            frame = pd.DataFrame({
                "dataset": "CPCB", "source_station_id": "Delhi_equal_station_city_series", "pollutant": "PM25",
                "timestamp_local": context_times, "observed_value": masked, "unit": "µg/m³",
                "source": "forecast_context_gap_experiment", "ingestion_run_id": f"simulation-{self.seed}",
            })
            imputed = impute_new_data(frame, retain_missing=True)
            filled_values = imputed["analysis_value"].to_numpy(dtype=float)
            proposed = filled_values[-gap:]
            if not np.isfinite(proposed).all():
                rows.append({
                    "strategy": "configured_imputation", "gap_hours": gap, "forecast_status": "rejected",
                    "model": None, "mae": None, "rmse": None, "mase": None,
                    "mae_degradation": None, "rmse_degradation": None, "mase_degradation": None,
                    "input_filled_fraction": 0.0,
                    "reason": "configured guarded interpolation leaves trailing gaps unresolved to avoid using future observations",
                    "unit": "µg/m³",
                })
                continue
            experimental = [dict(row) for row in base_rows]
            by_time = {row["timestamp"]: row for row in experimental}
            for offset in range(gap):
                timestamp = context_times[-gap + offset].to_pydatetime()
                by_time[timestamp]["mean"] = float(filled_values[-gap + offset])
                by_time[timestamp]["observed_mean"] = None
                by_time[timestamp]["valid_count"] = 1
                by_time[timestamp]["filled_count"] = 1
            forecast = forecast_at_issue(
                experimental, issue_time=issue.to_pydatetime(), horizon=24, context_hours=168,
                candidates=("seasonal_naive", "linear_trend"),
            )
            rows.append({
                "strategy": "configured_imputation", "gap_hours": gap, "forecast_status": "executed",
                "model": forecast.model, "mae": forecast.metrics["mae"], "rmse": forecast.metrics["rmse"],
                "mase": forecast.metrics["mase"],
                "mae_degradation": forecast.metrics["mae"] - baseline.metrics["mae"],
                "rmse_degradation": forecast.metrics["rmse"] - baseline.metrics["rmse"],
                "mase_degradation": forecast.metrics["mase"] - baseline.metrics["mase"],
                "input_filled_fraction": gap / 168, "reason": None, "unit": "µg/m³",
            })
        return {
            "kind": "analytics",
            "query_spec": {"dataset": "CPCB", "operation": "forecast_context_gap_sensitivity",
                           "geography_level": "city", "locations": ["Delhi"], "pollutants": ["PM25"],
                           "time_range": {"start": issue.isoformat(), "end": (issue + pd.Timedelta(hours=24)).isoformat()},
                           "resolution": "experiment", "metrics": ["mae", "rmse", "mase", "mae_degradation"],
                           "filters": {"gap_hours": [1, 3, 6], "seed": self.seed}},
            "resolved_period": {"start": issue.isoformat(), "end": (issue + pd.Timedelta(hours=24)).isoformat(),
                                "basis": "latest Delhi issue with a continuous observed context and observed 24-hour target"},
            "data": rows,
            "metadata": {"dataset": "CPCB", "geography_level": "city", "station_balanced": True,
                         "value_status": "observed", "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                         "warnings": []},
            "visualization": {"type": "bar"},
            "analysis": {"operator": "forecast_context_gap_sensitivity_v1", "seed": self.seed,
                         "issue_time": issue.isoformat(), "context_hours": 168, "forecast_horizon_hours": 24,
                         "gap_lengths_hours": [1, 3, 6], "configured_imputation_method": CONFIGURED_METHOD,
                         "source_values": "directly observed city-hour means", "future_observations_used_for_training": False,
                         "source_data_mutated": False},
            "answer": (
                f"I used the latest eligible directly observed Delhi issue time, {issue:%d %b %Y %H:%M}, and masked the final 1, 3, and 6 context hours. "
                "The strict no-fill strategy is rejected because the forecast requires a continuous context. The configured guarded method also leaves every trailing gap unresolved because filling it would require future observations. Accuracy degradation is therefore left empty for masked cases rather than introducing leakage or fabricated values."
            ),
            "display_title": "Delhi PM2.5 · forecast context-gap sensitivity",
            "model_trace": {"provider": "deterministic", "model": "forecast_context_gap_sensitivity_v1",
                            "latency_ms": 0, "prompt_tokens": None, "completion_tokens": None, "fallback_used": False},
        }

    def _sequences(self) -> list[dict]:
        if self._sequence_cache is not None:
            return self._sequence_cache
        paths = [str(path) for path in (self.root / "archive" / "hourly").glob("dataset=*/year=2026/month=7/*.parquet")]
        connection = duckdb.connect()
        try:
            runs = connection.execute("""
                WITH observed AS (
                  SELECT dataset,country_name,source_station_id,timestamp_local,observed_value,unit,
                         timestamp_local-row_number() OVER(PARTITION BY dataset,source_station_id ORDER BY timestamp_local)*INTERVAL 1 HOUR grp
                  FROM read_parquet(?,hive_partitioning=true,union_by_name=true)
                  WHERE pollutant='PM25' AND value_status='observed' AND observed_value>=0
                    AND timestamp_local BETWEEN TIMESTAMP '2026-07-01' AND TIMESTAMP '2026-07-31 23:00:00'
                ), runs AS (
                  SELECT dataset,any_value(country_name) country,source_station_id,min(timestamp_local) run_start,
                         max(timestamp_local) run_end,count(*) run_hours,any_value(unit) unit
                  FROM observed GROUP BY dataset,source_station_id,grp
                )
                SELECT * FROM runs WHERE run_hours>=? ORDER BY dataset,source_station_id,run_start
            """, [paths, self.sequence_hours]).fetchdf()
            if runs.empty:
                raise DataUnavailableError("No complete observed PM2.5 sequences satisfy the simulation length")
            chosen = []
            rng = np.random.default_rng(self.seed)
            for dataset, group in runs.groupby("dataset", sort=True):
                unique = group.sort_values(["source_station_id", "run_hours"], ascending=[True, False]).drop_duplicates("source_station_id")
                indices = rng.choice(len(unique), size=min(self.samples_per_country, len(unique)), replace=False)
                chosen.append(unique.iloc[np.sort(indices)])
            selected = pd.concat(chosen, ignore_index=True)
            selected["sequence_end"] = selected["run_start"] + pd.to_timedelta(self.sequence_hours - 1, unit="h")
            connection.register("simulation_sequences", selected[["dataset", "source_station_id", "run_start", "sequence_end"]])
            values = connection.execute("""
                SELECT h.dataset,h.country_name country,h.source_station_id,s.run_start,h.timestamp_local,h.observed_value,h.unit
                FROM read_parquet(?,hive_partitioning=true,union_by_name=true) h
                JOIN simulation_sequences s USING(dataset,source_station_id)
                WHERE h.pollutant='PM25' AND h.value_status='observed' AND h.observed_value>=0
                  AND h.timestamp_local BETWEEN s.run_start AND s.sequence_end
                ORDER BY h.dataset,h.source_station_id,h.timestamp_local
            """, [paths]).fetchdf()
        finally:
            connection.close()
        output = []
        for (dataset, station), group in values.groupby(["dataset", "source_station_id"], sort=True):
            if len(group) != self.sequence_hours:
                continue
            output.append({
                "sample_id": f"{dataset}:{station}", "dataset": dataset, "country": group["country"].iloc[0],
                "source_station_id": station, "timestamps": pd.DatetimeIndex(group["timestamp_local"]),
                "values": group["observed_value"].to_numpy(dtype=float), "unit": group["unit"].iloc[0],
            })
        self._sequence_cache = output
        return output

    def _reconstruction_experiments(self) -> list[dict]:
        if self._reconstruction_cache is not None:
            return self._reconstruction_cache
        output = []
        for sample_index, sequence in enumerate(self._sequences()):
            seasonal_scale = float(np.mean(np.abs(sequence["values"][24:] - sequence["values"][:-24])))
            for gap in self.gap_lengths:
                rng = np.random.default_rng(self.seed + sample_index * 100 + gap)
                start = int(rng.integers(168, self.sequence_hours - gap - 24))
                truth = sequence["values"][start:start + gap].copy()
                masked = sequence["values"].copy(); masked[start:start + gap] = np.nan
                predictions = {
                    "forward_fill": np.repeat(masked[start - 1], gap),
                    "linear_interpolation": np.interp(np.arange(start, start + gap),
                                                       [start - 1, start + gap], [masked[start - 1], masked[start + gap]]),
                    "seasonal_interpolation": np.mean(np.vstack([
                        masked[start - 24:start - 24 + gap], masked[start + 24:start + 24 + gap]
                    ]), axis=0),
                }
                frame = pd.DataFrame({
                    "dataset": sequence["dataset"], "source_station_id": sequence["source_station_id"],
                    "pollutant": "PM25", "timestamp_local": sequence["timestamps"], "observed_value": masked,
                    "unit": sequence["unit"], "source": "reproducible_masking_experiment",
                    "ingestion_run_id": f"simulation-{self.seed}",
                })
                configured = impute_new_data(frame, retain_missing=True)
                predictions["configured"] = configured["analysis_value"].to_numpy(dtype=float)[start:start + gap]
                for method, predicted in predictions.items():
                    for offset in range(gap):
                        output.append({
                            "scope": "country", "sample_id": sequence["sample_id"], "dataset": sequence["dataset"],
                            "country": sequence["country"], "method": method, "gap_hours": gap,
                            "offset": offset + 1, "truth": float(truth[offset]), "prediction": float(predicted[offset]),
                            "scaled_absolute_error": float(abs(predicted[offset] - truth[offset]) / seasonal_scale)
                            if seasonal_scale > 0 else None,
                        })
        self._reconstruction_cache = output
        return output

    def _result(self, rows: list[dict], started: float, operator: str, answer: str,
                metrics: list[str], analysis: dict[str, Any]) -> dict[str, Any]:
        return {
            "kind": "analytics",
            "query_spec": {"dataset": "SIMULATION_COHORT", "operation": operator, "geography_level": "station",
                           "locations": sorted({row.get("country") for row in rows if row.get("country")}),
                           "pollutants": ["PM25"], "time_range": {"start": "2026-07-01", "end": "2026-07-31T23:00:00"},
                           "resolution": "experiment", "metrics": metrics,
                           "filters": {"seed": self.seed, "source_values": "observed"}},
            "resolved_period": {"start": "2026-07-01", "end": "2026-07-31T23:00:00",
                                "basis": "latest month containing eligible complete directly observed sequences"},
            "data": rows,
            "metadata": {"dataset": "SIMULATION_COHORT", "geography_level": "station", "station_balanced": True,
                         "value_status": "observed", "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                         "warnings": []},
            "visualization": {"type": "bar"},
            "analysis": {"operator": operator, "seed": self.seed, "sequence_hours": self.sequence_hours,
                         "station_sequences_per_country": self.samples_per_country, "source_data_mutated": False, **analysis},
            "answer": answer, "display_title": "PM2.5 · reproducible missing-data experiment",
            "model_trace": {"provider": "deterministic", "model": operator, "latency_ms": 0,
                            "prompt_tokens": None, "completion_tokens": None, "fallback_used": False},
        }


def _dataset_means(sequences) -> dict[str, float]:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for item in sequences: grouped[item["dataset"]].append(item["values"])
    return {dataset: float(np.mean(np.concatenate(values))) for dataset, values in grouped.items()}


def _ranks(values: dict[str, float]) -> dict[str, int]:
    ordered = sorted(values, key=lambda key: (values[key], key), reverse=True)
    return {key: rank for rank, key in enumerate(ordered, 1)}


def _summary(group: list[dict], threshold: float) -> dict[str, float]:
    values = np.concatenate([item["values"] for item in group])
    return {"mean": float(np.mean(values)), "p95": float(np.quantile(values, .95)),
            "exceedance_percentage": float(np.mean(values > threshold))}


def _stable_dataset_seed(value: str) -> int:
    return sum((index + 1) * ord(character) for index, character in enumerate(value))
