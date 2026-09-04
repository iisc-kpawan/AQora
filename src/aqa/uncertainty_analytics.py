from __future__ import annotations

import re
import time
from datetime import datetime
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from .capabilities import CapabilityAssessment, PRIMARY_DATASETS
from .config import load_config
from .errors import DataUnavailableError
from .global_analytics import POLLUTANTS, _period
from .lakehouse_catalog import LakehouseCatalog


def is_sampling_uncertainty_question(question: str) -> bool:
    text = question.casefold()
    return "bootstrap" in text and "station" in text and any(term in text for term in ("mean", "sampling", "country-level"))


def is_aggregation_comparison_question(question: str) -> bool:
    text = question.casefold()
    return (
        "station-weighted" in text and ("city-weighted" in text or "city-balanced" in text)
    ) or ("aggregation schemes" in text and "equal-country" in text)


def is_matched_month_city_balanced_bootstrap_question(question: str) -> bool:
    text = question.casefold()
    return (
        re.search(r"matched months?", text) is not None
        and "city-balanced" in text
        and any(term in text for term in ("uncertainty interval", "confidence interval", "bootstrap"))
    )


class UncertaintyAnalytics:
    """Seeded station bootstrap, city-balanced matched-month bootstrap, and explicit aggregation-weight sensitivity."""

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.catalog = LakehouseCatalog(self.config)
        self.root = self.catalog.root

    @staticmethod
    def should_handle(question: str, assessment: CapabilityAssessment) -> bool:
        return (
            is_sampling_uncertainty_question(question)
            or is_aggregation_comparison_question(question)
            or is_matched_month_city_balanced_bootstrap_question(question)
        )

    def execute(self, question: str, assessment: CapabilityAssessment) -> dict[str, Any]:
        started = time.perf_counter()
        text = question.casefold()
        if is_matched_month_city_balanced_bootstrap_question(question):
            return self._matched_month_city_balanced_bootstrap(question, assessment, started)
        pollutant = next((name for pattern, name in POLLUTANTS if re.search(pattern, text)), "PM25")
        selected = list(self.catalog.selected(assessment.supported_datasets))
        if "seven primary" in text or "primary countries" in text or not selected:
            selected = [item for item in self.catalog.datasets if item.dataset in PRIMARY_DATASETS]
        start, end = _period(text, self.catalog, [item.dataset for item in selected])
        paths = self._paths(selected, start, end)
        if not paths:
            raise DataUnavailableError("No hourly rows overlap the requested uncertainty period")
        minimum_coverage = _coverage(text, float(self.config["analytics"].get("minimum_daily_coverage", .75)))
        frame = self._station_values(paths, pollutant, start, end, minimum_coverage)
        if frame.empty:
            raise DataUnavailableError("No directly measured station series meets the requested coverage rule")
        available = set(frame["dataset"])
        exclusions = [
            {"dataset": item.dataset, "country": item.country_name,
             "reason": "no directly measured station series meets the requested period and coverage rule"}
            for item in selected if item.dataset not in available
        ]
        if is_sampling_uncertainty_question(question):
            return self._bootstrap_result(
                frame, pollutant, start, end, minimum_coverage, exclusions, started,
            )
        return self._aggregation_result(
            frame, pollutant, start, end, minimum_coverage, exclusions, started,
        )

    def _matched_month_city_balanced_bootstrap(self, question: str, assessment: CapabilityAssessment,
                                                started: float) -> dict[str, Any]:
        text = question.casefold()
        pollutant = next((name for pattern, name in POLLUTANTS if re.search(pattern, text)), "PM25")
        selected = list(self.catalog.selected(assessment.supported_datasets))
        if len(selected) < 2:
            raise DataUnavailableError(
                "A matched-month city-balanced comparison needs at least two explicitly named countries"
            )
        start, end = _period(text, self.catalog, [item.dataset for item in selected])
        paths = self._paths(selected, start, end)
        if not paths:
            raise DataUnavailableError("No hourly rows overlap the requested uncertainty period")
        coverage_threshold = float(self.config["analytics"].get("cohort_monthly_coverage_threshold", .8))
        frame = self._city_month_values(paths, pollutant, start, end)
        if frame.empty:
            raise DataUnavailableError(
                "No city-month observed-only series is available for the requested countries and period"
            )
        qualifying = frame[frame["coverage"] >= coverage_threshold].copy()
        qualifying["month_period"] = pd.to_datetime(qualifying["month_start"]).dt.to_period("M")
        datasets = [item.dataset for item in selected]
        per_dataset_months = {
            dataset: set(group["month_period"])
            for dataset, group in qualifying.groupby("dataset")
        }
        missing = [dataset for dataset in datasets if not per_dataset_months.get(dataset)]
        if missing:
            names = [item.country_name for item in selected if item.dataset in missing]
            raise DataUnavailableError(
                f"No city meets the {coverage_threshold:.0%} monthly coverage criterion for: " + ", ".join(names)
            )
        matched_months = set.intersection(*(per_dataset_months[dataset] for dataset in datasets))
        if not matched_months:
            raise DataUnavailableError(
                f"No calendar month meets the {coverage_threshold:.0%} monthly coverage criterion for every named country"
            )
        qualifying = qualifying[qualifying["month_period"].isin(matched_months)]

        seed = int(self.config["analytics"].get("bootstrap_seed", 20260901))
        replicates = int(self.config["analytics"].get("bootstrap_replicates", 1000))
        rng = np.random.default_rng(seed)
        rows = []
        for dataset in datasets:
            group = qualifying[qualifying["dataset"] == dataset]
            if group.empty:
                continue
            city_values = group.groupby("city_id")["city_mean"].mean().to_numpy(dtype=float)
            city_count = len(city_values)
            indices = rng.integers(0, city_count, size=(replicates, city_count))
            samples = city_values[indices].mean(axis=1)
            rows.append({
                "dataset": dataset, "country": group["country"].iloc[0], "pollutant": pollutant,
                "mean": float(city_values.mean()), "bootstrap_standard_error": float(samples.std(ddof=1)),
                "ci_lower": float(np.quantile(samples, .025)), "ci_upper": float(np.quantile(samples, .975)),
                "confidence_level": .95, "city_count": int(city_count),
                "matched_month_count": len(matched_months), "unit": group["unit"].iloc[0],
            })
        if not rows:
            raise DataUnavailableError("No country has a qualifying city under the requested coverage criterion")
        rows.sort(key=lambda row: row["mean"], reverse=True)
        for rank, row in enumerate(rows, 1):
            row["rank"] = rank
        months_label = ", ".join(str(month) for month in sorted(matched_months))
        answer = (
            f"I restricted the comparison to the {len(matched_months)} calendar month(s) where every named country has at "
            f"least one city meeting the {coverage_threshold:.0%} monthly coverage criterion ({months_label}), using "
            "observed-only values. For each country I averaged its qualifying cities equally (city-balanced), then "
            f"estimated uncertainty by resampling cities with replacement across {replicates:,} deterministic bootstrap "
            "replicates."
        )
        return self._result(
            rows=rows, operation="matched_month_city_balanced_bootstrap",
            operator="matched_month_city_balanced_bootstrap_v1", answer=answer, pollutant=pollutant,
            start=start, end=end, coverage=coverage_threshold, exclusions=[], started=started, visualization="bar",
            analysis={"bootstrap_seed": seed, "bootstrap_replicates": replicates, "confidence_level": .95,
                      "resampling_unit": "city", "matched_months": [str(month) for month in sorted(matched_months)],
                      "value_status": "observed_only"},
        )

    def _city_month_values(self, paths: list[str], pollutant: str, start: datetime, end: datetime):
        control = duckdb.connect(str(self.catalog.control_path), read_only=True)
        try:
            dimensions = control.execute(
                "SELECT dataset,source_station_id,city FROM station_dimension"
            ).fetchdf()
        finally:
            control.close()
        connection = duckdb.connect()
        connection.execute("SET threads=8")
        connection.register("uncertainty_station_dimension", dimensions)
        try:
            return connection.execute("""
                WITH station_month AS (
                  SELECT dataset,any_value(country_name) country,source_station_id,
                         date_trunc('month',timestamp_local) month_start,
                         avg(observed_value) FILTER(WHERE value_status='observed' AND observed_value>=0) station_mean,
                         count(*) expected_hour_count,
                         count(*) FILTER(WHERE value_status='observed' AND observed_value>=0) observed_hour_count,
                         any_value(unit) unit
                  FROM read_parquet(?,hive_partitioning=true,union_by_name=true)
                  WHERE pollutant=? AND timestamp_local BETWEEN ? AND ?
                  GROUP BY dataset,source_station_id,month_start
                ), enriched AS (
                  SELECT s.*,d.city,lower(trim(d.city)) city_id
                  FROM station_month s LEFT JOIN uncertainty_station_dimension d USING(dataset,source_station_id)
                  WHERE d.city IS NOT NULL AND trim(d.city)<>''
                )
                SELECT dataset,any_value(country) country,city_id,any_value(city) city_name,month_start,
                       avg(station_mean) city_mean,
                       sum(observed_hour_count)::DOUBLE/nullif(sum(expected_hour_count),0) coverage,
                       any_value(unit) unit
                FROM enriched WHERE station_mean IS NOT NULL
                GROUP BY dataset,city_id,month_start
                ORDER BY dataset,city_name,month_start
            """, [paths, pollutant, start, end]).fetchdf()
        finally:
            connection.close()

    def _station_values(self, paths: list[str], pollutant: str, start: datetime,
                        end: datetime, minimum_coverage: float):
        control = duckdb.connect(str(self.catalog.control_path), read_only=True)
        try:
            dimensions = control.execute(
                "SELECT dataset,source_station_id,city FROM station_dimension"
            ).fetchdf()
        finally:
            control.close()
        connection = duckdb.connect()
        connection.execute("SET threads=8")
        connection.register("uncertainty_station_dimension", dimensions)
        expected = max(1, int((end - start).total_seconds() // 3600) + 1)
        try:
            return connection.execute("""
                WITH stations AS (
                  SELECT dataset,any_value(country_name) country,source_station_id,
                         avg(observed_value) FILTER(WHERE value_status='observed' AND observed_value>=0) station_mean,
                         count(*) FILTER(WHERE value_status='observed' AND observed_value>=0) observed_hours,
                         any_value(unit) unit
                  FROM read_parquet(?,hive_partitioning=true,union_by_name=true)
                  WHERE pollutant=? AND timestamp_local BETWEEN ? AND ?
                  GROUP BY dataset,source_station_id
                )
                SELECT s.*,d.city,s.observed_hours::DOUBLE/? coverage
                FROM stations s LEFT JOIN uncertainty_station_dimension d USING(dataset,source_station_id)
                WHERE station_mean IS NOT NULL AND s.observed_hours::DOUBLE/?>=?
                ORDER BY dataset,source_station_id
            """, [paths, pollutant, start, end, expected, expected, minimum_coverage]).fetchdf()
        finally:
            connection.close()

    def _bootstrap_result(self, frame, pollutant: str, start: datetime, end: datetime,
                          coverage: float, exclusions: list[dict], started: float) -> dict[str, Any]:
        seed = int(self.config["analytics"].get("bootstrap_seed", 20260901))
        replicates = int(self.config["analytics"].get("bootstrap_replicates", 1000))
        rng = np.random.default_rng(seed)
        rows = []
        for dataset, group in frame.groupby("dataset", sort=True):
            values = group["station_mean"].to_numpy(dtype=float)
            indices = rng.integers(0, len(values), size=(replicates, len(values)))
            samples = values[indices].mean(axis=1)
            rows.append({
                "dataset": dataset, "country": group["country"].iloc[0], "pollutant": pollutant,
                "mean": float(values.mean()), "bootstrap_standard_error": float(samples.std(ddof=1)),
                "ci_lower": float(np.quantile(samples, .025)), "ci_upper": float(np.quantile(samples, .975)),
                "confidence_level": .95, "station_count": int(len(values)),
                "mean_station_coverage": float(group["coverage"].mean()), "unit": group["unit"].iloc[0],
            })
        rows.sort(key=lambda row: row["mean"], reverse=True)
        for rank, row in enumerate(rows, 1): row["rank"] = rank
        answer = (
            f"I estimated station-sampling uncertainty for {len(rows)} {pollutant} country result(s) using "
            f"{replicates:,} deterministic bootstrap resamples per country and a fixed seed. "
            f"The table reports the equal-station mean, bootstrap standard error, and 95% interval."
        )
        if exclusions:
            answer += " Excluded: " + "; ".join(f"{item['country']} ({item['reason']})" for item in exclusions) + "."
        return self._result(
            rows=rows, operation="station_sampling_bootstrap", operator="station_bootstrap_v1",
            answer=answer, pollutant=pollutant, start=start, end=end, coverage=coverage,
            exclusions=exclusions, started=started, visualization="bar",
            analysis={"bootstrap_seed": seed, "bootstrap_replicates": replicates,
                      "confidence_level": .95, "resampling_unit": "station"},
        )

    def _aggregation_result(self, frame, pollutant: str, start: datetime, end: datetime,
                            coverage: float, exclusions: list[dict], started: float) -> dict[str, Any]:
        rows = []
        country_values = []
        all_station_values = []
        all_city_values = []
        for dataset, group in frame.groupby("dataset", sort=True):
            station_mean = float(group["station_mean"].mean())
            resolved = group[group["city"].notna() & (group["city"].str.strip() != "")]
            city_means = resolved.groupby("city")["station_mean"].mean()
            city_mean = float(city_means.mean()) if len(city_means) else None
            country = group["country"].iloc[0]
            country_values.append(station_mean)
            all_station_values.extend(group["station_mean"].astype(float).tolist())
            all_city_values.extend(city_means.astype(float).tolist())
            rows.extend([
                {"dataset": dataset, "country": country, "scope": "country", "aggregation_method": "equal_station",
                 "mean": station_mean, "difference_from_equal_station": 0.0,
                 "station_count": int(len(group)), "city_count": int(len(city_means)), "unit": group["unit"].iloc[0]},
                {"dataset": dataset, "country": country, "scope": "country", "aggregation_method": "equal_city",
                 "mean": city_mean, "difference_from_equal_station": city_mean - station_mean if city_mean is not None else None,
                 "station_count": int(len(group)), "city_count": int(len(city_means)), "unit": group["unit"].iloc[0]},
            ])
        unit = frame["unit"].iloc[0]
        global_values = {
            "pooled_equal_station": float(np.mean(all_station_values)),
            "pooled_equal_city": float(np.mean(all_city_values)) if all_city_values else None,
            "equal_country": float(np.mean(country_values)),
        }
        reference = global_values["equal_country"]
        for method, value in global_values.items():
            rows.append({
                "dataset": "GLOBAL", "country": "Eligible countries", "scope": "global",
                "aggregation_method": method, "mean": value,
                "difference_from_equal_country": value - reference if value is not None else None,
                "station_count": len(all_station_values), "city_count": len(all_city_values), "unit": unit,
            })
        reversals = _country_rank_reversals(rows)
        answer = (
            f"I compared equal-station and equal-city country estimates for {len(country_values)} eligible countries, "
            "then calculated pooled-station, pooled-city, and equal-country global means. "
            f"Country rank-order reversals: {('; '.join(reversals)) if reversals else 'none'}."
        )
        if exclusions:
            answer += " Excluded: " + "; ".join(f"{item['country']} ({item['reason']})" for item in exclusions) + "."
        return self._result(
            rows=rows, operation="aggregation_sensitivity", operator="aggregation_weight_sensitivity_v1",
            answer=answer, pollutant=pollutant, start=start, end=end, coverage=coverage,
            exclusions=exclusions, started=started, visualization="bar",
            analysis={"aggregation_methods": list(global_values), "country_rank_reversals": reversals,
                      "city_metadata_required_for_equal_city": True},
        )

    def _result(self, *, rows: list[dict], operation: str, operator: str, answer: str,
                pollutant: str, start: datetime, end: datetime, coverage: float,
                exclusions: list[dict], started: float, visualization: str,
                analysis: dict) -> dict[str, Any]:
        return {
            "kind": "analytics",
            "query_spec": {"dataset": "GLOBAL_LAKEHOUSE", "operation": operation,
                           "geography_level": "country", "locations": sorted({row["country"] for row in rows}),
                           "pollutants": [pollutant], "time_range": {"start": start.isoformat(), "end": end.isoformat()},
                           "resolution": "period", "metrics": ["mean"],
                           "filters": {"value_status": "observed", "minimum_station_coverage": coverage}},
            "resolved_period": {"start": start.isoformat(), "end": end.isoformat(), "basis": "explicit requested period"},
            "data": rows,
            "metadata": {"dataset": "GLOBAL_LAKEHOUSE", "geography_level": "country",
                         "station_count": max((row.get("station_count", 0) for row in rows), default=0),
                         "value_status": "observed", "station_balanced": True,
                         "latency_ms": round((time.perf_counter() - started) * 1000, 2), "warnings": []},
            "visualization": {"type": visualization},
            "analysis": {"operator": operator, "minimum_station_coverage": coverage,
                         "exclusions": exclusions, **analysis},
            "answer": answer, "display_title": f"{pollutant} · uncertainty and weighting sensitivity",
            "model_trace": {"provider": "deterministic", "model": operator, "latency_ms": 0,
                            "prompt_tokens": None, "completion_tokens": None, "fallback_used": False},
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


def _coverage(text: str, default: float) -> float:
    match = re.search(r"(?:at least|minimum|requiring)\s+(\d+(?:\.\d+)?)\s*%\s*coverage", text)
    return float(match.group(1)) / 100 if match else default


def _country_rank_reversals(rows: list[dict]) -> list[str]:
    stations = {row["dataset"]: row for row in rows if row["scope"] == "country" and row["aggregation_method"] == "equal_station"}
    cities = {row["dataset"]: row for row in rows if row["scope"] == "country" and row["aggregation_method"] == "equal_city" and row["mean"] is not None}
    datasets = sorted(stations.keys() & cities.keys())
    output = []
    for index, left in enumerate(datasets):
        for right in datasets[index + 1:]:
            if (stations[left]["mean"] - stations[right]["mean"]) * (cities[left]["mean"] - cities[right]["mean"]) < 0:
                output.append(f"{stations[left]['country']} versus {stations[right]['country']}")
    return output
