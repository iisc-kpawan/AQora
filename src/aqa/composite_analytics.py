from __future__ import annotations

import calendar
import math
import re
import time
from collections import defaultdict
from datetime import datetime
from typing import Any

import duckdb

from .aqi import DEFAULT_GUIDELINE, get_guideline
from .capabilities import CapabilityAssessment, is_flexible_country_cohort
from .config import load_config
from .errors import DataUnavailableError
from .global_analytics import POLLUTANTS, _period, _top_n
from .lakehouse_catalog import LakehouseCatalog


class CompositeAnalytics:
    """Deterministic multi-objective scores and Pareto comparisons."""

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.catalog = LakehouseCatalog(self.config)
        self.root = self.catalog.root

    @staticmethod
    def should_handle(question: str, assessment: CapabilityAssessment) -> bool:
        text = question.casefold()
        return any(term in text for term in (
            "composite", "multi-criteria", "analytics-readiness", "analytics readiness", "pareto",
        ))

    def execute(self, question: str, assessment: CapabilityAssessment) -> dict[str, Any]:
        text = question.casefold()
        if "pareto" in text:
            return self._pareto(question, assessment)
        if "readiness" in text:
            return self._readiness(question, assessment)
        if "composite" in text or "multi-criteria" in text:
            return self._pollution_score(question, assessment)
        raise DataUnavailableError("The requested multi-objective method is not recognized")

    def _pollution_score(self, question: str, assessment: CapabilityAssessment) -> dict[str, Any]:
        started = time.perf_counter()
        text = question.casefold()
        pollutants = [name for pattern, name in POLLUTANTS if re.search(pattern, text)]
        if len(pollutants) > 1:
            raise DataUnavailableError("Composite pollution scoring currently requires one pollutant")
        pollutant = pollutants[0] if pollutants else "PM25"
        start, end = _period(text, self.catalog, assessment.supported_datasets)
        if start.year != end.year or start.month != end.month:
            raise DataUnavailableError("Composite pollution scoring currently supports one complete calendar month")
        selected = list(self.catalog.selected(assessment.supported_datasets))
        geography = "city" if re.search(r"\bcit(?:y|ies)\b", text) else "country"
        flexible = is_flexible_country_cohort(question, assessment.intent) or geography == "city"
        cohort = self.catalog.resolve_cohort(
            [item.dataset for item in selected], pollutant, start, end,
            value_status="any", require_all_months=True,
        )
        exclusions = [item for item in cohort if not item.included]
        if exclusions and not flexible:
            detail = "; ".join(f"{item.country_name}: {item.reason}" for item in exclusions)
            raise DataUnavailableError("The requested scoring cohort is incomplete. " + detail)
        included = {item.dataset for item in cohort if item.included}
        selected = [item for item in selected if item.dataset in included]
        paths = [
            str(path) for item in selected
            for path in (self.root / "query" / "quality_monthly" / f"dataset={item.dataset}").glob("year=*/month=*/*.parquet")
        ]
        daily_paths = [
            str(path) for item in selected
            for path in (self.root / "query" / "area_daily" / f"dataset={item.dataset}").glob("year=*/month=*/*.parquet")
        ]
        if not paths or not daily_paths:
            raise DataUnavailableError("The quality and daily layers required for composite scoring are unavailable")
        minimum_coverage = float(self.config.get("analytics", {}).get("minimum_daily_coverage", .75))
        guideline = get_guideline(DEFAULT_GUIDELINE)
        threshold = guideline.thresholds.get(pollutant)
        if threshold is None:
            raise DataUnavailableError(f"No configured daily guideline is available for {pollutant}")
        connection = duckdb.connect()
        try:
            quality = _fetch_dicts(connection, """
                SELECT dataset,country_name country,area_id,area_name,mean,p95,coverage,
                       observed_fraction,imputed_fraction,active_station_count,unit,aggregation_method
                FROM read_parquet(?,hive_partitioning=true,union_by_name=true)
                WHERE pollutant=? AND geography_level=? AND month_start=date_trunc('month', ?)
                  AND valid_day_count>0
                ORDER BY dataset,area_name
            """, [paths, pollutant, geography, start])
            burden = _fetch_dicts(connection, """
                SELECT dataset,area_id,count(*) valid_day_count,
                       count(*) FILTER(WHERE mean>?) exceedance_count,
                       count(*) FILTER(WHERE mean>?)/count(*)::DOUBLE exceedance_percentage
                FROM read_parquet(?,hive_partitioning=true,union_by_name=true)
                WHERE pollutant=? AND geography_level=? AND bucket_day BETWEEN ? AND ? AND coverage>=?
                GROUP BY dataset,area_id
            """, [threshold, threshold, daily_paths, pollutant, geography, start, end, minimum_coverage])
        finally:
            connection.close()
        burden_by_key = {(row["dataset"], row["area_id"]): row for row in burden}
        rows = []
        for row in quality:
            extra = burden_by_key.get((row["dataset"], row["area_id"]))
            if extra and row["coverage"] is not None and row["coverage"] >= minimum_coverage:
                rows.append({**row, **extra})
        if not rows:
            raise DataUnavailableError(f"No {geography} satisfies the {minimum_coverage:.0%} coverage eligibility rule")
        include_data_risk = geography == "city" or "completeness" in text
        components = ["mean", "p95", "exceedance_percentage"]
        if include_data_risk:
            for row in rows:
                row["data_gap_fraction"] = 1.0 - row["coverage"]
            components.append("data_gap_fraction")
        weights = _weights(self.config, "city_burden" if include_data_risk else "severity", components)
        _score(rows, components, weights)
        eligible_count = len(rows)
        rows.sort(key=lambda row: (row["composite_score"], row["mean"]), reverse=True)
        rows = rows[:_top_n(text)]
        for index, row in enumerate(rows, 1):
            row["rank"] = index
            row["threshold"] = threshold
        formula = " + ".join(f"{weights[name]:.2f}×normalized_{name}" for name in components)
        answer = (
            f"{rows[0]['area_name']} ranks first with a composite score of {rows[0]['composite_score']:.3f}. "
            f"The reproducible formula is {formula}; each component is min–max normalized across the eligible cohort. "
            f"Eligibility required at least {minimum_coverage:.0%} coverage."
        )
        warnings = []
        if any((row.get("imputed_fraction") or 0) > .5 for row in rows):
            warnings.append("At least one ranked result is based mostly on policy-approved filled values.")
            answer += " At least one result is mostly filled rather than directly measured; use the provenance columns when interpreting the ranking."
        exclusion_rows = [{"country": item.country_name, "reason": item.reason, "reason_code": item.reason_code} for item in exclusions]
        return _result(
            rows=rows, answer=answer, started=started, operator="composite_pollution_v1",
            title=f"{pollutant} · composite burden", visualization="bar",
            query_spec={
                "dataset": "GLOBAL_COMPOSITE", "operation": "composite_score", "geography_level": geography,
                "locations": [item.country_name for item in selected], "pollutants": [pollutant],
                "time_range": {"start": start.isoformat(), "end": end.isoformat()}, "resolution": "monthly",
                "metrics": ["composite_score"] + components,
                "filters": {"minimum_coverage": minimum_coverage, "threshold": threshold},
            },
            analysis={
                "normalization": "cohort_min_max", "constant_component_value": 0.0,
                "weights": weights, "formula": formula, "threshold_source": guideline.id,
                "eligible_count": eligible_count, "returned_count": len(rows), "exclusions": exclusion_rows,
            },
            metadata={"geography_level": geography, "warnings": warnings},
            period=(start, end, "explicit month or latest loaded month"),
        )

    def _readiness(self, question: str, assessment: CapabilityAssessment) -> dict[str, Any]:
        started = time.perf_counter()
        text = question.casefold()
        selected = list(self.catalog.selected(assessment.supported_datasets))
        ids = [item.dataset for item in selected]
        connection = duckdb.connect(str(self.catalog.control_path), read_only=True)
        try:
            rows = _fetch_dicts(connection, """
                WITH availability AS (
                  SELECT dataset,any_value(country_name) country,
                         count(DISTINCT pollutant) FILTER(WHERE usable_count>0) pollutant_breadth,
                         count(DISTINCT month_start) FILTER(WHERE usable_count>0) usable_months,
                         date_diff('month',min(month_start),max(month_start))+1 possible_months,
                         sum(usable_count)::DOUBLE/nullif(sum(expected_count),0) coverage,
                         1-(sum(legacy_imputed_count+guarded_imputed_count)::DOUBLE/nullif(sum(usable_count),0)) low_imputation_fraction
                  FROM pollutant_monthly_availability WHERE dataset IN (SELECT unnest(?)) GROUP BY dataset
                ), stations AS (
                  SELECT dataset,count(DISTINCT source_station_id) station_count
                  FROM station_dimension WHERE dataset IN (SELECT unnest(?)) GROUP BY dataset
                )
                SELECT a.*,s.station_count,a.usable_months::DOUBLE/nullif(a.possible_months,0) continuity
                FROM availability a JOIN stations s USING(dataset) ORDER BY country
            """, [ids, ids])
        finally:
            connection.close()
        if len(rows) != len(selected):
            raise DataUnavailableError("Readiness inputs are incomplete for the explicitly requested cohort")
        for row in rows:
            row["log_station_count"] = math.log1p(row["station_count"])
        components = ["pollutant_breadth", "log_station_count", "continuity", "coverage", "low_imputation_fraction"]
        weights = _weights(self.config, "readiness", components)
        _score(rows, components, weights, output="readiness_score")
        rows.sort(key=lambda row: row["readiness_score"], reverse=True)
        rows = rows[:_top_n(text)]
        for index, row in enumerate(rows, 1):
            row["rank"] = index
        formula = " + ".join(f"{weights[name]:.2f}×normalized_{name}" for name in components)
        answer = (
            f"{rows[0]['country']} ranks first for analytics readiness at {rows[0]['readiness_score']:.3f}. "
            f"The score uses {formula}. Station count is log-transformed before min–max normalization so network size is not disproportionate."
        )
        bounds = [item.available_start for item in selected], [item.available_end for item in selected]
        return _result(
            rows=rows, answer=answer, started=started, operator="analytics_readiness_v1",
            title="Analytics readiness", visualization="bar",
            query_spec={
                "dataset": "GLOBAL_CONTROL_CATALOG", "operation": "readiness_score", "geography_level": "country",
                "locations": [item.country_name for item in selected], "pollutants": [],
                "time_range": {"start": min(bounds[0]).isoformat(), "end": max(bounds[1]).isoformat()},
                "resolution": "dataset_history", "metrics": ["readiness_score"] + components, "filters": {},
            },
            analysis={
                "normalization": "cohort_min_max", "constant_component_value": 0.0,
                "weights": weights, "formula": formula, "station_transform": "log1p",
            },
            metadata={"geography_level": "country", "warnings": [], "value_status": "metadata", "station_balanced": False},
            period=(min(bounds[0]), max(bounds[1]), "full loaded history for each dataset"),
        )

    def _pareto(self, question: str, assessment: CapabilityAssessment) -> dict[str, Any]:
        started = time.perf_counter()
        text = question.casefold()
        pollutants = [name for pattern, name in POLLUTANTS if re.search(pattern, text)]
        if len(pollutants) < 2:
            raise DataUnavailableError("Pareto analysis requires at least two explicitly requested pollutants")
        selected = list(self.catalog.selected(assessment.supported_datasets))
        paths = [
            str(path) for item in selected
            for path in (self.root / "query" / "quality_monthly" / f"dataset={item.dataset}").glob("year=*/month=*/*.parquet")
        ]
        if not paths:
            raise DataUnavailableError("No monthly quality rows are available for Pareto analysis")
        connection = duckdb.connect()
        try:
            if re.search(r"\b20\d{2}\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", text):
                start, _ = _period(text, self.catalog, assessment.supported_datasets)
                month = start.replace(day=1, hour=0, minute=0, second=0)
            else:
                month = connection.execute("""
                    WITH complete AS (
                      SELECT month_start,dataset,count(DISTINCT pollutant) pollutant_count
                      FROM read_parquet(?,hive_partitioning=true,union_by_name=true)
                      WHERE geography_level='country' AND pollutant IN (SELECT unnest(?)) AND valid_day_count>0
                      GROUP BY month_start,dataset
                    )
                    SELECT max(month_start) FROM (
                      SELECT month_start FROM complete WHERE pollutant_count=?
                      GROUP BY month_start HAVING count(DISTINCT dataset)=?
                    )
                """, [paths, pollutants, len(pollutants), len(selected)]).fetchone()[0]
                if month is None:
                    raise DataUnavailableError("The requested countries have no common complete month for all Pareto pollutants")
            raw = _fetch_dicts(connection, """
                SELECT dataset,any_value(country_name) country,pollutant,any_value(unit) unit,
                       avg(mean) mean,avg(coverage) coverage,avg(observed_fraction) observed_fraction,
                       avg(imputed_fraction) imputed_fraction,max(active_station_count) active_station_count
                FROM read_parquet(?,hive_partitioning=true,union_by_name=true)
                WHERE geography_level='country' AND pollutant IN (SELECT unnest(?)) AND month_start=?
                  AND valid_day_count>0 GROUP BY dataset,pollutant
            """, [paths, pollutants, month])
        finally:
            connection.close()
        grouped: dict[str, dict] = {}
        for row in raw:
            item = grouped.setdefault(row["dataset"], {"dataset": row["dataset"], "country": row["country"]})
            item[f"{row['pollutant'].lower()}_mean"] = row["mean"]
            item[f"{row['pollutant'].lower()}_unit"] = row["unit"]
            item[f"{row['pollutant'].lower()}_coverage"] = row["coverage"]
            item[f"{row['pollutant'].lower()}_observed_fraction"] = row["observed_fraction"]
            item[f"{row['pollutant'].lower()}_imputed_fraction"] = row["imputed_fraction"]
            item[f"{row['pollutant'].lower()}_station_count"] = row["active_station_count"]
        rows = [row for row in grouped.values() if all(f"{p.lower()}_mean" in row for p in pollutants)]
        if len(rows) != len(selected):
            raise DataUnavailableError("Pareto inputs are incomplete for the explicitly requested country cohort")
        for row in rows:
            row["data_completeness"] = sum(row[f"{p.lower()}_coverage"] for p in pollutants) / len(pollutants)
        for row in rows:
            dominators = [
                other["country"] for other in rows if other is not row and _dominates(other, row, pollutants)
            ]
            row["pareto_dominated"] = bool(dominators)
            row["dominated_by"] = dominators
        rows.sort(key=lambda row: (row["pareto_dominated"], sum(row[f"{p.lower()}_mean"] for p in pollutants), -row["data_completeness"]))
        dominated = [row["country"] for row in rows if row["pareto_dominated"]]
        frontier = [row["country"] for row in rows if not row["pareto_dominated"]]
        end = datetime(month.year, month.month, calendar.monthrange(month.year, month.month)[1], 23, 59, 59)
        answer = (
            f"For {month:%B %Y}, the Pareto frontier is {', '.join(frontier)}. "
            f"Pareto-dominated countries are {', '.join(dominated) if dominated else 'none'}. "
            "A country dominates another only when it has no higher concentration for every requested pollutant, no lower completeness, and is strictly better on at least one objective."
        )
        warnings = []
        if any(any((row.get(f"{p.lower()}_imputed_fraction") or 0) > .5 for p in pollutants) for row in rows):
            warnings.append("At least one country-pollutant result is based mostly on policy-approved filled values.")
            answer += " Some country-pollutant values are mostly filled rather than directly measured; their provenance fractions are included in the table."
        return _result(
            rows=rows, answer=answer, started=started, operator="pollution_completeness_pareto_v1",
            title="Pollution and completeness · Pareto comparison", visualization="table",
            query_spec={
                "dataset": "GLOBAL_QUALITY_CUBE", "operation": "pareto", "geography_level": "country",
                "locations": [item.country_name for item in selected], "pollutants": pollutants,
                "time_range": {"start": month.isoformat(), "end": end.isoformat()}, "resolution": "monthly",
                "metrics": [f"{p.lower()}_mean" for p in pollutants] + ["data_completeness", "pareto_dominated"],
                "filters": {},
            },
            analysis={
                "objectives": {**{f"{p.lower()}_mean": "minimize" for p in pollutants}, "data_completeness": "maximize"},
                "frontier": frontier, "dominated": dominated,
            },
            metadata={"geography_level": "country", "warnings": warnings},
            period=(month, end, "latest common complete month"),
        )


def _fetch_dicts(connection: duckdb.DuckDBPyConnection, sql: str, parameters: list) -> list[dict]:
    result = connection.execute(sql, parameters)
    names = [column[0] for column in result.description]
    return [dict(zip(names, row)) for row in result.fetchall()]


def _weights(config: dict, policy: str, components: list[str]) -> dict[str, float]:
    configured = config.get("analytics", {}).get("composite_weights", {}).get(policy, {})
    values = {name: float(configured.get(name, 1.0)) for name in components}
    if any(value < 0 for value in values.values()) or sum(values.values()) <= 0:
        raise DataUnavailableError("Composite weights must be non-negative and sum to more than zero")
    total = sum(values.values())
    return {name: value / total for name, value in values.items()}


def _score(rows: list[dict], components: list[str], weights: dict[str, float], output: str = "composite_score") -> None:
    for component in components:
        values = [float(row[component]) for row in rows]
        low, high = min(values), max(values)
        for row in rows:
            row[f"normalized_{component}"] = (float(row[component]) - low) / (high - low) if high > low else 0.0
    for row in rows:
        row[output] = sum(weights[name] * row[f"normalized_{name}"] for name in components)


def _dominates(candidate: dict, target: dict, pollutants: list[str]) -> bool:
    no_worse = all(candidate[f"{p.lower()}_mean"] <= target[f"{p.lower()}_mean"] for p in pollutants)
    no_worse = no_worse and candidate["data_completeness"] >= target["data_completeness"]
    strictly_better = any(candidate[f"{p.lower()}_mean"] < target[f"{p.lower()}_mean"] for p in pollutants)
    strictly_better = strictly_better or candidate["data_completeness"] > target["data_completeness"]
    return no_worse and strictly_better


def _result(*, rows: list[dict], answer: str, started: float, operator: str, title: str,
            visualization: str, query_spec: dict, analysis: dict, metadata: dict,
            period: tuple[datetime, datetime, str]) -> dict[str, Any]:
    start, end, basis = period
    return {
        "kind": "analytics", "query_spec": query_spec,
        "resolved_period": {"start": start.isoformat(), "end": end.isoformat(), "basis": basis},
        "data": rows,
        "metadata": {
            "dataset": query_spec["dataset"], "geography_level": metadata["geography_level"],
            "station_count": max((row.get("active_station_count") or row.get("station_count") or 0 for row in rows), default=0),
            "value_status": metadata.get("value_status", "observed_and_policy_approved_filled"),
            "station_balanced": metadata.get("station_balanced", True),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2), "warnings": metadata.get("warnings", []),
        },
        "visualization": {"type": visualization}, "analysis": {"operator": operator, **analysis},
        "answer": answer, "display_title": title,
        "model_trace": {"provider": "deterministic", "model": operator, "latency_ms": 0,
                        "prompt_tokens": None, "completion_tokens": None, "fallback_used": False},
    }
