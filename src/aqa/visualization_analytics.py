from __future__ import annotations

import math
import time
from datetime import datetime, timedelta
from typing import Any

import duckdb
import numpy as np

from .capabilities import CapabilityAssessment, PRIMARY_DATASETS
from .forecast_portfolio import ForecastPortfolioAnalytics
from .lakehouse_catalog import LakehouseCatalog
from .provenance_analytics import ProvenanceAnalytics
from .quality_analytics import QualityAnalytics


class VisualizationAnalytics:
    """Deterministic plotted-data builders with explicit visualization contracts."""

    def __init__(self, config: dict | None = None):
        self.catalog = LakehouseCatalog(config)
        self.root = self.catalog.root
        self.quality = QualityAnalytics(config)
        self.provenance = ProvenanceAnalytics(config)
        self.forecasts = ForecastPortfolioAnalytics(config)
        self.maximum_pm25 = float(self.quality.config.get("visualization", {}).get("maximum_plausible_pm25", 5000))
        self.episode_hours = int(self.quality.config.get("visualization", {}).get("episode_window_hours", 72))
        self.episode_count = int(self.quality.config.get("visualization", {}).get("episode_count", 3))

    @staticmethod
    def should_handle(question: str, assessment: CapabilityAssessment) -> bool:
        del assessment
        text = question.casefold()
        return any(pattern in text for pattern in (
            "mean versus pm2.5 data coverage", "dual-view analysis", "country-by-month pm2.5 mean",
            "small-multiple visualization", "year-over-year pm2.5 change versus",
            "24-hour forecast accuracy and data completeness", "integrated visual summary",
            "mean versus missing-data percentage", "multi-panel episode visualization",
        ))

    def execute(self, question: str, assessment: CapabilityAssessment) -> dict[str, Any]:
        text = question.casefold()
        if "multi-panel episode visualization" in text:
            return self._episode_panels()
        if "mean versus missing-data percentage" in text:
            return self._missingness_outlier_scatter()
        if "24-hour forecast accuracy and data completeness" in text:
            return self._forecast_quality()
        if "country-by-month pm2.5 mean" in text:
            return self._coverage_heatmap(assessment)
        if "small-multiple visualization" in text:
            return self._provenance_small_multiples(question, assessment)
        if "year-over-year pm2.5 change versus" in text:
            return self._yoy_scatter(assessment)
        if "integrated visual summary" in text:
            return self._integrated_summary(assessment)
        if "dual-view analysis" in text:
            return self._dual_view(assessment)
        result = self.quality.execute(question, assessment)
        result["visualization"] = {
            "type": "scatter", "x": "coverage", "y": "mean", "label": "country",
            "size": "active_station_count", "x_title": "PM2.5 data coverage", "y_title": "Mean PM2.5",
            "x_format": "percent",
        }
        result["analysis"]["operator"] = "pollution_coverage_scatter_v1"
        result["answer"] += " The scatter positions encode coverage and mean concentration; labels identify countries and marker size reflects active stations."
        return result

    def _missingness_outlier_scatter(self) -> dict[str, Any]:
        started = time.perf_counter()
        selected = list(self.catalog.datasets)
        raw = self._quality_rows([item.dataset for item in selected], datetime(2026, 7, 1), datetime(2026, 7, 31, 23, 59, 59))
        by_dataset = {row["dataset"]: row for row in raw}
        rows = []
        for item in selected:
            source = by_dataset.get(item.dataset)
            eligible = bool(source and source["mean"] is not None and source.get("expected_hour_count"))
            missing = (
                (source["missing_count"] + source["invalid_count"]) / source["expected_hour_count"]
                if eligible else None
            )
            rows.append({
                "dataset": item.dataset, "country": item.country_name, "pollutant": "PM25", "eligible": eligible,
                "mean": source["mean"] if eligible else None, "missing_percentage": missing,
                "coverage": source["coverage"] if source else None,
                "active_station_count": source["active_station_count"] if source else 0,
                "outlier_mean": False, "outlier_missingness": False, "outlier_label": "",
                "exclusion_reason": None if eligible else "no usable July 2026 PM2.5 country result",
                "unit": source["unit"] if source else "µg/m³",
            })
        eligible_rows = [row for row in rows if row["eligible"]]
        mean_flags = _robust_outliers([row["mean"] for row in eligible_rows])
        missing_flags = _robust_outliers([row["missing_percentage"] for row in eligible_rows])
        for row, mean_flag, missing_flag in zip(eligible_rows, mean_flags, missing_flags):
            row["outlier_mean"], row["outlier_missingness"] = mean_flag, missing_flag
            if mean_flag or missing_flag:
                dimensions = "mean and missingness" if mean_flag and missing_flag else "mean" if mean_flag else "missingness"
                row["outlier_label"] = f"{row['country']} · {dimensions}"
        outliers = [row for row in eligible_rows if row["outlier_label"]]
        return self._result(
            rows, started, "pollution_missingness_outlier_scatter_v1",
            f"{len(eligible_rows)} of {len(rows)} countries have usable July 2026 PM2.5 values. "
            f"{len(outliers)} point(s) are labelled as robust outliers in concentration, missingness, or both. "
            "Outliers use the deterministic modified-z rule; excluded countries remain in the downloadable table.",
            "July 2026 PM2.5 versus missing data", datetime(2026, 7, 1), datetime(2026, 7, 31, 23, 59, 59),
            {"type": "scatter", "x": "missing_percentage", "y": "mean", "label": "outlier_label",
             "hover_label": "country", "size": "active_station_count", "filter": {"eligible": True},
             "x_title": "Missing-data percentage", "y_title": "Mean PM2.5", "x_format": "percent"},
            {"outlier_method": "modified_z_score_absolute_above_3.5",
             "eligible_country_count": len(eligible_rows), "outlier_country_count": len(outliers),
             "outlier_countries": [row["country"] for row in outliers]},
        )

    def _episode_panels(self) -> dict[str, Any]:
        started = time.perf_counter()
        threshold = 15.0
        daily = self._delhi_daily_series()
        window_days = self.episode_hours // 24
        candidates = []
        for index in range(len(daily) - window_days + 1):
            window = daily[index:index + window_days]
            if window[-1]["timestamp"].date() != (window[0]["timestamp"] + timedelta(days=window_days - 1)).date():
                continue
            severity = sum(max(row["mean"] - threshold, 0) for row in window)
            candidates.append({
                "start": window[0]["timestamp"], "end": window[-1]["timestamp"] + timedelta(hours=23),
                "severity": severity, "peak_daily_mean": max(row["mean"] for row in window),
                "observed_fraction": sum(row["observed_fraction"] for row in window) / len(window),
                "imputed_fraction": sum(row["imputed_fraction"] for row in window) / len(window),
            })
        chosen = []
        for candidate in sorted(candidates, key=lambda row: row["severity"], reverse=True):
            if all(candidate["end"] < existing["start"] or candidate["start"] > existing["end"] for existing in chosen):
                chosen.append(candidate)
            if len(chosen) == self.episode_count:
                break
        rows = self._delhi_episode_hours(chosen, threshold)
        summaries = []
        for rank, episode in enumerate(chosen, 1):
            summaries.append({
                "rank": rank, "start": episode["start"].isoformat(), "end": episode["end"].isoformat(),
                "severity": episode["severity"], "peak_daily_mean": episode["peak_daily_mean"],
                "observed_fraction": episode["observed_fraction"], "imputed_fraction": episode["imputed_fraction"],
            })
        return self._result(
            rows, started, "delhi_pm25_episode_panels_v1",
            f"The panels show the {len(chosen)} highest-severity non-overlapping {self.episode_hours}-hour Delhi PM2.5 windows in the loaded history. "
            f"Severity is cumulative daily excess above the configured {threshold:g} µg/m³ WHO guideline. "
            "Each monitoring station receives equal weight in each hourly Delhi value. Marker colors distinguish "
            "measured hours, filled hours, hours containing both measured and filled station values, and missing hours.",
            "Delhi PM2.5 · three most severe episodes", daily[0]["timestamp"], daily[-1]["timestamp"] + timedelta(hours=23),
            {"type": "episode_panels", "panel": "episode_label", "x": "timestamp", "y": "concentration",
             "status": "value_status", "threshold": "guideline_threshold", "y_title": "PM2.5 µg/m³"},
            {"episode_definition": f"non-overlapping_{self.episode_hours}_hour_windows",
             "severity_definition": "sum(max(daily_city_mean-guideline,0))", "guideline_threshold": threshold,
             "guideline_id": "WHO_AQG_2021", "minimum_daily_coverage": .75, "episodes": summaries,
             "daily_selection_start": daily[0]["timestamp"].isoformat(),
             "daily_selection_end": daily[-1]["timestamp"].isoformat()},
        )

    def _delhi_daily_series(self) -> list[dict[str, Any]]:
        control = duckdb.connect(str(self.catalog.control_path), read_only=True)
        try:
            stations = control.execute(
                "SELECT source_station_id FROM station_dimension WHERE dataset='CPCB' AND lower(city)='delhi'"
            ).fetchdf()
        finally:
            control.close()
        old_paths = [str(path) for path in (self.root / "query" / "daily" / "dataset=CPCB").glob("year=*/month=*/*.parquet")]
        recent_paths = [str(path) for path in (self.root / "query" / "area_daily" / "dataset=CPCB").glob("year=*/month=*/*.parquet")]
        connection = duckdb.connect()
        connection.register("delhi_episode_stations", stations)
        try:
            old = connection.execute("""
                SELECT bucket_start bucket,avg(mean) mean,
                       sum(valid_hour_count)::DOUBLE/nullif(sum(expected_hour_count),0) coverage,
                       sum(observed_count)::DOUBLE/nullif(sum(valid_hour_count),0) observed_fraction,
                       sum(imputed_count)::DOUBLE/nullif(sum(valid_hour_count),0) imputed_fraction
                FROM read_parquet(?,hive_partitioning=true,union_by_name=true)
                JOIN delhi_episode_stations USING(source_station_id)
                WHERE pollutant='PM25' GROUP BY bucket_start HAVING coverage>=.75
            """, [old_paths]).fetchall()
            recent = connection.execute("""
                SELECT bucket_day bucket,mean,coverage,observed_fraction,imputed_fraction
                FROM read_parquet(?,hive_partitioning=true,union_by_name=true)
                WHERE pollutant='PM25' AND geography_level='city' AND lower(area_name)='delhi' AND coverage>=.75
            """, [recent_paths]).fetchall()
        finally:
            connection.close()
        by_time = {row[0]: row for row in old}
        by_time.update({row[0]: row for row in recent})
        return [{"timestamp": timestamp, "mean": row[1], "coverage": row[2],
                 "observed_fraction": row[3] or 0.0, "imputed_fraction": row[4] or 0.0}
                for timestamp, row in sorted(by_time.items()) if row[1] is not None and row[1] <= self.maximum_pm25]

    def _delhi_episode_hours(self, episodes: list[dict], threshold: float) -> list[dict[str, Any]]:
        control = duckdb.connect(str(self.catalog.control_path), read_only=True)
        try:
            stations = control.execute(
                "SELECT source_station_id FROM station_dimension WHERE dataset='CPCB' AND lower(city)='delhi'"
            ).fetchdf()
        finally:
            control.close()
        paths = []
        for episode in episodes:
            current = episode["start"].replace(day=1)
            while current <= episode["end"]:
                paths.extend(str(path) for path in (self.root / "archive" / "hourly" / "dataset=CPCB" /
                             f"year={current.year}" / f"month={current.month}").glob("*.parquet"))
                current = datetime(current.year + 1, 1, 1) if current.month == 12 else current.replace(month=current.month + 1)
        connection = duckdb.connect()
        connection.register("delhi_episode_stations", stations)
        try:
            result = connection.execute("""
                SELECT timestamp_local,avg(analysis_value) FILTER(WHERE analysis_value>=0) concentration,
                       count(*) FILTER(WHERE value_status='observed' AND analysis_value>=0) observed_station_count,
                       count(*) FILTER(WHERE value_status LIKE 'imputed%' AND analysis_value>=0) filled_station_count,
                       count(*) FILTER(WHERE analysis_value>=0) valid_station_count,count(*) series_row_count,
                       any_value(unit) unit
                FROM read_parquet(?,hive_partitioning=true,union_by_name=true)
                JOIN delhi_episode_stations USING(source_station_id)
                WHERE pollutant='PM25' GROUP BY timestamp_local ORDER BY timestamp_local
            """, [sorted(set(paths))])
            names = [column[0] for column in result.description]
            hourly = {row[0]: dict(zip(names, row)) for row in result.fetchall()}
        finally:
            connection.close()
        rows = []
        for rank, episode in enumerate(episodes, 1):
            timestamp = episode["start"]
            label = f"#{rank} · {episode['start']:%d %b %Y}–{episode['end']:%d %b %Y}"
            while timestamp <= episode["end"]:
                source = hourly.get(timestamp)
                concentration = source["concentration"] if source else None
                observed = int(source["observed_station_count"] or 0) if source else 0
                filled = int(source["filled_station_count"] or 0) if source else 0
                if concentration is None: status = "missing"
                elif filled and observed: status = "mixed"
                elif filled: status = "imputed"
                else: status = "observed"
                rows.append({
                    "episode_rank": rank, "episode_label": label, "timestamp": timestamp.isoformat(),
                    "location": "Delhi", "pollutant": "PM25", "concentration": concentration,
                    "marker_value": concentration if concentration is not None else 0.0,
                    "value_status": status, "observed_station_count": observed, "filled_station_count": filled,
                    "valid_station_count": int(source["valid_station_count"] or 0) if source else 0,
                    "missing_station_count": max(0, int(source["series_row_count"] or 0) - int(source["valid_station_count"] or 0)) if source else None,
                    "guideline_threshold": threshold, "unit": source["unit"] if source else "µg/m³",
                })
                timestamp += timedelta(hours=1)
        return rows

    def _dual_view(self, assessment: CapabilityAssessment) -> dict[str, Any]:
        started = time.perf_counter()
        selected = list(self.catalog.selected(assessment.supported_datasets))
        raw = self._quality_rows([item.dataset for item in selected], datetime(2026, 7, 1), datetime(2026, 7, 31, 23, 59, 59))
        rows = [{
            "dataset": row["dataset"], "country": row["country"], "pollutant": "PM25",
            "mean": row["mean"], "coverage": row["coverage"],
            "active_station_count": row["active_station_count"], "unit": row["unit"],
        } for row in raw if row["mean"] is not None]
        rows.sort(key=lambda row: row["mean"], reverse=True)
        rows = rows[:10]
        for rank, row in enumerate(rows, 1): row["rank"] = rank
        return self._result(
            rows, started, "rank_coverage_dual_view_v1",
            f"The dual view ranks {len(rows)} eligible countries by July 2026 mean PM2.5. Bars encode concentration and companion markers encode coverage on a separate percentage axis, so data completeness is not confused with pollution level.",
            "July 2026 PM2.5 ranking with coverage", datetime(2026, 7, 1), datetime(2026, 7, 31, 23, 59, 59),
            {"type": "dual_axis", "x": "country", "y": "mean", "secondary_y": "coverage",
             "y_title": "Mean PM2.5", "secondary_y_title": "Coverage", "secondary_format": "percent"},
            {"ranking_metric": "mean", "eligible_country_count": len(rows)},
        )

    def _coverage_heatmap(self, assessment: CapabilityAssessment) -> dict[str, Any]:
        started = time.perf_counter()
        selected = [item for item in self.catalog.selected(assessment.supported_datasets) if item.dataset in PRIMARY_DATASETS]
        start, end = datetime(2025, 8, 1), datetime(2026, 7, 31, 23, 59, 59)
        raw = self._quality_rows([item.dataset for item in selected], start, end)
        lookup = {(row["dataset"], row["month_start"].year, row["month_start"].month): row for row in raw}
        rows = []
        current = start
        while current <= end:
            for item in selected:
                source = lookup.get((item.dataset, current.year, current.month))
                coverage = source["coverage"] if source else None
                masked = source is None or source["mean"] is None or coverage is None or coverage < .80
                rows.append({
                    "dataset": item.dataset, "country": item.country_name, "timestamp": current.isoformat(),
                    "pollutant": "PM25", "mean": None if masked else source["mean"],
                    "unmasked_mean": source["mean"] if source else None, "coverage": coverage,
                    "masked": masked, "mask_reason": "no usable monthly value" if source is None else
                    ("invalid extreme value masked" if source["mean"] is None and coverage else
                     "coverage below 80%" if masked else None), "unit": source["unit"] if source else "µg/m³",
                })
            current = datetime(current.year + 1, 1, 1) if current.month == 12 else datetime(current.year, current.month + 1, 1)
        masked_count = sum(row["masked"] for row in rows)
        return self._result(
            rows, started, "coverage_masked_country_month_heatmap_v1",
            f"The heatmap contains {len(rows)} country-month cells; {masked_count} are visibly masked because coverage is below 80% or no usable value exists. Missing countries or months were not substituted.",
            "PM2.5 country-by-month mean · coverage masked", start, end,
            {"type": "heatmap_grid", "x": "timestamp", "y": "country", "z": "mean", "mask": "masked",
             "hover_fields": ["coverage", "mask_reason"], "color_title": "PM2.5 µg/m³"},
            {"coverage_threshold": .80, "masked_cell_count": masked_count, "total_cell_count": len(rows)},
        )

    def _provenance_small_multiples(self, question: str, assessment: CapabilityAssessment) -> dict[str, Any]:
        result = self.provenance.execute(question, assessment)
        masked = 0
        for row in result["data"]:
            value = row.get("mean")
            if value is not None and (not math.isfinite(value) or value > self.maximum_pm25):
                row["mean"] = None
                row["visualization_exclusion_reason"] = "invalid extreme value masked"
                masked += 1
        result["visualization"] = {
            "type": "small_multiples", "panel": "country", "x": "timestamp", "y": "mean",
            "series": "policy_label", "include_series": ["measured only", "all usable values"],
            "y_title": "Monthly PM2.5",
        }
        result["analysis"]["operator"] = "provenance_small_multiples_v1"
        country_count = len({row["country"] for row in result["data"]})
        result["answer"] = (
            f"I created {country_count} country panels for monthly PM2.5, with measured-only and all-usable "
            "trajectories shown on the same axes. Missing policy-month values remain gaps, and filled values "
            "are not described as measurements."
        )
        if masked:
            result["answer"] += f" {masked} extreme plotted value(s) were masked under the configured display sanity bound and remain disclosed in the table."
        result["analysis"]["extreme_value_mask_count"] = masked
        result["analysis"]["maximum_plausible_pm25"] = self.maximum_pm25
        result["display_title"] = "PM2.5 · measured and filled monthly trajectories"
        return result

    def _yoy_scatter(self, assessment: CapabilityAssessment) -> dict[str, Any]:
        started = time.perf_counter()
        selected = list(self.catalog.selected(assessment.supported_datasets))
        if len(selected) == 1 and selected[0].dataset == "CPCB":
            selected = list(self.catalog.datasets)
        rows = self._yoy_rows(selected)
        eligible = [row for row in rows if row["eligible"]]
        return self._result(
            rows, started, "yoy_completeness_scatter_v1",
            f"{len(eligible)} of {len(rows)} candidate countries have usable PM2.5 results in both July 2025 and July 2026. The scatter uses year-over-year change on one axis, July 2026 completeness on the other, and station count for marker size; exclusions remain in the downloadable table.",
            "PM2.5 year-over-year change versus completeness", datetime(2025, 7, 1), datetime(2026, 7, 31, 23, 59, 59),
            {"type": "scatter", "x": "coverage_2026", "y": "yoy_change", "size": "station_count",
             "label": "country", "filter": {"eligible": True}, "x_title": "July 2026 completeness",
             "y_title": "PM2.5 change vs July 2025", "x_format": "percent"},
            {"eligible_country_count": len(eligible), "candidate_country_count": len(rows)},
        )

    def _integrated_summary(self, assessment: CapabilityAssessment) -> dict[str, Any]:
        started = time.perf_counter()
        selected = list(self.catalog.selected(assessment.supported_datasets))
        rows = [row for row in self._yoy_rows(selected) if row["eligible"]]
        rows.sort(key=lambda row: row["mean_2026"], reverse=True)
        rows = rows[:10]
        for rank, row in enumerate(rows, 1):
            row["rank"] = rank
        return self._result(
            rows, started, "integrated_pollution_quality_summary_v1",
            f"The integrated summary ranks {len(rows)} eligible countries by July 2026 PM2.5 and combines concentration, completeness, temporal-sampling uncertainty, and year-over-year change in coordinated panels. Countries lacking either July comparison value are excluded rather than substituted.",
            "July 2026 PM2.5 · integrated top-10 summary", datetime(2025, 7, 1), datetime(2026, 7, 31, 23, 59, 59),
            {"type": "integrated", "x": "country", "rank_y": "mean_2026", "coverage_y": "coverage_2026",
             "change_y": "yoy_change", "error_lower": "ci_lower", "error_upper": "ci_upper"},
            {"ranking_metric": "mean_2026", "uncertainty_method": "normal interval over valid daily values",
             "country_count": len(rows)},
        )

    def _forecast_quality(self) -> dict[str, Any]:
        result = self.forecasts._rolling_pm25()
        result["visualization"] = {
            "type": "scatter", "x": "mean_input_filled_fraction", "y": "mae", "label": "city",
            "filter": {"eligible": True}, "x_title": "Filled fraction in forecast input",
            "y_title": "24-hour forecast MAE", "x_format": "percent", "color": "beats_seasonal_naive",
        }
        result["analysis"]["operator"] = "forecast_accuracy_completeness_visual_v1"
        result["query_spec"]["operation"] = "forecast_accuracy_completeness_visual_v1"
        result["answer"] += " The visualization places forecast error and filled-input fraction on separate axes; color indicates whether the model beat seasonal naive, so missing-data effects are not merged into model error."
        result["display_title"] = "PM2.5 forecast accuracy versus input completeness"
        return result

    def _yoy_rows(self, selected) -> list[dict[str, Any]]:
        raw = self._quality_rows([item.dataset for item in selected], datetime(2025, 7, 1), datetime(2026, 7, 31, 23, 59, 59))
        by_key = {(row["dataset"], row["month_start"].year): row for row in raw if row["month_start"].month == 7}
        output = []
        for item in selected:
            prior, current = by_key.get((item.dataset, 2025)), by_key.get((item.dataset, 2026))
            eligible = bool(prior and current and prior["mean"] is not None and current["mean"] is not None)
            standard_error = current["std"] / math.sqrt(current["valid_day_count"]) if eligible and current["std"] is not None and current["valid_day_count"] else None
            output.append({
                "dataset": item.dataset, "country": item.country_name, "pollutant": "PM25", "eligible": eligible,
                "exclusion_reason": None if eligible else "usable July PM2.5 is missing in 2025 or 2026",
                "mean_2025": prior["mean"] if prior else None, "mean_2026": current["mean"] if current else None,
                "yoy_change": current["mean"] - prior["mean"] if eligible else None,
                "yoy_percent_change": (current["mean"] - prior["mean"]) / prior["mean"] if eligible and prior["mean"] else None,
                "coverage_2026": current["coverage"] if current else None,
                "station_count": current["active_station_count"] if current else 0,
                "ci_lower": current["mean"] - 1.96 * standard_error if standard_error is not None else None,
                "ci_upper": current["mean"] + 1.96 * standard_error if standard_error is not None else None,
                "unit": current["unit"] if current else (prior["unit"] if prior else "µg/m³"),
            })
        return output

    def _quality_rows(self, datasets: list[str], start: datetime, end: datetime) -> list[dict[str, Any]]:
        paths = [str(path) for dataset in datasets for path in
                 (self.root / "query" / "quality_monthly" / f"dataset={dataset}").glob("year=*/month=*/*.parquet")]
        if not paths:
            return []
        connection = duckdb.connect()
        try:
            result = connection.execute("""
                SELECT dataset,any_value(country_name) country,month_start,mean,std,coverage,
                       valid_day_count,active_station_count,expected_hour_count,missing_count,invalid_count,
                       any_value(unit) unit
                FROM read_parquet(?,hive_partitioning=true,union_by_name=true)
                WHERE geography_level='country' AND pollutant='PM25'
                  AND month_start BETWEEN date_trunc('month',?) AND date_trunc('month',?)
                GROUP BY dataset,month_start,mean,std,coverage,valid_day_count,active_station_count,
                         expected_hour_count,missing_count,invalid_count
                ORDER BY dataset,month_start
            """, [paths, start, end])
            names = [column[0] for column in result.description]
            rows = [dict(zip(names, row)) for row in result.fetchall()]
            for row in rows:
                value = row.get("mean")
                if value is not None and (not math.isfinite(value) or value > self.maximum_pm25):
                    row["mean"] = None
                    row["extreme_value_masked"] = True
            return rows
        finally:
            connection.close()

    @staticmethod
    def _result(rows: list[dict[str, Any]], started: float, operator: str, answer: str, title: str,
                start: datetime, end: datetime, visualization: dict, analysis: dict) -> dict[str, Any]:
        return {
            "kind": "analytics",
            "query_spec": {"dataset": "GLOBAL_LAKEHOUSE", "operation": operator, "geography_level": "country",
                           "locations": sorted({row.get("country") for row in rows if row.get("country")}),
                           "pollutants": ["PM25"], "time_range": {"start": start.isoformat(), "end": end.isoformat()},
                           "resolution": "monthly", "metrics": ["mean", "coverage"], "filters": {}},
            "resolved_period": {"start": start.isoformat(), "end": end.isoformat(), "basis": "explicit requested period"},
            "data": rows,
            "metadata": {"dataset": "GLOBAL_LAKEHOUSE", "geography_level": "country", "station_balanced": True,
                         "value_status": "observed_and_policy_approved_filled",
                         "latency_ms": round((time.perf_counter() - started) * 1000, 2), "warnings": []},
            "visualization": visualization, "analysis": {"operator": operator, **analysis}, "answer": answer,
            "display_title": title,
            "model_trace": {"provider": "deterministic", "model": operator, "latency_ms": 0,
                            "prompt_tokens": None, "completion_tokens": None, "fallback_used": False},
        }


def _robust_outliers(values: list[float]) -> list[bool]:
    numeric = np.asarray(values, dtype=float)
    if not len(numeric):
        return []
    center = float(np.median(numeric))
    mad = float(np.median(np.abs(numeric - center)))
    if mad == 0:
        return [bool(abs(value - center) > 0) for value in numeric]
    scores = .6745 * np.abs(numeric - center) / mad
    return [bool(score > 3.5) for score in scores]
