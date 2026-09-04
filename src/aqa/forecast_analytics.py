from __future__ import annotations

import calendar
import re
import time
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import duckdb
import numpy as np

from .capabilities import CapabilityAssessment, MONTHS
from .config import load_config
from .errors import DataUnavailableError, ValidationError
from .forecasting import forecast_at_issue, rolling_backtest
from .global_analytics import POLLUTANTS
from .lakehouse_catalog import LakehouseCatalog
from .quality_analytics import _requested_cities


DATASET_TIMEZONES = {
    "CPCB": "Asia/Kolkata", "CNEMC": "Asia/Shanghai", "AURN": "Europe/London",
    "SINAICA": "America/Mexico_City", "EPA": "America/New_York",
}

LOCATION_RULES = {
    ("CNEMC", "beijing"): ("city", "北京"),
    ("AURN", "london"): ("admin1", "Greater London"),
    ("SINAICA", "mexico city"): ("city", "CDMX"),
    ("CPCB", "bangalore"): ("city", "Bengaluru"),
    ("CPCB", "bengaluru"): ("city", "Bengaluru"),
}


class ForecastAnalytics:
    """Leakage-safe conversational hourly forecasting and historical verification."""

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.catalog = LakehouseCatalog(self.config)
        self.root = self.catalog.root

    @staticmethod
    def should_handle(question: str, assessment: CapabilityAssessment) -> bool:
        text = question.casefold()
        return assessment.intent == "forecast" or any(term in text for term in ("forecast", "backtest", "mase"))

    def execute(self, question: str, assessment: CapabilityAssessment) -> dict[str, Any]:
        horizons = _comparison_horizons(question.casefold())
        if len(horizons) >= 2:
            return self._multi_horizon_comparison(question, assessment, horizons)
        return self._execute_single(question, assessment)

    def _execute_single(self, question: str, assessment: CapabilityAssessment,
                        horizon_override: int | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        text = question.casefold()
        selected = list(self.catalog.selected(assessment.supported_datasets))
        if len(selected) != 1:
            raise DataUnavailableError("Forecasting currently requires one country dataset per request")
        item = selected[0]
        timezone_name = DATASET_TIMEZONES.get(item.dataset)
        if not timezone_name:
            raise DataUnavailableError(f"A verified forecast timezone is not configured for {item.country_name}")
        cities = _requested_cities(text)
        if len(cities) != 1:
            raise DataUnavailableError("Conversational forecasting currently requires exactly one named city")
        location = "Bengaluru" if cities[0] == "bangalore" else cities[0].title()
        pollutant_matches = [name for pattern, name in POLLUTANTS if re.search(pattern, text)]
        if len(pollutant_matches) != 1:
            raise DataUnavailableError("Concentration forecasting requires exactly one pollutant")
        pollutant = pollutant_matches[0]
        horizon = horizon_override or _horizon(text, int(self.config["forecasting"].get("horizon_hours", 24)))
        issue_times_utc, issue_basis = _issue_times(text, item.available_end, timezone_name)
        issue_utc = issue_times_utc[0]
        local_zone = ZoneInfo(timezone_name)
        local_issues = [value.astimezone(local_zone).replace(tzinfo=None) for value in issue_times_utc]
        issue_local = local_issues[0]
        context_hours = int(self.config["forecasting"].get("context_hours", 168))
        maximum_filled = float(self.config["forecasting"].get("maximum_input_filled_fraction", .20))
        minimum_targets = float(self.config["forecasting"].get("minimum_observed_target_fraction", .75))
        candidates = tuple(self.config["forecasting"].get("candidates", ["seasonal_naive", "linear_trend", "tsfm_api"]))
        model = _model(text, self.config["forecasting"].get("default_model", "auto"))
        dimension = self._stations(item.dataset, cities[0])
        if dimension.empty:
            raise DataUnavailableError(f"No monitoring stations resolve to {location}")
        paths = [str(path) for path in (self.root / "query" / "hourly" / f"dataset={item.dataset}").glob("year=*/month=*/*.parquet")]
        if not paths:
            raise DataUnavailableError("No recent hourly partitions are available for forecasting")
        local_issues = [value.replace(minute=0, second=0, microsecond=0) for value in local_issues]
        start = local_issues[0]
        query_start = min(local_issues) - _hours(context_hours - 1)
        query_end = max(local_issues) + _hours(horizon)
        connection = duckdb.connect()
        connection.register("forecast_stations", dimension[["source_station_id"]])
        try:
            result = connection.execute("""
                WITH station_hour AS (
                  SELECT h.timestamp_local,h.source_station_id,
                         avg(h.analysis_value) FILTER(WHERE h.analysis_value>=0) station_value,
                         avg(h.observed_value) FILTER(WHERE h.value_status='observed' AND h.observed_value>=0) observed_station_value,
                         count(*) FILTER(WHERE h.analysis_value>=0) valid_count,
                         count(*) FILTER(WHERE h.value_status LIKE 'imputed%' AND h.analysis_value>=0) filled_count,
                         any_value(h.unit) unit
                  FROM read_parquet(?,hive_partitioning=true,union_by_name=true) h
                  JOIN forecast_stations s USING(source_station_id)
                  WHERE h.pollutant=? AND h.timestamp_local BETWEEN ? AND ?
                  GROUP BY h.timestamp_local,h.source_station_id
                )
                SELECT timestamp_local AS "timestamp",avg(station_value) mean,
                       avg(observed_station_value) observed_mean,
                       sum(valid_count) valid_count,sum(filled_count) filled_count,
                       count(station_value) active_station_count,any_value(unit) unit
                FROM station_hour GROUP BY timestamp_local ORDER BY timestamp_local
            """, [paths, pollutant, query_start, query_end])
            names = [column[0] for column in result.description]
            rows = [dict(zip(names, row)) for row in result.fetchall()]
        finally:
            connection.close()
        if not rows:
            raise DataUnavailableError("No hourly values matched the forecast city, pollutant, and issue time")
        if len(local_issues) > 1:
            try:
                backtest = rolling_backtest(
                    rows, issue_times=local_issues, horizon=horizon, context_hours=context_hours,
                    maximum_filled_fraction=maximum_filled,
                    minimum_observed_target_fraction=minimum_targets,
                    model=model, candidates=candidates,
                )
            except ValidationError as exc:
                raise DataUnavailableError(exc.message) from exc
            return _rolling_result(
                backtest, item=item, location=location, pollutant=pollutant, unit=next((row["unit"] for row in rows if row.get("unit")), "concentration"),
                local_zone=local_zone, timezone_name=timezone_name, issue_basis=issue_basis,
                horizon=horizon, maximum_filled=maximum_filled, station_count=int(dimension["source_station_id"].nunique()),
                started=started,
            )
        try:
            forecast = forecast_at_issue(
                rows, issue_time=start, horizon=horizon, context_hours=context_hours,
                maximum_filled_fraction=maximum_filled,
                minimum_observed_target_fraction=minimum_targets,
                model=model, candidates=candidates,
                interval_level=float(self.config["forecasting"].get("interval_level", .90)),
            )
        except ValidationError as exc:
            raise DataUnavailableError(exc.message) from exc
        evaluation_requested = any(term in text for term in (
            "evaluate", "against", "backtest", "accuracy", "mae", "rmse", "mase", "smape", "verify",
        ))
        output = []
        unit = next((row["unit"] for row in rows if row.get("unit")), "concentration")
        for value in forecast.values:
            local_timestamp = datetime.fromisoformat(value["timestamp"]).replace(tzinfo=local_zone)
            row = {
                **value, "timestamp_local": value["timestamp"],
                "timestamp": local_timestamp.astimezone(timezone.utc).isoformat(),
                "location": location, "pollutant": pollutant, "unit": unit, "model": forecast.model,
            }
            if not evaluation_requested:
                row.pop("observed", None)
            output.append(row)
        metrics = forecast.metrics if evaluation_requested else None
        baseline_metrics = forecast.baseline_metrics if evaluation_requested else None
        predicted_mean = sum(row["forecast"] for row in output) / len(output)
        predicted_max = max(row["forecast"] for row in output)
        baseline_mean = sum(row["baseline_forecast"] for row in output) / len(output)
        if evaluation_requested and metrics:
            answer = (
                f"The leakage-safe {_model_label(forecast.model)} backtest for {location} {pollutant} has MAE {metrics['mae']:.2f}, "
                f"RMSE {metrics['rmse']:.2f}, MASE {_format_optional(metrics['mase'])}, sMAPE {metrics['smape']:.1f}%, and bias {metrics['bias']:.2f} {unit}. "
                f"It was scored against {metrics['observed_target_count']} directly observed target hours; seasonal naive MAE was {baseline_metrics['mae']:.2f}."
            )
        else:
            answer = (
                f"The {horizon}-hour {location} {pollutant} forecast uses {_model_label(forecast.model)}; its mean is {predicted_mean:.2f} "
                f"{unit} and maximum is {predicted_max:.2f} {unit}. The seasonal-naive mean is {baseline_mean:.2f} {unit}."
            )
            if "compare" in text and forecast.model == "seasonal_naive":
                answer += " The selected model is seasonal naive itself, so the selected and baseline forecasts are identical."
        if "interval" in text or "probab" in text:
            answer += f" The table includes a {forecast.interval_level:.0%} empirical prediction interval for every hour."
        answer += (
            f" Input context was {forecast.input_quality['filled_fraction']:.1%} filled, below the configured "
            f"{maximum_filled:.0%} maximum. Forecasts are estimates, not measurements."
        )
        issue_utc_text = issue_utc.isoformat()
        return {
            "kind": "analytics",
            "query_spec": {
                "dataset": item.dataset, "operation": "forecast_backtest" if evaluation_requested else "forecast",
                "geography_level": "city", "locations": [location], "pollutants": [pollutant],
                "time_range": {"start": issue_utc_text, "end": output[-1]["timestamp"]},
                "resolution": "hourly", "metrics": ["forecast", "lower", "upper", "baseline_forecast"],
                "filters": {"maximum_input_filled_fraction": maximum_filled, "observed_targets_only": True},
            },
            "resolved_period": {
                "start": output[0]["timestamp"], "end": output[-1]["timestamp"],
                "basis": issue_basis + f"; converted to {timezone_name} for the stored local-hour series",
            },
            "data": output,
            "metadata": {
                "dataset": item.dataset, "geography_level": "city",
                "station_count": int(dimension["source_station_id"].nunique()),
                "value_status": "model_estimate", "station_balanced": True,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2), "warnings": [],
                "timezone": timezone_name, "issue_time_utc": issue_utc_text,
            },
            "visualization": {"type": "line"},
            "analysis": {
                "operator": "hourly_forecast_v1", "model": forecast.model,
                "selection_scores_observed_mae": forecast.selection_scores,
                "metrics": metrics, "baseline_metrics": baseline_metrics,
                "input_quality": forecast.input_quality, "interval_level": forecast.interval_level,
                "training_cutoff_local": start.isoformat(), "training_cutoff_utc": issue_utc_text,
                "future_observations_used_for_training": False,
            },
            "answer": answer, "display_title": f"{location} {pollutant} · {horizon}-hour forecast",
            "model_trace": {
                "provider": "deterministic", "model": "hourly_forecast_v1", "latency_ms": 0,
                "prompt_tokens": None, "completion_tokens": None, "fallback_used": False,
            },
        }

    def _multi_horizon_comparison(self, question: str, assessment: CapabilityAssessment,
                                  horizons: list[int]) -> dict[str, Any]:
        started = time.perf_counter()
        results = [self._execute_single(question, assessment, horizon_override=horizon) for horizon in horizons]
        rows = []
        for horizon, result in zip(horizons, results):
            values = result["data"]
            rows.append({
                "horizon_hours": horizon,
                "predicted_mean": float(np.mean([row["forecast"] for row in values])),
                "predicted_maximum": float(np.max([row["forecast"] for row in values])),
                "seasonal_naive_mean": float(np.mean([row["baseline_forecast"] for row in values])),
                "model": result["analysis"]["model"],
                "issue_time_utc": result["metadata"]["issue_time_utc"],
                "input_filled_fraction": float(result["analysis"]["input_quality"]["filled_fraction"]),
                "location": result["query_spec"]["locations"][0],
                "pollutant": result["query_spec"]["pollutants"][0],
                "unit": values[0]["unit"],
            })
        issue_times = {row["issue_time_utc"] for row in rows}
        if len(issue_times) != 1:
            raise DataUnavailableError("Multi-horizon comparison did not resolve to one common issue time")
        location = rows[0]["location"]; pollutant = rows[0]["pollutant"]; unit = rows[0]["unit"]
        summary = ", ".join(f"{row['horizon_hours']}h: {row['predicted_mean']:.2f} {unit}" for row in rows)
        answer = (
            f"Using the same {rows[0]['issue_time_utc']} issue time and identical leakage-safe context, the predicted means for "
            f"{location} {pollutant} are {summary}. Each horizon was selected and calculated independently with the configured model registry. "
            "Forecasts are estimates, not measurements."
        )
        return {
            "kind": "analytics",
            "query_spec": {
                "dataset": results[0]["query_spec"]["dataset"], "operation": "multi_horizon_forecast_comparison",
                "geography_level": "city", "locations": [location], "pollutants": [pollutant],
                "time_range": {"start": rows[0]["issue_time_utc"], "end": results[-1]["resolved_period"]["end"]},
                "resolution": "forecast_horizon", "metrics": ["predicted_mean", "predicted_maximum", "seasonal_naive_mean"],
                "filters": {"same_issue_time": True, "horizons": horizons},
            },
            "resolved_period": {"start": rows[0]["issue_time_utc"], "end": results[-1]["resolved_period"]["end"],
                                "basis": "one common issue time with independently evaluated forecast horizons"},
            "data": rows,
            "metadata": {
                **results[0]["metadata"], "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "value_status": "model_estimate",
            },
            "visualization": {"type": "bar"},
            "analysis": {
                "operator": "multi_horizon_forecast_comparison_v1", "horizons": horizons,
                "same_issue_time": True, "future_observations_used_for_training": False,
                "per_horizon_models": {str(row["horizon_hours"]): row["model"] for row in rows},
            },
            "answer": answer, "display_title": f"{location} {pollutant} · multi-horizon forecast",
            "model_trace": {"provider": "deterministic", "model": "multi_horizon_forecast_comparison_v1",
                            "latency_ms": 0, "prompt_tokens": None, "completion_tokens": None, "fallback_used": False},
        }

    def _stations(self, dataset: str, city: str):
        connection = duckdb.connect(str(self.catalog.control_path), read_only=True)
        try:
            rule = LOCATION_RULES.get((dataset, city))
            if rule:
                column, value = rule
                return connection.execute(
                    f"SELECT source_station_id FROM station_dimension WHERE dataset=? AND {column}=?",
                    [dataset, value],
                ).fetchdf()
            return connection.execute(
                "SELECT source_station_id FROM station_dimension WHERE dataset=? AND lower(city)=lower(?)",
                [dataset, city],
            ).fetchdf()
        finally:
            connection.close()


def _issue_times(text: str, available_end: datetime, timezone_name: str) -> tuple[list[datetime], str]:
    names = "|".join(sorted(MONTHS, key=len, reverse=True))
    date_range = re.search(rf"\b(?:from\s+)?(\d{{1,2}})\s*[–-]\s*(\d{{1,2}})\s+({names})\s+(20\d{{2}})", text)
    if date_range:
        first, last = int(date_range.group(1)), int(date_range.group(2))
        if last < first:
            raise DataUnavailableError("Forecast issue-date range must run from an earlier day to a later day")
        zone = ZoneInfo(timezone_name)
        values = [
            datetime(int(date_range.group(4)), MONTHS[date_range.group(3)], day, 23, tzinfo=zone).astimezone(timezone.utc)
            for day in range(first, last + 1)
        ]
        return values, f"daily 23:00 issue times in {timezone_name}"
    match = re.search(rf"\b(\d{{1,2}})\s+({names})\s+(20\d{{2}})(?:\s+(\d{{1,2}}):(\d{{2}}))?\s*(utc)?", text)
    if match:
        value = datetime(
            int(match.group(3)), MONTHS[match.group(2)], int(match.group(1)),
            int(match.group(4) or 23), int(match.group(5) or 0), tzinfo=timezone.utc if match.group(6) else ZoneInfo(timezone_name),
        )
        basis = "explicit UTC issue time" if match.group(6) else f"explicit {timezone_name} issue time"
        return [value.astimezone(timezone.utc)], basis
    local = available_end.replace(tzinfo=ZoneInfo(timezone_name))
    return [local.astimezone(timezone.utc)], "latest loaded local timestamp"


def _horizon(text: str, default: int) -> int:
    patterns = (
        r"next\s+(\d+)\s+hours?", r"for\s+(?:the\s+)?(?:next\s+)?(\d+)[- ]hours?",
        r"(\d+)[- ]hour\s+(?:forecast|horizon|backtest)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = int(match.group(1))
            if value < 1 or value > 720:
                raise DataUnavailableError("Forecast horizon must be between 1 and 720 hours")
            return value
    return default


def _comparison_horizons(text: str) -> list[int]:
    match = re.search(r"for\s+([0-9,\sand]+?)\s+hours?\b", text)
    if not match or "same issue time" not in text:
        return []
    values = list(dict.fromkeys(int(value) for value in re.findall(r"\d+", match.group(1))))
    if any(value < 1 or value > 720 for value in values):
        raise DataUnavailableError("Forecast horizons must be between 1 and 720 hours")
    return values


def _model(text: str, default: str) -> str:
    if "linear trend" in text:
        return "linear_trend"
    if "seasonal naive only" in text or "using seasonal naive" in text:
        return "seasonal_naive"
    return default


def _hours(value: int):
    from datetime import timedelta
    return timedelta(hours=value)


def _format_optional(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "not available"


def _model_label(value: str) -> str:
    return value.replace("_", " ")


def _rolling_result(backtest: dict, *, item, location: str, pollutant: str, unit: str,
                    local_zone: ZoneInfo, timezone_name: str, issue_basis: str, horizon: int,
                    maximum_filled: float, station_count: int, started: float) -> dict[str, Any]:
    rows = []
    for origin in backtest["origins"]:
        forecast = origin["forecast"]
        local_issue = datetime.fromisoformat(origin["issue_time"]).replace(tzinfo=local_zone)
        metrics = forecast.metrics or {}
        rows.append({
            "issue_time": local_issue.astimezone(timezone.utc).isoformat(),
            "issue_time_local": origin["issue_time"], "location": location, "pollutant": pollutant,
            "model": forecast.model, "mae": metrics.get("mae"), "rmse": metrics.get("rmse"),
            "mase": metrics.get("mase"), "smape": metrics.get("smape"), "bias": metrics.get("bias"),
            "interval_coverage": metrics.get("interval_coverage"),
            "observed_target_count": metrics.get("observed_target_count"),
            "baseline_mae": (forecast.baseline_metrics or {}).get("mae"),
            "filled_fraction": (forecast.input_quality or {}).get("filled_fraction"),
            "context_mean": (forecast.input_quality or {}).get("context_mean"),
            "context_std": (forecast.input_quality or {}).get("context_std"),
            "context_coefficient_of_variation": (forecast.input_quality or {}).get("context_coefficient_of_variation"),
            "unit": unit,
        })
    metrics = backtest["metrics"]; baseline = backtest["baseline_metrics"]
    answer = (
        f"The {backtest['origin_count']}-origin {location} {pollutant} backtest has aggregate MAE {metrics['mae']:.2f}, "
        f"RMSE {metrics['rmse']:.2f}, MASE {_format_optional(metrics.get('mase'))}, sMAPE {metrics['smape']:.1f}%, and bias {metrics['bias']:.2f} {unit}. "
        f"It used {backtest['observed_target_count']} directly observed target hours; seasonal-naive MAE was {baseline['mae']:.2f}. "
        f"Every origin enforced the configured {maximum_filled:.0%} maximum filled input fraction."
    )
    return {
        "kind": "analytics",
        "query_spec": {
            "dataset": item.dataset, "operation": "rolling_backtest", "geography_level": "city",
            "locations": [location], "pollutants": [pollutant],
            "time_range": {"start": rows[0]["issue_time"], "end": rows[-1]["issue_time"]},
            "resolution": "daily_origins", "metrics": ["mae", "rmse", "mase", "smape", "bias", "interval_coverage"],
            "filters": {"maximum_input_filled_fraction": maximum_filled, "observed_targets_only": True},
        },
        "resolved_period": {"start": rows[0]["issue_time"], "end": rows[-1]["issue_time"], "basis": issue_basis},
        "data": rows,
        "metadata": {
            "dataset": item.dataset, "geography_level": "city", "station_count": station_count,
            "value_status": "model_estimate", "station_balanced": True,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2), "warnings": [], "timezone": timezone_name,
        },
        "visualization": {"type": "line"},
        "analysis": {
            "operator": "rolling_hourly_backtest_v1", "metrics": metrics, "baseline_metrics": baseline,
            "origin_count": backtest["origin_count"], "future_observations_used_for_training": False,
            "evaluation_targets": "directly_observed_only",
        },
        "answer": answer, "display_title": f"{location} {pollutant} · rolling {horizon}-hour backtest",
        "model_trace": {"provider": "deterministic", "model": "rolling_hourly_backtest_v1", "latency_ms": 0,
                        "prompt_tokens": None, "completion_tokens": None, "fallback_used": False},
    }
