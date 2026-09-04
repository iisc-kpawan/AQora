from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Iterable

import numpy as np
import httpx

from .errors import ValidationError


@dataclass(frozen=True)
class Forecast:
    model: str
    horizon: int
    values: list[dict]
    selection_scores: dict[str, float]
    metrics: dict[str, float | int | None] | None = None
    baseline_metrics: dict[str, float | int | None] | None = None
    input_quality: dict[str, float | int | str] | None = None
    interval_level: float | None = None
    disclaimer: str = "Forecasts are model estimates, not measurements."


def _seasonal_naive(values: np.ndarray, horizon: int, season: int = 24) -> np.ndarray:
    if len(values) < season:
        return np.repeat(values[-1], horizon)
    pattern = values[-season:]
    return np.resize(pattern, horizon)


def _linear_trend(values: np.ndarray, horizon: int) -> np.ndarray:
    window = values[-min(len(values), 168):]
    x = np.arange(len(window), dtype=float)
    slope, intercept = np.polyfit(x, window, 1)
    predicted = intercept + slope * np.arange(len(window), len(window) + horizon)
    return np.maximum(predicted, 0)


def _tsfm_api(values: np.ndarray, horizon: int) -> np.ndarray:
    endpoint = os.environ.get("AQA_TSFM_ENDPOINT")
    if not endpoint:
        raise ValidationError("AQA_TSFM_ENDPOINT is not configured")
    response = httpx.post(endpoint, timeout=180, json={"values": values.tolist(), "horizon": horizon, "frequency": "1h"})
    response.raise_for_status()
    predicted = np.asarray(response.json()["forecast"], dtype=float)
    if predicted.shape != (horizon,) or not np.isfinite(predicted).all():
        raise ValidationError("TSFM endpoint returned an invalid forecast")
    return np.maximum(predicted, 0)


MODELS: dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    "seasonal_naive": _seasonal_naive,
    "linear_trend": _linear_trend,
    "tsfm_api": _tsfm_api,
}


def forecast_series(
    rows: Iterable[dict], *, horizon: int = 24, model: str = "auto",
    candidates: tuple[str, ...] = ("seasonal_naive", "linear_trend", "tsfm_api"),
) -> Forecast:
    ordered = sorted(rows, key=lambda row: row["timestamp"])
    numeric = [(datetime.fromisoformat(row["timestamp"]), float(row["mean"])) for row in ordered if row.get("mean") is not None]
    if len(numeric) < 48:
        raise ValidationError("Forecasting requires at least 48 valid hourly values")
    timestamps, raw = zip(*numeric)
    values = np.asarray(raw, dtype=float)
    if horizon < 1 or horizon > 720:
        raise ValidationError("Forecast horizon must be between 1 and 720 hours")
    scores: dict[str, float] = {}
    validation = min(horizon, max(24, len(values) // 5))
    train, actual = values[:-validation], values[-validation:]
    for name in candidates:
        if name not in MODELS:
            continue
        try:
            predicted = MODELS[name](train, validation)
            scores[name] = float(np.mean(np.abs(predicted - actual)))
        except (ValidationError, httpx.HTTPError):
            continue
    selected = min(scores, key=scores.get) if model == "auto" else model
    if selected not in MODELS:
        raise ValidationError(f"Forecast model is unavailable: {selected}")
    predicted = MODELS[selected](values, horizon)
    start = timestamps[-1]
    output = [
        {"timestamp": (start + timedelta(hours=index + 1)).isoformat(), "forecast": round(float(value), 6)}
        for index, value in enumerate(predicted)
    ]
    return Forecast(selected, horizon, output, scores)


def forecast_at_issue(
    rows: Iterable[dict], *, issue_time: datetime, horizon: int = 24, context_hours: int = 168,
    maximum_filled_fraction: float = .20, minimum_observed_target_fraction: float = .75,
    model: str = "auto", candidates: tuple[str, ...] = ("seasonal_naive", "linear_trend", "tsfm_api"),
    interval_level: float = .90,
) -> Forecast:
    """Forecast one hourly series without using data after the issue time for training or model selection."""
    if horizon < 1 or horizon > 720:
        raise ValidationError("Forecast horizon must be between 1 and 720 hours")
    if context_hours < 48:
        raise ValidationError("Forecast context must contain at least 48 hours")
    ordered = sorted((_forecast_row(row) for row in rows), key=lambda row: row["timestamp"])
    by_time = {row["timestamp"]: row for row in ordered}
    expected_context = [issue_time - timedelta(hours=context_hours - 1 - index) for index in range(context_hours)]
    missing_context = [timestamp for timestamp in expected_context if timestamp not in by_time or by_time[timestamp]["mean"] is None]
    if missing_context:
        near_origin = max(missing_context)
        raise ValidationError(
            f"Forecast context is not continuous: {len(missing_context)} hourly value(s) are missing; "
            f"the latest missing hour is {near_origin.isoformat()}"
        )
    context = [by_time[timestamp] for timestamp in expected_context]
    valid_count = sum(row["valid_count"] for row in context)
    filled_count = sum(row["filled_count"] for row in context)
    filled_fraction = filled_count / valid_count if valid_count else 0.0
    if filled_fraction > maximum_filled_fraction:
        raise ValidationError(
            f"Forecast input is {filled_fraction:.1%} filled, above the configured {maximum_filled_fraction:.1%} maximum"
        )
    values = np.asarray([row["mean"] for row in context], dtype=float)
    validation = min(horizon, 24, max(1, context_hours - 48))
    train = values[:-validation]
    validation_rows = context[-validation:]
    validation_actual = np.asarray([
        row["observed_mean"] if row["observed_mean"] is not None else np.nan for row in validation_rows
    ], dtype=float)
    observed_mask = np.isfinite(validation_actual)
    if observed_mask.sum() < math.ceil(validation * minimum_observed_target_fraction):
        raise ValidationError("Model selection does not have enough directly observed validation targets")
    scores: dict[str, float] = {}
    predictions: dict[str, np.ndarray] = {}
    for name in candidates:
        if name not in MODELS:
            continue
        try:
            predicted = MODELS[name](train, validation)
            predictions[name] = predicted
            scores[name] = float(np.mean(np.abs(predicted[observed_mask] - validation_actual[observed_mask])))
        except (ValidationError, httpx.HTTPError, ValueError, np.linalg.LinAlgError):
            continue
    if not scores:
        raise ValidationError("No forecasting candidate produced an observed-target validation score")
    selected = min(scores, key=scores.get) if model == "auto" else model
    if selected not in MODELS or selected not in scores:
        raise ValidationError(f"Forecast model is unavailable or ineligible: {selected}")
    predicted = MODELS[selected](values, horizon)
    baseline = MODELS["seasonal_naive"](values, horizon)
    validation_residuals = validation_actual[observed_mask] - predictions[selected][observed_mask]
    radius = float(np.quantile(np.abs(validation_residuals), interval_level, method="higher"))
    lower = np.maximum(predicted - radius, 0)
    upper = predicted + radius
    expected_targets = [issue_time + timedelta(hours=index + 1) for index in range(horizon)]
    output = []
    for index, timestamp in enumerate(expected_targets):
        actual = by_time.get(timestamp, {}).get("observed_mean")
        output.append({
            "timestamp": timestamp.isoformat(), "forecast": round(float(predicted[index]), 6),
            "lower": round(float(lower[index]), 6), "upper": round(float(upper[index]), 6),
            "baseline_forecast": round(float(baseline[index]), 6),
            "observed": round(float(actual), 6) if actual is not None else None,
        })
    observed_targets = [row for row in output if row["observed"] is not None]
    metrics = baseline_metrics = None
    required_targets = math.ceil(horizon * minimum_observed_target_fraction)
    if len(observed_targets) >= required_targets:
        actual = np.asarray([row["observed"] for row in observed_targets], dtype=float)
        chosen = np.asarray([row["forecast"] for row in observed_targets], dtype=float)
        naive = np.asarray([row["baseline_forecast"] for row in observed_targets], dtype=float)
        scale = _mase_scale(context)
        intervals = np.asarray([
            row["lower"] <= row["observed"] <= row["upper"] for row in observed_targets
        ], dtype=float)
        metrics = _metrics(actual, chosen, scale, intervals)
        baseline_metrics = _metrics(actual, naive, scale, None)
    return Forecast(
        selected, horizon, output, scores, metrics, baseline_metrics,
        {
            "context_hours": context_hours, "valid_input_count": int(valid_count),
            "filled_input_count": int(filled_count), "filled_fraction": filled_fraction,
            "context_mean": float(np.mean(values)), "context_std": float(np.std(values)),
            "context_coefficient_of_variation": float(np.std(values) / np.mean(values)) if np.mean(values) else None,
            "maximum_filled_fraction": maximum_filled_fraction,
            "model_selection_target": "directly_observed_only",
        },
        interval_level,
    )


def rolling_backtest(
    rows: Iterable[dict], *, issue_times: Iterable[datetime], horizon: int = 24, context_hours: int = 168,
    maximum_filled_fraction: float = .20, minimum_observed_target_fraction: float = .75,
    model: str = "auto", candidates: tuple[str, ...] = ("seasonal_naive", "linear_trend", "tsfm_api"),
) -> dict:
    """Run leakage-safe rolling origins and aggregate observed-target errors."""
    origins = []
    for issue_time in issue_times:
        result = forecast_at_issue(
            rows, issue_time=issue_time, horizon=horizon, context_hours=context_hours,
            maximum_filled_fraction=maximum_filled_fraction,
            minimum_observed_target_fraction=minimum_observed_target_fraction,
            model=model, candidates=candidates,
        )
        origins.append({"issue_time": issue_time.isoformat(), "forecast": result})
    selected_rows = []
    baseline_rows = []
    scaled_errors = []
    coverages = []
    for origin in origins:
        forecast = origin["forecast"]
        for row in forecast.values:
            if row["observed"] is None:
                continue
            selected_rows.append((row["observed"], row["forecast"]))
            baseline_rows.append((row["observed"], row["baseline_forecast"]))
        if forecast.metrics and forecast.metrics.get("mase") is not None:
            scaled_errors.append((forecast.metrics["mase"], forecast.metrics["observed_target_count"]))
            coverages.append((forecast.metrics["interval_coverage"], forecast.metrics["observed_target_count"]))
    if not selected_rows:
        raise ValidationError("Rolling backtest has no directly observed targets")
    actual = np.asarray([row[0] for row in selected_rows]); predicted = np.asarray([row[1] for row in selected_rows])
    naive = np.asarray([row[1] for row in baseline_rows]); count = len(selected_rows)
    aggregate = _metrics(actual, predicted, None, None)
    baseline_aggregate = _metrics(actual, naive, None, None)
    if scaled_errors:
        aggregate["mase"] = sum(value * weight for value, weight in scaled_errors) / sum(weight for _, weight in scaled_errors)
        aggregate["interval_coverage"] = sum(value * weight for value, weight in coverages) / sum(weight for _, weight in coverages)
    return {
        "origin_count": len(origins), "observed_target_count": count,
        "metrics": aggregate, "baseline_metrics": baseline_aggregate,
        "origins": origins,
    }


def _forecast_row(row: dict) -> dict:
    timestamp = row["timestamp"]
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp)
    valid_count = int(row.get("valid_count", 1 if row.get("mean") is not None else 0) or 0)
    filled_count = int(row.get("filled_count", 0) or 0)
    return {
        "timestamp": timestamp, "mean": float(row["mean"]) if row.get("mean") is not None else None,
        "observed_mean": float(row.get("observed_mean", row.get("mean"))) if row.get("observed_mean", row.get("mean")) is not None else None,
        "valid_count": valid_count, "filled_count": filled_count,
    }


def _mase_scale(context: list[dict], season: int = 24) -> float | None:
    observed = np.asarray([
        row["observed_mean"] if row["observed_mean"] is not None else np.nan for row in context
    ], dtype=float)
    if len(observed) <= season:
        return None
    left, right = observed[:-season], observed[season:]
    mask = np.isfinite(left) & np.isfinite(right)
    if not mask.any():
        return None
    scale = float(np.mean(np.abs(right[mask] - left[mask])))
    return scale if scale > 0 else None


def _metrics(actual: np.ndarray, predicted: np.ndarray, scale: float | None,
             interval_hits: np.ndarray | None) -> dict[str, float | int | None]:
    error = predicted - actual
    mae = float(np.mean(np.abs(error)))
    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    smape_terms = np.divide(np.abs(error), denominator, out=np.zeros_like(error), where=denominator > 0)
    return {
        "mae": mae, "rmse": float(np.sqrt(np.mean(error ** 2))), "mase": mae / scale if scale else None,
        "smape": float(np.mean(smape_terms)) * 100,
        "bias": float(np.mean(error)),
        "interval_coverage": float(np.mean(interval_hits)) if interval_hits is not None else None,
        "observed_target_count": int(len(actual)),
    }
