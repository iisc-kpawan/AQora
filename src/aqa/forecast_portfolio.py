from __future__ import annotations

import math
import time
from datetime import datetime
from typing import Any

import numpy as np

from .capabilities import CapabilityAssessment
from .errors import AQAError
from .forecast_analytics import ForecastAnalytics


class ForecastPortfolioAnalytics:
    """Reproducible multi-city readiness audits and leakage-safe rolling backtests."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.single = ForecastAnalytics(config)
        settings = self.single.config["forecasting"]
        self.cities = tuple(settings.get("portfolio_cities", ()))
        self.pollutants = tuple(settings.get("portfolio_pollutants", ("PM2.5",)))
        self.replicates = int(settings.get("bootstrap_replicates", 1000))
        self.seed = int(settings.get("bootstrap_seed", 20260901))

    @staticmethod
    def should_handle(question: str, assessment: CapabilityAssessment) -> bool:
        text = question.casefold()
        return (
            "suitable for 24-hour forecasting evaluation" in text
            or ("rolling 24-hour" in text and "eligible cities" in text and "seasonal naive" in text)
            or "fails to beat seasonal naive" in text
        )

    def execute(self, question: str, assessment: CapabilityAssessment) -> dict[str, Any]:
        del assessment
        if "suitable for 24-hour forecasting evaluation" in question.casefold():
            return self._readiness()
        if "fails to beat seasonal naive" in question.casefold():
            return self._failure_diagnostics()
        return self._rolling_pm25()

    def _readiness(self) -> dict[str, Any]:
        started = time.perf_counter()
        rows: list[dict[str, Any]] = []
        for candidate in self.cities:
            for pollutant in self.pollutants:
                result, reason = self._evaluate_one(candidate["dataset"], candidate["city"], pollutant)
                if result:
                    analysis = result["analysis"]
                    rows.append({
                        "dataset": candidate["dataset"], "city": candidate["city"], "pollutant": pollutant,
                        "eligible": True, "reason": None,
                        "origin_count": analysis["origin_count"],
                        "observed_target_count": analysis["metrics"]["observed_target_count"],
                        "maximum_input_filled_fraction": max(row["filled_fraction"] for row in result["data"]),
                        "station_count": result["metadata"]["station_count"],
                    })
                else:
                    rows.append({
                        "dataset": candidate["dataset"], "city": candidate["city"], "pollutant": pollutant,
                        "eligible": False, "reason": reason, "origin_count": 0,
                        "observed_target_count": 0, "maximum_input_filled_fraction": None,
                        "station_count": None,
                    })
        eligible = [row for row in rows if row["eligible"]]
        answer = (
            f"{len(eligible)} of {len(rows)} configured city–pollutant candidates are suitable for a leakage-safe "
            "24-hour rolling evaluation over 15–21 August 2026. Eligibility requires a continuous 168-hour "
            "context at every daily origin, no more than 20% filled input values, and at least 75% directly "
            "observed target hours. The table includes every candidate and the exact exclusion reason."
        )
        return self._result(
            rows, started=started, operator="forecast_readiness_portfolio_v1", answer=answer,
            title="24-hour forecast evaluation readiness",
            metrics=["eligible", "origin_count", "observed_target_count", "maximum_input_filled_fraction"],
            analysis={"eligible_count": len(eligible), "candidate_count": len(rows)},
        )

    def _rolling_pm25(self) -> dict[str, Any]:
        started = time.perf_counter()
        rows: list[dict[str, Any]] = []
        origin_mase: list[float] = []
        city_mase: list[float] = []
        total_targets = 0
        weighted_mase = 0.0
        wins = 0
        for candidate in self.cities:
            result, reason = self._evaluate_one(candidate["dataset"], candidate["city"], "PM2.5")
            if not result:
                rows.append({
                    "dataset": candidate["dataset"], "city": candidate["city"], "pollutant": "PM2.5",
                    "eligible": False, "reason": reason, "origin_count": 0, "observed_target_count": 0,
                    "model": None, "mae": None, "rmse": None, "mase": None, "mase_ci_lower": None,
                    "mase_ci_upper": None, "seasonal_naive_mae": None, "mae_improvement": None,
                    "beats_seasonal_naive": None, "mean_input_filled_fraction": None,
                    "mean_context_level": None, "mean_context_volatility": None,
                    "mean_context_coefficient_of_variation": None,
                })
                continue
            metrics = result["analysis"]["metrics"]
            baseline = result["analysis"]["baseline_metrics"]
            valid_origin_mase = [float(row["mase"]) for row in result["data"] if row.get("mase") is not None]
            ci_lower, ci_upper = _bootstrap_mean_interval(valid_origin_mase, self.replicates, self.seed)
            count = int(metrics["observed_target_count"])
            mase = float(metrics["mase"])
            improvement = float(baseline["mae"] - metrics["mae"])
            won = improvement > 0
            origin_mase.extend(valid_origin_mase)
            city_mase.append(mase)
            total_targets += count
            weighted_mase += mase * count
            wins += int(won)
            models = sorted({row["model"] for row in result["data"]})
            rows.append({
                "dataset": candidate["dataset"], "city": candidate["city"], "pollutant": "PM2.5",
                "eligible": True, "reason": None, "origin_count": result["analysis"]["origin_count"],
                "observed_target_count": count, "model": ", ".join(models),
                "mae": metrics["mae"], "rmse": metrics["rmse"], "mase": mase,
                "mase_ci_lower": ci_lower, "mase_ci_upper": ci_upper,
                "seasonal_naive_mae": baseline["mae"], "mae_improvement": improvement,
                "beats_seasonal_naive": won,
                "mean_input_filled_fraction": float(np.mean([row["filled_fraction"] for row in result["data"]])),
                "mean_context_level": float(np.mean([row["context_mean"] for row in result["data"]])),
                "mean_context_volatility": float(np.mean([row["context_std"] for row in result["data"]])),
                "mean_context_coefficient_of_variation": float(np.mean([
                    row["context_coefficient_of_variation"] for row in result["data"]
                    if row["context_coefficient_of_variation"] is not None
                ])),
            })
        eligible = [row for row in rows if row["eligible"]]
        if not eligible:
            answer = "No configured city passed every rolling-origin forecast-quality requirement for this window."
            aggregate = {"eligible_city_count": 0, "candidate_city_count": len(rows)}
        else:
            portfolio_lower, portfolio_upper = _bootstrap_mean_interval(origin_mase, self.replicates, self.seed)
            equal_city_mase = float(np.mean(city_mase))
            target_weighted_mase = weighted_mase / total_targets
            win_rate = wins / len(eligible)
            aggregate = {
                "eligible_city_count": len(eligible), "candidate_city_count": len(rows),
                "equal_city_mase": equal_city_mase, "observed_target_weighted_mase": target_weighted_mase,
                "origin_level_mase_ci_lower": portfolio_lower, "origin_level_mase_ci_upper": portfolio_upper,
                "win_count_vs_seasonal_naive": wins, "win_rate_vs_seasonal_naive": win_rate,
                "observed_target_count": total_targets,
            }
            answer = (
                f"{len(eligible)} of {len(rows)} configured cities passed all seven daily rolling origins. "
                f"Their equal-city mean MASE is {equal_city_mase:.2f}; the observed-target-weighted MASE is "
                f"{target_weighted_mase:.2f}, with an origin-level bootstrap interval of "
                f"{portfolio_lower:.2f}–{portfolio_upper:.2f}. The configured model beat seasonal naive in "
                f"{wins} of {len(eligible)} cities ({win_rate:.0%}). Excluded cities are retained in the table "
                "with their data-quality reason; no city or target was substituted."
            )
        return self._result(
            rows, started=started, operator="rolling_forecast_portfolio_v1", answer=answer,
            title="PM2.5 · multi-city rolling 24-hour backtest",
            metrics=["mae", "rmse", "mase", "mase_ci_lower", "mase_ci_upper", "seasonal_naive_mae"],
            analysis=aggregate,
        )

    def _failure_diagnostics(self) -> dict[str, Any]:
        result = self._rolling_pm25()
        eligible = [row for row in result["data"] if row["eligible"]]
        failures = [row for row in eligible if not row["beats_seasonal_naive"]]
        associations: dict[str, float | None] = {}
        reason = None
        if len(eligible) < 3 or not failures or len(failures) == len(eligible):
            associations = {"missingness": None, "volatility": None, "pollutant_level": None}
            reason = (
                "association is not estimable: at least three eligible cities and both outcome groups "
                "are required for a descriptive point-biserial correlation"
            )
        else:
            outcome = np.asarray([not row["beats_seasonal_naive"] for row in eligible], dtype=float)
            for name, field in (
                ("missingness", "mean_input_filled_fraction"),
                ("volatility", "mean_context_volatility"),
                ("pollutant_level", "mean_context_level"),
            ):
                values = np.asarray([row[field] for row in eligible], dtype=float)
                associations[name] = float(np.corrcoef(outcome, values)[0, 1]) if np.std(values) else None
        names = ", ".join(row["city"] for row in failures) or "No eligible city"
        result["analysis"]["operator"] = "forecast_failure_diagnostics_v1"
        result["analysis"]["failure_city_count"] = len(failures)
        result["analysis"]["failure_cities"] = [row["city"] for row in failures]
        result["analysis"]["association_method"] = "descriptive point-biserial correlation"
        result["analysis"]["associations"] = associations
        result["analysis"]["association_limitation"] = reason
        result["query_spec"]["operation"] = "forecast_failure_diagnostics_v1"
        result["query_spec"]["metrics"].extend([
            "mean_input_filled_fraction", "mean_context_volatility", "mean_context_level",
        ])
        result["display_title"] = "PM2.5 · forecast failure diagnostics"
        result["answer"] = (
            f"{names} failed to beat seasonal naive among the {len(eligible)} eligible cities. "
            "The table reports each city's filled-input fraction, context volatility, and mean pollutant level. "
        )
        if reason:
            result["answer"] += (
                f"A reliable association test cannot be calculated from only {len(eligible)} eligible cities; "
                "the correlation fields are therefore left empty rather than presenting a spurious result. "
            )
        else:
            result["answer"] += "The analysis section reports descriptive point-biserial associations. "
        excluded = len(result["data"]) - len(eligible)
        if excluded:
            result["answer"] += f"{excluded} configured cities were excluded by the predeclared data-quality rules."
        return result

    def _evaluate_one(self, dataset: str, city: str, pollutant: str) -> tuple[dict[str, Any] | None, str | None]:
        available_end = self.single.catalog.by_dataset(dataset).available_end
        if available_end < datetime(2026, 8, 22, 23):
            return None, (
                "recent-data criterion failed: "
                f"{dataset} is available through {available_end:%d %b %Y %H:%M}, before the evaluation targets"
            )
        question = (
            f"Backtest {city} {pollutant} 24-hour forecasts at daily issue times from 15–21 August 2026 "
            "and report aggregate MAE, RMSE, MASE, and interval coverage."
        )
        assessment = CapabilityAssessment(intent="forecast", supported_datasets=[dataset])
        try:
            return self.single.execute(question, assessment), None
        except AQAError as exc:
            return None, exc.message

    def _result(self, rows: list[dict[str, Any]], *, started: float, operator: str, answer: str,
                title: str, metrics: list[str], analysis: dict[str, Any]) -> dict[str, Any]:
        settings = self.single.config["forecasting"]
        return {
            "kind": "analytics",
            "query_spec": {
                "dataset": "configured_portfolio", "operation": operator, "geography_level": "city",
                "locations": [item["city"] for item in self.cities], "pollutants": sorted({row["pollutant"] for row in rows}),
                "time_range": {"start": "2026-08-15", "end": "2026-08-21"},
                "resolution": "daily_origins", "metrics": metrics,
                "filters": {
                    "context_hours": int(settings.get("context_hours", 168)),
                    "maximum_input_filled_fraction": float(settings.get("maximum_input_filled_fraction", .20)),
                    "minimum_observed_target_fraction": float(settings.get("minimum_observed_target_fraction", .75)),
                    "observed_targets_only": True,
                },
            },
            "resolved_period": {
                "start": "2026-08-15", "end": "2026-08-21",
                "basis": "daily 23:00 issue times in each city's configured local timezone",
            },
            "data": rows,
            "metadata": {
                "dataset": "configured_portfolio", "geography_level": "city", "station_balanced": True,
                "value_status": "model_estimate", "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "warnings": [],
            },
            "visualization": {"type": "table"},
            "analysis": {
                "operator": operator, "future_observations_used_for_training": False,
                "evaluation_targets": "directly_observed_only", "portfolio_defined_in_config": True,
                "bootstrap_replicates": self.replicates, "bootstrap_seed": self.seed, **analysis,
            },
            "answer": answer, "display_title": title,
            "model_trace": {"provider": "deterministic", "model": operator, "latency_ms": 0,
                            "prompt_tokens": None, "completion_tokens": None, "fallback_used": False},
        }


def _bootstrap_mean_interval(values: list[float], replicates: int, seed: int) -> tuple[float | None, float | None]:
    numeric = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if not len(numeric):
        return None, None
    generator = np.random.default_rng(seed)
    samples = generator.choice(numeric, size=(replicates, len(numeric)), replace=True).mean(axis=1)
    lower, upper = np.quantile(samples, [.025, .975])
    return float(lower), float(upper)
