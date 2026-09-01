from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

from .agent import Planner
from .analytics import AnalyticsEngine
from .queryspec import QuerySpec


PLAN_FIELDS = ("dataset", "operation", "geography_level", "locations", "pollutants", "time_range", "resolution", "metrics", "filters")


def evaluate_file(path: str | Path, *, provider: str | None = None, model: str | None = None) -> dict[str, Any]:
    cases = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    providers = (provider or "deterministic").split(",")
    models = (model or "rule_planner_v1").split(",")
    if len(models) == 1 and len(providers) > 1:
        models *= len(providers)
    runs = []
    for selected_provider, selected_model in zip(providers, models):
        runs.append(_evaluate(cases, selected_provider.strip(), selected_model.strip()))
    return {"cases": len(cases), "runs": runs, "manual_review_fields": ["answer_correctness_1_5", "scientific_transparency_1_5", "usefulness_1_5", "notes"]}


def _evaluate(cases: list[dict], provider: str, model: str) -> dict:
    planner = Planner()
    engine = AnalyticsEngine()
    records = []
    for case in cases:
        started = time.perf_counter()
        error = None
        try:
            if provider == "deterministic":
                from .agent import rule_based_plan
                actual = rule_based_plan(case["question"])
                trace = None
            else:
                actual, trace = planner.plan(case["question"], provider=provider, model=model)
            expected = QuerySpec.model_validate(case["ground_truth_queryspec"])
            actual_json, expected_json = actual.model_dump(mode="json"), expected.model_dump(mode="json")
            field_scores = {field: actual_json[field] == expected_json[field] for field in PLAN_FIELDS}
            execution_ok = True
            numeric_ok = None
            if case.get("execute", False):
                result = engine.execute(actual)
                if "ground_truth_value" in case:
                    metric = case.get("ground_truth_metric", "mean")
                    tolerance = case.get("tolerance", 1e-6)
                    numeric_ok = abs(float(result["data"][0][metric]) - float(case["ground_truth_value"])) <= tolerance
            records.append({
                "id": case["id"], "valid": True, "exact_plan": all(field_scores.values()),
                "field_scores": field_scores, "execution_ok": execution_ok,
                "numeric_ok": numeric_ok, "latency_ms": (time.perf_counter() - started) * 1000,
                "prompt_tokens": getattr(trace, "prompt_tokens", None), "completion_tokens": getattr(trace, "completion_tokens", None),
            })
        except Exception as exc:
            error = str(exc)
            records.append({"id": case["id"], "valid": False, "exact_plan": False, "field_scores": {}, "execution_ok": False, "numeric_ok": False, "latency_ms": (time.perf_counter() - started) * 1000, "error": error})
    valid = [item for item in records if item["valid"]]
    latencies = [item["latency_ms"] for item in records]
    tokens = [sum(value for value in (item.get("prompt_tokens"), item.get("completion_tokens")) if value is not None) for item in records]
    field_accuracy = {field: sum(bool(item.get("field_scores", {}).get(field)) for item in records) / len(records) for field in PLAN_FIELDS}
    mesa = {
        "M_meaning_and_entities": statistics.mean([field_accuracy["dataset"], field_accuracy["geography_level"], field_accuracy["locations"], field_accuracy["pollutants"]]),
        "E_execution_validity": sum(item["execution_ok"] for item in records) / len(records),
        "S_scientific_specification": statistics.mean([field_accuracy["time_range"], field_accuracy["resolution"], field_accuracy["filters"]]),
        "A_analytic_alignment": field_accuracy["metrics"],
    }
    return {
        "provider": provider, "model": model, "valid_queryspec_rate": len(valid) / len(records),
        "exact_plan_rate": sum(item["exact_plan"] for item in records) / len(records),
        "field_accuracy": field_accuracy, "mesa": mesa,
        "latency_ms": {"mean": statistics.mean(latencies), "p50": statistics.median(latencies), "p95": sorted(latencies)[max(0, int(len(latencies) * .95) - 1)]},
        "tokens": {"total": sum(tokens), "mean": statistics.mean(tokens) if tokens else 0},
        "records": records,
    }
