from __future__ import annotations

import json
import os
import re
import socket
import time
import calendar
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx

from .analytics import AnalyticsEngine
from .capabilities import CapabilityAssessment, assess_question, available_bounds
from .config import load_config
from .errors import AQAError, ValidationError
from .metadata import MetadataService
from .global_analytics import GlobalAnalytics
from .quality_analytics import QualityAnalytics
from .threshold_analytics import ThresholdAnalytics
from .aqi_analytics import AQIAnalytics
from .temporal_analytics import TemporalAnalytics
from .composite_analytics import CompositeAnalytics
from .forecast_analytics import ForecastAnalytics
from .forecast_portfolio import ForecastPortfolioAnalytics
from .provenance_analytics import ProvenanceAnalytics
from .uncertainty_analytics import UncertaintyAnalytics
from .visualization_analytics import VisualizationAnalytics
from .simulation_analytics import SimulationAnalytics
from .cohort_analytics import CohortAnalytics
from .derived_temporal_analytics import DerivedTemporalAnalytics
from .advanced_aqi_analytics import AdvancedAQIAnalytics
from .compound_analytics import CompoundAnalytics
from .burden_episode_analytics import BurdenEpisodeAnalytics
from .gap_statistics_analytics import GapStatisticsAnalytics
from .queryspec import Dataset, QuerySpec, TimeRange


SYSTEM_PROMPT = """You plan air-quality analytics. Return only a JSON object valid for the supplied QuerySpec schema. Never calculate measurements, emit SQL, use file paths, or invent entities. Relative periods must remain phrases so deterministic code resolves them against data availability."""


@dataclass
class ModelTrace:
    provider: str
    model: str
    latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    fallback_used: bool


class Planner:
    def __init__(self, config: dict | None = None):
        self.config = config or load_config()

    def plan(self, question: str, *, provider: str | None = None, model: str | None = None) -> tuple[QuerySpec, ModelTrace]:
        provider = provider or self.config["llm"]["default_provider"]
        model = model or self.config["llm"]["default_model"]
        started = time.perf_counter()
        try:
            if provider == "ollama":
                spec, usage = self._ollama(question, model)
            elif provider == "openai_compatible":
                spec, usage = self._openai_compatible(question, model)
            else:
                raise ValidationError(f"Unsupported LLM provider: {provider}")
            trace = ModelTrace(provider, model, (time.perf_counter() - started) * 1000, usage.get("prompt_tokens"), usage.get("completion_tokens"), False)
            return spec, trace
        except Exception:
            if not self.config["llm"].get("allow_provider_fallback", True):
                raise
            spec = rule_based_plan(question)
            trace = ModelTrace("deterministic", "rule_planner_v1", (time.perf_counter() - started) * 1000, None, None, True)
            return spec, trace

    def _ollama(self, question: str, model: str) -> tuple[QuerySpec, dict]:
        url = self.config["llm"]["ollama_base_url"].rstrip("/") + "/api/chat"
        parsed = httpx.URL(url)
        try:
            with socket.create_connection((parsed.host, parsed.port or 80), timeout=0.25):
                pass
        except OSError as exc:
            raise ConnectionError(f"Ollama is unavailable at {parsed.host}:{parsed.port}") from exc
        timeout = httpx.Timeout(float(self.config["llm"]["request_timeout_seconds"]), connect=1.0)
        with httpx.Client(timeout=timeout, trust_env=False) as client:
            response = client.post(url, json={
                "model": model, "stream": False, "format": QuerySpec.model_json_schema(),
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}],
                "options": {"temperature": 0},
            })
        response.raise_for_status()
        payload = response.json()
        spec = QuerySpec.model_validate_json(payload["message"]["content"])
        return spec, {"prompt_tokens": payload.get("prompt_eval_count"), "completion_tokens": payload.get("eval_count")}

    def _openai_compatible(self, question: str, model: str) -> tuple[QuerySpec, dict]:
        base = os.environ.get("AQA_LLM_BASE_URL")
        key = os.environ.get("AQA_LLM_API_KEY")
        if not base or not key:
            raise ValidationError("AQA_LLM_BASE_URL and AQA_LLM_API_KEY are required")
        response = httpx.post(base.rstrip("/") + "/chat/completions", timeout=self.config["llm"]["request_timeout_seconds"], headers={"Authorization": f"Bearer {key}"}, json={
            "model": model, "temperature": 0,
            "response_format": {"type": "json_schema", "json_schema": {"name": "QuerySpec", "strict": True, "schema": QuerySpec.model_json_schema()}},
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}],
        })
        response.raise_for_status()
        payload = response.json()
        spec = QuerySpec.model_validate_json(payload["choices"][0]["message"]["content"])
        usage = payload.get("usage", {})
        return spec, {"prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens")}


class ConversationalAgent:
    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.planner = Planner(self.config)
        self.analytics = AnalyticsEngine(self.config)
        self.metadata = MetadataService(self.analytics.catalog, self.config)
        self.global_analytics = GlobalAnalytics(self.config)
        self.quality_analytics = QualityAnalytics(self.config)
        self.threshold_analytics = ThresholdAnalytics(self.config)
        self.aqi_analytics = AQIAnalytics(self.config)
        self.temporal_analytics = TemporalAnalytics(self.config)
        self.composite_analytics = CompositeAnalytics(self.config)
        self.forecast_analytics = ForecastAnalytics(self.config)
        self.forecast_portfolio = ForecastPortfolioAnalytics(self.config)
        self.provenance_analytics = ProvenanceAnalytics(self.config)
        self.uncertainty_analytics = UncertaintyAnalytics(self.config)
        self.visualization_analytics = VisualizationAnalytics(self.config)
        self.simulation_analytics = SimulationAnalytics(self.config)
        self.cohort_analytics = CohortAnalytics(self.config)
        self.derived_temporal_analytics = DerivedTemporalAnalytics(self.config)
        self.advanced_aqi_analytics = AdvancedAQIAnalytics(self.config)
        self.compound_analytics = CompoundAnalytics(self.config)
        self.burden_episode_analytics = BurdenEpisodeAnalytics(self.config)
        self.gap_statistics_analytics = GapStatisticsAnalytics(self.config)

    def ask(
        self, question: str, *, provider: str | None = None, model: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        assessment = assess_question(question, self.config)
        inherited_location_note = None
        if assessment.location_ambiguous:
            candidate_locations = (context or {}).get("locations") or []
            if len(candidate_locations) == 1:
                inherited_location = candidate_locations[0]
                augmented_question = f"For {inherited_location}: {question}"
                reassessed = assess_question(augmented_question, self.config)
                if not reassessed.location_ambiguous:
                    question, assessment = augmented_question, reassessed
                    inherited_location_note = (
                        f"(Continuing about {inherited_location} from your previous question, "
                        "since this one didn't name a location.) "
                    )
        result = self._ask_dispatch(question, assessment, provider=provider, model=model)
        if inherited_location_note and result.get("kind") != "limitation":
            result["answer"] = inherited_location_note + result["answer"]
        return result

    def _ask_dispatch(
        self, question: str, assessment: CapabilityAssessment, *,
        provider: str | None = None, model: str | None = None,
    ) -> dict[str, Any]:
        metadata_period_audit = (
            assessment.intent == "metadata" and assessment.unavailable_periods
            and not assessment.unsupported_places and not assessment.unsupported_methods
            and "provenance" in question.casefold()
        )
        if not assessment.can_execute and not metadata_period_audit:
            return limitation_result(question, assessment, self.config)
        datasets = assessment.supported_datasets
        if self.burden_episode_analytics.should_handle(question, assessment):
            try:
                return self.burden_episode_analytics.execute(question, assessment)
            except AQAError as exc:
                return limitation_result(question, assessment, self.config, extra_reason=exc.message)
        if self.gap_statistics_analytics.should_handle(question, assessment):
            try:
                return self.gap_statistics_analytics.execute(question, assessment)
            except AQAError as exc:
                return limitation_result(question, assessment, self.config, extra_reason=exc.message)
        if self.compound_analytics.should_handle(question, assessment):
            try:
                return self.compound_analytics.execute(question, assessment)
            except AQAError as exc:
                return limitation_result(question, assessment, self.config, extra_reason=exc.message)
        if self.advanced_aqi_analytics.should_handle(question, assessment):
            try:
                return self.advanced_aqi_analytics.execute(question, assessment)
            except AQAError as exc:
                return limitation_result(question, assessment, self.config, extra_reason=exc.message)
        if self.cohort_analytics.should_handle(question, assessment):
            try:
                return self.cohort_analytics.execute(question, assessment)
            except AQAError as exc:
                return limitation_result(question, assessment, self.config, extra_reason=exc.message)
        if self.derived_temporal_analytics.should_handle(question, assessment):
            try:
                return self.derived_temporal_analytics.execute(question, assessment)
            except AQAError as exc:
                return limitation_result(question, assessment, self.config, extra_reason=exc.message)
        if self.simulation_analytics.should_handle(question, assessment):
            try:
                return self.simulation_analytics.execute(question, assessment)
            except AQAError as exc:
                return limitation_result(question, assessment, self.config, extra_reason=exc.message)
        if self.visualization_analytics.should_handle(question, assessment):
            try:
                return self.visualization_analytics.execute(question, assessment)
            except AQAError as exc:
                return limitation_result(question, assessment, self.config, extra_reason=exc.message)
        if self.forecast_portfolio.should_handle(question, assessment):
            try:
                return self.forecast_portfolio.execute(question, assessment)
            except AQAError as exc:
                return limitation_result(question, assessment, self.config, extra_reason=exc.message)
        if self.composite_analytics.should_handle(question, assessment):
            try:
                return self.composite_analytics.execute(question, assessment)
            except AQAError as exc:
                return limitation_result(question, assessment, self.config, extra_reason=exc.message)
        if self.provenance_analytics.should_handle(question, assessment):
            try:
                return self.provenance_analytics.execute(question, assessment)
            except AQAError as exc:
                return limitation_result(question, assessment, self.config, extra_reason=exc.message)
        if self.uncertainty_analytics.should_handle(question, assessment):
            try:
                return self.uncertainty_analytics.execute(question, assessment)
            except AQAError as exc:
                return limitation_result(question, assessment, self.config, extra_reason=exc.message)
        if (("weekday" in question.casefold() and "weekend" in question.casefold())
                or ("daytime" in question.casefold() and "nighttime" in question.casefold())
                or "jointly observed" in question.casefold()):
            try:
                return self.temporal_analytics.execute(question, assessment)
            except AQAError as exc:
                return limitation_result(question, assessment, self.config, extra_reason=exc.message)
        if self.quality_analytics.should_handle(question, assessment):
            try:
                return self.quality_analytics.execute(question, assessment)
            except AQAError as exc:
                return limitation_result(question, assessment, self.config, extra_reason=exc.message)
        if assessment.intent == "metadata":
            text = question.casefold()
            if any(term in text for term in ("countries", "all data", "all source", "availability matrix")):
                datasets = [item.dataset for item in self.metadata.lakehouse.datasets]
            result = self.metadata.answer(question, datasets)
            if assessment.unavailable_periods:
                result["metadata"].setdefault("warnings", []).extend(assessment.unavailable_periods)
                result["answer"] += " Requested-period limitation: " + "; ".join(assessment.unavailable_periods) + "."
            result["model_trace"] = ModelTrace("deterministic", "metadata_v1", 0, None, None, False).__dict__
            return result
        if self.threshold_analytics.should_handle(question, assessment):
            try:
                return self.threshold_analytics.execute(question, assessment)
            except AQAError as exc:
                return limitation_result(question, assessment, self.config, extra_reason=exc.message)
        if self.aqi_analytics.should_handle(question, assessment):
            try:
                return self.aqi_analytics.execute(question, assessment)
            except AQAError as exc:
                return limitation_result(question, assessment, self.config, extra_reason=exc.message)
        if self.temporal_analytics.should_handle(question, assessment):
            try:
                return self.temporal_analytics.execute(question, assessment)
            except AQAError as exc:
                return limitation_result(question, assessment, self.config, extra_reason=exc.message)
        if self.forecast_analytics.should_handle(question, assessment):
            try:
                return self.forecast_analytics.execute(question, assessment)
            except AQAError as exc:
                return limitation_result(question, assessment, self.config, extra_reason=exc.message)
        if self.global_analytics.should_handle(question, assessment):
            try:
                return self.global_analytics.execute(question, assessment)
            except AQAError as exc:
                return limitation_result(question, assessment, self.config, extra_reason=exc.message)
        try:
            spec, trace = self.planner.plan(question, provider=provider, model=model)
        except AQAError as exc:
            return limitation_result(question, assessment, self.config, extra_reason=exc.message)
        except Exception as exc:
            return limitation_result(question, assessment, self.config, extra_reason=f"The request could not be represented safely: {exc}")
        text = question.casefold()
        if "aqi" in text or (any(word in text for word in ("highest", "top", "rank")) and any(word in text for word in ("city", "cities"))):
            guarded = rule_based_plan(question)
            if guarded != spec:
                spec = guarded
                trace = ModelTrace("deterministic_guardrail", "rule_planner_v2", trace.latency_ms, trace.prompt_tokens, trace.completion_tokens, True)
        confident_time_range = _confident_time_range_from_question(text)
        if confident_time_range:
            # A model can hallucinate dates for a relative phrase, pick the wrong
            # phrase entirely (e.g. "last month" for a "last year" question), or
            # simply misread an explicit date the question names outright (e.g.
            # "March 2026"); whatever the question's own wording deterministically
            # implies always wins over the model's guess.
            deterministic_range = TimeRange.model_validate(confident_time_range)
            if deterministic_range != spec.time_range:
                spec = spec.model_copy(update={"time_range": deterministic_range})
                trace = ModelTrace("deterministic_guardrail", "time_range_guardrail_v1", trace.latency_ms, trace.prompt_tokens, trace.completion_tokens, True)
        expected_dataset = _expected_dataset_from_text(text)
        if expected_dataset and spec.dataset.value != expected_dataset:
            # A model can name the right city but still pick the wrong dataset
            # enum (e.g. CPCB for a Beijing question), which then fails to
            # resolve the city at all; the country actually named in the
            # question always wins over the model's own dataset choice.
            spec = spec.model_copy(update={"dataset": Dataset(expected_dataset)})
            trace = ModelTrace("deterministic_guardrail", "dataset_country_guardrail_v1", trace.latency_ms, trace.prompt_tokens, trace.completion_tokens, True)
        expected_pollutants = _pollutants_from_text(text)
        if expected_pollutants and spec.pollutants != expected_pollutants:
            # A model can garble a pollutant mention entirely (e.g. "ozone
            # levels" planned as the literal pollutant "ozone_levels", which
            # fails to resolve at all), pick a different one than the question
            # named, or just leave it in a non-canonical form (e.g. "OZONE"
            # instead of "O3") that computes correctly but displays inconsistently;
            # the pollutant(s) actually named in the question always win, in
            # their canonical form, over the model's own output.
            spec = spec.model_copy(update={"pollutants": expected_pollutants})
            trace = ModelTrace("deterministic_guardrail", "pollutant_guardrail_v1", trace.latency_ms, trace.prompt_tokens, trace.completion_tokens, True)
        try:
            result = self.analytics.execute(spec)
        except AQAError as exc:
            return limitation_result(question, assessment, self.config, extra_reason=exc.message)
        result["answer"] = explain_result(question, result)
        result["kind"] = "analytics"
        result["model_trace"] = trace.__dict__
        return result


def limitation_result(
    question: str,
    assessment: CapabilityAssessment,
    config: dict,
    *,
    extra_reason: str | None = None,
) -> dict[str, Any]:
    reasons = []
    if assessment.location_ambiguous:
        reasons.append(
            "No country or city was named in the question, so I can't determine which dataset to use. "
            "This system does not carry context between questions -- please name a location explicitly "
            "(e.g. \"Delhi\", \"China\", \"United Kingdom\") in every question that needs one."
        )
    if assessment.unsupported_places:
        reasons.append("The requested geography is not loaded: " + ", ".join(assessment.unsupported_places) + ".")
    if assessment.unavailable_periods:
        reasons.extend(item + "." for item in assessment.unavailable_periods)
    if assessment.unsupported_methods:
        reasons.append("This build does not yet implement: " + ", ".join(assessment.unsupported_methods) + ".")
    if extra_reason:
        reasons.append(extra_reason.rstrip(".") + ".")
    from .lakehouse_catalog import LakehouseCatalog
    lakehouse = LakehouseCatalog(config)
    requested = set(assessment.supported_datasets)
    selected = [item for item in lakehouse.datasets if not requested or item.dataset in requested]
    if len(selected) > 8:
        scope = [f"{len(selected)} loaded country datasets; exact per-country bounds are available from the metadata query"]
    else:
        scope = [f"{item.country_name} ({item.dataset}): {item.available_start:%d %b %Y}–{item.available_end:%d %b %Y %H:%M}" for item in selected]
    answer = "I can’t calculate a reliable answer from the currently loaded data. " + " ".join(reasons)
    answer += " Loaded scope: " + "; ".join(scope) + ". I have not substituted another country, date, or method."
    return {
        "kind": "limitation",
        "query_spec": {"operation": "limitation", "datasets": assessment.supported_datasets},
        "data": [],
        "metadata": {
            "dataset": "unavailable", "station_count": 0, "value_status": "unavailable",
            "latency_ms": 0, "warnings": reasons,
        },
        "resolved_period": {}, "visualization": {"type": "none"},
        "analysis": {"operation": "limitation", "intent": assessment.intent, "reasons": reasons},
        "answer": answer, "display_title": "Request outside the loaded analytical scope",
        "model_trace": ModelTrace("deterministic", "capability_guard_v1", 0, None, None, False).__dict__,
    }


def rule_based_plan(question: str) -> QuerySpec:
    text = question.casefold()
    has_china = any(word in text for word in ("china", "beijing", "shanghai", "guangzhou", "shenzhen", "chinese"))
    has_india = any(word in text for word in ("india", "delhi", "mumbai", "bengaluru", "bangalore", "chennai", "kolkata", "indian"))
    dataset = "ALL" if has_china and has_india else "CNEMC" if has_china else "CPCB"
    locations = ["global"]
    level = "country"
    known = {
        "delhi": "Delhi", "bengaluru": "Bengaluru", "bangalore": "Bengaluru",
        "mumbai": "Mumbai", "kolkata": "Kolkata", "chennai": "Chennai",
        "beijing": "Beijing", "shanghai": "Shanghai", "guangzhou": "Guangzhou",
    }
    matched = [name for token, name in known.items() if token in text]
    if matched:
        locations, level = matched, "city"
    if level == "country":
        locations = ["India", "China"] if dataset == "ALL" else ["China" if dataset == "CNEMC" else "India"]
    pollutants = _pollutants_from_text(text) or []
    asks_aqi = bool(re.search(r"\baqi\b|air quality index", text))
    if asks_aqi:
        pollutants = ["PM25", "PM10", "NO2", "SO2", "CO", "O3"]
    elif not pollutants:
        pollutants = ["PM25"]
    metric = "mean"
    for word, mapped in (("median", "median"), ("maximum", "max"), ("highest", "max"), ("minimum", "min"), ("variability", "std"), ("95th", "p95")):
        if word in text:
            metric = mapped
            break
    resolution = "hourly" if "hour" in text or "time of day" in text else "monthly" if "monthly" in text or "month by month" in text else "yearly" if any(term in text for term in ("annual", "yearly", "over years", "by year", "across years")) else "daily"
    time_range = _time_range_from_question(text)
    year = re.search(r"\b(20\d{2})\b", text)
    operation = "aggregate"
    if asks_aqi: operation = "aqi"
    elif any(word in text for word in ("compare", "versus", " vs ")): operation = "compare"
    elif "rank" in text or (any(word in text for word in ("highest", "top")) and any(word in text for word in ("city", "cities"))): operation = "rank"
    elif any(word in text for word in ("trend", "increased", "decreased", "improved")): operation = "trend"
    elif "correlation" in text or "correlat" in text: operation = "correlation"
    elif any(word in text for word in ("distribution", "histogram", "spread")): operation = "distribution"
    elif any(word in text for word in ("coverage", "data quality", "missing data")):
        operation, metric = "quality", "coverage"
    threshold_match = re.search(r"(?:above|over|exceed(?:ed|ing)?)\s+(\d+(?:\.\d+)?)", text)
    filters = {"value_status": "any"}
    if threshold_match:
        operation = "exceedance"; filters["threshold"] = float(threshold_match.group(1))
    if operation == "correlation":
        if "spearman" in text: filters["correlation_method"] = "spearman"
        elif "kendall" in text: filters["correlation_method"] = "kendall"
    if operation == "rank":
        metric = "mean"
        if year and resolution == "daily":
            resolution = "yearly"
    payload = {
        "dataset": dataset, "operation": operation, "geography_level": level, "locations": locations,
        "pollutants": pollutants, "time_range": time_range,
        "resolution": resolution, "metrics": [metric], "filters": filters,
    }
    if operation == "rank":
        payload["group_by"] = "city"
    return QuerySpec.model_validate(payload)


POLLUTANT_PATTERNS = (
    (r"pm\s*2(?:\.|\s*)5|\bpm25\b|pm₂\.₅", "PM25"),
    (r"\bpm\s*10\b|pm₁₀", "PM10"),
    (r"\bno\s*2\b|no₂", "NO2"),
    (r"\bso\s*2\b|so₂", "SO2"),
    (r"\bo\s*3\b|o₃|\bozone\b", "O3"),
    (r"\bco\b", "CO"),
)


def _pollutants_from_text(text: str) -> list[str] | None:
    """Deterministically detect explicit pollutant mentions in a question.

    Returns None for an AQI question (which implies all six pollutants
    regardless of any single mention) or when nothing matched, so callers
    only override a model's pollutant choice when the text gives clear,
    specific evidence -- not when the question is silent on the point.
    """
    if re.search(r"\baqi\b|air quality index", text):
        return None
    pollutants = []
    for pattern, canonical in POLLUTANT_PATTERNS:
        if re.search(pattern, text) and canonical not in pollutants:
            pollutants.append(canonical)
    return pollutants or None


def _expected_dataset_from_text(text: str) -> str | None:
    """Deterministically infer which Dataset enum a question's own wording implies.

    Returns None when the text names neither country, so callers only override
    a model's dataset choice when the question itself gives clear evidence.
    """
    has_china = any(word in text for word in ("china", "beijing", "shanghai", "guangzhou", "shenzhen", "chinese"))
    has_india = any(word in text for word in ("india", "delhi", "mumbai", "bengaluru", "bangalore", "chennai", "kolkata", "indian"))
    return "ALL" if has_china and has_india else "CNEMC" if has_china else "CPCB" if has_india else None


def _confident_relative_period(text: str) -> str | None:
    """Extract a period phrase for canonical relative/current-period wording.

    Returns None when the question doesn't use one of these forms, so callers
    can fall back to explicit-date extraction or a default instead of forcing
    an unrelated period onto a question that never mentioned one.
    """
    relative = re.search(r"\b(?:last|past)\s+(?:\d+\s+)?(?:complete\s+)?(?:hours?|days?|weeks?|months?|years?)\b", text)
    if relative:
        return relative.group(0)
    if re.search(r"\btoday\b", text):
        return "today"
    if re.search(r"\byesterday\b", text):
        return "yesterday"
    if re.search(r"\b(?:this|current)\s+week\b", text):
        return "this week"
    current_month = re.search(r"\b(?:this month|current month|latest available|latest month)\b", text)
    if current_month:
        return current_month.group(0)
    if re.search(
        r"\b(?:this|current)\s+year\b|\byear\s+to\s+date\b|\bytd\b|\b(?:start|beginning)\s+of\s+(?:the\s+)?year\b",
        text,
    ):
        return "this year"
    if any(term in text for term in ("over years", "by year", "across years", "all years")):
        return "all available"
    return None


def _confident_time_range_from_question(text: str) -> dict[str, str] | None:
    """Like _confident_relative_period, but also covers explicit dates the
    question names outright (a day, a month, a year, or a range of these).
    Returns None when the question gives no textual time evidence at all, so
    callers don't force a period onto a question that never mentioned one.
    """
    confident = _confident_relative_period(text)
    if confident:
        return {"period": confident}
    month_names = {name.casefold(): number for number, name in enumerate(calendar.month_name) if name}
    month_names.update({name.casefold(): number for number, name in enumerate(calendar.month_abbr) if name})
    names = "|".join(sorted(month_names, key=len, reverse=True))
    day_matches = list(re.finditer(rf"\b(\d{{1,2}})\s+({names})\s+(20\d{{2}})(?:\s+(\d{{1,2}}):(\d{{2}}))?\b", text))
    if day_matches:
        # An explicit trailing time (e.g. a forecast issue timestamp like "31
        # May 2026 23:00") means exactly that instant, not end-of-day --
        # matches how available_end is always stored (on the hour, never at
        # 23:59:59), so forcing 23:59:59 here would push the requested end
        # 59 minutes past data that is actually available through this exact
        # moment.
        values = [
            datetime(int(item.group(3)), month_names[item.group(2)], int(item.group(1)),
                     int(item.group(4)), int(item.group(5))) if item.group(4)
            else datetime(int(item.group(3)), month_names[item.group(2)], int(item.group(1)))
            for item in day_matches
        ]
        start = values[0]
        end = values[-1] if day_matches[-1].group(4) else values[-1].replace(hour=23, minute=59, second=59)
        return {"start": start.isoformat(), "end": end.isoformat()}
    month_matches = list(re.finditer(rf"\b({names})\s+(20\d{{2}})\b", text))
    if month_matches:
        first, last = month_matches[0], month_matches[-1]
        start = datetime(int(first.group(2)), month_names[first.group(1)], 1)
        last_year, last_month = int(last.group(2)), month_names[last.group(1)]
        end = datetime(last_year, last_month, calendar.monthrange(last_year, last_month)[1], 23, 59, 59)
        if len(month_matches) == 1:
            return {"period": f"{last_year:04d}-{last_month:02d}"}
        return {"start": start.isoformat(), "end": end.isoformat()}
    years = [int(value) for value in re.findall(r"\b(20\d{2})\b", text)]
    if len(years) >= 2:
        return {"start": datetime(years[0], 1, 1).isoformat(), "end": datetime(years[-1], 12, 31, 23, 59, 59).isoformat()}
    if years:
        return {"period": str(years[0])}
    return None


def _time_range_from_question(text: str) -> dict[str, str]:
    return _confident_time_range_from_question(text) or {"period": "last month"}


METRIC_LABELS = {
    "mean": "average", "median": "median", "max": "maximum", "min": "minimum",
    "std": "standard deviation", "variance": "variance", "p05": "5th percentile",
    "p25": "25th percentile", "p75": "75th percentile", "p95": "95th percentile",
    "count": "count", "coverage": "coverage",
}


def explain_result(question: str, result: dict) -> str:
    data, meta, period = result["data"], result["metadata"], result["resolved_period"]
    spec, analysis = result["query_spec"], result.get("analysis") or {}
    if not data:
        return "No valid rows matched the requested geography, pollutant, period, and quality filters. I did not replace the request with a different scope."
    metric = next((key for key in spec.get("metrics", []) if key in data[0]), None) or next((key for key in ("mean", "median", "max", "min", "p95", "std", "coverage") if key in data[0]), None)
    unit = data[0].get("unit", "")
    start, end = _date_label(period["start"]), _date_label(period["end"])
    if spec["operation"] == "aqi" and metric:
        valid = [row for row in data if row.get(metric) is not None]
        completed = [row for row in valid if spec["resolution"] != "yearly" or row.get("reporting_days", 0) >= 350]
        comparable = completed or valid
        highest, lowest = max(comparable, key=lambda row: row[metric]), min(comparable, key=lambda row: row[metric])
        location = spec["locations"][0]
        standard = meta.get("aqi_standard", {}).get("name", "the selected national AQI standard")
        summary = (f"For {location}, the average daily AQI was highest in {_date_label(highest['timestamp'])} at {highest[metric]:.1f} "
                   f"and lowest in {_date_label(lowest['timestamp'])} at {lowest[metric]:.1f}. The chart shows {len(valid)} {spec['resolution']} values from {start} to {end}, calculated with {standard}.")
        partial = [row for row in valid if row not in completed] if spec["resolution"] == "yearly" else []
        if partial:
            latest_partial = partial[-1]
            summary += f" {_date_label(latest_partial['timestamp'])} is incomplete ({latest_partial['reporting_days']} days, average {latest_partial[metric]:.1f}) and is not used for the highest/lowest full-year comparison."
    elif spec["operation"] == "rank" and metric:
        top = data[0]
        pollutant = top.get("pollutant", spec["pollutants"][0])
        summary = (f"{top['location']} had the highest {spec['resolution']} average {pollutant} among the matched Indian cities in {start}, "
                   f"at {top[metric]:.2f} {unit}. The ranking compares {len(data)} cities and uses {meta['station_count']} monitoring stations in total.")
    elif spec["operation"] == "exceedance":
        summary = f"The threshold was exceeded in {analysis.get('exceedance_count', len(data))} of {analysis.get('total_buckets', len(data))} {spec['resolution']} periods between {start} and {end}."
    elif spec["operation"] == "correlation" and analysis.get("correlation") is not None:
        method = analysis.get("correlation_method", "pearson").capitalize()
        summary = f"The {method} correlation was {analysis['correlation']:.3f} across {analysis['paired_buckets']} matched {spec['resolution']} periods between {start} and {end}."
    elif metric and len(data) == 1:
        row = data[0]
        summary = f"The {metric} {row.get('pollutant', spec['pollutants'][0])} value for {row.get('location', spec['locations'][0])} was {row[metric]:.2f} {unit} between {start} and {end}."
    elif metric:
        valid = [row for row in data if row.get(metric) is not None]
        values = [row[metric] for row in valid]
        overall = sum(values) / len(values)
        highest, lowest = max(valid, key=lambda row: row[metric]), min(valid, key=lambda row: row[metric])
        metric_label = METRIC_LABELS.get(metric, metric)
        summary = (f"From {start} to {end}, the {metric_label} {spec['pollutants'][0]} was {overall:.2f} {unit} "
                   f"(daily values ranged from {lowest[metric]:.2f} {unit} on {_date_label(lowest['timestamp'])} "
                   f"to {highest[metric]:.2f} {unit} on {_date_label(highest['timestamp'])}). "
                   f"The result contains {len(valid)} {spec['resolution']} periods from {meta['station_count']} stations.")
    else:
        summary = f"The query returned {len(data)} results covering {start} to {end}."
    if meta.get("station_balanced") and meta.get("station_count", 0) > 1:
        summary += " Each monitoring station receives equal weight in area averages."
    status = meta.get("value_status")
    if status == "historical_unknown":
        summary += " The legacy historical files do not record which values were measured and which were filled, so that distinction cannot be shown for this period."
    elif status == "observed":
        summary += " These values are labeled as directly observed in the newly ingested data."
    elif status == "mixed":
        summary += " The result combines values with different measurement/filling labels; the counts are included in the downloadable data."
    return summary


def _date_label(value: str) -> str:
    timestamp = datetime.fromisoformat(value)
    if timestamp.month == 1 and timestamp.day == 1 and timestamp.hour == 0:
        return str(timestamp.year)
    return timestamp.strftime("%d %b %Y")
