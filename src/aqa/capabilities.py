from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import duckdb

from .config import load_config
from .lakehouse_catalog import LakehouseCatalog


COUNTRY_ALIASES = {
    "CPCB": r"\bindia\b|\bindian\b|\bcpcb\b|\bdelhi\b|\bmumbai\b|\bchennai\b|\bkolkata\b|\bbengaluru\b|\bbangalore\b",
    "CNEMC": r"\bchina\b|\bchinese\b|\bcnemc\b|\bbeijing\b|\bshanghai\b|\bguangzhou\b|\bshenzhen\b",
    "EPA": r"\bunited states\b|\bu\.s\.(?:a\.)?\b|\busa\b|\bamerican\b",
    "AURN": r"\bunited kingdom\b|\bu\.k\.\b|\buk\b|\bbritain\b|\bbritish\b|\blondon\b",
    "SINAICA": r"\bmexico\b|\bmexican\b|\bsinaica\b",
    "EEA_FR": r"\bfrance\b|\bfrench\b",
    "EEA_DE": r"\bgermany\b|\bgerman\b",
    "OPENAQ_KR": r"\bsouth korea\b|\bkorean\b|\bseoul\b",
}
PRIMARY_DATASETS = ("CPCB", "CNEMC", "EPA", "SINAICA", "AURN", "EEA_DE", "EEA_FR")
ADVANCED_METHODS = {
    "bootstrap uncertainty": r"\bbootstrap\b|confidence interval|uncertainty interval",
    "change-point detection": r"change[- ]?point",
    "robust/monotonic regression": r"robust regression|monotonic trend|theil|mann.?kendall",
    "simulation/sensitivity experiment": r"artificially mask|reproducibly mask|simulate \d+%|masking the final|alternate aggregation|sensitivity",
    "historical observed/imputed split": r"observed[- ]only.*(?:imputed|filled)|(?:imputed|filled).*observed[- ]only|imputed fraction|fraction .* imputed|measured (?:versus|vs\.?) filled",
    "year-over-year country ranking": r"rank countries by year-over-year|rank countries by year over year",
    "weekday/diurnal cohort comparison": r"weekday.*weekend|weekday diurnal|hour-of-day by day-of-week|jointly observed in both cities",
    "cross-standard AQI comparison": r"local aqi frameworks|common .*exceedance metric.*aqi|comparable from non-comparable",
    "compound seasonal/cohort analysis": r"predefined seasonal windows|seasonal-strength metric|matching months|matched non-missing|active monitoring-station composition",
    "multi-constraint ranking or sensitivity": r"rank stability|different coverage thresholds|station-weighted versus city-balanced|robust to coverage|own 95th-percentile|duration [×x] mean exceedance",
    "AQI category-burden comparison": r"aqi category burden|classification as high-burden|using only dates for which both cities",
    "configured guideline comparison": r"two different configured .*guideline",
    "ingestion reliability decomposition": r"ingestion success|late-arrival recovery|delayed, missing, invalid",
    "forecast evaluation experiment": r"forecast-model ranking|forecast accuracy across|rolling[- ]origin|backtests? for eligible|context gaps|fails? to beat seasonal naive|same issue time|forecasting evaluation under",
    "compound year-over-year attribution": r"year over year.*three months|year-over-year.*contributing|pm2\.5 improved.*o3 worsened|worsening pm2\.5 and o3",
    "temporal completeness cohort": r"complete coverage for all six|valid-hour coverage.*first/last|minimum history, continuity, coverage",
    "segmented trend comparison": r"trend slopes before and after|before and after january",
    "annualized burden uncertainty": r"uncertainty in annualized|annualized .* incomplete data",
    "derived temporal statistics": r"statistically meaningful monotonic|monthly anomalies|decreasing .* trends?.*increasing .* trends?|both slopes|active-station count.*changed|direction of change",
    "coverage cohort construction": r"observations in every month|reproducible cohort|inclusion and exclusion reasons|station-days meeting .*coverage threshold",
    "aggregation-method comparison": r"station-weighted.*city-weighted|city-weighted.*station-weighted|aggregation schemes|identify rank reversals",
    "compound data-quality uncertainty": r"uncertainty due to missing data|estimate gap|rank changes .*filled data.*observed-only",
    "specialized multi-view visualization": r"scatter plot|heatmap.*mask|small-multiple|multi-panel|dual-view|companion marker|integrated visual summary",
    "thresholded metadata cohort": r"more than \d+ monitoring stations.*(?:coverage window|data-coverage window)",
    "cross-pollutant derived comparison": r"pollutant with greater relative variability|relative variability in each country",
    "compound provenance trend sensitivity": r"(?:observed[- ]only|filled|imputed).*trend slope|trend slope.*(?:observed[- ]only|filled|imputed)",
}
MONTHS = {name.casefold(): index for index, name in enumerate(calendar.month_name) if name}
MONTHS.update({name.casefold(): index for index, name in enumerate(calendar.month_abbr) if name})


@dataclass
class CapabilityAssessment:
    intent: str
    supported_datasets: list[str] = field(default_factory=list)
    unsupported_places: list[str] = field(default_factory=list)
    unavailable_periods: list[str] = field(default_factory=list)
    unsupported_methods: list[str] = field(default_factory=list)
    location_ambiguous: bool = False

    @property
    def can_execute(self) -> bool:
        return not (
            self.unsupported_places or self.unavailable_periods
            or self.unsupported_methods or self.location_ambiguous
        )


def classify_intent(question: str) -> str:
    text = question.casefold()
    if "forecast" in text or "backtest" in text or "mase" in text or "rmse" in text or "smape" in text:
        return "forecast"
    if any(term in text for term in (
        "source network", "source family", "station–pollutant", "station-pollutant",
        "available pollutant", "all pollutants", "monitoring station", "number of stations",
        "countries in aqa", "metadata", "date range", "earliest", "latest timestamp",
        "availability matrix", "provenance chain", "canonical units", "conversion rules",
        "provenance for", "value provenance", "measurement lineage",
    )):
        return "metadata"
    if "aqi" in text or "exceed" in text or "guideline" in text:
        return "aqi_or_exceedance"
    if any(term in text for term in ("rank", "top ", "highest", "lowest")):
        return "ranking"
    if any(term in text for term in ("missing", "coverage", "imputed", "data quality", "gap length", "completeness")):
        return "quality"
    if any(term in text for term in ("compare", " versus ", " vs ", "difference")):
        return "comparison"
    if any(term in text for term in ("trend", "moving average", "change", "profile", "season")):
        return "temporal"
    return "statistics"


def _implies_global_scope_without_a_handler(text: str) -> bool:
    """Catches cross-country/global wording the earlier global_scope/
    primary_scope checks miss (e.g. "for each eligible country" rather than
    "for each primary network", "every country" rather than "all countries",
    a cities ranking naming no city, "selected ... cities"). None of these
    phrasings are backed by an implemented specialized handler today, so the
    caller reports them as unsupported rather than asking for a location that
    would not actually resolve anything.
    """
    return bool(
        re.search(r"\bfor\s+each\s+(?:primary|eligible)\s+countr", text)
        or re.search(r"\bevery\s+countr", text)
        or re.search(r"\bglobal\s+summary\b", text)
        or re.search(r"\bcities\s+with\s+at\s+least\b", text)
        or re.search(r"\b\d+\s+cities\s+with\s+the\s+(?:highest|lowest)\b", text)
        or re.search(r"\bselected\s+(?:[\w-]+\s+)?cit(?:y|ies)\b", text)
    )


def is_flexible_country_cohort(question: str, intent: str | None = None) -> bool:
    text = question.casefold()
    intent = intent or classify_intent(question)
    if any(term in text for term in ("available countries", "eligible countries", "countries with", "which country", "across countries")):
        return True
    if re.search(r"\bcountries\b|\bcountry[–-]pollutant pairs\b", text):
        explicitly_named = sum(bool(re.search(pattern, text)) for pattern in COUNTRY_ALIASES.values())
        return intent != "comparison" or explicitly_named == 0
    return intent == "ranking" and bool(re.search(r"\brank (?:all )?countries\b|\brank all available\b", text))


def assess_question(question: str, config: dict | None = None) -> CapabilityAssessment:
    config = config or load_config()
    text = question.casefold()
    assessment = CapabilityAssessment(intent=classify_intent(question))
    lakehouse = LakehouseCatalog(config)
    for item in lakehouse.datasets:
        pattern = COUNTRY_ALIASES.get(item.dataset)
        name_pattern = rf"\b{re.escape(item.country_name.casefold())}\b"
        dataset_pattern = rf"\b{re.escape(item.dataset.casefold())}\b"
        if re.search(pattern, text) if pattern else re.search(name_pattern, text) or re.search(dataset_pattern, text):
            if item.dataset not in assessment.supported_datasets:
                assessment.supported_datasets.append(item.dataset)
    primary_scope = any(term in text for term in (
        "seven primary", "all seven", "each primary network", "for each primary network",
    ))
    global_scope = any(term in text for term in (
        "all available countries", "all countries", "countries in aqa", "five countries",
    ))
    if "city-level" in text and not assessment.supported_datasets:
        global_scope = True
    if _burden_episode_supported(text):
        global_scope = True
    if assessment.intent == "ranking" and "countr" in text:
        global_scope = True
    if primary_scope:
        assessment.supported_datasets = [item.dataset for item in lakehouse.datasets if item.dataset in PRIMARY_DATASETS]
    elif global_scope:
        assessment.supported_datasets = [item.dataset for item in lakehouse.datasets]
    for method, pattern in ADVANCED_METHODS.items():
        if re.search(pattern, text):
            if method in {"configured guideline comparison", "cross-standard AQI comparison", "AQI category-burden comparison"} and _advanced_aqi_supported(text):
                continue
            if method in {"segmented trend comparison", "change-point detection", "robust/monotonic regression", "compound seasonal/cohort analysis", "derived temporal statistics"} and _advanced_temporal_supported(text):
                continue
            if method == "change-point detection" and _changepoint_detection_supported(text):
                continue
            if method in {"derived temporal statistics", "year-over-year country ranking", "bootstrap uncertainty", "compound seasonal/cohort analysis", "compound year-over-year attribution"} and _derived_temporal_supported(text):
                continue
            if method in {"cross-pollutant derived comparison", "compound seasonal/cohort analysis",
                          "compound year-over-year attribution", "derived temporal statistics",
                          "multi-constraint ranking or sensitivity"} and _compound_derived_supported(text):
                continue
            if method in {"temporal completeness cohort", "coverage cohort construction", "thresholded metadata cohort"} and _coverage_cohort_supported(text):
                continue
            if method in {"multi-constraint ranking or sensitivity", "aggregation-method comparison"} and _rank_stability_supported(text):
                continue
            if method in {"annualized burden uncertainty", "multi-constraint ranking or sensitivity"} and _burden_episode_supported(text):
                continue
            if method == "simulation/sensitivity experiment" and (
                _simulation_experiment_supported(text) or _coverage_sensitivity_supported(text)
            ):
                continue
            if method == "specialized multi-view visualization" and _advanced_visualization_supported(text):
                continue
            if method in {"forecast evaluation experiment", "bootstrap uncertainty", "temporal completeness cohort"} and _forecast_portfolio_supported(text):
                continue
            if method == "forecast evaluation experiment" and _multi_horizon_forecast_supported(text):
                continue
            if method == "historical observed/imputed split" and _provenance_policy_supported(text, assessment.intent):
                continue
            if method == "compound provenance trend sensitivity" and _provenance_policy_supported(text, assessment.intent):
                continue
            if method in {"compound data-quality uncertainty", "multi-constraint ranking or sensitivity", "simulation/sensitivity experiment"} and _provenance_policy_supported(text, assessment.intent):
                continue
            if method == "bootstrap uncertainty" and (
                _sampling_bootstrap_supported(text) or _matched_month_city_balanced_bootstrap_supported(text)
            ):
                continue
            if method == "aggregation-method comparison" and _aggregation_comparison_supported(text):
                continue
            if method == "weekday/diurnal cohort comparison" and (
                _weekday_weekend_supported(text) or _hour_by_weekday_supported(text)
            ):
                continue
            assessment.unsupported_methods.append(method)
    flexible_country_cohort = (
        is_flexible_country_cohort(question, assessment.intent)
        or _forecast_portfolio_supported(text) or _advanced_visualization_supported(text)
        or _simulation_experiment_supported(text) or _rank_stability_supported(text)
        or _coverage_cohort_supported(text)
        or _derived_temporal_supported(text)
        or _advanced_temporal_supported(text)
        or _advanced_aqi_supported(text)
        or _burden_episode_supported(text)
    )
    if not assessment.supported_datasets and not assessment.unsupported_places:
        if assessment.intent == "metadata" or flexible_country_cohort:
            # Conceptual/global metadata questions, and operations that compute
            # their own flexible cross-country cohort, are meaningful without
            # naming a single country; keep the existing concrete default.
            assessment.supported_datasets = ["CPCB"]
        elif _implies_global_scope_without_a_handler(text):
            # The question is unambiguously cross-country/global by its own
            # wording ("for each country", "every country", a cities ranking
            # with no city named, "selected ... cities") -- naming a location
            # would not help, and none of the specialized modules implement
            # this particular operation, so silently narrowing it to CPCB/CNEMC
            # (the only datasets the fallback planner knows) would be worse
            # than saying so isn't supported yet.
            assessment.unsupported_methods.append("cross-country/global-scope research operation")
        else:
            # No country or city was named at all -- do not silently guess one
            # (this previously always defaulted to India/CPCB regardless of what,
            # if anything, was discussed earlier in the conversation).
            assessment.location_ambiguous = True
    requested_end = requested_period_end(question)
    if requested_end and not flexible_country_cohort:
        for dataset in assessment.supported_datasets:
            _, available_end = available_bounds(dataset, config)
            if requested_end > available_end:
                assessment.unavailable_periods.append(
                    f"{dataset} requested through {requested_end:%d %b %Y}, available through {available_end:%d %b %Y %H:%M}"
                )
    return assessment


def _provenance_policy_supported(text: str, intent: str) -> bool:
    if intent == "forecast" or "aqi" in text:
        return False
    return (
        ("observed" in text and any(term in text for term in ("imputed", "filled")))
        or "imputed fraction" in text or "imputation changes" in text
        or "imputation rate" in text or "imputation dependence" in text
    )


def _sampling_bootstrap_supported(text: str) -> bool:
    return "bootstrap" in text and "station" in text and any(term in text for term in ("mean", "sampling", "country-level"))


def _matched_month_city_balanced_bootstrap_supported(text: str) -> bool:
    return (
        re.search(r"matched months?", text) is not None
        and "city-balanced" in text
        and any(term in text for term in ("uncertainty interval", "confidence interval", "bootstrap"))
    )


def _aggregation_comparison_supported(text: str) -> bool:
    return (
        "station-weighted" in text and ("city-weighted" in text or "city-balanced" in text)
    ) or ("aggregation schemes" in text and "equal-country" in text)


def _weekday_weekend_supported(text: str) -> bool:
    return ("jointly observed" in text and "diurnal" in text) or (
        "weekday" in text and "weekend" in text
        and "jointly observed" not in text
        and "hour-of-day by day-of-week" not in text
    )


def _hour_by_weekday_supported(text: str) -> bool:
    return bool(re.search(r"hour-of-day\s+by\s+day-of-week|hour\s+of\s+day\s+by\s+day\s+of\s+week", text))


def _forecast_portfolio_supported(text: str) -> bool:
    """Only admit portfolio operations implemented by ForecastPortfolioAnalytics."""
    readiness = (
        "suitable for 24-hour forecasting evaluation" in text
        and "minimum history" in text and "continuity" in text and "coverage" in text
    )
    rolling = (
        "rolling 24-hour" in text and "eligible cities" in text
        and bool(re.search(r"seasonal[ -]naive", text)) and "15" in text and "21" in text
    )
    failure_diagnostics = bool(re.search(r"fails to beat seasonal[ -]naive", text)) and all(
        term in text for term in ("missingness", "volatility", "pollutant level")
    )
    return readiness or rolling or failure_diagnostics


def _advanced_visualization_supported(text: str) -> bool:
    patterns = (
        "mean versus pm2.5 data coverage", "dual-view analysis", "country-by-month pm2.5 mean",
        "small-multiple visualization", "year-over-year pm2.5 change versus",
        "24-hour forecast accuracy and data completeness", "integrated visual summary",
        "mean versus missing-data percentage", "multi-panel episode visualization",
    )
    return any(pattern in text for pattern in patterns)


def _simulation_experiment_supported(text: str) -> bool:
    return any(pattern in text for pattern in (
        "artificially masked 1-hour and 3-hour", "benchmark the configured short-gap imputation",
        "simulate 5%, 10%, and 20% additional missingness",
        "downstream ranking sensitivity to imputation",
        "masking the final 1, 3, and 6 context hours",
        "apparent year-over-year pm2.5 improvement in india",
        "india experienced meaningful pm2.5 improvement over the last 12 complete months",
    ))


def _coverage_sensitivity_supported(text: str) -> bool:
    """A plain area-mean-versus-coverage-threshold sensitivity request.

    Kept narrow so it never intercepts the harder compound cases that
    ADVANCED_METHODS already routes elsewhere: an observed/imputed policy
    angle belongs to ProvenanceAnalytics, and a ranking angle belongs to the
    rank-stability/multi-constraint methods.
    """
    if any(term in text for term in ("imputed", "filled", "observed-only", "observed only")):
        return False
    if any(term in text for term in ("rank", "ranking")):
        return False
    if not ("coverage" in text and re.search(r"sensitiv", text)):
        return False
    return len(re.findall(r"\d+(?:\.\d+)?\s*%", text)) >= 2


def _changepoint_detection_supported(text: str) -> bool:
    """A single-area daily-series changepoint request (TemporalAnalytics).

    Excludes the multi-country "regime changes" compound already handled by
    _advanced_temporal_supported, which needs trend/seasonality/change-point
    analysis combined across a whole cohort rather than one area's single
    strongest breakpoint.
    """
    if not re.search(r"change[- ]?point", text):
        return False
    return not any(term in text for term in (
        "regime", "seven primary", "each country", "every country", "all countries", "primary countries",
    ))


def _rank_stability_supported(text: str) -> bool:
    return any(pattern in text for pattern in (
        "rankings produced by station-weighted versus city-balanced",
        "heatmap of rank stability for countries",
        "minimum coverage threshold needed to keep july 2026 top-10 country",
    ))


def _coverage_cohort_supported(text: str) -> bool:
    return any(pattern in text for pattern in (
        "complete coverage for all six aqa pollutants",
        "station-days meeting the configured minimum coverage threshold",
        "pm2.5 is available from more than 100 monitoring stations",
        "pm2.5 observations in every month from august 2025 through july 2026",
        "country–pollutant combinations that have at least 80% coverage in every month",
        "country-pollutant combinations that have at least 80% coverage in every month",
        "reproducible cohort of countries eligible for cross-country pm2.5 analysis",
        "country–pollutant availability matrix for all six pollutants",
        "country-pollutant availability matrix for all six pollutants",
    ))


def _derived_temporal_supported(text: str) -> bool:
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
    ))


def _advanced_temporal_supported(text: str) -> bool:
    return any(pattern in text for pattern in (
        "monthly pm2.5 trend slopes before and after january 2026 for india and china",
        "regime changes across the seven primary countries",
        "long-term pm2.5 trend estimates from ordinary linear regression",
        "countries with strong winter pm2.5 peaks also show stronger year-over-year variability",
        "changes in active monitoring-station composition materially alter the estimated pm2.5 trend",
    ))


def _advanced_aqi_supported(text: str) -> bool:
    return any(pattern in text for pattern in (
        "delhi and bengaluru cpcb aqi category distributions using only dates",
        "pm2.5 exceedance percentages for india and china under two different configured concentration guidelines",
        "delhi, bengaluru, beijing, london, and mexico city using a common pm2.5 concentration-based exceedance metric",
        "classification as high-burden changes when ranking by mean pm2.5",
    ))


def _burden_episode_supported(text: str) -> bool:
    annualized = (
        bool(re.search(r"annuali[sz]ed", text))
        and "uncertainty" in text and "exceedance" in text
        and bool(re.search(r"\bcit(?:y|ies)\b", text))
        and any(term in text for term in ("incomplete", "missing"))
    )
    episodes = (
        bool(re.search(r"multi[ -]?day", text))
        and "episode" in text and bool(re.search(r"\bcit(?:y|ies)\b", text))
        and any(term in text for term in ("rank", "most severe", "severity"))
        and "duration" in text and "mean exceedance" in text
    )
    return annualized or episodes


def _compound_derived_supported(text: str) -> bool:
    relative_variability = "relative variability" in text and sum(
        bool(re.search(pattern, text)) for pattern in (
            r"pm\s*2(?:\.|\s*)5|\bpm25\b", r"\bpm\s*10\b", r"\bno\s*2\b",
            r"\bso\s*2\b", r"\bo\s*3\b|\bozone\b", r"\bco\b",
        )
    ) >= 2
    seasonal = "winter" in text and "summer" in text and "pm2.5" in text
    attribution = bool(re.search(r"year[- ]over[- ]year", text)) and "month" in text and any(
        term in text for term in ("contribut", "driv", "explain")
    )
    station_change = "active-station count" in text and any(
        term in text for term in ("change", "changed", "direction")
    )
    historical_extremes = "own 95th-percentile" in text or "own 95th percentile" in text
    return relative_variability or seasonal or attribution or station_change or historical_extremes


def _multi_horizon_forecast_supported(text: str) -> bool:
    match = re.search(r"for\s+([0-9,\sand]+?)\s+hours?\b", text)
    return bool(match and "same issue time" in text and len(re.findall(r"\d+", match.group(1))) >= 2)


def requested_period_end(question: str) -> datetime | None:
    text = question.casefold()
    dates = []
    month_pattern = "|".join(sorted(MONTHS, key=len, reverse=True))
    for match in re.finditer(rf"\b(\d{{1,2}})\s+({month_pattern})\s+(20\d{{2}})\b", text):
        dates.append(datetime(int(match.group(3)), MONTHS[match.group(2)], int(match.group(1)), 23, 59, 59))
    for match in re.finditer(rf"(?<!\d )\b({month_pattern})\s+(20\d{{2}})\b", text):
        month, year = MONTHS[match.group(1)], int(match.group(2))
        dates.append(datetime(year, month, calendar.monthrange(year, month)[1], 23, 59, 59))
    for match in re.finditer(r"\b(20\d{2})-(\d{1,2})(?:-(\d{1,2}))?\b", text):
        year, month = int(match.group(1)), int(match.group(2))
        day = int(match.group(3)) if match.group(3) else calendar.monthrange(year, month)[1]
        dates.append(datetime(year, month, day, 23, 59, 59))
    if not dates:
        for match in re.finditer(r"\b(20\d{2})\b", text):
            dates.append(datetime(int(match.group(1)), 12, 31, 23, 59, 59))
    return max(dates) if dates else None


def available_bounds(dataset: str, config: dict | None = None) -> tuple[datetime, datetime]:
    config = config or load_config()
    try:
        item = LakehouseCatalog(config).by_dataset(dataset)
        return item.available_start, item.available_end
    except (FileNotFoundError, KeyError):
        pass
    cfg = config["datasets"][dataset]
    root = Path(config["project"]["data_root"]).resolve() / "curated" / "hourly" / f"dataset={dataset}"
    return _available_bounds_cached(dataset, cfg["historical_start"], cfg["historical_end"], str(root))


@lru_cache(maxsize=8)
def _available_bounds_cached(dataset: str, historical_start: str, historical_end: str, root_value: str) -> tuple[datetime, datetime]:
    start, end = datetime.fromisoformat(historical_start), datetime.fromisoformat(historical_end)
    root = Path(root_value)
    paths = [str(path) for path in root.glob("year=*/month=*/*.parquet")]
    if paths:
        row = duckdb.sql("SELECT min(timestamp_local), max(timestamp_local) FROM read_parquet(?, union_by_name=true)", params=[paths]).fetchone()
        if row and row[0] is not None:
            start, end = min(start, row[0]), max(end, row[1])
    return start, end
