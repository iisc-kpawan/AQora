from aqa.agent import _confident_time_range_from_question, _expected_dataset_from_text, _pollutants_from_text, rule_based_plan


def test_pollutants_from_text_recognizes_ozone_synonym():
    # _pollutants_from_text expects already-casefolded text, matching how
    # every caller (rule_based_plan, the agent guardrail) invokes it.
    assert _pollutants_from_text("ozone levels in delhi march 2026") == ["O3"]
    assert _pollutants_from_text("average so2 in delhi") == ["SO2"]
    assert _pollutants_from_text("compare pm2.5 and no2 in delhi") == ["PM25", "NO2"]


def test_pollutants_from_text_ignores_aqi_questions():
    assert _pollutants_from_text("what is the aqi in delhi") is None


def test_pollutants_from_text_returns_none_when_nothing_matches():
    assert _pollutants_from_text("data coverage in delhi") is None


def test_expected_dataset_from_text_detects_country_from_city_names():
    assert _expected_dataset_from_text("pm 10 of beijing last year") == "CNEMC"
    assert _expected_dataset_from_text("average pm2.5 in delhi") == "CPCB"
    assert _expected_dataset_from_text("compare delhi and beijing pm2.5") == "ALL"
    assert _expected_dataset_from_text("average pm2.5 in london") is None


def test_rule_planner_resolves_china_alias_and_pollutant():
    spec=rule_based_plan("Average PM2.5 in Beijing last month")
    assert spec.dataset.value == "CNEMC"
    assert spec.locations == ["Beijing"]
    assert spec.pollutants == ["PM25"]


def test_rule_planner_threshold():
    spec=rule_based_plan("How many days did PM10 in Delhi exceed 100 in 2025?")
    assert spec.operation.value == "exceedance"
    assert spec.filters.threshold == 100


def test_rule_planner_cross_country_compare():
    spec = rule_based_plan("Compare PM2.5 in Delhi and Beijing in 2025")
    assert spec.dataset.value == "ALL"
    assert spec.operation.value == "compare"
    assert spec.locations == ["Delhi", "Beijing"]


def test_rule_planner_rank_and_quality():
    ranked = rule_based_plan("Rank Indian cities by PM10 in 2025")
    assert ranked.operation.value == "rank"
    assert ranked.group_by.value == "city"
    quality = rule_based_plan("Show data quality coverage for Delhi in 2025")
    assert quality.operation.value == "quality"
    assert quality.metrics[0].value == "coverage"


def test_rule_planner_resolves_bare_last_year_and_this_month():
    spec = rule_based_plan("Average PM2.5 in Delhi last year")
    assert spec.time_range.period == "last year"
    spec = rule_based_plan("Average PM2.5 in Delhi this month")
    assert spec.time_range.period == "this month"


def test_rule_planner_resolves_today_yesterday_and_this_year():
    assert rule_based_plan("Average PM2.5 in Delhi today").time_range.period == "today"
    assert rule_based_plan("Average PM2.5 in Delhi yesterday").time_range.period == "yesterday"
    assert rule_based_plan("Average PM2.5 in Delhi this year").time_range.period == "this year"
    assert rule_based_plan("Average PM2.5 in Delhi year to date").time_range.period == "this year"
    assert rule_based_plan("Average PM2.5 in Delhi past 5 years").time_range.period == "past 5 years"


def test_rule_planner_understands_aqi_over_years():
    spec = rule_based_plan("chennai aqi over years")
    assert spec.operation.value == "aqi"
    assert spec.locations == ["Chennai"]
    assert spec.resolution.value == "yearly"
    assert spec.time_range.period == "all available"
    assert set(spec.pollutants) == {"PM25", "PM10", "NO2", "SO2", "CO", "O3"}


def test_rule_planner_ranks_indian_cities_for_highest_co():
    spec = rule_based_plan("whats the highest co in the year 2024 of indian city")
    assert spec.operation.value == "rank"
    assert spec.geography_level.value == "country"
    assert spec.locations == ["India"]
    assert spec.group_by.value == "city"
    assert spec.resolution.value == "yearly"
    assert spec.metrics[0].value == "mean"


def test_rule_planner_preserves_exact_day_and_month_ranges():
    day = rule_based_plan("Mean PM2.5 in Delhi on 27 August 2024")
    assert day.time_range.start.isoformat() == "2024-08-27T00:00:00"
    assert day.time_range.end.isoformat() == "2024-08-27T23:59:59"
    months = rule_based_plan("Compare PM2.5 in Delhi from August 2024 to July 2025")
    assert months.time_range.start.isoformat() == "2024-08-01T00:00:00"
    assert months.time_range.end.isoformat() == "2025-07-31T23:59:59"


def test_confident_time_range_respects_an_explicit_forecast_issue_time():
    # Regression: a date immediately followed by an explicit time used to be
    # silently pushed to end-of-day (23:59:59), one minute later than any
    # dataset's available_end could ever be (always stored on the hour),
    # so a forecast issue-time question naming the exact currently-available
    # moment always resolved to an end past what's actually loaded.
    result = _confident_time_range_from_question(
        "forecast delhi pm25 for the next 24 hours using data through 31 may 2026 23:00."
    )
    assert result == {"start": "2026-05-31T23:00:00", "end": "2026-05-31T23:00:00"}
