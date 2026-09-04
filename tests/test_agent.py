from aqa.agent import rule_based_plan


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
