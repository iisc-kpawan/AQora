from aqa.compound_analytics import compound_operation
from aqa.forecast_analytics import _comparison_horizons


def test_compound_operation_recognizes_semantic_intent_variants():
    assert compound_operation(
        "For Germany, say which of NO2 and ozone has greater relative variability in July 2026."
    ) == "cross_pollutant_relative_variability"
    assert compound_operation(
        "Contrast summer and winter PM2.5 concentrations for India and China."
    ) == "seasonal_contrast"
    assert compound_operation(
        "Explain which months drove India's year-over-year PM2.5 difference."
    ) == "year_over_year_month_attribution"
    assert compound_operation(
        "Report the direction of each active-station count change between two months."
    ) == "active_station_change"
    assert compound_operation(
        "Rank countries above their own 95th percentile historical PM2.5 baseline."
    ) == "historical_percentile_extreme_burden"


def test_compound_operation_does_not_capture_simple_queries():
    assert compound_operation("What was mean PM2.5 in Germany last month?") is None
    assert compound_operation("Forecast Beijing PM2.5 for 24 hours.") is None


def test_multi_horizon_parser_requires_common_issue_time():
    assert _comparison_horizons(
        "Forecast Beijing PM2.5 for 6, 12, and 24 hours from the same issue time."
    ) == [6, 12, 24]
    assert _comparison_horizons("Forecast Beijing PM2.5 for 24 hours.") == []
