from datetime import datetime, timedelta

import pytest

from aqa.forecasting import forecast_at_issue, forecast_series, rolling_backtest


def rows(count=240):
    start=datetime(2026,1,1)
    return [{"timestamp":(start+timedelta(hours=i)).isoformat(),"mean":float(i%24)} for i in range(count)]


def test_auto_forecast_selects_and_returns_horizon():
    result=forecast_series(rows(),horizon=24)
    assert result.model == "seasonal_naive"
    assert len(result.values) == 24
    assert result.values[0]["forecast"] == 0


def test_forecast_rejects_short_series():
    with pytest.raises(Exception): forecast_series(rows(20))


def provenance_rows(count=240, filled_count=0):
    start = datetime(2026, 1, 1)
    return [{
        "timestamp": start + timedelta(hours=index), "mean": float(index % 24),
        "observed_mean": float(index % 24), "valid_count": 10, "filled_count": filled_count,
    } for index in range(count)]


def test_provenance_forecast_enforces_filled_fraction_threshold():
    issue = datetime(2026, 1, 8, 23)
    with pytest.raises(Exception, match="above the configured"):
        forecast_at_issue(provenance_rows(filled_count=3), issue_time=issue, maximum_filled_fraction=.20)
    accepted = forecast_at_issue(provenance_rows(filled_count=2), issue_time=issue, maximum_filled_fraction=.20)
    assert accepted.input_quality["filled_fraction"] == pytest.approx(.20)


def test_future_values_do_not_change_the_forecast():
    issue = datetime(2026, 1, 8, 23)
    original = provenance_rows()
    changed = [dict(row) for row in original]
    for row in changed:
        if row["timestamp"] > issue:
            row["mean"] += 10_000
    first = forecast_at_issue(original, issue_time=issue)
    second = forecast_at_issue(changed, issue_time=issue)
    assert [row["forecast"] for row in first.values] == [row["forecast"] for row in second.values]


def test_rolling_backtest_aggregates_multiple_leakage_safe_origins():
    result = rolling_backtest(
        provenance_rows(264), issue_times=[datetime(2026, 1, 8, 23), datetime(2026, 1, 9, 23)],
    )
    assert result["origin_count"] == 2
    assert result["observed_target_count"] == 48
    assert result["metrics"]["mae"] == pytest.approx(0)


def test_lenient_forecast_does_not_reject_on_high_filled_fraction():
    # strict_input_quality=True (the default, used by the readiness audit,
    # rolling portfolio backtests, and simulation experiments) must keep
    # rejecting exactly as before. strict_input_quality=False (used only by
    # the conversational/API forecast entry point) must never reject for
    # this reason -- it should always produce a forecast and disclose the
    # actual filled fraction instead.
    issue = datetime(2026, 1, 8, 23)
    data = provenance_rows(filled_count=8)  # 8 of 10 valid_count per row => 80% filled
    with pytest.raises(Exception, match="above the configured"):
        forecast_at_issue(data, issue_time=issue, maximum_filled_fraction=.20, strict_input_quality=True)
    lenient = forecast_at_issue(data, issue_time=issue, maximum_filled_fraction=.20, strict_input_quality=False)
    assert lenient.input_quality["exceeds_maximum_filled_fraction"] is True
    assert lenient.input_quality["filled_fraction"] == pytest.approx(.80)
    assert lenient.input_quality["model_selection_target"] == "directly_observed_only"
    assert len(lenient.values) == 24


def test_lenient_forecast_falls_back_to_filled_validation_when_no_observed_targets():
    issue = datetime(2026, 1, 8, 23)
    data = provenance_rows()
    # Blank out observed_mean for the whole model-selection validation
    # window (the 24 hours up to and including the issue time) while
    # keeping filled_count low, so only the "not enough directly observed
    # validation targets" gate is actually at stake here, not the filled-
    # fraction one.
    for row in data:
        if issue - timedelta(hours=23) <= row["timestamp"] <= issue:
            row["observed_mean"] = None
    with pytest.raises(Exception, match="directly observed validation targets"):
        forecast_at_issue(data, issue_time=issue, strict_input_quality=True)
    lenient = forecast_at_issue(data, issue_time=issue, strict_input_quality=False)
    assert lenient.input_quality["model_selection_target"] == "policy_approved_filled_fallback"
    assert lenient.input_quality["exceeds_maximum_filled_fraction"] is False
    assert len(lenient.values) == 24
