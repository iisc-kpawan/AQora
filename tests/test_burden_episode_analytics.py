from datetime import datetime

import pytest

from aqa.burden_episode_analytics import (
    build_multiday_episodes,
    burden_episode_operation,
    wilson_interval,
)


def test_burden_episode_operation_recognizes_semantic_variants():
    assert burden_episode_operation(
        "Estimate uncertainty in annualised PM2.5 exceedance days for incomplete city records."
    ) == "annualized_exceedance_uncertainty"
    assert burden_episode_operation(
        "Rank the most severe multi day city episodes using duration times mean exceedance."
    ) == "severe_multiday_episode_ranking"


def test_burden_episode_operation_does_not_capture_simple_threshold_requests():
    assert burden_episode_operation("How many PM2.5 exceedance days did Delhi have?") is None
    assert burden_episode_operation("Show Beijing's longest PM2.5 episode.") is None


def test_wilson_interval_handles_boundary_and_central_counts():
    assert wilson_interval(0, 100)[0] == 0
    assert wilson_interval(0, 100)[1] == pytest.approx(0.0369935, rel=1e-5)
    lower, upper = wilson_interval(50, 100)
    assert lower == pytest.approx(0.4038315, rel=1e-5)
    assert upper == pytest.approx(0.5961685, rel=1e-5)


def test_wilson_interval_rejects_invalid_counts():
    with pytest.raises(ValueError):
        wilson_interval(2, 1)
    with pytest.raises(ValueError):
        wilson_interval(0, 0)


def test_multiday_episode_builder_breaks_on_missing_and_non_exceeding_days():
    daily = [
        _day("2026-01-01", 20),
        _day("2026-01-02", 25),
        _day("2026-01-04", 30),  # Missing 03 Jan breaks the run.
        _day("2026-01-05", 10),  # A non-exceedance breaks the run.
        _day("2026-01-06", 35),
        _day("2026-01-07", 45),
        _day("2026-01-08", 55),
    ]
    episodes = build_multiday_episodes(daily, threshold=15)
    assert len(episodes) == 2
    assert episodes[0]["start"][:10] == "2026-01-01"
    assert episodes[0]["duration_days"] == 2
    assert episodes[0]["mean_exceedance"] == pytest.approx(7.5)
    assert episodes[0]["severity"] == pytest.approx(15)
    assert episodes[1]["start"][:10] == "2026-01-06"
    assert episodes[1]["duration_days"] == 3
    assert episodes[1]["severity"] == pytest.approx(90)


def _day(day: str, mean: float) -> dict:
    return {
        "dataset": "TEST",
        "country": "Example",
        "area_id": "city-1",
        "area_name": "Example City",
        "bucket_day": datetime.fromisoformat(day),
        "mean": mean,
        "coverage": 1.0,
        "observed_fraction": 0.8,
        "imputed_fraction": 0.2,
        "active_station_count": 3,
        "unit": "µg/m³",
    }
