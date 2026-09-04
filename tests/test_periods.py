from datetime import datetime

from aqa.periods import resolve_period


def test_last_month_means_latest_available_month():
    start, end = resolve_period("last month", datetime(2026, 5, 31, 23))
    assert start == datetime(2026, 5, 1)
    assert end == datetime(2026, 5, 31, 23)


def test_bare_last_year_and_last_week_resolve_without_a_count():
    start, end = resolve_period("last year", datetime(2026, 5, 31, 23))
    assert start == datetime(2025, 1, 1)
    assert end == datetime(2026, 5, 31, 23)
    start, end = resolve_period("last week", datetime(2025, 12, 31, 23))
    assert end == datetime(2025, 12, 31, 23)
    assert (end - start).total_seconds() == 167 * 3600


def test_relative_window_uses_data_end():
    start, end = resolve_period("last 7 days", datetime(2025, 12, 31, 23))
    assert end == datetime(2025, 12, 31, 23)
    assert (end - start).total_seconds() == 167 * 3600


def test_complete_months_exclude_partial_latest_month():
    start, end = resolve_period("last 3 complete months", datetime(2026, 5, 18, 23))
    assert start == datetime(2026, 2, 1)
    assert end == datetime(2026, 4, 30, 23, 59, 59)
