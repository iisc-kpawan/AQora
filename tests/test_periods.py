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


def test_today_yesterday_and_this_week_resolve_against_available_end():
    start, end = resolve_period("today", datetime(2026, 5, 31, 17, 30))
    assert start == datetime(2026, 5, 31)
    assert end == datetime(2026, 5, 31, 17, 30)
    start, end = resolve_period("yesterday", datetime(2026, 5, 31, 17, 30))
    assert start == datetime(2026, 5, 30)
    assert end == datetime(2026, 5, 30, 23, 59, 59)
    start, end = resolve_period("this week", datetime(2026, 5, 31, 17, 30))  # a Sunday
    assert start == datetime(2026, 5, 25)  # the preceding Monday
    assert end == datetime(2026, 5, 31, 17, 30)


def test_this_year_and_synonyms_resolve_to_year_to_date():
    for phrase in ("this year", "current year", "year to date", "ytd", "start of year", "beginning of the year"):
        start, end = resolve_period(phrase, datetime(2026, 5, 31, 23))
        assert start == datetime(2026, 1, 1), phrase
        assert end == datetime(2026, 5, 31, 23), phrase


def test_past_is_a_synonym_for_last():
    start, end = resolve_period("past 7 days", datetime(2025, 12, 31, 23))
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
