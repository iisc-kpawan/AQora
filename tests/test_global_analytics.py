from aqa.global_analytics import _station_constraint


def test_station_constraint_recognizes_explicit_minimum():
    assert _station_constraint("countries with at least 10 active pm2.5 stations") == 10
    assert _station_constraint("minimum 5 stations") == 5


def test_station_constraint_recognizes_elided_at_least():
    text = "countries with at least 80% coverage and 10 active pm2.5 stations"
    assert _station_constraint(text) == 10


def test_station_constraint_returns_none_without_a_count():
    assert _station_constraint("rank the top 10 countries by mean pm2.5") is None
