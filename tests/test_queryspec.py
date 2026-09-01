import pytest
from pydantic import ValidationError

from aqa.queryspec import QuerySpec


BASE = {"dataset":"CPCB","geography_level":"city","locations":["Delhi"],"pollutants":["PM25"]}


def test_defaults_are_bounded():
    spec = QuerySpec.model_validate(BASE)
    assert spec.limit == 10_000
    assert spec.time_range.period == "last month"


def test_raw_sql_and_paths_are_rejected():
    with pytest.raises(ValidationError):
        QuerySpec.model_validate({**BASE, "sql":"DROP TABLE observations"})
    with pytest.raises(ValidationError):
        QuerySpec.model_validate({**BASE, "limit":100_001})


def test_exceedance_requires_threshold():
    with pytest.raises(ValidationError):
        QuerySpec.model_validate({**BASE, "operation":"exceedance"})


def test_correlation_requires_two_pollutants():
    with pytest.raises(ValidationError):
        QuerySpec.model_validate({**BASE, "operation":"correlation"})

