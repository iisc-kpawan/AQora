import pytest

from datetime import date

from aqa.aqi import calculate_aqi, standard_for_date, sub_index


@pytest.mark.parametrize("standard,pollutant,concentration,expected", [
    ("IN_CPCB_NAQI_2014","PM25",30,50),
    ("IN_CPCB_NAQI_2014","PM25",60,100),
    ("IN_CPCB_NAQI_2014","CO",2,100),
    ("CN_HJ633_2026","PM25",35,50),
    ("CN_HJ633_2026","PM25",60,100),
    ("CN_HJ633_2012","PM25",75,100),
])
def test_breakpoints(standard,pollutant,concentration,expected):
    assert sub_index(pollutant,concentration,standard) == expected


def test_dominant_pollutant():
    result=calculate_aqi({"PM25":60,"PM10":50},"IN_CPCB_NAQI_2014")
    assert result["aqi"] == 100
    assert result["dominant_pollutants"] == ["PM25"]


def test_china_standard_is_selected_by_effective_date():
    assert standard_for_date("CNEMC", date(2026, 2, 28)).id == "CN_HJ633_2012"
    assert standard_for_date("CNEMC", date(2026, 3, 1)).id == "CN_HJ633_2026"
