from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Mapping

from .errors import ValidationError


@dataclass(frozen=True)
class AQIStandard:
    id: str
    name: str
    country_code: str
    index_points: tuple[int, ...]
    breakpoints: Mapping[str, tuple[float, ...]]
    categories: tuple[str, ...]
    source_url: str
    effective_from: date | None = None
    effective_to: date | None = None
    averaging_hours: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ConcentrationGuideline:
    id: str
    name: str
    thresholds: Mapping[str, float]
    averaging_hours: Mapping[str, int]
    units: Mapping[str, str]
    effective_from: date
    source_url: str


INDIA_CPCB_2014 = AQIStandard(
    id="IN_CPCB_NAQI_2014",
    name="India National Air Quality Index (CPCB 2014)",
    country_code="IN",
    index_points=(0, 50, 100, 200, 300, 400, 500),
    breakpoints={
        "PM10": (0, 50, 100, 250, 350, 430, 500),
        "PM25": (0, 30, 60, 90, 120, 250, 350),
        "NO2": (0, 40, 80, 180, 280, 400, 500),
        "O3": (0, 50, 100, 168, 208, 748, 1000),
        "CO": (0, 1, 2, 10, 17, 34, 50),
        "SO2": (0, 40, 80, 380, 800, 1600, 2000),
    },
    categories=("Good", "Satisfactory", "Moderately polluted", "Poor", "Very poor", "Severe"),
    source_url="https://cpcb.nic.in/displaypdf.php?id=bWFudWFsLW1vbml0b3JpbmcvQVFJX05BTVBfUmVwX1NlcHRlbWJlcjIwMTYucGRm",
    effective_from=date(2014, 10, 17),
    averaging_hours={"PM25": 24, "PM10": 24, "NO2": 24, "SO2": 24, "CO": 8, "O3": 8},
)


CHINA_HJ633_2026 = AQIStandard(
    id="CN_HJ633_2026",
    name="China HJ 633-2026",
    country_code="CN",
    index_points=(0, 50, 100, 150, 200, 300, 400, 500),
    breakpoints={
        "SO2": (0, 50, 150, 475, 800, 1600, 2100, 2620),
        "NO2": (0, 40, 80, 180, 280, 565, 750, 940),
        "CO": (0, 2, 4, 14, 24, 36, 48, 60),
        "O3": (0, 100, 160, 215, 265, 800, 1000, 1200),
        "PM10": (0, 50, 120, 250, 350, 420, 500, 600),
        "PM25": (0, 35, 60, 115, 150, 250, 350, 500),
    },
    categories=("Excellent", "Good", "Lightly polluted", "Moderately polluted", "Heavily polluted", "Severely polluted"),
    source_url="https://www.mee.gov.cn/ywgz/fgbz/bz/bzwb/jcffbz/202602/W020260225366493492011.pdf",
    effective_from=date(2026, 3, 1),
    averaging_hours={"PM25": 24, "PM10": 24, "NO2": 24, "SO2": 24, "CO": 24, "O3": 8},
)


CHINA_HJ633_2012 = AQIStandard(
    id="CN_HJ633_2012",
    name="China HJ 633-2012 (superseded 2026-03-01)",
    country_code="CN",
    index_points=(0, 50, 100, 150, 200, 300, 400, 500),
    breakpoints={
        "SO2": (0, 50, 150, 475, 800, 1600, 2100, 2620),
        "NO2": (0, 40, 80, 180, 280, 565, 750, 940),
        "CO": (0, 2, 4, 14, 24, 36, 48, 60),
        "O3": (0, 100, 160, 215, 265, 800, 1000, 1200),
        "PM10": (0, 50, 150, 250, 350, 420, 500, 600),
        "PM25": (0, 35, 75, 115, 150, 250, 350, 500),
    },
    categories=("Excellent", "Good", "Lightly polluted", "Moderately polluted", "Heavily polluted", "Severely polluted"),
    source_url="https://www.mee.gov.cn/ywgz/fgbz/bz/bzwb/jcffbz/201203/W020250407406362045613.pdf",
    effective_from=date(2016, 1, 1), effective_to=date(2026, 2, 28),
    averaging_hours={"PM25": 24, "PM10": 24, "NO2": 24, "SO2": 24, "CO": 24, "O3": 8},
)


STANDARDS = {item.id: item for item in (INDIA_CPCB_2014, CHINA_HJ633_2026, CHINA_HJ633_2012)}
DEFAULT_STANDARD = {"CPCB": INDIA_CPCB_2014.id, "CNEMC": CHINA_HJ633_2026.id}

WHO_AQG_2021 = ConcentrationGuideline(
    id="WHO_AQG_2021", name="WHO Global Air Quality Guidelines 2021",
    thresholds={"PM25": 15.0, "PM10": 45.0, "NO2": 25.0, "SO2": 40.0, "O3": 100.0},
    averaging_hours={"PM25": 24, "PM10": 24, "NO2": 24, "SO2": 24, "O3": 8},
    units={"PM25": "µg/m³", "PM10": "µg/m³", "NO2": "µg/m³", "SO2": "µg/m³", "O3": "µg/m³"},
    effective_from=date(2021, 9, 22),
    source_url="https://www.who.int/publications/i/item/9789240034228",
)
WHO_IT1_2021 = ConcentrationGuideline(
    id="WHO_IT1_2021", name="WHO Global Air Quality Guidelines 2021 Interim Target 1",
    thresholds={"PM25": 75.0, "PM10": 150.0, "NO2": 120.0, "SO2": 125.0, "O3": 160.0},
    averaging_hours={"PM25": 24, "PM10": 24, "NO2": 24, "SO2": 24, "O3": 8},
    units={"PM25": "µg/m³", "PM10": "µg/m³", "NO2": "µg/m³", "SO2": "µg/m³", "O3": "µg/m³"},
    effective_from=date(2021, 9, 22),
    source_url="https://iris.who.int/bitstream/handle/10665/345329/9789240034228-eng.pdf",
)
GUIDELINES = {item.id: item for item in (WHO_AQG_2021, WHO_IT1_2021)}
DEFAULT_GUIDELINE = WHO_AQG_2021.id


def sub_index(pollutant: str, concentration: float, standard_id: str) -> int:
    standard = get_standard(standard_id)
    pollutant = pollutant.upper().replace(".", "")
    if pollutant == "PM25":
        pollutant = "PM25"
    if pollutant not in standard.breakpoints:
        raise ValidationError(f"{pollutant} is not supported by {standard.id}")
    if concentration < 0 or not math.isfinite(concentration):
        raise ValidationError("AQI concentration must be finite and non-negative")
    concentrations = standard.breakpoints[pollutant]
    points = standard.index_points
    if concentration >= concentrations[-1]:
        return 500
    for index in range(1, len(concentrations)):
        if concentration <= concentrations[index]:
            lo_c, hi_c = concentrations[index - 1], concentrations[index]
            lo_i, hi_i = points[index - 1], points[index]
            value = (hi_i - lo_i) * (concentration - lo_c) / (hi_c - lo_c) + lo_i
            return min(500, math.ceil(value))
    return 500


def calculate_aqi(concentrations: Mapping[str, float], standard_id: str) -> dict:
    standard = get_standard(standard_id)
    indices = {pollutant: sub_index(pollutant, value, standard_id) for pollutant, value in concentrations.items() if pollutant in standard.breakpoints}
    if not indices:
        raise ValidationError("No supported pollutant concentrations were supplied")
    aqi = max(indices.values())
    dominant = sorted(name for name, value in indices.items() if value == aqi)
    return {
        "aqi": aqi,
        "category": category(aqi, standard),
        "dominant_pollutants": dominant,
        "sub_indices": indices,
        "standard": {"id": standard.id, "name": standard.name, "source_url": standard.source_url},
    }


def category(aqi: int, standard: AQIStandard) -> str:
    if standard.country_code == "IN":
        bounds = (50, 100, 200, 300, 400, 500)
    else:
        bounds = (50, 100, 150, 200, 300, 500)
    for upper, label in zip(bounds, standard.categories):
        if aqi <= upper:
            return label
    return standard.categories[-1]


def get_standard(standard_id: str) -> AQIStandard:
    try:
        return STANDARDS[standard_id]
    except KeyError as exc:
        raise ValidationError(f"Unknown AQI standard: {standard_id}") from exc


def standard_for_date(dataset: str, when: date | datetime) -> AQIStandard:
    value = when.date() if isinstance(when, datetime) else when
    if dataset == "CPCB":
        return INDIA_CPCB_2014
    if dataset == "CNEMC":
        return CHINA_HJ633_2026 if value >= CHINA_HJ633_2026.effective_from else CHINA_HJ633_2012
    raise ValidationError(f"No country AQI standard is configured for {dataset}")


def get_guideline(guideline_id: str = DEFAULT_GUIDELINE) -> ConcentrationGuideline:
    try:
        return GUIDELINES[guideline_id]
    except KeyError as exc:
        raise ValidationError(f"Unknown concentration guideline: {guideline_id}") from exc
