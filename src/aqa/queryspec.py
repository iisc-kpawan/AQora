from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Dataset(str, Enum):
    CPCB = "CPCB"
    CNEMC = "CNEMC"
    ALL = "ALL"


class GeographyLevel(str, Enum):
    station = "station"
    city = "city"
    state = "state"
    country = "country"
    global_ = "global"


class Resolution(str, Enum):
    hourly = "hourly"
    daily = "daily"
    weekly = "weekly"
    monthly = "monthly"
    yearly = "yearly"


class Metric(str, Enum):
    mean = "mean"
    median = "median"
    min = "min"
    max = "max"
    std = "std"
    variance = "variance"
    p05 = "p05"
    p25 = "p25"
    p75 = "p75"
    p95 = "p95"
    count = "count"
    coverage = "coverage"


class Operation(str, Enum):
    aggregate = "aggregate"
    compare = "compare"
    rank = "rank"
    trend = "trend"
    distribution = "distribution"
    correlation = "correlation"
    exceedance = "exceedance"
    quality = "quality"
    aqi = "aqi"


class ValueStatus(str, Enum):
    any = "any"
    observed = "observed"
    imputed = "imputed"
    historical_unknown = "historical_unknown"


class CorrelationMethod(str, Enum):
    pearson = "pearson"
    spearman = "spearman"
    kendall = "kendall"


class TimeRange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: datetime | None = None
    end: datetime | None = None
    period: str | None = None

    @field_validator("start", "end", mode="after")
    @classmethod
    def _drop_timezone(cls, value: datetime | None) -> datetime | None:
        # All dataset bounds and downstream comparisons are naive local timestamps;
        # a model-supplied offset (e.g. "...+05:30" or "...Z") must not survive here.
        if value is not None and value.tzinfo is not None:
            return value.replace(tzinfo=None)
        return value

    @model_validator(mode="after")
    def one_mode(self):
        explicit = self.start is not None or self.end is not None
        if self.period and explicit:
            raise ValueError("Use either period or explicit start/end, not both")
        if explicit and (self.start is None or self.end is None):
            raise ValueError("Both start and end are required")
        if not self.period and not explicit:
            self.period = "last month"
        if self.start and self.end and self.start > self.end:
            raise ValueError("start must not be after end")
        return self


class QueryFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value_status: ValueStatus = ValueStatus.any
    minimum_coverage: float | None = Field(default=None, ge=0, le=1)
    threshold: float | None = None
    correlation_method: CorrelationMethod = CorrelationMethod.pearson


class QuerySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: Dataset
    operation: Operation = Operation.aggregate
    geography_level: GeographyLevel = GeographyLevel.city
    locations: list[str] = Field(min_length=1, max_length=50)
    pollutants: list[str] = Field(min_length=1, max_length=6)
    time_range: TimeRange = Field(default_factory=TimeRange)
    resolution: Resolution = Resolution.daily
    metrics: list[Metric] = Field(default_factory=lambda: [Metric.mean], min_length=1, max_length=8)
    filters: QueryFilters = Field(default_factory=QueryFilters)
    limit: int = Field(default=10_000, ge=1, le=100_000)
    visualization: str | None = Field(default=None, pattern="^(line|bar|heatmap|table|scatter|map|none)$")
    group_by: GeographyLevel | None = None
    top_n: int = Field(default=10, ge=1, le=100)
    aqi_standard: str | None = Field(default=None, pattern="^(IN_CPCB_NAQI_2014|CN_HJ633_2026|CN_HJ633_2012)$")

    @model_validator(mode="after")
    def geography_is_meaningful(self):
        if self.geography_level == GeographyLevel.global_ and self.locations != ["global"]:
            raise ValueError("global geography requires locations=['global']")
        if self.operation == Operation.correlation and len(self.pollutants) != 2:
            raise ValueError("correlation requires exactly two pollutants")
        if self.operation == Operation.exceedance and self.filters.threshold is None:
            raise ValueError("exceedance requires filters.threshold")
        if self.operation == Operation.aqi and self.dataset == Dataset.ALL:
            raise ValueError("AQI queries must use CPCB or CNEMC because national standards differ")
        return self
