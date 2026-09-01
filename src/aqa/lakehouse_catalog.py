from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import duckdb

from .config import load_config


@dataclass(frozen=True)
class DatasetAvailability:
    dataset: str
    network: str
    country_code: str
    country_name: str
    available_start: datetime
    available_end: datetime
    hourly_start: datetime


@dataclass(frozen=True)
class CohortCandidate:
    dataset: str
    country_code: str
    country_name: str
    included: bool
    reason_code: str | None
    reason: str | None
    coverage: float
    active_stations: int
    expected_count: int
    selected_count: int


class LakehouseCatalog:
    """Read-only metadata access for the atomically published global lakehouse."""

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.root = Path(self.config["project"]["data_root"]).resolve() / "lakehouse"
        self.control_path = self.root / "control" / "retention.duckdb"
        if not self.control_path.is_file():
            raise FileNotFoundError(f"Lakehouse control database is missing: {self.control_path}")

    @property
    def datasets(self) -> tuple[DatasetAvailability, ...]:
        return _dataset_rows(str(self.control_path), self.control_path.stat().st_mtime_ns)

    def by_dataset(self, dataset: str) -> DatasetAvailability:
        for item in self.datasets:
            if item.dataset == dataset:
                return item
        raise KeyError(f"Unknown lakehouse dataset: {dataset}")

    def selected(self, datasets: Iterable[str] | None = None) -> tuple[DatasetAvailability, ...]:
        wanted = set(datasets or ())
        return tuple(item for item in self.datasets if not wanted or item.dataset in wanted)

    @property
    def availability_rows(self) -> tuple[dict, ...]:
        monthly_root = self.root / "query" / "monthly"
        marker = self.control_path.stat().st_mtime_ns
        return _pollutant_rows(str(monthly_root), marker)

    @property
    def country_rows(self) -> tuple[dict, ...]:
        monthly_root = self.root / "query" / "monthly"
        marker = self.control_path.stat().st_mtime_ns
        return _country_rows(str(monthly_root), marker)

    def provenance_rows(
        self,
        datasets: Iterable[str],
        pollutant: str | None = None,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> tuple[dict, ...]:
        """Return value-lineage counts without treating filled values as measurements."""
        selected = self.selected(datasets)
        if not selected:
            return ()
        first_month = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0) if start else None
        last_month = end.replace(day=1, hour=0, minute=0, second=0, microsecond=0) if end else None
        connection = duckdb.connect(str(self.control_path), read_only=True)
        try:
            result = connection.execute("""
                SELECT dataset, any_value(network) network,
                       any_value(country_code) country_code, any_value(country_name) country,
                       pollutant, any_value(unit) unit,
                       sum(expected_count)::BIGINT expected_count,
                       sum(usable_count)::BIGINT usable_count,
                       sum(observed_count)::BIGINT observed_count,
                       sum(legacy_imputed_count)::BIGINT legacy_filled_count,
                       sum(guarded_imputed_count)::BIGINT guarded_filled_count,
                       sum(unresolved_count)::BIGINT unresolved_count,
                       sum(invalid_count)::BIGINT invalid_count,
                       min(first_usable_timestamp) first_usable_timestamp,
                       max(last_usable_timestamp) last_usable_timestamp,
                       min(month_start) FILTER (WHERE observed_count>0) first_observed_month,
                       max(month_start) FILTER (WHERE observed_count>0) last_observed_month,
                       max(active_station_count)::BIGINT active_station_count
                FROM pollutant_monthly_availability
                WHERE dataset IN (SELECT unnest(?))
                  AND (? IS NULL OR pollutant=?)
                  AND (? IS NULL OR month_start>=?)
                  AND (? IS NULL OR month_start<=?)
                GROUP BY dataset, pollutant ORDER BY dataset, pollutant
            """, [
                [item.dataset for item in selected], pollutant, pollutant,
                first_month, first_month, last_month, last_month,
            ])
            names = [item[0] for item in result.description]
            rows = [dict(zip(names, row)) for row in result.fetchall()]
        finally:
            connection.close()
        found = {(row["dataset"], row["pollutant"]) for row in rows}
        requested_pollutants = [pollutant] if pollutant else sorted({row["pollutant"] for row in rows})
        for item in selected:
            for requested in requested_pollutants:
                if (item.dataset, requested) not in found:
                    rows.append({
                        "dataset": item.dataset, "network": item.network,
                        "country_code": item.country_code, "country": item.country_name,
                        "pollutant": requested, "unit": None, "expected_count": 0,
                        "usable_count": 0, "observed_count": 0, "legacy_filled_count": 0,
                        "guarded_filled_count": 0, "unresolved_count": 0, "invalid_count": 0,
                        "first_usable_timestamp": None, "last_usable_timestamp": None,
                        "first_observed_month": None, "last_observed_month": None,
                        "active_station_count": 0,
                    })
        return tuple(sorted(rows, key=lambda row: (row["dataset"], row["pollutant"])))

    def resolve_cohort(
        self, datasets: Iterable[str], pollutant: str, start: datetime, end: datetime, *,
        value_status: str = "any", minimum_coverage: float | None = None,
        minimum_stations: int | None = None, require_all_months: bool = False,
    ) -> tuple[CohortCandidate, ...]:
        selected = self.selected(datasets)
        first_month = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        last_month = end.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        connection = duckdb.connect(str(self.control_path), read_only=True)
        try:
            rows = connection.execute("""
                SELECT dataset, any_value(country_code), any_value(country_name),
                       count(DISTINCT month_start) month_count,
                       count(DISTINCT month_start) FILTER (
                           WHERE CASE WHEN ?='observed' THEN observed_count ELSE usable_count END > 0
                       ) value_month_count,
                       sum(expected_count) expected_count,
                       sum(CASE WHEN ?='observed' THEN observed_count ELSE usable_count END) selected_count,
                       max(CASE WHEN ?='observed' THEN observed_station_count ELSE usable_station_count END) active_stations
                FROM pollutant_monthly_availability
                WHERE pollutant=? AND month_start BETWEEN ? AND ?
                  AND dataset IN (SELECT unnest(?))
                GROUP BY dataset
            """, [value_status, value_status, value_status, pollutant, first_month, last_month, [item.dataset for item in selected]]).fetchall()
        finally:
            connection.close()
        found = {row[0]: row for row in rows}
        required_months = (last_month.year - first_month.year) * 12 + last_month.month - first_month.month + 1
        output = []
        for item in selected:
            row = found.get(item.dataset)
            if row is None:
                output.append(CohortCandidate(item.dataset, item.country_code, item.country_name, False,
                                              "no_rows", f"no {pollutant} rows overlap the requested period", 0.0, 0, 0, 0))
                continue
            month_count, value_month_count = int(row[3]), int(row[4])
            expected, selected_count, stations = int(row[5] or 0), int(row[6] or 0), int(row[7] or 0)
            coverage = selected_count / expected if expected else 0.0
            reason_code = None
            reason = None
            if selected_count == 0:
                reason_code = "no_observed_values" if value_status == "observed" else "no_usable_values"
                reason = f"rows are present but contain no {value_status if value_status != 'any' else 'usable'} values"
            elif require_all_months and value_month_count < required_months:
                reason_code = "partial_period"
                reason = f"only {value_month_count} of {required_months} requested months contain {value_status if value_status != 'any' else 'usable'} values"
            elif minimum_coverage is not None and coverage < minimum_coverage:
                reason_code = "coverage_below_threshold"
                reason = f"coverage {coverage:.1%} is below the required {minimum_coverage:.1%}"
            elif minimum_stations is not None and stations < minimum_stations:
                reason_code = "stations_below_threshold"
                reason = f"{stations} active stations is below the required {minimum_stations}"
            output.append(CohortCandidate(item.dataset, item.country_code, item.country_name, reason is None,
                                          reason_code, reason, coverage, stations, expected, selected_count))
        return tuple(output)


@lru_cache(maxsize=4)
def _dataset_rows(path: str, marker: int) -> tuple[DatasetAvailability, ...]:
    del marker
    connection = duckdb.connect(path, read_only=True)
    try:
        rows = connection.execute("""
            SELECT dataset, network, country_code, country_name,
                   available_start, available_end, hourly_start
            FROM dataset_availability ORDER BY dataset
        """).fetchall()
    finally:
        connection.close()
    return tuple(DatasetAvailability(*row) for row in rows)


@lru_cache(maxsize=4)
def _pollutant_rows(root: str, marker: int) -> tuple[dict, ...]:
    del marker
    paths = [str(path) for path in Path(root).glob("dataset=*/year=*/month=*/*.parquet")]
    connection = duckdb.connect()
    try:
        result = connection.execute("""
            SELECT dataset, any_value(network) AS network,
                   any_value(country_code) AS country_code,
                   any_value(country_name) AS country,
                   pollutant, count(DISTINCT source_station_id) AS station_count,
                   count(DISTINCT source_station_id || '|' || pollutant) AS series_count,
                   min(first_valid_timestamp) AS earliest,
                   max(last_valid_timestamp) AS latest,
                   any_value(unit) AS unit
            FROM read_parquet(?, hive_partitioning=true, union_by_name=true)
            GROUP BY dataset, pollutant ORDER BY dataset, pollutant
        """, [paths])
        names = [item[0] for item in result.description]
        return tuple(dict(zip(names, row)) for row in result.fetchall())
    finally:
        connection.close()


@lru_cache(maxsize=4)
def _country_rows(root: str, marker: int) -> tuple[dict, ...]:
    del marker
    paths = [str(path) for path in Path(root).glob("dataset=*/year=*/month=*/*.parquet")]
    connection = duckdb.connect()
    try:
        result = connection.execute("""
            SELECT dataset, any_value(network) AS network,
                   any_value(country_code) AS country_code,
                   any_value(country_name) AS country,
                   count(DISTINCT source_station_id) AS monitoring_sites,
                   count(DISTINCT source_station_id || '|' || pollutant) AS station_pollutant_series,
                   count(DISTINCT network) AS source_families
            FROM read_parquet(?, hive_partitioning=true, union_by_name=true)
            GROUP BY dataset ORDER BY dataset
        """, [paths])
        names = [item[0] for item in result.description]
        return tuple(dict(zip(names, row)) for row in result.fetchall())
    finally:
        connection.close()
