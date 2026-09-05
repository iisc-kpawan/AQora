from __future__ import annotations

import re
import time
from datetime import datetime
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from .capabilities import CapabilityAssessment, is_flexible_country_cohort
from .config import load_config
from .errors import DataUnavailableError
from .global_analytics import POLLUTANTS, _period, _top_n
from .lakehouse_catalog import LakehouseCatalog
from .provenance_analytics import _percentage_after


GAP_LENGTH_PATTERN = re.compile(r"\bgap[- ]lengths?\b|\bgap\s+statistics\b|\blongest\b.{0,30}\bgap\b")


def is_gap_length_question(text: str) -> bool:
    if "reliability" in text and "coverage" in text and "imputation fraction" in text:
        return False
    return bool(GAP_LENGTH_PATTERN.search(text))


class GapStatisticsAnalytics:
    """Detects missing-data episodes per station and reports gap-length statistics.

    A "gap" is a maximal run of consecutive hours where a station's value was
    not directly measured (value_status != 'observed'), regardless of whether
    the pipeline later filled it -- the same "observed vs not" boundary
    ProvenanceAnalytics already uses. Episodes are found with the standard
    SQL gaps-and-islands technique (a row's rank within its own timestamp
    order minus its rank within same-status runs is constant inside one run
    and changes at every transition), which only works because the raw
    hourly layer carries one row per expected hour even when that hour is
    missing -- confirmed against the real data before relying on it.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.catalog = LakehouseCatalog(self.config)
        self.root = self.catalog.root

    @staticmethod
    def should_handle(question: str, assessment: CapabilityAssessment) -> bool:
        del assessment
        return is_gap_length_question(question.casefold())

    def execute(self, question: str, assessment: CapabilityAssessment) -> dict[str, Any]:
        started = time.perf_counter()
        text = question.casefold()
        pollutant = next((name for pattern, name in POLLUTANTS if re.search(pattern, text)), "PM25")
        selected = list(self.catalog.selected(assessment.supported_datasets))
        if is_flexible_country_cohort(question, assessment.intent):
            selected = list(self.catalog.datasets)
        start, end = _period(text, self.catalog, [item.dataset for item in selected])
        paths = self._paths(selected, start, end)
        if not paths:
            raise DataUnavailableError("No hourly lakehouse partitions overlap the requested gap analysis")

        gaps, totals = self._gaps_and_totals(paths, pollutant, start, end)
        if totals.empty:
            raise DataUnavailableError("No hourly station series overlap the requested gap analysis")
        country_by_dataset = {item.dataset: item.country_name for item in selected}
        totals["country"] = totals["dataset"].map(country_by_dataset)
        totals["missingness"] = totals["missing_hours"] / totals["expected_hours"]

        minimum_missingness = _percentage_after(text, ("missingness",), comparison="more")
        ranked_by_missingness = bool(re.search(r"highest missingness|most missing", text))
        station_mode = "station" in text or minimum_missingness is not None or ranked_by_missingness

        if minimum_missingness is not None:
            totals = totals[totals["missingness"] > minimum_missingness]
        if ranked_by_missingness:
            totals = totals.sort_values("missingness", ascending=False).head(_requested_count(text))
        if totals.empty:
            condition = f"more than {minimum_missingness:.0%} missingness" if minimum_missingness is not None else "the requested missingness criterion"
            raise DataUnavailableError(f"No station meets {condition}")

        qualifying = totals[["dataset", "source_station_id"]]
        gaps = gaps.merge(qualifying, on=["dataset", "source_station_id"], how="inner")

        if station_mode:
            rows = _station_rows(totals, gaps)
            visualization = {"type": "bar", "x": "source_station_id", "y": "missingness"}
        else:
            rows = _country_rows(totals, gaps)
            visualization = {"type": "bar", "x": "country", "y": "longest_gap_hours"}
        if not rows:
            raise DataUnavailableError("No gap statistics are available for the requested scope")

        answer = _answer(rows, pollutant, start, end, station_mode, minimum_missingness, ranked_by_missingness)
        return {
            "kind": "analytics",
            "query_spec": {
                "dataset": "GLOBAL_LAKEHOUSE" if len(selected) > 1 else selected[0].dataset,
                "operation": "gap_length_statistics",
                "geography_level": "station" if station_mode else "country",
                "locations": sorted({row["country"] for row in rows}), "pollutants": [pollutant],
                "time_range": {"start": start.isoformat(), "end": end.isoformat()}, "resolution": "period",
                "metrics": ["gap_count", "longest_gap_hours", "mean_gap_hours", "median_gap_hours", "p95_gap_hours"],
                "filters": {"minimum_missingness": minimum_missingness},
            },
            "resolved_period": {"start": start.isoformat(), "end": end.isoformat(), "basis": "explicit requested period"},
            "data": rows,
            "metadata": {
                "dataset": "GLOBAL_LAKEHOUSE", "geography_level": "station" if station_mode else "country",
                "station_count": len(totals), "value_status": "observed vs not observed",
                "station_balanced": not station_mode,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2), "warnings": [],
            },
            "visualization": visualization,
            "analysis": {
                "operator": "gap_length_statistics_v1",
                "gap_definition": "maximal run of consecutive hours with value_status != 'observed'",
                "minimum_missingness": minimum_missingness, "ranked_by_missingness": ranked_by_missingness,
            },
            "answer": answer, "display_title": f"{pollutant} · gap-length statistics",
            "model_trace": {"provider": "deterministic", "model": "gap_length_statistics_v1", "latency_ms": 0,
                            "prompt_tokens": None, "completion_tokens": None, "fallback_used": False},
        }

    def _paths(self, selected, start: datetime, end: datetime) -> list[str]:
        paths = []
        for item in selected:
            layer = "query/hourly" if start >= item.hourly_start else "archive/hourly"
            current = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            while current <= end:
                directory = self.root / layer / f"dataset={item.dataset}" / f"year={current.year}" / f"month={current.month}"
                paths.extend(str(path) for path in directory.glob("*.parquet"))
                current = current.replace(year=current.year + 1, month=1) if current.month == 12 else current.replace(month=current.month + 1)
        return paths

    def _gaps_and_totals(self, paths: list[str], pollutant: str, start: datetime, end: datetime):
        connection = duckdb.connect()
        connection.execute("SET threads=8")
        try:
            gaps = connection.execute("""
                WITH raw AS (
                  SELECT dataset,source_station_id,timestamp_local,
                         CASE WHEN value_status='observed' THEN 1 ELSE 0 END is_observed
                  FROM read_parquet(?,hive_partitioning=true,union_by_name=true)
                  WHERE pollutant=? AND timestamp_local BETWEEN ? AND ?
                ), numbered AS (
                  SELECT *,
                         row_number() OVER (PARTITION BY dataset,source_station_id ORDER BY timestamp_local) rn,
                         row_number() OVER (PARTITION BY dataset,source_station_id,is_observed ORDER BY timestamp_local) rn2
                  FROM raw
                )
                SELECT dataset,source_station_id,is_observed,(rn-rn2) grp,
                       min(timestamp_local) gap_start,max(timestamp_local) gap_end,count(*) gap_hours
                FROM numbered GROUP BY dataset,source_station_id,is_observed,grp
            """, [paths, pollutant, start, end]).fetchdf()
            totals = connection.execute("""
                SELECT dataset,source_station_id,count(*) expected_hours,
                       sum(CASE WHEN value_status='observed' THEN 0 ELSE 1 END) missing_hours
                FROM read_parquet(?,hive_partitioning=true,union_by_name=true)
                WHERE pollutant=? AND timestamp_local BETWEEN ? AND ?
                GROUP BY dataset,source_station_id
            """, [paths, pollutant, start, end]).fetchdf()
        finally:
            connection.close()
        gaps = gaps[gaps["is_observed"] == 0].drop(columns=["is_observed", "grp"]).reset_index(drop=True)
        return gaps, totals


def _requested_count(text: str, default: int = 10) -> int:
    match = re.search(r"\b(\d+)\s+(?:stations?|series|countr(?:y|ies)|cities|pairs)\b", text)
    if match:
        return int(match.group(1))
    return _top_n(text) if re.search(r"top\s+\d+|\btop\b", text) else default


def _gap_stats(gap_hours: np.ndarray) -> dict[str, float | None]:
    if len(gap_hours) == 0:
        return {"gap_count": 0, "longest_gap_hours": 0.0, "mean_gap_hours": None,
                "median_gap_hours": None, "p95_gap_hours": None}
    return {
        "gap_count": int(len(gap_hours)), "longest_gap_hours": float(gap_hours.max()),
        "mean_gap_hours": float(gap_hours.mean()), "median_gap_hours": float(np.median(gap_hours)),
        "p95_gap_hours": float(np.quantile(gap_hours, .95)),
    }


def _station_rows(totals: pd.DataFrame, gaps: pd.DataFrame) -> list[dict]:
    rows = []
    for _, station in totals.iterrows():
        station_gaps = gaps[
            (gaps["dataset"] == station["dataset"]) & (gaps["source_station_id"] == station["source_station_id"])
        ]
        gap_hours = station_gaps["gap_hours"].to_numpy(dtype=float)
        stats = _gap_stats(gap_hours)
        longest = station_gaps.loc[station_gaps["gap_hours"].idxmax()] if len(station_gaps) else None
        rows.append({
            "dataset": station["dataset"], "country": station["country"],
            "source_station_id": station["source_station_id"],
            "missingness": float(station["missingness"]),
            "longest_gap_start": longest["gap_start"].isoformat() if longest is not None else None,
            "longest_gap_end": longest["gap_end"].isoformat() if longest is not None else None,
            **stats,
        })
    return sorted(rows, key=lambda row: row["missingness"], reverse=True)


def _country_rows(totals: pd.DataFrame, gaps: pd.DataFrame) -> list[dict]:
    rows = []
    for country, group in totals.groupby("country", sort=True):
        keys = group[["dataset", "source_station_id"]]
        country_gaps = gaps.merge(keys, on=["dataset", "source_station_id"], how="inner")
        gap_hours = country_gaps["gap_hours"].to_numpy(dtype=float)
        stats = _gap_stats(gap_hours)
        longest = country_gaps.loc[country_gaps["gap_hours"].idxmax()] if len(country_gaps) else None
        rows.append({
            "dataset": group["dataset"].iloc[0], "country": country, "station_count": int(len(group)),
            "overall_missingness": float(group["missing_hours"].sum() / group["expected_hours"].sum()),
            "longest_gap_station": longest["source_station_id"] if longest is not None else None,
            "longest_gap_start": longest["gap_start"].isoformat() if longest is not None else None,
            "longest_gap_end": longest["gap_end"].isoformat() if longest is not None else None,
            **stats,
        })
    return rows


def _answer(rows: list[dict], pollutant: str, start: datetime, end: datetime,
            station_mode: bool, minimum_missingness: float | None, ranked_by_missingness: bool) -> str:
    period = f"{start:%d %b %Y} to {end:%d %b %Y}"
    scope = "station" if station_mode else "country"
    parts = []
    for row in rows:
        label = row["source_station_id"] if station_mode else row["country"]
        if row["gap_count"] == 0:
            parts.append(f"{label} has no non-observed hours (no gaps).")
            continue
        parts.append(
            f"{label}: {row['gap_count']} gap(s), longest {row['longest_gap_hours']:.0f}h "
            f"(mean {row['mean_gap_hours']:.1f}h, median {row['median_gap_hours']:.1f}h, p95 {row['p95_gap_hours']:.1f}h)"
        )
    answer = (
        f"For {period}, I detected {pollutant} hourly missing-data episodes (maximal runs where a station's "
        f"value was not directly measured) per {scope}. " + "; ".join(parts) + "."
    )
    if minimum_missingness is not None:
        answer += f" Restricted to stations with more than {minimum_missingness:.0%} missingness."
    if ranked_by_missingness:
        answer += " Stations are ranked by missingness, highest first."
    return answer
