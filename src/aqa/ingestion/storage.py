from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import duckdb
import pandas as pd


KEY = ["dataset", "source_station_id", "pollutant", "timestamp_local"]


def write_partitions(frame: pd.DataFrame, root: str | Path, *, source_priority: list[str], conflict_root: str | Path | None = None, run_id: str | None = None) -> dict[str, int | str | None]:
    root = Path(root)
    frame = frame.copy()
    frame["year"] = frame["timestamp_local"].dt.year
    frame["month"] = frame["timestamp_local"].dt.month
    inserted = updated = conflicts = 0
    conflict_frames = []
    for (dataset, year, month), incoming in frame.groupby(["dataset", "year", "month"]):
        directory = root / "hourly" / f"dataset={dataset}" / f"year={year:04d}" / f"month={month:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "observations.parquet"
        incoming = incoming.drop(columns=["year", "month"]).drop_duplicates(KEY, keep="last")
        if target.exists():
            previous = duckdb.sql("SELECT * FROM read_parquet(?)", params=[str(target)]).df()
            previous["_arrival"] = 0
        else:
            previous = pd.DataFrame(columns=incoming.columns)
            previous["_arrival"] = pd.Series(dtype=int)
        incoming["_arrival"] = 1
        combined = pd.concat([previous, incoming], ignore_index=True)
        rank = {name: index for index, name in enumerate(source_priority)}
        combined["_priority"] = combined["source"].map(rank).fillna(len(rank))
        conflict_keys = (
            combined.groupby(KEY, dropna=False)
            .agg(source_count=("source", "nunique"), value_count=("analysis_value", lambda values: values.nunique(dropna=False)))
            .reset_index()
            .query("source_count > 1 and value_count > 1")
        )
        conflicts += len(conflict_keys)
        if not conflict_keys.empty:
            audit = combined.merge(conflict_keys[KEY], on=KEY, how="inner")[KEY + ["source", "analysis_value", "ingestion_run_id"]].copy()
            audit["recorded_for_run"] = run_id
            conflict_frames.append(audit)
        combined = combined.sort_values(KEY + ["_priority", "_arrival"], ascending=[True] * len(KEY) + [True, False]).drop_duplicates(KEY, keep="first")
        if target.exists():
            comparison = combined.merge(previous[KEY + ["analysis_value", "value_status", "source"]], on=KEY, how="left", suffixes=("", "_previous"), indicator=True)
            inserted += int((comparison["_merge"] == "left_only").sum())
            existing = comparison["_merge"] == "both"
            value_changed = comparison["analysis_value"].fillna(float("inf")) != comparison["analysis_value_previous"].fillna(float("inf"))
            metadata_changed = (comparison["value_status"] != comparison["value_status_previous"]) | (comparison["source"] != comparison["source_previous"])
            updated += int((existing & (value_changed | metadata_changed)).sum())
        else:
            inserted += len(combined)
        combined = combined.drop(columns=["_priority", "_arrival"])
        temporary = directory / f".{target.name}.{uuid4().hex}.tmp"
        connection = duckdb.connect()
        try:
            connection.register("output_frame", combined.sort_values(["pollutant", "source_station_id", "timestamp_local"]))
            connection.execute("COPY output_frame TO ? (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)", [str(temporary)])
        finally:
            connection.close()
        os.replace(temporary, target)
    conflict_path = None
    if conflict_frames and conflict_root:
        conflict_dir = Path(conflict_root)
        conflict_dir.mkdir(parents=True, exist_ok=True)
        conflict_path = conflict_dir / f"{run_id or uuid4().hex}.parquet"
        audit_frame = pd.concat(conflict_frames, ignore_index=True)
        duckdb.sql("COPY audit_frame TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", params=[str(conflict_path)])
    return {"rows_inserted": inserted, "rows_updated": updated, "conflicting_candidates": conflicts, "conflict_artifact": str(conflict_path) if conflict_path else None}
