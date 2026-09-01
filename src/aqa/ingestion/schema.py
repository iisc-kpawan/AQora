from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import duckdb


CONTROL_SCHEMA = """
CREATE TABLE IF NOT EXISTS ingestion_runs (
    ingestion_run_id VARCHAR PRIMARY KEY,
    dataset VARCHAR NOT NULL,
    source VARCHAR NOT NULL,
    requested_start TIMESTAMP,
    requested_end TIMESTAMP,
    actual_start TIMESTAMP,
    actual_end TIMESTAMP,
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    rows_downloaded BIGINT DEFAULT 0,
    rows_parsed BIGINT DEFAULT 0,
    rows_valid BIGINT DEFAULT 0,
    rows_rejected BIGINT DEFAULT 0,
    rows_observed BIGINT DEFAULT 0,
    rows_imputed BIGINT DEFAULT 0,
    rows_inserted BIGINT DEFAULT 0,
    rows_updated_within_grace BIGINT DEFAULT 0,
    status VARCHAR NOT NULL,
    error VARCHAR,
    artifact_sha256 VARCHAR,
    processing_version VARCHAR NOT NULL
);
CREATE TABLE IF NOT EXISTS source_checkpoints (
    dataset VARCHAR NOT NULL,
    source VARCHAR NOT NULL,
    last_successful_ingestion TIMESTAMP,
    last_finalized_timestamp TIMESTAMP,
    latest_source_timestamp_seen TIMESTAMP,
    PRIMARY KEY(dataset, source)
);
CREATE TABLE IF NOT EXISTS source_conflicts (
    ingestion_run_id VARCHAR NOT NULL,
    dataset VARCHAR NOT NULL,
    source_station_id VARCHAR NOT NULL,
    pollutant VARCHAR NOT NULL,
    timestamp_utc TIMESTAMP NOT NULL,
    chosen_source VARCHAR NOT NULL,
    rejected_source VARCHAR NOT NULL,
    chosen_value DOUBLE,
    rejected_value DOUBLE,
    recorded_at TIMESTAMP NOT NULL
);
"""


def connect_control(path: str) -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect(path)
    connection.execute(CONTROL_SCHEMA)
    return connection


def begin_run(connection, dataset: str, source: str, start: datetime, end: datetime, version: str) -> str:
    run_id = f"{dataset.lower()}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    connection.execute(
        "INSERT INTO ingestion_runs (ingestion_run_id,dataset,source,requested_start,requested_end,started_at,status,processing_version) VALUES (?,?,?,?,?,?,'running',?)",
        [run_id, dataset, source, start, end, datetime.now(timezone.utc), version],
    )
    return run_id

