from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..errors import IngestionError


POLLUTANT_COLUMNS = {
    "PM2.5 (µg/m³)": ("PM25", "µg/m³"), "PM10 (µg/m³)": ("PM10", "µg/m³"),
    "NO2 (µg/m³)": ("NO2", "µg/m³"), "SO2 (µg/m³)": ("SO2", "µg/m³"),
    "CO (mg/m³)": ("CO", "mg/m³"), "Ozone (µg/m³)": ("O3", "µg/m³"),
}
CNEMC_TYPES = {"PM2.5": "PM25", "PM10": "PM10", "NO2": "NO2", "SO2": "SO2", "CO": "CO", "O3": "O3"}


def parse_cpcb(path: str | Path, *, source: str, run_id: str) -> pd.DataFrame:
    path = Path(path)
    frame = pd.read_csv(path)
    if "Timestamp" not in frame.columns:
        raise IngestionError(f"CPCB schema drift: Timestamp missing in {path.name}")
    present = [column for column in POLLUTANT_COLUMNS if column in frame.columns]
    if not present:
        raise IngestionError(f"CPCB schema drift: no known pollutant columns in {path.name}")
    timestamps = pd.to_datetime(frame["Timestamp"], errors="coerce")
    if timestamps.isna().any():
        raise IngestionError(f"CPCB contains {timestamps.isna().sum()} invalid timestamps")
    station_id = _cpcb_station_id(path.name)
    pieces = []
    for column in present:
        pollutant, unit = POLLUTANT_COLUMNS[column]
        values = pd.to_numeric(frame[column], errors="coerce")
        hourly = pd.DataFrame({"timestamp_local": timestamps, "observed_value": values}).set_index("timestamp_local").resample("1h").median().reset_index()
        hourly["pollutant"], hourly["unit"] = pollutant, unit
        pieces.append(hourly)
    return _decorate(pd.concat(pieces, ignore_index=True), "CPCB", station_id, source, run_id)


def parse_cnemc(path: str | Path, *, source: str, run_id: str) -> pd.DataFrame:
    path = Path(path)
    frame = pd.read_csv(path)
    required = {"date", "hour", "type"}
    if not required.issubset(frame.columns):
        raise IngestionError(f"CNEMC schema drift: required columns are {sorted(required)}")
    frame = frame[frame["type"].isin(CNEMC_TYPES)].copy()
    frame["timestamp_local"] = pd.to_datetime(frame["date"].astype(str), format="%Y%m%d", errors="coerce") + pd.to_timedelta(frame["hour"], unit="h")
    if frame["timestamp_local"].isna().any():
        raise IngestionError("CNEMC contains invalid date/hour values")
    station_columns = [column for column in frame.columns if column not in {"date", "hour", "type", "timestamp_local"}]
    long = frame.melt(id_vars=["timestamp_local", "type"], value_vars=station_columns, var_name="source_station_id", value_name="observed_value")
    long["observed_value"] = pd.to_numeric(long["observed_value"], errors="coerce")
    long["pollutant"] = long["type"].map(CNEMC_TYPES)
    long["unit"] = long["pollutant"].map(lambda item: "mg/m³" if item == "CO" else "µg/m³")
    long = long.drop(columns="type")
    long["dataset"], long["source"], long["ingestion_run_id"] = "CNEMC", source, run_id
    return long


def _decorate(frame: pd.DataFrame, dataset: str, station_id: str, source: str, run_id: str) -> pd.DataFrame:
    frame["dataset"] = dataset
    frame["source_station_id"] = station_id
    frame["source"] = source
    frame["ingestion_run_id"] = run_id
    return frame


def _cpcb_station_id(filename: str) -> str:
    parts = filename.split("_")
    if "site" not in parts:
        raise IngestionError(f"Cannot resolve CPCB station id from {filename}")
    index = parts.index("site")
    if index + 1 >= len(parts):
        raise IngestionError(f"Cannot resolve CPCB station id from {filename}")
    return f"site_{parts[index + 1]}"
