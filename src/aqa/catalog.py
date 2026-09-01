from __future__ import annotations

import csv
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd

from .config import load_config
from .errors import NotFoundError


POLLUTANT_SUFFIXES = {
    "PM2.5": ("PM25", "PM2.5", "PM2.5 (µg/m³)", "µg/m³"),
    "PM10": ("PM10", "PM10", "PM10 (µg/m³)", "µg/m³"),
    "NO2": ("NO2", "NO2", "NO2 (µg/m³)", "µg/m³"),
    "SO2": ("SO2", "SO2", "SO2 (µg/m³)", "µg/m³"),
    "Ozone": ("O3", "Ozone", "Ozone (µg/m³)", "µg/m³"),
    "CO": ("CO", "CO", "CO (mg/m³)", "mg/m³"),
}

POLLUTANT_ALIASES = {
    "pm2.5": "PM25", "pm25": "PM25", "pm₂.₅": "PM25",
    "pm10": "PM10", "pm₁₀": "PM10", "no2": "NO2", "no₂": "NO2",
    "so2": "SO2", "so₂": "SO2", "co": "CO", "o3": "O3",
    "o₃": "O3", "ozone": "O3",
}

CHINA_CITY_ALIASES = {
    "beijing": "北京", "shanghai": "上海", "guangzhou": "广州",
    "shenzhen": "深圳", "tianjin": "天津", "chongqing": "重庆",
    "chengdu": "成都", "wuhan": "武汉", "nanjing": "南京",
    "hangzhou": "杭州", "xian": "西安", "xi'an": "西安",
}

BOARD_STATES = {
    "APPCB": "Andhra Pradesh", "APSPCB": "Andhra Pradesh",
    "ASPCB": "Assam", "PCBA": "Assam", "BSPCB": "Bihar",
    "CGPCB": "Chhattisgarh", "DPCC": "Delhi", "IMD": None,
    "GPCB": "Gujarat", "HSPCB": "Haryana", "HPSPCB": "Himachal Pradesh",
    "JSPCB": "Jharkhand", "KSPCB": "Karnataka", "KSPCB": "Karnataka",
    "KSPCB": "Karnataka", "MPCB": "Maharashtra", "MPPCB": "Madhya Pradesh",
    "OSPCB": "Odisha", "PPCB": "Punjab", "RSPCB": "Rajasthan",
    "TSPCB": "Telangana", "TNPCB": "Tamil Nadu", "UPPCB": "Uttar Pradesh",
    "UKPCB": "Uttarakhand", "WBPCB": "West Bengal",
}


@dataclass(frozen=True)
class Station:
    dataset: str
    station_id: str
    station_name: str | None
    city: str | None
    state: str | None
    country: str
    country_code: str
    latitude: float | None
    longitude: float | None
    geography_status: str


@dataclass(frozen=True)
class SeriesFile:
    dataset: str
    station_id: str
    pollutant: str
    header: str
    unit: str
    path: str


def canonical_pollutant(value: str) -> str:
    key = value.strip().lower().replace(" ", "")
    try:
        return POLLUTANT_ALIASES[key]
    except KeyError as exc:
        raise NotFoundError(f"Unknown pollutant: {value}") from exc


def _parse_pollutant(filename: str) -> tuple[str, str, str]:
    stem = Path(filename).stem
    for suffix, (pollutant, _, header, unit) in POLLUTANT_SUFFIXES.items():
        if stem.endswith("_" + suffix):
            return pollutant, header, unit
    raise ValueError(f"Unsupported series filename: {filename}")


def _cpcb_geography(name: str) -> tuple[str | None, str | None, str]:
    tokens = name.split("_")
    agency = tokens[-1] if tokens else ""
    if len(tokens) >= 2 and tokens[-2:] == ["Kerala", "PCB"]:
        agency = "Kerala_PCB"
        city = tokens[-3].replace("-", " ") if len(tokens) >= 3 else None
        return city, "Kerala", "derived_from_station_name"
    city = tokens[-2].replace("-", " ") if len(tokens) >= 2 else None
    state = BOARD_STATES.get(agency)
    if state is None and city and city.casefold() == "delhi":
        state = "Delhi"
    return city, state, "derived_from_station_name"


class Catalog:
    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.stations: list[Station] = []
        self.series: list[SeriesFile] = []
        self._build()

    def _build(self) -> None:
        for dataset, cfg in self.config["datasets"].items():
            metadata = self._metadata(dataset, cfg)
            root = Path(cfg["historical_path"])
            available: set[str] = set()
            for path in sorted(root.glob("*.csv")):
                pollutant, header, unit = _parse_pollutant(path.name)
                station_id = path.name.split("_", 2)[1]
                if dataset == "CPCB":
                    station_id = "site_" + station_id
                available.add(station_id)
                self.series.append(SeriesFile(dataset, station_id, pollutant, header, unit, str(path.resolve())))
            incremental = Path(self.config["project"]["data_root"]) / "curated" / "hourly" / f"dataset={dataset}"
            incremental_paths = [str(path) for path in incremental.glob("year=*/month=*/*.parquet")]
            if incremental_paths:
                rows = duckdb.sql("SELECT DISTINCT source_station_id FROM read_parquet(?, hive_partitioning=true)", params=[incremental_paths]).fetchall()
                available.update(row[0] for row in rows)
            for station_id in sorted(available):
                row = metadata.get(station_id)
                if row:
                    self.stations.append(row)
                else:
                    self.stations.append(Station(
                        dataset, station_id, None, None, None, cfg["country_name"],
                        cfg["country_code"], None, None, "unresolved",
                    ))

    def _metadata(self, dataset: str, cfg: dict) -> dict[str, Station]:
        out: dict[str, Station] = {}
        with Path(cfg["metadata_path"]).open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if dataset == "CPCB":
                    station_id = row["site_ids"]
                    name = row["site_names"]
                    city, state, status = _cpcb_geography(name)
                else:
                    station_id = row["Site_code"]
                    name, city, state, status = row["Site_name"], row["City"], None, "authoritative_city"
                out[station_id] = Station(
                    dataset, station_id, name, city, state, cfg["country_name"],
                    cfg["country_code"], _float(row.get("Latitude")),
                    _float(row.get("Longitude")), status,
                )
        return out

    def persist(self, path: str | Path | None = None) -> Path:
        target = Path(path or self.config["project"]["catalog_database"])
        target.parent.mkdir(parents=True, exist_ok=True)
        connection = duckdb.connect(str(target))
        try:
            connection.execute("DROP TABLE IF EXISTS stations")
            connection.execute("DROP TABLE IF EXISTS series_files")
            station_frame = pd.DataFrame([asdict(item) for item in self.stations])
            series_frame = pd.DataFrame([asdict(item) for item in self.series])
            connection.register("station_frame", station_frame)
            connection.register("series_frame", series_frame)
            connection.execute("CREATE TABLE stations AS SELECT * FROM station_frame")
            connection.execute('CREATE TABLE series_files AS SELECT * FROM series_frame')
        finally:
            connection.close()
        return target

    def resolve_stations(self, dataset: str, level: str, locations: Iterable[str]) -> list[Station]:
        candidates = [station for station in self.stations if station.dataset == dataset]
        wanted = [self._location_alias(dataset, item).casefold() for item in locations]
        if level == "global":
            return candidates
        field = {"station": "station_name", "city": "city", "state": "state", "country": "country"}[level]
        found = [station for station in candidates if any(
            target == str(getattr(station, field) or "").casefold()
            or target == station.station_id.casefold()
            for target in wanted
        )]
        if not found:
            raise NotFoundError(f"No {dataset} {level} matched {list(locations)}")
        return found

    @staticmethod
    def _location_alias(dataset: str, value: str) -> str:
        if dataset == "CNEMC":
            return CHINA_CITY_ALIASES.get(value.strip().casefold(), value.strip())
        aliases = {"bangalore": "Bengaluru", "new delhi": "Delhi"}
        return aliases.get(value.strip().casefold(), value.strip())

    def files_for(self, dataset: str, station_ids: set[str], pollutant: str) -> list[SeriesFile]:
        return [item for item in self.series if item.dataset == dataset and item.station_id in station_ids and item.pollutant == pollutant]


def _float(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None
