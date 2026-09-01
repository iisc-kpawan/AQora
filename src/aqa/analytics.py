from __future__ import annotations

import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from .aqi import DEFAULT_STANDARD, calculate_aqi, category, get_standard
from .catalog import Catalog, canonical_pollutant
from .config import load_config
from .errors import DataUnavailableError, ValidationError
from .periods import resolve_period
from .queryspec import GeographyLevel, Operation, QuerySpec, Resolution, ValueStatus


DATE_TRUNC = {
    Resolution.hourly: "hour", Resolution.daily: "day", Resolution.weekly: "week",
    Resolution.monthly: "month", Resolution.yearly: "year",
}

METRIC_SQL = {
    "mean": "avg(station_value)", "median": "median(station_value)",
    "min": "min(station_value)", "max": "max(station_value)",
    "std": "stddev_samp(station_value)", "variance": "var_samp(station_value)",
    "p05": "quantile_cont(station_value, 0.05)", "p25": "quantile_cont(station_value, 0.25)",
    "p75": "quantile_cont(station_value, 0.75)", "p95": "quantile_cont(station_value, 0.95)",
    "count": "count(station_value)",
}


class AnalyticsEngine:
    def __init__(self, config: dict | None = None, catalog: Catalog | None = None, *, prefer_parquet: bool = True):
        self.config = config or load_config()
        self.catalog = catalog or Catalog(self.config)
        self.prefer_parquet = prefer_parquet

    def execute(self, spec: QuerySpec) -> dict[str, Any]:
        started = time.perf_counter()
        dataset = spec.dataset.value
        if dataset == "ALL":
            return self._execute_all(spec, started)
        if spec.operation == Operation.aqi:
            return self._execute_aqi(spec, started)
        level = "global" if spec.geography_level == GeographyLevel.global_ else spec.geography_level.value
        cfg = self.config["datasets"][dataset]
        available_start, available_end = self._available_bounds(dataset)
        if spec.time_range.period and spec.time_range.period.casefold() in {"all", "all available", "all years", "over years"}:
            start, end = available_start, available_end
        elif spec.time_range.period:
            start, end = resolve_period(spec.time_range.period, available_end)
        else:
            start, end = spec.time_range.start, spec.time_range.end
        assert start is not None and end is not None
        start, end = max(start, available_start), min(end, available_end)
        if start > end:
            raise DataUnavailableError("Requested period does not overlap available data")
        stations = self.catalog.resolve_stations(dataset, level, spec.locations)
        station_ids = {station.station_id for station in stations}
        labels = self._labels(spec, stations)
        results: list[dict[str, Any]] = []
        warnings = []
        if self.config["analytics"].get("exclude_negative_values", True):
            warnings.append("Negative historical concentrations were excluded as invalid")
        for requested_pollutant in spec.pollutants:
            pollutant = canonical_pollutant(requested_pollutant)
            files = self.catalog.files_for(dataset, station_ids, pollutant)
            if not files and not self._incremental_paths(dataset, start, end):
                warnings.append(f"No {pollutant} series matched the requested geography")
                continue
            results.extend(self._query_files(files, pollutant, spec, start, end, labels))
        if not results:
            raise DataUnavailableError("No valid observations matched the query")
        results, analysis = self._postprocess(spec, results)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        return {
            "query_spec": spec.model_dump(mode="json"),
            "resolved_period": {"start": start.isoformat(), "end": end.isoformat(), "basis": "latest available data"},
            "data": results,
            "metadata": {
                "dataset": dataset, "geography_level": level, "station_count": len(stations),
                "value_status": self._result_status(results), "station_balanced": True,
                "latency_ms": elapsed_ms, "warnings": warnings,
                "locations": self._location_metadata(stations, labels),
            },
            "visualization": self._visualization(spec, len(results)),
            "analysis": analysis,
        }

    def _execute_aqi(self, spec: QuerySpec, started: float) -> dict[str, Any]:
        if spec.resolution == Resolution.hourly:
            raise ValidationError("AQI is reported daily or at a broader time resolution")
        raw = spec.model_dump(mode="json")
        raw.update({
            "operation": "aggregate", "pollutants": ["PM25", "PM10", "NO2", "SO2", "CO", "O3"],
            "resolution": "hourly", "metrics": ["mean"], "limit": 100_000,
        })
        hourly = self.execute(QuerySpec.model_validate(raw))
        frame = pd.DataFrame(hourly["data"])
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        daily_parts = []
        for (location, pollutant), group in frame.groupby(["location", "pollutant"]):
            series = group.sort_values("timestamp").set_index("timestamp")["mean"].astype(float).asfreq("h")
            if pollutant in {"CO", "O3"}:
                values = series.rolling(8, min_periods=6).mean().resample("D").max()
            else:
                daily = series.resample("D").agg(["mean", "count"])
                values = daily["mean"].where(daily["count"] >= 18)
            daily_parts.append(pd.DataFrame({"timestamp": values.index, "location": location, "pollutant": pollutant, "concentration": values.values}))
        concentrations = pd.concat(daily_parts, ignore_index=True).dropna(subset=["concentration"])
        standard_id = spec.aqi_standard or DEFAULT_STANDARD[spec.dataset.value]
        standard = get_standard(standard_id)
        daily_aqi = []
        for (timestamp, location), group in concentrations.groupby(["timestamp", "location"]):
            values = dict(zip(group["pollutant"], group["concentration"]))
            if not ({"PM25", "PM10"} & values.keys()) or len(values) < 3:
                continue
            computed = calculate_aqi(values, standard_id)
            daily_aqi.append({"timestamp": timestamp, "location": location, "aqi": computed["aqi"], "dominant_pollutants": computed["dominant_pollutants"]})
        if not daily_aqi:
            raise DataUnavailableError("Not enough pollutant coverage to calculate AQI")
        daily_frame = pd.DataFrame(daily_aqi)
        if spec.resolution == Resolution.daily:
            daily_frame["bucket"] = daily_frame["timestamp"].dt.floor("D")
        elif spec.resolution == Resolution.weekly:
            daily_frame["bucket"] = daily_frame["timestamp"] - pd.to_timedelta(daily_frame["timestamp"].dt.weekday, unit="D")
        elif spec.resolution == Resolution.monthly:
            daily_frame["bucket"] = daily_frame["timestamp"].dt.to_period("M").dt.start_time
        else:
            daily_frame["bucket"] = daily_frame["timestamp"].dt.to_period("Y").dt.start_time
        output = []
        for (bucket, location), group in daily_frame.groupby(["bucket", "location"]):
            values = group["aqi"].astype(float)
            row = {
                "timestamp": bucket.isoformat(), "location": location, "pollutant": "AQI", "unit": "AQI",
                "reporting_days": len(values), "coverage": round(len(values) / max(1, (group["timestamp"].max() - group["timestamp"].min()).days + 1), 6),
            }
            for metric in spec.metrics:
                name = metric.value
                if name == "mean": row[name] = float(values.mean())
                elif name == "median": row[name] = float(values.median())
                elif name == "min": row[name] = float(values.min())
                elif name == "max": row[name] = float(values.max())
                elif name == "std": row[name] = float(values.std()) if len(values) > 1 else None
                elif name == "variance": row[name] = float(values.var()) if len(values) > 1 else None
                elif name.startswith("p") and name[1:].isdigit(): row[name] = float(values.quantile(int(name[1:]) / 100))
                elif name == "count": row[name] = len(values)
                elif name == "coverage": row[name] = row["coverage"]
            summary_value = row.get(spec.metrics[0].value)
            if summary_value is not None and spec.metrics[0].value not in {"count", "coverage"}:
                row["category"] = category(round(summary_value), standard)
            dominant = [item for items in group["dominant_pollutants"] for item in items]
            row["dominant_pollutant"] = pd.Series(dominant).mode().iloc[0] if dominant else None
            output.append(row)
        return {
            "query_spec": spec.model_dump(mode="json"), "resolved_period": hourly["resolved_period"], "data": output,
            "metadata": {**hourly["metadata"], "latency_ms": round((time.perf_counter() - started) * 1000, 2), "aqi_standard": {"id": standard.id, "name": standard.name, "source_url": standard.source_url}},
            "visualization": {"type": "line" if len(output) > 1 else "table"},
            "analysis": {"operation": "aqi", "method": "Daily AQI from station-balanced area concentrations; 24-hour means for PM/NO2/SO2 and daily maximum rolling 8-hour means for CO/O3; broader buckets summarize daily AQI", "standard_id": standard.id},
        }

    def _execute_all(self, spec: QuerySpec, started: float) -> dict[str, Any]:
        children = []
        for dataset in ("CPCB", "CNEMC"):
            level = "global" if spec.geography_level == GeographyLevel.global_ else spec.geography_level.value
            if level == "global":
                locations = ["global"]
            else:
                locations = []
                for location in spec.locations:
                    try:
                        self.catalog.resolve_stations(dataset, level, [location])
                        locations.append(location)
                    except Exception:
                        continue
            if not locations:
                continue
            raw = spec.model_dump(mode="json")
            raw.update({"dataset": dataset, "locations": locations})
            children.append(self.execute(QuerySpec.model_validate(raw)))
        if not children:
            raise DataUnavailableError("No locations resolved across CPCB and CNEMC")
        data = []
        for child in children:
            for row in child["data"]:
                data.append({"dataset": child["metadata"]["dataset"], **row})
        starts = [datetime.fromisoformat(child["resolved_period"]["start"]) for child in children]
        ends = [datetime.fromisoformat(child["resolved_period"]["end"]) for child in children]
        statuses = {child["metadata"]["value_status"] for child in children}
        return {
            "query_spec": spec.model_dump(mode="json"),
            "resolved_period": {"start": min(starts).isoformat(), "end": max(ends).isoformat(), "basis": "resolved independently against each dataset"},
            "resolved_periods": {child["metadata"]["dataset"]: child["resolved_period"] for child in children},
            "data": data,
            "metadata": {
                "dataset": "ALL", "geography_level": spec.geography_level.value,
                "station_count": sum(child["metadata"]["station_count"] for child in children),
                "value_status": next(iter(statuses)) if len(statuses) == 1 else "mixed",
                "station_balanced": True, "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "warnings": [f"{child['metadata']['dataset']}: {warning}" for child in children for warning in child["metadata"]["warnings"]],
                "locations": [location for child in children for location in child["metadata"]["locations"]],
            },
            "visualization": self._visualization(spec, len(data)),
            "analysis": {"operation": spec.operation.value, "dataset_results": len(children)},
        }

    def _query_files(self, files, pollutant: str, spec: QuerySpec, start: datetime, end: datetime, labels: dict[str, str]) -> list[dict]:
        defaults = {"PM25": ("PM2.5 (µg/m³)", "µg/m³"), "PM10": ("PM10 (µg/m³)", "µg/m³"), "NO2": ("NO2 (µg/m³)", "µg/m³"), "SO2": ("SO2 (µg/m³)", "µg/m³"), "O3": ("Ozone (µg/m³)", "µg/m³"), "CO": ("CO (mg/m³)", "mg/m³")}
        column, unit = (files[0].header, files[0].unit) if files else defaults[pollutant]
        connection = duckdb.connect()
        connection.execute("SET threads TO 4")
        connection.execute("CREATE TEMP TABLE selected_stations(station_id VARCHAR, location VARCHAR)")
        connection.executemany("INSERT INTO selected_stations VALUES (?, ?)", list(labels.items()))
        parquet_paths = self._parquet_paths(spec.dataset.value, start, end)
        bucket = DATE_TRUNC[spec.resolution]
        metrics = [metric.value for metric in spec.metrics]
        projections = [f"{METRIC_SQL[item]} AS {item}" for item in metrics if item in METRIC_SQL]
        projections.append("count(station_value) AS reporting_stations")
        projections.append("sum(station_count) AS observation_count")
        projections.append("avg(station_coverage) AS coverage")
        projections.append("sum(observed_count) AS observed_count")
        projections.append("sum(imputed_count) AS imputed_count")
        projections.append("sum(historical_unknown_count) AS historical_unknown_count")
        expected = {"hour": 1, "day": 24, "week": 168, "month": 730.5, "year": 8766}[bucket]
        if parquet_paths:
            source_sql = """
                SELECT h.timestamp_local AS ts, h.analysis_value AS value,
                       s.station_id, s.location, h.value_status
                FROM read_parquet(?, hive_partitioning=true, union_by_name=true) h
                JOIN selected_stations s ON h.source_station_id = s.station_id
                WHERE h.pollutant = ? AND h.timestamp_local BETWEEN ? AND ?
                QUALIFY row_number() OVER (
                    PARTITION BY h.source_station_id, h.pollutant, h.timestamp_local
                    ORDER BY CASE h.value_status WHEN 'observed' THEN 0 WHEN 'imputed' THEN 1 ELSE 2 END
                ) = 1
            """
            parameters = [parquet_paths, pollutant, start, end]
        else:
            connection.execute("CREATE TEMP TABLE selected_files(path VARCHAR, station_id VARCHAR)")
            connection.executemany("INSERT INTO selected_files VALUES (?, ?)", [(item.path, item.station_id) for item in files])
            paths = [item.path for item in files]
            source_sql = f"""
                SELECT try_cast(r.Timestamp AS TIMESTAMP) AS ts,
                       try_cast(r.{_quote(column)} AS DOUBLE) AS value,
                       f.station_id, s.location, 'historical_unknown'::VARCHAR AS value_status
                FROM read_csv(?, header=true, filename=true, union_by_name=true) r
                JOIN selected_files f ON r.filename = f.path
                JOIN selected_stations s ON f.station_id = s.station_id
                WHERE try_cast(r.Timestamp AS TIMESTAMP) BETWEEN ? AND ?
            """
            parameters = [paths, start, end]
        sql = f"""
            WITH raw_unfiltered AS (
                {source_sql}
            ), raw AS (
                SELECT * FROM raw_unfiltered
                WHERE ? = 'any' OR value_status = ?
            ), station_bucket AS (
                SELECT date_trunc('{bucket}', ts) AS timestamp,
                       station_id, location, avg(value) AS station_value,
                       count(value) AS station_count,
                       least(count(value) / {expected}, 1.0) AS station_coverage,
                       count(*) FILTER (WHERE value_status='observed') AS observed_count,
                       count(*) FILTER (WHERE value_status='imputed') AS imputed_count,
                       count(*) FILTER (WHERE value_status='historical_unknown') AS historical_unknown_count
                FROM raw
                WHERE value IS NOT NULL AND value >= 0
                GROUP BY 1, 2, 3
            )
            SELECT timestamp, location, {', '.join(projections)}
            FROM station_bucket
            GROUP BY timestamp, location
            ORDER BY timestamp, location
            LIMIT ?
        """
        try:
            status = spec.filters.value_status.value
            rows = connection.execute(sql, [*parameters, status, status, spec.limit]).fetchall()
            names = [item[0] for item in connection.description]
        finally:
            connection.close()
        output = []
        for row in rows:
            item = dict(zip(names, row))
            item["timestamp"] = item["timestamp"].isoformat()
            for key, value in list(item.items()):
                if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                    item[key] = None
            item.update({"pollutant": pollutant, "unit": unit})
            output.append(item)
        return output

    @staticmethod
    def _result_status(rows: list[dict]) -> str:
        counts = {
            "observed": sum(int(row.get("observed_count") or 0) for row in rows),
            "imputed": sum(int(row.get("imputed_count") or 0) for row in rows),
            "historical_unknown": sum(int(row.get("historical_unknown_count") or 0) for row in rows),
        }
        present = [name for name, count in counts.items() if count]
        return present[0] if len(present) == 1 else "mixed" if present else "unavailable"

    def _parquet_paths(self, dataset: str, start: datetime, end: datetime) -> list[str]:
        if not self.prefer_parquet:
            return []
        paths = []
        for layer in ("historical", "hourly"):
            root = Path(self.config["project"]["data_root"]) / "curated" / layer / f"dataset={dataset}"
            current = start.replace(day=1)
            while current <= end:
                directory = root / f"year={current.year}" / f"month={current.month:02d}"
                if not directory.exists():
                    directory = root / f"year={current.year}" / f"month={current.month}"
                paths.extend(str(path.resolve()) for path in directory.glob("*.parquet"))
                if current.month == 12:
                    current = current.replace(year=current.year + 1, month=1)
                else:
                    current = current.replace(month=current.month + 1)
        return paths

    def _incremental_paths(self, dataset: str, start: datetime, end: datetime) -> list[str]:
        root = Path(self.config["project"]["data_root"]) / "curated" / "hourly" / f"dataset={dataset}"
        if not root.exists():
            return []
        return [path for path in self._parquet_paths(dataset, start, end) if "/curated/hourly/" in path]

    def _available_bounds(self, dataset: str) -> tuple[datetime, datetime]:
        cfg = self.config["datasets"][dataset]
        start = datetime.fromisoformat(cfg["historical_start"])
        end = datetime.fromisoformat(cfg["historical_end"])
        root = Path(self.config["project"]["data_root"]) / "curated" / "hourly" / f"dataset={dataset}"
        paths = [str(path) for path in root.glob("year=*/month=*/*.parquet")]
        if paths:
            row = duckdb.sql("SELECT min(timestamp_local), max(timestamp_local) FROM read_parquet(?, union_by_name=true)", params=[paths]).fetchone()
            if row[0] is not None:
                start, end = min(start, row[0]), max(end, row[1])
        return start, end

    def _labels(self, spec: QuerySpec, stations) -> dict[str, str]:
        if spec.operation == Operation.rank and spec.group_by:
            field = "global" if spec.group_by == GeographyLevel.global_ else spec.group_by.value
            return {station.station_id: str(getattr(station, field, None) or "Unresolved") for station in stations}
        if spec.operation == Operation.compare and len(spec.locations) > 1:
            labels = {}
            level = "global" if spec.geography_level == GeographyLevel.global_ else spec.geography_level.value
            for location in spec.locations:
                for station in self.catalog.resolve_stations(spec.dataset.value, level, [location]):
                    labels[station.station_id] = location
            return labels
        label = spec.locations[0]
        return {station.station_id: label for station in stations}

    @staticmethod
    def _location_metadata(stations, labels: dict[str, str]) -> list[dict]:
        grouped: dict[str, list] = {}
        for station in stations:
            grouped.setdefault(labels[station.station_id], []).append(station)
        output = []
        for label, members in grouped.items():
            latitudes = [item.latitude for item in members if item.latitude is not None]
            longitudes = [item.longitude for item in members if item.longitude is not None]
            output.append({"location": label, "latitude": sum(latitudes) / len(latitudes) if latitudes else None, "longitude": sum(longitudes) / len(longitudes) if longitudes else None, "station_count": len(members)})
        return output

    @staticmethod
    def _postprocess(spec: QuerySpec, rows: list[dict]) -> tuple[list[dict], dict | None]:
        if spec.operation == Operation.rank:
            metric = spec.metrics[0].value
            rows = sorted(rows, key=lambda item: item.get(metric) if item.get(metric) is not None else float("-inf"), reverse=True)[:spec.top_n]
            for index, row in enumerate(rows, 1):
                row["rank"] = index
            return rows, {"operation": "rank", "metric": metric, "top_n": spec.top_n}
        if spec.operation == Operation.exceedance:
            metric, threshold = spec.metrics[0].value, spec.filters.threshold
            matches = [row for row in rows if row.get(metric) is not None and row[metric] > threshold]
            return matches, {"operation": "exceedance", "threshold": threshold, "exceedance_count": len(matches), "total_buckets": len(rows)}
        if spec.operation == Operation.trend:
            metric = spec.metrics[0].value
            values = [row.get(metric) for row in rows if row.get(metric) is not None]
            if len(values) >= 2:
                import numpy as np
                slope = float(np.polyfit(np.arange(len(values)), values, 1)[0])
                return rows, {"operation": "trend", "slope_per_bucket": slope, "direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat"}
        if spec.operation == Operation.correlation:
            by_pollutant = {}
            for row in rows:
                by_pollutant.setdefault(row["pollutant"], {})[(row["timestamp"], row["location"])] = row.get("mean")
            names = list(by_pollutant)
            if len(names) == 2:
                common = sorted(set(by_pollutant[names[0]]) & set(by_pollutant[names[1]]))
                pairs = [(by_pollutant[names[0]][key], by_pollutant[names[1]][key]) for key in common]
                pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
                if len(pairs) >= 2:
                    import numpy as np
                    value = float(np.corrcoef([a for a, _ in pairs], [b for _, b in pairs])[0, 1])
                    return rows, {"operation": "correlation", "pollutants": names, "pearson_r": value, "paired_buckets": len(pairs)}
        return rows, {"operation": spec.operation.value}

    @staticmethod
    def _visualization(spec: QuerySpec, rows: int) -> dict:
        requested = spec.visualization
        if requested == "none":
            return {"type": "none", "reason": "Explicitly disabled"}
        if requested:
            return {"type": requested}
        if spec.operation == Operation.rank:
            return {"type": "bar"}
        if spec.operation == Operation.correlation:
            return {"type": "scatter"}
        if rows <= 1:
            return {"type": "table"}
        if spec.resolution == Resolution.hourly and rows > 48:
            return {"type": "heatmap", "fallback": "line"}
        return {"type": "line"}


def _quote(identifier: str) -> str:
    if not identifier or "\x00" in identifier:
        raise ValidationError("Invalid column identifier")
    return '"' + identifier.replace('"', '""') + '"'
