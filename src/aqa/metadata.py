from __future__ import annotations

import re
from typing import Any

from .catalog import Catalog
from .config import load_config
from .lakehouse_catalog import LakehouseCatalog
from .global_analytics import _period


POLLUTANT_PATTERNS = (
    (r"pm\s*2(?:\.|\s*)5|\bpm25\b", "PM25"), (r"\bpm\s*10\b", "PM10"),
    (r"\bno\s*2\b", "NO2"), (r"\bso\s*2\b", "SO2"),
    (r"\bo\s*3\b|\bozone\b", "O3"), (r"\bco\b", "CO"),
)


class MetadataService:
    def __init__(self, catalog: Catalog | None = None, config: dict | None = None):
        self.config = config or load_config()
        self.catalog = catalog or Catalog(self.config)
        self.lakehouse = LakehouseCatalog(self.config)

    def answer(self, question: str, datasets: list[str]) -> dict[str, Any]:
        text = question.casefold()
        selected = list(self.lakehouse.selected(datasets))
        if not selected:
            selected = list(self.lakehouse.datasets)
        selected_ids = {item.dataset for item in selected}
        pollutant = next((name for pattern, name in POLLUTANT_PATTERNS if re.search(pattern, text)), None)
        rows = [dict(row) for row in self.lakehouse.availability_rows if row["dataset"] in selected_ids]
        if pollutant:
            rows = [row for row in rows if row["pollutant"] == pollutant]

        if "provenance" in text or "measured versus filled" in text or "measured vs filled" in text:
            start, end = _period(text, self.lakehouse, [item.dataset for item in selected])
            audit = [dict(row) for row in self.lakehouse.provenance_rows(
                [item.dataset for item in selected], pollutant, start=start, end=end,
            )]
            for row in audit:
                usable = row["usable_count"] or 0
                filled = row["legacy_filled_count"] + row["guarded_filled_count"]
                row["observed_fraction_of_usable"] = row["observed_count"] / usable if usable else None
                row["filled_fraction_of_usable"] = filled / usable if usable else None
                for key in ("first_usable_timestamp", "last_usable_timestamp", "first_observed_month", "last_observed_month"):
                    if row[key] is not None:
                        row[key] = row[key].isoformat()
                row["measurement_lineage"] = "matched processed-source value"
                row["legacy_fill_lineage"] = "imputed-source value where no processed measurement matched"
                row["guarded_fill_lineage"] = "internal-gap interpolation allowed by guarded-30pct-336h-v1"
                row["aggregate_lineage"] = "hourly station values; stations are equal-weighted within each requested area"
            available = [row for row in audit if row["usable_count"]]
            unavailable = [row for row in audit if not row["usable_count"]]
            period = f"{start:%d %b %Y} to {end:%d %b %Y}"
            if available:
                details = "; ".join(
                    f"{row['country']} {row['pollutant']}: {row['observed_count']:,} measured, "
                    f"{row['legacy_filled_count'] + row['guarded_filled_count']:,} filled, "
                    f"{row['unresolved_count'] + row['invalid_count']:,} unusable"
                    for row in available[:8]
                )
                answer = f"For {period}, the provenance audit found {details}."
            else:
                answer = f"No usable rows exist for the requested provenance audit period, {period}."
            if unavailable:
                answer += " No rows were available for " + ", ".join(
                    f"{row['country']} {row['pollutant']}" for row in unavailable[:8]
                ) + "."
            answer += (
                " ‘Measured’ means the timestamp and value were recovered from the processed source file. "
                "‘Legacy filled’ means the processed value was absent but the corresponding imputed file supplied a value; "
                "it is not represented as a measurement. Guarded fills use the stricter 30%-missing/336-hour internal-gap policy. "
                "Canonical units are recorded per pollutant, invalid negative values are excluded, and area aggregates give each station equal weight."
            )
            data = audit
        elif "list all pollutants" in text or "available pollutants" in text:
            values = sorted({row["pollutant"] for row in rows})
            answer = f"{self._scope_label(selected)} contain {len(values)} pollutants: {', '.join(values)}."
            data = [{"pollutant": value} for value in values]
        elif "how many monitoring stations" in text and pollutant and len(selected) == 1:
            count = rows[0]["station_count"] if rows else 0
            item = selected[0]
            answer = f"{item.country_name} has {count:,} monitoring stations with an available {pollutant} series in the published snapshot."
            data = [{"dataset": item.dataset, "country": item.country_name, "pollutant": pollutant, "station_count": count}]
        elif "earliest" in text or "latest timestamp" in text or "date range" in text:
            data = [{
                "dataset": item.dataset, "country": item.country_name,
                "earliest": item.available_start.isoformat(), "latest": item.available_end.isoformat(),
            } for item in selected]
            answer = " ".join(f"{row['dataset']} runs from {row['earliest']} through {row['latest']}." for row in data)
        elif "countries" in text and pollutant:
            names = sorted({row["country"] for row in rows if row["station_count"]})
            answer = f"{pollutant} series are available in {len(names)} countries in the published AQA lakehouse."
            data = [{"country": country, "pollutant": pollutant} for country in names]
        elif "at least four" in text and "pollutant" in text:
            data = []
            for item in selected:
                qualifying = [row for row in rows if row["dataset"] == item.dataset and row["station_count"] >= 50]
                if len(qualifying) >= 4:
                    data.append({
                        "dataset": item.dataset, "country": item.country_name,
                        "qualifying_pollutants": [row["pollutant"] for row in qualifying],
                        "qualifying_pollutant_count": len(qualifying),
                    })
            answer = f"{len(data)} countries have at least four pollutants with 50 or more monitoring stations."
        elif "largest number" in text and "share" in text and "network" in text:
            data = []
            for item in selected:
                candidates = [row for row in rows if row["dataset"] == item.dataset]
                if not candidates:
                    continue
                total = sum(row["series_count"] for row in candidates)
                largest = max(candidates, key=lambda row: (row["series_count"], row["pollutant"]))
                data.append({
                    "dataset": item.dataset, "source_network": item.network,
                    "pollutant": largest["pollutant"], "series_count": largest["series_count"],
                    "share_of_network_series": largest["series_count"] / total if total else None,
                })
            answer = f"I calculated the leading pollutant and its series share for {len(data)} source datasets."
        elif "ratio of station" in text and "monitoring sites" in text:
            data = []
            for row in self.lakehouse.country_rows:
                if row["dataset"] not in selected_ids:
                    continue
                data.append({**row, "series_per_site": row["station_pollutant_series"] / row["monitoring_sites"] if row["monitoring_sites"] else None})
            data.sort(key=lambda row: (row["series_per_site"] or 0, row["country"]), reverse=True)
            data = data[:5]
            answer = f"The table ranks the top {len(data)} countries by station–pollutant series per monitoring site."
        elif "greatest number" in text and "monitoring sites" in text:
            data = [dict(row) for row in self.lakehouse.country_rows if row["dataset"] in selected_ids]
            data.sort(key=lambda row: (row["monitoring_sites"], row["country"]), reverse=True)
            data = data[:5]
            answer = f"The largest loaded monitoring-site inventory is {data[0]['country']} with {data[0]['monitoring_sites']:,} sites."
        elif "more than one source" in text or "overlapping pm2.5 observations" in text:
            data = [dict(row) for row in self.lakehouse.country_rows if row["dataset"] in selected_ids and row["source_families"] > 1]
            if data:
                answer = f"{len(data)} countries contain more than one source family; these require station-level overlap review."
            else:
                answer = "No loaded country contains more than one source family in the published lakehouse, so cross-source overlap is not present at country level. City-level duplicate mapping requires city metadata and cannot be inferred from country-level partitions alone."
        elif "source network" in text or "source networks" in text:
            data = [{
                "dataset": item.dataset, "country": item.country_name, "source_network": item.network,
                "earliest": item.available_start.isoformat(), "latest": item.available_end.isoformat(),
            } for item in selected if not pollutant or any(row["dataset"] == item.dataset for row in rows)]
            answer = f"{len(data)} source datasets contribute to the requested scope; their exact date ranges are in the table."
        elif "canonical units" in text or "conversion rules" in text:
            units = {(row["pollutant"], row["unit"]) for row in rows}
            data = [{"pollutant": name, "canonical_unit": unit} for name, unit in sorted(units)]
            answer = "Canonical units are " + ", ".join(f"{row['pollutant']}: {row['canonical_unit']}" for row in data) + ". The lakehouse records canonical units, but not a per-row source-conversion audit trail."
        else:
            data = rows
            answer = f"I returned pollutant availability for {len(selected)} loaded country datasets. Station counts describe available series, not complete hourly coverage."

        country_rows = [row for row in self.lakehouse.country_rows if row["dataset"] in selected_ids]
        return {
            "kind": "metadata", "query_spec": {"operation": "metadata", "datasets": sorted(selected_ids)},
            "data": data,
            "metadata": {
                "dataset": ",".join(sorted(selected_ids)),
                "station_count": sum(row["monitoring_sites"] for row in country_rows),
                "value_status": "metadata", "latency_ms": 0,
            },
            "resolved_period": ({"start": start.isoformat(), "end": end.isoformat(), "basis": "explicit requested provenance period"}
                                if 'audit' in locals() else {}),
            "visualization": {"type": "table"},
            "analysis": {"operation": "provenance_audit" if 'audit' in locals() else "metadata", "country_count": len(selected)},
            "answer": answer, "display_title": "AQA global data availability",
        }

    @staticmethod
    def _scope_label(selected) -> str:
        if len(selected) == 1:
            return f"The published {selected[0].country_name} data"
        return f"The published data for {len(selected)} countries"
