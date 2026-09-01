from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import MSTL


METHOD = "mstl_24_168_deseasonalized_linear"
POLICY_VERSION = "1"


def impute_new_data(frame: pd.DataFrame, *, max_missing_fraction: float = 0.30,
                    max_gap_hours: int = 336, retain_missing: bool = False) -> pd.DataFrame:
    required = {"dataset", "source_station_id", "pollutant", "timestamp_local", "observed_value", "unit", "source", "ingestion_run_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Canonical staging fields missing: {sorted(missing)}")
    pieces = []
    keys = ["dataset", "source_station_id", "pollutant", "unit", "source", "ingestion_run_id"]
    for key, group in frame.groupby(keys, dropna=False):
        group = group.sort_values("timestamp_local").drop_duplicates("timestamp_local", keep="last")
        index = pd.date_range(group["timestamp_local"].min(), group["timestamp_local"].max(), freq="h")
        observed = group.set_index("timestamp_local")["observed_value"].reindex(index).astype(float)
        observed = observed.mask(observed < 0)
        missing_mask = observed.isna()
        analysis = observed.copy()
        method = pd.Series(pd.NA, index=index, dtype="object")
        eligible = missing_mask.mean() <= max_missing_fraction and _max_internal_gap(missing_mask) <= max_gap_hours and observed.notna().sum() >= 336
        if eligible and missing_mask.any():
            scaffold = observed.interpolate("linear").ffill().bfill()
            result = MSTL(scaffold, periods=(24, 168), iterate=2, stl_kwargs={"robust": True}).fit()
            seasonal = result.seasonal.sum(axis=1) if isinstance(result.seasonal, pd.DataFrame) else result.seasonal
            filled = (observed - seasonal).interpolate("linear", limit_area="inside") + seasonal
            analysis.loc[missing_mask] = filled.loc[missing_mask]
            analysis = analysis.clip(lower=0)
            method.loc[missing_mask & analysis.notna()] = METHOD
        out = pd.DataFrame({"timestamp_local": index, "observed_value": observed, "analysis_value": analysis})
        out["value_status"] = np.where(out["observed_value"].notna(), "observed", np.where(out["analysis_value"].notna(), "imputed", "missing"))
        out["imputation_method"] = method.to_numpy()
        out["imputation_policy_version"] = np.where(out["value_status"] == "imputed", POLICY_VERSION, None)
        for name, value in zip(keys, key):
            out[name] = value
        pieces.append(out)
    result = pd.concat(pieces, ignore_index=True)
    return result if retain_missing else result[result["value_status"] != "missing"]


def _max_internal_gap(mask: pd.Series) -> int:
    seen = mask.copy()
    valid_positions = np.flatnonzero(~seen.to_numpy())
    if len(valid_positions) < 2:
        return len(seen)
    internal = seen.iloc[valid_positions[0]:valid_positions[-1] + 1].to_numpy()
    longest = current = 0
    for value in internal:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest
