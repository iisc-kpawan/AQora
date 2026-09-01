from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigurationError


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_paths(value: Any, root: Path) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_paths(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_paths(item, root) for item in value]
    return value


@lru_cache(maxsize=4)
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    selected = path or os.environ.get("AQA_CONFIG") or PROJECT_ROOT / "configs" / "aqa.yaml"
    config_path = Path(selected).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or "datasets" not in config:
        raise ConfigurationError("Configuration must define datasets")
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(PROJECT_ROOT)
    for dataset in config["datasets"].values():
        for key in ("historical_path", "metadata_path"):
            raw = Path(dataset[key]).expanduser()
            dataset[key] = str(raw if raw.is_absolute() else PROJECT_ROOT / raw)
    for key in ("data_root", "control_database", "catalog_database", "export_root"):
        raw = Path(config["project"][key]).expanduser()
        config["project"][key] = str(raw if raw.is_absolute() else PROJECT_ROOT / raw)
    return _resolve_paths(config, PROJECT_ROOT)
