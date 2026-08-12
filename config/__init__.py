from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "hesm.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return data


def get_config(path: str | Path | None = None) -> dict[str, Any]:
    return copy.deepcopy(load_config(path))


def get_config_value(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "PACKAGE_ROOT",
    "get_config",
    "get_config_value",
    "load_config",
]
