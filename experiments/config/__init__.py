"""Configuration loader owned exclusively by the experiment package."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


CONFIG_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CONFIG_ROOT.parents[1]
DEFAULT_CONFIG_PATH = CONFIG_ROOT / "locomo.yaml"


def load_experiment_config(
    path: str | Path | None = None,
) -> dict[str, Any]:
    config_path = Path(path).expanduser() if path else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Experiment config must be a mapping: {config_path}")
    return copy.deepcopy(config)


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


__all__ = [
    "CONFIG_ROOT",
    "DEFAULT_CONFIG_PATH",
    "PROJECT_ROOT",
    "load_experiment_config",
    "project_path",
]
