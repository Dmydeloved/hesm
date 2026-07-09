from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

from configs import DEFAULT_CONFIG_PATH as ROOT_DEFAULT_CONFIG_PATH
from configs import load_config as load_root_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT_DEFAULT_CONFIG_PATH
CONFIG_PATH_ENV = "TOPIC_MEMORY_CONFIG"

def get_config_path() -> Path:
    override = os.getenv(CONFIG_PATH_ENV, "").strip()
    return Path(override).expanduser() if override else DEFAULT_CONFIG_PATH


def get_config() -> dict[str, Any]:
    return copy.deepcopy(load_root_config(get_config_path()))


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    target = Path(path).expanduser() if path else get_config_path()
    return copy.deepcopy(load_root_config(target))


def reload_config() -> dict[str, Any]:
    return get_config()


def config_section(name: str) -> dict[str, Any]:
    section = get_config().get(name)
    return section if isinstance(section, dict) else {}


def get(section: str, key: str, fallback: Any = None) -> Any:
    section_config = config_section(section)
    value = section_config.get(key)
    if value is None:
        return fallback
    if isinstance(value, str) and not value.strip():
        return fallback
    return value


def config_value(section: str, key: str, default: Any = None) -> Any:
    return get(section, key, default)


def config_path(section: str, key: str) -> Path:
    value = str(config_value(section, key, "")).strip()
    if not value:
        raise ValueError(f"Missing config path: {section}.{key}")
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


__all__ = [
    "CONFIG_PATH_ENV",
    "DEFAULT_CONFIG_PATH",
    "PROJECT_ROOT",
    "config_path",
    "config_section",
    "config_value",
    "get",
    "get_config",
    "get_config_path",
    "load_config",
    "reload_config",
]
