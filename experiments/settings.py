"""Load the standalone HESM configuration used only by OmniMemEval."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
MEMORY_MODEL = "gpt-4.1-mini-2025-04-14"
ANSWER_MODEL = "gpt-4.1-mini-2025-04-14"
JUDGE_MODEL = "gpt-4o-mini-2024-07-18"
EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_MAX_INPUT_TOKENS = 8192


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_env(value: Any, location: str = "config") -> Any:
    """Resolve exact ``${NAME}`` YAML scalars without writing secrets to disk."""
    if isinstance(value, dict):
        return {
            key: _resolve_env(item, f"{location}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _resolve_env(item, f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        name = value[2:-1].strip()
        resolved = os.environ.get(name, "")
        if not resolved:
            raise ValueError(f"Missing environment variable {name} for {location}")
        return resolved
    return value


def read_settings(path: str | Path) -> dict[str, Any]:
    target = Path(path).expanduser().resolve(strict=True)
    raw_data = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(raw_data, dict) or raw_data.get("schema_version") != 2:
        raise ValueError("Expected schema_version: 2 standalone experiment config")
    data = _resolve_env(raw_data)
    experiment = data.get("experiment")
    if not isinstance(experiment, dict):
        raise ValueError("Missing experiment configuration section")
    required = {"profile", "data_root", "host", "port", "api_token"}
    if required - experiment.keys():
        raise ValueError(
            f"Missing experiment settings: {sorted(required - experiment.keys())}"
        )
    native_required = {
        "topic_extraction", "embedding", "summarization",
        "memory_management", "api",
    }
    if native_required - data.keys():
        raise ValueError(
            f"Missing native HESM sections: {sorted(native_required - data.keys())}"
        )
    for section in ("topic_extraction", "summarization", "chat"):
        model = str((data.get(section) or {}).get("model") or "")
        if model != MEMORY_MODEL:
            raise ValueError(
                f"{section}.model must be {MEMORY_MODEL}, got {model or '<unset>'}"
            )
    embedding_model = str((data.get("embedding") or {}).get("model") or "")
    if embedding_model != EMBEDDING_MODEL:
        raise ValueError(
            f"embedding.model must be {EMBEDDING_MODEL}, got {embedding_model or '<unset>'}"
        )
    embedding = data.get("embedding") or {}
    max_input_tokens = int(embedding.get("max_input_tokens", 0))
    chunk_tokens = int(embedding.get("chunk_tokens", 0))
    if max_input_tokens != EMBEDDING_MAX_INPUT_TOKENS:
        raise ValueError(
            f"embedding.max_input_tokens must be {EMBEDDING_MAX_INPUT_TOKENS}"
        )
    if not 1 <= chunk_tokens < max_input_tokens:
        raise ValueError(
            "embedding.chunk_tokens must be positive and below max_input_tokens"
        )
    root = (target.parent / str(experiment["data_root"])).resolve()
    production = (ROOT / "memory").resolve()
    if root == production or production in root.parents or root in production.parents:
        raise ValueError("Experiment storage must be separate from production memory")
    file_sha256 = _sha256(target)
    # The reproducibility fingerprint covers the YAML-controlled models,
    # thresholds and paths, but never derives a stored hash from credentials.
    config_sha256 = file_sha256
    result = dict(experiment)
    result.update({
        "config_path": str(target),
        "data_root": str(root),
        # HESMService reads native settings from this same experiment file.
        "hesm_config_path": str(target),
        "hesm_config_sha256": file_sha256,
        "config_fingerprint": config_sha256,
        "native_config": data,
        "max_top_k": int(experiment.get("max_top_k", 100)),
        "native_models": {
            "topic_extraction": (data.get("topic_extraction") or {}).get("model"),
            "summarization": (data.get("summarization") or {}).get("model"),
            "chat": (data.get("chat") or {}).get("model"),
            "embedding": (data.get("embedding") or {}).get("model"),
        },
    })
    return result


def load_env(path: str | Path) -> None:
    from dotenv import dotenv_values
    for key, value in dotenv_values(path).items():
        if value is not None:
            os.environ[key] = value


def validate_omnimemeval_models() -> None:
    expected = {"ANSWER_MODEL": ANSWER_MODEL, "EVAL_MODEL": JUDGE_MODEL}
    for name, required in expected.items():
        actual = os.environ.get(name, "").strip()
        if actual != required:
            raise ValueError(f"{name} must be {required}, got {actual or '<unset>'}")
