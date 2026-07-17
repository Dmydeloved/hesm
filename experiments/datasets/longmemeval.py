from __future__ import annotations

from pathlib import Path

from .base import ExperimentExample
from .locomo import load_generic_memory_jsonl


def load_longmemeval(path: Path, limit: int = 0) -> list[ExperimentExample]:
    return load_generic_memory_jsonl(path, dataset_name="longmemeval", limit=limit)

