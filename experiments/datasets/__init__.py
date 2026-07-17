from __future__ import annotations

from pathlib import Path

from .base import ExperimentExample
from .locomo import load_locomo
from .longmemeval import load_longmemeval
from .multiwoz import load_multiwoz


def load_dataset(
    name: str,
    path: str | Path,
    limit: int = 0,
    dialogue_limit: int = 0,
) -> list[ExperimentExample]:
    loaders = {
        "multiwoz": load_multiwoz,
        "locomo": load_locomo,
        "longmemeval": load_longmemeval,
    }
    try:
        loader = loaders[name]
    except KeyError as error:
        raise ValueError(f"Unknown dataset: {name}") from error
    examples = loader(Path(path), limit=limit)
    if dialogue_limit <= 0:
        return examples

    kept_dialogues: list[str] = []
    filtered: list[ExperimentExample] = []
    for example in examples:
        if example.dialogue_id not in kept_dialogues:
            if len(kept_dialogues) >= dialogue_limit:
                break
            kept_dialogues.append(example.dialogue_id)
        filtered.append(example)
    return filtered


__all__ = ["ExperimentExample", "load_dataset"]
