from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from experiments.datasets import ExperimentExample


@dataclass
class MethodResult:
    answer: str = ""
    context_text: str = ""
    retrieved_items: list[dict[str, Any]] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)


class MemoryMethod(Protocol):
    name: str

    def __init__(self, output_dir: Path, config: dict[str, Any]) -> None:
        ...

    def reset_dialogue(self, dialogue_id: str) -> None:
        ...

    def predict(self, example: ExperimentExample) -> MethodResult:
        ...

    def observe(self, example: ExperimentExample) -> None:
        ...

