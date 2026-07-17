from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.datasets import ExperimentExample
from .base import MethodResult
from .scoring import memory_text


class FullContextMethod:
    name = "full_context"

    def __init__(self, output_dir: Path, config: dict[str, Any]) -> None:
        del output_dir, config
        self.memories: list[ExperimentExample] = []

    def reset_dialogue(self, dialogue_id: str) -> None:
        del dialogue_id
        self.memories = []

    def predict(self, example: ExperimentExample) -> MethodResult:
        del example
        items = [
            {
                "id": memory.case_id,
                "user_input": memory.user_input,
                "assistant_output": memory.assistant_output,
                "score": 1.0,
            }
            for memory in self.memories
        ]
        context = "\n\n".join(memory_text(memory) for memory in self.memories)
        return MethodResult(answer=context, context_text=context, retrieved_items=items)

    def observe(self, example: ExperimentExample) -> None:
        self.memories.append(example)

