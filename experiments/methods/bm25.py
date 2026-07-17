from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.datasets import ExperimentExample
from .base import MethodResult
from .scoring import bm25_score, memory_text


class BM25Method:
    name = "bm25"

    def __init__(self, output_dir: Path, config: dict[str, Any]) -> None:
        del output_dir
        self.top_k = int(config.get("top_k") or config.get("top_qa") or 8)
        self.memories: list[tuple[ExperimentExample, str]] = []

    def reset_dialogue(self, dialogue_id: str) -> None:
        del dialogue_id
        self.memories = []

    def predict(self, example: ExperimentExample) -> MethodResult:
        ranked = sorted(
            self.memories,
            key=lambda item: bm25_score(example.user_input, item[1]),
            reverse=True,
        )[: self.top_k]
        retrieved = [
            {
                "id": memory.case_id,
                "user_input": memory.user_input,
                "assistant_output": memory.assistant_output,
                "score": bm25_score(example.user_input, text),
            }
            for memory, text in ranked
        ]
        context = "\n\n".join(text for _, text in ranked)
        return MethodResult(answer=context, context_text=context, retrieved_items=retrieved)

    def observe(self, example: ExperimentExample) -> None:
        self.memories.append((example, memory_text(example)))

