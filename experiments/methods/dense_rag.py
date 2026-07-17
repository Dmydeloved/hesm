from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.datasets import ExperimentExample
from memory.embedder import HashingEmbedder
from .base import MethodResult
from .scoring import cosine, memory_text


class DenseRAGMethod:
    name = "dense_rag"

    def __init__(self, output_dir: Path, config: dict[str, Any]) -> None:
        del output_dir
        self.top_k = int(config.get("top_k") or config.get("top_qa") or 8)
        self.embedder = HashingEmbedder(dimensions=int(config.get("dimensions", 256)))
        self.memories: list[tuple[ExperimentExample, list[float], str]] = []

    def reset_dialogue(self, dialogue_id: str) -> None:
        del dialogue_id
        self.memories = []

    def predict(self, example: ExperimentExample) -> MethodResult:
        query_embedding = self.embedder.embed(example.user_input)
        ranked = sorted(
            self.memories,
            key=lambda item: cosine(query_embedding, item[1]),
            reverse=True,
        )[: self.top_k]
        retrieved = [
            {
                "id": memory.case_id,
                "user_input": memory.user_input,
                "assistant_output": memory.assistant_output,
                "score": cosine(query_embedding, embedding),
            }
            for memory, embedding, _ in ranked
        ]
        context = "\n\n".join(text for _, _, text in ranked)
        return MethodResult(answer=context, context_text=context, retrieved_items=retrieved)

    def observe(self, example: ExperimentExample) -> None:
        text = memory_text(example)
        self.memories.append((example, self.embedder.embed(text), text))

