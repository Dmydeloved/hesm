from __future__ import annotations

import shutil
import os
from pathlib import Path
from typing import Any

from experiments.datasets import ExperimentExample
from memory.config import get as config_get
from memory.embedder import BailianEmbedder, HashingEmbedder
from memory.manager import MemoryManager
from memory.retriever import HybridRetriever, LLMRetrievalReranker
from memory.storage import MemoryStorage
from memory.vector_store import ChromaVectorStore
from .base import MethodResult


class HeuristicReranker:
    def rerank(
        self,
        layer: str,
        query_text: str,
        candidates: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        del layer, query_text
        ranked = sorted(
            candidates,
            key=lambda item: (bool(item.get("vector_recalled")), str(item.get("updated_at") or "")),
            reverse=True,
        )
        return [
            {
                "id": item["id"],
                "score": 1.0 if item.get("vector_recalled") else 0.5,
                "reason": "heuristic experiment reranker",
            }
            for item in ranked[:limit]
        ]


class HESMMethod:
    name = "hesm"

    def __init__(self, output_dir: Path, config: dict[str, Any]) -> None:
        self.output_dir = output_dir
        self.top_experience = int(config.get("top_experience", 3))
        self.top_segment = int(config.get("top_segment", 5))
        self.top_qa = int(config.get("top_qa", 8))
        self.embedder = build_embedder(config)
        self.reranker = build_reranker(config)
        self.storage: MemoryStorage | None = None
        self.vector_store: ChromaVectorStore | None = None
        self.manager: MemoryManager | None = None
        self.retriever: HybridRetriever | None = None

    def reset_dialogue(self, dialogue_id: str) -> None:
        self.close()
        dialogue_dir = self.output_dir / "state" / safe_name(dialogue_id)
        if dialogue_dir.exists():
            shutil.rmtree(dialogue_dir)
        dialogue_dir.mkdir(parents=True, exist_ok=True)
        self.storage = MemoryStorage(dialogue_dir / "memory.sqlite3")
        self.vector_store = ChromaVectorStore(dialogue_dir / "chroma", ephemeral=True)
        self.manager = MemoryManager(
            storage=self.storage,
            vector_store=self.vector_store,
            embedder=self.embedder,
        )
        self.retriever = HybridRetriever(
            storage=self.storage,
            vector_store=self.vector_store,
            embedder=self.embedder,
            reranker=self.reranker,
        )

    def predict(self, example: ExperimentExample) -> MethodResult:
        if self.retriever is None:
            self.reset_dialogue(example.dialogue_id)
        assert self.retriever is not None
        topic = example.topic_result
        result = self.retriever.recall(
            topic=str(topic.get("topic") or ""),
            core_entity=str(topic.get("core_entity") or ""),
            intent=str(topic.get("intent") or ""),
            entities=topic.get("entities") or [],
            top_experience=self.top_experience,
            top_segment=self.top_segment,
            top_qa=self.top_qa,
            state_key=f"eval-{safe_name(example.dialogue_id)}",
            use_cache=True,
        )
        retrieved_items = [
            {
                "id": item["qa_id"],
                "user_input": item.get("user_input", ""),
                "assistant_output": item.get("assistant_output", ""),
                "score": item.get("score", 0.0),
            }
            for item in result["qas"]
        ]
        return MethodResult(
            answer=result.get("context_text", ""),
            context_text=result.get("context_text", ""),
            retrieved_items=retrieved_items,
            debug=result.get("debug", {}),
        )

    def observe(self, example: ExperimentExample) -> None:
        if self.manager is None:
            self.reset_dialogue(example.dialogue_id)
        assert self.manager is not None
        self.manager.add_qa(
            topic_result=example.topic_result,
            user_input=example.user_input,
            assistant_output=example.assistant_output,
            tools=[],
            state_key=f"eval-{safe_name(example.dialogue_id)}",
        )

    def close(self) -> None:
        if self.storage is not None:
            self.storage.close()
        self.storage = None
        self.vector_store = None
        self.manager = None
        self.retriever = None


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def build_embedder(config: dict[str, Any]):
    name = str(config.get("embedder", "hashing")).lower()
    if name == "hashing":
        return HashingEmbedder(dimensions=int(config.get("dimensions", 256)))
    if name == "bailian":
        api_key = config.get("embedding_api_key")
        api_key_env = config.get("embedding_api_key_env")
        if not api_key and api_key_env:
            api_key = os.getenv(str(api_key_env))
        return BailianEmbedder(
            api_key=str(api_key) if api_key else None,
            model=config.get("embedding_model") or config_get("embedding", "model"),
            base_url=config.get("embedding_base_url") or config_get("embedding", "base_url"),
        )
    raise ValueError(f"Unknown HESM embedder: {name}")


def build_reranker(config: dict[str, Any]):
    name = str(config.get("rerank", "heuristic")).lower()
    if name == "heuristic":
        return HeuristicReranker()
    if name == "llm":
        api_key = config.get("retrieval_api_key")
        api_key_env = config.get("retrieval_api_key_env")
        if not api_key and api_key_env:
            api_key = os.getenv(str(api_key_env))
        return LLMRetrievalReranker(
            api_key=str(api_key) if api_key else None,
            model=config.get("retrieval_model") or config_get("retrieval", "model"),
            base_url=config.get("retrieval_base_url") or config_get("retrieval", "base_url"),
            max_retries=config.get("retrieval_max_retries"),
            retry_delay=config.get("retrieval_retry_delay"),
        )
    raise ValueError(f"Unknown HESM reranker: {name}")
