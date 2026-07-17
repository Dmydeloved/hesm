"""
Vector RAG memory system.

Embeds each conversation turn with BailianEmbedder and stores it in a
per-conversation ChromaDB collection. At retrieval time, embeds the question
and returns the top-K most similar turns.

Uses the HESM project's existing ChromaVectorStore and BailianEmbedder —
no new vector infrastructure needed, just separate persist paths per conv_id.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from experiments.locomo.data.loader import Session
from experiments.locomo.evaluation.token_metrics import count_tokens
from experiments.locomo.methods.base import MemorySystem, RetrievalResult

logger = logging.getLogger(__name__)


class VectorRAGMemory(MemorySystem):
    """
    Standard dense retrieval baseline: turn-level vector search without any
    hierarchical or semantic structuring.

    Each turn is indexed as: "[Speaker]: text"  (identical to QA user_input format
    in HESM so embeddings are in a comparable space).

    Storage isolation: each (conv_id) gets its own Chroma directory under
    outputs/locomo/memory/vector_rag_{conv_id}/chroma/.
    """

    def __init__(self, memory_root: str | Path) -> None:
        """
        Args:
            memory_root: base directory for isolated Chroma stores,
                         e.g. "outputs/locomo/memory"
        """
        self._memory_root = Path(memory_root)
        self._vector_store: Any = None
        self._embedder: Any = None
        self._conv_id: str = ""
        self._turn_texts: dict[str, str] = {}  # dia_id → turn text

    @property
    def method_name(self) -> str:
        return "vector_rag"

    def reset(self) -> None:
        # Close existing store (ChromaVectorStore has no explicit close, just drop ref)
        self._vector_store = None
        self._conv_id = ""
        self._turn_texts = {}

    def build_memory(
        self,
        conv_id: str,
        sessions: list[Session],
        speaker_a: str,
        speaker_b: str,
    ) -> None:
        from memory.embedder import BailianEmbedder
        from memory.vector_store import ChromaVectorStore

        self._conv_id = conv_id
        chroma_path = self._memory_root / f"vector_rag_{conv_id}" / "chroma"
        chroma_path.mkdir(parents=True, exist_ok=True)

        self._vector_store = ChromaVectorStore(persist_path=str(chroma_path))
        self._embedder = BailianEmbedder()
        self._turn_texts = {}

        total = 0
        for session in sessions:
            for turn in session.turns:
                if not turn.dia_id or not turn.text.strip():
                    continue
                text = f"[{turn.speaker}]: {turn.text}"
                self._turn_texts[turn.dia_id] = text
                try:
                    embedding = self._embedder.embed(text)
                    # Store as memory_type="qa" so ChromaVectorStore IDs are
                    # "qa:{safe_dia_id}" — consistent with HESM conventions
                    safe_id = turn.dia_id.replace(":", "_")
                    self._vector_store.upsert(
                        memory_type="qa",
                        memory_id=safe_id,
                        text=text,
                        embedding=embedding,
                        updated_at=turn.timestamp,
                        metadata={
                            "dia_id": turn.dia_id,
                            "speaker": turn.speaker,
                            "session_num": turn.session_num,
                        },
                    )
                    total += 1
                except Exception as exc:
                    logger.warning("Failed to embed turn %s: %s", turn.dia_id, exc)

        logger.info("[VectorRAG] %s: indexed %d turns", conv_id, total)

    def retrieve(self, question: str, top_k: int = 5) -> RetrievalResult:
        if self._vector_store is None or self._embedder is None:
            return RetrievalResult("", [], 0, {"error": "memory not built"})

        try:
            query_emb = self._embedder.embed(question)
            results = self._vector_store.query(
                query_embedding=query_emb,
                memory_type="qa",
                top_k=top_k,
            )
        except Exception as exc:
            logger.warning("[VectorRAG] retrieve failed: %s", exc)
            return RetrievalResult("", [], 0, {"error": str(exc)})

        retrieved_ids: list[str] = []
        lines: list[str] = []
        for r in results:
            dia_id = r.get("metadata", {}).get("dia_id", "")
            if dia_id:
                retrieved_ids.append(dia_id)
            doc = r.get("document", "")
            if doc:
                lines.append(doc)

        context_text = "\n".join(lines)
        return RetrievalResult(
            context_text=context_text,
            retrieved_ids=retrieved_ids,
            token_count=count_tokens(context_text),
            raw_result={"num_results": len(results)},
        )
