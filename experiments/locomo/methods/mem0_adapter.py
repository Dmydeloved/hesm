"""
Mem0 memory system adapter.

Uses the mem0ai library (pip install mem0ai) to build and query memory.
Configures Mem0 to use the same OpenAI-compatible LLM and embedding API
as the rest of the HESM experiment (topic_extraction and embedding sections
from configs/config.yaml).

dia_id tracking: stored as Mem0 metadata {"dia_id": "D1:3"} and recovered
from search results.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from experiments.locomo.data.loader import Session
from experiments.locomo.evaluation.token_metrics import count_tokens
from experiments.locomo.methods.base import MemorySystem, RetrievalResult

logger = logging.getLogger(__name__)


class Mem0Memory(MemorySystem):
    """
    Wraps Mem0 (mem0ai) for the LoCoMo benchmark.

    Each conversation uses a dedicated Mem0 user_id (conv_id) so memories
    are isolated between conversations.

    Mem0 is configured to use:
    - LLM:       topic_extraction config (OpenAI-compatible)
    - Embedder:  embedding config (Alibaba DashScope text-embedding-v4)
    - VectorDB:  Chroma (persisted per conv_id)
    """

    def __init__(
        self,
        memory_root: str | Path,
        hesm_config: dict[str, Any],
        collection_prefix: str = "mem0_locomo",
    ) -> None:
        self._memory_root = Path(memory_root)
        self._hesm_config = hesm_config  # full configs/config.yaml content
        self._collection_prefix = collection_prefix
        self._memory: Any = None
        self._user_id: str = ""

    @property
    def method_name(self) -> str:
        return "mem0"

    def reset(self) -> None:
        self._memory = None
        self._user_id = ""

    def build_memory(
        self,
        conv_id: str,
        sessions: list[Session],
        speaker_a: str,
        speaker_b: str,
    ) -> None:
        self._user_id = conv_id
        self._memory = self._create_mem0(conv_id)
        if self._memory is None:
            return

        total = 0
        for session in sessions:
            for turn in session.turns:
                if not turn.text.strip():
                    continue
                text = f"[{turn.speaker}]: {turn.text}"
                try:
                    self._memory.add(
                        text,
                        user_id=conv_id,
                        metadata={
                            "dia_id": turn.dia_id,
                            "speaker": turn.speaker,
                            "session_num": turn.session_num,
                            "timestamp": turn.timestamp,
                        },
                    )
                    total += 1
                except Exception as exc:
                    logger.warning("[Mem0] turn %s failed: %s", turn.dia_id, exc)

        logger.info("[Mem0] %s: added %d turns", conv_id, total)

    def retrieve(self, question: str, top_k: int = 5) -> RetrievalResult:
        if self._memory is None:
            return RetrievalResult("", [], 0, {"error": "mem0 not initialised"})

        try:
            results = self._memory.search(
                query=question,
                user_id=self._user_id,
                limit=top_k,
            )
        except Exception as exc:
            logger.warning("[Mem0] search failed: %s", exc)
            return RetrievalResult("", [], 0, {"error": str(exc)})

        # mem0 search returns: {"results": [{"memory": str, "metadata": dict, ...}]}
        items = results if isinstance(results, list) else results.get("results", [])

        retrieved_ids: list[str] = []
        lines: list[str] = []
        for item in items:
            memory_text = item.get("memory", "") or item.get("text", "")
            if memory_text:
                lines.append(memory_text)
            meta = item.get("metadata", {}) or {}
            dia_id = meta.get("dia_id", "")
            if dia_id:
                retrieved_ids.append(dia_id)

        context_text = "\n".join(lines)
        return RetrievalResult(
            context_text=context_text,
            retrieved_ids=retrieved_ids,
            token_count=count_tokens(context_text),
            raw_result={"num_results": len(items)},
        )

    # ─── Internal ─────────────────────────────────────────────────────────────

    def _create_mem0(self, conv_id: str) -> Any:
        """Construct a Mem0 Memory instance with project-consistent config."""
        try:
            from mem0 import Memory
        except ImportError:
            logger.error(
                "mem0ai not installed. Run: pip install mem0ai"
            )
            return None

        cfg = self._hesm_config
        te = cfg.get("topic_extraction", {})
        emb = cfg.get("embedding", {})

        api_key = te.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
        emb_api_key = emb.get("api_key") or api_key

        chroma_path = str(
            self._memory_root / f"mem0_{conv_id}" / "chroma"
        )
        Path(chroma_path).mkdir(parents=True, exist_ok=True)

        mem0_config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": te.get("model", "gpt-4"),
                    "api_key": api_key,
                    "openai_base_url": te.get("base_url", ""),
                    "temperature": 0.0,
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": emb.get("model", "text-embedding-v4"),
                    "api_key": emb_api_key,
                    "openai_base_url": emb.get("base_url", ""),
                },
            },
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": f"{self._collection_prefix}_{conv_id}",
                    "path": chroma_path,
                },
            },
        }

        try:
            return Memory.from_config(mem0_config)
        except Exception as exc:
            logger.error("[Mem0] Failed to initialise: %s", exc)
            return None
