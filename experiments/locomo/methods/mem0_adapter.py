"""
Mem0 memory system adapter.

Uses the mem0ai library (pip install mem0ai) to build and query memory.
Configures Mem0 from the isolated memory_methods.mem0 section in
configs/config.yaml, with legacy top-level fallbacks.

dia_id tracking: stored as Mem0 metadata {"dia_id": "D1:3"} and recovered
from search results.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from typing import Any

from experiments.locomo.data.loader import Session
from experiments.locomo.evaluation.token_metrics import count_tokens
from experiments.locomo.methods.base import MemorySystem, RetrievalResult
from experiments.locomo.methods.build_state import (
    expected_dia_ids,
    load_completed_dia_ids,
    save_build_state,
)

logger = logging.getLogger(__name__)


class Mem0Memory(MemorySystem):
    """
    Wraps Mem0 (mem0ai) for the LoCoMo benchmark.

    Each conversation uses a dedicated Mem0 user_id (conv_id) so memories
    are isolated between conversations.

    Mem0 is configured to use:
    - LLM:       memory_methods.mem0.llm (OpenAI-compatible)
    - Embedder:  memory_methods.mem0.embedding
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
            raise RuntimeError(f"Mem0 could not attach or initialise memory for {conv_id}")

        expected = expected_dia_ids(sessions)
        base = self._memory_root / f"mem0_{conv_id}"
        state_path = base / "build_state.json"
        completed = load_completed_dia_ids(state_path)
        if not completed:
            # Backward compatibility for stores built before build_state.json
            # existed. Chroma persists the source dia_id in embedding metadata.
            completed.update(self._legacy_chroma_dia_ids(base / "chroma"))

        if expected and expected <= completed:
            save_build_state(
                state_path,
                method=self.method_name,
                conv_id=conv_id,
                expected=expected,
                completed=completed,
            )
            logger.info(
                "[Mem0] %s: memory already complete (%d turns), skipping build",
                conv_id,
                len(expected),
            )
            return

        total = 0
        skipped = 0
        for session in sessions:
            for turn in session.turns:
                if not turn.text.strip() or not turn.dia_id:
                    continue
                if turn.dia_id in completed:
                    skipped += 1
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
                    completed.add(turn.dia_id)
                    save_build_state(
                        state_path,
                        method=self.method_name,
                        conv_id=conv_id,
                        expected=expected,
                        completed=completed,
                    )
                except Exception as exc:
                    logger.warning("[Mem0] turn %s failed: %s", turn.dia_id, exc)

        save_build_state(
            state_path,
            method=self.method_name,
            conv_id=conv_id,
            expected=expected,
            completed=completed,
        )
        logger.info(
            "[Mem0] %s: added %d turns, skipped %d completed turns",
            conv_id,
            total,
            skipped,
        )

    def retrieve(self, question: str, top_k: int = 5) -> RetrievalResult:
        if self._memory is None:
            return RetrievalResult("", [], 0, {"error": "mem0 not initialised"})

        try:
            results = self._memory.search(
                query=question,
                filters={"user_id": self._user_id},
                top_k=top_k,
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
        method_cfg = cfg.get("memory_methods", {}).get("mem0", {})
        te = method_cfg.get("llm") or cfg.get("topic_extraction", {})
        emb = method_cfg.get("embedding") or cfg.get("embedding", {})

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
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            logger.error("[Mem0] Failed to initialise: %s", exc)
            return None

    @staticmethod
    def _legacy_chroma_dia_ids(chroma_path: Path) -> set[str]:
        """Recover source IDs from a legacy Mem0 Chroma store without loading Mem0."""
        sqlite_path = chroma_path / "chroma.sqlite3"
        if not sqlite_path.exists():
            return set()
        try:
            connection = sqlite3.connect(str(sqlite_path))
            try:
                rows = connection.execute(
                    "SELECT string_value FROM embedding_metadata "
                    "WHERE key = 'dia_id' AND string_value IS NOT NULL"
                ).fetchall()
            finally:
                connection.close()
            return {str(row[0]) for row in rows if str(row[0]).strip()}
        except sqlite3.Error as exc:
            logger.warning("[Mem0] failed to inspect legacy build state: %s", exc)
            return set()
