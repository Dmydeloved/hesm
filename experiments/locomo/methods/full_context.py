"""
Full Context memory system.

Concatenates ALL conversation turns and passes them directly to the LLM as
context. No retrieval, no compression. Serves as the oracle upper-bound for
retrieval quality but lower-bound for context efficiency.

Retrieved ids = all turn dia_ids (in session order).
"""

from __future__ import annotations

import logging
from typing import Any

from experiments.locomo.data.loader import Session
from experiments.locomo.evaluation.token_metrics import count_tokens
from experiments.locomo.methods.base import MemorySystem, RetrievalResult

logger = logging.getLogger(__name__)


class FullContextMemory(MemorySystem):
    """
    Baseline: inject the complete conversation history into every LLM call.
    build_memory() simply stores all turns in RAM.
    retrieve() returns all turns regardless of the question.
    """

    def __init__(self) -> None:
        self._turns: list[Any] = []  # list[Turn]
        self._speaker_a: str = ""
        self._speaker_b: str = ""

    @property
    def method_name(self) -> str:
        return "full_context"

    def reset(self) -> None:
        self._turns = []
        self._speaker_a = ""
        self._speaker_b = ""

    def build_memory(
        self,
        conv_id: str,
        sessions: list[Session],
        speaker_a: str,
        speaker_b: str,
    ) -> None:
        self._speaker_a = speaker_a
        self._speaker_b = speaker_b
        self._turns = []
        for session in sessions:
            for turn in session.turns:
                self._turns.append(turn)
        logger.info(
            "[FullContext] %s: stored %d turns from %d sessions",
            conv_id, len(self._turns), len(sessions),
        )

    def retrieve(self, question: str, top_k: int = 5) -> RetrievalResult:
        # Full context: ignore question, return everything
        lines: list[str] = []
        for turn in self._turns:
            lines.append(f"[{turn.speaker}]: {turn.text}")
        context_text = "\n".join(lines)
        retrieved_ids = [t.dia_id for t in self._turns if t.dia_id]
        token_count = count_tokens(context_text)
        return RetrievalResult(
            context_text=context_text,
            retrieved_ids=retrieved_ids,
            token_count=token_count,
            raw_result={"num_turns": len(self._turns)},
        )
