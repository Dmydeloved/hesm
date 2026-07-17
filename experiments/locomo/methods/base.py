"""
Base interface for all memory systems in the LoCoMo benchmark.

All five methods (Full Context, Vector RAG, Mem0, A-MEM, HESM) implement
MemorySystem so the QA runner can treat them uniformly.

Also defines the shared LLMAnswerGenerator used by the runner for all methods
(ensures identical answer-generation conditions across methods).
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import openai

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Data transfer objects
# ─────────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """Structured retrieval output — passed to generate_answer and evaluators."""

    context_text: str            # text block injected into the LLM prompt
    retrieved_ids: list[str]     # dia_ids of retrieved turns, e.g. ["D1:3", "D2:8"]
    token_count: int             # number of tokens in context_text
    raw_result: dict[str, Any] = field(default_factory=dict)  # full result for debug


# ─────────────────────────────────────────────────────────────
# Unified MemorySystem interface
# ─────────────────────────────────────────────────────────────

class MemorySystem(ABC):
    """
    Abstract base for all memory systems evaluated on LoCoMo.

    Lifecycle per conversation:
      1. reset()           — clear all per-conversation state
      2. build_memory(...)  — ingest turns in temporal order
      3. retrieve(...)      — retrieve context for a question
    """

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Short identifier used in output file names and tables."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """
        Clear all memory state for the current conversation.
        Must be called before build_memory() for each new conversation.
        """
        ...

    @abstractmethod
    def build_memory(
        self,
        conv_id: str,
        sessions: list[Any],   # list[Session] — imported at call site to avoid circular
        speaker_a: str,
        speaker_b: str,
    ) -> None:
        """
        Build memory from all sessions of a conversation in temporal order.

        Implementations must process sessions strictly in the provided order
        (ascending session_num) and must not access QA pairs or future sessions.

        Args:
            conv_id:   unique conversation identifier (used for storage isolation)
            sessions:  ordered list of Session objects
            speaker_a: name of first speaker
            speaker_b: name of second speaker
        """
        ...

    @abstractmethod
    def retrieve(self, question: str, top_k: int = 5) -> RetrievalResult:
        """
        Retrieve relevant memory context for a question.

        Args:
            question: the natural-language question
            top_k:    maximum number of memory items to include

        Returns:
            RetrievalResult with context_text, retrieved_ids, token_count
        """
        ...


# ─────────────────────────────────────────────────────────────
# Shared LLM answer generator (identical for all methods)
# ─────────────────────────────────────────────────────────────

_QA_PROMPT = """\
You are a helpful assistant with access to memory notes from a long-term \
conversation between {speaker_a} and {speaker_b}.

Using ONLY the information in the memory context below, answer the question \
concisely and accurately. If the answer is not in the context, say "Unknown".

Memory Context:
{context}

Question: {question}
Answer:"""


class LLMAnswerGenerator:
    """
    Generates answers for all memory systems using the topic_extraction LLM config.

    Constructed once and shared across all MemorySystem instances to ensure
    identical answer-generation conditions for a fair comparison.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        te: dict[str, Any] = config.get("topic_extraction", {})
        api_key = te.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
        base_url = te.get("base_url", "https://api.openai.com/v1/")
        self.model: str = te.get("model", "gpt-4")
        self.max_retries: int = int(te.get("max_retries", 3))
        self.retry_delay: float = float(te.get("retry_delay", 2.0))
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def generate(
        self,
        question: str,
        context: str,
        speaker_a: str = "Speaker A",
        speaker_b: str = "Speaker B",
    ) -> str:
        """Call LLM and return the answer string. Returns "" on total failure."""
        prompt = _QA_PROMPT.format(
            speaker_a=speaker_a,
            speaker_b=speaker_b,
            context=context if context.strip() else "(no memory available)",
            question=question,
        )
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                return resp.choices[0].message.content.strip()
            except Exception as exc:
                logger.warning(
                    "Answer generation attempt %d/%d failed: %s",
                    attempt, self.max_retries, exc,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)
        return ""
