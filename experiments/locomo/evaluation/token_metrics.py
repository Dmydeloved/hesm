"""
Context efficiency metrics.

Counts tokens in the retrieved context using tiktoken (cl100k_base encoding)
and computes the compression ratio versus the full conversation.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

# Lazily imported so the module is importable even if tiktoken has issues
_encoder: Any = None


def _get_encoder(encoding: str = "cl100k_base") -> Any:
    global _encoder
    if _encoder is None:
        try:
            import tiktoken
            _encoder = tiktoken.get_encoding(encoding)
        except Exception as exc:
            logger.warning("tiktoken unavailable, falling back to word count: %s", exc)
            _encoder = _FallbackEncoder()
    return _encoder


class _FallbackEncoder:
    """Simple word-count fallback when tiktoken is not available."""

    def encode(self, text: str) -> list[str]:  # type: ignore[return]
        return text.split()


def count_tokens(text: str, encoding: str = "cl100k_base") -> int:
    """Count tokens in *text* using tiktoken (or word count as fallback)."""
    enc = _get_encoder(encoding)
    return len(enc.encode(text))


def compute_compression_ratio(
    retrieved_tokens: int,
    total_conversation_tokens: int,
) -> float:
    """
    Token Compression Ratio = total_conversation_tokens / retrieved_tokens.

    A ratio of 10× means the retrieved context is 10× smaller than the full
    conversation history. Returns 1.0 if retrieved_tokens is 0 or if the
    ratio would be ≤ 1.
    """
    if retrieved_tokens <= 0:
        return float(total_conversation_tokens) if total_conversation_tokens > 0 else 1.0
    ratio = total_conversation_tokens / retrieved_tokens
    return max(ratio, 1.0)  # ratio < 1 means context larger than conversation (edge case)
