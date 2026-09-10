from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import Any, Protocol

from .config import get as config_get


logger = logging.getLogger(__name__)


TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+")
DEFAULT_MAX_INPUT_TOKENS = 8192
DEFAULT_CHUNK_TOKENS = 7800


class TextEmbedder(Protocol):
    def embed(self, text: str) -> list[float]:
        """Encode one text into a vector."""


class BailianEmbedder:
    """Alibaba Bailian embedding client using its OpenAI-compatible endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
        chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    ) -> None:
        from openai import OpenAI

        api_key = api_key or config_get("embedding", "api_key")
        if not api_key:
            raise ValueError("Set embedding.api_key in config/hesm.yaml.")
        model = model or config_get("embedding", "model", "text-embedding-v4")
        base_url = base_url or config_get(
            "embedding", "base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.client = OpenAI(api_key=str(api_key), base_url=str(base_url))
        self.model = str(model)
        self.max_input_tokens = int(max_input_tokens)
        self.chunk_tokens = int(chunk_tokens)
        if self.max_input_tokens < 1:
            raise ValueError("embedding.max_input_tokens must be positive")
        if not 1 <= self.chunk_tokens <= self.max_input_tokens:
            raise ValueError(
                "embedding.chunk_tokens must be between 1 and max_input_tokens"
            )

    def embed(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Embedding text cannot be empty.")
        chunks, token_count = split_embedding_text(
            text,
            max_input_tokens=self.max_input_tokens,
            chunk_tokens=self.chunk_tokens,
        )
        logger.info(
            "Embedding request model=%s text_length=%s token_count=%s chunks=%s",
            self.model,
            len(text),
            token_count,
            len(chunks),
        )
        if len(chunks) > 1:
            logger.warning(
                "Embedding content exceeds one-call limit; model=%s "
                "token_count=%s max_input_tokens=%s chunk_tokens=%s chunks=%s",
                self.model,
                token_count,
                self.max_input_tokens,
                self.chunk_tokens,
                len(chunks),
            )
        vectors = []
        for chunk in chunks:
            response = self.client.embeddings.create(
                model=self.model,
                input=chunk,
            )
            vectors.append(list(response.data[0].embedding))
        embedding = _mean_normalized(vectors)
        logger.info(
            "Embedding response model=%s dimension=%s chunks=%s",
            self.model,
            len(embedding),
            len(chunks),
        )
        return embedding


def embedding_token_count(text: str) -> int:
    """Count tokens with the experiment tokenizer, with a safe byte fallback."""
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        # A byte is the smallest possible tokenizer unit, so byte length is a
        # conservative upper bound when tiktoken is unavailable.
        return len(text.encode("utf-8"))


def split_embedding_text(
    text: str,
    *,
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
) -> tuple[list[str], int]:
    """Return nonempty chunks that are safe for one embedding API call each."""
    if not text or not text.strip():
        raise ValueError("Embedding text cannot be empty.")
    max_input_tokens = int(max_input_tokens)
    chunk_tokens = int(chunk_tokens)
    if max_input_tokens < 1 or not 1 <= chunk_tokens <= max_input_tokens:
        raise ValueError("Invalid embedding token limits")
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        token_ids = encoding.encode(text)
        token_count = len(token_ids)
        if token_count <= max_input_tokens:
            return [text], token_count
        # Split on Python character boundaries. Decoding arbitrary token slices
        # can split the UTF-8 bytes of one CJK character across two chunks.
        chunks = []
        start = 0
        while start < len(text):
            low, high = start + 1, len(text)
            best = start
            while low <= high:
                middle = (low + high) // 2
                if len(encoding.encode(text[start:middle])) <= chunk_tokens:
                    best = middle
                    low = middle + 1
                else:
                    high = middle - 1
            if best == start:
                best = start + 1
            chunks.append(text[start:best])
            start = best
        return [chunk for chunk in chunks if chunk.strip()], token_count
    except ImportError:
        raw = text.encode("utf-8")
        token_count = len(raw)
        if token_count <= max_input_tokens:
            return [text], token_count
        chunks: list[str] = []
        current: list[str] = []
        current_bytes = 0
        for character in text:
            encoded_size = len(character.encode("utf-8"))
            if current and current_bytes + encoded_size > chunk_tokens:
                chunks.append("".join(current))
                current = []
                current_bytes = 0
            current.append(character)
            current_bytes += encoded_size
        if current:
            chunks.append("".join(current))
        return [chunk for chunk in chunks if chunk.strip()], token_count


def _mean_normalized(vectors: list[list[float]]) -> list[float]:
    if not vectors or not vectors[0]:
        raise ValueError("Embedding provider returned an empty vector")
    if len(vectors) == 1:
        return vectors[0]
    dimensions = len(vectors[0])
    if any(len(vector) != dimensions for vector in vectors):
        raise ValueError("Embedding provider returned inconsistent dimensions")
    averaged = [
        sum(vector[index] for vector in vectors) / len(vectors)
        for index in range(dimensions)
    ]
    norm = math.sqrt(sum(value * value for value in averaged))
    return [value / norm for value in averaged] if norm else averaged


class HashingEmbedder:
    """Deterministic offline embedder used only by unit tests."""

    def __init__(self, dimensions: int = 128) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            vector[int.from_bytes(digest[:4], "big") % self.dimensions] += (
                1.0 if digest[4] % 2 == 0 else -1.0
            )
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text or "")]


def topic_entity_text(topic: str, core_entity: str, extra: object = None) -> str:
    parts = [topic, core_entity]
    if isinstance(extra, list):
        parts.extend(str(item) for item in extra)
    elif extra:
        parts.append(str(extra))
    return " ".join(part for part in parts if part)


def _text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def build_experience_embedding_text(experience: dict[str, Any]) -> str:
    """Build a bounded Experience vector document without aggregating QA entities."""

    state = experience.get("state") if isinstance(experience.get("state"), dict) else {}
    history = (
        experience.get("history_experience")
        if isinstance(experience.get("history_experience"), dict)
        else {}
    )
    recent_segments = experience.get("recent_segments") or []
    recent_text = "；".join(
        " / ".join(
            part
            for part in (
                str(segment.get("intent") or "").strip(),
                str(segment.get("summary") or "").strip(),
            )
            if part
        )
        for segment in recent_segments[-3:]
        if isinstance(segment, dict)
    )
    state_text = "，".join(
        f"{key}={state[key]}"
        for key in ("status", "current_segment_id")
        if state.get(key)
    )
    return "\n".join(
        [
            f"主题：{experience.get('topic', '')}",
            f"核心实体：{experience.get('core_entity', '')}",
            f"相关意图：{'、'.join(_text_list(experience.get('intents_link')))}",
            f"长期摘要：{experience.get('summary', '')}",
            "历史经验：" + " / ".join(
                str(history.get(key) or "").strip()
                for key in ("topic", "core_entity", "summary")
                if str(history.get(key) or "").strip()
            ),
            f"当前状态：{state_text}",
            f"最近阶段：{recent_text}",
        ]
    )


def build_segment_embedding_text(segment: dict[str, Any]) -> str:
    """Build a Segment vector document from intent, summary and recent questions."""

    recent_inputs = _text_list(segment.get("recent_qa_inputs"))[-3:]
    return "\n".join(
        [
            f"主题：{segment.get('topic', '')}",
            f"核心实体：{segment.get('core_entity', '')}",
            f"阶段意图：{segment.get('intent', '')}",
            f"阶段摘要：{segment.get('summary', '')}",
            f"片段状态：{segment.get('status', '')}",
            f"最近问题：{'；'.join(recent_inputs)}",
        ]
    )


def build_qa_embedding_text(qa: dict[str, Any]) -> str:
    """Build a QA vector document containing the original evidence text."""

    answer_excerpt = str(qa.get("assistant_output") or "")[:500]
    return "\n".join(
        [
            f"主题：{qa.get('topic', '')}",
            f"核心实体：{qa.get('core_entity', '')}",
            f"用户意图：{qa.get('intent', '')}",
            f"相关实体：{'、'.join(_text_list(qa.get('entities')))}",
            f"用户问题：{qa.get('user_input', '')}",
            f"助手回答摘要：{answer_excerpt}",
        ]
    )
