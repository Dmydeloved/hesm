from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Protocol

from .config import get as config_get


TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+")


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

    def embed(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Embedding text cannot be empty.")
        response = self.client.embeddings.create(model=self.model, input=text)
        return list(response.data[0].embedding)


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
