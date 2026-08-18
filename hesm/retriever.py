"""Retrieve the current Experience and its recent descendants."""

from __future__ import annotations

import json
from typing import Any, Callable

from .storage import MemoryStorage


def _experience_status(experience: dict[str, Any]) -> str:
    state = experience.get("state")
    return str(state.get("status") or "") if isinstance(state, dict) else ""


def build_context_text(
    experiences: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    qas: list[dict[str, Any]],
) -> str:
    """Build the prompt context from one Experience and recent descendants."""
    if not experiences:
        return "【长期记忆 Experience】\n未找到相关长期记忆。"

    experience = experiences[0]
    lines = [
        "【长期记忆 Experience】",
        f"主题：{experience.get('topic', '')}",
        f"核心实体：{experience.get('core_entity', '')}",
        f"状态：{_experience_status(experience)}",
        f"摘要：{experience.get('summary', '')}",
    ]
    history = experience.get("history_experience")
    if isinstance(history, dict) and history:
        history = history.get("summary") or json.dumps(
            history, ensure_ascii=False
        )
    if str(history or "").strip():
        lines.append(f"历史经验：{str(history).strip()}")

    lines.append("\n【最近 Segment】")
    if not segments:
        lines.append("无。")
    for index, segment in enumerate(segments, 1):
        lines.extend(
            [
                f"{index}. 意图：{segment.get('intent', '')}",
                f"   摘要：{segment.get('summary', '')}",
                f"   更新时间：{segment.get('updated_at', '')}",
            ]
        )

    lines.append("\n【最近 QA】")
    if not qas:
        lines.append("无。")
    for index, qa in enumerate(qas, 1):
        lines.extend(
            [
                f"{index}. 用户：{qa.get('user_input', '')}",
                f"   助手：{qa.get('assistant_output', '')}",
                f"   时间：{qa.get('timestamp', '')}",
            ]
        )
    return "\n".join(lines)


class HybridRetriever:
    """Retrieve an active Experience or create one, then load recent context."""

    def __init__(
        self,
        storage: MemoryStorage,
        create_experience: Callable[..., dict[str, Any]],
    ) -> None:
        self.storage = storage
        self.create_experience = create_experience

    def retriever(
        self,
        topic: str,
        core_entity: str,
        query: str,
    ) -> dict[str, Any]:
        """Return the current Experience and its recent memory context."""
        from .config import get as config_get

        topic = str(topic or "").strip()
        core_entity = str(core_entity or "").strip()
        query = str(query or "").strip()
        if not topic or not core_entity or not query:
            raise ValueError("topic, core_entity and query must not be empty")

        # 先从 SQLite 获取主题、实体完全一致的最新进行中 Experience。
        experience = self.storage.find_active_experience(topic, core_entity)
        if experience is None:
            experience = self.create_experience(
                topic=topic,
                core_entity=core_entity,
                query=query,
            )

        # 只加载当前 Experience 下最近的有限数量 Segment。
        segment_limit = max(1, int(config_get("api", "top_segment", 2)))
        latest_segments = self.storage.list_latest_segments(
            str(experience["experience_id"]),
            segment_limit,
        )
        segments = sorted(
            latest_segments,
            key=lambda item: (
                str(item.get("updated_at") or ""),
                str(item.get("created_at") or ""),
                str(item.get("segment_id") or ""),
            ),
        )

        # 从上述 Segment 中取最近四轮 QA，再恢复为时间升序。
        segment_ids = [segment["segment_id"] for segment in segments]
        latest_qas = self.storage.list_latest_qas(segment_ids, 4)
        qas = sorted(
            latest_qas,
            key=lambda item: (
                str(item.get("timestamp") or ""),
                str(item.get("qa_id") or ""),
            ),
        )

        experiences = [experience]
        context = build_context_text(experiences, segments, qas)
        return {
            "experiences": experiences,
            "segments": segments,
            "qas": qas,
            "context": context,
        }


__all__ = ["HybridRetriever", "build_context_text"]
