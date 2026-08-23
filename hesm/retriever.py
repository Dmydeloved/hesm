"""检索当前 Experience 及其最近的下层记忆。"""

from __future__ import annotations

import json
from typing import Any

from .manager import MemoryManager


def _experience_status(experience: dict[str, Any]) -> str:
    if experience.get("status"):
        return str(experience["status"])
    state = experience.get("state")
    return str(state.get("status") or "") if isinstance(state, dict) else ""


def _memory_text(value: Any) -> str:
    """把结构化记忆转换为可读且稳定的 Prompt 文本。"""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value or "")


def build_context_text(
    experiences: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    qas: list[dict[str, Any]],
) -> str:
    """根据一个 Experience 及其最近下层记忆构造 Prompt 上下文。"""
    if not experiences:
        return "【长期记忆 Experience】\n未找到相关长期记忆。"

    experience = experiences[0]
    lines = [
        "【长期记忆 Experience】",
        f"主题：{experience.get('topic', '')}",
        f"核心实体：{experience.get('core_entity', '')}",
        f"状态：{_experience_status(experience)}",
        f"摘要：{_memory_text(experience.get('summary'))}",
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
                f"   摘要：{_memory_text(segment.get('summary'))}",
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
    """通过 MemoryManager 路由当前 Experience，并加载最近记忆。"""

    def __init__(
        self,
        manager: MemoryManager,
    ) -> None:
        self.manager = manager
        self.storage = manager.storage

    def retriever(
        self,
        topic: str,
        core_entity: str,
        query: str,
        intent: str = "",
        state_key: str = "default",
    ) -> dict[str, Any]:
        """返回当前 Experience、最近两个 Segment 和最近五条 QA。"""
        topic = str(topic or "").strip()
        core_entity = str(core_entity or "").strip()
        query = str(query or "").strip()
        intent = str(intent or "").strip() or "查询"
        if not topic or not core_entity or not query:
            raise ValueError("topic, core_entity and query must not be empty")

        # 统一复用 MemoryManager 的路由规则和 runtime 维护逻辑。
        experience, _current_segment = self.manager.route_experience(
            state_key=state_key,
            topic=topic,
            core_entity=core_entity,
            intent=intent,
            query=query,
        )
        self.storage.commit()

        # 固定加载当前 Experience 下最近两个 Segment。
        latest_segments = self.storage.list_latest_segments(
            str(experience["experience_id"]),
            2,
        )
        segments = sorted(
            latest_segments,
            key=lambda item: (
                str(item.get("updated_at") or ""),
                str(item.get("created_at") or ""),
                str(item.get("segment_id") or ""),
            ),
        )

        # 从上述 Segment 中取最近五条 QA，再恢复为时间升序。
        segment_ids = [segment["segment_id"] for segment in segments]
        latest_qas = self.storage.list_latest_qas(segment_ids, 5)
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
