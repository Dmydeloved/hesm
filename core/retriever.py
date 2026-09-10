"""检索当前 Experience 及其最近的下层记忆。"""

from __future__ import annotations

import json
import logging
from typing import Any

from .manager import MemoryManager


logger = logging.getLogger(__name__)


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


def build_history_context(history_experience: Any) -> str:
    """把未命中新 Experience 时召回的历史经验直接转换为上下文。"""
    if isinstance(history_experience, (dict, list)):
        text = json.dumps(history_experience, ensure_ascii=False)
    else:
        text = str(history_experience or "").strip()
    if not text or text in {"{}", "[]"}:
        text = "未召回可复用的历史经验。"
    return f"【历史经验】\n{text}"


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
        logger.info(
            "Hybrid retrieval started state_key=%s topic=%s core_entity=%s intent=%s query=%s",
            state_key,
            topic,
            core_entity,
            intent,
            query,
        )

        # 统一复用 MemoryManager 的路由规则和 runtime 维护逻辑。
        experience, _current_segment = self.manager.route_experience(
            state_key=state_key,
            topic=topic,
            core_entity=core_entity,
            intent=intent,
            query=query,
        )
        self.storage.commit()
        logger.info(
            "Hybrid retrieval routed state_key=%s experience_id=%s current_segment_id=%s",
            state_key,
            experience.get("experience_id", ""),
            (_current_segment or {}).get("segment_id", ""),
        )

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
        logger.info(
            "Hybrid retrieval completed state_key=%s experience_count=%s segment_count=%s qa_count=%s context=%s",
            state_key,
            len(experiences),
            len(segments),
            len(qas),
            context,
        )
        return {
            "experiences": experiences,
            "segments": segments,
            "qas": qas,
            "context": context,
        }


class ReadOnlyHybridRetriever:
    """复用 HESM 路由规则检索记忆，但不创建或更新任何业务记忆。"""

    def __init__(
        self,
        manager: MemoryManager,
        *,
        segment_limit: int = 2,
        qa_limit: int = 5,
    ) -> None:
        self.manager = manager
        self.storage = manager.storage
        self.segment_limit = max(1, int(segment_limit))
        self.qa_limit = max(1, int(qa_limit))

    def _route_existing(
        self,
        *,
        state_key: str,
        topic: str,
        core_entity: str,
    ) -> tuple[dict[str, Any] | None, str]:
        """按 runtime、SQLite、路由向量的顺序查找已有 Experience。"""
        runtime = self.storage.get_runtime_state(state_key)
        current = self.storage.get_experience(
            runtime.get("current_experience_id") if runtime else None
        )
        if self.manager._same_experience(current, topic, core_entity):
            return current, "runtime"

        current = self.storage.find_active_experience(topic, core_entity)
        if current:
            return current, "sqlite"

        current = self.manager._find_experience_by_vector(topic, core_entity)
        if current:
            return current, "vector"
        return None, "history_only"

    def retriever(
        self,
        topic: str,
        core_entity: str,
        query: str,
        intent: str = "",
        state_key: str = "default",
    ) -> dict[str, Any]:
        """检索已有层级；未命中时只召回历史经验，不创建 Experience。"""
        topic = str(topic or "").strip()
        core_entity = str(core_entity or "").strip()
        query = str(query or "").strip()
        intent = str(intent or "").strip() or "查询"
        if not topic or not core_entity or not query:
            raise ValueError("topic, core_entity and query must not be empty")

        experience, route_source = self._route_existing(
            state_key=state_key,
            topic=topic,
            core_entity=core_entity,
        )
        if experience is None:
            recaller = self.manager.experience_recaller
            if recaller is None:
                raise RuntimeError("experience_recaller is not configured")
            recalled = recaller.recall(
                topic=topic,
                core_entity=core_entity,
                query=query,
                intent=intent,
            )
            history = (
                recalled.get("history_experience", {})
                if isinstance(recalled, dict)
                else {}
            )
            return {
                "route_status": route_source,
                "experiences": [],
                "segments": [],
                "qas": [],
                "history_experience": history,
                "history_recall": recalled,
                "context": build_history_context(history),
            }

        latest_segments = self.storage.list_latest_segments(
            str(experience["experience_id"]),
            self.segment_limit,
        )
        segments = sorted(
            latest_segments,
            key=lambda item: (
                str(item.get("updated_at") or ""),
                str(item.get("created_at") or ""),
                str(item.get("segment_id") or ""),
            ),
        )
        segment_ids = [str(segment["segment_id"]) for segment in segments]
        latest_qas = self.storage.list_latest_qas(segment_ids, self.qa_limit)
        qas = sorted(
            latest_qas,
            key=lambda item: (
                str(item.get("timestamp") or ""),
                str(item.get("qa_id") or ""),
            ),
        )
        experiences = [experience]
        return {
            "route_status": route_source,
            "experiences": experiences,
            "segments": segments,
            "qas": qas,
            "history_experience": experience.get("history_experience") or {},
            "context": build_context_text(experiences, segments, qas),
        }


__all__ = [
    "HybridRetriever",
    "ReadOnlyHybridRetriever",
    "build_context_text",
    "build_history_context",
]
