from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from .embedder import (
    TextEmbedder,
    build_experience_embedding_text,
    build_qa_embedding_text,
    build_segment_embedding_text,
)
from .storage import MemoryStorage
from .summarizer import TemplateSummarizer
from .vector_store import ChromaVectorStore


logger = logging.getLogger(__name__)
SHANGHAI_TZ = timezone(timedelta(hours=8))


class MemoryManager:
    """Structured QA -> Segment -> Experience memory manager."""

    def __init__(
        self,
        storage: MemoryStorage,
        vector_store: ChromaVectorStore,
        embedder: TextEmbedder,
        summarizer: TemplateSummarizer | None = None,
        segment_summary_qa_threshold: int = 5,
        experience_summary_segment_threshold: int = 5,
    ) -> None:
        self.storage = storage
        self.summarizer = summarizer or TemplateSummarizer()
        self.vector_store = vector_store
        self.embedder = embedder
        self.segment_summary_qa_threshold = segment_summary_qa_threshold
        self.experience_summary_segment_threshold = experience_summary_segment_threshold

    def add_qa(
        self,
        topic_result: dict[str, Any],
        user_input: str,
        assistant_output: str = "",
        tools: list[dict[str, Any]] | None = None,
        timestamp: str | None = None,
        state_key: str = "default",
    ) -> dict[str, Any]:
        """Add one structured topic result into QA/Segment/Experience memory."""

        topic = self._required_text(topic_result, "topic")
        core_entity = self._required_text(topic_result, "core_entity")
        intent = self._required_text(topic_result, "intent")
        entities = topic_result.get("entities") or [core_entity]
        confidence = float(topic_result.get("confidence", 0.0))
        reasoning = str(topic_result.get("reasoning") or "")
        timestamp = timestamp or self._now()

        logger.info(
            "写入主题记忆 state=%s topic=%s core_entity=%s intent=%s",
            state_key,
            topic,
            core_entity,
            intent,
        )

        try:
            runtime = self.storage.get_runtime_state(state_key)
            current_experience = self.storage.get_experience(
                runtime["current_experience_id"] if runtime else None
            )
            current_segment = self.storage.get_segment(
                runtime["current_segment_id"] if runtime else None
            )

            action = "append_segment"

            # Experience 是长期聚合键。只要 topic/core_entity 变了，就先沉淀旧状态，
            # 再去库里找目标 Experience，避免不同长期记忆串线。
            if not self._same_experience(current_experience, topic, core_entity):
                if current_segment:
                    self._summarize_segment(current_segment, timestamp, reason="experience_switch")
                if current_experience:
                    self._summarize_experience(current_experience, timestamp, reason="experience_switch")

                current_experience = self.storage.find_experience(topic, core_entity)
                if current_experience:
                    logger.info("命中已有 Experience experience_id=%s", current_experience["experience_id"])
                    current_segment = self.storage.find_latest_segment(current_experience["experience_id"])
                    action = "switch_experience"
                else:
                    current_experience = self._create_experience(topic, core_entity, timestamp)
                    current_segment = None
                    action = "new_experience"

            # Segment 是同一 Experience 内的意图聚合。intent 不同就切 Segment；
            # intent 相同则继续追加 QA。
            if not current_segment or current_segment["intent"] != intent:
                if current_segment:
                    self._summarize_segment(current_segment, timestamp, reason="intent_switch")
                current_segment = self._create_segment(current_experience, topic, core_entity, intent, timestamp)
                action = "new_segment" if action == "append_segment" else action

            qa = {
                "qa_id": self._new_id("qa"),
                "timestamp": timestamp,
                "user_input": user_input,
                "assistant_output": assistant_output,
                "tools": tools or [],
                "topic": topic,
                "intent": intent,
                "core_entity": core_entity,
                "entities": [str(entity) for entity in entities if str(entity).strip()],
                "segment_id": current_segment["segment_id"],
                "status": "active",
                "confidence": confidence,
                "reasoning": reasoning,
            }
            self.storage.insert_qa(qa)
            self.upsert_qa_vector(qa["qa_id"])
            logger.info(
                "QA 已创建 qa_id=%s segment_id=%s experience_id=%s",
                qa["qa_id"],
                current_segment["segment_id"],
                current_experience["experience_id"],
            )

            current_segment["qa_ids"].append(qa["qa_id"])
            current_segment["updated_at"] = timestamp
            self._maybe_summarize_segment_by_threshold(current_segment, timestamp)
            self.storage.update_segment(current_segment)
            self.upsert_segment_vector(current_segment["segment_id"])

            self._attach_segment_to_experience(current_experience, current_segment, intent, timestamp)
            self._maybe_summarize_experience_by_threshold(current_experience, timestamp)
            self.storage.update_experience(current_experience)
            self.upsert_experience_vector(current_experience["experience_id"])
            self.storage.upsert_runtime_state(
                state_key=state_key,
                current_experience_id=current_experience["experience_id"],
                current_segment_id=current_segment["segment_id"],
                updated_at=timestamp,
            )
            self.storage.commit()
            return {
                "qa_id": qa["qa_id"],
                "segment_id": current_segment["segment_id"],
                "experience_id": current_experience["experience_id"],
                "action": action,
            }
        except Exception:
            self.storage.rollback()
            logger.exception("结构化记忆写入失败，已回滚")
            raise

    def _create_experience(self, topic: str, core_entity: str, now: str) -> dict[str, Any]:
        experience = {
            "experience_id": self._new_id("exp"),
            "topic": topic,
            "core_entity": core_entity,
            "intents_link": [],
            "segment_ids": [],
            "summary": "",
            "state": {"status": "in_progress", "current_segment_id": ""},
            "created_at": now,
            "updated_at": now,
            "version": 1,
            "last_summarized_segment_count": 0,
        }
        self.storage.insert_experience(experience)
        self.upsert_experience_vector(experience["experience_id"])
        logger.info("新建 Experience experience_id=%s topic=%s entity=%s", experience["experience_id"], topic, core_entity)
        return experience

    def _create_segment(
        self,
        experience: dict[str, Any],
        topic: str,
        core_entity: str,
        intent: str,
        now: str,
    ) -> dict[str, Any]:
        segment = {
            "segment_id": self._new_id("seg"),
            "topic": topic,
            "intent": intent,
            "core_entity": core_entity,
            "qa_ids": [],
            "status": "open",
            "summary": "",
            "experience_id": experience["experience_id"],
            "created_at": now,
            "updated_at": now,
            "version": 1,
            "last_summarized_qa_count": 0,
        }
        self.storage.insert_segment(segment)
        self.upsert_segment_vector(segment["segment_id"])
        logger.info(
            "新建 Segment segment_id=%s experience_id=%s intent=%s",
            segment["segment_id"],
            experience["experience_id"],
            intent,
        )
        return segment

    def _attach_segment_to_experience(
        self,
        experience: dict[str, Any],
        segment: dict[str, Any],
        intent: str,
        now: str,
    ) -> None:
        if segment["segment_id"] not in experience["segment_ids"]:
            experience["segment_ids"].append(segment["segment_id"])
        if intent not in experience["intents_link"]:
            experience["intents_link"].append(intent)
        experience["state"] = {
            "status": "in_progress",
            "current_segment_id": segment["segment_id"],
        }
        experience["updated_at"] = now

    def _maybe_summarize_segment_by_threshold(self, segment: dict[str, Any], now: str) -> None:
        qa_count = len(segment["qa_ids"])
        if qa_count - segment["last_summarized_qa_count"] >= self.segment_summary_qa_threshold:
            self._summarize_segment(segment, now, reason="qa_threshold")

    def _maybe_summarize_experience_by_threshold(self, experience: dict[str, Any], now: str) -> None:
        segment_count = len(experience["segment_ids"])
        if segment_count - experience["last_summarized_segment_count"] >= self.experience_summary_segment_threshold:
            self._summarize_experience(experience, now, reason="segment_threshold")

    def _summarize_segment(self, segment: dict[str, Any], now: str, reason: str) -> None:
        qa_items = [
            self._get_qa(qa_id)
            for qa_id in segment.get("qa_ids", [])
        ]
        qa_items = [qa for qa in qa_items if qa]
        if not qa_items:
            return
        segment["summary"] = self.summarizer.summarize_segment(segment, qa_items)
        segment["last_summarized_qa_count"] = len(segment["qa_ids"])
        segment["version"] += 1
        segment["updated_at"] = now
        self.storage.update_segment(segment)
        self.upsert_segment_vector(segment["segment_id"])
        logger.info(
            "Segment 总结已更新 segment_id=%s reason=%s qa_count=%s version=%s",
            segment["segment_id"],
            reason,
            len(segment["qa_ids"]),
            segment["version"],
        )

    def _summarize_experience(self, experience: dict[str, Any], now: str, reason: str) -> None:
        segments = [
            self.storage.get_segment(segment_id)
            for segment_id in experience.get("segment_ids", [])
        ]
        segments = [segment for segment in segments if segment]
        experience["summary"] = self.summarizer.summarize_experience(experience, segments)
        experience["last_summarized_segment_count"] = len(experience["segment_ids"])
        experience["version"] += 1
        experience["updated_at"] = now
        self.storage.update_experience(experience)
        self.upsert_experience_vector(experience["experience_id"])
        logger.info(
            "Experience 总结已更新 experience_id=%s reason=%s segment_count=%s version=%s",
            experience["experience_id"],
            reason,
            len(experience["segment_ids"]),
            experience["version"],
        )

    def _get_qa(self, qa_id: str) -> dict[str, Any] | None:
        return self.storage.get_qa(qa_id)

    def upsert_experience_vector(self, experience_id: str) -> None:
        experience = self.storage.get_experience(experience_id)
        if not experience:
            raise ValueError(f"Unknown experience_id: {experience_id}")
        recent_segments = self.storage.list_segments_by_experience_ids([experience_id])[:3]
        vector_memory = {**experience, "recent_segments": recent_segments}
        text = build_experience_embedding_text(vector_memory)
        self.vector_store.upsert(
            memory_type="experience",
            memory_id=experience_id,
            text=text,
            embedding=self.embedder.embed(text),
            updated_at=experience["updated_at"],
            metadata={
                "experience_id": experience_id,
                "topic": experience["topic"],
                "core_entity": experience["core_entity"],
                "intents": experience.get("intents_link") or [],
                "version": experience["version"],
            },
        )
        logger.debug("Experience 向量已写入 id=%s", experience_id)

    def upsert_segment_vector(self, segment_id: str) -> None:
        segment = self.storage.get_segment(segment_id)
        if not segment:
            raise ValueError(f"Unknown segment_id: {segment_id}")
        recent_qas = [
            self.storage.get_qa(qa_id)
            for qa_id in (segment.get("qa_ids") or [])[-3:]
        ]
        vector_memory = {
            **segment,
            "recent_qa_inputs": [
                qa["user_input"] for qa in recent_qas if qa
            ],
        }
        text = build_segment_embedding_text(vector_memory)
        self.vector_store.upsert(
            memory_type="segment",
            memory_id=segment_id,
            text=text,
            embedding=self.embedder.embed(text),
            updated_at=segment["updated_at"],
            metadata={
                "segment_id": segment_id,
                "experience_id": segment["experience_id"],
                "topic": segment["topic"],
                "core_entity": segment["core_entity"],
                "intent": segment["intent"],
                "status": segment["status"],
                "version": segment["version"],
            },
        )
        logger.debug("Segment 向量已写入 id=%s", segment_id)

    def upsert_qa_vector(self, qa_id: str) -> None:
        qa = self.storage.get_qa(qa_id)
        if not qa:
            raise ValueError(f"Unknown qa_id: {qa_id}")
        text = build_qa_embedding_text(qa)
        self.vector_store.upsert(
            memory_type="qa",
            memory_id=qa_id,
            text=text,
            embedding=self.embedder.embed(text),
            updated_at=qa["timestamp"],
            metadata={
                "qa_id": qa_id,
                "segment_id": qa["segment_id"],
                "topic": qa["topic"],
                "core_entity": qa["core_entity"],
                "intent": qa["intent"],
                "timestamp": qa["timestamp"],
                "status": qa["status"],
                "confidence": float(qa["confidence"]),
            },
        )
        logger.debug("QA 向量已写入 id=%s", qa_id)

    def _same_experience(
        self, experience: dict[str, Any] | None, topic: str, core_entity: str
    ) -> bool:
        return bool(
            experience
            and experience["topic"] == topic
            and experience["core_entity"] == core_entity
        )

    def _required_text(self, value: dict[str, Any], key: str) -> str:
        text = str(value.get(key) or "").strip()
        if not text:
            raise ValueError(f"{key} is required")
        return text

    def _new_id(self, prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:12]}"

    def _now(self) -> str:
        return datetime.now(SHANGHAI_TZ).isoformat()
