from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from .embedder import TextEmbedder
from .recaller import ExperienceRecaller
from .storage import MemoryStorage
from .summarizer import SummarizerProtocol, TemplateSummarizer
from .time_utils import format_timestamp
from .vector_store import (
    ChromaVectorStore,
    build_vector_document,
    build_vector_metadata,
)


logger = logging.getLogger(__name__)


class MemoryManager:
    """管理 QA → Segment → Experience 的分层结构化记忆。"""

    def __init__(
        self,
        storage: MemoryStorage,
        vector_store: ChromaVectorStore,
        embedder: TextEmbedder,
        summarizer: SummarizerProtocol | None = None,
        segment_summary_qa_threshold: int = 5,
        experience_summary_segment_threshold: int = 5,
        experience_similarity_threshold: float = 0.82,
        experience_route_margin: float = 0.05,
        min_segment_qas: int = 2,
        experience_recaller: ExperienceRecaller | None = None,
    ) -> None:
        self.storage = storage
        self.summarizer = summarizer or TemplateSummarizer()
        self.vector_store = vector_store
        self.embedder = embedder
        self.segment_summary_qa_threshold = segment_summary_qa_threshold
        self.experience_summary_segment_threshold = experience_summary_segment_threshold
        # 语义路由阈值。
        self.experience_similarity_threshold = experience_similarity_threshold
        self.experience_route_margin = max(0.0, float(experience_route_margin))
        self.min_segment_qas = min_segment_qas
        self.experience_recaller = experience_recaller

    def add_qa(
        self,
        topic_result: dict[str, Any],
        user_input: str,
        assistant_output: str = "",
        tools: list[dict[str, Any]] | None = None,
        timestamp: str | None = None,
        state_key: str = "default",
        source_id: str | None = None,
    ) -> dict[str, Any]:
        """把一条结构化主题结果写入三层记忆。"""

        topic = self._required_text(topic_result, "topic")
        core_entity = self._required_text(topic_result, "core_entity")
        intent = self._required_text(topic_result, "intent")
        entities = topic_result.get("entities") or [core_entity]
        confidence = float(topic_result.get("confidence", 0.0))
        reasoning = str(topic_result.get("reasoning") or "")
        timestamp = format_timestamp(timestamp)
        if source_id is not None:
            source_id = str(source_id).strip() or None

        logger.info(
            "写入主题记忆 state=%s topic=%s core_entity=%s intent=%s",
            state_key,
            topic,
            core_entity,
            intent,
        )
        logger.info(
            "Memory add payload state=%s source_id=%s user_input=%s "
            "assistant_output=%s topic_result=%s",
            state_key,
            source_id,
            user_input,
            assistant_output,
            json.dumps(topic_result, ensure_ascii=False),
        )

        try:
            current_experience, current_segment = self.route_experience(
                state_key=state_key,
                topic=topic,
                core_entity=core_entity,
                intent=intent,
                query=user_input,
                timestamp=timestamp,
            )

            # Segment intent 边界只在写入路径判断。父层更新统一交给 outbox。
            action = "append_segment"
            if not current_segment or self._should_cut_segment(current_segment, intent):
                if current_segment:
                    old_qa_count = self.storage.count_qas_by_segment(
                        current_segment["segment_id"]
                    )
                    self._enqueue_job(
                        "update_segment",
                        "segment",
                        current_segment["segment_id"],
                        old_qa_count,
                        timestamp,
                        payload={
                            "desired_status": "completed",
                            "force_summary": True,
                            "force_experience_summary": True,
                        },
                    )
                    action = "new_segment"
                else:
                    action = (
                        "new_experience"
                        if self.storage.count_segments_by_experience(
                            current_experience["experience_id"]
                        ) == 0
                        else "new_segment"
                    )
                current_segment = self._create_segment(
                    current_experience,
                    topic,
                    core_entity,
                    intent,
                    timestamp,
                )
            self.storage.upsert_runtime_state(
                state_key=state_key,
                current_experience_id=current_experience["experience_id"],
                current_segment_id=current_segment["segment_id"],
                updated_at=timestamp,
            )

            qa = {
                "qa_id": self._new_id("qa"),
                "source_id": source_id,
                "timestamp": timestamp,
                "user_input": user_input,
                "assistant_output": assistant_output,
                "tools": tools or [],
                "topic": topic,
                "intent": intent,
                "core_entity": core_entity,
                "entities": [str(entity) for entity in entities if str(entity).strip()],
                "segment_id": current_segment["segment_id"],
                "experience_id": current_experience["experience_id"],
                "status": "open",
                "confidence": confidence,
                "reason": reasoning,
            }
            self.storage.insert_qa(qa)
            self.upsert_qa_vector(qa["qa_id"])
            logger.info(
                "QA 已创建 qa_id=%s segment_id=%s experience_id=%s",
                qa["qa_id"],
                current_segment["segment_id"],
                current_experience["experience_id"],
            )

            qa_count = self.storage.count_qas_by_segment(current_segment["segment_id"])
            self._enqueue_job(
                "update_segment",
                "segment",
                current_segment["segment_id"],
                qa_count,
                timestamp,
                payload={
                    "desired_status": "open",
                    "force_summary": False,
                    "force_experience_summary": False,
                },
            )
            self.storage.commit()
            logger.info(
                "Memory add committed state=%s qa_id=%s segment_id=%s experience_id=%s action=%s",
                state_key,
                qa["qa_id"],
                current_segment["segment_id"],
                current_experience["experience_id"],
                action,
            )
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

    def route_experience(
        self,
        *,
        state_key: str,
        topic: str,
        core_entity: str,
        intent: str,
        query: str,
        timestamp: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """路由 Experience，并返回其最新 Segment（若不存在则为 ``None``）。"""
        timestamp = format_timestamp(timestamp)
        logger.info(
            "Route experience started state_key=%s topic=%s core_entity=%s intent=%s query=%s",
            state_key,
            topic,
            core_entity,
            intent,
            query,
        )
        runtime = self.storage.get_runtime_state(state_key)
        current_experience = self.storage.get_experience(
            runtime["current_experience_id"] if runtime else None
        )
        current_segment = self.storage.get_segment(
            runtime["current_segment_id"] if runtime else None
        )
        logger.info(
            "Route runtime loaded state_key=%s runtime=%s current_experience_id=%s current_segment_id=%s",
            state_key,
            json.dumps(runtime or {}, ensure_ascii=False),
            (current_experience or {}).get("experience_id", ""),
            (current_segment or {}).get("segment_id", ""),
        )

        # runtime 中的 Segment 必须属于当前 Experience，避免复用失效指针。
        if (
            current_experience
            and current_segment
            and current_segment.get("experience_id")
            != current_experience.get("experience_id")
        ):
            logger.info(
                "Route discarded stale segment state_key=%s segment_id=%s experience_id=%s expected_experience_id=%s",
                state_key,
                current_segment.get("segment_id", ""),
                current_segment.get("experience_id", ""),
                current_experience.get("experience_id", ""),
            )
            current_segment = None

        # 当前 Experience 不可复用时，依次尝试关系库、路由向量和新建流程。
        if not self._same_experience(current_experience, topic, core_entity):
            logger.info(
                "Route current experience not reusable state_key=%s current_experience_id=%s",
                state_key,
                (current_experience or {}).get("experience_id", ""),
            )
            if current_segment:
                old_qa_count = self.storage.count_qas_by_segment(
                    current_segment["segment_id"]
                )
                self._enqueue_job(
                    "update_segment",
                    "segment",
                    current_segment["segment_id"],
                    old_qa_count,
                    timestamp,
                    payload={
                        "desired_status": "completed",
                        "force_summary": True,
                        "force_experience_summary": True,
                    },
                )
            elif current_experience:
                old_segment_count = self.storage.count_segments_by_experience(
                    current_experience["experience_id"]
                )
                self._enqueue_job(
                    "update_experience",
                    "experience",
                    current_experience["experience_id"],
                    old_segment_count,
                    timestamp,
                    payload={
                        "desired_status": "open",
                        "force_summary": True,
                    },
                )

            current_experience = self.storage.find_active_experience(
                topic,
                core_entity,
            )
            if current_experience:
                logger.info(
                    "SQLite 命中进行中 Experience id=%s topic=%s entity=%s",
                    current_experience["experience_id"],
                    current_experience["topic"],
                    current_experience["core_entity"],
                )
            else:
                current_experience = self._find_experience_by_vector(
                    topic,
                    core_entity,
                )

            if current_experience:
                current_segment = self.storage.find_latest_segment(
                    current_experience["experience_id"]
                )
            else:
                current_experience = self.create_experience(
                    topic=topic,
                    core_entity=core_entity,
                    query=query,
                    intent=intent,
                    timestamp=timestamp,
                )
                current_segment = None
        else:
            logger.info(
                "Route reused current experience state_key=%s experience_id=%s segment_id=%s",
                state_key,
                current_experience.get("experience_id", ""),
                (current_segment or {}).get("segment_id", ""),
            )


        self.storage.upsert_runtime_state(
            state_key=state_key,
            current_experience_id=current_experience["experience_id"],
            current_segment_id=current_segment["segment_id"] if current_segment else "",
            updated_at=timestamp,
        )
        logger.info(
            "Route experience completed state_key=%s experience_id=%s segment_id=%s",
            state_key,
            current_experience["experience_id"],
            current_segment["segment_id"] if current_segment else "",
        )
        return current_experience, current_segment

    def create_experience(
        self,
        *,
        topic: str,
        core_entity: str,
        query: str,
        intent: str = "",
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        """Create the minimal Experience and enqueue its one-time derivations."""
        topic = str(topic or "").strip()
        core_entity = str(core_entity or "").strip()
        query = str(query or "").strip()
        if not topic or not core_entity or not query:
            raise ValueError("topic, core_entity and query must not be empty")

        now = format_timestamp(timestamp)
        experience = {
            "experience_id": self._new_id("exp"),
            "topic": topic,
            "core_entity": core_entity,
            "intents_link": [],
            "segment_ids": [],
            "summary": {},
            "status": "open",
            "created_at": now,
            "updated_at": now,
            "version": 1,
            "last_summarized_segment_count": 0,
            "history_experience": {},
        }
        self.storage.insert_experience(experience)
        self._enqueue_job(
            "recall_experience_history",
            "experience",
            experience["experience_id"],
            experience["version"],
            now,
            payload={
                "topic": topic,
                "core_entity": core_entity,
                "query": query,
                "intent": intent,
            },
        )
        self._enqueue_job(
            "create_experience_route_vector",
            "experience",
            experience["experience_id"],
            experience["version"],
            now,
            payload={
                "topic": topic,
                "core_entity": core_entity,
                "status": "open",
                "created_at": now,
                "updated_at": now,
            },
        )
        logger.info(
            "新建 Experience experience_id=%s topic=%s entity=%s",
            experience["experience_id"],
            topic,
            core_entity,
        )
        return experience

    def _mark_experience_completed(
        self, experience: dict[str, Any], now: str
    ) -> None:
        self._enqueue_job(
            "update_experience",
            "experience",
            experience["experience_id"],
            self.storage.count_segments_by_experience(experience["experience_id"]),
            now,
            payload={"desired_status": "completed", "force_summary": True},
        )

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
            "summary": {},
            "experience_id": experience["experience_id"],
            "created_at": now,
            "updated_at": now,
            "version": 1,
            "last_summarized_qa_count": 0,
        }
        self.storage.insert_segment(segment)
        logger.info(
            "新建 Segment segment_id=%s experience_id=%s intent=%s",
            segment["segment_id"],
            experience["experience_id"],
            intent,
        )
        return segment

    def upsert_experience_content_vector(self, experience_id: str) -> None:
        """Update the Experience content vector only."""
        experience = self.storage.get_experience(experience_id)
        if not experience:
            raise ValueError(f"Unknown experience_id: {experience_id}")
        recent_segments = self.storage.list_segments_by_experience_ids([experience_id])[:3]
        vector_memory = {**experience, "recent_segments": recent_segments}
        text = build_vector_document("experience", vector_memory)
        normal_embedding = self.embedder.embed(text)
        metadata = build_vector_metadata("experience", experience)
        self.vector_store.upsert(
            memory_type="experience",
            memory_id=experience_id,
            text=text,
            embedding=normal_embedding,
            updated_at=experience["updated_at"],
            metadata=metadata,
        )

    def create_experience_route_vector(
        self,
        experience_id: str,
        creation_snapshot: dict[str, Any],
    ) -> None:
        """Create an idempotent route vector from immutable creation fields."""
        experience = {
            "experience_id": experience_id,
            "topic": str(creation_snapshot.get("topic") or ""),
            "core_entity": str(creation_snapshot.get("core_entity") or ""),
            "status": "open",
            "created_at": creation_snapshot.get("created_at"),
            "updated_at": creation_snapshot.get("updated_at")
            or creation_snapshot.get("created_at"),
        }
        if not experience["topic"] or not experience["core_entity"]:
            raise ValueError("Experience route creation snapshot is incomplete")
        route_text = build_vector_document("experience_route", experience)
        route_embedding = self.embedder.embed(route_text)
        metadata = build_vector_metadata("experience_route", experience)
        self.vector_store.upsert(
            memory_type="experience_route",
            memory_id=experience_id,
            text=route_text,
            embedding=route_embedding,
            updated_at=experience["updated_at"],
            metadata=metadata,
        )
        logger.debug("Experience 路由向量已创建 id=%s", experience_id)

    def upsert_experience_vector(self, experience_id: str) -> None:
        """Compatibility helper for the mutable Experience content vector."""
        self.upsert_experience_content_vector(experience_id)

    def _enqueue_job(
        self,
        job_type: str,
        memory_type: str,
        memory_id: str,
        target_version: int,
        timestamp: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.storage.enqueue_memory_job(
            job_id=self._new_id("job"),
            job_type=job_type,
            memory_type=memory_type,
            memory_id=memory_id,
            target_version=target_version,
            timestamp=timestamp,
            payload=payload,
        )

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
        text = build_vector_document("segment", vector_memory)
        self.vector_store.upsert(
            memory_type="segment",
            memory_id=segment_id,
            text=text,
            embedding=self.embedder.embed(text),
            updated_at=segment["updated_at"],
            metadata=build_vector_metadata("segment", segment),
        )
        logger.debug("Segment 向量已写入 id=%s", segment_id)

    def upsert_qa_vector(self, qa_id: str) -> None:
        qa = self.storage.get_qa(qa_id)
        if not qa:
            raise ValueError(f"Unknown qa_id: {qa_id}")
        segment = self.storage.get_segment(qa.get("segment_id"))
        vector_memory = {
            **qa,
            "experience_id": (segment or {}).get("experience_id", ""),
        }
        text = build_vector_document("qa", vector_memory)
        self.vector_store.upsert(
            memory_type="qa",
            memory_id=qa_id,
            text=text,
            embedding=self.embedder.embed(text),
            updated_at=qa["timestamp"],
            metadata=build_vector_metadata("qa", vector_memory),
        )
        logger.debug("QA 向量已写入 id=%s", qa_id)

    def _same_experience(
        self,
        experience: dict[str, Any] | None,
        topic: str,
        core_entity: str,
    ) -> bool:
        """仅在主题、核心实体和开放状态均匹配时返回真。"""
        return bool(
            experience
            and experience["topic"] == topic
            and experience["core_entity"] == core_entity
            and experience.get("status") == "open"
        )

    def _find_experience_by_vector(
        self, topic: str, core_entity: str
    ) -> dict[str, Any] | None:
        """Route through the dedicated topic+core_entity Experience vectors."""
        query_text = f"主题：{topic}\n核心实体：{core_entity}"
        try:
            query_embedding = self.embedder.embed(query_text)
        except Exception:
            logger.warning("Experience 路由向量嵌入失败", exc_info=True)
            return None
        ## TODO 添加筛选条件状态为open
        results = self.vector_store.query(
            query_embedding,
            memory_type="experience_route",
            top_k=5,
        )
        candidates: list[tuple[dict[str, Any], float]] = []
        for item in results:
            if item["similarity"] <= self.experience_similarity_threshold:
                break  # results are sorted by similarity desc; no point checking rest
            metadata = item.get("metadata") or {}
            exp_id = str(
                metadata.get("experience_id")
                or metadata.get("memory_id")
                or ""
            )
            if not exp_id:
                continue
            experience = self.storage.get_experience(exp_id)
            if (
                experience
                and experience.get("status") == "open"
            ):
                candidates.append((experience, float(item["similarity"])))
        if not candidates:
            return None
        if (
            len(candidates) > 1
            and candidates[0][1] - candidates[1][1] < self.experience_route_margin
        ):
            logger.info(
                "Experience 路由向量候选不明确 first=%.3f second=%.3f margin=%.3f",
                candidates[0][1],
                candidates[1][1],
                self.experience_route_margin,
            )
            return None
        experience, similarity = candidates[0]
        logger.info(
            "Experience 路由向量命中 id=%s sim=%.3f topic=%s entity=%s",
            experience["experience_id"],
            similarity,
            experience["topic"],
            experience["core_entity"],
        )
        return experience

    def _should_cut_segment(self, segment: dict[str, Any], intent: str) -> bool:
        """判断是否应该切断当前 Segment，开启新 Segment。

        三个条件必须同时满足才切：
        1. intent 字符串不完全相同
        2. 当前 intent 与 segment intent 的 token 余弦相似度 < 0.8
           （措辞不同但语义相近时保留，避免碎片化）
        3. 当前 Segment 已积累了足够的 QA（>= min_segment_qas）
        """
        if segment.get("status") != "open":
            logger.info(
                "Segment cut because status is not open segment_id=%s status=%s",
                segment.get("segment_id", ""),
                segment.get("status", ""),
            )
            return True

        if segment["intent"] == intent:
            logger.info(
                "Segment retained because intent is unchanged segment_id=%s intent=%s",
                segment.get("segment_id", ""),
                intent,
            )
            return False

        similarity = self._intent_similarity(segment["intent"], intent)
        if similarity >= 0.8:
            logger.info(
                "Segment retained because intent similarity is high segment_id=%s old_intent=%s new_intent=%s similarity=%.3f",
                segment.get("segment_id", ""),
                segment["intent"],
                intent,
                similarity,
            )
            return False

        qa_count = self.storage.count_qas_by_segment(segment["segment_id"])
        if qa_count < self.min_segment_qas:
            logger.info(
                "Segment retained because QA count is below threshold segment_id=%s qa_count=%s min_segment_qas=%s similarity=%.3f",
                segment.get("segment_id", ""),
                qa_count,
                self.min_segment_qas,
                similarity,
            )
            return False

        logger.info(
            "Segment cut because intent changed segment_id=%s old_intent=%s new_intent=%s qa_count=%s similarity=%.3f",
            segment.get("segment_id", ""),
            segment["intent"],
            intent,
            qa_count,
            similarity,
        )
        return True

    @staticmethod
    def _intent_similarity(a: str, b: str) -> float:
        """本地 token bag-of-words 余弦相似度，无 API 调用。"""
        from .embedder import tokenize
        import math

        tokens_a = tokenize(a)
        tokens_b = tokenize(b)
        if not tokens_a or not tokens_b:
            return 0.0

        freq_a: dict[str, int] = {}
        freq_b: dict[str, int] = {}
        for t in tokens_a:
            freq_a[t] = freq_a.get(t, 0) + 1
        for t in tokens_b:
            freq_b[t] = freq_b.get(t, 0) + 1

        vocab = set(freq_a) | set(freq_b)
        dot  = sum(freq_a.get(t, 0) * freq_b.get(t, 0) for t in vocab)
        norm_a = math.sqrt(sum(v * v for v in freq_a.values()))
        norm_b = math.sqrt(sum(v * v for v in freq_b.values()))
        if not norm_a or not norm_b:
            return 0.0
        return dot / (norm_a * norm_b)

    def _required_text(self, value: dict[str, Any], key: str) -> str:
        text = str(value.get(key) or "").strip()
        if not text:
            raise ValueError(f"{key} is required")
        return text

    @staticmethod
    def _summary_object(value: Any) -> dict[str, Any]:
        """把总结器响应规范化为结构化摘要对象。"""
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    def _new_id(self, prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:12]}"

    def _now(self) -> str:
        return format_timestamp()
