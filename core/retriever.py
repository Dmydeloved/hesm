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
    """Compatibility formatter for an Experience's stored historical context."""
    if isinstance(history_experience, (dict, list)):
        text = json.dumps(history_experience, ensure_ascii=False)
    else:
        text = str(history_experience or "").strip()
    if not text or text in {"{}", "[]"}:
        text = "未召回可复用的历史经验。"
    return f"【历史经验】\n{text}"


class HybridRetriever:
    """只读检索已有记忆；检索阶段绝不创建或更新 Experience。"""

    def __init__(
        self,
        manager: MemoryManager,
        *,
        segment_limit: int = 2,
        qa_limit: int = 5,
        experience_route_margin: float = 0.05,
        experience_qa_hybrid_enabled: bool = False,
        qa_rrf_constant: int = 60,
        qa_dense_candidate_limit: int = 30,
        qa_bm25_candidate_limit: int = 30,
    ) -> None:
        self.manager = manager
        self.storage = manager.storage
        self.segment_limit = max(1, int(segment_limit))
        self.qa_limit = max(1, int(qa_limit))
        self.experience_route_margin = max(0.0, float(experience_route_margin))
        self.experience_qa_hybrid_enabled = bool(experience_qa_hybrid_enabled)
        self.qa_rrf_constant = max(1, int(qa_rrf_constant))
        self.qa_dense_candidate_limit = max(self.qa_limit, int(qa_dense_candidate_limit))
        self.qa_bm25_candidate_limit = max(self.qa_limit, int(qa_bm25_candidate_limit))

    def _retrieve_experience_qas(self, experience: dict[str, Any], query: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Fuse experience-scoped dense and FTS5 QA rankings with reciprocal rank fusion."""
        experience_id = str(experience.get("experience_id") or "")
        ranked_ids: dict[str, dict[str, Any]] = {}
        try:
            embedding = self.manager.embedder.embed(query)
            dense = self.manager.vector_store.query(
                embedding,
                memory_type="qa",
                top_k=self.qa_dense_candidate_limit,
                metadata_filter={"experience_id": experience_id},
            )
        except Exception:
            logger.warning("Experience-scoped QA vector retrieval failed", exc_info=True)
            dense = []
        for rank, item in enumerate(dense, 1):
            metadata = item.get("metadata") or {}
            qa_id = str(metadata.get("qa_id") or metadata.get("memory_id") or "")
            if not qa_id:
                continue
            ranked_ids.setdefault(qa_id, {"dense_rank": None, "bm25_rank": None})
            ranked_ids[qa_id]["dense_rank"] = rank
            ranked_ids[qa_id]["dense_similarity"] = float(item.get("similarity") or 0.0)

        try:
            bm25 = self.storage.search_qas_bm25(
                experience_id=experience_id,
                query=query,
                limit=self.qa_bm25_candidate_limit,
            )
        except Exception:
            logger.warning("Experience-scoped QA BM25 retrieval failed", exc_info=True)
            bm25 = []
        for rank, qa in enumerate(bm25, 1):
            qa_id = str(qa.get("qa_id") or "")
            if not qa_id:
                continue
            ranked_ids.setdefault(qa_id, {"dense_rank": None, "bm25_rank": None})
            ranked_ids[qa_id]["bm25_rank"] = rank
            ranked_ids[qa_id]["bm25_score"] = float(qa.get("bm25_score") or 0.0)

        scores: dict[str, float] = {}
        for qa_id, item in ranked_ids.items():
            score = 0.0
            for rank_key in ("dense_rank", "bm25_rank"):
                rank = item.get(rank_key)
                if rank is not None:
                    score += 1.0 / (self.qa_rrf_constant + int(rank))
            item["rrf_score"] = score
            scores[qa_id] = score
        ordered_ids = sorted(
            scores,
            key=lambda qa_id: (scores[qa_id], str(ranked_ids[qa_id].get("dense_rank") or 10**9), qa_id),
            reverse=True,
        )
        qas_by_id = {
            str(qa.get("qa_id") or ""): qa
            for qa in self.storage.get_qas(ordered_ids)
        }
        valid_ids = []
        for qa_id in ordered_ids:
            qa = qas_by_id.get(qa_id)
            if not qa or qa.get("status") != "open":
                continue
            segment = self.storage.get_segment(str(qa.get("segment_id") or ""))
            if (
                segment
                and segment.get("status") != "deleted"
                and str(segment.get("experience_id") or "") == experience_id
            ):
                valid_ids.append(qa_id)
        ordered_ids = valid_ids[: self.qa_limit]
        qas = [qas_by_id[qa_id] for qa_id in ordered_ids]
        matches = [
            {
                "qa_id": qa_id,
                "experience_id": experience_id,
                "dense_rank": ranked_ids[qa_id].get("dense_rank"),
                "bm25_rank": ranked_ids[qa_id].get("bm25_rank"),
                "dense_similarity": round(float(ranked_ids[qa_id].get("dense_similarity") or 0.0), 6),
                "bm25_score": ranked_ids[qa_id].get("bm25_score"),
                "rrf_score": round(scores[qa_id], 8),
                "channels": [
                    name for name, key in (("vector", "dense_rank"), ("bm25", "bm25_rank"))
                    if ranked_ids[qa_id].get(key) is not None
                ],
            }
            for qa_id in ordered_ids
        ]
        return qas, matches

    def _route_existing(
        self,
        *,
        state_key: str,
        topic: str,
        core_entity: str,
        query: str = "",
    ) -> tuple[dict[str, Any] | None, str, float, str]:
        """按 runtime、SQLite、路由向量的顺序查找已有 Experience。"""
        runtime = self.storage.get_runtime_state(state_key)
        current = self.storage.get_experience(
            runtime.get("current_experience_id") if runtime else None
        )
        if self.manager._same_experience(current, topic, core_entity):
            return current, "runtime", 1.0, ""

        current = self.storage.find_active_experience(topic, core_entity)
        if current:
            return current, "sqlite", 1.0, ""

        candidates = self._experience_vector_candidates(topic, core_entity)
        if not candidates:
            return None, "no_route", 0.0, "no_vector_candidate"

        current, confidence = candidates[0]
        threshold = float(self.manager.experience_similarity_threshold)
        if confidence <= threshold:
            return None, "no_route", confidence, "low_confidence"
        if len(candidates) > 1:
            margin = confidence - candidates[1][1]
            if margin < self.experience_route_margin:
                selector = getattr(self.manager, "_select_ambiguous_by_goal", None)
                if selector is None:
                    current = max(
                        (item[0] for item in candidates[:2]),
                        key=lambda item: (
                            str(item.get("updated_at") or ""),
                            str(item.get("created_at") or ""),
                        ),
                    )
                else:
                    current = selector(candidates[:2], query)
                confidence = next(
                    score for candidate, score in candidates[:2]
                    if candidate.get("experience_id") == current.get("experience_id")
                )
        return current, "vector", confidence, ""

    def _experience_vector_candidates(
        self,
        topic: str,
        core_entity: str,
    ) -> list[tuple[dict[str, Any], float]]:
        query_text = f"主题：{topic}\n核心实体：{core_entity}"
        try:
            embedding = self.manager.embedder.embed(query_text)
            items = self.manager.vector_store.query(
                embedding,
                memory_type="experience_route",
                top_k=5,
                metadata_filter=None,
            )
        except Exception:
            logger.warning("Experience vector routing failed", exc_info=True)
            return []

        candidates: list[tuple[dict[str, Any], float]] = []
        seen: set[str] = set()
        for item in items:
            metadata = item.get("metadata") or {}
            experience_id = str(
                metadata.get("experience_id")
                or metadata.get("memory_id")
                or ""
            )
            if not experience_id or experience_id in seen:
                continue
            experience = self.storage.get_experience(experience_id)
            if not experience or experience.get("status") != "open":
                continue
            seen.add(experience_id)
            candidates.append((experience, float(item.get("similarity") or 0.0)))
        return candidates

    def retriever(
        self,
        topic: str,
        core_entity: str,
        query: str,
        intent: str = "",
        state_key: str = "default",
        entities: list[str] | None = None,
        query_candidates: list[dict[str, Any]] | None = None,
        target_experience_id: str = "",
    ) -> dict[str, Any]:
        """Retrieve within a routed Experience; route misses return empty results."""
        topic = str(topic or "").strip()
        core_entity = str(core_entity or "").strip()
        query = str(query or "").strip()
        intent = str(intent or "").strip() or "查询"
        if not topic or not core_entity or not query:
            raise ValueError("topic, core_entity and query must not be empty")
        route_variants = [(topic, core_entity)]
        for candidate in query_candidates or []:
            key = (
                str(candidate.get("topic") or "").strip(),
                str(candidate.get("core_entity") or "").strip(),
            )
            if key[0] and key[1] and key not in route_variants:
                route_variants.append(key)
        experience = None
        route_source = "no_route"
        route_confidence = 0.0
        fallback_reason = "no_vector_candidate"
        if target_experience_id:
            targeted = self.storage.get_experience(str(target_experience_id))
            if targeted and targeted.get("status") != "deleted":
                experience = targeted
                route_source = "evidence_experience"
                route_confidence = 1.0
                fallback_reason = ""
        for route_topic, route_entity in (() if experience else route_variants):
            routed, source, confidence, reason = self._route_existing(
                state_key=state_key,
                topic=route_topic,
                core_entity=route_entity,
                query=query,
            )
            if confidence > route_confidence:
                route_confidence = confidence
                fallback_reason = reason
            if routed is not None:
                experience = routed
                route_source = source
                topic = route_topic
                core_entity = route_entity
                break
        if experience is None:
            return {
                "route_status": route_source,
                "route_confidence": route_confidence,
                "fallback_reason": fallback_reason,
                "experiences": [],
                "segments": [],
                "qas": [],
                "history_experience": {},
                "history_recall": {},
                "context": "",
            }

        if hasattr(self.storage, "list_unupdated_segments"):
            latest_segments = self.storage.list_unupdated_segments(experience)
        else:
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

        if segments and hasattr(self.storage, "list_unupdated_qas"):
            latest_qas = []
            seen_qa_ids: set[str] = set()
            for segment in segments:
                for qa in self.storage.list_unupdated_qas(segment):
                    qa_id = str(qa.get("qa_id") or "")
                    if qa_id and qa_id not in seen_qa_ids:
                        seen_qa_ids.add(qa_id)
                        latest_qas.append(qa)
        else:
            segment_ids = [str(segment["segment_id"]) for segment in segments]
            latest_qas = self.storage.list_latest_qas(segment_ids, self.qa_limit)
        qas = sorted(
            latest_qas,
            key=lambda item: (
                str(item.get("timestamp") or ""),
                str(item.get("qa_id") or ""),
            ),
        )

        qa_matches: list[dict[str, Any]] = []
        retrieval_channels: list[str] = []
        if self.experience_qa_hybrid_enabled:
            qas, qa_matches = self._retrieve_experience_qas(experience, query)
            segment_map: dict[str, dict[str, Any]] = {}
            for qa in qas:
                segment_id = str(qa.get("segment_id") or "")
                if segment_id and segment_id not in segment_map:
                    segment = self.storage.get_segment(segment_id)
                    if segment:
                        segment_map[segment_id] = segment
            segments = list(segment_map.values())
            retrieval_channels = sorted(
                {channel for match in qa_matches for channel in match["channels"]}
            )

        experiences = [experience]
        return {
            "route_status": route_source,
            "route_confidence": route_confidence,
            "fallback_reason": "",
            "experiences": experiences,
            "segments": segments,
            "qas": qas,
            **({"qa_matches": qa_matches, "retrieval_channels": retrieval_channels} if self.experience_qa_hybrid_enabled else {}),
            "history_experience": experience.get("history_experience") or {},
            "context": build_context_text(experiences, segments, qas),
        }


class ReadOnlyHybridRetriever(HybridRetriever):
    """兼容旧调用名；所有 HybridRetriever 现在都保证只读。"""


__all__ = [
    "HybridRetriever",
    "ReadOnlyHybridRetriever",
    "build_context_text",
    "build_history_context",
]
