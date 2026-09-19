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


def build_qa_fallback_context(
    experiences: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    qas: list[dict[str, Any]],
) -> str:
    """Build evidence-first context when QA recall reconstructs the hierarchy."""
    experience_by_id = {
        str(item.get("experience_id") or ""): item for item in experiences
    }
    segment_by_id = {str(item.get("segment_id") or ""): item for item in segments}
    lines = ["【QA 托底检索】"]
    for index, qa in enumerate(qas, 1):
        segment = segment_by_id.get(str(qa.get("segment_id") or ""), {})
        experience = experience_by_id.get(
            str(segment.get("experience_id") or ""), {}
        )
        lines.extend(
            [
                f"{index}. 用户：{qa.get('user_input', '')}",
                f"   助手：{qa.get('assistant_output', '')}",
                f"   主题：{qa.get('topic', '')}",
                f"   核心实体：{qa.get('core_entity', '')}",
                f"   所属阶段：{segment.get('intent', '')}",
                f"   阶段摘要：{_memory_text(segment.get('summary'))}",
                f"   长期经验：{_memory_text(experience.get('summary'))}",
                f"   时间：{qa.get('timestamp', '')}",
            ]
        )
    return "\n".join(lines)


class HybridRetriever:
    """只读检索已有记忆；检索阶段绝不创建或更新 Experience。"""

    def __init__(
        self,
        manager: MemoryManager,
        *,
        segment_limit: int = 2,
        qa_limit: int = 5,
        qa_candidate_limit: int = 40,
        qa_similarity_threshold: float = 0.45,
        experience_route_margin: float = 0.05,
    ) -> None:
        self.manager = manager
        self.storage = manager.storage
        self.segment_limit = max(1, int(segment_limit))
        self.qa_limit = max(1, int(qa_limit))
        self.qa_candidate_limit = max(self.qa_limit, int(qa_candidate_limit))
        self.qa_similarity_threshold = float(qa_similarity_threshold)
        self.experience_route_margin = max(0.0, float(experience_route_margin))

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
            return None, "qa_fallback", 0.0, "no_vector_candidate"

        current, confidence = candidates[0]
        threshold = float(self.manager.experience_similarity_threshold)
        if confidence <= threshold:
            return None, "qa_fallback", confidence, "low_confidence"
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

    def _qa_fallback(
        self,
        *,
        topic: str,
        core_entity: str,
        intent: str,
        query: str,
        entities: list[str] | None = None,
        query_candidates: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        candidate_map: dict[str, dict[str, Any]] = {}
        segment_scores: dict[str, float] = {}
        experience_scores: dict[str, float] = {}
        segment_cache: dict[str, dict[str, Any]] = {}
        experience_cache: dict[str, dict[str, Any]] = {}
        query_entities = list(
            dict.fromkeys(
                value
                for value in [
                    core_entity,
                    *(entities or []),
                    *[
                        str(candidate.get("core_entity") or "")
                        for candidate in (query_candidates or [])
                    ],
                    *[
                        str(entity)
                        for candidate in (query_candidates or [])
                        for entity in candidate.get("entities") or []
                    ],
                ]
                if str(value or "").strip()
            )
        )
        structured_queries = [
            {
                "topic": topic,
                "core_entity": core_entity,
                "intent": intent,
                "entities": query_entities,
            }
        ]
        seen_structured = {(topic, core_entity, intent)}
        for candidate in query_candidates or []:
            variant = {
                "topic": str(candidate.get("topic") or "").strip(),
                "core_entity": str(candidate.get("core_entity") or "").strip(),
                "intent": str(candidate.get("intent") or "").strip(),
                "entities": [
                    str(value)
                    for value in candidate.get("entities") or []
                    if str(value).strip()
                ],
            }
            key = (variant["topic"], variant["core_entity"], variant["intent"])
            if variant["topic"] and variant["core_entity"] and key not in seen_structured:
                seen_structured.add(key)
                structured_queries.append(variant)

        def add_ranked(
            rows: list[tuple[dict[str, Any], float]],
            channel: str,
            *,
            dense: bool = False,
        ) -> None:
            for rank, (qa, similarity) in enumerate(rows, 1):
                qa_id = str(qa.get("qa_id") or "")
                if not qa_id:
                    continue
                candidate = candidate_map.setdefault(
                    qa_id,
                    {
                        "qa": qa,
                        "rrf_score": 0.0,
                        "dense_similarity": 0.0,
                        "parent_similarity": 0.0,
                        "channels": [],
                    },
                )
                candidate["rrf_score"] += 1.0 / (60 + rank)
                if dense:
                    candidate["dense_similarity"] = max(
                        float(candidate["dense_similarity"]), similarity
                    )
                else:
                    candidate["parent_similarity"] = max(
                        float(candidate["parent_similarity"]), similarity
                    )
                if channel not in candidate["channels"]:
                    candidate["channels"].append(channel)

        dense_rows: list[tuple[dict[str, Any], float]] = []
        embedding: list[float] | None = None
        query_document = (
            f"主题：{topic}\n核心实体：{core_entity}\n"
            f"意图：{intent}\n实体：{'、'.join(query_entities)}\n用户问题：{query}"
        )
        try:
            embedding = self.manager.embedder.embed(query_document)
            vector_items = self.manager.vector_store.query(
                embedding,
                memory_type="qa",
                top_k=self.qa_candidate_limit,
                metadata_filter={"status": "open"},
            )
            qa_ids = [
                str((item.get("metadata") or {}).get("qa_id")
                    or (item.get("metadata") or {}).get("memory_id")
                    or "")
                for item in vector_items
            ]
            qa_rows = (
                self.storage.get_qas(qa_ids)
                if hasattr(self.storage, "get_qas")
                else [self.storage.get_qa(qa_id) for qa_id in qa_ids]
            )
            qa_items_by_id = {
                str(qa.get("qa_id") or ""): qa
                for qa in qa_rows
                if qa
            }
            for item in vector_items:
                metadata = item.get("metadata") or {}
                qa_id = str(metadata.get("qa_id") or metadata.get("memory_id") or "")
                qa = qa_items_by_id.get(qa_id)
                if qa and qa.get("status") == "open":
                    dense_rows.append((qa, float(item.get("similarity") or 0.0)))
        except Exception:
            logger.warning("QA vector fallback failed", exc_info=True)
        add_ranked(dense_rows, "dense", dense=True)

        # Parent summaries are navigation indexes rather than answer evidence.
        # Their vectors rerank QA candidates, while Segment provenance can add
        # the exact source QAs referenced by key facts/state changes.
        summary_source_ids: list[str] = []
        try:
            if embedding is None:
                raise RuntimeError("query embedding is unavailable")
            segment_items = self.manager.vector_store.query(
                embedding,
                memory_type="segment",
                top_k=min(20, self.qa_candidate_limit),
                metadata_filter=None,
            )
            parent_segment_ids = [
                str(
                    (item.get("metadata") or {}).get("segment_id")
                    or (item.get("metadata") or {}).get("memory_id")
                    or ""
                )
                for item in segment_items
            ]
            parent_segments = (
                self.storage.get_segments(parent_segment_ids)
                if hasattr(self.storage, "get_segments")
                else [self.storage.get_segment(value) for value in parent_segment_ids]
            )
            segment_cache.update(
                {
                    str(segment.get("segment_id") or ""): segment
                    for segment in parent_segments
                    if segment
                }
            )
            for item in segment_items:
                metadata = item.get("metadata") or {}
                segment_id = str(
                    metadata.get("segment_id") or metadata.get("memory_id") or ""
                )
                segment = segment_cache.get(segment_id)
                if not segment or segment.get("status") == "deleted":
                    continue
                similarity = float(item.get("similarity") or 0.0)
                segment_scores[segment_id] = max(
                    segment_scores.get(segment_id, 0.0),
                    similarity,
                )
                summary = segment.get("summary") or {}
                if isinstance(summary, dict):
                    for field in ("key_facts", "state_changes"):
                        for fact in summary.get(field) or []:
                            if isinstance(fact, dict):
                                summary_source_ids.extend(
                                    str(value)
                                    for value in fact.get("source_qa_ids") or []
                                    if str(value).strip()
                                )

            experience_items = self.manager.vector_store.query(
                embedding,
                memory_type="experience",
                top_k=min(10, self.qa_candidate_limit),
                metadata_filter=None,
            )
            parent_experience_ids = [
                str(
                    (item.get("metadata") or {}).get("experience_id")
                    or (item.get("metadata") or {}).get("memory_id")
                    or ""
                )
                for item in experience_items
            ]
            parent_experiences = (
                self.storage.get_experiences(parent_experience_ids)
                if hasattr(self.storage, "get_experiences")
                else [
                    self.storage.get_experience(value)
                    for value in parent_experience_ids
                ]
            )
            experience_cache.update(
                {
                    str(experience.get("experience_id") or ""): experience
                    for experience in parent_experiences
                    if experience
                }
            )
            for item in experience_items:
                metadata = item.get("metadata") or {}
                experience_id = str(
                    metadata.get("experience_id")
                    or metadata.get("memory_id")
                    or ""
                )
                experience = experience_cache.get(experience_id)
                if not experience or experience.get("status") == "deleted":
                    continue
                experience_scores[experience_id] = max(
                    experience_scores.get(experience_id, 0.0),
                    float(item.get("similarity") or 0.0),
                )
        except Exception:
            logger.warning("Parent-memory fallback navigation failed", exc_info=True)

        if summary_source_ids:
            source_ids = list(dict.fromkeys(summary_source_ids))
            source_qas = (
                self.storage.get_qas(source_ids)
                if hasattr(self.storage, "get_qas")
                else [self.storage.get_qa(qa_id) for qa_id in source_ids]
            )
            add_ranked(
                [
                    (
                        qa,
                        segment_scores.get(str(qa.get("segment_id") or ""), 0.0),
                    )
                    for qa in source_qas
                    if qa.get("status") == "open"
                ],
                "summary",
            )

        try:
            keyword_by_id: dict[str, dict[str, Any]] = {}
            for structured in structured_queries:
                for qa in self.storage.search_qas(
                    topic=structured["topic"],
                    core_entity=structured["core_entity"],
                    entities=structured["entities"],
                    intent=structured["intent"],
                    limit=self.qa_candidate_limit,
                ):
                    qa_id = str(qa.get("qa_id") or "")
                    previous = keyword_by_id.get(qa_id)
                    if previous is None or float(qa.get("keyword_score") or 0.0) > float(
                        previous.get("keyword_score") or 0.0
                    ):
                        keyword_by_id[qa_id] = qa
            keyword_qas = sorted(
                keyword_by_id.values(),
                key=lambda qa: (
                    float(qa.get("keyword_score") or 0.0),
                    str(qa.get("timestamp") or ""),
                ),
                reverse=True,
            )[: self.qa_candidate_limit]
        except Exception:
            logger.warning("QA SQLite keyword fallback failed", exc_info=True)
            keyword_qas = []
        add_ranked([(qa, 0.0) for qa in keyword_qas], "keyword")

        candidate_segment_ids = list(
            dict.fromkeys(
                str(candidate["qa"].get("segment_id") or "")
                for candidate in candidate_map.values()
                if str(candidate["qa"].get("segment_id") or "")
                not in segment_cache
            )
        )
        if candidate_segment_ids:
            rows = (
                self.storage.get_segments(candidate_segment_ids)
                if hasattr(self.storage, "get_segments")
                else [self.storage.get_segment(value) for value in candidate_segment_ids]
            )
            segment_cache.update(
                {
                    str(segment.get("segment_id") or ""): segment
                    for segment in rows
                    if segment
                }
            )
        candidate_experience_ids = list(
            dict.fromkeys(
                str(segment.get("experience_id") or "")
                for segment in segment_cache.values()
                if str(segment.get("experience_id") or "")
                and str(segment.get("experience_id") or "")
                not in experience_cache
            )
        )
        if candidate_experience_ids:
            rows = (
                self.storage.get_experiences(candidate_experience_ids)
                if hasattr(self.storage, "get_experiences")
                else [
                    self.storage.get_experience(value)
                    for value in candidate_experience_ids
                ]
            )
            experience_cache.update(
                {
                    str(experience.get("experience_id") or ""): experience
                    for experience in rows
                    if experience
                }
            )

        ranked = []
        for candidate in candidate_map.values():
            qa = candidate["qa"]
            channels = candidate["channels"]
            dense_similarity = float(candidate["dense_similarity"])
            keyword_score = float(qa.get("keyword_score") or 0.0)
            strong_structured_match = "keyword" in channels and keyword_score >= 6.0
            if (
                not strong_structured_match
                and "summary" not in channels
                and dense_similarity < self.qa_similarity_threshold
            ):
                continue
            feature_score = float(candidate["rrf_score"])
            if str(qa.get("topic") or "") == topic:
                feature_score += 0.01
            if str(qa.get("core_entity") or "") == core_entity:
                feature_score += 0.015
            if str(qa.get("intent") or "") == intent:
                feature_score += 0.005
            feature_score += 0.003 * keyword_score
            segment = segment_cache.get(str(qa.get("segment_id") or ""))
            experience = (
                experience_cache.get(str(segment.get("experience_id") or ""))
                if segment
                else None
            )
            if segment and segment.get("status") != "deleted":
                feature_score += 0.01 * segment_scores.get(
                    str(segment.get("segment_id") or ""),
                    0.0,
                )
            if experience and experience.get("status") != "deleted":
                feature_score += 0.008 * experience_scores.get(
                    str(experience.get("experience_id") or ""),
                    0.0,
                )
            candidate["parent_similarity"] = max(
                float(candidate["parent_similarity"]),
                segment_scores.get(str(qa.get("segment_id") or ""), 0.0),
                experience_scores.get(
                    str((experience or {}).get("experience_id") or ""),
                    0.0,
                ),
            )
            candidate["segment"] = segment
            candidate["experience"] = experience
            candidate["score"] = feature_score
            ranked.append(candidate)
        ranked.sort(
            key=lambda item: (
                float(item["score"]),
                float(item["dense_similarity"]),
                str(item["qa"].get("timestamp") or ""),
            ),
            reverse=True,
        )

        selected_qas: list[dict[str, Any]] = []
        selected_segments: list[dict[str, Any]] = []
        selected_experiences: list[dict[str, Any]] = []
        segment_ids: set[str] = set()
        experience_ids: set[str] = set()
        matches: list[dict[str, Any]] = []
        effective_segment_limit = min(
            4,
            max(self.segment_limit, len(query_entities)),
        )
        for candidate in ranked:
            qa = candidate["qa"]
            segment = candidate.get("segment")
            if not segment or segment.get("status") == "deleted":
                continue
            experience = candidate.get("experience")
            if not experience or experience.get("status") == "deleted":
                continue
            segment_id = str(segment["segment_id"])
            experience_id = str(experience["experience_id"])
            if segment_id not in segment_ids and len(selected_segments) >= effective_segment_limit:
                continue
            selected_qas.append(qa)
            if segment_id not in segment_ids and len(selected_segments) < effective_segment_limit:
                segment_ids.add(segment_id)
                selected_segments.append(segment)
            if experience_id not in experience_ids:
                experience_ids.add(experience_id)
                selected_experiences.append(experience)
            matches.append(
                {
                    "qa_id": qa["qa_id"],
                    "segment_id": segment_id,
                    "experience_id": experience_id,
                    "score": round(float(candidate["score"]), 6),
                    "dense_similarity": round(float(candidate["dense_similarity"]), 6),
                    "parent_similarity": round(
                        float(candidate["parent_similarity"]),
                        6,
                    ),
                    "channels": list(candidate["channels"]),
                }
            )
            if len(selected_qas) >= self.qa_limit:
                break

        if not selected_qas:
            return None
        return {
            "route_status": "qa_fallback",
            "experiences": selected_experiences,
            "segments": selected_segments,
            "qas": selected_qas,
            "qa_matches": matches,
            "retrieval_channels": sorted(
                {channel for match in matches for channel in match["channels"]}
            ),
            "context": build_qa_fallback_context(
                selected_experiences,
                selected_segments,
                selected_qas,
            ),
        }

    def retriever(
        self,
        topic: str,
        core_entity: str,
        query: str,
        intent: str = "",
        state_key: str = "default",
        entities: list[str] | None = None,
        query_candidates: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Retrieve existing memory; route misses go directly to QA hybrid search."""
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
        route_source = "qa_fallback"
        route_confidence = 0.0
        fallback_reason = "no_vector_candidate"
        for route_topic, route_entity in route_variants:
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
            qa_fallback = self._qa_fallback(
                topic=topic,
                core_entity=core_entity,
                intent=intent,
                query=query,
                entities=entities,
                query_candidates=query_candidates,
            )
            if qa_fallback:
                return {
                    **qa_fallback,
                    "route_confidence": route_confidence,
                    "fallback_reason": fallback_reason,
                    "history_experience": {},
                    "history_recall": {},
                }
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

        latest_segment = segments[-1] if segments else None
        if latest_segment and hasattr(self.storage, "list_unupdated_qas"):
            latest_qas = self.storage.list_unupdated_qas(latest_segment)
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

        experiences = [experience]
        return {
            "route_status": route_source,
            "route_confidence": route_confidence,
            "fallback_reason": "",
            "experiences": experiences,
            "segments": segments,
            "qas": qas,
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
    "build_qa_fallback_context",
]
