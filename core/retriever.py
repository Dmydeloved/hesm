"""并发召回 QA 候选，再在候选池内检索并组装上下文。"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .manager import MemoryManager


logger = logging.getLogger(__name__)
_KEYWORD_PARTS = re.compile(r"[\u3400-\u9fff]+|[^\W\u3400-\u9fff]+")
_KEYWORD_FIELDS = ("user_input", "assistant_output", "topic", "core_entity", "intent", "entities")


def _memory_text(value: Any) -> str:
    """把结构化记忆转换为可读的上下文文本。"""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value or "")


def build_context_text(
    experiences: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    qas: list[dict[str, Any]],
) -> str:
    """用最终入选的 QA 及其父级摘要构建上下文。"""
    lines = ["【检索到的 QA 证据】"]
    if not qas:
        lines.append("无。")
    for index, qa in enumerate(qas, 1):
        lines.extend(
            [
                f"{index}. 用户：{qa.get('user_input', '')}",
                f"   助手：{qa.get('assistant_output', '')}",
                f"   时间：{qa.get('timestamp', '')}",
                f"   主题/实体/意图：{qa.get('topic', '')} / {qa.get('core_entity', '')} / {qa.get('intent', '')}",
            ]
        )
    lines.append("\n【相关阶段背景】")
    for segment in segments:
        lines.append(f"- {segment.get('intent', '')}：{_memory_text(segment.get('summary'))}")
    lines.append("\n【相关长期经历背景】")
    for experience in experiences:
        lines.append(
            f"- {experience.get('topic', '')} / {experience.get('core_entity', '')}"
            f"；状态：{experience.get('status', '')}；摘要：{_memory_text(experience.get('summary'))}"
        )
    return "\n".join(lines)


def _keyword_tokens(text: str, *, include_cjk_chars: bool) -> list[str]:
    """中文使用相邻双字组合，其他文本使用空白或标点分隔的词。"""
    tokens: list[str] = []
    for part in _KEYWORD_PARTS.findall(text.casefold()):
        if "\u3400" <= part[0] <= "\u9fff" and len(part) > 1:
            tokens.extend(part[index:index + 2] for index in range(len(part) - 1))
            if include_cjk_chars:
                tokens.extend(part)
        else:
            tokens.append(part)
    return tokens


def _rank_keywords(query: str, pool: dict[str, dict[str, Any]], limit: int) -> list[str]:
    """在 QA 池内计算 BM25，保留至少命中一个关键词的记录。"""
    terms = set(_keyword_tokens(query, include_cjk_chars=False))
    if not terms or not pool:
        return []

    documents = {}
    document_frequencies: Counter[str] = Counter()
    total_length = 0
    for qa_id, qa in pool.items():
        content = " ".join(_memory_text(qa.get(field)) for field in _KEYWORD_FIELDS)
        tokens = _keyword_tokens(content, include_cjk_chars=True)
        frequencies = Counter(token for token in tokens if token in terms)
        document_frequencies.update(frequencies.keys())
        documents[qa_id] = (len(tokens), frequencies)
        total_length += len(tokens)

    if not document_frequencies:
        return []

    average_length = total_length / len(pool)
    inverse_frequencies = {
        term: math.log1p((len(pool) - count + 0.5) / (count + 0.5))
        for term, count in document_frequencies.items()
    }
    k1, b = 1.2, 0.75
    scores = {}
    for qa_id, (length, frequencies) in documents.items():
        if not frequencies:
            continue
        length_factor = k1 * (1 - b + b * length / average_length)
        scores[qa_id] = sum(
            inverse_frequencies[term] * count * (k1 + 1) / (count + length_factor)
            for term, count in frequencies.items()
        )

    return sorted(scores, key=lambda qa_id: (-scores[qa_id], qa_id))[:limit]


class HybridRetriever:
    """从主题、意图和近期时间三条路径只读检索记忆。"""

    def __init__(
        self,
        manager: MemoryManager,
        *,
        qa_limit: int = 5,
        qa_rrf_constant: int = 60,
        qa_dense_candidate_limit: int = 30,
        qa_bm25_candidate_limit: int = 30,
        qa_similarity_threshold: float = 0.0,
        segment_candidate_limit: int = 30,
        experience_candidate_limit: int = 20,
        recent_qa_limit: int = 10,
        experience_qa_candidate_limit: int = 100,
    ) -> None:
        self.manager = manager
        self.storage = manager.storage
        self.qa_limit = max(1, int(qa_limit))
        self.qa_rrf_constant = max(1, int(qa_rrf_constant))
        self.qa_dense_candidate_limit = max(1, int(qa_dense_candidate_limit))
        self.qa_bm25_candidate_limit = max(1, int(qa_bm25_candidate_limit))
        self.qa_similarity_threshold = float(qa_similarity_threshold)
        self.segment_candidate_limit = max(1, int(segment_candidate_limit))
        self.experience_candidate_limit = max(1, int(experience_candidate_limit))
        self.recent_qa_limit = max(1, int(recent_qa_limit))
        self.experience_qa_candidate_limit = max(1, int(experience_qa_candidate_limit))

    def retriever(
        self,
        topic: str,
        core_entity: str,
        query: str,
        intent: str,
        entities: list[str] | None,
        model: str = "default",
    ) -> dict[str, Any]:
        """合并启用路径的 QA，再按关键词和语义名次融合并构建上下文。"""
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        model = model.strip().lower()
        if model not in {"default", "no_topic", "no_intent", "no_time"}:
            raise ValueError("model must be default, no_topic, no_intent, or no_time")

        with ThreadPoolExecutor(max_workers=3) as executor:
            topic_future = (
                executor.submit(self._recall_topic, topic, core_entity, entities)
                if model != "no_topic" else None
            )
            intent_future = (
                executor.submit(self._recall_intent, intent, query)
                if model != "no_intent" else None
            )
            time_future = (
                executor.submit(self._recall_recent)
                if model != "no_time" else None
            )
            experience_ids, topic_qas = topic_future.result() if topic_future else ([], [])
            segment_ids, intent_qas = intent_future.result() if intent_future else ([], [])
            recent_qas = time_future.result() if time_future else []

        candidate_ids = list(dict.fromkeys(
            qa["qa_id"]
            for route in (topic_qas, intent_qas, recent_qas)
            for qa in route
        ))
        with self.storage.retrieval_reader() as reader:
            candidates = {
                qa["qa_id"]: qa for qa in reader.get_qas(candidate_ids)
                if qa["status"] == "open"
            }
            candidate_segment_ids = list(dict.fromkeys(qa["segment_id"] for qa in candidates.values()))
            segments = {
                row["segment_id"]: row
                for row in reader.get_segments(candidate_segment_ids)
                if row["status"] != "deleted"
            }
            candidate_experience_ids = list(dict.fromkeys(
                segment["experience_id"] for segment in segments.values()
            ))
            experiences = {
                row["experience_id"]: row
                for row in reader.get_experiences(candidate_experience_ids)
                if row["status"] != "deleted"
            }
        pool = {}
        for qa_id in candidate_ids:
            qa = candidates.get(qa_id)
            if qa is None:
                continue
            segment = segments.get(qa["segment_id"])
            if segment and segment["experience_id"] in experiences:
                pool[qa_id] = qa

        keyword_ids = _rank_keywords(query, pool, self.qa_bm25_candidate_limit)
        semantic_ids = self._rank_semantic(query, pool)
        final_qas = self._fuse_qas(pool, keyword_ids, semantic_ids)
        final_segments = [
            segments[segment_id]
            for segment_id in dict.fromkeys(qa["segment_id"] for qa in final_qas)
        ]
        final_experiences = [
            experiences[experience_id]
            for experience_id in dict.fromkeys(segment["experience_id"] for segment in final_segments)
        ]

        return {
            "model": model,
            "experiences": final_experiences,
            "segments": final_segments,
            "qas": final_qas,
            "context": build_context_text(final_experiences, final_segments, final_qas),
            "debug": {
                "topic_experience_ids": experience_ids,
                "intent_segment_ids": segment_ids,
                "recent_qa_ids": [qa["qa_id"] for qa in recent_qas],
                "candidate_qa_ids": list(pool),
                "candidate_qa_pool": list(pool.values()),
                "keyword_qa_ids": keyword_ids,
                "semantic_qa_ids": semantic_ids,
            },
        }

    def _recall_topic(
        self, topic: str, core_entity: str, entities: list[str] | None
    ) -> tuple[list[str], list[dict[str, Any]]]:
        with self.storage.retrieval_reader() as reader:
            experiences = reader.search_experiences_by_topic_entity(
                topic, core_entity, entities, self.experience_candidate_limit
            )
            experience_ids = [row["experience_id"] for row in experiences]
            segments = reader.list_segments_by_experience_ids(experience_ids)
            qas = reader.list_qas_by_segment_ids(
                [row["segment_id"] for row in segments],
                limit=self.experience_qa_candidate_limit,
            )
        return experience_ids, qas

    def _recall_intent(self, intent: str, query: str) -> tuple[list[str], list[dict[str, Any]]]:
        """合并意图标签与输入语义命中的 Segment，再展开 QA。"""
        intent = intent.strip()
        semantic_query = f"意图：{intent}\n用户输入：{query}" if intent else query
        semantic_ids = []
        try:
            hits = self.manager.vector_store.query(
                self.manager.embedder.embed(semantic_query),
                memory_type="segment",
                top_k=self.segment_candidate_limit,
                metadata_filter={"status": ["open", "completed"]},
            )
            for hit in hits:
                segment_id = (hit.get("metadata") or {}).get("segment_id")
                if segment_id and hit["similarity"] > 0:
                    semantic_ids.append(segment_id)
        except Exception:
            logger.warning("意图路径 Segment 语义检索失败", exc_info=True)

        semantic_ids = list(dict.fromkeys(semantic_ids))
        with self.storage.retrieval_reader() as reader:
            exact_matches = reader.search_segments_by_intent(intent, self.segment_candidate_limit)
            semantic_matches = reader.get_segments(semantic_ids)
            experience_ids = list(dict.fromkeys(segment["experience_id"] for segment in semantic_matches))
            valid_experiences = {
                row["experience_id"] for row in reader.get_experiences(experience_ids)
                if row["status"] != "deleted"
            }
            valid_semantic_ids = {
                segment["segment_id"] for segment in semantic_matches
                if segment["status"] != "deleted" and segment["experience_id"] in valid_experiences
            }
            segment_ids = list(dict.fromkeys([
                *(segment["segment_id"] for segment in exact_matches),
                *(segment_id for segment_id in semantic_ids if segment_id in valid_semantic_ids),
            ]))
            return segment_ids, reader.list_qas_by_segment_ids(segment_ids)

    def _recall_recent(self) -> list[dict[str, Any]]:
        with self.storage.retrieval_reader() as reader:
            return reader.list_recent_qas(self.recent_qa_limit)

    def _rank_semantic(self, query: str, pool: dict[str, dict[str, Any]]) -> list[str]:
        if not pool:
            return []
        try:
            hits = self.manager.vector_store.query(
                self.manager.embedder.embed(query),
                memory_type="qa",
                top_k=min(self.qa_dense_candidate_limit, len(pool)),
                metadata_filter={"qa_id": list(pool)},
            )
        except Exception:
            logger.warning("候选 QA 语义检索失败", exc_info=True)
            return []
        return list(dict.fromkeys(
            hit["metadata"]["qa_id"]
            for hit in hits
            if (hit.get("metadata") or {}).get("qa_id") in pool
            and hit["similarity"] > self.qa_similarity_threshold
        ))

    def _fuse_qas(
        self,
        pool: dict[str, dict[str, Any]],
        keyword_ids: list[str],
        semantic_ids: list[str],
    ) -> list[dict[str, Any]]:
        ranks: dict[str, dict[str, int]] = {}
        for channel, ids in (("keyword", keyword_ids), ("semantic", semantic_ids)):
            for rank, qa_id in enumerate(ids, start=1):
                ranks.setdefault(qa_id, {})[channel] = rank
        scores = {
            qa_id: sum(1 / (self.qa_rrf_constant + rank) for rank in channels.values())
            for qa_id, channels in ranks.items()
        }
        ordered = sorted(ranks, key=lambda qa_id: (-scores[qa_id], min(ranks[qa_id].values()), qa_id))
        return [
            {
                **pool[qa_id],
                "retrieval": {
                    "channels": ranks[qa_id],
                    "rrf_score": round(scores[qa_id], 8),
                },
            }
            for qa_id in ordered[:self.qa_limit]
        ]
