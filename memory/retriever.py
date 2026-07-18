from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

from .config import get as config_get
from .embedder import TextEmbedder
from .storage import MemoryStorage
from prompts.topic_memory import (
    experience_retrieval_prompt,
    qa_retrieval_prompt,
    segment_retrieval_prompt,
)
from .vector_store import ChromaVectorStore


logger = logging.getLogger(__name__)

DEFAULT_RETRIEVAL_RERANK_MODEL = str(config_get("retrieval", "model", "qwen-plus"))
DEFAULT_RETRIEVAL_RERANK_BASE_URL = str(
    config_get("retrieval", "base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
)


def strip_markdown_code_fence(content: str) -> str:
    text = content.strip()
    if not text.startswith("```"):
        return text

    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_rerank_response(content: str) -> list[dict[str, Any]]:
    payload = json.loads(strip_markdown_code_fence(content))
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        payload = payload["results"]
    if not isinstance(payload, list):
        raise ValueError("Retrieval rerank response must be a JSON array.")

    results: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        memory_id = str(item.get("id") or "").strip()
        if not memory_id:
            continue
        results.append({
            "id": memory_id,
            "score": clamp01(float(item.get("score", 0.0))),
            "reason": str(item.get("reason") or "").strip(),
        })
    return results


class LLMRetrievalReranker:
    """OpenAI-compatible LLM reranker for each memory retrieval layer."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        max_retries: int | None = None,
        retry_delay: float | None = None,
        client: Any | None = None,
    ) -> None:
        api_key = api_key or config_get("retrieval", "api_key")
        model = model or config_get("retrieval", "model", DEFAULT_RETRIEVAL_RERANK_MODEL)
        base_url = base_url or config_get(
            "retrieval", "base_url", DEFAULT_RETRIEVAL_RERANK_BASE_URL
        )
        max_retries = max_retries if max_retries is not None else config_get(
            "retrieval", "max_retries", 3
        )
        retry_delay = retry_delay if retry_delay is not None else config_get(
            "retrieval", "retry_delay", 2.0
        )
        if client is None:
            from openai import OpenAI

            if not api_key:
                raise ValueError("Set retrieval.api_key in configs/config.yaml.")
            client = OpenAI(api_key=str(api_key), base_url=str(base_url))

        self.client = client
        self.model = str(model)
        self.max_retries = int(max_retries)
        self.retry_delay = float(retry_delay)

    def rerank(
        self,
        layer: str,
        query_text: str,
        candidates: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        prompt_builders = {
            "experience": experience_retrieval_prompt,
            "segment": segment_retrieval_prompt,
            "qa": qa_retrieval_prompt,
        }
        prompt_builder = prompt_builders[layer]
        prompt = prompt_builder(query_text, candidates, limit)
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                content = (response.choices[0].message.content or "").strip()
                return parse_rerank_response(content)
            except Exception as error:
                last_error = error
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)

        raise RuntimeError(
            f"{layer} retrieval rerank failed after {self.max_retries} attempts: {last_error}"
        ) from last_error



def safe_json_loads(value: Any, default: Any) -> Any:
    if value is None or value == "":
        return default.copy() if hasattr(default, "copy") else default
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default.copy() if hasattr(default, "copy") else default


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def parse_summary(value: Any) -> str:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ""
        parsed = safe_json_loads(stripped, None)
        if parsed is None:
            return "" if stripped[:1] in "[{" else stripped
    else:
        parsed = value
    if isinstance(parsed, str):
        return parsed.strip()
    if isinstance(parsed, dict):
        if isinstance(parsed.get("summary"), str):
            return parsed["summary"].strip()
        for key in ("long", "short"):
            if isinstance(parsed.get(key), str) and parsed[key].strip():
                return parsed[key].strip()
    return ""



def parse_state(value: Any) -> dict[str, Any]:
    parsed = safe_json_loads(value, {})
    return parsed if isinstance(parsed, dict) else {}


def parse_intents(value: Any) -> list[str]:
    parsed = safe_json_loads(value, [])
    return [str(item) for item in parsed if str(item).strip()] if isinstance(parsed, list) else []


def parse_entities(value: Any) -> list[str]:
    parsed = safe_json_loads(value, [])
    return [str(item) for item in parsed if str(item).strip()] if isinstance(parsed, list) else []


def build_query_text(
    topic: str,
    core_entity: str,
    intent: str | None = None,
    entities: list[str] | None = None,
) -> str:
    return "\n".join(
        [
            f"主题: {topic}",
            f"核心实体: {core_entity}",
            f"用户意图: {intent or ''}",
            f"相关实体: {'、'.join(entities or [])}",
        ]
    )


def build_context_text(
    experiences: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    qas: list[dict[str, Any]],
) -> str:
    lines = ["【长期记忆 Experience】"]
    if not experiences:
        lines.append("未找到相关长期记忆。")
    for index, experience in enumerate(experiences, 1):
        prefix = f"{index}. " if len(experiences) > 1 else ""
        lines.extend(
            [
                f"{prefix}主题: {experience['topic']}",
                f"核心实体: {experience['core_entity']}",
                f"相关意图: {'、'.join(experience['intents'])}",
                f"摘要: {experience['summary']}",
                "",
            ]
        )

    lines.append("【相关片段 Segment】")
    if not segments:
        lines.append("未找到相关片段。")
    for index, segment in enumerate(segments, 1):
        lines.extend(
            [
                f"{index}. 意图: {segment['intent']}",
                f"   摘要: {segment['summary']}",
                "",
            ]
        )

    lines.append("【原始问答 QA】")
    if not qas:
        lines.append("未找到相关原始问答。")
    for index, qa in enumerate(qas, 1):
        lines.extend(
            [
                f"{index}. 时间: {qa['timestamp']}",
                f"用户: {qa['user_input']}",
                f"助手: {qa['assistant_output']}",
                "",
            ]
        )
    return "\n".join(lines).rstrip()

class HybridRetriever:
    """Hierarchical structured-first Experience -> Segment -> QA retriever."""

    def __init__(
        self,
        storage: MemoryStorage,
        vector_store: ChromaVectorStore | None = None,
        embedder: TextEmbedder | None = None,
        reranker: LLMRetrievalReranker | None = None,
        rerank_with_llm: bool = True,
        retrieval_api_key: str | None = None,
        retrieval_model: str | None = None,
        retrieval_base_url: str | None = None,
        retrieval_max_retries: int | None = None,
        retrieval_retry_delay: float | None = None,
        **_: Any,
    ) -> None:
        self.storage = storage
        self.vector_store = vector_store
        self.embedder = embedder
        self.reranker = reranker

        if self.reranker is None and rerank_with_llm:
            self.reranker = LLMRetrievalReranker(
                api_key=retrieval_api_key,
                model=retrieval_model,
                base_url=retrieval_base_url,
                max_retries=retrieval_max_retries,
                retry_delay=retrieval_retry_delay,
            )

    def recall(
        self,
        topic: str,
        core_entity: str,
        intent: str | None = None,
        entities: list[str] | None = None,
        top_experience: int = 3,
        top_segment: int = 5,
        top_qa: int = 8,
        top_k: int | None = None,
        state_key: str = "default",
        use_cache: bool = True,
    ) -> dict[str, Any]:
        if top_k is not None:
            top_qa = top_k
        if min(top_experience, top_segment, top_qa) <= 0:
            raise ValueError("top_experience, top_segment and top_qa must be positive")

        query_entities = [str(item) for item in entities or [] if str(item).strip()]
        query_text = build_query_text(topic, core_entity, intent, query_entities)
        query_embedding = self._embed_query(query_text)
        retrieval_cache = self._load_retrieval_cache(state_key) if use_cache else {}
        experience_cache_hit = False
        segment_cache_hit = False

        experience_vector_ids: set[str] = set()
        experiences: list[dict[str, Any]] = []
        if use_cache and self._cache_same_experience(retrieval_cache, topic, core_entity):
            experiences = self._cached_experiences(retrieval_cache, top_experience)
            experience_cache_hit = bool(experiences)

        if experience_cache_hit:
            experience_count = len(experiences)
        else:
            experience_vector_ids = self._vector_candidate_ids(
                "experience", query_embedding, max(20, top_experience * 8)
            )
            experiences, experience_count = self._recall_experiences(
                topic, core_entity, intent, query_text, experience_vector_ids, top_experience
            )

        segment_vector_ids: set[str] = set()
        segments: list[dict[str, Any]] = []
        if experience_cache_hit and self._cache_same_segment(retrieval_cache, intent):
            segments = self._cached_segments(retrieval_cache, experiences, top_segment)
            segment_cache_hit = bool(segments)

        if segment_cache_hit:
            segment_count = len(segments)
        else:
            segment_vector_ids = self._vector_candidate_ids(
                "segment", query_embedding, max(20, top_segment * 8)
            )
            segments, segment_count = self._recall_segments(
                experiences, intent, query_text, segment_vector_ids, top_segment
            )

        qa_vector_ids = self._vector_candidate_ids(
            "qa", query_embedding, max(20, top_qa * 8)
        )
        qas, qa_count = self._recall_qas(
            segments, query_text, qa_vector_ids, top_qa
        )

        debug = {
            "experience_candidates": experience_count,
            "segment_candidates": segment_count,
            "qa_candidates": qa_count,
            "vector_experience_candidates": len(experience_vector_ids),
            "vector_segment_candidates": len(segment_vector_ids),
            "vector_qa_candidates": len(qa_vector_ids),
            "experience_cache_hit": experience_cache_hit,
            "segment_cache_hit": segment_cache_hit,
        }
        logger.info(
            "閸掑棗鐪伴崣顒€娲栫€瑰本鍨?experience=%s/%s segment=%s/%s qa=%s/%s vectors=%s/%s/%s cache=%s/%s",
            len(experiences), experience_count, len(segments), segment_count,
            len(qas), qa_count, len(experience_vector_ids), len(segment_vector_ids),
            len(qa_vector_ids), experience_cache_hit, segment_cache_hit,
        )
        if use_cache:
            self._store_retrieval_cache(
                state_key, topic, core_entity, intent, query_entities, experiences, segments
            )
        return {
            "query": {
                "topic": topic,
                "core_entity": core_entity,
                "intent": intent or "",
                "entities": query_entities,
            },
            "experiences": experiences,
            "segments": segments,
            "qas": qas,
            "context_text": build_context_text(experiences, segments, qas),
            "debug": debug,
            "results": [{"qa": qa, "score": qa["score"]} for qa in qas],
        }

    def _cache_signature(
        self,
        topic: str,
        core_entity: str,
        intent: str | None,
        entities: list[str],
    ) -> dict[str, Any]:
        return {
            "topic": str(topic or "").strip(),
            "core_entity": str(core_entity or "").strip(),
            "intent": str(intent or "").strip(),
            "entities": self._normalized_cache_entities(entities),
        }

    def _normalized_cache_entities(self, entities: list[str]) -> list[str]:
        return sorted({str(item).strip() for item in entities if str(item).strip()})

    def _load_retrieval_cache(self, state_key: str) -> dict[str, Any]:
        runtime = self.storage.get_runtime_state(state_key)
        if not runtime:
            return {}
        cache = runtime.get("retrieval_cache")
        return cache if isinstance(cache, dict) else {}

    def _cache_same_experience(
        self,
        cache: dict[str, Any],
        topic: str,
        core_entity: str,
    ) -> bool:
        query = cache.get("query") if isinstance(cache, dict) else None
        if not isinstance(query, dict):
            return False
        return (
            query.get("topic") == str(topic or "").strip()
            and query.get("core_entity") == str(core_entity or "").strip()
        )

    def _cache_same_segment(
        self,
        cache: dict[str, Any],
        intent: str | None,
    ) -> bool:
        query = cache.get("query") if isinstance(cache, dict) else None
        if not isinstance(query, dict):
            return False
        return query.get("intent", "") == str(intent or "").strip()

    def _cached_experiences(
        self,
        cache: dict[str, Any],
        limit: int,
    ) -> list[dict[str, Any]]:
        experience_ids = [
            str(item) for item in cache.get("experience_ids") or [] if str(item).strip()
        ]
        if not experience_ids:
            return []
        experiences: list[dict[str, Any]] = []
        for item in self.storage.get_experiences(experience_ids):
            experience = self._prepare_experience(item, set())
            experience["score"] = 1.0
            experience["cache_hit"] = True
            experiences.append(experience)
            if len(experiences) >= limit:
                break
        return experiences

    def _cached_segments(
        self,
        cache: dict[str, Any],
        experiences: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        segment_ids = [str(item) for item in cache.get("segment_ids") or [] if str(item).strip()]
        if not segment_ids:
            return []
        experience_ids = {item["experience_id"] for item in experiences}
        segments: list[dict[str, Any]] = []
        for item in self.storage.get_segments(segment_ids):
            if item.get("experience_id") not in experience_ids:
                continue
            if item.get("status") == "deleted":
                continue
            segment = self._prepare_segment(item, set())
            segment["score"] = 1.0
            segment["cache_hit"] = True
            segments.append(segment)
            if len(segments) >= limit:
                break
        return segments

    def _store_retrieval_cache(
        self,
        state_key: str,
        topic: str,
        core_entity: str,
        intent: str | None,
        entities: list[str],
        experiences: list[dict[str, Any]],
        segments: list[dict[str, Any]],
    ) -> None:
        cache = {
            "query": self._cache_signature(topic, core_entity, intent, entities),
            "experience_ids": [item["experience_id"] for item in experiences],
            "segment_ids": [item["segment_id"] for item in segments],
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            self.storage.update_runtime_retrieval_cache(
                state_key=state_key,
                retrieval_cache=cache,
                updated_at=cache["cached_at"],
            )
            self.storage.commit()
        except Exception:
            self.storage.rollback()
            logger.exception("Failed to update retrieval cache")

    def _embed_query(self, query_text: str) -> list[float] | None:
        if self.vector_store is None or self.embedder is None:
            return None
        try:
            return self.embedder.embed(query_text)
        except Exception:
            logger.exception(
                "Query embedding failed, continuing with structured candidates"
            )
            return None

    def _vector_candidate_ids(
        self,
        memory_type: str,
        query_embedding: list[float] | None,
        top_k: int,
    ) -> set[str]:
        if self.vector_store is None or query_embedding is None:
            return set()
        try:
            items = self.vector_store.query(
                query_embedding=query_embedding,
                memory_type=memory_type,
                top_k=top_k,
            )
        except Exception:
            logger.exception("%s vector recall failed, continuing with structured candidates", memory_type)
            return set()

        candidate_ids: set[str] = set()
        for item in items:
            metadata = item.get("metadata") or {}
            memory_id = metadata.get("memory_id") or metadata.get(f"{memory_type}_id")
            if memory_id:
                candidate_ids.add(str(memory_id))
        return candidate_ids

    def _truncate_for_prompt(self, value: Any, limit: int = 600) -> str:
        text = str(value or "").strip()
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + "..."

    def _prompt_candidate(
        self, item: dict[str, Any], id_field: str
    ) -> dict[str, Any]:
        candidate: dict[str, Any] = {
            "id": item[id_field],
            "topic": item.get("topic", ""),
            "core_entity": item.get("core_entity", ""),
        }
        for key in (
            "intents",
            "intent",
            "entities",
            "confidence",
            "updated_at",
            "timestamp",
            "status",
            "vector_recalled",
        ):
            if key in item:
                candidate[key] = item[key]
        for key in ("summary", "state", "user_input", "assistant_output", "reasoning"):
            if key in item:
                candidate[key] = self._truncate_for_prompt(item[key])
        return candidate

    def _prepare_experience(
        self, item: dict[str, Any], vector_candidate_ids: set[str]
    ) -> dict[str, Any]:
        return {
            "experience_id": item["experience_id"],
            "topic": item.get("topic", ""),
            "core_entity": item.get("core_entity", ""),
            "intents": parse_intents(item.get("intents_link")),
            "summary": parse_summary(item.get("summary")),
            "state": parse_state(item.get("state")),
            "vector_recalled": item["experience_id"] in vector_candidate_ids,
            "updated_at": item.get("updated_at", ""),
        }

    def _prepare_segment(
        self, item: dict[str, Any], vector_candidate_ids: set[str]
    ) -> dict[str, Any]:
        return {
            "segment_id": item["segment_id"],
            "experience_id": item["experience_id"],
            "topic": item.get("topic", ""),
            "core_entity": item.get("core_entity", ""),
            "intent": str(item.get("intent") or ""),
            "status": item.get("status", ""),
            "summary": str(item.get("summary") or ""),
            "vector_recalled": item["segment_id"] in vector_candidate_ids,
            "updated_at": item.get("updated_at", ""),
        }

    def _prepare_qa(
        self, item: dict[str, Any], vector_candidate_ids: set[str]
    ) -> dict[str, Any]:
        return {
            "qa_id": item["qa_id"],
            "segment_id": item["segment_id"],
            "timestamp": item.get("timestamp", ""),
            "user_input": item.get("user_input", ""),
            "assistant_output": item.get("assistant_output", ""),
            "topic": item.get("topic", ""),
            "core_entity": item.get("core_entity", ""),
            "intent": item.get("intent", ""),
            "entities": parse_entities(item.get("entities")),
            "confidence": clamp01(float(item.get("confidence") or 0.0)),
            "vector_recalled": item["qa_id"] in vector_candidate_ids,
            "reasoning": item.get("reasoning", ""),
            # tools is deserialized by _row_to_dict (tools_json → tools);
            # carried through so callers can extract dia_ids for retrieval metrics.
            "tools": item.get("tools") or [],
        }

    def _select_candidates_with_llm(
        self,
        layer: str,
        query_text: str,
        candidates: list[dict[str, Any]],
        id_field: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        if self.reranker is None:
            raise RuntimeError("LLM retrieval reranker is required for memory scoring.")

        prompt_candidates = [self._prompt_candidate(item, id_field) for item in candidates]
        selections = self.reranker.rerank(
            layer=layer,
            query_text=query_text,
            candidates=prompt_candidates,
            limit=limit,
        )

        by_id = {str(item[id_field]): item for item in candidates}
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        for selection in selections:
            memory_id = str(selection.get("id") or "").strip()
            if not memory_id or memory_id in seen or memory_id not in by_id:
                continue
            item = dict(by_id[memory_id])
            item["score"] = clamp01(float(selection.get("score", 0.0)))
            reason = str(selection.get("reason") or "").strip()
            if reason:
                item["llm_reason"] = reason
            selected.append(item)
            seen.add(memory_id)
            if len(selected) >= limit:
                break

        if not selected:
            logger.warning("%s LLM selection returned no valid candidates", layer)
        return selected

    def _recall_experiences(
        self,
        topic: str,
        core_entity: str,
        intent: str | None,
        query_text: str,
        vector_candidate_ids: set[str],
        limit: int,
    ) -> tuple[list[dict[str, Any]], int]:
        exact = self.storage.find_experiences(topic, core_entity, max(limit * 4, limit))
        candidates = list(exact)
        if not candidates:
            candidates = self.storage.list_experiences_by_topic(topic, max(limit * 8, 20))
            known = {item["experience_id"] for item in candidates}
            vector_rows = self.storage.get_experiences(list(vector_candidate_ids))
            candidates.extend(
                item for item in vector_rows if item["experience_id"] not in known
            )

        prepared = [self._prepare_experience(item, vector_candidate_ids) for item in candidates]
        ranked = self._select_candidates_with_llm(
            "experience", query_text, prepared, "experience_id", limit
        )
        logger.info("Experience 閸婃瑩鈧鏆熼柌?%s 閸氭垿鍣洪崐娆撯偓澶嬫殶闁?%s", len(candidates), len(vector_candidate_ids))
        return ranked, len(candidates)

    def _recall_segments(
        self,
        experiences: list[dict[str, Any]],
        intent: str | None,
        query_text: str,
        vector_candidate_ids: set[str],
        limit: int,
    ) -> tuple[list[dict[str, Any]], int]:
        experience_ids = [item["experience_id"] for item in experiences]
        candidates = self.storage.list_segments_by_experience_ids(experience_ids)
        prepared = [self._prepare_segment(item, vector_candidate_ids) for item in candidates]
        ranked = self._select_candidates_with_llm(
            "segment", query_text, prepared, "segment_id", limit
        )
        logger.info("Segment 閸婃瑩鈧鏆熼柌?%s 閸氭垿鍣洪崐娆撯偓澶嬫殶闁?%s", len(candidates), len(vector_candidate_ids))
        return ranked, len(candidates)

    def _recall_qas(
        self,
        segments: list[dict[str, Any]],
        query_text: str,
        vector_candidate_ids: set[str],
        limit: int,
    ) -> tuple[list[dict[str, Any]], int]:
        segment_ids = [item["segment_id"] for item in segments]
        candidates = self.storage.list_qas_by_segment_ids(segment_ids)
        prepared = [self._prepare_qa(item, vector_candidate_ids) for item in candidates]
        ranked = self._select_candidates_with_llm("qa", query_text, prepared, "qa_id", limit)
        ranked.sort(key=lambda item: item["timestamp"])
        logger.info("QA 閸婃瑩鈧鏆熼柌?%s 閸氭垿鍣洪崐娆撯偓澶嬫殶闁?%s", len(candidates), len(vector_candidate_ids))
        return ranked, len(candidates)


class StructuredRetriever(HybridRetriever):
    """Backward-compatible alias."""
