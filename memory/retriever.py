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


def _parse_score(value: Any) -> float:
    try:
        return clamp01(float(value))
    except (TypeError, ValueError):
        return 0.0


def _parse_hierarchy_items(items: Any, level: str) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    specs = {
        "experience": ("experience_id", "segments", "segment"),
        "segment": ("segment_id", "qas", "qa"),
        "qa": ("qa_id", "", ""),
    }
    id_field, children_field, child_level = specs[level]
    parsed: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        memory_id = str(item.get("id") or item.get(id_field) or "").strip()
        if not memory_id:
            continue
        result = {
            "id": memory_id,
            "score": _parse_score(item.get("score")),
            "reason": str(item.get("reason") or "").strip(),
        }
        if children_field:
            result[children_field] = _parse_hierarchy_items(
                item.get(children_field), child_level
            )
        parsed.append(result)
    return parsed


def parse_hierarchical_rerank_response(content: str) -> dict[str, Any]:
    """Parse a single-call Experience -> Segment -> QA selection tree."""
    payload = json.loads(strip_markdown_code_fence(content))
    if not isinstance(payload, dict) or not isinstance(payload.get("experiences"), list):
        raise ValueError("Hierarchical rerank response must contain experiences[].")
    return {
        "experiences": _parse_hierarchy_items(payload["experiences"], "experience")
    }


def build_hierarchical_retrieval_prompt(
    query_text: str,
    candidate_tree: list[dict[str, Any]],
    top_experience: int,
    top_segment: int,
    top_qa: int,
) -> str:
    """Build the joint prompt locally so retrieval remains a one-file change."""
    candidates_json = json.dumps(candidate_tree, ensure_ascii=False, separators=(",", ":"))
    return f"""# Role

You are the retrieval filter for a hierarchical long-term memory system.
Select the best Experience -> Segment -> QA subtree for the query in one pass.

# Query

{query_text}

# Candidate Tree

{candidates_json}

# Selection Rules

1. Judge the complete path jointly. A QA must directly help answer the query; its
   Segment intent and Experience topic/core_entity must also be compatible.
2. vector_similarity and keyword_score are complementary recall hints, not final
   relevance scores.
3. Prefer specific, traceable QA evidence. Use summaries to understand context,
   but do not choose an unrelated QA only because its parent summary is relevant.
4. Select at most {top_experience} Experiences, {top_segment} Segments globally,
   and {top_qa} QAs globally.
5. Every selected Segment must be nested under its real selected Experience, and
   every selected QA must be nested under its real selected Segment.
6. Select only IDs present in Candidate Tree. Do not invent information.

# Output

Return only a JSON object, without Markdown or extra text:

{{
  "experiences": [
    {{
      "id": "experience id",
      "score": 0.0,
      "reason": "brief reason",
      "segments": [
        {{
          "id": "segment id",
          "score": 0.0,
          "reason": "brief reason",
          "qas": [
            {{"id": "qa id", "score": 0.0, "reason": "brief reason"}}
          ]
        }}
      ]
    }}
  ]
}}

Sort every list from most relevant to least relevant. Scores must be numbers from
0.0 to 1.0. Return an empty experiences list only when no candidate is useful."""


class LLMRetrievalReranker:
    """OpenAI-compatible LLM reranker for hierarchical memory retrieval."""

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

    def rerank_hierarchy(
        self,
        query_text: str,
        candidate_tree: list[dict[str, Any]],
        top_experience: int,
        top_segment: int,
        top_qa: int,
    ) -> dict[str, Any]:
        """Jointly select all three layers with exactly one API request."""
        prompt = build_hierarchical_retrieval_prompt(
            query_text=query_text,
            candidate_tree=candidate_tree,
            top_experience=top_experience,
            top_segment=top_segment,
            top_qa=top_qa,
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        content = (response.choices[0].message.content or "").strip()
        return parse_hierarchical_rerank_response(content)



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
    query: str | None = None
) -> str:
    return "\n".join(
        [
            f"主题: {topic}",
            f"核心实体: {core_entity}",
            f"用户意图: {intent or ''}",
            f"相关实体: {'、'.join(entities or [])}",
            f"问题: {query}",
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
        query: str | None = None,
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
        query_text = build_query_text(topic, core_entity, intent, query_entities, query)
        keyword_query_text = "\n".join(
            value
            for value in (
                str(topic or "").strip(),
                str(core_entity or "").strip(),
                str(intent or "").strip(),
                *(str(item).strip() for item in query_entities),
                str(query or "").strip(),
            )
            if value
        )
        query_embedding = self._embed_query(query_text)
        retrieval_cache = self._load_retrieval_cache(state_key) if use_cache else {}

        candidate_data = self._collect_hierarchical_candidates(
            topic=topic,
            core_entity=core_entity,
            intent=intent,
            keyword_query_text=keyword_query_text,
            query_embedding=query_embedding,
            retrieval_cache=retrieval_cache,
            top_experience=top_experience,
            top_segment=top_segment,
            top_qa=top_qa,
        )
        candidate_tree = self._build_prompt_candidate_tree(candidate_data)

        llm_calls = 0
        selection: dict[str, Any] = {"experiences": []}
        if candidate_tree and self.reranker is not None:
            rerank_hierarchy = getattr(self.reranker, "rerank_hierarchy", None)
            if callable(rerank_hierarchy):
                llm_calls = 1
                try:
                    selection = rerank_hierarchy(
                        query_text=query_text,
                        candidate_tree=candidate_tree,
                        top_experience=top_experience,
                        top_segment=top_segment,
                        top_qa=top_qa,
                    )
                except Exception:
                    logger.exception(
                        "Hierarchical LLM rerank failed; using deterministic fallback"
                    )
            else:
                logger.warning(
                    "Injected reranker has no rerank_hierarchy(); using local fallback"
                )

        experiences, segments, qas = self._hydrate_hierarchical_selection(
            selection,
            candidate_data,
            top_experience,
            top_segment,
            top_qa,
        )
        if not qas and candidate_data["qas"]:
            experiences, segments, qas = self._local_hierarchical_selection(
                candidate_data, top_experience, top_segment, top_qa
            )

        raw_counts = candidate_data["raw_counts"]
        experience_count = raw_counts["experience"]
        segment_count = raw_counts["segment"]
        qa_count = raw_counts["qa"]
        vector_counts = candidate_data["vector_counts"]
        keyword_counts = candidate_data["keyword_counts"]
        cached_experience_ids = set(retrieval_cache.get("experience_ids") or [])
        cached_segment_ids = set(retrieval_cache.get("segment_ids") or [])
        experience_cache_hit = bool(
            use_cache
            and self._cache_same_experience(retrieval_cache, topic, core_entity)
            and cached_experience_ids
        )
        segment_cache_hit = bool(
            experience_cache_hit
            and self._cache_same_segment(retrieval_cache, intent)
            and cached_segment_ids
        )

        debug = {
            "experience_candidates": experience_count,
            "segment_candidates": segment_count,
            "qa_candidates": qa_count,
            "prompt_experience_candidates": len(candidate_data["experiences"]),
            "prompt_segment_candidates": len(candidate_data["segments"]),
            "prompt_qa_candidates": len(candidate_data["qas"]),
            "vector_experience_candidates": vector_counts["experience"],
            "vector_segment_candidates": vector_counts["segment"],
            "vector_qa_candidates": vector_counts["qa"],
            "keyword_experience_candidates": keyword_counts["experience"],
            "keyword_segment_candidates": keyword_counts["segment"],
            "keyword_qa_candidates": keyword_counts["qa"],
            "experience_cache_hit": experience_cache_hit,
            "segment_cache_hit": segment_cache_hit,
            "retrieval_strategy": "hierarchical_hybrid_joint",
            "llm_calls": llm_calls,
        }
        logger.info(
            "检索结果：experience=%s/%s segment=%s/%s qa=%s/%s vectors=%s/%s/%s cache=%s/%s",
            len(experiences), experience_count, len(segments), segment_count,
            len(qas), qa_count, vector_counts["experience"],
            vector_counts["segment"], vector_counts["qa"],
            experience_cache_hit, segment_cache_hit,
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
            # "results": [{"qa": qa, "score": qa["score"]} for qa in qas],
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

    def _vector_candidates(
        self,
        memory_type: str,
        query_embedding: list[float] | None,
        top_k: int,
        allowed_ids: set[str] | None = None,
        min_allowed: int = 0,
    ) -> dict[str, float]:
        if self.vector_store is None or query_embedding is None:
            return {}
        requested = max(1, top_k)
        while True:
            try:
                items = self.vector_store.query(
                    query_embedding=query_embedding,
                    memory_type=memory_type,
                    top_k=requested,
                )
            except Exception:
                logger.exception(
                    "%s vector recall failed, continuing with structured candidates",
                    memory_type,
                )
                return {}

            candidates: dict[str, float] = {}
            for item in items:
                metadata = item.get("metadata") or {}
                memory_id = metadata.get("memory_id") or metadata.get(
                    f"{memory_type}_id"
                )
                memory_id = str(memory_id or "")
                if memory_id and (allowed_ids is None or memory_id in allowed_ids):
                    candidates[memory_id] = clamp01(
                        float(item.get("similarity") or 0.0)
                    )
            if allowed_ids is None or len(candidates) >= min_allowed:
                return candidates
            if len(items) < requested:
                return candidates
            count_method = getattr(self.vector_store, "count", None)
            total = int(count_method()) if callable(count_method) else requested
            if requested >= total:
                return candidates
            requested = min(total, requested * 2)

    def _vector_candidate_ids(
        self,
        memory_type: str,
        query_embedding: list[float] | None,
        top_k: int,
    ) -> set[str]:
        """Backward-compatible ID-only vector recall helper."""
        return set(self._vector_candidates(memory_type, query_embedding, top_k))

    def _lexical_units(self, value: Any) -> set[str]:
        text = str(value or "").lower()
        words = set("".join(char if char.isalnum() else " " for char in text).split())
        words = {word for word in words if len(word) > 1}
        chinese = [char for char in text if "\u4e00" <= char <= "\u9fff"]
        words.update(
            chinese[index] + chinese[index + 1]
            for index in range(len(chinese) - 1)
        )
        return words

    def _lexical_overlap(self, query_text: str, candidate_text: str) -> float:
        query_units = self._lexical_units(query_text)
        candidate_units = self._lexical_units(candidate_text)
        if not query_units or not candidate_units:
            return 0.0
        return len(query_units & candidate_units) / max(1, min(len(query_units), 12))

    def _score_candidate(
        self,
        item: dict[str, Any],
        topic: str,
        core_entity: str,
        intent: str | None,
        query_text: str,
        cached: bool = False,
    ) -> float:
        vector_similarity = max(
            float(item.get("vector_similarity") or 0.0),
            float(item.get("descendant_similarity") or 0.0) * 0.9,
        )
        keyword_score = float(item.get("keyword_score") or 0.0)
        score = vector_similarity * 0.45 + keyword_score * 0.15
        if str(item.get("topic") or "").strip() == str(topic or "").strip():
            score += 0.15
        if str(item.get("core_entity") or "").strip() == str(core_entity or "").strip():
            score += 0.1

        item_intents = item.get("intents") or [item.get("intent")]
        normalized_intents = {str(value or "").strip() for value in item_intents}
        if str(intent or "").strip() in normalized_intents:
            score += 0.1

        candidate_text = " ".join(
            str(item.get(key) or "")
            for key in (
                "topic",
                "core_entity",
                "intents",
                "intent",
                "summary",
                "user_input",
                "assistant_output",
                "entities",
            )
        )
        if not keyword_score:
            score += self._lexical_overlap(query_text, candidate_text) * 0.15
        if cached:
            score += 0.05
        return clamp01(score)

    def _rank_with_parent_coverage(
        self,
        items: list[dict[str, Any]],
        id_field: str,
        limit: int,
        parent_field: str | None = None,
    ) -> list[dict[str, Any]]:
        ranked = sorted(
            items,
            key=lambda item: (
                float(item.get("_candidate_score") or 0.0),
                str(item.get("updated_at") or item.get("timestamp") or ""),
                str(item.get(id_field) or ""),
            ),
            reverse=True,
        )
        if not parent_field:
            return ranked[:limit]

        selected: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        seen_parents: set[str] = set()
        for item in ranked:
            parent_id = str(item.get(parent_field) or "")
            memory_id = str(item.get(id_field) or "")
            if parent_id in seen_parents:
                continue
            selected.append(item)
            seen_ids.add(memory_id)
            seen_parents.add(parent_id)
            if len(selected) >= limit:
                return selected
        for item in ranked:
            memory_id = str(item.get(id_field) or "")
            if memory_id in seen_ids:
                continue
            selected.append(item)
            if len(selected) >= limit:
                break
        return selected

    def _rank_hybrid_candidates(
        self,
        items: list[dict[str, Any]],
        id_field: str,
        limit: int,
        parent_field: str | None = None,
    ) -> list[dict[str, Any]]:
        """Keep explicit vector and keyword recall quotas, then fill by fusion score."""
        combined = sorted(
            items,
            key=lambda item: (
                float(item.get("_candidate_score") or 0.0),
                str(item.get("updated_at") or item.get("timestamp") or ""),
            ),
            reverse=True,
        )
        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set()

        def add(item: dict[str, Any]) -> None:
            memory_id = str(item.get(id_field) or "")
            if memory_id and memory_id not in selected_ids and len(selected) < limit:
                selected.append(item)
                selected_ids.add(memory_id)

        if parent_field:
            seen_parents: set[str] = set()
            for item in combined:
                parent_id = str(item.get(parent_field) or "")
                if parent_id in seen_parents:
                    continue
                add(item)
                seen_parents.add(parent_id)

        channel_quota = max(1, limit // 3)
        vector_ranked = sorted(
            (
                item
                for item in items
                if float(item.get("vector_similarity") or 0.0) > 0.0
            ),
            key=lambda item: float(item.get("vector_similarity") or 0.0),
            reverse=True,
        )
        keyword_ranked = sorted(
            (
                item
                for item in items
                if float(item.get("keyword_score") or 0.0) > 0.0
            ),
            key=lambda item: float(item.get("keyword_score") or 0.0),
            reverse=True,
        )
        for item in vector_ranked[:channel_quota]:
            add(item)
        for item in keyword_ranked[:channel_quota]:
            add(item)
        for item in combined:
            add(item)

        return sorted(
            selected,
            key=lambda item: float(item.get("_candidate_score") or 0.0),
            reverse=True,
        )

    def _keyword_experience_rows(
        self,
        topic: str,
        core_entity: str,
        intent: str | None,
        query_text: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        ignored = {
            "主题",
            "核心实体",
            "用户意图",
            "相关实体",
            "问题",
            "topic",
            "core",
            "entity",
            "intent",
            "query",
            "question",
            "user",
            "related",
        }
        terms: list[str] = []
        for value in (topic, core_entity, intent):
            term = str(value or "").strip()
            if term and term not in terms:
                terms.append(term)
        lexical_terms = sorted(
            (
                term
                for term in self._lexical_units(query_text)
                if term not in ignored and 1 < len(term) <= 32
            ),
            key=lambda value: (len(value), value),
            reverse=True,
        )
        for term in lexical_terms:
            if term not in terms:
                terms.append(term)
            if len(terms) >= 12:
                break
        if not terms:
            return []

        searchable = (
            "COALESCE(topic, '') || ' ' || COALESCE(core_entity, '') || ' ' || "
            "COALESCE(intents_link_json, '') || ' ' || COALESCE(summary_json, '') || "
            "' ' || COALESCE(state_json, '')"
        )
        where_clause = " OR ".join(f"({searchable}) LIKE ?" for _ in terms)
        parameters = [f"%{term}%" for term in terms]
        rows = self.storage.connection.execute(
            f"""
            SELECT experience_id
            FROM experience_memory
            WHERE {where_clause}
            ORDER BY updated_at DESC, created_at DESC
            LIMIT ?
            """,
            (*parameters, limit),
        ).fetchall()
        return self.storage.get_experiences(
            [str(row["experience_id"]) for row in rows]
        )

    def _candidate_keyword_score(
        self, item: dict[str, Any], query_text: str
    ) -> float:
        candidate_text = " ".join(
            str(item.get(key) or "")
            for key in (
                "topic",
                "core_entity",
                "intents",
                "intent",
                "summary",
                "user_input",
                "assistant_output",
                "entities",
            )
        )
        return clamp01(self._lexical_overlap(query_text, candidate_text))

    def _collect_hierarchical_candidates(
        self,
        *,
        topic: str,
        core_entity: str,
        intent: str | None,
        keyword_query_text: str,
        query_embedding: list[float] | None,
        retrieval_cache: dict[str, Any],
        top_experience: int,
        top_segment: int,
        top_qa: int,
    ) -> dict[str, Any]:
        experience_limit = max(
            top_experience,
            int(config_get("retrieval", "experience_candidate_limit", 12)),
            top_experience * 6,
        )
        segment_limit = max(
            top_segment,
            int(config_get("retrieval", "segment_candidate_limit", 36)),
            top_segment * 12,
        )
        qa_limit = max(
            top_qa,
            int(config_get("retrieval", "qa_candidate_limit", 80)),
            top_qa * 10,
        )
        cache_matches_experience = self._cache_same_experience(
            retrieval_cache, topic, core_entity
        )
        cache_matches_segment = cache_matches_experience and self._cache_same_segment(
            retrieval_cache, intent
        )
        cached_experience_ids = (
            {str(value) for value in retrieval_cache.get("experience_ids") or []}
            if cache_matches_experience
            else set()
        )
        cached_segment_ids = (
            {str(value) for value in retrieval_cache.get("segment_ids") or []}
            if cache_matches_segment
            else set()
        )

        # Layer 1: Experience recall. Only this layer may define the parent scope.
        experience_vectors = self._vector_candidates(
            "experience", query_embedding, max(20, experience_limit * 4)
        )
        experience_rows: dict[str, dict[str, Any]] = {}

        def add_experience(row: dict[str, Any] | None) -> None:
            if row and row.get("experience_id"):
                experience_rows[str(row["experience_id"])] = row

        structured_limit = max(experience_limit * 2, 20)
        for row in self.storage.find_experiences(topic, core_entity, structured_limit):
            add_experience(row)
        for row in self.storage.list_experiences_by_topic(topic, structured_limit):
            add_experience(row)
        for row in self._keyword_experience_rows(
            topic, core_entity, intent, keyword_query_text, structured_limit
        ):
            add_experience(row)
        for row in self.storage.get_experiences(list(experience_vectors)):
            add_experience(row)
        for row in self.storage.get_experiences(list(cached_experience_ids)):
            add_experience(row)

        prepared_experiences: list[dict[str, Any]] = []
        for experience_id, row in experience_rows.items():
            item = self._prepare_experience(row, set(experience_vectors))
            item["vector_similarity"] = experience_vectors.get(experience_id, 0.0)
            item["keyword_score"] = self._candidate_keyword_score(
                item, keyword_query_text
            )
            item["keyword_recalled"] = item["keyword_score"] > 0.0
            item["_candidate_score"] = self._score_candidate(
                item,
                topic,
                core_entity,
                intent,
                keyword_query_text,
                experience_id in cached_experience_ids,
            )
            prepared_experiences.append(item)
        raw_experience_count = len(prepared_experiences)
        prepared_experiences = self._rank_hybrid_candidates(
            prepared_experiences, "experience_id", experience_limit
        )
        selected_experience_ids = {
            item["experience_id"] for item in prepared_experiences
        }

        # Layer 2: Segment recall is strictly limited to selected Experiences.
        segment_rows = self.storage.list_segments_by_experience_ids(
            list(selected_experience_ids)
        )
        allowed_segment_ids = {
            str(row["segment_id"]) for row in segment_rows if row.get("segment_id")
        }
        segment_vectors = self._vector_candidates(
            "segment",
            query_embedding,
            max(20, segment_limit * 4),
            allowed_ids=allowed_segment_ids,
            min_allowed=min(segment_limit, len(allowed_segment_ids)),
        )
        prepared_segments: list[dict[str, Any]] = []
        for row in segment_rows:
            segment_id = str(row["segment_id"])
            item = self._prepare_segment(row, set(segment_vectors))
            item["vector_similarity"] = segment_vectors.get(segment_id, 0.0)
            item["keyword_score"] = self._candidate_keyword_score(
                item, keyword_query_text
            )
            item["keyword_recalled"] = item["keyword_score"] > 0.0
            item["_candidate_score"] = self._score_candidate(
                item,
                topic,
                core_entity,
                intent,
                keyword_query_text,
                segment_id in cached_segment_ids,
            )
            prepared_segments.append(item)
        raw_segment_count = len(prepared_segments)
        prepared_segments = self._rank_hybrid_candidates(
            prepared_segments,
            "segment_id",
            segment_limit,
            "experience_id",
        )
        selected_segment_ids = {item["segment_id"] for item in prepared_segments}

        # Layer 3: QA recall is strictly limited to selected Segments.
        qa_rows = self.storage.list_qas_by_segment_ids(list(selected_segment_ids))
        allowed_qa_ids = {
            str(row["qa_id"]) for row in qa_rows if row.get("qa_id")
        }
        qa_vectors = self._vector_candidates(
            "qa",
            query_embedding,
            max(20, qa_limit * 4),
            allowed_ids=allowed_qa_ids,
            min_allowed=min(qa_limit, len(allowed_qa_ids)),
        )
        prepared_qas: list[dict[str, Any]] = []
        for row in qa_rows:
            qa_id = str(row["qa_id"])
            item = self._prepare_qa(row, set(qa_vectors))
            item["vector_similarity"] = qa_vectors.get(qa_id, 0.0)
            item["keyword_score"] = self._candidate_keyword_score(
                item, keyword_query_text
            )
            item["keyword_recalled"] = item["keyword_score"] > 0.0
            item["_candidate_score"] = self._score_candidate(
                item, topic, core_entity, intent, keyword_query_text
            )
            prepared_qas.append(item)
        raw_qa_count = len(prepared_qas)
        prepared_qas = self._rank_hybrid_candidates(
            prepared_qas, "qa_id", qa_limit, "segment_id"
        )

        return {
            "experiences": prepared_experiences,
            "segments": prepared_segments,
            "qas": prepared_qas,
            "raw_counts": {
                "experience": raw_experience_count,
                "segment": raw_segment_count,
                "qa": raw_qa_count,
            },
            "vector_counts": {
                "experience": len(experience_vectors),
                "segment": len(segment_vectors),
                "qa": len(qa_vectors),
            },
            "keyword_counts": {
                "experience": sum(
                    bool(item.get("keyword_recalled"))
                    for item in prepared_experiences
                ),
                "segment": sum(
                    bool(item.get("keyword_recalled")) for item in prepared_segments
                ),
                "qa": sum(bool(item.get("keyword_recalled")) for item in prepared_qas),
            },
        }

    def _build_prompt_candidate_tree(
        self, candidate_data: dict[str, Any]
    ) -> list[dict[str, Any]]:
        segments_by_experience: dict[str, list[dict[str, Any]]] = {}
        qas_by_segment: dict[str, list[dict[str, Any]]] = {}
        for qa in candidate_data["qas"]:
            qas_by_segment.setdefault(str(qa["segment_id"]), []).append(qa)
        for segment in candidate_data["segments"]:
            segments_by_experience.setdefault(
                str(segment["experience_id"]), []
            ).append(segment)

        tree: list[dict[str, Any]] = []
        for experience in candidate_data["experiences"]:
            experience_node = {
                "id": experience["experience_id"],
                "topic": experience.get("topic", ""),
                "core_entity": experience.get("core_entity", ""),
                "intents": experience.get("intents", []),
                "summary": self._truncate_for_prompt(experience.get("summary"), 500),
                "state": self._truncate_for_prompt(
                    json.dumps(experience.get("state") or {}, ensure_ascii=False), 300
                ),
                "vector_similarity": round(
                    float(experience.get("vector_similarity") or 0.0), 4
                ),
                "keyword_score": round(
                    float(experience.get("keyword_score") or 0.0), 4
                ),
                "local_score": round(
                    float(experience.get("_candidate_score") or 0.0), 4
                ),
                "segments": [],
            }
            for segment in segments_by_experience.get(
                str(experience["experience_id"]), []
            ):
                segment_node = {
                    "id": segment["segment_id"],
                    "intent": segment.get("intent", ""),
                    "summary": self._truncate_for_prompt(segment.get("summary"), 350),
                    "vector_similarity": round(
                        float(segment.get("vector_similarity") or 0.0), 4
                    ),
                    "keyword_score": round(
                        float(segment.get("keyword_score") or 0.0), 4
                    ),
                    "local_score": round(
                        float(segment.get("_candidate_score") or 0.0), 4
                    ),
                    "qas": [],
                }
                for qa in qas_by_segment.get(str(segment["segment_id"]), []):
                    segment_node["qas"].append({
                        "id": qa["qa_id"],
                        "timestamp": qa.get("timestamp", ""),
                        "user_input": self._truncate_for_prompt(
                            qa.get("user_input"), 320
                        ),
                        "assistant_output": self._truncate_for_prompt(
                            qa.get("assistant_output"), 240
                        ),
                        "entities": qa.get("entities", []),
                        "confidence": qa.get("confidence", 0.0),
                        "vector_similarity": round(
                            float(qa.get("vector_similarity") or 0.0), 4
                        ),
                        "keyword_score": round(
                            float(qa.get("keyword_score") or 0.0), 4
                        ),
                        "local_score": round(
                            float(qa.get("_candidate_score") or 0.0), 4
                        ),
                    })
                experience_node["segments"].append(segment_node)
            tree.append(experience_node)
        return tree

    def _public_candidate(self, item: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in item.items()
            if not key.startswith("_") and key != "descendant_similarity"
        }

    def _apply_selection_metadata(
        self, item: dict[str, Any], selection: dict[str, Any]
    ) -> dict[str, Any]:
        result = self._public_candidate(item)
        result["score"] = _parse_score(selection.get("score"))
        reason = str(selection.get("reason") or "").strip()
        if reason:
            result["llm_reason"] = reason
        return result

    def _hydrate_hierarchical_selection(
        self,
        selection: dict[str, Any],
        candidate_data: dict[str, Any],
        top_experience: int,
        top_segment: int,
        top_qa: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        experience_by_id = {
            str(item["experience_id"]): item for item in candidate_data["experiences"]
        }
        segment_by_id = {
            str(item["segment_id"]): item for item in candidate_data["segments"]
        }
        qa_by_id = {str(item["qa_id"]): item for item in candidate_data["qas"]}

        experiences: list[dict[str, Any]] = []
        segments: list[dict[str, Any]] = []
        qas: list[dict[str, Any]] = []
        seen_experiences: set[str] = set()
        seen_segments: set[str] = set()
        seen_qas: set[str] = set()
        experience_selections = (
            selection.get("experiences") if isinstance(selection, dict) else []
        )
        if not isinstance(experience_selections, list):
            return experiences, segments, qas

        for experience_selection in experience_selections:
            if not isinstance(experience_selection, dict):
                continue
            experience_id = str(experience_selection.get("id") or "")
            if (
                experience_id in seen_experiences
                or experience_id not in experience_by_id
                or len(experiences) >= top_experience
            ):
                continue
            experiences.append(self._apply_selection_metadata(
                experience_by_id[experience_id], experience_selection
            ))
            seen_experiences.add(experience_id)

            for segment_selection in experience_selection.get("segments") or []:
                if not isinstance(segment_selection, dict):
                    continue
                segment_id = str(segment_selection.get("id") or "")
                segment = segment_by_id.get(segment_id)
                if (
                    segment is None
                    or str(segment.get("experience_id")) != experience_id
                    or segment_id in seen_segments
                    or len(segments) >= top_segment
                ):
                    continue
                segments.append(self._apply_selection_metadata(
                    segment, segment_selection
                ))
                seen_segments.add(segment_id)

                for qa_selection in segment_selection.get("qas") or []:
                    if not isinstance(qa_selection, dict):
                        continue
                    qa_id = str(qa_selection.get("id") or "")
                    qa = qa_by_id.get(qa_id)
                    if (
                        qa is None
                        or str(qa.get("segment_id")) != segment_id
                        or qa_id in seen_qas
                        or len(qas) >= top_qa
                    ):
                        continue
                    qas.append(self._apply_selection_metadata(qa, qa_selection))
                    seen_qas.add(qa_id)
        return experiences, segments, qas

    def _local_hierarchical_selection(
        self,
        candidate_data: dict[str, Any],
        top_experience: int,
        top_segment: int,
        top_qa: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        experience_items = candidate_data["experiences"][:top_experience]
        experience_ids = {item["experience_id"] for item in experience_items}
        segment_items = [
            item
            for item in candidate_data["segments"]
            if item.get("experience_id") in experience_ids
        ][:top_segment]
        segment_ids = {item["segment_id"] for item in segment_items}
        qa_items = [
            item
            for item in candidate_data["qas"]
            if item.get("segment_id") in segment_ids
        ][:top_qa]

        def fallback_item(item: dict[str, Any]) -> dict[str, Any]:
            result = self._public_candidate(item)
            result["score"] = clamp01(float(item.get("_candidate_score") or 0.0))
            result["llm_reason"] = "deterministic retrieval fallback"
            return result

        return (
            [fallback_item(item) for item in experience_items],
            [fallback_item(item) for item in segment_items],
            [fallback_item(item) for item in qa_items],
        )

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
        for key in ("summary", "state", "user_input", "assistant_output"):
            if key in item:
                candidate[key] = item[key]
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
        logger.info("Experience 候选个数%s  向量检索个数%s 已经被选中的Experience如下%s", len(candidates), len(vector_candidate_ids), ranked)
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
        logger.info("Segment 候选个数%s 向量检索候选个数%s 被挑选的Segment %s", len(candidates), len(vector_candidate_ids), ranked)
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
        candidates.sort(key=lambda item: item["timestamp"])
        prepared = [self._prepare_qa(item, vector_candidate_ids) for item in candidates]
        ranked = self._select_candidates_with_llm("qa", query_text, prepared, "qa_id", limit)
        # ranked.sort(key=lambda item: item["timestamp"])
        logger.info("QA 候选个数%s 向量检索个数%s ", len(candidates), len(vector_candidate_ids))
        return ranked, len(candidates)


class StructuredRetriever(HybridRetriever):
    """Backward-compatible alias."""
