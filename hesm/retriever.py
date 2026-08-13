from __future__ import annotations

import json
import logging
import math
import re
import time
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from .config import get as config_get
from .embedder import TextEmbedder
from .storage import MemoryStorage
from .prompts.topic_memory import (
    experience_retrieval_prompt,
    qa_retrieval_prompt,
    segment_retrieval_prompt,
)
from .vector_store import ChromaVectorStore


logger = logging.getLogger(__name__)

DEFAULT_RETRIEVAL_RERANK_MODEL = "qwen-plus"
DEFAULT_RETRIEVAL_RERANK_BASE_URL = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
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
2. vector_similarity, relation_score, keyword_score, and retrieval_sources are
   complementary recall hints, not final relevance scores.
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
                raise ValueError("Set retrieval.api_key in config/hesm.yaml.")
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

class _BaseHybridRetriever:
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
        retrieval_config: dict[str, Any] | None = None,
        **_: Any,
    ) -> None:
        self.storage = storage
        self.vector_store = vector_store
        self.embedder = embedder
        self.reranker = reranker
        has_explicit_settings = isinstance(retrieval_config, dict)
        settings = retrieval_config or {}

        def setting(key: str, default: Any) -> Any:
            if key in settings and settings[key] is not None:
                return settings[key]
            if has_explicit_settings:
                return default
            return config_get("retrieval", key, default)

        self.max_context_tokens = max(
            1, int(setting("max_context_tokens", 30000))
        )
        self.min_prompt_experiences = max(
            1, int(setting("min_prompt_experiences", 6))
        )
        self.token_encoding = str(setting("token_encoding", "cl100k_base"))
        self._token_encoder: Any | None = None
        self._token_encoder_initialized = False

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
        state_key: str = "default",
        use_cache: bool = True,
    ) -> dict[str, Any]:
        recall_started_at = time.perf_counter()
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
        embedding_started_at = time.perf_counter()
        query_embedding = self._embed_query(query_text)
        embedding_elapsed_ms = round(
            (time.perf_counter() - embedding_started_at) * 1000, 3
        )
        retrieval_cache = self._load_retrieval_cache(state_key) if use_cache else {}

        candidate_collection_started_at = time.perf_counter()
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
        candidate_collection_elapsed_ms = round(
            (time.perf_counter() - candidate_collection_started_at) * 1000, 3
        )

        tree_build_started_at = time.perf_counter()
        candidate_tree = self._build_prompt_candidate_tree(candidate_data)
        tree_build_elapsed_ms = round(
            (time.perf_counter() - tree_build_started_at) * 1000, 3
        )
        original_candidate_tree = deepcopy(candidate_tree)

        tree_pruning_started_at = time.perf_counter()
        candidate_tree, tree_fit_debug = self._fit_candidate_tree_to_context(
            candidate_tree
        )
        tree_pruning_elapsed_ms = round(
            (time.perf_counter() - tree_pruning_started_at) * 1000, 3
        )

        llm_calls = 0
        selection: dict[str, Any] = {"experiences": []}
        reranking_started_at = time.perf_counter()
        if (
            candidate_tree
            and self.reranker is not None
            and not tree_fit_debug['context_overflow_unresolved']
        ):
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
        elif tree_fit_debug['context_overflow_unresolved']:
            logger.warning(
                'Candidate tree still exceeds max_context_tokens after pruning and '
                'summary compression; skipping LLM rerank'
            )
        reranking_elapsed_ms = round(
            (time.perf_counter() - reranking_started_at) * 1000, 3
        )

        result_processing_started_at = time.perf_counter()
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
        result_processing_elapsed_ms = round(
            (time.perf_counter() - result_processing_started_at) * 1000, 3
        )

        direct_unique_counts = candidate_data["direct_unique_counts"]
        source_counts = candidate_data["source_counts"]
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

        prompt_counts = {
            "experience": len(candidate_tree),
            "segment": sum(len(item.get("segments") or []) for item in candidate_tree),
            "qa": sum(
                len(segment.get("qas") or [])
                for item in candidate_tree
                for segment in item.get("segments") or []
            ),
        }
        debug = {
            # This value is finalized immediately before return so it includes
            # context rendering and retrieval-cache persistence as well.
            "total_retrieval_ms": 0.0,
            "embedding": {
                "elapsed_ms": embedding_elapsed_ms,
                "vector_dimensions": len(query_embedding or []),
            },
            "candidate_collection": {
                "elapsed_ms": candidate_collection_elapsed_ms,
                **candidate_data["timing"],
            },
            "candidate_tree_build": {
                "elapsed_ms": tree_build_elapsed_ms,
            },
            "tree_pruning": {
                "elapsed_ms": tree_pruning_elapsed_ms,
                "counts_before": self._tree_counts(original_candidate_tree),
                "counts_after": self._tree_counts(candidate_tree),
            },
            "reranking": {
                "elapsed_ms": reranking_elapsed_ms,
                "attempted": bool(llm_calls),
                "llm_calls": llm_calls,
                "skipped_for_context_overflow": tree_fit_debug[
                    "context_overflow_unresolved"
                ],
            },
            "result_processing": {
                "elapsed_ms": result_processing_elapsed_ms,
            },
            "candidate_trees": {
                "original": original_candidate_tree,
                "pruned": deepcopy(candidate_tree),
            },
            "constraints": {
                "requested_top_k": {
                    "experience": top_experience,
                    "segment": top_segment,
                    "qa": top_qa,
                },
                "candidate_limits": candidate_data["limits"],
                "context": {
                    "max_context_tokens": self.max_context_tokens,
                    "min_prompt_experiences": self.min_prompt_experiences,
                    "effective_min_prompt_experiences": max(
                        1,
                        min(
                            self.min_prompt_experiences,
                            len(original_candidate_tree),
                        ),
                    ),
                },
                "cache": {
                    "enabled": use_cache,
                    "state_key": state_key,
                },
            },
            # # strategy：本次召回采用的融合与层级处理策略。
            # "strategy": "hierarchical_equal_weight_rrf",
            # # source_candidates：各层每种数据源在去重前返回的原始条数。
            # "source_candidates": source_counts,
            # # pipeline_counts：候选在召回、补祖先、质量裁剪、上下文裁剪各阶段的数量。
            # "pipeline_counts": {
            #     # direct_unique：各数据源合并去重后的直接召回数量。
            #     "direct_unique": direct_unique_counts,
            #     # direct_selected：每层第一次按融合分和候选上限筛选后的数量。
            #     "direct_selected": candidate_data["direct_selected_counts"],
            #     # ancestors_added：为保持树完整而补入的 Experience/Segment 数量。
            #     "ancestors_added": candidate_data["completion_counts"],
            #     # after_completion：补齐祖先后、质量上限裁剪前的三层数量。
            #     "after_completion": candidate_data["counts_after_completion"],
            #     # after_quality_pruning：按路径质量和分层上限裁剪后的数量。
            #     "after_quality_pruning": candidate_data[
            #         "counts_after_quality_pruning"
            #     ],
            #     # submitted_to_llm：真正放入最终 candidate_tree 的三层数量。
            #     "submitted_to_llm": prompt_counts,
            #     # final_selected：LLM 或本地 fallback 最终返回的三层数量。
            #     "final_selected": {
            #         "experience": len(experiences),
            #         "segment": len(segments),
            #         "qa": len(qas),
            #     },
            # },
            # # candidate_limits：补祖先后的质量裁剪上限，不是最终 LLM top-k。
            # "candidate_limits": candidate_data["limits"],
            # # context_budget：只针对 candidate_tree 的 token 预算与裁剪结果。
            # "context_budget": {
            #     "limit_tokens": tree_fit_debug["max_context_tokens"],
            #     "tokens_before": tree_fit_debug["candidate_tokens_before_pruning"],
            #     "tokens_after": tree_fit_debug[
            #         "candidate_tokens_after_summary_compression"
            #     ],
            #     "removed_experience_ids": tree_fit_debug[
            #         "removed_experience_ids"
            #     ],
            #     "summary_compressed": tree_fit_debug["summary_compressed"],
            #     "overflow_unresolved": tree_fit_debug[
            #         "context_overflow_unresolved"
            #     ],
            # },
            # # cache_hits：当前查询是否复用了 Experience/Segment 检索缓存。
            # "cache_hits": {
            #     "experience": experience_cache_hit,
            #     "segment": segment_cache_hit,
            # },
            # # llm_calls：本次层级 rerank 实际调用 LLM 的次数。
            # "llm_calls": llm_calls,
        }
        logger.info(
            "检索结果：experience=%s/%s segment=%s/%s qa=%s/%s vectors=%s/%s/%s cache=%s/%s",
            len(experiences), direct_unique_counts["experience"],
            len(segments), direct_unique_counts["segment"],
            len(qas), direct_unique_counts["qa"],
            source_counts["experience"]["vector"],
            source_counts["segment"]["vector"],
            source_counts["qa"]["vector"],
            experience_cache_hit, segment_cache_hit,
        )
        context_text = build_context_text(experiences, segments, qas)
        if use_cache:
            self._store_retrieval_cache(
                state_key, topic, core_entity, intent, query_entities, experiences, segments
            )
        result = {
            "query": {
                "topic": topic,
                "core_entity": core_entity,
                "intent": intent or "",
                "entities": query_entities,
            },
            "experiences": experiences,
            "segments": segments,
            "qas": qas,
            "context_text": context_text,
            "debug": debug,
            # "results": [{"qa": qa, "score": qa["score"]} for qa in qas],
        }
        debug["total_retrieval_ms"] = round(
            (time.perf_counter() - recall_started_at) * 1000, 3
        )
        return result

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

    def _lexical_units(self, value: Any) -> set[str]:
        text = str(value or "").lower()
        words = set("".join(char if char.isalnum() else " " for char in text).split())
        words = {word for word in words if len(word) > 1}
        # 补充轻量英文词形归一化，使 move/moved、year/years 能互相命中。
        normalized_words = set(words)
        for word in words:
            if word.endswith("ies") and len(word) > 4:
                normalized_words.add(word[:-3] + "y")
            elif word.endswith("s") and len(word) > 3:
                normalized_words.add(word[:-1])
            if word.endswith("ed") and len(word) > 4:
                normalized_words.add(word[:-2])
                normalized_words.add(word[:-1])
            if word.endswith("ing") and len(word) > 5:
                normalized_words.add(word[:-3])
                normalized_words.add(word[:-3] + "e")
        words = normalized_words
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
        item_intents = item.get("intents") or [item.get("intent")]
        normalized_intents = {str(value or "").strip() for value in item_intents}
        structure_matches = [
            float(
                str(item.get("topic") or "").strip()
                == str(topic or "").strip()
            ),
            float(
                str(item.get("core_entity") or "").strip()
                == str(core_entity or "").strip()
            ),
        ]
        if str(intent or "").strip():
            structure_matches.append(
                float(str(intent or "").strip() in normalized_intents)
            )
        structure_score = sum(structure_matches) / len(structure_matches)
        item["_structure_score"] = clamp01(structure_score)
        item["_vector_channel_score"] = (
            clamp01(0.75 * vector_similarity + 0.25 * structure_score)
            if vector_similarity > 0.0
            else 0.0
        )
        item["_keyword_channel_score"] = (
            clamp01(0.75 * keyword_score + 0.25 * structure_score)
            if keyword_score > 0.0
            else 0.0
        )
        # 两个召回通道各占 35%，topic/core_entity/intent 等权共享 30%。
        score = (
            vector_similarity * 0.35
            + keyword_score * 0.35
            + structure_score * 0.3
        )
        if cached:
            score += 0.03
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
        """Fuse equal-weight vector/keyword ranks and keep quotas for both channels."""
        del parent_field  # 父级多样性改在完整树裁剪时处理，避免候选分支膨胀。
        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set()

        def add(item: dict[str, Any]) -> None:
            memory_id = str(item.get(id_field) or "")
            if memory_id and memory_id not in selected_ids and len(selected) < limit:
                selected.append(item)
                selected_ids.add(memory_id)

        vector_ranked = sorted(
            (
                item
                for item in items
                if float(item.get("_vector_channel_score") or 0.0) > 0.0
            ),
            key=lambda item: float(item.get("_vector_channel_score") or 0.0),
            reverse=True,
        )
        keyword_ranked = sorted(
            (
                item
                for item in items
                if float(item.get("_keyword_channel_score") or 0.0) > 0.0
            ),
            key=lambda item: float(item.get("_keyword_channel_score") or 0.0),
            reverse=True,
        )
        vector_ranks = {
            str(item.get(id_field) or ""): rank
            for rank, item in enumerate(vector_ranked, 1)
        }
        keyword_ranks = {
            str(item.get(id_field) or ""): rank
            for rank, item in enumerate(keyword_ranked, 1)
        }
        rrf_constant = 60.0
        for item in items:
            memory_id = str(item.get(id_field) or "")
            vector_rank = vector_ranks.get(memory_id)
            keyword_rank = keyword_ranks.get(memory_id)
            vector_rrf = (
                (rrf_constant + 1.0) / (rrf_constant + vector_rank)
                if vector_rank is not None
                else 0.0
            )
            keyword_rrf = (
                (rrf_constant + 1.0) / (rrf_constant + keyword_rank)
                if keyword_rank is not None
                else 0.0
            )
            fusion_score = 0.5 * vector_rrf + 0.5 * keyword_rrf
            item["_fusion_score"] = clamp01(fusion_score)
            item["_candidate_score"] = clamp01(
                0.75 * fusion_score
                + 0.25 * float(item.get("_candidate_score") or 0.0)
            )

        combined = sorted(
            items,
            key=lambda item: (
                float(item.get("_candidate_score") or 0.0),
                str(item.get("updated_at") or item.get("timestamp") or ""),
            ),
            reverse=True,
        )
        channel_quota = max(1, limit // 3)
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

    def _structured_experience_rows(
        self,
        topic: str,
        core_entity: str,
        intent: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Recall Experiences by equal-weight topic/core_entity/intent matches."""
        conditions: list[tuple[str, Any]] = []
        if str(topic or "").strip():
            conditions.append(("topic = ?", str(topic).strip()))
        if str(core_entity or "").strip():
            conditions.append(("core_entity = ?", str(core_entity).strip()))
        if str(intent or "").strip():
            conditions.append(("intents_link_json LIKE ?", f'%"{str(intent).strip()}"%'))
        if not conditions:
            return []
        score_expression = " + ".join(
            f"CASE WHEN {condition} THEN 1 ELSE 0 END"
            for condition, _ in conditions
        )
        where_expression = " OR ".join(condition for condition, _ in conditions)
        values = [value for _, value in conditions]
        rows = self.storage.connection.execute(
            f"""
            SELECT experience_id, ({score_expression}) AS structure_score
            FROM experience_memory
            WHERE {where_expression}
            ORDER BY structure_score DESC, updated_at DESC, created_at DESC
            LIMIT ?
            """,
            (*values, *values, limit),
        ).fetchall()
        return self.storage.get_experiences(
            [str(row["experience_id"]) for row in rows]
        )

    def _keyword_experience_rows(
        self,
        topic: str,
        core_entity: str,
        intent: str | None,
        query_text: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        # topic/core_entity/intent 由结构化通道处理，内容关键词通道只检索
        # entities 和原始问题，避免一个常见人物名召回该人物的全部记忆。
        terms = self._keyword_terms(
            query_text,
            excluded_values=(topic, core_entity, intent or ""),
        )
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
        self,
        item: dict[str, Any],
        query_text: str,
        topic: str = "",
        core_entity: str = "",
        intent: str | None = None,
    ) -> float:
        """Content-keyword relevance; structure fields are scored separately."""
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
        content_terms = self._keyword_terms(
            query_text,
            excluded_values=(topic, core_entity, intent or ""),
        )
        if not content_terms:
            return 0.0
        return clamp01(
            self._lexical_overlap(" ".join(content_terms), candidate_text)
        )

    def _keyword_terms(
        self,
        query_text: str,
        limit: int = 12,
        excluded_values: tuple[Any, ...] = (),
    ) -> list[str]:
        ignored = {
            '主题', '核心实体', '用户意图', '相关实体', '问题',
            'topic', 'core', 'entity', 'intent', 'query', 'question',
            'user', 'related', 'what', 'when', 'where', 'who', 'why', 'how',
            'did', 'does', 'do', 'is', 'are', 'was', 'were', 'the', 'a', 'an',
            'of', 'to', 'from', 'in', 'on', 'for', 'with', 'about', 'this',
            'that', 'it',
        }
        for value in excluded_values:
            ignored.update(self._lexical_units(value))
        return sorted(
            (
                term
                for term in self._lexical_units(query_text)
                if term not in ignored and 1 < len(term) <= 32
            ),
            key=lambda value: (len(value), value),
            reverse=True,
        )[:limit]

    def _structured_child_rows(
        self,
        memory_type: str,
        topic: str,
        core_entity: str,
        intent: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Recall child rows by equal-weight topic/core_entity/intent matches."""
        specs = {
            "segment": (
                "segment_memory", "segment_id", "status != 'deleted'",
                self.storage.get_segments,
            ),
            "qa": (
                "qa_memory", "qa_id", "status = 'active'",
                self.storage.get_qas,
            ),
        }
        table, id_field, status_filter, loader = specs[memory_type]
        field_values = [
            ("topic", str(topic or "").strip()),
            ("core_entity", str(core_entity or "").strip()),
            ("intent", str(intent or "").strip()),
        ]
        field_values = [(field, value) for field, value in field_values if value]
        if not field_values:
            return []
        score_expression = " + ".join(
            f"CASE WHEN {field} = ? THEN 1 ELSE 0 END"
            for field, _ in field_values
        )
        where_expression = " OR ".join(
            f"{field} = ?" for field, _ in field_values
        )
        values = [value for _, value in field_values]
        order_field = "timestamp" if memory_type == "qa" else "updated_at"
        rows = self.storage.connection.execute(
            f"""
            SELECT {id_field}, ({score_expression}) AS structure_score
            FROM {table}
            WHERE {status_filter} AND ({where_expression})
            ORDER BY structure_score DESC, {order_field} DESC
            LIMIT ?
            """,
            (*values, *values, limit),
        ).fetchall()
        return loader([str(row[id_field]) for row in rows])

    def _keyword_segment_rows(
        self,
        query_text: str,
        limit: int,
        topic: str = "",
        core_entity: str = "",
        intent: str | None = None,
    ) -> list[dict[str, Any]]:
        terms = self._keyword_terms(
            query_text,
            excluded_values=(topic, core_entity, intent or ""),
        )
        if not terms:
            return []
        searchable = (
            '''COALESCE(topic, '') || ' ' || COALESCE(core_entity, '') || ' ' || '''
            '''COALESCE(intent, '') || ' ' || COALESCE(summary, '')'''
        )
        where_clause = ' OR '.join(f'({searchable}) LIKE ?' for _ in terms)
        rows = self.storage.connection.execute(
            f'''
            SELECT segment_id
            FROM segment_memory
            WHERE status != 'deleted' AND ({where_clause})
            ORDER BY updated_at DESC, created_at DESC
            LIMIT ?
            ''',
            (*[f'%{term}%' for term in terms], limit),
        ).fetchall()
        return self.storage.get_segments(
            [str(row['segment_id']) for row in rows]
        )

    def _keyword_qa_rows(
        self,
        query_text: str,
        limit: int,
        topic: str = "",
        core_entity: str = "",
        intent: str | None = None,
    ) -> list[dict[str, Any]]:
        terms = self._keyword_terms(
            query_text,
            excluded_values=(topic, core_entity, intent or ""),
        )
        if not terms:
            return []
        searchable = (
            '''COALESCE(topic, '') || ' ' || COALESCE(core_entity, '') || ' ' || '''
            '''COALESCE(intent, '') || ' ' || COALESCE(user_input, '') || ' ' || '''
            '''COALESCE(assistant_output, '') || ' ' || COALESCE(entities_json, '')'''
        )
        where_clause = ' OR '.join(f'({searchable}) LIKE ?' for _ in terms)
        rows = self.storage.connection.execute(
            f'''
            SELECT qa_id
            FROM qa_memory
            WHERE status = 'active' AND ({where_clause})
            ORDER BY timestamp DESC
            LIMIT ?
            ''',
            (*[f'%{term}%' for term in terms], limit),
        ).fetchall()
        return self.storage.get_qas([str(row['qa_id']) for row in rows])

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
        collection_started_at = time.perf_counter()
        timing_seconds: dict[str, Any] = {
            "sql_recall": {"total": 0.0, "by_layer": {}},
            "chroma_vector_recall": {"total": 0.0, "by_layer": {}},
            "ranking_processing": {"total": 0.0, "by_layer": {}},
        }

        def record_elapsed(
            category: str,
            layer: str,
            operation: str,
            elapsed: float,
        ) -> None:
            bucket = timing_seconds[category]
            bucket["total"] += elapsed
            layer_bucket = bucket["by_layer"].setdefault(layer, {})
            layer_bucket[operation] = layer_bucket.get(operation, 0.0) + elapsed

        def measure(
            category: str,
            layer: str,
            operation: str,
            callback: Any,
        ) -> Any:
            started_at = time.perf_counter()
            value = callback()
            elapsed = time.perf_counter() - started_at
            record_elapsed(category, layer, operation, elapsed)
            return value

        experience_limit = max(
            self.min_prompt_experiences,
            top_experience,
            int(config_get("retrieval", "experience_candidate_limit", 8)),
            top_experience * 3,
        )
        segment_limit = max(
            top_segment,
            int(config_get("retrieval", "segment_candidate_limit", 20)),
            top_segment * 4,
        )
        qa_limit = max(
            top_qa,
            int(config_get("retrieval", "qa_candidate_limit", 32)),
            top_qa * 4,
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

        # Layer 1: recall Experiences independently from child-layer candidates.
        experience_vectors = measure(
            "chroma_vector_recall",
            "experience",
            "query",
            lambda: self._vector_candidates(
                "experience", query_embedding, max(12, experience_limit * 2)
            ),
        )
        experience_rows: dict[str, dict[str, Any]] = {}

        def add_experience(row: dict[str, Any] | None) -> None:
            if row and row.get("experience_id"):
                experience_rows[str(row["experience_id"])] = row

        structured_limit = max(experience_limit * 2, 12)
        exact_experience_rows = measure(
            "sql_recall",
            "experience",
            "exact",
            lambda: self.storage.find_experiences(
                topic, core_entity, structured_limit
            ),
        )
        topic_experience_rows = measure(
            "sql_recall",
            "experience",
            "topic",
            lambda: self.storage.list_experiences_by_topic(
                topic, structured_limit
            ),
        )
        structured_experience_rows = measure(
            "sql_recall",
            "experience",
            "structured",
            lambda: self._structured_experience_rows(
                topic, core_entity, intent, structured_limit
            ),
        )
        keyword_experience_rows = measure(
            "sql_recall",
            "experience",
            "keyword",
            lambda: self._keyword_experience_rows(
                topic,
                core_entity,
                intent,
                keyword_query_text,
                structured_limit,
            ),
        )
        vector_experience_rows = measure(
            "sql_recall",
            "experience",
            "hydrate_vector_ids",
            lambda: self.storage.get_experiences(list(experience_vectors)),
        )
        cached_experience_rows = measure(
            "sql_recall",
            "experience",
            "hydrate_cache_ids",
            lambda: self.storage.get_experiences(list(cached_experience_ids)),
        )
        for row in exact_experience_rows:
            add_experience(row)
        for row in topic_experience_rows:
            add_experience(row)
        for row in structured_experience_rows:
            add_experience(row)
        for row in keyword_experience_rows:
            add_experience(row)
        for row in vector_experience_rows:
            add_experience(row)
        for row in cached_experience_rows:
            add_experience(row)

        experience_ranking_started_at = time.perf_counter()
        prepared_experiences: list[dict[str, Any]] = []
        for experience_id, row in experience_rows.items():
            item = self._prepare_experience(row, set(experience_vectors))
            item["vector_similarity"] = experience_vectors.get(experience_id, 0.0)
            item["keyword_score"] = self._candidate_keyword_score(
                item, keyword_query_text, topic, core_entity, intent
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
        record_elapsed(
            "ranking_processing",
            "experience",
            "score_deduplicate_and_rank",
            time.perf_counter() - experience_ranking_started_at,
        )
        for item in prepared_experiences:
            item["_direct_recalled"] = True
        logger.info(
            "Experience candidate sources: exact=%s topic=%s structured=%s keyword=%s "
            "vector=%s cache=%s deduplicated=%s selected=%s",
            len(exact_experience_rows),
            len(topic_experience_rows),
            len(structured_experience_rows),
            len(keyword_experience_rows),
            len(vector_experience_rows),
            len(cached_experience_rows),
            raw_experience_count,
            len(prepared_experiences),
        )

        # Layer 2: recall Segments globally. Parent Experiences are completed later.
        segment_vectors = measure(
            "chroma_vector_recall",
            "segment",
            "query",
            lambda: self._vector_candidates(
                "segment",
                query_embedding,
                max(20, segment_limit * 2),
            ),
        )
        segment_rows_by_id: dict[str, dict[str, Any]] = {}

        def add_segment(row: dict[str, Any] | None) -> None:
            if (
                row
                and row.get("segment_id")
                and row.get("status") != "deleted"
            ):
                segment_rows_by_id[str(row["segment_id"])] = row

        structured_segment_rows = measure(
            "sql_recall",
            "segment",
            "structured",
            lambda: self._structured_child_rows(
                "segment", topic, core_entity, intent, segment_limit * 2
            ),
        )
        keyword_segment_rows = measure(
            "sql_recall",
            "segment",
            "keyword",
            lambda: self._keyword_segment_rows(
                keyword_query_text,
                segment_limit * 2,
                topic,
                core_entity,
                intent,
            ),
        )
        vector_segment_rows = measure(
            "sql_recall",
            "segment",
            "hydrate_vector_ids",
            lambda: self.storage.get_segments(list(segment_vectors)),
        )
        cached_segment_rows = measure(
            "sql_recall",
            "segment",
            "hydrate_cache_ids",
            lambda: self.storage.get_segments(list(cached_segment_ids)),
        )
        for row in structured_segment_rows:
            add_segment(row)
        for row in keyword_segment_rows:
            add_segment(row)
        for row in vector_segment_rows:
            add_segment(row)
        for row in cached_segment_rows:
            add_segment(row)

        segment_ranking_started_at = time.perf_counter()
        prepared_segments: list[dict[str, Any]] = []
        for row in segment_rows_by_id.values():
            segment_id = str(row["segment_id"])
            item = self._prepare_segment(row, set(segment_vectors))
            item["vector_similarity"] = segment_vectors.get(segment_id, 0.0)
            item["keyword_score"] = self._candidate_keyword_score(
                item, keyword_query_text, topic, core_entity, intent
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
            item["_direct_recalled"] = True
            prepared_segments.append(item)
        raw_segment_count = len(prepared_segments)
        prepared_segments = self._rank_hybrid_candidates(
            prepared_segments,
            "segment_id",
            segment_limit,
            "experience_id",
        )
        record_elapsed(
            "ranking_processing",
            "segment",
            "score_deduplicate_and_rank",
            time.perf_counter() - segment_ranking_started_at,
        )
        logger.info(
            "Segment candidate sources: structured=%s keyword=%s vector=%s cache=%s "
            "deduplicated=%s selected=%s",
            len(structured_segment_rows),
            len(keyword_segment_rows),
            len(vector_segment_rows),
            len(cached_segment_rows),
            raw_segment_count,
            len(prepared_segments),
        )
        # Layer 3: recall QAs globally. Segment and Experience ancestors are
        # completed after the direct candidates have been capped.
        qa_vectors = measure(
            "chroma_vector_recall",
            "qa",
            "query",
            lambda: self._vector_candidates(
                "qa",
                query_embedding,
                max(20, qa_limit * 2),
            ),
        )
        qa_rows_by_id: dict[str, dict[str, Any]] = {}

        def add_qa(row: dict[str, Any] | None) -> None:
            if row and row.get("qa_id") and row.get("status") == "active":
                qa_rows_by_id[str(row["qa_id"])] = row

        structured_qa_rows = measure(
            "sql_recall",
            "qa",
            "structured",
            lambda: self._structured_child_rows(
                "qa", topic, core_entity, intent, qa_limit * 2
            ),
        )
        keyword_qa_rows = measure(
            "sql_recall",
            "qa",
            "keyword",
            lambda: self._keyword_qa_rows(
                keyword_query_text,
                qa_limit * 2,
                topic,
                core_entity,
                intent,
            ),
        )
        vector_qa_rows = measure(
            "sql_recall",
            "qa",
            "hydrate_vector_ids",
            lambda: self.storage.get_qas(list(qa_vectors)),
        )
        for row in structured_qa_rows:
            add_qa(row)
        for row in keyword_qa_rows:
            add_qa(row)
        for row in vector_qa_rows:
            add_qa(row)

        qa_ranking_started_at = time.perf_counter()
        prepared_qas: list[dict[str, Any]] = []
        for row in qa_rows_by_id.values():
            qa_id = str(row["qa_id"])
            item = self._prepare_qa(row, set(qa_vectors))
            item["vector_similarity"] = qa_vectors.get(qa_id, 0.0)
            item["keyword_score"] = self._candidate_keyword_score(
                item, keyword_query_text, topic, core_entity, intent
            )
            item["keyword_recalled"] = item["keyword_score"] > 0.0
            item["_candidate_score"] = self._score_candidate(
                item, topic, core_entity, intent, keyword_query_text
            )
            item["_direct_recalled"] = True
            prepared_qas.append(item)
        raw_qa_count = len(prepared_qas)
        prepared_qas = self._rank_hybrid_candidates(
            prepared_qas, "qa_id", qa_limit, "segment_id"
        )
        record_elapsed(
            "ranking_processing",
            "qa",
            "score_deduplicate_and_rank",
            time.perf_counter() - qa_ranking_started_at,
        )
        logger.info(
            "QA candidate sources: structured=%s keyword=%s vector=%s "
            "deduplicated=%s selected=%s",
            len(structured_qa_rows),
            len(keyword_qa_rows),
            len(vector_qa_rows),
            raw_qa_count,
            len(prepared_qas),
        )

        ancestry_started_at = time.perf_counter()
        ancestry_sql_before = timing_seconds["sql_recall"]["total"]
        prepared_experiences, prepared_segments, prepared_qas = (
            self._complete_candidate_ancestry(
                experiences=prepared_experiences,
                segments=prepared_segments,
                qas=prepared_qas,
                experience_vectors=experience_vectors,
                segment_vectors=segment_vectors,
                cached_experience_ids=cached_experience_ids,
                cached_segment_ids=cached_segment_ids,
                topic=topic,
                core_entity=core_entity,
                intent=intent,
                keyword_query_text=keyword_query_text,
                timing_recorder=measure,
            )
        )
        ancestry_elapsed = time.perf_counter() - ancestry_started_at
        ancestry_sql_elapsed = (
            timing_seconds["sql_recall"]["total"] - ancestry_sql_before
        )
        record_elapsed(
            "ranking_processing",
            "ancestry",
            "complete_score_and_sort",
            max(0.0, ancestry_elapsed - ancestry_sql_elapsed),
        )
        counts_after_completion = {
            "experience": len(prepared_experiences),
            "segment": len(prepared_segments),
            "qa": len(prepared_qas),
        }
        completion_counts = {
            "experience": sum(
                bool(item.get("_ancestor_completed"))
                for item in prepared_experiences
            ),
            "segment": sum(
                bool(item.get("_ancestor_completed"))
                for item in prepared_segments
            ),
        }
        quality_pruning_started_at = time.perf_counter()
        prepared_experiences, prepared_segments, prepared_qas = (
            self._prune_completed_candidates(
                prepared_experiences,
                prepared_segments,
                prepared_qas,
                experience_limit,
                segment_limit,
                qa_limit,
            )
        )
        record_elapsed(
            "ranking_processing",
            "quality_pruning",
            "path_score_sort_and_limit",
            time.perf_counter() - quality_pruning_started_at,
        )
        counts_after_quality_pruning = {
            "experience": len(prepared_experiences),
            "segment": len(prepared_segments),
            "qa": len(prepared_qas),
        }

        def milliseconds(value: float) -> float:
            return round(value * 1000, 3)

        timing = {
            category: {
                "total_ms": milliseconds(bucket["total"]),
                "by_layer": {
                    layer: {
                        operation: milliseconds(elapsed)
                        for operation, elapsed in operations.items()
                    }
                    for layer, operations in bucket["by_layer"].items()
                },
            }
            for category, bucket in timing_seconds.items()
        }
        instrumented_elapsed = time.perf_counter() - collection_started_at
        timing["instrumented_elapsed_ms"] = milliseconds(instrumented_elapsed)
        timing["other_processing_ms"] = milliseconds(max(
            0.0,
            instrumented_elapsed - sum(
                bucket["total"] for bucket in timing_seconds.values()
            ),
        ))

        return {
            "experiences": prepared_experiences,
            "segments": prepared_segments,
            "qas": prepared_qas,
            "direct_unique_counts": {
                "experience": raw_experience_count,
                "segment": raw_segment_count,
                "qa": raw_qa_count,
            },
            "direct_selected_counts": {
                "experience": min(raw_experience_count, experience_limit),
                "segment": min(raw_segment_count, segment_limit),
                "qa": min(raw_qa_count, qa_limit),
            },
            "source_counts": {
                "experience": {
                    "exact": len(exact_experience_rows),
                    "topic": len(topic_experience_rows),
                    "structured": len(structured_experience_rows),
                    "keyword": len(keyword_experience_rows),
                    "vector": len(vector_experience_rows),
                    "cache": len(cached_experience_rows),
                },
                "segment": {
                    "structured": len(structured_segment_rows),
                    "keyword": len(keyword_segment_rows),
                    "vector": len(vector_segment_rows),
                    "cache": len(cached_segment_rows),
                },
                "qa": {
                    "structured": len(structured_qa_rows),
                    "keyword": len(keyword_qa_rows),
                    "vector": len(vector_qa_rows),
                },
            },
            "completion_counts": completion_counts,
            "counts_after_completion": counts_after_completion,
            "counts_after_quality_pruning": counts_after_quality_pruning,
            "limits": {
                "experience": experience_limit,
                "segment": segment_limit,
                "qa": qa_limit,
            },
            "timing": timing,
        }

    def _complete_candidate_ancestry(
        self,
        *,
        experiences: list[dict[str, Any]],
        segments: list[dict[str, Any]],
        qas: list[dict[str, Any]],
        experience_vectors: dict[str, float],
        segment_vectors: dict[str, float],
        cached_experience_ids: set[str],
        cached_segment_ids: set[str],
        topic: str,
        core_entity: str,
        intent: str | None,
        keyword_query_text: str,
        timing_recorder: Any | None = None,
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        '''Complete every recalled QA/Segment path up to its real Experience.'''
        experience_by_id = {
            str(item['experience_id']): item for item in experiences
        }
        segment_by_id = {str(item['segment_id']): item for item in segments}

        missing_segment_ids = {
            str(qa.get('segment_id') or '')
            for qa in qas
            if str(qa.get('segment_id') or '') not in segment_by_id
        }
        missing_segment_ids.discard('')
        if timing_recorder is None:
            missing_segment_rows = self.storage.get_segments(
                list(missing_segment_ids)
            )
        else:
            missing_segment_rows = timing_recorder(
                "sql_recall",
                "ancestry",
                "hydrate_missing_segments",
                lambda: self.storage.get_segments(list(missing_segment_ids)),
            )
        for row in missing_segment_rows:
            if row.get('status') == 'deleted':
                continue
            segment_id = str(row['segment_id'])
            item = self._prepare_segment(row, set(segment_vectors))
            item['vector_similarity'] = segment_vectors.get(segment_id, 0.0)
            item['keyword_score'] = self._candidate_keyword_score(
                item, keyword_query_text, topic, core_entity, intent
            )
            item['keyword_recalled'] = item['keyword_score'] > 0.0
            item['_candidate_score'] = self._score_candidate(
                item,
                topic,
                core_entity,
                intent,
                keyword_query_text,
                segment_id in cached_segment_ids,
            )
            item['_direct_recalled'] = False
            item['_ancestor_completed'] = True
            segment_by_id[segment_id] = item

        missing_experience_ids = {
            str(segment.get('experience_id') or '')
            for segment in segment_by_id.values()
            if str(segment.get('experience_id') or '') not in experience_by_id
        }
        missing_experience_ids.discard('')
        if timing_recorder is None:
            missing_experience_rows = self.storage.get_experiences(
                list(missing_experience_ids)
            )
        else:
            missing_experience_rows = timing_recorder(
                "sql_recall",
                "ancestry",
                "hydrate_missing_experiences",
                lambda: self.storage.get_experiences(
                    list(missing_experience_ids)
                ),
            )
        for row in missing_experience_rows:
            experience_id = str(row['experience_id'])
            item = self._prepare_experience(row, set(experience_vectors))
            item['vector_similarity'] = experience_vectors.get(experience_id, 0.0)
            item['keyword_score'] = self._candidate_keyword_score(
                item, keyword_query_text, topic, core_entity, intent
            )
            item['keyword_recalled'] = item['keyword_score'] > 0.0
            item['_candidate_score'] = self._score_candidate(
                item,
                topic,
                core_entity,
                intent,
                keyword_query_text,
                experience_id in cached_experience_ids,
            )
            item['_direct_recalled'] = False
            item['_ancestor_completed'] = True
            experience_by_id[experience_id] = item

        valid_experience_ids = set(experience_by_id)
        valid_segments = [
            item
            for item in segment_by_id.values()
            if str(item.get('experience_id') or '') in valid_experience_ids
        ]
        valid_segment_ids = {
            str(item['segment_id']) for item in valid_segments
        }
        valid_qas = [
            item
            for item in qas
            if str(item.get('segment_id') or '') in valid_segment_ids
        ]
        return (
            sorted(
                experience_by_id.values(),
                key=lambda item: float(item.get('_candidate_score') or 0.0),
                reverse=True,
            ),
            sorted(
                valid_segments,
                key=lambda item: float(item.get('_candidate_score') or 0.0),
                reverse=True,
            ),
            valid_qas,
        )

    def _prune_completed_candidates(
        self,
        experiences: list[dict[str, Any]],
        segments: list[dict[str, Any]],
        qas: list[dict[str, Any]],
        experience_limit: int,
        segment_limit: int,
        qa_limit: int,
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        """Apply quality limits after ancestry completion without breaking paths."""
        qas_by_segment: dict[str, list[dict[str, Any]]] = {}
        segments_by_experience: dict[str, list[dict[str, Any]]] = {}
        for qa in qas:
            qas_by_segment.setdefault(str(qa["segment_id"]), []).append(qa)
        for segment in segments:
            segments_by_experience.setdefault(
                str(segment["experience_id"]), []
            ).append(segment)

        def segment_path_score(segment: dict[str, Any]) -> float:
            return clamp01(
                0.7 * max(
                    (
                        float(qa.get("_candidate_score") or 0.0)
                        for qa in qas_by_segment.get(
                            str(segment["segment_id"]), []
                        )
                    ),
                    default=0.0,
                )
                + 0.3 * float(segment.get("_candidate_score") or 0.0)
            )

        def experience_branch_score(experience: dict[str, Any]) -> float:
            branch_segments = segments_by_experience.get(
                str(experience["experience_id"]), []
            )
            return clamp01(
                0.6 * max(
                    (
                        float(qa.get("_candidate_score") or 0.0)
                        for segment in branch_segments
                        for qa in qas_by_segment.get(
                            str(segment["segment_id"]), []
                        )
                    ),
                    default=0.0,
                )
                + 0.3 * max(
                    (segment_path_score(segment) for segment in branch_segments),
                    default=0.0,
                )
                + 0.1 * float(experience.get("_candidate_score") or 0.0)
            )

        selected_experiences = sorted(
            experiences, key=experience_branch_score, reverse=True
        )[:experience_limit]
        selected_experience_ids = {
            str(item["experience_id"]) for item in selected_experiences
        }
        selected_segments = sorted(
            (
                item
                for item in segments
                if str(item.get("experience_id") or "")
                in selected_experience_ids
            ),
            key=segment_path_score,
            reverse=True,
        )[:segment_limit]
        selected_segment_ids = {
            str(item["segment_id"]) for item in selected_segments
        }
        selected_qas = sorted(
            (
                item
                for item in qas
                if str(item.get("segment_id") or "") in selected_segment_ids
            ),
            key=lambda item: float(item.get("_candidate_score") or 0.0),
            reverse=True,
        )[:qa_limit]
        return selected_experiences, selected_segments, selected_qas

    def _tree_counts(
        self, candidate_tree: list[dict[str, Any]]
    ) -> dict[str, int]:
        return {
            "experience": len(candidate_tree),
            "segment": sum(
                len(experience.get("segments") or [])
                for experience in candidate_tree
            ),
            "qa": sum(
                len(segment.get("qas") or [])
                for experience in candidate_tree
                for segment in experience.get("segments") or []
            ),
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
                "summary": str(experience.get("summary") or ""),
                "state": self._truncate_for_prompt(
                    json.dumps(experience.get("state") or {}, ensure_ascii=False), 300
                ),
                "vector_similarity": round(
                    float(experience.get("vector_similarity") or 0.0), 4
                ),
                "keyword_score": round(
                    float(experience.get("keyword_score") or 0.0), 4
                ),
                "relation_score": round(
                    float(experience.get("relation_score") or 0.0), 4
                ),
                "retrieval_sources": experience.get("retrieval_sources", []),
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
                    "summary": str(segment.get("summary") or ""),
                    "vector_similarity": round(
                        float(segment.get("vector_similarity") or 0.0), 4
                    ),
                    "keyword_score": round(
                        float(segment.get("keyword_score") or 0.0), 4
                    ),
                    "retrieval_sources": segment.get("retrieval_sources", []),
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
                        "retrieval_sources": qa.get("retrieval_sources", []),
                        "local_score": round(
                            float(qa.get("_candidate_score") or 0.0), 4
                        ),
                    })
                experience_node["segments"].append(segment_node)
            tree.append(experience_node)
        return tree

    def _serialize_candidate_tree(self, candidate_tree: list[dict[str, Any]]) -> str:
        return json.dumps(
            candidate_tree, ensure_ascii=False, separators=(',', ':')
        )

    def _count_candidate_tree_tokens(
        self, candidate_tree: list[dict[str, Any]]
    ) -> int:
        text = self._serialize_candidate_tree(candidate_tree)
        if not self._token_encoder_initialized:
            self._token_encoder_initialized = True
            try:
                import tiktoken

                self._token_encoder = tiktoken.get_encoding(self.token_encoding)
            except Exception:
                self._token_encoder = None
                logger.warning(
                    'Tokenizer %s is unavailable; using conservative UTF-8 estimate',
                    self.token_encoding,
                )
        if self._token_encoder is not None:
            return len(self._token_encoder.encode(text))
        return max(
            math.ceil(len(text) / 3),
            math.ceil(len(text.encode('utf-8')) / 3),
        )

    def _experience_branch_score(self, experience: dict[str, Any]) -> float:
        segments = experience.get('segments') or []
        segment_scores = [
            float(segment.get('local_score') or 0.0) for segment in segments
        ]
        qa_scores = [
            float(qa.get('local_score') or 0.0)
            for segment in segments
            for qa in segment.get('qas') or []
        ]
        experience_score = float(experience.get('local_score') or 0.0)
        return clamp01(
            0.6 * max(qa_scores, default=0.0)
            + 0.3 * max(segment_scores, default=0.0)
            + 0.1 * experience_score
        )

    def _extract_summary_sections(
        self, summary: Any, headers: tuple[str, ...]
    ) -> dict[str, str]:
        text = str(summary or '').strip()
        if not text:
            return {}
        header_pattern = '|'.join(re.escape(header) for header in headers)
        matches = list(re.finditer(
            rf'(?m)^\s*({header_pattern})\s*[：:]\s*', text
        ))
        sections: dict[str, str] = {}
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            sections[match.group(1)] = text[match.end():end].strip()
        return sections

    def _retain_summary_sections(
        self,
        summary: Any,
        all_headers: tuple[str, ...],
        retained_headers: tuple[str, ...],
    ) -> tuple[str, bool]:
        original = str(summary or '').strip()
        sections = self._extract_summary_sections(original, all_headers)
        if not sections:
            return original, False
        rendered = [
            f'{header}：{sections[header]}'
            for header in retained_headers
            if header in sections
        ]
        return '\n\n'.join(rendered), True

    def _compress_tree_summaries(
        self, candidate_tree: list[dict[str, Any]]
    ) -> dict[str, int]:
        stats = {
            'compressed_experience_summaries': 0,
            'compressed_segment_summaries': 0,
            'unstructured_summary_count': 0,
        }
        experience_headers = (
            '目标', '总体状态', '阶段总结', '当前推进', '长期信息', '下一步'
        )
        segment_headers = ('阶段概述', '关键过程', '阶段结论', '关键事实')
        for experience in candidate_tree:
            compressed, structured = self._retain_summary_sections(
                experience.get('summary'),
                experience_headers,
                ('阶段总结', '长期信息'),
            )
            if structured:
                if compressed != str(experience.get('summary') or '').strip():
                    stats['compressed_experience_summaries'] += 1
                experience['summary'] = compressed
            elif experience.get('summary'):
                stats['unstructured_summary_count'] += 1

            for segment in experience.get('segments') or []:
                compressed, structured = self._retain_summary_sections(
                    segment.get('summary'),
                    segment_headers,
                    ('关键过程', '阶段结论', '关键事实'),
                )
                if structured:
                    if compressed != str(segment.get('summary') or '').strip():
                        stats['compressed_segment_summaries'] += 1
                    segment['summary'] = compressed
                elif segment.get('summary'):
                    stats['unstructured_summary_count'] += 1
        return stats

    def _shrink_structured_summary(self, summary: str) -> str:
        headers = ('阶段总结', '长期信息', '关键过程', '阶段结论', '关键事实')
        sections = self._extract_summary_sections(summary, headers)
        if not sections:
            if len(summary) <= 96:
                return summary
            return self._truncate_for_prompt(summary, max(96, int(len(summary) * 0.8)))
        changed = False
        rendered: list[str] = []
        for header, body in sections.items():
            target = max(32, int(len(body) * 0.8))
            if len(body) > target:
                body = self._truncate_for_prompt(body, target)
                changed = True
            rendered.append(f'{header}：{body}')
        return '\n\n'.join(rendered) if changed else summary

    def _shrink_summaries_to_budget(
        self, candidate_tree: list[dict[str, Any]]
    ) -> None:
        for _ in range(24):
            if self._count_candidate_tree_tokens(candidate_tree) <= self.max_context_tokens:
                return
            nodes = [
                node
                for experience in candidate_tree
                for node in [experience, *(experience.get('segments') or [])]
                if node.get('summary')
            ]
            nodes.sort(key=lambda node: len(str(node.get('summary') or '')), reverse=True)
            changed = False
            for node in nodes:
                original = str(node.get('summary') or '')
                shrunk = self._shrink_structured_summary(original)
                if shrunk != original:
                    node['summary'] = shrunk
                    changed = True
                    break
            if not changed:
                return

    def _fit_candidate_tree_to_context(
        self, candidate_tree: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        fitted_tree = deepcopy(candidate_tree)
        fitted_tree.sort(key=self._experience_branch_score, reverse=True)
        for experience in fitted_tree:
            experience['segments'].sort(
                key=lambda segment: float(segment.get('local_score') or 0.0),
                reverse=True,
            )
            for segment in experience.get('segments') or []:
                segment['qas'].sort(
                    key=lambda qa: float(qa.get('local_score') or 0.0),
                    reverse=True,
                )

        tokens_before = self._count_candidate_tree_tokens(fitted_tree)
        minimum = max(1, min(self.min_prompt_experiences, len(fitted_tree)))
        removed_experience_ids: list[str] = []
        while (
            self._count_candidate_tree_tokens(fitted_tree) > self.max_context_tokens
            and len(fitted_tree) > minimum
        ):
            removed = fitted_tree.pop()
            removed_experience_ids.append(str(removed.get('id') or ''))

        tokens_after_pruning = self._count_candidate_tree_tokens(fitted_tree)
        compression_stats = {
            'compressed_experience_summaries': 0,
            'compressed_segment_summaries': 0,
            'unstructured_summary_count': 0,
        }
        summary_compressed = tokens_after_pruning > self.max_context_tokens
        if summary_compressed:
            compression_stats = self._compress_tree_summaries(fitted_tree)
            self._shrink_summaries_to_budget(fitted_tree)
        tokens_after_compression = self._count_candidate_tree_tokens(fitted_tree)

        debug = {
            'candidate_tokens_before_pruning': tokens_before,
            'candidate_tokens_after_branch_pruning': tokens_after_pruning,
            'candidate_tokens_after_summary_compression': tokens_after_compression,
            'max_context_tokens': self.max_context_tokens,
            'min_prompt_experiences': self.min_prompt_experiences,
            'removed_experience_ids': removed_experience_ids,
            'prompt_experiences_after_pruning': len(fitted_tree),
            'summary_compressed': summary_compressed,
            'context_overflow_unresolved': (
                tokens_after_compression > self.max_context_tokens
            ),
            **compression_stats,
        }
        return fitted_tree, debug

    def _public_candidate(self, item: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in item.items()
            if not key.startswith("_") and key != "descendant_similarity"
        }

    def _public_candidate_tree(
        self,
        prompt_tree: list[dict[str, Any]],
        candidate_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Hydrate the recall tree with complete E/S/QA records for API callers."""
        experiences = {
            str(item["experience_id"]): item
            for item in candidate_data["experiences"]
        }
        segments = {
            str(item["segment_id"]): item for item in candidate_data["segments"]
        }
        qas = {str(item["qa_id"]): item for item in candidate_data["qas"]}
        result: list[dict[str, Any]] = []
        for experience_node in prompt_tree:
            experience_id = str(experience_node["id"])
            source_experience = experiences.get(experience_id)
            if source_experience is None:
                continue
            experience = self._public_candidate(source_experience)
            experience["id"] = experience_id
            experience["local_score"] = float(
                experience_node.get("local_score") or 0.0
            )
            experience["segments"] = []
            for segment_node in experience_node.get("segments") or []:
                segment_id = str(segment_node["id"])
                source_segment = segments.get(segment_id)
                if source_segment is None:
                    continue
                segment = self._public_candidate(source_segment)
                segment["id"] = segment_id
                segment["local_score"] = float(
                    segment_node.get("local_score") or 0.0
                )
                segment["qas"] = []
                for qa_node in segment_node.get("qas") or []:
                    qa_id = str(qa_node["id"])
                    source_qa = qas.get(qa_id)
                    if source_qa is None:
                        continue
                    qa = self._public_candidate(source_qa)
                    qa["id"] = qa_id
                    qa["local_score"] = float(
                        qa_node.get("local_score") or 0.0
                    )
                    segment["qas"].append(qa)
                experience["segments"].append(segment)
            result.append(experience)
        return result

    def _selected_result_tree(
        self,
        experiences: list[dict[str, Any]],
        segments: list[dict[str, Any]],
        qas: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Return the final selection as the same explicit three-level tree."""
        segments_by_experience: dict[str, list[dict[str, Any]]] = {}
        qas_by_segment: dict[str, list[dict[str, Any]]] = {}
        for qa in qas:
            qas_by_segment.setdefault(str(qa["segment_id"]), []).append(qa)
        for segment in segments:
            segments_by_experience.setdefault(
                str(segment["experience_id"]), []
            ).append(segment)
        tree: list[dict[str, Any]] = []
        for source_experience in experiences:
            experience = dict(source_experience)
            experience["id"] = str(experience["experience_id"])
            experience["segments"] = []
            for source_segment in segments_by_experience.get(
                experience["experience_id"], []
            ):
                segment = dict(source_segment)
                segment["id"] = str(segment["segment_id"])
                segment["qas"] = []
                for source_qa in qas_by_segment.get(segment["segment_id"], []):
                    qa = dict(source_qa)
                    qa["id"] = str(qa["qa_id"])
                    segment["qas"].append(qa)
                experience["segments"].append(segment)
            tree.append(experience)
        return tree

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
        qas_by_segment: dict[str, list[dict[str, Any]]] = {}
        segments_by_experience: dict[str, list[dict[str, Any]]] = {}
        for qa in candidate_data["qas"]:
            qas_by_segment.setdefault(str(qa["segment_id"]), []).append(qa)
        for segment in candidate_data["segments"]:
            segments_by_experience.setdefault(
                str(segment["experience_id"]), []
            ).append(segment)

        def segment_path_score(segment: dict[str, Any]) -> float:
            qa_score = max(
                (
                    float(qa.get("_candidate_score") or 0.0)
                    for qa in qas_by_segment.get(str(segment["segment_id"]), [])
                ),
                default=0.0,
            )
            return clamp01(
                0.7 * qa_score
                + 0.3 * float(segment.get("_candidate_score") or 0.0)
            )

        def experience_branch_score(experience: dict[str, Any]) -> float:
            branch_segments = segments_by_experience.get(
                str(experience["experience_id"]), []
            )
            qa_score = max(
                (
                    float(qa.get("_candidate_score") or 0.0)
                    for segment in branch_segments
                    for qa in qas_by_segment.get(str(segment["segment_id"]), [])
                ),
                default=0.0,
            )
            return clamp01(
                0.6 * qa_score
                + 0.3 * max(
                    (segment_path_score(segment) for segment in branch_segments),
                    default=0.0,
                )
                + 0.1 * float(experience.get("_candidate_score") or 0.0)
            )

        experience_items = sorted(
            candidate_data["experiences"],
            key=experience_branch_score,
            reverse=True,
        )[:top_experience]
        experience_ids = {item["experience_id"] for item in experience_items}
        segment_items = sorted(
            (
                item
                for item in candidate_data["segments"]
                if item.get("experience_id") in experience_ids
            ),
            key=segment_path_score,
            reverse=True,
        )[:top_segment]
        segment_ids = {item["segment_id"] for item in segment_items}
        qa_items = sorted(
            (
                item
                for item in candidate_data["qas"]
                if item.get("segment_id") in segment_ids
            ),
            key=lambda item: float(item.get("_candidate_score") or 0.0),
            reverse=True,
        )[:top_qa]

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
            "segment_ids": list(item.get("segment_ids") or []),
            "vector_recalled": item["experience_id"] in vector_candidate_ids,
            "created_at": item.get("created_at", ""),
            "updated_at": item.get("updated_at", ""),
            "version": int(item.get("version") or 0),
            "last_summarized_segment_count": int(
                item.get("last_summarized_segment_count") or 0
            ),
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
            "qa_ids": list(item.get("qa_ids") or []),
            "vector_recalled": item["segment_id"] in vector_candidate_ids,
            "created_at": item.get("created_at", ""),
            "updated_at": item.get("updated_at", ""),
            "version": int(item.get("version") or 0),
            "last_summarized_qa_count": int(
                item.get("last_summarized_qa_count") or 0
            ),
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
            "status": item.get("status", ""),
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


class HybridRetriever(_BaseHybridRetriever):
    """Experience-first hierarchical retriever."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        retrieval_config = kwargs.pop("retrieval_config", None)
        super().__init__(
            *args,
            retrieval_config=retrieval_config,
            **kwargs,
        )
        has_explicit_settings = isinstance(retrieval_config, dict)
        settings = retrieval_config if has_explicit_settings else {}

        def setting(key: str, default: Any) -> Any:
            if key in settings and settings[key] is not None:
                return settings[key]
            if has_explicit_settings:
                return default
            return config_get("retrieval", key, default)

        self.max_context_tokens = max(1, int(setting(
            "max_context_tokens", 30000
        )))
        self.token_encoding = str(setting("token_encoding", "cl100k_base"))
        self.qa_full_search_confidence_threshold = clamp01(float(setting(
            "qa_full_search_confidence_threshold", 0.8
        )))
        self.experience_vector_recall_limit = max(1, int(setting(
            "experience_vector_recall_limit", 12
        )))
        self.experience_relational_recall_limit = max(1, int(setting(
            "experience_relational_recall_limit", 12
        )))
        self.experience_candidate_limit = max(1, int(setting(
            "experience_candidate_limit", 8
        )))
        self.scoped_segment_vector_limit = max(1, int(setting(
            "scoped_segment_vector_limit", 32
        )))
        self.scoped_qa_vector_limit = max(1, int(setting(
            "scoped_qa_vector_limit", 64
        )))
        self.qa_rescue_vector_limit = max(1, int(setting(
            "qa_rescue_vector_limit", 40
        )))
        self.qa_rescue_relational_limit = max(1, int(setting(
            "qa_rescue_relational_limit", 40
        )))
        self.min_retained_experiences = max(1, int(setting(
            "min_retained_experiences", 1
        )))
        self.min_retained_segments = max(1, int(setting(
            "min_retained_segments", 2
        )))
        self.summary_soft_experience_chars = max(64, int(setting(
            "summary_soft_experience_chars", 800
        )))
        self.summary_soft_segment_chars = max(32, int(setting(
            "summary_soft_segment_chars", 400
        )))
        self.summary_extreme_experience_chars = max(64, int(setting(
            "summary_extreme_experience_chars", 300
        )))
        self.summary_extreme_segment_chars = max(32, int(setting(
            "summary_extreme_segment_chars", 160
        )))
        if self.min_retained_segments < self.min_retained_experiences:
            raise ValueError(
                "retrieval.min_retained_segments must be >= "
                "retrieval.min_retained_experiences"
            )

    def recall(
        self,
        topic: str,
        core_entity: str,
        intent: str | None = None,
        entities: list[str] | None = None,
        query: str | None = None,
        *,
        query_confidence: float,
        top_experience: int = 3,
        top_segment: int = 5,
        top_qa: int = 8,
    ) -> dict[str, Any]:
        """Recall a bounded, structurally valid Experience -> Segment -> QA tree."""
        recall_started_at = time.perf_counter()
        if min(top_experience, top_segment, top_qa) <= 0:
            raise ValueError(
                "top_experience, top_segment and top_qa must be positive"
            )
        if not 0.0 <= float(query_confidence) <= 1.0:
            raise ValueError("query_confidence must be between 0.0 and 1.0")

        query_entities = [
            str(value).strip() for value in entities or [] if str(value).strip()
        ]
        query_text = build_query_text(
            topic, core_entity, intent, query_entities, query
        )
        keyword_query_text = chr(10).join(
            value
            for value in (
                str(topic or "").strip(),
                str(core_entity or "").strip(),
                str(intent or "").strip(),
                *query_entities,
                str(query or "").strip(),
            )
            if value
        )
        embedding_started_at = time.perf_counter()
        query_embedding = self._embed_query(query_text)
        embedding_elapsed_ms = round(
            (time.perf_counter() - embedding_started_at) * 1000, 3
        )
        low_confidence = (
            float(query_confidence)
            < self.qa_full_search_confidence_threshold
        )
        logger.info(
            "Starting Experience-first recall confidence=%.3f threshold=%.3f "
            "qa_global_rescue=%s",
            query_confidence,
            self.qa_full_search_confidence_threshold,
            low_confidence,
        )

        candidate_collection_started_at = time.perf_counter()
        candidate_data = self._collect_experience_first_candidates(
            topic=topic,
            core_entity=core_entity,
            intent=intent,
            query_entities=query_entities,
            keyword_query_text=keyword_query_text,
            query_embedding=query_embedding,
            low_confidence=low_confidence,
            top_experience=top_experience,
        )
        candidate_collection_elapsed_ms = round(
            (time.perf_counter() - candidate_collection_started_at) * 1000, 3
        )

        tree_build_started_at = time.perf_counter()
        candidate_tree = self._build_prompt_candidate_tree(candidate_data)
        tree_build_elapsed_ms = round(
            (time.perf_counter() - tree_build_started_at) * 1000, 3
        )
        original_candidate_tree = deepcopy(candidate_tree)

        tree_pruning_started_at = time.perf_counter()
        candidate_tree, tree_fit_debug = self._fit_candidate_tree_to_context(
            candidate_tree
        )
        tree_pruning_elapsed_ms = round(
            (time.perf_counter() - tree_pruning_started_at) * 1000, 3
        )
        self._validate_candidate_tree(candidate_tree)
        fitted_data = self._filter_candidate_data_to_tree(
            candidate_data, candidate_tree
        )

        selection: dict[str, Any] = {"experiences": []}
        llm_calls = 0
        reranking_started_at = time.perf_counter()
        if (
            candidate_tree
            and self.reranker is not None
            and not tree_fit_debug["context_overflow_unresolved"]
        ):
            rerank_hierarchy = getattr(self.reranker, "rerank_hierarchy", None)
            if callable(rerank_hierarchy):
                try:
                    llm_calls = 1
                    selection = rerank_hierarchy(
                        query_text=query_text,
                        candidate_tree=candidate_tree,
                        top_experience=top_experience,
                        top_segment=top_segment,
                        top_qa=top_qa,
                    )
                except Exception:
                    logger.exception(
                        "Hierarchical LLM rerank failed; using local fallback"
                    )
        elif tree_fit_debug["context_overflow_unresolved"]:
            logger.warning(
                "Candidate tree remains over budget after all pruning and "
                "compression stages; skipping LLM rerank"
            )
        reranking_elapsed_ms = round(
            (time.perf_counter() - reranking_started_at) * 1000, 3
        )

        result_processing_started_at = time.perf_counter()
        experiences, segments, qas = self._hydrate_hierarchical_selection(
            selection,
            fitted_data,
            top_experience,
            top_segment,
            top_qa,
        )
        if not qas and fitted_data["qas"]:
            experiences, segments, qas = self._local_hierarchical_selection(
                fitted_data, top_experience, top_segment, top_qa
            )
        experiences, segments, qas = self._cohere_selected_results(
            experiences, segments, qas
        )
        result_processing_elapsed_ms = round(
            (time.perf_counter() - result_processing_started_at) * 1000, 3
        )

        final_counts = {
            "experience": len(experiences),
            "segment": len(segments),
            "qa": len(qas),
        }
        logger.info(
            "Recall complete candidates=%s fitted=%s final=%s tokens=%s/%s",
            candidate_data["counts"],
            self._tree_counts(candidate_tree),
            final_counts,
            tree_fit_debug["candidate_tokens_final"],
            self.max_context_tokens,
        )
        context_text = build_context_text(experiences, segments, qas)
        public_candidate_tree = self._public_candidate_tree(
            candidate_tree, fitted_data
        )
        selected_tree = self._selected_result_tree(experiences, segments, qas)
        result = {
            "query": {
                "topic": topic,
                "core_entity": core_entity,
                "intent": intent or "",
                "entities": query_entities,
                "confidence": float(query_confidence),
            },
            "experiences": experiences,
            "segments": segments,
            "qas": qas,
            "candidate_tree": public_candidate_tree,
            "selected_tree": selected_tree,
            "context_text": context_text,
            "debug": {
                "total_retrieval_ms": 0.0,
                "embedding": {
                    "elapsed_ms": embedding_elapsed_ms,
                    "vector_dimensions": len(query_embedding or []),
                },
                "candidate_collection": {
                    "elapsed_ms": candidate_collection_elapsed_ms,
                    **candidate_data["timing"],
                },
                "candidate_tree_build": {
                    "elapsed_ms": tree_build_elapsed_ms,
                },
                "tree_pruning": {
                    "elapsed_ms": tree_pruning_elapsed_ms,
                    "counts_before": self._tree_counts(
                        original_candidate_tree
                    ),
                    "counts_after": self._tree_counts(candidate_tree),
                },
                "reranking": {
                    "elapsed_ms": reranking_elapsed_ms,
                    "attempted": bool(llm_calls),
                    "llm_calls": llm_calls,
                    "skipped_for_context_overflow": tree_fit_debug[
                        "context_overflow_unresolved"
                    ],
                },
                "result_processing": {
                    "elapsed_ms": result_processing_elapsed_ms,
                },
                "candidate_trees": {
                    "original": original_candidate_tree,
                    "pruned": deepcopy(candidate_tree),
                },
                "constraints": {
                    "query_confidence": float(query_confidence),
                    "qa_rescue_confidence_threshold": (
                        self.qa_full_search_confidence_threshold
                    ),
                    "requested_top_k": {
                        "experience": top_experience,
                        "segment": top_segment,
                        "qa": top_qa,
                    },
                    "candidate_limits": {
                        "experience": self.experience_candidate_limit,
                        "experience_vector": self.experience_vector_recall_limit,
                        "experience_relational": (
                            self.experience_relational_recall_limit
                        ),
                        "scoped_segment_vector": (
                            self.scoped_segment_vector_limit
                        ),
                        "scoped_qa_vector": self.scoped_qa_vector_limit,
                        "qa_rescue_vector": self.qa_rescue_vector_limit,
                        "qa_rescue_relational": (
                            self.qa_rescue_relational_limit
                        ),
                    },
                    "context": {
                        "max_context_tokens": self.max_context_tokens,
                        "min_retained_experiences": (
                            self.min_retained_experiences
                        ),
                        "min_retained_segments": self.min_retained_segments,
                    },
                },
                "strategy": "experience_first_confidence_gated_qa_rescue",
                "low_confidence_qa_rescue": low_confidence,
                "source_candidates": candidate_data["source_counts"],
                "pipeline_counts": {
                    "before_context_pruning": candidate_data["counts"],
                    "submitted_to_llm": self._tree_counts(candidate_tree),
                    "final_selected": final_counts,
                },
                "context_budget": tree_fit_debug,
                "llm_calls": llm_calls,
            },
        }
        result["debug"]["total_retrieval_ms"] = round(
            (time.perf_counter() - recall_started_at) * 1000, 3
        )
        return result

    def _recall_experience_roots(
        self,
        *,
        topic: str,
        core_entity: str,
        intent: str | None,
        query_text: str,
        query_embedding: list[float] | None,
        top_experience: int,
        timing_recorder: Any | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Fuse vector and topic/core_entity relational Experience recall."""
        vector_callback = lambda: self._vector_candidates(
            "experience", query_embedding, self.experience_vector_recall_limit
        )
        vector_hits = (
            timing_recorder(
                "chroma_vector_recall", "experience", "query", vector_callback
            )
            if timing_recorder is not None
            else vector_callback()
        )
        relation_callback = lambda: self.storage.search_experiences(
            topic, core_entity, self.experience_relational_recall_limit
        )
        relation_rows = (
            timing_recorder(
                "sql_recall", "experience", "relational", relation_callback
            )
            if timing_recorder is not None
            else relation_callback()
        )
        hydrate_callback = lambda: self.storage.get_experiences(list(vector_hits))
        vector_rows = (
            timing_recorder(
                "sql_recall",
                "experience",
                "hydrate_vector_ids",
                hydrate_callback,
            )
            if timing_recorder is not None
            else hydrate_callback()
        )
        rows_by_id: dict[str, dict[str, Any]] = {}
        for row in [*relation_rows, *vector_rows]:
            rows_by_id[str(row["experience_id"])] = row

        vector_rank = {
            memory_id: rank
            for rank, (memory_id, _) in enumerate(
                sorted(
                    vector_hits.items(),
                    key=lambda item: item[1],
                    reverse=True,
                ),
                1,
            )
        }
        relation_ids = [str(row["experience_id"]) for row in relation_rows]
        relation_rank = {
            memory_id: rank for rank, memory_id in enumerate(relation_ids, 1)
        }
        prepared: list[dict[str, Any]] = []
        rrf_constant = 60.0
        for experience_id, row in rows_by_id.items():
            item = self._prepare_experience(row, set(vector_hits))
            vector_similarity = vector_hits.get(experience_id, 0.0)
            relation_score = clamp01(float(row.get("relation_score") or 0.0) / 2.0)
            item["vector_similarity"] = vector_similarity
            item["keyword_score"] = 0.0
            item["relation_score"] = relation_score
            item["retrieval_sources"] = [
                source
                for source, present in (
                    ("experience_vector", experience_id in vector_rank),
                    ("experience_relational", experience_id in relation_rank),
                )
                if present
            ]
            vector_rrf = (
                (rrf_constant + 1.0)
                / (rrf_constant + vector_rank[experience_id])
                if experience_id in vector_rank else 0.0
            )
            relation_rrf = (
                (rrf_constant + 1.0)
                / (rrf_constant + relation_rank[experience_id])
                if experience_id in relation_rank else 0.0
            )
            structure_score = self._score_candidate(
                item, topic, core_entity, intent, query_text
            )
            item["_candidate_score"] = clamp01(
                0.4 * vector_rrf
                + 0.4 * relation_rrf
                + 0.2 * structure_score
            )
            prepared.append(item)

        limit = max(top_experience, self.experience_candidate_limit)
        prepared.sort(
            key=lambda item: (
                float(item.get("_candidate_score") or 0.0),
                str(item.get("updated_at") or ""),
            ),
            reverse=True,
        )
        selected = prepared[:limit]
        logger.info(
            "Experience roots recalled vector=%d relational=%d unique=%d selected=%d",
            len(vector_rows),
            len(relation_rows),
            len(prepared),
            len(selected),
        )
        return selected, {
            "experience_vector": len(vector_rows),
            "experience_relational": len(relation_rows),
        }

    def _load_descendants_for_experiences(
        self,
        experiences: list[dict[str, Any]],
        *,
        topic: str,
        core_entity: str,
        intent: str | None,
        query_text: str,
        query_embedding: list[float] | None,
        timing_recorder: Any | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Batch-load every active Segment and QA below the initial roots."""
        experience_ids = [str(item["experience_id"]) for item in experiences]
        segment_callback = lambda: self.storage.list_segments_by_experience_ids(
            experience_ids
        )
        segment_rows = (
            timing_recorder(
                "sql_recall",
                "segment",
                "load_experience_descendants",
                segment_callback,
            )
            if timing_recorder is not None
            else segment_callback()
        )
        segment_ids = {str(row["segment_id"]) for row in segment_rows}
        qa_callback = lambda: self.storage.list_qas_by_segment_ids(
            list(segment_ids)
        )
        qa_rows = (
            timing_recorder(
                "sql_recall", "qa", "load_segment_descendants", qa_callback
            )
            if timing_recorder is not None
            else qa_callback()
        )

        segment_vector_callback = lambda: self._scoped_vector_candidates(
            "segment", query_embedding, self.scoped_segment_vector_limit,
            metadata_filter={"experience_id": experience_ids}
        )
        segment_vectors = (
            timing_recorder(
                "chroma_vector_recall",
                "segment",
                "scoped_query",
                segment_vector_callback,
            )
            if timing_recorder is not None
            else segment_vector_callback()
        )
        qa_vector_callback = lambda: self._scoped_vector_candidates(
            "qa", query_embedding, self.scoped_qa_vector_limit,
            metadata_filter={"segment_id": list(segment_ids)}
        )
        qa_vectors = (
            timing_recorder(
                "chroma_vector_recall", "qa", "scoped_query", qa_vector_callback
            )
            if timing_recorder is not None
            else qa_vector_callback()
        )

        segments: list[dict[str, Any]] = []
        for row in segment_rows:
            segment_id = str(row["segment_id"])
            item = self._prepare_segment(row, set(segment_vectors))
            item["vector_similarity"] = segment_vectors.get(segment_id, 0.0)
            ### TODO 关键字打分
            item["keyword_score"] = self._candidate_keyword_score(
                item, query_text, topic, core_entity, intent
            )
            item["retrieval_sources"] = ["initial_experience_descendant"]
            if segment_id in segment_vectors:
                item["retrieval_sources"].append("scoped_segment_vector")
            item["_candidate_score"] = self._score_candidate(
                item, topic, core_entity, intent, query_text
            )
            segments.append(item)

        qas: list[dict[str, Any]] = []
        for row in qa_rows:
            qa_id = str(row["qa_id"])
            item = self._prepare_qa(row, set(qa_vectors))
            item["vector_similarity"] = qa_vectors.get(qa_id, 0.0)
            item["keyword_score"] = self._candidate_keyword_score(
                item, query_text, topic, core_entity, intent
            )
            item["retrieval_sources"] = ["initial_experience_descendant"]
            if qa_id in qa_vectors:
                item["retrieval_sources"].append("scoped_qa_vector")
            local_score = self._score_candidate(
                item, topic, core_entity, intent, query_text
            )
            # Stored QA confidence measures record quality, not query confidence.
            item["_candidate_score"] = clamp01(
                0.95 * local_score
                + 0.05 * float(item.get("confidence") or 0.0)
            )
            qas.append(item)

        logger.info(
            "Expanded initial Experience roots experiences=%d segments=%d qas=%d",
            len(experiences),
            len(segments),
            len(qas),
        )
        return segments, qas

    def _scoped_vector_candidates(
        self,
        memory_type: str,
        query_embedding: list[float] | None,
        limit: int,
        *,
        metadata_filter: dict[str, Any],
    ) -> dict[str, float]:
        """Vector-score descendants without opening a global QA search."""
        if self.vector_store is None or query_embedding is None:
            return {}
        if any(isinstance(value, list) and not value for value in metadata_filter.values()):
            return {}
        try:
            items = self.vector_store.query(
                query_embedding=query_embedding,
                memory_type=memory_type,
                top_k=max(1, int(limit)),
                metadata_filter=metadata_filter,
            )
        except Exception:
            logger.exception(
                "Scoped %s vector scoring failed filters=%s",
                memory_type,
                metadata_filter,
            )
            return {}
        candidates: dict[str, float] = {}
        for item in items:
            metadata = item.get("metadata") or {}
            memory_id = str(
                metadata.get("memory_id")
                or metadata.get(f"{memory_type}_id")
                or ""
            )
            if memory_id:
                candidates[memory_id] = clamp01(
                    float(item.get("similarity") or 0.0)
                )
        logger.debug(
            "Scoped vector scoring layer=%s filters=%s hits=%d",
            memory_type,
            metadata_filter,
            len(candidates),
        )
        return candidates

    def _merge_low_confidence_qa_paths(
        self,
        *,
        experiences: list[dict[str, Any]],
        segments: list[dict[str, Any]],
        qas: list[dict[str, Any]],
        topic: str,
        core_entity: str,
        intent: str | None,
        entities: list[str],
        query_text: str,
        query_embedding: list[float] | None,
        timing_recorder: Any | None = None,
    ) -> dict[str, int]:
        """Add global QA hits and only their real Segment/Experience ancestors."""
        vector_callback = lambda: self._vector_candidates(
            "qa", query_embedding, self.qa_rescue_vector_limit
        )
        vector_hits = (
            timing_recorder(
                "chroma_vector_recall", "qa_rescue", "query", vector_callback
            )
            if timing_recorder is not None
            else vector_callback()
        )
        relational_callback = lambda: self.storage.search_qas(
            topic=topic, core_entity=core_entity, entities=entities,
            keywords=self._keyword_terms(query_text),
            limit=self.qa_rescue_relational_limit
        )
        relational_rows = (
            timing_recorder(
                "sql_recall", "qa_rescue", "relational", relational_callback
            )
            if timing_recorder is not None
            else relational_callback()
        )
        hydrate_qa_callback = lambda: self.storage.get_qas(list(vector_hits))
        vector_rows = (
            timing_recorder(
                "sql_recall",
                "qa_rescue",
                "hydrate_vector_ids",
                hydrate_qa_callback,
            )
            if timing_recorder is not None
            else hydrate_qa_callback()
        )
        relational_ids = {
            str(row["qa_id"]) for row in relational_rows
        }
        rows_by_id: dict[str, dict[str, Any]] = {}
        for row in [*relational_rows, *vector_rows]:
            if row.get("status") == "active":
                rows_by_id[str(row["qa_id"])] = row

        qa_by_id = {str(item["qa_id"]): item for item in qas}
        for qa_id, row in rows_by_id.items():
            item = self._prepare_qa(row, set(vector_hits))
            item["vector_similarity"] = vector_hits.get(qa_id, 0.0)
            item["keyword_score"] = self._candidate_keyword_score(
                item, query_text, topic, core_entity, intent
            )
            item["retrieval_sources"] = [
                source
                for source, present in (
                    ("global_qa_vector", qa_id in vector_hits),
                    ("global_qa_relational", qa_id in relational_ids),
                )
                if present
            ]
            local_score = self._score_candidate(
                item, topic, core_entity, intent, query_text
            )
            item["_candidate_score"] = clamp01(
                0.95 * local_score
                + 0.05 * float(item.get("confidence") or 0.0)
            )
            existing = qa_by_id.get(qa_id)
            if existing is None or float(item["_candidate_score"]) > float(
                existing.get("_candidate_score") or 0.0
            ):
                qa_by_id[qa_id] = item

        segment_by_id = {
            str(item["segment_id"]): item for item in segments
        }
        missing_segment_ids = {
            str(qa.get("segment_id") or "")
            for qa in qa_by_id.values()
            if str(qa.get("segment_id") or "") not in segment_by_id
        }
        missing_segment_ids.discard("")
        hydrate_segments_callback = lambda: self.storage.get_segments(
            list(missing_segment_ids)
        )
        missing_segment_rows = (
            timing_recorder(
                "sql_recall",
                "qa_rescue",
                "hydrate_missing_segments",
                hydrate_segments_callback,
            )
            if timing_recorder is not None
            else hydrate_segments_callback()
        )
        for row in missing_segment_rows:
            if row.get("status") == "deleted":
                continue
            segment_id = str(row["segment_id"])
            item = self._prepare_segment(row, set())
            item["vector_similarity"] = 0.0
            item["keyword_score"] = self._candidate_keyword_score(
                item, query_text, topic, core_entity, intent
            )
            item["retrieval_sources"] = ["qa_rescue_ancestor"]
            item["_candidate_score"] = self._score_candidate(
                item, topic, core_entity, intent, query_text
            )
            segment_by_id[segment_id] = item

        experience_by_id = {
            str(item["experience_id"]): item for item in experiences
        }
        missing_experience_ids = {
            str(segment.get("experience_id") or "")
            for segment in segment_by_id.values()
            if str(segment.get("experience_id") or "") not in experience_by_id
        }
        missing_experience_ids.discard("")
        hydrate_experiences_callback = lambda: self.storage.get_experiences(
            list(missing_experience_ids)
        )
        missing_experience_rows = (
            timing_recorder(
                "sql_recall",
                "qa_rescue",
                "hydrate_missing_experiences",
                hydrate_experiences_callback,
            )
            if timing_recorder is not None
            else hydrate_experiences_callback()
        )
        for row in missing_experience_rows:
            experience_id = str(row["experience_id"])
            item = self._prepare_experience(row, set())
            item["vector_similarity"] = 0.0
            item["keyword_score"] = 0.0
            item["relation_score"] = 0.0
            item["retrieval_sources"] = ["qa_rescue_ancestor"]
            item["_candidate_score"] = self._score_candidate(
                item, topic, core_entity, intent, query_text
            )
            experience_by_id[experience_id] = item

        valid_experience_ids = set(experience_by_id)
        valid_segments = {
            segment_id: item
            for segment_id, item in segment_by_id.items()
            if str(item.get("experience_id") or "") in valid_experience_ids
        }
        valid_qas = {
            qa_id: item
            for qa_id, item in qa_by_id.items()
            if str(item.get("segment_id") or "") in valid_segments
        }
        experiences[:] = list(experience_by_id.values())
        segments[:] = list(valid_segments.values())
        qas[:] = list(valid_qas.values())
        logger.info(
            "Low-confidence QA rescue vector=%d relational=%d unique=%d "
            "added_segments=%d added_experiences=%d",
            len(vector_rows),
            len(relational_rows),
            len(rows_by_id),
            len(missing_segment_ids),
            len(missing_experience_ids),
        )
        return {
            "qa_rescue_vector": len(vector_rows),
            "qa_rescue_relational": len(relational_rows),
            "qa_rescue_unique": len(rows_by_id),
        }

    def _collect_experience_first_candidates(
        self,
        *,
        topic: str,
        core_entity: str,
        intent: str | None,
        query_entities: list[str],
        keyword_query_text: str,
        query_embedding: list[float] | None,
        low_confidence: bool,
        top_experience: int,
    ) -> dict[str, Any]:
        collection_started_at = time.perf_counter()
        timing_seconds: dict[str, Any] = {
            "sql_recall": {"total": 0.0, "by_layer": {}},
            "chroma_vector_recall": {"total": 0.0, "by_layer": {}},
            "ranking_processing": {"total": 0.0, "by_layer": {}},
        }

        def record_elapsed(
            category: str,
            layer: str,
            operation: str,
            elapsed: float,
        ) -> None:
            bucket = timing_seconds[category]
            bucket["total"] += elapsed
            layer_bucket = bucket["by_layer"].setdefault(layer, {})
            layer_bucket[operation] = layer_bucket.get(operation, 0.0) + elapsed

        def measure(
            category: str,
            layer: str,
            operation: str,
            callback: Any,
        ) -> Any:
            started_at = time.perf_counter()
            value = callback()
            record_elapsed(
                category, layer, operation, time.perf_counter() - started_at
            )
            return value

        roots_started_at = time.perf_counter()
        roots_accounted_before = sum(
            bucket["total"] for bucket in timing_seconds.values()
        )
        experiences, source_counts = self._recall_experience_roots(
            topic=topic,
            core_entity=core_entity,
            intent=intent,
            query_text=keyword_query_text,
            query_embedding=query_embedding,
            top_experience=top_experience,
            timing_recorder=measure,
        )
        roots_elapsed = time.perf_counter() - roots_started_at
        roots_accounted = sum(
            bucket["total"] for bucket in timing_seconds.values()
        ) - roots_accounted_before
        record_elapsed(
            "ranking_processing",
            "experience",
            "fuse_score_and_rank",
            max(0.0, roots_elapsed - roots_accounted),
        )

        descendants_started_at = time.perf_counter()
        descendants_accounted_before = sum(
            bucket["total"] for bucket in timing_seconds.values()
        )
        segments, qas = self._load_descendants_for_experiences(
            experiences,
            topic=topic,
            core_entity=core_entity,
            intent=intent,
            query_text=keyword_query_text,
            query_embedding=query_embedding,
            timing_recorder=measure,
        )
        descendants_elapsed = time.perf_counter() - descendants_started_at
        descendants_accounted = sum(
            bucket["total"] for bucket in timing_seconds.values()
        ) - descendants_accounted_before
        record_elapsed(
            "ranking_processing",
            "descendants",
            "prepare_and_score",
            max(0.0, descendants_elapsed - descendants_accounted),
        )
        source_counts.update({
            "initial_descendant_segments": len(segments),
            "initial_descendant_qas": len(qas),
            "qa_rescue_vector": 0,
            "qa_rescue_relational": 0,
            "qa_rescue_unique": 0,
        })
        if low_confidence:
            rescue_started_at = time.perf_counter()
            rescue_accounted_before = sum(
                bucket["total"] for bucket in timing_seconds.values()
            )
            source_counts.update(self._merge_low_confidence_qa_paths(
                experiences=experiences,
                segments=segments,
                qas=qas,
                topic=topic,
                core_entity=core_entity,
                intent=intent,
                entities=query_entities,
                query_text=keyword_query_text,
                query_embedding=query_embedding,
                timing_recorder=measure,
            ))
            rescue_elapsed = time.perf_counter() - rescue_started_at
            rescue_accounted = sum(
                bucket["total"] for bucket in timing_seconds.values()
            ) - rescue_accounted_before
            record_elapsed(
                "ranking_processing",
                "qa_rescue",
                "merge_score_and_complete_ancestry",
                max(0.0, rescue_elapsed - rescue_accounted),
            )

        # Remove unusable/orphaned nodes once, before the canonical tree is built.
        canonical_started_at = time.perf_counter()
        experience_ids = {str(item["experience_id"]) for item in experiences}
        segments = [
            item for item in segments
            if str(item.get("experience_id") or "") in experience_ids
        ]
        segment_ids = {str(item["segment_id"]) for item in segments}
        qas = [
            item for item in qas
            if str(item.get("segment_id") or "") in segment_ids
        ]
        segment_ids_with_qas = {
            str(item["segment_id"]) for item in qas
        }
        segments = [
            item for item in segments
            if str(item["segment_id"]) in segment_ids_with_qas
        ]
        experience_ids_with_segments = {
            str(item["experience_id"]) for item in segments
        }
        experiences = [
            item for item in experiences
            if str(item["experience_id"]) in experience_ids_with_segments
        ]
        experiences.sort(
            key=lambda item: float(item.get("_candidate_score") or 0.0),
            reverse=True,
        )
        segments.sort(
            key=lambda item: float(item.get("_candidate_score") or 0.0),
            reverse=True,
        )
        qas.sort(
            key=lambda item: float(item.get("_candidate_score") or 0.0),
            reverse=True,
        )
        record_elapsed(
            "ranking_processing",
            "canonical_tree",
            "filter_orphans_and_sort",
            time.perf_counter() - canonical_started_at,
        )
        counts = {
            "experience": len(experiences),
            "segment": len(segments),
            "qa": len(qas),
        }
        logger.info("Canonical candidate paths prepared counts=%s", counts)
        def milliseconds(value: float) -> float:
            return round(value * 1000, 3)

        timing = {
            category: {
                "total_ms": milliseconds(bucket["total"]),
                "by_layer": {
                    layer: {
                        operation: milliseconds(elapsed)
                        for operation, elapsed in operations.items()
                    }
                    for layer, operations in bucket["by_layer"].items()
                },
            }
            for category, bucket in timing_seconds.items()
        }
        instrumented_elapsed = time.perf_counter() - collection_started_at
        timing["instrumented_elapsed_ms"] = milliseconds(instrumented_elapsed)
        timing["other_processing_ms"] = milliseconds(max(
            0.0,
            instrumented_elapsed - sum(
                bucket["total"] for bucket in timing_seconds.values()
            ),
        ))
        return {
            "experiences": experiences,
            "segments": segments,
            "qas": qas,
            "counts": counts,
            "source_counts": source_counts,
            "timing": timing,
        }

    def _tree_counts(
        self, candidate_tree: list[dict[str, Any]]
    ) -> dict[str, int]:
        return {
            "experience": len(candidate_tree),
            "segment": sum(
                len(experience.get("segments") or [])
                for experience in candidate_tree
            ),
            "qa": sum(
                len(segment.get("qas") or [])
                for experience in candidate_tree
                for segment in experience.get("segments") or []
            ),
        }

    def _validate_candidate_tree(
        self, candidate_tree: list[dict[str, Any]]
    ) -> None:
        """Fail fast if pruning or rescue introduced an orphan/duplicate node."""
        experience_ids: set[str] = set()
        segment_ids: set[str] = set()
        qa_ids: set[str] = set()
        for experience in candidate_tree:
            experience_id = str(experience.get("id") or "")
            segments = experience.get("segments") or []
            if not experience_id or experience_id in experience_ids or not segments:
                raise ValueError("Invalid candidate tree Experience node")
            experience_ids.add(experience_id)
            for segment in segments:
                segment_id = str(segment.get("id") or "")
                qas = segment.get("qas") or []
                if not segment_id or segment_id in segment_ids or not qas:
                    raise ValueError("Invalid candidate tree Segment node")
                segment_ids.add(segment_id)
                for qa in qas:
                    qa_id = str(qa.get("id") or "")
                    if not qa_id or qa_id in qa_ids:
                        raise ValueError("Invalid candidate tree QA node")
                    qa_ids.add(qa_id)

    def _filter_candidate_data_to_tree(
        self,
        candidate_data: dict[str, Any],
        candidate_tree: list[dict[str, Any]],
    ) -> dict[str, Any]:
        experience_ids = {
            str(experience["id"]) for experience in candidate_tree
        }
        segment_ids = {
            str(segment["id"])
            for experience in candidate_tree
            for segment in experience.get("segments") or []
        }
        qa_ids = {
            str(qa["id"])
            for experience in candidate_tree
            for segment in experience.get("segments") or []
            for qa in segment.get("qas") or []
        }
        return {
            "experiences": [
                item for item in candidate_data["experiences"]
                if str(item["experience_id"]) in experience_ids
            ],
            "segments": [
                item for item in candidate_data["segments"]
                if str(item["segment_id"]) in segment_ids
            ],
            "qas": [
                item for item in candidate_data["qas"]
                if str(item["qa_id"]) in qa_ids
            ],
        }

    def _segment_branch_score(self, segment: dict[str, Any]) -> float:
        qa_scores = sorted(
            (
                float(qa.get("local_score") or 0.0)
                for qa in segment.get("qas") or []
            ),
            reverse=True,
        )
        top_scores = qa_scores[:3]
        return clamp01(
            0.65 * max(qa_scores, default=0.0)
            + 0.20 * (
                sum(top_scores) / len(top_scores) if top_scores else 0.0
            )
            + 0.15 * float(segment.get("local_score") or 0.0)
        )

    def _experience_branch_score(self, experience: dict[str, Any]) -> float:
        segment_scores = sorted(
            (
                self._segment_branch_score(segment)
                for segment in experience.get("segments") or []
            ),
            reverse=True,
        )
        top_scores = segment_scores[:2]
        return clamp01(
            0.60 * max(segment_scores, default=0.0)
            + 0.25 * (
                sum(top_scores) / len(top_scores) if top_scores else 0.0
            )
            + 0.15 * float(experience.get("local_score") or 0.0)
        )

    def _remove_lowest_qa(
        self,
        tree: list[dict[str, Any]],
    ) -> tuple[str | None, list[str], list[str]]:
        candidates = sorted(
            (
                (
                    float(qa.get("local_score") or 0.0),
                    str(experience["id"]),
                    str(segment["id"]),
                    str(qa["id"]),
                )
                for experience in tree
                for segment in experience.get("segments") or []
                for qa in segment.get("qas") or []
            ),
            key=lambda item: (item[0], item[3]),
        )
        for _, experience_id, segment_id, qa_id in candidates:
            experience = next(
                item for item in tree if str(item["id"]) == experience_id
            )
            segment = next(
                item for item in experience["segments"]
                if str(item["id"]) == segment_id
            )
            # Keep the strongest evidence leaf in every Segment. Once each
            # Segment has one QA, branch-level decisions belong to the Segment
            # stage rather than being driven by a single low-scoring leaf.
            if len(segment["qas"]) <= 1:
                continue

            segment["qas"] = [
                qa for qa in segment["qas"] if str(qa["id"]) != qa_id
            ]
            removed_segments: list[str] = []
            removed_experiences: list[str] = []
            return qa_id, removed_segments, removed_experiences
        return None, [], []

    def _remove_lowest_segment(
        self,
        tree: list[dict[str, Any]],
        min_segments: int,
    ) -> tuple[str | None, str | None]:
        candidates = sorted(
            (
                (
                    self._segment_branch_score(segment),
                    str(experience["id"]),
                    str(segment["id"]),
                )
                for experience in tree
                for segment in experience.get("segments") or []
            ),
            key=lambda item: (item[0], item[2]),
        )
        counts = self._tree_counts(tree)
        for _, experience_id, segment_id in candidates:
            experience = next(
                item for item in tree if str(item["id"]) == experience_id
            )
            if counts["segment"] <= min_segments:
                continue
            # Preserve one Segment per Experience here. Root-level deletion is
            # intentionally deferred to the Experience pruning stage.
            if len(experience["segments"]) <= 1:
                continue
            experience["segments"] = [
                segment for segment in experience["segments"]
                if str(segment["id"]) != segment_id
            ]
            return segment_id, None
        return None, None

    def _remove_lowest_experience(
        self,
        tree: list[dict[str, Any]],
        min_experiences: int,
        min_segments: int,
    ) -> str | None:
        counts = self._tree_counts(tree)
        candidates = sorted(
            tree,
            key=lambda item: (
                self._experience_branch_score(item),
                str(item.get("id") or ""),
            ),
        )
        for experience in candidates:
            branch_segments = len(experience.get("segments") or [])
            if counts["experience"] <= min_experiences:
                return None
            if counts["segment"] - branch_segments < min_segments:
                continue
            experience_id = str(experience["id"])
            tree[:] = [
                item for item in tree if str(item["id"]) != experience_id
            ]
            return experience_id
        return None

    def _compress_tree_to_char_limits(
        self,
        tree: list[dict[str, Any]],
        *,
        experience_limit: int,
        segment_limit: int,
    ) -> dict[str, int]:
        changed_experiences = 0
        changed_segments = 0
        for experience in tree:
            summary = str(experience.get("summary") or "")
            compressed = self._truncate_for_prompt(summary, experience_limit)
            if compressed != summary:
                experience["summary"] = compressed
                changed_experiences += 1
            for segment in experience.get("segments") or []:
                summary = str(segment.get("summary") or "")
                compressed = self._truncate_for_prompt(summary, segment_limit)
                if compressed != summary:
                    segment["summary"] = compressed
                    changed_segments += 1
        return {
            "experience_summaries": changed_experiences,
            "segment_summaries": changed_segments,
        }

    def _clip_qa_payloads_to_budget(
        self, tree: list[dict[str, Any]]
    ) -> int:
        """Last-resort clipping when the protected minimum tree is itself huge."""
        clipped = 0
        for _ in range(256):
            if self._count_candidate_tree_tokens(tree) <= self.max_context_tokens:
                break
            fields = [
                (qa, field)
                for experience in tree
                for segment in experience.get("segments") or []
                for qa in segment.get("qas") or []
                for field in ("assistant_output", "user_input")
                if len(str(qa.get(field) or "")) > 96
            ]
            if not fields:
                break
            qa, field = max(
                fields, key=lambda item: len(str(item[0].get(item[1]) or ""))
            )
            original = str(qa.get(field) or "")
            target = max(96, int(len(original) * 0.75))
            # Keep the ellipsis inside the hard target so every iteration makes
            # progress and cannot stall at target + len("...").
            qa[field] = original[:target - 1].rstrip() + "…"
            clipped += 1
        return clipped

    def _fit_candidate_tree_to_context(
        self, candidate_tree: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Fit the tree by QA, Segment, Experience, then two summary passes."""
        tree = deepcopy(candidate_tree)
        tree.sort(key=self._experience_branch_score, reverse=True)
        for experience in tree:
            experience["segments"].sort(
                key=self._segment_branch_score, reverse=True
            )
            for segment in experience.get("segments") or []:
                segment["qas"].sort(
                    key=lambda qa: float(qa.get("local_score") or 0.0),
                    reverse=True,
                )

        initial_counts = self._tree_counts(tree)
        min_experiences = min(
            self.min_retained_experiences, initial_counts["experience"]
        )
        min_segments = min(
            self.min_retained_segments, initial_counts["segment"]
        )
        tokens_before = self._count_candidate_tree_tokens(tree)
        removed_qa_ids: list[str] = []
        removed_segment_ids: list[str] = []
        removed_experience_ids: list[str] = []

        # Stage 1: remove the least relevant leaves. Empty parents are removed
        # in the same operation, but only when the configured minima survive.
        while self._count_candidate_tree_tokens(tree) > self.max_context_tokens:
            qa_id, segment_ids, experience_ids = self._remove_lowest_qa(
                tree
            )
            if qa_id is None:
                break
            removed_qa_ids.append(qa_id)
            removed_segment_ids.extend(segment_ids)
            removed_experience_ids.extend(experience_ids)
        tokens_after_qa = self._count_candidate_tree_tokens(tree)
        logger.info(
            "Context pruning QA stage removed=%d cascaded_segments=%d "
            "cascaded_experiences=%d tokens=%d",
            len(removed_qa_ids),
            len(removed_segment_ids),
            len(removed_experience_ids),
            tokens_after_qa,
        )

        # Stage 2: remove complete Segment branches once QA leaf pruning can no
        # longer make progress without violating the structural minimum.
        segment_stage_removed: list[str] = []
        while self._count_candidate_tree_tokens(tree) > self.max_context_tokens:
            segment_id, experience_id = self._remove_lowest_segment(
                tree, min_segments
            )
            if segment_id is None:
                break
            segment_stage_removed.append(segment_id)
            removed_segment_ids.append(segment_id)
            if experience_id:
                removed_experience_ids.append(experience_id)
        tokens_after_segment = self._count_candidate_tree_tokens(tree)
        logger.info(
            "Context pruning Segment stage removed=%d tokens=%d",
            len(segment_stage_removed),
            tokens_after_segment,
        )

        # Stage 3: whole Experience branches are the last structural deletion.
        experience_stage_removed: list[str] = []
        while self._count_candidate_tree_tokens(tree) > self.max_context_tokens:
            experience_id = self._remove_lowest_experience(
                tree, min_experiences, min_segments
            )
            if experience_id is None:
                break
            experience_stage_removed.append(experience_id)
            removed_experience_ids.append(experience_id)
        tokens_after_experience = self._count_candidate_tree_tokens(tree)
        logger.info(
            "Context pruning Experience stage removed=%d tokens=%d",
            len(experience_stage_removed),
            tokens_after_experience,
        )

        normal_compression: dict[str, Any] = {}
        if tokens_after_experience > self.max_context_tokens:
            normal_compression.update(self._compress_tree_summaries(tree))
            normal_compression.update(self._compress_tree_to_char_limits(
                tree,
                experience_limit=self.summary_soft_experience_chars,
                segment_limit=self.summary_soft_segment_chars,
            ))
        tokens_after_normal_compression = self._count_candidate_tree_tokens(tree)
        logger.info(
            "Context normal summary compression applied=%s tokens=%d",
            bool(normal_compression),
            tokens_after_normal_compression,
        )

        extreme_compression: dict[str, int] = {}
        if tokens_after_normal_compression > self.max_context_tokens:
            extreme_compression = self._compress_tree_to_char_limits(
                tree,
                experience_limit=self.summary_extreme_experience_chars,
                segment_limit=self.summary_extreme_segment_chars,
            )
        tokens_after_extreme_compression = self._count_candidate_tree_tokens(tree)
        logger.info(
            "Context extreme summary compression applied=%s tokens=%d",
            bool(extreme_compression),
            tokens_after_extreme_compression,
        )

        emergency_qa_clips = 0
        if tokens_after_extreme_compression > self.max_context_tokens:
            emergency_qa_clips = self._clip_qa_payloads_to_budget(tree)
        final_tokens = self._count_candidate_tree_tokens(tree)
        if emergency_qa_clips:
            logger.warning(
                "Protected minimum tree required emergency QA clipping "
                "operations=%d final_tokens=%d",
                emergency_qa_clips,
                final_tokens,
            )
        self._validate_candidate_tree(tree)
        return tree, {
            "limit_tokens": self.max_context_tokens,
            "candidate_tokens_before_pruning": tokens_before,
            "candidate_tokens_after_qa_pruning": tokens_after_qa,
            "candidate_tokens_after_segment_pruning": tokens_after_segment,
            "candidate_tokens_after_experience_pruning": tokens_after_experience,
            "candidate_tokens_after_normal_compression": (
                tokens_after_normal_compression
            ),
            "candidate_tokens_after_extreme_compression": (
                tokens_after_extreme_compression
            ),
            "candidate_tokens_final": final_tokens,
            "minimum_experiences": min_experiences,
            "minimum_segments": min_segments,
            "removed_qa_ids": removed_qa_ids,
            "removed_segment_ids": removed_segment_ids,
            "removed_experience_ids": removed_experience_ids,
            "normal_summary_compression": normal_compression,
            "extreme_summary_compression": extreme_compression,
            "emergency_qa_clip_operations": emergency_qa_clips,
            "context_overflow_unresolved": final_tokens > self.max_context_tokens,
        }

    def _cohere_selected_results(
        self,
        experiences: list[dict[str, Any]],
        segments: list[dict[str, Any]],
        qas: list[dict[str, Any]],
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        """Remove selected parents that have no selected descendants."""
        segment_ids_with_qas = {
            str(qa.get("segment_id") or "") for qa in qas
        }
        coherent_segments = [
            segment for segment in segments
            if str(segment.get("segment_id") or "") in segment_ids_with_qas
        ]
        experience_ids_with_segments = {
            str(segment.get("experience_id") or "")
            for segment in coherent_segments
        }
        coherent_experiences = [
            experience for experience in experiences
            if str(experience.get("experience_id") or "")
            in experience_ids_with_segments
        ]
        return coherent_experiences, coherent_segments, qas

class StructuredRetriever(HybridRetriever):
    """Alias retained for callers that prefer the structural name."""
