"""Historical Experience recall independent from active-memory retrieval."""

from __future__ import annotations

from copy import deepcopy
import json
import logging
import time
from typing import Any

from .embedder import TextEmbedder
from .storage import MemoryStorage
from .vector_store import ChromaVectorStore
from .prompts.topic_memory import build_historical_experience_prompt


logger = logging.getLogger(__name__)


class ExperienceRecaller:
    """Recall and compress historical Experiences from SQLite and Chroma."""

    def __init__(
        self,
        storage: MemoryStorage,
        vector_store: ChromaVectorStore,
        embedder: TextEmbedder,
    ) -> None:
        self.storage = storage
        self.vector_store = vector_store
        self.embedder = embedder

    def recall(
        self,
        topic: str,
        core_entity: str,
        query: str,
        intent: str = "",
    ) -> dict[str, Any]:
        """Compress deduplicated SQL/Chroma Experience summaries with an LLM."""
        started_at = time.perf_counter()
        topic = str(topic or "").strip()
        core_entity = str(core_entity or "").strip()
        query = str(query or "").strip()
        intent = str(intent or "").strip()
        if not topic or not core_entity or not query:
            raise ValueError("topic, core_entity and query must not be empty")

        sql_rows = self.storage.search_completed_experiences(
            topic=topic,
            core_entity=core_entity,
            limit=3,
        )

        query_text = f"主题：{topic}\n核心实体：{core_entity}\n查询：{query}"
        vector_items: list[dict[str, Any]] = []
        try:
            vector_items = self.vector_store.query(
                self.embedder.embed(query_text),
                memory_type="experience",
                top_k=3,
                metadata_filter={"status": "completed"},
            )
        except Exception:
            logger.warning("Experience vector recall failed", exc_info=True)

        experience_map: dict[str, dict[str, Any]] = {}
        for row in sql_rows:
            experience_id = str(row.get("experience_id") or "")
            if experience_id:
                experience_map[experience_id] = deepcopy(row)

        chroma_candidates = 0
        for item in vector_items[:3]:
            metadata = item.get("metadata") or {}
            experience_id = str(
                metadata.get("experience_id")
                or metadata.get("memory_id")
                or ""
            )
            experience = self.storage.get_experience(experience_id)
            if (
                not experience
                or experience.get("status") != "completed"
            ):
                continue
            chroma_candidates += 1
            merged = experience_map.get(experience_id, {})
            merged.update(deepcopy(experience))
            experience_map[experience_id] = merged

        summaries = [
            {
                "experience_id": experience_id,
                "topic": experience.get("topic", ""),
                "core_entity": experience.get("core_entity", ""),
                "summary_json": experience.get("summary") or {},
                "status": experience.get("status", ""),
            }
            for experience_id, experience in experience_map.items()
        ]

        historical_segments = self.storage.list_segments_by_experience_ids(
            list(experience_map)
        )

        prompt = ""
        prompt_tokens = 0
        history_experience: dict[str, Any] = {}
        llm_called = False
        llm_fallback = False
        if summaries:
            prompt = build_historical_experience_prompt(
                current_topic=topic,
                current_core_entity=core_entity,
                current_intent=intent,
                current_context=query,
                historical_experiences=summaries,
                historical_segments=historical_segments,
            )
            try:
                import tiktoken

                prompt_tokens = len(
                    tiktoken.get_encoding("cl100k_base").encode(prompt)
                )
            except Exception:
                prompt_tokens = max(1, (len(prompt) + 3) // 4)

            fallback_payload = {
                "relevance_score": 0.0,
                "prior_context": "候选历史任务尚未完成可迁移性判断。",
                "reusable_knowledge": [],
                "prior_outcome": "",
                "applicable_condition": [],
                "provenance": {
                    "experience_ids": [
                        item["experience_id"] for item in summaries
                    ]
                },
            }
            fallback = json.dumps(fallback_payload, ensure_ascii=False)
            try:
                from .summarizer import LLMSummarizer

                llm_called = True
                response_text = LLMSummarizer()._generate_summary(
                    prompt, fallback=fallback
                ).strip()
                parsed = json.loads(response_text)
                history_experience = (
                    parsed if isinstance(parsed, dict) else fallback_payload
                )
                llm_fallback = response_text == fallback
            except Exception:
                history_experience = fallback_payload
                llm_fallback = True
                logger.warning(
                    "Experience summary compression failed; using fallback",
                    exc_info=True,
                )
        print(
            "检索结果：experiences:"
            f"{json.dumps(experience_map, ensure_ascii=False, indent=2)}\n"
            "history_experience:"
            f"{json.dumps(history_experience, ensure_ascii=False, indent=2)}"
        )
        return {
            "experiences": experience_map,
            "history_experience": history_experience,
            "debug": {
                "sqlite_candidates": len(sql_rows),
                "chroma_candidates": chroma_candidates,
                "merged_experiences": len(experience_map),
                "llm_called": llm_called,
                "llm_fallback": llm_fallback,
                "prompt": prompt,
                "prompt_tokens": prompt_tokens,
                "total_recall_ms": round(
                    (time.perf_counter() - started_at) * 1000, 3
                ),
            },
        }


__all__ = ["ExperienceRecaller"]
