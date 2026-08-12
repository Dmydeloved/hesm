"""Public HESM application service for memory ingestion and retrieval."""

from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Any

from .config import PROJECT_ROOT, load_config
from .embedder import BailianEmbedder
from .extractor import TopicExtractor
from .manager import MemoryManager
from .retriever import HybridRetriever
from .storage import MemoryStorage
from .summarizer import LLMSummarizer
from .vector_store import ChromaVectorStore


class HESMService:
    """Own all production HESM components behind two public operations."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.config = load_config(config_path)
        self._lock = RLock()

        paths = self.config.get("paths", {})
        database_path = self._project_path(paths.get("memory_db", "memory/hesm.sqlite3"))
        chroma_path = self._project_path(paths.get("chroma", "memory/chroma"))

        embedding_config = self.config.get("embedding", {})
        topic_config = self.config.get("topic_extraction", {})
        summary_config = self.config.get("summarization", {})
        retrieval_config = self.config.get("retrieval", {})
        management_config = self.config.get("memory_management", {})
        self.api_config = self.config.get("api", {})

        self.storage = MemoryStorage(
            database_path,
            check_same_thread=False,
        )
        self.vector_store = ChromaVectorStore(persist_path=chroma_path)
        self.embedder = BailianEmbedder(
            api_key=embedding_config.get("api_key"),
            model=embedding_config.get("model"),
            base_url=embedding_config.get("base_url"),
        )
        self.extractor = TopicExtractor(
            api_key=topic_config.get("api_key"),
            model=topic_config.get("model"),
            base_url=topic_config.get("base_url"),
            max_retries=topic_config.get("max_retries"),
            retry_delay=topic_config.get("retry_delay"),
        )
        summarizer = LLMSummarizer(
            api_key=summary_config.get("api_key"),
            model=summary_config.get("model"),
            base_url=summary_config.get("base_url"),
            max_retries=summary_config.get("max_retries"),
            retry_delay=summary_config.get("retry_delay"),
        )
        self.manager = MemoryManager(
            storage=self.storage,
            vector_store=self.vector_store,
            embedder=self.embedder,
            summarizer=summarizer,
            segment_summary_qa_threshold=int(
                management_config.get("segment_qa_threshold", 5)
            ),
            experience_summary_segment_threshold=int(
                management_config.get("experience_segment_threshold", 5)
            ),
            experience_similarity_threshold=float(
                management_config.get("experience_similarity_threshold", 0.82)
            ),
            min_segment_qas=int(management_config.get("min_segment_qas", 2)),
        )
        self.retriever = HybridRetriever(
            storage=self.storage,
            vector_store=self.vector_store,
            embedder=self.embedder,
            rerank_with_llm=True,
            retrieval_api_key=retrieval_config.get("api_key"),
            retrieval_model=retrieval_config.get("model"),
            retrieval_base_url=retrieval_config.get("base_url"),
            retrieval_max_retries=retrieval_config.get("max_retries"),
            retrieval_retry_delay=retrieval_config.get("retry_delay"),
            retrieval_config=retrieval_config,
        )

    def add_memory(
        self,
        *,
        user_input: str,
        assistant_output: str = "",
        context: str = "",
        topic_result: dict[str, Any] | list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        timestamp: str | None = None,
        state_key: str = "default",
    ) -> dict[str, Any]:
        """Extract and persist one interaction into hierarchical memory."""
        text = str(user_input).strip()
        if not text:
            raise ValueError("user_input must not be empty")
        with self._lock:
            extracted = topic_result or self.extractor.extract(
                user_input=text,
                context=str(context or ""),
            )
            topic_results = extracted if isinstance(extracted, list) else [extracted]
            if not topic_results or not all(isinstance(item, dict) for item in topic_results):
                raise ValueError("Topic extraction returned no valid records")
            memories = [
                self.manager.add_qa(
                    topic_result=item,
                    user_input=text,
                    assistant_output=str(assistant_output or ""),
                    tools=tools,
                    timestamp=timestamp,
                    state_key=state_key,
                )
                for item in topic_results
            ]
        return {
            "memories": memories,
            "topic_results": topic_results,
            "state_key": state_key,
        }

    def retrieve(
        self,
        *,
        question: str,
        top_experience: int | None = None,
        top_segment: int | None = None,
        top_qa: int | None = None,
    ) -> dict[str, Any]:
        """Extract a query and retrieve a bounded hierarchical memory tree."""
        text = str(question).strip()
        if not text:
            raise ValueError("question must not be empty")
        limits = {
            "top_experience": int(
                top_experience or self.api_config.get("top_experience", 2)
            ),
            "top_segment": int(top_segment or self.api_config.get("top_segment", 3)),
            "top_qa": int(top_qa or self.api_config.get("top_qa", 8)),
        }
        with self._lock:
            extracted = self.extractor.extract(user_input=text)
            candidates = extracted if isinstance(extracted, list) else [extracted]
            if not candidates or not isinstance(candidates[0], dict):
                raise ValueError("Topic extraction returned no valid query")
            primary = candidates[0]
            result = self.retriever.recall(
                topic=str(primary.get("topic") or ""),
                core_entity=str(primary.get("core_entity") or ""),
                intent=str(primary.get("intent") or ""),
                entities=list(primary.get("entities") or []),
                query=text,
                query_confidence=float(primary.get("confidence") or 0.0),
                **limits,
            )
        return {
            "question": text,
            "query_extraction": primary,
            "query_candidates": candidates,
            "limits": limits,
            **result,
        }

    def close(self) -> None:
        with self._lock:
            self.storage.close()

    @staticmethod
    def _project_path(value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else PROJECT_ROOT / path


__all__ = ["HESMService"]
