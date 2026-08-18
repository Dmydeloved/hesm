"""Public HESM application service for memory ingestion and retrieval."""

from __future__ import annotations

from pathlib import Path
from threading import RLock
import time
from typing import Any

from hesm.chat import LLMAnswerer, build_chat_prompt
from hesm.config import PROJECT_ROOT, load_config
from hesm.embedder import BailianEmbedder
from hesm.extractor import TopicExtractor
from hesm.manager import MemoryManager
from hesm.recaller import ExperienceRecaller
from hesm.retriever import HybridRetriever
from hesm.session import SessionManager
from hesm.storage import MemoryStorage
from hesm.summarizer import LLMSummarizer
from hesm.vector_store import ChromaVectorStore


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
        chat_config = self.config.get("chat", summary_config)
        management_config = self.config.get("memory_management", {})
        self.api_config = self.config.get("api", {})

        self.storage = MemoryStorage(
            database_path,
            check_same_thread=False,
        )
        self.sessions = SessionManager(database_path)
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
        self.recaller = ExperienceRecaller(
            storage=self.storage,
            vector_store=self.vector_store,
            embedder=self.embedder,
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
            experience_recall=self.recaller.recall,
        )
        self.retriever = HybridRetriever(
            storage=self.storage,
            create_experience=self.manager.create_experience,
        )
        self.answerer = LLMAnswerer(
            api_key=str(chat_config.get("api_key") or ""),
            model=str(chat_config.get("model") or ""),
            base_url=str(chat_config.get("base_url") or ""),
            max_retries=int(chat_config.get("max_retries", 3)),
            retry_delay=float(chat_config.get("retry_delay", 2.0)),
        )
        self.chat_model = str(chat_config.get("model") or "")

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
    ) -> dict[str, Any]:
        """Extract a query and retrieve a bounded hierarchical memory tree."""
        text = str(question).strip()
        if not text:
            raise ValueError("question must not be empty")
        limits = {
            "top_experience": 1,
            "top_segment": max(1, int(self.api_config.get("top_segment", 2))),
            "top_qa": 4,
        }
        started_at = time.perf_counter()
        with self._lock:
            extraction_started_at = time.perf_counter()
            extracted = self.extractor.extract(user_input=text)
            extraction_ms = round(
                (time.perf_counter() - extraction_started_at) * 1000, 3
            )
            candidates = extracted if isinstance(extracted, list) else [extracted]
            if not candidates or not isinstance(candidates[0], dict):
                raise ValueError("Topic extraction returned no valid query")
            primary = candidates[0]
            retrieval_started_at = time.perf_counter()
            result = self.retriever.retriever(
                topic=str(primary.get("topic") or ""),
                core_entity=str(primary.get("core_entity") or ""),
                query=text,
            )
            retrieval_ms = round(
                (time.perf_counter() - retrieval_started_at) * 1000, 3
            )
        total_ms = round((time.perf_counter() - started_at) * 1000, 3)
        return {
            "question": text,
            "query_extraction": primary,
            "query_candidates": candidates,
            "limits": limits,
            "timing": {
                "topic_extraction_ms": extraction_ms,
                "retrieval_ms": retrieval_ms,
                "response_assembly_ms": round(
                    max(0.0, total_ms - extraction_ms - retrieval_ms), 3
                ),
                "total_ms": total_ms,
            },
            **result,
        }

    def chat(
        self,
        *,
        message: str,
        history: list[dict[str, str]] | None = None,
        state_key: str = "default",
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Retrieve memory, answer the user, then persist the complete turn."""
        text = str(message).strip()
        if not text:
            raise ValueError("message must not be empty")
        session_identifier = str(session_id or state_key or "web_chat")
        session = self.sessions.ensure(session_identifier)
        persisted_history = session.get("messages") or []
        requested_history = history or []
        history_source = persisted_history if persisted_history else requested_history
        normalized_history = [
            {
                "role": str(item.get("role") or ""),
                "content": str(item.get("content") or "")[:20_000],
            }
            for item in history_source[-20:]
            if item.get("role") in {"user", "assistant"}
            and str(item.get("content") or "").strip()
        ]

        total_started_at = time.perf_counter()
        with self._lock:
            retrieval_result = self.retrieve(
                question=text,
            )
            prompt_started_at = time.perf_counter()
            prompt = build_chat_prompt(
                question=text,
                extraction=retrieval_result["query_extraction"],
                memory_context=retrieval_result.get("context") or "",
                history=normalized_history,
            )
            prompt_ms = round((time.perf_counter() - prompt_started_at) * 1000, 3)

            generation_started_at = time.perf_counter()
            answer = self.answerer.answer(prompt)
            generation_ms = round(
                (time.perf_counter() - generation_started_at) * 1000, 3
            )

            storage_started_at = time.perf_counter()
            stored = self.add_memory(
                user_input=text,
                assistant_output=answer,
                context=prompt,
                topic_result=retrieval_result["query_extraction"],
                # tools is reserved exclusively for actual tool invocations.
                # Topic extraction, retrieval and prompt diagnostics are not tools.
                tools=[],
                state_key=state_key,
            )
            storage_ms = round((time.perf_counter() - storage_started_at) * 1000, 3)
            self.sessions.append_turn(
                session_identifier,
                user_content=text,
                assistant_content=answer,
                metadata={
                    "last_qa_id": stored["memories"][0].get("qa_id", ""),
                    "model": self.chat_model,
                },
            )

        total_ms = round((time.perf_counter() - total_started_at) * 1000, 3)
        return {
            "message": text,
            "answer": answer,
            "prompt": prompt,
            "history": normalized_history,
            "extraction": retrieval_result["query_extraction"],
            "retrieval": retrieval_result,
            "stored": stored,
            "timing": {
                **retrieval_result.get("timing", {}),
                "prompt_assembly_ms": prompt_ms,
                "generation_ms": generation_ms,
                "storage_ms": storage_ms,
                "chat_total_ms": total_ms,
            },
            "model": self.chat_model,
            "session_id": session_identifier,
        }

    def close(self) -> None:
        with self._lock:
            self.storage.close()

    @staticmethod
    def _project_path(value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else PROJECT_ROOT / path


__all__ = ["HESMService"]
