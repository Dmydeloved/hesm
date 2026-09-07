"""Public HESM application service for memory ingestion and retrieval."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from threading import RLock
import time
from typing import Any, Iterator

from core.chat import LLMAnswerer, build_chat_prompt
from core.config import PROJECT_ROOT, load_config
from core.embedder import BailianEmbedder
from core.extractor import TopicExtractor
from core.logging_config import configure_logging
from core.manager import MemoryManager
from core.recaller import ExperienceRecaller
from core.retriever import HybridRetriever
from core.session import SessionManager
from core.storage import MemoryStorage
from core.summarizer import LLMSummarizer
from core.vector_store import ChromaVectorStore


logger = logging.getLogger(__name__)


class HESMService:
    """Own all production HESM components behind two public operations."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        configure_logging()
        self.config = load_config(config_path)
        self._lock = RLock()
        logger.info("Initializing HESMService config_path=%s", config_path or "default")

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
        # 启动时同步迁移已有向量元数据，保证 SQLite 与 Chroma 的时间格式一致。
        self.vector_store.normalize_timestamps()
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
            experience_recaller=self.recaller,
        )
        self.retriever = HybridRetriever(
            manager=self.manager,
        )
        self.answerer = LLMAnswerer(
            api_key=str(chat_config.get("api_key") or ""),
            model=str(chat_config.get("model") or ""),
            base_url=str(chat_config.get("base_url") or ""),
            max_retries=int(chat_config.get("max_retries", 3)),
            retry_delay=float(chat_config.get("retry_delay", 2.0)),
        )
        self.chat_model = str(chat_config.get("model") or "")
        logger.info(
            "HESMService initialized db=%s chroma=%s chat_model=%s",
            database_path,
            chroma_path,
            self.chat_model,
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
        source_id: str | None = None,
    ) -> dict[str, Any]:
        """Extract and persist one interaction into hierarchical memory."""
        text = str(user_input).strip()
        if not text:
            raise ValueError("user_input must not be empty")
        logger.info(
            "Add memory started state_key=%s user_input=%s",
            state_key,
            text,
        )
        with self._lock:
            extracted = topic_result or self.extractor.extract(
                user_input=text,
                context=str(context or ""),
            )
            topic_results = extracted if isinstance(extracted, list) else [extracted]
            if not topic_results or not all(isinstance(item, dict) for item in topic_results):
                raise ValueError("Topic extraction returned no valid records")
            logger.info(
                "Add memory topic extraction result state_key=%s topic_results=%s",
                state_key,
                json.dumps(topic_results, ensure_ascii=False),
            )
            memories = [
                self.manager.add_qa(
                    topic_result=item,
                    user_input=text,
                    assistant_output=str(assistant_output or ""),
                    tools=tools,
                    timestamp=timestamp,
                    state_key=state_key,
                    source_id=source_id,
                )
                for item in topic_results
            ]
        logger.info(
            "Add memory completed state_key=%s memory_count=%s memories=%s",
            state_key,
            len(memories),
            json.dumps(memories, ensure_ascii=False),
        )
        return {
            "memories": memories,
            "topic_results": topic_results,
            "state_key": state_key,
        }

    def retrieve(
        self,
        *,
        question: str,
        state_key: str = "default",
        extraction_context: str = "",
    ) -> dict[str, Any]:
        """Extract a query and retrieve a bounded hierarchical memory tree."""
        text = str(question).strip()
        if not text:
            raise ValueError("question must not be empty")
        logger.info("Retrieve started state_key=%s question=%s", state_key, text)
        started_at = time.perf_counter()
        with self._lock:
            extraction_started_at = time.perf_counter()
            extracted = self.extractor.extract(
                user_input=text,
                context=str(extraction_context or ""),
            )
            extraction_ms = round(
                (time.perf_counter() - extraction_started_at) * 1000, 3
            )
            candidates = extracted if isinstance(extracted, list) else [extracted]
            if not candidates or not isinstance(candidates[0], dict):
                raise ValueError("Topic extraction returned no valid query")
            primary = candidates[0]
            logger.info(
                "Retrieve extraction completed state_key=%s timing_ms=%s extraction=%s",
                state_key,
                extraction_ms,
                json.dumps(primary, ensure_ascii=False),
            )
            retrieval_started_at = time.perf_counter()
            result = self.retriever.retriever(
                topic=str(primary.get("topic") or ""),
                core_entity=str(primary.get("core_entity") or ""),
                query=text,
                intent=str(primary.get("intent") or ""),
                state_key=state_key,
            )
            retrieval_ms = round(
                (time.perf_counter() - retrieval_started_at) * 1000, 3
            )
            logger.info(
                "Retrieve memory completed state_key=%s timing_ms=%s experience_count=%s segment_count=%s qa_count=%s",
                state_key,
                retrieval_ms,
                len(result.get("experiences") or []),
                len(result.get("segments") or []),
                len(result.get("qas") or []),
            )
        total_ms = round((time.perf_counter() - started_at) * 1000, 3)
        logger.info("Retrieve completed state_key=%s total_ms=%s", state_key, total_ms)
        return {
            "question": text,
            "query_extraction": primary,
            "query_candidates": candidates,
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
        history: list[dict[str, Any]] | None = None,
        state_key: str = "default",
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Retrieve memory, answer the user, then persist the complete turn."""
        final_result: dict[str, Any] | None = None
        for event in self.chat_events(
            message=message,
            history=history,
            state_key=state_key,
            session_id=session_id,
        ):
            if event.get("event") == "final":
                final_result = event["result"]
        if final_result is None:
            raise RuntimeError("Chat completed without final result")
        return final_result

    def chat_events(
        self,
        *,
        message: str,
        history: list[dict[str, Any]] | None = None,
        state_key: str = "default",
        session_id: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield real chat processing milestones as each backend stage completes."""
        text = str(message).strip()
        if not text:
            raise ValueError("message must not be empty")
        session_identifier = str(session_id or state_key or "web_chat")
        logger.info(
            "Chat events started session_id=%s state_key=%s message=%s",
            session_identifier,
            state_key,
            text,
        )
        session = self.sessions.ensure(session_identifier)
        persisted_history = session.get("messages") or []
        requested_history = history or []
        # The browser only needs to submit session_id. history is retained as a
        # compatibility fallback for older callers whose session has not been saved.
        history_source = persisted_history or requested_history
        recent_turns = self._recent_session_turns(
            history_source,
            session.get("metadata") or {},
            limit=5,
        )
        extraction_context = (
            json.dumps(recent_turns, ensure_ascii=False, indent=2)
            if recent_turns else ""
        )

        total_started_at = time.perf_counter()
        timings: dict[str, float] = {}
        yield {
            "event": "start",
            "session_id": session_identifier,
            "history_turn_count": len(recent_turns),
        }
        with self._lock:
            yield {
                "event": "stage_started",
                "stage": "extract",
                "session_id": session_identifier,
                "history_turn_count": len(recent_turns),
            }
            extraction_started_at = time.perf_counter()
            extracted = self.extractor.extract(
                user_input=text,
                context=extraction_context,
            )
            timings["topic_extraction_ms"] = round(
                (time.perf_counter() - extraction_started_at) * 1000, 3
            )
            candidates = extracted if isinstance(extracted, list) else [extracted]
            if not candidates or not isinstance(candidates[0], dict):
                raise ValueError("Topic extraction returned no valid query")
            primary = candidates[0]
            logger.info(
                "Chat extract completed session_id=%s timing_ms=%s extraction=%s candidates=%s",
                session_identifier,
                timings["topic_extraction_ms"],
                json.dumps(primary, ensure_ascii=False),
                json.dumps(candidates, ensure_ascii=False),
            )
            yield {
                "event": "stage_completed",
                "stage": "extract",
                "timing_ms": timings["topic_extraction_ms"],
                "extraction": primary,
                "query_candidates": candidates,
                "history_turn_count": len(recent_turns),
            }

            yield {"event": "stage_started", "stage": "retrieve"}
            retrieval_started_at = time.perf_counter()
            retrieval_payload = self.retriever.retriever(
                topic=str(primary.get("topic") or ""),
                core_entity=str(primary.get("core_entity") or ""),
                query=text,
                intent=str(primary.get("intent") or ""),
                state_key=session_identifier,
            )
            timings["retrieval_ms"] = round(
                (time.perf_counter() - retrieval_started_at) * 1000, 3
            )
            retrieval_result = {
                "question": text,
                "query_extraction": primary,
                "query_candidates": candidates,
                "timing": {
                    "topic_extraction_ms": timings["topic_extraction_ms"],
                    "retrieval_ms": timings["retrieval_ms"],
                },
                **retrieval_payload,
            }
            logger.info(
                "Chat retrieve completed session_id=%s timing_ms=%s experience_count=%s segment_count=%s qa_count=%s",
                session_identifier,
                timings["retrieval_ms"],
                len(retrieval_payload.get("experiences") or []),
                len(retrieval_payload.get("segments") or []),
                len(retrieval_payload.get("qas") or []),
            )
            yield {
                "event": "stage_completed",
                "stage": "retrieve",
                "timing_ms": timings["retrieval_ms"],
                "retrieval": retrieval_result,
            }

            yield {"event": "stage_started", "stage": "prompt"}
            prompt_started_at = time.perf_counter()
            prompt = build_chat_prompt(
                question=text,
                memory_context=retrieval_result.get("context") or "",
            )
            timings["prompt_assembly_ms"] = round(
                (time.perf_counter() - prompt_started_at) * 1000, 3
            )
            logger.info(
                "Chat prompt built session_id=%s timing_ms=%s prompt=%s",
                session_identifier,
                timings["prompt_assembly_ms"],
                prompt,
            )
            yield {
                "event": "stage_completed",
                "stage": "prompt",
                "timing_ms": timings["prompt_assembly_ms"],
                "prompt": prompt,
                "history_turn_count": len(recent_turns),
                "retrieval_qa_count": len(retrieval_result.get("qas") or []),
            }

            yield {"event": "stage_started", "stage": "generate"}
            generation_started_at = time.perf_counter()
            answer = self.answerer.answer(prompt)
            timings["generation_ms"] = round(
                (time.perf_counter() - generation_started_at) * 1000, 3
            )
            logger.info(
                "Chat generation completed session_id=%s timing_ms=%s model=%s answer=%s",
                session_identifier,
                timings["generation_ms"],
                self.chat_model,
                answer,
            )
            yield {
                "event": "stage_completed",
                "stage": "generate",
                "timing_ms": timings["generation_ms"],
                "answer": answer,
                "model": self.chat_model,
            }

            yield {"event": "stage_started", "stage": "store"}
            storage_started_at = time.perf_counter()
            stored = self.add_memory(
                user_input=text,
                assistant_output=answer,
                context=prompt,
                topic_result=retrieval_result["query_extraction"],
                # tools is reserved exclusively for actual tool invocations.
                # Topic extraction, retrieval and prompt diagnostics are not tools.
                tools=[],
                state_key=session_identifier,
            )
            timings["storage_ms"] = round((time.perf_counter() - storage_started_at) * 1000, 3)
            logger.info(
                "Chat store completed session_id=%s timing_ms=%s stored=%s",
                session_identifier,
                timings["storage_ms"],
                json.dumps(stored, ensure_ascii=False),
            )
            self.sessions.append_turn(
                session_identifier,
                user_content=text,
                assistant_content=answer,
                metadata={
                    "last_qa_id": stored["memories"][0].get("qa_id", ""),
                    "model": self.chat_model,
                    "topic": retrieval_result["query_extraction"].get(
                        "topic", ""
                    ),
                    "core_entity": retrieval_result["query_extraction"].get(
                        "core_entity", ""
                    ),
                },
            )
            yield {
                "event": "stage_completed",
                "stage": "store",
                "timing_ms": timings["storage_ms"],
                "stored": stored,
            }

        total_ms = round((time.perf_counter() - total_started_at) * 1000, 3)
        retrieval_result["timing"] = {
            **retrieval_result.get("timing", {}),
            "response_assembly_ms": round(
                max(
                    0.0,
                    total_ms
                    - timings.get("topic_extraction_ms", 0.0)
                    - timings.get("retrieval_ms", 0.0),
                ),
                3,
            ),
            "total_ms": round(
                timings.get("topic_extraction_ms", 0.0)
                + timings.get("retrieval_ms", 0.0),
                3,
            ),
        }
        result = {
            "message": text,
            "answer": answer,
            "prompt": prompt,
            "extraction": retrieval_result["query_extraction"],
            "retrieval": retrieval_result,
            "stored": stored,
            "timing": {
                **retrieval_result.get("timing", {}),
                "prompt_assembly_ms": timings["prompt_assembly_ms"],
                "generation_ms": timings["generation_ms"],
                "storage_ms": timings["storage_ms"],
                "chat_total_ms": total_ms,
            },
            "model": self.chat_model,
            "session_id": session_identifier,
            "history_turn_count": len(recent_turns),
        }
        logger.info(
            "Chat events completed session_id=%s total_ms=%s result=%s",
            session_identifier,
            total_ms,
            json.dumps(result, ensure_ascii=False),
        )
        yield {"event": "final", "result": result}

    def close(self) -> None:
        with self._lock:
            self.storage.close()

    @staticmethod
    def _build_topic_extraction_context(
        messages: list[dict[str, Any]],
        session_metadata: dict[str, Any],
    ) -> str:
        """构造最近五轮对话及其既有主题、核心实体上下文。"""
        turns = HESMService._recent_session_turns(messages, session_metadata, limit=5)
        return json.dumps(turns, ensure_ascii=False, indent=2) if turns else ""

    @staticmethod
    def _recent_session_turns(
        messages: list[dict[str, Any]],
        session_metadata: dict[str, Any],
        *,
        limit: int = 5,
    ) -> list[dict[str, str]]:
        """Return the most recent user/assistant turns from a saved session."""
        turns: list[dict[str, str]] = []
        pending: dict[str, str] | None = None
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "")
            content = str(message.get("content") or "").strip()
            if role not in {"user", "assistant"} or not content:
                continue
            metadata = message.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            topic = str(
                metadata.get("topic") or message.get("topic") or ""
            ).strip()
            core_entity = str(
                metadata.get("core_entity")
                or message.get("core_entity")
                or ""
            ).strip()

            if role == "user":
                if pending:
                    turns.append(pending)
                pending = {
                    "user_input": content[:20_000],
                    "assistant_output": "",
                    "topic": topic,
                    "core_entity": core_entity,
                }
                continue

            if pending is None:
                continue
            pending["assistant_output"] = content[:20_000]
            pending["topic"] = pending["topic"] or topic
            pending["core_entity"] = pending["core_entity"] or core_entity
            turns.append(pending)
            pending = None

        if pending:
            turns.append(pending)
        turns = turns[-max(1, int(limit)):]

        # 兼容尚未在消息中保存主题信息的旧会话，回填最近一轮元数据。
        if turns:
            turns[-1]["topic"] = turns[-1]["topic"] or str(
                session_metadata.get("topic") or ""
            )
            turns[-1]["core_entity"] = turns[-1]["core_entity"] or str(
                session_metadata.get("core_entity") or ""
            )
        return turns

    @staticmethod
    def _project_path(value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else PROJECT_ROOT / path


__all__ = ["HESMService"]
