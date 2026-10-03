"""Thin user-scoped OmniMemEval adapter around native HESM operations."""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from core.retriever import ReadOnlyHybridRetriever
from service.hesm_service import HESMService


REQUEST_SCHEMA = """
PRAGMA journal_mode = WAL;
CREATE TABLE IF NOT EXISTS request_log (
    request_id TEXT PRIMARY KEY,
    request_hash TEXT NOT NULL,
    operation TEXT NOT NULL,
    user_hash TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('processing', 'success', 'failed')),
    idempotent_hit INTEGER NOT NULL DEFAULT 0,
    original_request_id TEXT,
    result_count INTEGER NOT NULL DEFAULT 0,
    error_type TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_request_idempotency
ON request_log (operation, user_hash, request_hash, status);

CREATE TABLE IF NOT EXISTS topic_extraction_history (
    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_hash TEXT NOT NULL,
    source_id TEXT NOT NULL UNIQUE,
    session_id TEXT NOT NULL DEFAULT '',
    topic_results_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_topic_extraction_history_user
ON topic_extraction_history (user_hash, history_id DESC);

CREATE TABLE IF NOT EXISTS evidence_topic_map (
    user_hash TEXT NOT NULL,
    evidence_key TEXT NOT NULL,
    source_id TEXT NOT NULL,
    topic_results_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (user_hash, evidence_key)
);
CREATE INDEX IF NOT EXISTS idx_evidence_topic_map_source
ON evidence_topic_map (user_hash, source_id);
"""

TOPIC_EXTRACTION_HISTORY_LIMIT = 5
logger = logging.getLogger(__name__)


class EvaluationBackend:
    """Isolate native HESM stores by dataset and OmniMemEval ``user_id``."""

    def __init__(self, settings, dataset_id: str):
        self.settings = settings
        self.dataset_id = str(dataset_id).strip()
        if not self.dataset_id:
            raise ValueError("dataset_id must not be empty")
        self.root = Path(settings["data_root"]) / self.dataset_id
        self.users_root = self.root / "users"
        self.users_root.mkdir(parents=True, exist_ok=True)
        self.request_db_path = self.root / "requests.sqlite3"
        self._registry_lock = threading.RLock()
        self._services: dict[str, HESMService] = {}
        self._write_locks: dict[str, threading.RLock] = {}
        with self._request_connection() as connection:
            connection.executescript(REQUEST_SCHEMA)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _user_id(user_id) -> str:
        value = str(user_id or "").strip()
        if not value:
            raise ValueError("user_id must be nonempty")
        return value

    @staticmethod
    def _digest(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @classmethod
    def _content_hash(cls, operation: str, user_id: str, payload) -> str:
        canonical = json.dumps(
            {"operation": operation, "user_id": user_id, "payload": payload},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return cls._digest(canonical)

    def _request_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.request_db_path, timeout=60)
        connection.row_factory = sqlite3.Row
        return connection

    def _write_lock(self, user_hash: str) -> threading.RLock:
        with self._registry_lock:
            return self._write_locks.setdefault(user_hash, threading.RLock())

    def _service(self, user_hash: str) -> HESMService:
        with self._registry_lock:
            service = self._services.get(user_hash)
        if service is not None:
            return service

        # Callers hold the user-specific lock, so different users can initialize
        # concurrently while one user can create only one HESMService.
        user_root = self.users_root / user_hash
        service = HESMService(
            config_data=self.settings["native_config"],
            storage_root=user_root,
            retriever_class=ReadOnlyHybridRetriever,
        )
        with self._registry_lock:
            self._services[user_hash] = service
        return service

    def _start_request(
        self,
        operation: str,
        user_hash: str,
        request_hash: str,
        *,
        idempotent: bool,
    ) -> tuple[str, sqlite3.Row | None]:
        request_id = uuid.uuid4().hex
        now = self._now()
        with self._request_connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            previous = None
            if idempotent:
                previous = connection.execute(
                    """
                    SELECT request_id, result_count
                    FROM request_log
                    WHERE operation = ? AND user_hash = ? AND request_hash = ?
                      AND status = 'success'
                    ORDER BY finished_at DESC
                    LIMIT 1
                    """,
                    (operation, user_hash, request_hash),
                ).fetchone()
            if previous is not None:
                connection.execute(
                    """
                    INSERT INTO request_log (
                        request_id, request_hash, operation, user_hash, status,
                        idempotent_hit, original_request_id, result_count,
                        started_at, finished_at
                    ) VALUES (?, ?, ?, ?, 'success', 1, ?, ?, ?, ?)
                    """,
                    (
                        request_id, request_hash, operation, user_hash,
                        previous["request_id"], previous["result_count"], now, now,
                    ),
                )
            else:
                connection.execute(
                    """
                    INSERT INTO request_log (
                        request_id, request_hash, operation, user_hash,
                        status, started_at
                    ) VALUES (?, ?, ?, ?, 'processing', ?)
                    """,
                    (request_id, request_hash, operation, user_hash, now),
                )
        return request_id, previous

    def _finish_request(
        self,
        request_id: str,
        status: str,
        *,
        result_count: int = 0,
        error_type: str | None = None,
    ) -> None:
        with self._request_connection() as connection:
            connection.execute(
                """
                UPDATE request_log
                SET status = ?, result_count = ?, error_type = ?, finished_at = ?
                WHERE request_id = ?
                """,
                (status, result_count, error_type, self._now(), request_id),
            )

    def _recent_topic_extraction_context(self, user_hash: str) -> str:
        """Return the last five successful extraction rounds for one user."""
        with self._request_connection() as connection:
            rows = connection.execute(
                """
                SELECT topic_results_json
                FROM topic_extraction_history
                WHERE user_hash = ?
                ORDER BY history_id DESC
                LIMIT ?
                """,
                (user_hash, TOPIC_EXTRACTION_HISTORY_LIMIT),
            ).fetchall()
        if not rows:
            return ""
        results = [json.loads(row["topic_results_json"]) for row in reversed(rows)]
        return (
            "最近5轮主题提取结果（按时间从早到晚，仅供当前主题提取参考）：\n"
            + json.dumps(results, ensure_ascii=False, separators=(",", ":"))
        )

    def _record_topic_extraction(
        self,
        *,
        user_hash: str,
        source_id: str,
        session_id: str,
        topic_results,
    ) -> None:
        """Persist one successful extraction round in the experiment store."""
        if isinstance(topic_results, dict):
            topic_results = [topic_results]
        if not isinstance(topic_results, list) or not topic_results:
            raise RuntimeError("HESM add_memory returned no topic extraction results")
        with self._request_connection() as connection:
            connection.execute(
                """
                INSERT INTO topic_extraction_history (
                    user_hash, source_id, session_id, topic_results_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(source_id) DO NOTHING
                """,
                (
                    user_hash,
                    source_id,
                    str(session_id or ""),
                    json.dumps(topic_results, ensure_ascii=False),
                    self._now(),
                ),
            )

    @staticmethod
    def _normalize_evidence(evidence) -> list[str]:
        if evidence is None:
            return []
        values = evidence if isinstance(evidence, (list, tuple, set)) else [evidence]
        normalized = []
        for item in values:
            if isinstance(item, dict):
                item = (
                    item.get("evidence_key")
                    or item.get("evidence_id")
                    or item.get("id")
                    or item.get("key")
                )
            value = str(item or "").strip()
            if value and value not in normalized:
                normalized.append(value)
        return normalized

    def _record_evidence_topic_results(
        self,
        *,
        user_hash: str,
        source_id: str,
        evidence,
        topic_results,
    ) -> None:
        evidence_keys = self._normalize_evidence(evidence)
        if not evidence_keys:
            return
        payload = json.dumps(topic_results, ensure_ascii=False)
        with self._request_connection() as connection:
            connection.executemany(
                """
                INSERT INTO evidence_topic_map (
                    user_hash, evidence_key, source_id, topic_results_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_hash, evidence_key) DO UPDATE SET
                    source_id = excluded.source_id,
                    topic_results_json = excluded.topic_results_json,
                    created_at = excluded.created_at
                """,
                [
                    (user_hash, key, source_id, payload, self._now())
                    for key in evidence_keys
                ],
            )

    def _topic_results_for_evidence(self, user_hash: str, evidence) -> list[dict]:
        # A retrieval query is anchored to its first evidence item only.  Keep
        # this constraint here as well as at the call site so future callers
        # cannot accidentally merge topic records from multiple turns.
        evidence_keys = self._normalize_evidence(evidence)[:1]
        if not evidence_keys:
            return []
        # Prefer the native HESM database: the topic/entity values used here
        # are the exact values stored on the source QA rows.  The adapter index
        # below remains a compatibility fallback for older isolated stores.
        database_path = self.users_root / user_hash / "hesm.sqlite3"
        native_by_key: dict[str, dict] = {}
        if database_path.exists():
            with sqlite3.connect(database_path, timeout=60) as connection:
                connection.row_factory = sqlite3.Row
                placeholders = ",".join("?" for _ in evidence_keys)
                rows = connection.execute(
                    f"""
                    SELECT source_id, topic, core_entity, intent, entities_json,
                           confidence, reason
                    FROM qa_memory
                    WHERE source_id IN ({placeholders})
                    """,
                    tuple(f"evidence:{key}" for key in evidence_keys),
                ).fetchall()
            for row in rows:
                source_id = str(row["source_id"] or "")
                native_by_key[source_id.removeprefix("evidence:")] = {
                    "topic": row["topic"],
                    "core_entity": row["core_entity"],
                    "intent": row["intent"],
                    "entities": json.loads(row["entities_json"] or "[]"),
                    "confidence": row["confidence"],
                    "reasoning": row["reason"],
                }
        if len(native_by_key) == len(evidence_keys):
            return [native_by_key[key] for key in evidence_keys]
        placeholders = ",".join("?" for _ in evidence_keys)
        with self._request_connection() as connection:
            rows = connection.execute(
                f"""
                SELECT evidence_key, topic_results_json
                FROM evidence_topic_map
                WHERE user_hash = ? AND evidence_key IN ({placeholders})
                """,
                (user_hash, *evidence_keys),
            ).fetchall()
        by_key = {str(row["evidence_key"]): row for row in rows}
        missing = [key for key in evidence_keys if key not in by_key]
        if missing:
            raise ValueError(
                "No extracted topic/entity record for evidence: "
                + ", ".join(missing)
            )
        results = []
        for key in evidence_keys:
            values = json.loads(by_key[key]["topic_results_json"])
            if isinstance(values, dict):
                values = [values]
            if not isinstance(values, list) or not values:
                raise RuntimeError(f"Invalid topic/entity record for evidence {key}")
            results.extend(item for item in values if isinstance(item, dict))
        if not results:
            raise RuntimeError("Evidence lookup returned no valid topic/entity records")
        return results

    def _wait_for_outbox(self, user_hash: str) -> None:
        """Wait until experiment-only memory derivations are queryable."""
        database_path = self.users_root / user_hash / "hesm.sqlite3"
        if not database_path.exists():
            return
        timeout = float(self.settings.get("outbox_settle_timeout_seconds", 3600))
        deadline = time.monotonic() + max(1.0, timeout)
        while True:
            with sqlite3.connect(database_path, timeout=60) as connection:
                rows = connection.execute(
                    """
                    SELECT status, COUNT(*)
                    FROM memory_outbox
                    WHERE status IN ('pending', 'processing', 'failed')
                    GROUP BY status
                    """
                ).fetchall()
            counts = {str(status): int(count) for status, count in rows}
            if counts.get("failed", 0):
                raise RuntimeError(
                    "HESM derivation outbox contains failed jobs: "
                    f"{counts['failed']}"
                )
            if not counts.get("pending", 0) and not counts.get("processing", 0):
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Timed out waiting for HESM derivation outbox to settle "
                    f"(pending={counts.get('pending', 0)}, "
                    f"processing={counts.get('processing', 0)})"
                )
            time.sleep(0.1)

    def add(self, user_id, messages, session_id=""):
        """Apply request idempotency, then call native HESM ``add_memory``."""
        raw_user_id = str(user_id or "").strip()
        user_hash = self._digest(raw_user_id)
        request_hash = self._content_hash("add", raw_user_id, messages)
        write_lock = self._write_lock(user_hash)
        with write_lock:
            request_id, previous = self._start_request(
                "add", user_hash, request_hash, idempotent=True
            )
            if previous is not None:
                return {
                    "status": "success",
                    "messages_added": int(previous["result_count"]),
                    "idempotent": True,
                    "request_hash": request_hash,
                }

            added = 0
            try:
                user_id = self._user_id(raw_user_id)
                if not isinstance(messages, list) or not messages:
                    raise ValueError("messages must be a nonempty list")
                for message in messages:
                    if not isinstance(message, dict):
                        raise ValueError("Each message must be an object")
                    if not isinstance(message.get("content"), str):
                        raise ValueError("Each message requires string content")
                service = self._service(user_hash)
                for index, message in enumerate(messages):
                    content = message["content"]
                    if not content.strip():
                        continue
                    source_id = f"{request_hash}:{index}"
                    evidence_keys = self._normalize_evidence(
                        message.get("evidence")
                        or message.get("evidence_id")
                        or message.get("evidence_key")
                    )
                    if evidence_keys:
                        source_id = f"evidence:{evidence_keys[0]}"
                    result = service.add_memory(
                        user_input=content,
                        assistant_output="",
                        context=self._recent_topic_extraction_context(user_hash),
                        tools=message.get("tools") or [],
                        timestamp=message.get("chat_time"),
                        state_key="evaluation",
                        source_id=source_id,
                    )
                    self._record_topic_extraction(
                        user_hash=user_hash,
                        source_id=source_id,
                        session_id=session_id,
                        topic_results=result.get("topic_results") or [],
                    )
                    self._record_evidence_topic_results(
                        user_hash=user_hash,
                        source_id=source_id,
                        evidence=message.get("evidence")
                        or message.get("evidence_id")
                        or message.get("evidence_key"),
                        topic_results=result.get("topic_results") or [],
                    )
                    added += 1
            except Exception as exc:
                self._finish_request(
                    request_id, "failed", result_count=added,
                    error_type=type(exc).__name__,
                )
                raise
            self._finish_request(request_id, "success", result_count=added)
            return {
                "status": "success",
                "messages_added": added,
                "idempotent": False,
                "request_hash": request_hash,
            }

    def search(self, user_id, query, top_k=20, question_date=None, evidence=None):
        """Retrieve only from the HESM store belonging to ``user_id``."""
        raw_user_id = str(user_id or "").strip()
        user_hash = self._digest(raw_user_id)
        request_hash = self._content_hash(
            "search",
            raw_user_id,
            {
                "query": query,
                "top_k": top_k,
                "question_date": question_date,
                "evidence": self._normalize_evidence(evidence),
            },
        )
        request_id, _ = self._start_request(
            "search", user_hash, request_hash, idempotent=False
        )
        try:
            user_id = self._user_id(raw_user_id)
            if not isinstance(query, str) or not query.strip():
                raise ValueError("query must be nonempty")
            if type(top_k) is not int or not 1 <= top_k <= self.settings["max_top_k"]:
                raise ValueError("top_k outside configured range")
            evidence_keys = self._normalize_evidence(evidence)
            evidence_lookup = (
                str(self.settings.get("retrieval_topic_source", "llm"))
                == "evidence_topic_lookup"
            )
            topic_result = None
            if evidence_lookup and evidence_keys:
                try:
                    topic_result = self._topic_results_for_evidence(
                        user_hash, evidence_keys[:1]
                    )
                except Exception as exc:
                    # Missing evidence rows and SQLite/JSON lookup failures are
                    # non-fatal: native HESM retrieval falls back to its LLM
                    # topic extractor when topic_result is None.
                    logger.warning(
                        "Evidence topic lookup failed; falling back to LLM "
                        "user_hash=%s evidence=%s error=%s",
                        user_hash,
                        evidence_keys[0],
                        type(exc).__name__,
                    )
            # The experiment requires a complete-memory barrier before scoring.
            # Holding the user lock also prevents a concurrent add from opening
            # a new derivation window between the barrier and retrieval.
            with self._write_lock(user_hash):
                service = self._service(user_hash)
                self._wait_for_outbox(user_hash)
                result = service.retrieve(
                    question=query,
                    state_key="evaluation",
                    extraction_context=(
                        f"Question date: {question_date}" if question_date else ""
                    ),
                    topic_result=topic_result,
                )
        except Exception as exc:
            self._finish_request(
                request_id, "failed", error_type=type(exc).__name__
            )
            raise
        self._finish_request(
            request_id,
            "success",
            result_count=len(result.get("qas") or []),
        )
        return {
            "context": result.get("context") or "",
            "route_status": result.get("route_status", ""),
            "query_extraction": result.get("query_extraction") or {},
            "candidate_counts": {
                "experiences": len(result.get("experiences") or []),
                "segments": len(result.get("segments") or []),
                "qas": len(result.get("qas") or []),
            },
            "query_date": question_date,
            "evidence": self._normalize_evidence(evidence),
            "topic_extraction_source": result.get(
                "topic_extraction_source", "llm"
            ),
            "request_hash": request_hash,
            "config_fingerprint": self.settings["config_fingerprint"],
        }

    def delete(self, user_id):
        """Keep OmniMemEval cleanup compatible without deleting HESM memory."""
        raw_user_id = str(user_id or "").strip()
        user_hash = self._digest(raw_user_id)
        request_hash = self._content_hash("delete", raw_user_id, {})
        request_id, _ = self._start_request(
            "delete", user_hash, request_hash, idempotent=False
        )
        try:
            self._user_id(raw_user_id)
        except Exception as exc:
            self._finish_request(
                request_id, "failed", error_type=type(exc).__name__
            )
            raise
        self._finish_request(request_id, "success")
        return {
            "status": "ignored_user_store_preserved",
            "request_hash": request_hash,
        }

    def health(self):
        capabilities = [
            "dataset_isolation", "user_isolation", "native_add",
            "readonly_retrieval", "request_hash_idempotency",
            "five_round_topic_extraction_context",
            "derivation_completion_barrier",
        ]
        if self.settings.get("retrieval_topic_source") == "evidence_topic_lookup":
            capabilities.extend([
                "evidence_topic_lookup", "evidence_lookup_llm_fallback",
                "native_qa_topic_entity_source",
            ])
        return {
            "status": "ok",
            "profile": self.settings["profile"],
            "dataset_id": self.dataset_id,
            "config_fingerprint": self.settings["config_fingerprint"],
            "capabilities": capabilities,
        }

    def close(self):
        with self._registry_lock:
            services = list(self._services.values())
            self._services.clear()
        for service in services:
            service.close()
