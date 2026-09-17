from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from .time_utils import format_timestamp


logger = logging.getLogger(__name__)


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS qa_memory (
    qa_id TEXT PRIMARY KEY,
    source_id TEXT,
    timestamp TEXT NOT NULL,
    user_input TEXT NOT NULL,
    assistant_output TEXT NOT NULL,
    tools_json TEXT NOT NULL,
    topic TEXT NOT NULL,
    intent TEXT NOT NULL,
    core_entity TEXT NOT NULL,
    entities_json TEXT NOT NULL,
    segment_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('open', 'deleted')),
    confidence REAL NOT NULL,
    reason TEXT NOT NULL,
    FOREIGN KEY (segment_id) REFERENCES segment_memory(segment_id)
);

CREATE TABLE IF NOT EXISTS segment_memory (
    segment_id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    intent TEXT NOT NULL,
    core_entity TEXT NOT NULL,
    qa_ids_json TEXT NOT NULL,
    summary_json TEXT NOT NULL,
    experience_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    version INTEGER NOT NULL,
    last_summarized_qa_count INTEGER NOT NULL,
    summarized_qa_ids_json TEXT NOT NULL DEFAULT '[]',
    summary_version INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL CHECK (status IN ('open', 'completed', 'deleted')),
    FOREIGN KEY (experience_id) REFERENCES experience_memory(experience_id)
);

CREATE TABLE IF NOT EXISTS experience_memory (
    experience_id TEXT PRIMARY KEY,
    history_experience_json TEXT NOT NULL DEFAULT '{}',
    topic TEXT NOT NULL,
    core_entity TEXT NOT NULL,
    intents_link_json TEXT NOT NULL,
    segment_ids_json TEXT NOT NULL,
    summary_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    version INTEGER NOT NULL,
    last_summarized_segment_count INTEGER NOT NULL,
    last_summarized_child_revision INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL CHECK (status IN ('open', 'completed', 'deleted'))
);

CREATE TABLE IF NOT EXISTS runtime_state (
    state_key TEXT PRIMARY KEY,
    current_experience_id TEXT NOT NULL,
    current_segment_id TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    retrieval_cache_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS chat_session (
    session_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    messages_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS memory_outbox (
    job_id TEXT PRIMARY KEY,
    job_type TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    target_version INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    retry_count INTEGER NOT NULL DEFAULT 0,
    last_error TEXT NOT NULL DEFAULT '',
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(job_type, memory_type, memory_id)
);

CREATE INDEX IF NOT EXISTS idx_qa_segment_id ON qa_memory(segment_id);
CREATE INDEX IF NOT EXISTS idx_qa_topic_entity_intent
ON qa_memory(topic, core_entity, intent);
CREATE INDEX IF NOT EXISTS idx_segment_experience_id
ON segment_memory(experience_id);
CREATE INDEX IF NOT EXISTS idx_experience_topic_entity
ON experience_memory(topic, core_entity);
CREATE INDEX IF NOT EXISTS idx_experience_core_entity
ON experience_memory(core_entity);
CREATE INDEX IF NOT EXISTS idx_qa_topic
ON qa_memory(topic);
CREATE INDEX IF NOT EXISTS idx_chat_session_updated_at
ON chat_session(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_memory_outbox_status_created
ON memory_outbox(status, created_at);
"""


JSON_FIELDS = {
    "tools_json",
    "entities_json",
    "qa_ids_json",
    "summarized_qa_ids_json",
    "intents_link_json",
    "segment_ids_json",
    "summary_json",
    "history_experience_json",
    "retrieval_cache_json",
    "vector_json",
    "messages_json",
    "metadata_json",
    "payload_json",
}

JSON_DEFAULTS: dict[str, Any] = {
    "tools_json": [],
    "entities_json": [],
    "qa_ids_json": [],
    "summarized_qa_ids_json": [],
    "intents_link_json": [],
    "segment_ids_json": [],
    "summary_json": {},
    "history_experience_json": {},
    "retrieval_cache_json": {},
    "vector_json": [],
    "messages_json": [],
    "metadata_json": {},
}


class MemoryStorage:
    """管理 QA、Segment、Experience 以及运行时状态的 SQLite 仓储。"""

    def __init__(
        self,
        db_path: str | Path,
        *,
        read_only: bool = False,
        check_same_thread: bool = True,
    ) -> None:
        self.db_path = Path(db_path)
        self.read_only = read_only
        if read_only:
            if not self.db_path.exists():
                raise FileNotFoundError(f"Memory database not found: {self.db_path}")
            self.connection = sqlite3.connect(
                f"file:{self.db_path.resolve().as_posix()}?mode=ro",
                uri=True,
                check_same_thread=check_same_thread,
            )
        else:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(
                self.db_path,
                check_same_thread=check_same_thread,
            )
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA busy_timeout = 15000")
        if not read_only:
            self.connection.execute("PRAGMA journal_mode = WAL")
            self.connection.execute("PRAGMA synchronous = NORMAL")
            self.connection.executescript(SCHEMA)
            self._ensure_runtime_state_columns()
            self._ensure_memory_columns()
            self._drop_legacy_qa_search_index()
            self._normalize_stored_timestamps()
            self.connection.commit()
        logger.info("结构化记忆库已初始化 db=%s", self.db_path.resolve())

    def _ensure_runtime_state_columns(self) -> None:
        columns = {
            row["name"]
            for row in self.connection.execute("PRAGMA table_info(runtime_state)").fetchall()
        }
        if "retrieval_cache_json" not in columns:
            self.connection.execute(
                "ALTER TABLE runtime_state "
                "ADD COLUMN retrieval_cache_json TEXT NOT NULL DEFAULT '{}'"
            )

    def _table_columns(self, table: str) -> set[str]:
        """读取表字段，用于兼容已有数据库的增量迁移。"""
        return {
            str(row["name"])
            for row in self.connection.execute(f"PRAGMA table_info({table})").fetchall()
        }

    def _ensure_memory_columns(self) -> None:
        """为旧版数据库补齐文档定义的字段，并迁移可复用的数据。"""
        qa_columns = self._table_columns("qa_memory")
        if "source_id" not in qa_columns:
            self.connection.execute(
                "ALTER TABLE qa_memory ADD COLUMN source_id TEXT"
            )
        if "reason" not in qa_columns:
            self.connection.execute(
                "ALTER TABLE qa_memory ADD COLUMN reason TEXT NOT NULL DEFAULT ''"
            )
            if "reasoning" in qa_columns:
                self.connection.execute(
                    "UPDATE qa_memory SET reason = reasoning WHERE reason = ''"
                )
        cursor = self.connection.execute(
            "UPDATE qa_memory SET status = 'open' WHERE status = 'active'"
        )

        segment_columns = self._table_columns("segment_memory")
        if "summary_json" not in segment_columns:
            self.connection.execute(
                "ALTER TABLE segment_memory "
                "ADD COLUMN summary_json TEXT NOT NULL DEFAULT '{}'"
            )
            if "summary" in segment_columns:
                rows = self.connection.execute(
                    "SELECT segment_id, summary FROM segment_memory"
                ).fetchall()
                for row in rows:
                    payload = self._segment_summary_payload(row["summary"])
                    self.connection.execute(
                        "UPDATE segment_memory SET summary_json = ? WHERE segment_id = ?",
                        (json.dumps(payload, ensure_ascii=False), row["segment_id"]),
                    )
        if "summary_version" not in segment_columns:
            self.connection.execute(
                "ALTER TABLE segment_memory "
                "ADD COLUMN summary_version INTEGER NOT NULL DEFAULT 0"
            )
            self.connection.execute(
                "UPDATE segment_memory SET summary_version = 1 "
                "WHERE last_summarized_qa_count > 0"
            )
        if "summarized_qa_ids_json" not in segment_columns:
            self.connection.execute(
                "ALTER TABLE segment_memory ADD COLUMN "
                "summarized_qa_ids_json TEXT NOT NULL DEFAULT '[]'"
            )
            rows = self.connection.execute(
                "SELECT segment_id, qa_ids_json, last_summarized_qa_count "
                "FROM segment_memory"
            ).fetchall()
            for row in rows:
                try:
                    qa_ids = json.loads(row["qa_ids_json"] or "[]")
                except (json.JSONDecodeError, TypeError):
                    qa_ids = []
                summarized_ids = (
                    qa_ids[: max(0, int(row["last_summarized_qa_count"] or 0))]
                    if isinstance(qa_ids, list)
                    else []
                )
                self.connection.execute(
                    "UPDATE segment_memory SET summarized_qa_ids_json = ? "
                    "WHERE segment_id = ?",
                    (json.dumps(summarized_ids, ensure_ascii=False), row["segment_id"]),
                )

        experience_columns = self._table_columns("experience_memory")
        if "history_experience_json" not in experience_columns:
            self.connection.execute(
                "ALTER TABLE experience_memory "
                "ADD COLUMN history_experience_json TEXT NOT NULL DEFAULT '{}'"
            )
        if "status" not in experience_columns:
            self.connection.execute(
                "ALTER TABLE experience_memory "
                "ADD COLUMN status TEXT NOT NULL DEFAULT 'open'"
            )
            if "state_json" in experience_columns:
                self.connection.execute(
                    """
                    UPDATE experience_memory
                    SET status = CASE
                        WHEN json_extract(state_json, '$.status') = 'completed'
                            THEN 'completed'
                        WHEN json_extract(state_json, '$.status') = 'deleted'
                            THEN 'deleted'
                        ELSE 'open'
                    END
                    """
                )
        if "last_summarized_child_revision" not in experience_columns:
            self.connection.execute(
                "ALTER TABLE experience_memory ADD COLUMN "
                "last_summarized_child_revision INTEGER NOT NULL DEFAULT 0"
            )

        outbox_columns = self._table_columns("memory_outbox")
        if "payload_json" not in outbox_columns:
            self.connection.execute(
                "ALTER TABLE memory_outbox "
                "ADD COLUMN payload_json TEXT NOT NULL DEFAULT '{}'"
            )

    def _drop_legacy_qa_search_index(self) -> None:
        """Remove the retired full-text virtual table from existing databases."""
        self.connection.execute("DROP TABLE IF EXISTS qa_memory_fts")

    def _normalize_stored_timestamps(self) -> None:
        """将已有记忆及会话中的时间统一迁移为标准格式。"""
        time_columns = {
            "qa_memory": ("timestamp",),
            "segment_memory": ("created_at", "updated_at"),
            "experience_memory": ("created_at", "updated_at"),
            "runtime_state": ("updated_at",),
            "chat_session": ("created_at", "updated_at"),
        }
        for table, columns in time_columns.items():
            selected_columns = ", ".join(columns)
            if table == "chat_session":
                selected_columns = f"{selected_columns}, messages_json"
            rows = self.connection.execute(
                f"SELECT rowid, {selected_columns} FROM {table}"
            ).fetchall()
            for row in rows:
                updates: dict[str, str] = {}
                for column in columns:
                    original = str(row[column] or "").strip()
                    if not original:
                        continue
                    try:
                        normalized = format_timestamp(original)
                    except ValueError:
                        logger.warning(
                            "跳过无法识别的历史时间 table=%s column=%s value=%s",
                            table,
                            column,
                            original,
                        )
                        continue
                    if normalized != original:
                        updates[column] = normalized

                # 会话消息的创建时间存放在 JSON 中，也需要随表字段一起迁移。
                if table == "chat_session":
                    try:
                        messages = json.loads(row["messages_json"] or "[]")
                    except (json.JSONDecodeError, TypeError):
                        messages = []
                    messages_changed = False
                    if isinstance(messages, list):
                        for message in messages:
                            if not isinstance(message, dict) or not message.get("created_at"):
                                continue
                            original = str(message["created_at"]).strip()
                            try:
                                normalized = format_timestamp(original)
                            except ValueError:
                                continue
                            if normalized != original:
                                message["created_at"] = normalized
                                messages_changed = True
                    if messages_changed:
                        updates["messages_json"] = json.dumps(
                            messages,
                            ensure_ascii=False,
                        )

                if not updates:
                    continue
                assignments = ", ".join(f"{column} = ?" for column in updates)
                self.connection.execute(
                    f"UPDATE {table} SET {assignments} WHERE rowid = ?",
                    (*updates.values(), row["rowid"]),
                )

    def close(self) -> None:
        self.connection.close()

    def commit(self) -> None:
        self.connection.commit()

    def rollback(self) -> None:
        self.connection.rollback()

    def _row_to_dict(self, row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        data: dict[str, Any] = {}
        for key in row.keys():
            value = row[key]
            if key in JSON_FIELDS:
                default = JSON_DEFAULTS[key]
                try:
                    data[key[:-5]] = json.loads(value) if value else default.copy()
                except (json.JSONDecodeError, TypeError):
                    data[key[:-5]] = default.copy()
            else:
                data[key] = value
        # 旧库可能仍保留旧字段；对外只暴露文档定义的统一名称。
        if "reason" not in data and "reasoning" in data:
            data["reason"] = data["reasoning"]
        if "reasoning" not in data and "reason" in data:
            data["reasoning"] = data["reason"]
        row_keys = set(row.keys())
        if "summary_json" in row_keys and "summary" not in data:
            data["summary"] = {}
        if (
            "status" in data
            and "state" not in data
            and {"history_experience_json", "state_json"}.intersection(row_keys)
        ):
            data["state"] = {
                "status": "in_progress" if data["status"] == "open" else data["status"]
            }
        return data

    @staticmethod
    def _json_object(value: Any) -> dict[str, Any]:
        """把模型文本或字典规范化为可落库的 JSON 对象。"""
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    @classmethod
    def _segment_summary_payload(cls, value: Any) -> dict[str, Any]:
        """生成符合设计文档的 Segment Summary 默认结构。"""
        payload = cls._json_object(value)
        if payload:
            return payload
        conclusion = str(value or "").strip()
        return {
            "goal": "",
            "key_facts": [],
            "state_changes": [],
            "state": {"status": "ongoing", "current_conclusion": conclusion},
        }

    @classmethod
    def _experience_summary_payload(cls, value: Any) -> dict[str, Any]:
        """生成符合设计文档的 Experience Summary 默认结构。"""
        payload = cls._json_object(value)
        if payload:
            return payload
        summary = str(value or "").strip()
        return {
            "goal": "",
            "stage_trajectory": [],
            "stable_facts": [],
            "current_state": {"status": "ongoing", "summary": summary},
        }

    @staticmethod
    def _normalize_status(value: Any) -> str:
        """把旧版状态值映射为文档约定的状态枚举。"""
        status = str(value or "").strip()
        if status in {"active", "in_progress", "ongoing"}:
            return "open"
        return status if status in {"open", "completed", "deleted"} else "open"

    def get_runtime_state(self, state_key: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM runtime_state WHERE state_key = ?",
            (state_key,),
        ).fetchone()
        return self._row_to_dict(row)

    def upsert_runtime_state(
        self,
        state_key: str,
        current_experience_id: str,
        current_segment_id: str,
        updated_at: str,
        retrieval_cache: dict[str, Any] | None = None,
    ) -> None:
        updated_at = format_timestamp(updated_at)
        retrieval_cache_json = json.dumps(retrieval_cache or {}, ensure_ascii=False)
        if retrieval_cache is None:
            self.connection.execute(
                """
                INSERT INTO runtime_state (
                    state_key, current_experience_id, current_segment_id, updated_at,
                    retrieval_cache_json
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(state_key) DO UPDATE SET
                    current_experience_id = excluded.current_experience_id,
                    current_segment_id = excluded.current_segment_id,
                    updated_at = excluded.updated_at
                """,
                (
                    state_key,
                    current_experience_id,
                    current_segment_id,
                    updated_at,
                    retrieval_cache_json,
                ),
            )
            return

        self.connection.execute(
            """
            INSERT INTO runtime_state (
                state_key, current_experience_id, current_segment_id, updated_at,
                retrieval_cache_json
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(state_key) DO UPDATE SET
                current_experience_id = excluded.current_experience_id,
                current_segment_id = excluded.current_segment_id,
                updated_at = excluded.updated_at,
                retrieval_cache_json = excluded.retrieval_cache_json
            """,
            (
                state_key,
                current_experience_id,
                current_segment_id,
                updated_at,
                retrieval_cache_json,
            ),
        )

    def update_runtime_retrieval_cache(
        self,
        state_key: str,
        retrieval_cache: dict[str, Any],
        updated_at: str,
    ) -> None:
        updated_at = format_timestamp(updated_at)
        self.connection.execute(
            """
            INSERT INTO runtime_state (
                state_key, current_experience_id, current_segment_id, updated_at,
                retrieval_cache_json
            ) VALUES (?, '', '', ?, ?)
            ON CONFLICT(state_key) DO UPDATE SET
                retrieval_cache_json = excluded.retrieval_cache_json
            """,
            (
                state_key,
                updated_at,
                json.dumps(retrieval_cache, ensure_ascii=False),
            ),
        )

    def enqueue_memory_job(
        self,
        *,
        job_id: str,
        job_type: str,
        memory_type: str,
        memory_id: str,
        target_version: int,
        timestamp: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Persist or coalesce one asynchronous derived-memory job."""
        now = format_timestamp(timestamp)
        incoming_payload = dict(payload or {})
        existing = self.connection.execute(
            """
            SELECT payload_json FROM memory_outbox
            WHERE job_type = ? AND memory_type = ? AND memory_id = ?
            """,
            (job_type, memory_type, memory_id),
        ).fetchone()
        if existing:
            try:
                previous_payload = json.loads(existing["payload_json"] or "{}")
            except (json.JSONDecodeError, TypeError):
                previous_payload = {}
            if not isinstance(previous_payload, dict):
                previous_payload = {}
            incoming_payload = self._merge_memory_job_payload(
                previous_payload,
                incoming_payload,
            )
        self.connection.execute(
            """
            INSERT INTO memory_outbox (
                job_id, job_type, memory_type, memory_id, target_version,
                status, retry_count, last_error, payload_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, 'pending', 0, '', ?, ?, ?)
            ON CONFLICT(job_type, memory_type, memory_id) DO UPDATE SET
                target_version = MAX(memory_outbox.target_version, excluded.target_version),
                status = 'pending',
                retry_count = 0,
                last_error = '',
                payload_json = excluded.payload_json,
                updated_at = excluded.updated_at
            """,
            (
                job_id,
                job_type,
                memory_type,
                memory_id,
                max(0, int(target_version)),
                json.dumps(incoming_payload, ensure_ascii=False),
                now,
                now,
            ),
        )

    @staticmethod
    def _merge_memory_job_payload(
        previous: dict[str, Any],
        incoming: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge coalesced update intent without losing force/status signals."""
        merged = {**previous, **incoming}
        for key in (
            "force_summary",
            "force_experience_summary",
            "allow_experience_completion",
            "allow_summary_completion",
        ):
            if key in previous or key in incoming:
                merged[key] = bool(previous.get(key)) or bool(incoming.get(key))
        if "desired_status" in previous or "desired_status" in incoming:
            status_rank = {"open": 0, "completed": 1, "deleted": 2}
            statuses = [
                str(value or "open")
                for value in (
                    previous.get("desired_status"),
                    incoming.get("desired_status"),
                )
            ]
            merged["desired_status"] = max(
                statuses,
                key=lambda value: status_rank.get(value, 0),
            )
        return merged

    def recover_processing_memory_jobs(self) -> int:
        """Return interrupted jobs to the pending queue after a restart."""
        cursor = self.connection.execute(
            """
            UPDATE memory_outbox
            SET status = 'pending', updated_at = created_at
            WHERE status = 'processing'
            """
        )
        return int(cursor.rowcount)

    def list_pending_memory_jobs(self, limit: int = 10) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT * FROM memory_outbox
            WHERE status = 'pending'
            ORDER BY created_at, job_id
            LIMIT ?
            """,
            (max(1, int(limit)),),
        ).fetchall()
        jobs = []
        for row in rows:
            job = dict(row)
            try:
                job["payload"] = json.loads(job.pop("payload_json") or "{}")
            except (json.JSONDecodeError, TypeError):
                job["payload"] = {}
            jobs.append(job)
        return jobs

    def get_memory_job(self, job_id: str) -> dict[str, Any] | None:
        """Reload the latest coalesced job state after it has been claimed."""
        row = self.connection.execute(
            "SELECT * FROM memory_outbox WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        if not row:
            return None
        job = dict(row)
        try:
            job["payload"] = json.loads(job.pop("payload_json") or "{}")
        except (json.JSONDecodeError, TypeError):
            job["payload"] = {}
        return job

    def has_pending_terminal_update(
        self,
        *,
        job_type: str,
        memory_type: str,
        memory_id: str,
    ) -> bool:
        """Return whether an outstanding coalesced job intends to close memory."""
        row = self.connection.execute(
            """
            SELECT payload_json FROM memory_outbox
            WHERE job_type = ? AND memory_type = ? AND memory_id = ?
              AND status IN ('pending', 'processing')
            """,
            (job_type, memory_type, memory_id),
        ).fetchone()
        if not row:
            return False
        try:
            payload = json.loads(row["payload_json"] or "{}")
        except (json.JSONDecodeError, TypeError):
            return False
        return str(payload.get("desired_status") or "open") in {
            "completed",
            "deleted",
        }

    def mark_memory_job_processing(self, job_id: str, timestamp: str) -> bool:
        cursor = self.connection.execute(
            """
            UPDATE memory_outbox
            SET status = 'processing', updated_at = ?
            WHERE job_id = ? AND status = 'pending'
            """,
            (format_timestamp(timestamp), job_id),
        )
        return bool(cursor.rowcount)

    def mark_memory_job_completed(
        self,
        job_id: str,
        target_version: int,
        timestamp: str,
    ) -> bool:
        cursor = self.connection.execute(
            """
            UPDATE memory_outbox
            SET status = 'completed', last_error = '', updated_at = ?
            WHERE job_id = ?
              AND status = 'processing'
              AND target_version <= ?
            """,
            (format_timestamp(timestamp), job_id, int(target_version)),
        )
        return bool(cursor.rowcount)

    def mark_memory_job_failed(
        self,
        job_id: str,
        error: str,
        timestamp: str,
        *,
        max_retries: int,
    ) -> None:
        row = self.connection.execute(
            "SELECT retry_count FROM memory_outbox WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        retry_count = int(row["retry_count"] if row else 0) + 1
        status = "failed" if retry_count >= max(1, int(max_retries)) else "pending"
        self.connection.execute(
            """
            UPDATE memory_outbox
            SET status = ?, retry_count = ?, last_error = ?, updated_at = ?
            WHERE job_id = ? AND status = 'processing'
            """,
            (
                status,
                retry_count,
                str(error or "")[:4000],
                format_timestamp(timestamp),
                job_id,
            ),
        )

    def insert_qa(self, qa: dict[str, Any]) -> None:
        columns = [
            "qa_id", "source_id", "timestamp", "user_input", "assistant_output", "tools_json",
            "topic", "intent", "core_entity", "entities_json", "segment_id",
            "status", "confidence", "reason",
        ]
        source_id = qa.get("source_id")
        if source_id is not None:
            source_id = str(source_id).strip() or None
        reason = qa.get("reason", qa.get("reasoning", ""))
        values: list[Any] = [
            qa["qa_id"],
            source_id,
            format_timestamp(qa["timestamp"]),
            qa["user_input"],
            qa["assistant_output"],
            json.dumps(qa["tools"], ensure_ascii=False),
            qa["topic"],
            qa["intent"],
            qa["core_entity"],
            json.dumps(qa["entities"], ensure_ascii=False),
            qa["segment_id"],
            self._normalize_status(qa.get("status")),
            qa["confidence"],
            reason,
        ]
        # 旧库的 reasoning 字段仍为必填，迁移期同步写入以保持可用。
        if "reasoning" in self._table_columns("qa_memory"):
            columns.append("reasoning")
            values.append(reason)
        placeholders = ", ".join("?" for _ in columns)
        self.connection.execute(
            f"INSERT INTO qa_memory ({', '.join(columns)}) VALUES ({placeholders})",
            values,
        )

    def get_qa(self, qa_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM qa_memory WHERE qa_id = ?",
            (qa_id,),
        ).fetchone()
        return self._row_to_dict(row)

    def get_qas(self, qa_ids: list[str]) -> list[dict[str, Any]]:
        if not qa_ids:
            return []
        placeholders = ", ".join("?" for _ in qa_ids)
        rows = self.connection.execute(
            f"SELECT * FROM qa_memory WHERE qa_id IN ({placeholders})",
            qa_ids,
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def count_qas_by_segment(self, segment_id: str) -> int:
        """Count live QA rows without relying on the async Segment projection."""
        row = self.connection.execute(
            "SELECT COUNT(*) AS count FROM qa_memory "
            "WHERE segment_id = ? AND status = 'open'",
            (segment_id,),
        ).fetchone()
        return int(row["count"] if row else 0)

    def list_qas_by_segment(self, segment_id: str) -> list[dict[str, Any]]:
        """Load live QA evidence directly from the normalized relation."""
        rows = self.connection.execute(
            """
            SELECT * FROM qa_memory
            WHERE segment_id = ? AND status = 'open'
            ORDER BY timestamp, qa_id
            """,
            (segment_id,),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_segment(self, segment_id: str | None) -> dict[str, Any] | None:
        if not segment_id:
            return None
        row = self.connection.execute(
            "SELECT * FROM segment_memory WHERE segment_id = ?",
            (segment_id,),
        ).fetchone()
        return self._row_to_dict(row)

    def get_segments(self, segment_ids: list[str]) -> list[dict[str, Any]]:
        if not segment_ids:
            return []
        placeholders = ", ".join("?" for _ in segment_ids)
        rows = self.connection.execute(
            f"SELECT * FROM segment_memory WHERE segment_id IN ({placeholders})",
            segment_ids,
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def insert_segment(self, segment: dict[str, Any]) -> None:
        summary = self._segment_summary_payload(segment.get("summary"))
        columns = [
            "segment_id", "topic", "intent", "core_entity", "qa_ids_json",
            "summary_json", "experience_id", "created_at", "updated_at",
            "version", "last_summarized_qa_count", "summarized_qa_ids_json",
            "summary_version", "status",
        ]
        values: list[Any] = [
            segment["segment_id"],
            segment["topic"],
            segment["intent"],
            segment["core_entity"],
            json.dumps(segment["qa_ids"], ensure_ascii=False),
            json.dumps(summary, ensure_ascii=False),
            segment["experience_id"],
            format_timestamp(segment["created_at"]),
            format_timestamp(segment["updated_at"]),
            segment["version"],
            segment["last_summarized_qa_count"],
            json.dumps(segment.get("summarized_qa_ids") or [], ensure_ascii=False),
            int(segment.get("summary_version") or 0),
            self._normalize_status(segment.get("status")),
        ]
        # 旧库的 summary 字段仍为必填，保留一份 JSON 文本兼容旧读端。
        if "summary" in self._table_columns("segment_memory"):
            columns.append("summary")
            values.append(json.dumps(summary, ensure_ascii=False))
        placeholders = ", ".join("?" for _ in columns)
        self.connection.execute(
            f"INSERT INTO segment_memory ({', '.join(columns)}) "
            f"VALUES ({placeholders})",
            values,
        )

    def update_segment(self, segment: dict[str, Any]) -> None:
        summary = self._segment_summary_payload(segment.get("summary"))
        legacy_assignment = (
            ", summary = ?"
            if "summary" in self._table_columns("segment_memory")
            else ""
        )
        parameters: list[Any] = [
            json.dumps(segment["qa_ids"], ensure_ascii=False),
            json.dumps(summary, ensure_ascii=False),
            format_timestamp(segment["updated_at"]),
            segment["version"],
            segment["last_summarized_qa_count"],
            json.dumps(segment.get("summarized_qa_ids") or [], ensure_ascii=False),
            int(segment.get("summary_version") or 0),
            self._normalize_status(segment.get("status")),
        ]
        if legacy_assignment:
            parameters.append(json.dumps(summary, ensure_ascii=False))
        parameters.append(segment["segment_id"])
        self.connection.execute(
            f"""
            UPDATE segment_memory SET
                qa_ids_json = ?,
                summary_json = ?,
                updated_at = ?,
                version = ?,
                last_summarized_qa_count = ?,
                summarized_qa_ids_json = ?,
                summary_version = ?,
                status = ?
                {legacy_assignment}
            WHERE segment_id = ?
            """,
            parameters,
        )

    def update_segment_activity(self, segment: dict[str, Any]) -> None:
        """Update request-owned Segment fields while preserving derived summary fields."""
        self.connection.execute(
            """
            UPDATE segment_memory SET
                qa_ids_json = ?,
                updated_at = ?,
                status = ?
            WHERE segment_id = ?
            """,
            (
                json.dumps(segment.get("qa_ids") or [], ensure_ascii=False),
                format_timestamp(segment["updated_at"]),
                self._normalize_status(segment.get("status")),
                segment["segment_id"],
            ),
        )

    def reconcile_segment(
        self,
        *,
        segment_id: str,
        desired_status: str = "open",
        updated_at: str,
        summary: dict[str, Any] | None = None,
        summarized_qa_count: int | None = None,
    ) -> None:
        """Apply one aggregate Segment update from normalized QA evidence."""
        qa_rows = self.connection.execute(
            """
            SELECT qa_id, timestamp FROM qa_memory
            WHERE segment_id = ? AND status = 'open'
            ORDER BY timestamp, qa_id
            """,
            (segment_id,),
        ).fetchall()
        qa_ids = [str(row["qa_id"]) for row in qa_rows]
        latest_qa_at = str(qa_rows[-1]["timestamp"]) if qa_rows else ""
        effective_updated_at = max(
            format_timestamp(updated_at),
            format_timestamp(latest_qa_at) if latest_qa_at else "",
        )
        normalized_status = self._normalize_status(desired_status)
        summary_assignment = ""
        legacy_assignment = ""
        parameters: list[Any] = [json.dumps(qa_ids, ensure_ascii=False)]
        if summary is not None:
            summary_payload = self._segment_summary_payload(summary)
            summary_assignment = (
                ", summary_json = ?, "
                "last_summarized_qa_count = MAX(last_summarized_qa_count, ?), "
                "summarized_qa_ids_json = ?, "
                "summary_version = summary_version + 1"
            )
            parameters.extend(
                [
                    json.dumps(summary_payload, ensure_ascii=False),
                    max(0, int(summarized_qa_count or 0)),
                    json.dumps(qa_ids, ensure_ascii=False),
                ]
            )
            if "summary" in self._table_columns("segment_memory"):
                legacy_assignment = ", summary = ?"
                parameters.append(json.dumps(summary_payload, ensure_ascii=False))
        parameters.extend(
            [
                effective_updated_at,
                normalized_status,
                normalized_status,
                segment_id,
            ]
        )
        self.connection.execute(
            f"""
            UPDATE segment_memory SET
                qa_ids_json = ?
                {summary_assignment}
                {legacy_assignment},
                updated_at = MAX(updated_at, ?),
                status = CASE
                    WHEN status = 'deleted' THEN 'deleted'
                    WHEN ? = 'deleted' THEN 'deleted'
                    WHEN status = 'completed' OR ? = 'completed' THEN 'completed'
                    ELSE 'open'
                END,
                version = version + 1
            WHERE segment_id = ?
            """,
            parameters,
        )

    def update_segment_summary(
        self,
        *,
        segment_id: str,
        summary: dict[str, Any],
        last_summarized_qa_count: int,
        status: str,
        updated_at: str,
    ) -> None:
        """Update derived Segment fields without overwriting concurrently added QA ids."""
        summary_payload = self._segment_summary_payload(summary)
        summarized_ids = [
            str(row["qa_id"])
            for row in self.connection.execute(
                "SELECT qa_id FROM qa_memory "
                "WHERE segment_id = ? AND status = 'open' "
                "ORDER BY timestamp, qa_id",
                (segment_id,),
            ).fetchall()
        ]
        legacy_assignment = (
            ", summary = ?"
            if "summary" in self._table_columns("segment_memory")
            else ""
        )
        parameters: list[Any] = [
            json.dumps(summary_payload, ensure_ascii=False),
            max(0, int(last_summarized_qa_count)),
            json.dumps(summarized_ids, ensure_ascii=False),
            self._normalize_status(status),
            format_timestamp(updated_at),
        ]
        if legacy_assignment:
            parameters.append(json.dumps(summary_payload, ensure_ascii=False))
        parameters.append(segment_id)
        self.connection.execute(
            f"""
            UPDATE segment_memory SET
                summary_json = ?,
                last_summarized_qa_count = MAX(last_summarized_qa_count, ?),
                summarized_qa_ids_json = ?,
                summary_version = summary_version + 1,
                status = CASE
                    WHEN status = 'deleted' THEN 'deleted'
                    WHEN ? = 'completed' THEN 'completed'
                    ELSE status
                END,
                updated_at = ?,
                version = version + 1
                {legacy_assignment}
            WHERE segment_id = ?
            """,
            parameters,
        )

    def find_latest_segment(self, experience_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            """
            SELECT * FROM segment_memory
            WHERE experience_id = ?
            ORDER BY updated_at DESC, created_at DESC
            LIMIT 1
            """,
            (experience_id,),
        ).fetchone()
        return self._row_to_dict(row)

    def list_recent_open_segments(
        self,
        experience_id: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Return recent reusable Segment candidates for write routing."""
        rows = self.connection.execute(
            """
            SELECT * FROM segment_memory
            WHERE experience_id = ? AND status = 'open'
            ORDER BY updated_at DESC, created_at DESC, segment_id DESC
            LIMIT ?
            """,
            (experience_id, max(1, int(limit))),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_experience(self, experience_id: str | None) -> dict[str, Any] | None:
        if not experience_id:
            return None
        row = self.connection.execute(
            "SELECT * FROM experience_memory WHERE experience_id = ?",
            (experience_id,),
        ).fetchone()
        return self._row_to_dict(row)

    def find_experience(self, topic: str, core_entity: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            """
            SELECT * FROM experience_memory
            WHERE topic = ? AND core_entity = ?
            ORDER BY updated_at DESC, created_at DESC
            LIMIT 1
            """,
            (topic, core_entity),
        ).fetchone()
        return self._row_to_dict(row)

    def find_active_experience(
        self, topic: str, core_entity: str
    ) -> dict[str, Any] | None:
        """Return the latest in-progress Experience with an exact identity match."""
        row = self.connection.execute(
            """
            SELECT * FROM experience_memory
            WHERE lower(replace(trim(topic), ' ', '')) =
                  lower(replace(trim(?), ' ', ''))
              AND lower(replace(trim(core_entity), ' ', '')) =
                  lower(replace(trim(?), ' ', ''))
              AND status = 'open'
            ORDER BY updated_at DESC, created_at DESC
            LIMIT 1
            """,
            (str(topic or "").strip(), str(core_entity or "").strip()),
        ).fetchone()
        return self._row_to_dict(row)

    def find_experience_by_topic(self, topic: str) -> dict[str, Any] | None:
        """topic 精确匹配回退查询（向量检索嵌入失败时使用）。
        不依赖 core_entity，避免同一主题不同说话人造成匹配失败。
        """
        row = self.connection.execute(
            """
            SELECT * FROM experience_memory
            WHERE topic = ?
            ORDER BY updated_at DESC, created_at DESC
            LIMIT 1
            """,
            (topic,),
        ).fetchone()
        return self._row_to_dict(row)

    def find_experiences(
        self, topic: str, core_entity: str, limit: int
    ) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT * FROM experience_memory
            WHERE topic = ? AND core_entity = ?
            ORDER BY updated_at DESC, created_at DESC
            LIMIT ?
            """,
            (topic, core_entity, limit),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def list_experiences_by_topic(
        self, topic: str, limit: int
    ) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT * FROM experience_memory
            WHERE topic = ?
            ORDER BY updated_at DESC, created_at DESC
            LIMIT ?
            """,
            (topic, limit),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def search_experiences(
        self,
        topic: str,
        core_entity: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Recall Experience rows by topic/core_entity only."""
        topic = str(topic or "").strip()
        core_entity = str(core_entity or "").strip()
        conditions: list[str] = []
        parameters: list[Any] = []
        score_parts: list[str] = []
        score_parameters: list[Any] = []
        if topic:
            conditions.append("topic = ?")
            parameters.append(topic)
            score_parts.append("CASE WHEN topic = ? THEN 1 ELSE 0 END")
            score_parameters.append(topic)
        if core_entity:
            conditions.append("core_entity = ?")
            parameters.append(core_entity)
            score_parts.append("CASE WHEN core_entity = ? THEN 1 ELSE 0 END")
            score_parameters.append(core_entity)
        if not conditions:
            return []

        rows = self.connection.execute(
            f"""
            SELECT *, ({' + '.join(score_parts)}) AS relation_score
            FROM experience_memory
            WHERE {' OR '.join(conditions)}
            ORDER BY relation_score DESC, updated_at DESC, created_at DESC
            LIMIT ?
            """,
            (*score_parameters, *parameters, max(1, int(limit))),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def search_completed_experiences(
        self,
        topic: str,
        core_entity: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Recall completed Experiences matching topic, entity, or both."""
        topic = str(topic or "").strip()
        core_entity = str(core_entity or "").strip()
        conditions: list[str] = []
        parameters: list[Any] = []
        score_parts: list[str] = []
        score_parameters: list[Any] = []
        if topic:
            conditions.append("topic = ?")
            parameters.append(topic)
            score_parts.append("CASE WHEN topic = ? THEN 1 ELSE 0 END")
            score_parameters.append(topic)
        if core_entity:
            conditions.append("core_entity = ?")
            parameters.append(core_entity)
            score_parts.append(
                "CASE WHEN core_entity = ? THEN 1 ELSE 0 END"
            )
            score_parameters.append(core_entity)
        if not conditions:
            return []

        rows = self.connection.execute(
            f"""
            SELECT *, ({' + '.join(score_parts)}) AS relation_score
            FROM experience_memory
            WHERE status = 'completed'
              AND ({' OR '.join(conditions)})
            ORDER BY relation_score DESC, updated_at DESC, created_at DESC
            LIMIT ?
            """,
            (*score_parameters, *parameters, max(1, int(limit))),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def search_qas(
        self,
        *,
        topic: str,
        core_entity: str,
        entities: list[str],
        limit: int,
        intent: str = "",
        keywords: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Recall QA rows by keywords already stored in structured SQLite fields."""
        conditions: list[str] = []
        parameters: list[Any] = []
        score_parts: list[str] = []
        score_parameters: list[Any] = []
        topic = str(topic or "").strip()
        core_entity = str(core_entity or "").strip()
        intent = str(intent or "").strip()
        normalized_entities = sorted({
            str(value).strip().casefold()
            for value in entities
            if str(value).strip()
        })

        if topic:
            conditions.append("q.topic = ?")
            parameters.append(topic)
            score_parts.append("CASE WHEN q.topic = ? THEN 4 ELSE 0 END")
            score_parameters.append(topic)
        if core_entity:
            conditions.append("q.core_entity = ?")
            parameters.append(core_entity)
            score_parts.append("CASE WHEN q.core_entity = ? THEN 6 ELSE 0 END")
            score_parameters.append(core_entity)
        if intent:
            conditions.append("q.intent = ?")
            parameters.append(intent)
            score_parts.append("CASE WHEN q.intent = ? THEN 2 ELSE 0 END")
            score_parameters.append(intent)
        if normalized_entities:
            placeholders = ", ".join("?" for _ in normalized_entities)
            # JSON1 membership avoids substring matches in serialized entity data.
            entity_match = (
                "EXISTS ("
                "SELECT 1 FROM json_each("
                "CASE WHEN json_valid(q.entities_json) THEN q.entities_json ELSE '[]' END"
                ") AS entity WHERE lower(trim(CAST(entity.value AS TEXT))) "
                f"IN ({placeholders})"
                ")"
            )
            conditions.append(entity_match)
            parameters.extend(normalized_entities)
            score_parts.append(f"CASE WHEN {entity_match} THEN 3 ELSE 0 END")
            score_parameters.extend(normalized_entities)
        if not conditions:
            return []

        rows = self.connection.execute(
            f"""
            SELECT q.*, ({' + '.join(score_parts)}) AS keyword_score
            FROM qa_memory AS q
            WHERE q.status = 'open' AND ({' OR '.join(conditions)})
            ORDER BY keyword_score DESC, q.timestamp DESC
            LIMIT ?
            """,
            (*score_parameters, *parameters, max(1, int(limit))),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_experiences(self, experience_ids: list[str]) -> list[dict[str, Any]]:
        if not experience_ids:
            return []
        placeholders = ", ".join("?" for _ in experience_ids)
        rows = self.connection.execute(
            f"SELECT * FROM experience_memory WHERE experience_id IN ({placeholders})",
            experience_ids,
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def count_segments_by_experience(self, experience_id: str) -> int:
        """Count live Segment rows without relying on async Experience fields."""
        row = self.connection.execute(
            "SELECT COUNT(*) AS count FROM segment_memory "
            "WHERE experience_id = ? AND status != 'deleted'",
            (experience_id,),
        ).fetchone()
        return int(row["count"] if row else 0)

    def insert_experience(self, experience: dict[str, Any]) -> None:
        summary = self._experience_summary_payload(experience.get("summary"))
        status = self._normalize_status(
            experience.get("status")
            or (experience.get("state") or {}).get("status")
        )
        history = json.dumps(
            experience.get("history_experience") or {},
            ensure_ascii=False,
        )
        columns = [
            "experience_id", "history_experience_json", "topic", "core_entity",
            "intents_link_json", "segment_ids_json", "summary_json", "created_at",
            "updated_at", "version", "last_summarized_segment_count",
            "last_summarized_child_revision", "status",
        ]
        values: list[Any] = [
            experience["experience_id"],
            history,
            experience["topic"],
            experience["core_entity"],
            json.dumps(experience["intents_link"], ensure_ascii=False),
            json.dumps(experience["segment_ids"], ensure_ascii=False),
            json.dumps(summary, ensure_ascii=False),
            format_timestamp(experience["created_at"]),
            format_timestamp(experience["updated_at"]),
            experience["version"],
            experience["last_summarized_segment_count"],
            int(experience.get("last_summarized_child_revision") or 0),
            status,
        ]
        # 旧库的 state_json 字段仍为必填，同步生成兼容状态对象。
        if "state_json" in self._table_columns("experience_memory"):
            columns.append("state_json")
            values.append(json.dumps({"status": status}, ensure_ascii=False))
        placeholders = ", ".join("?" for _ in columns)
        self.connection.execute(
            f"INSERT INTO experience_memory ({', '.join(columns)}) "
            f"VALUES ({placeholders})",
            values,
        )

    def update_experience(self, experience: dict[str, Any]) -> None:
        summary = self._experience_summary_payload(experience.get("summary"))
        status = self._normalize_status(
            experience.get("status")
            or (experience.get("state") or {}).get("status")
        )
        legacy_assignment = (
            ", state_json = ?"
            if "state_json" in self._table_columns("experience_memory")
            else ""
        )
        parameters: list[Any] = [
            json.dumps(experience["intents_link"], ensure_ascii=False),
            json.dumps(experience["segment_ids"], ensure_ascii=False),
            json.dumps(summary, ensure_ascii=False),
            format_timestamp(experience["updated_at"]),
            experience["version"],
            experience["last_summarized_segment_count"],
            int(experience.get("last_summarized_child_revision") or 0),
            json.dumps(
                experience.get("history_experience") or {},
                ensure_ascii=False,
            ),
            status,
        ]
        if legacy_assignment:
            parameters.append(json.dumps({"status": status}, ensure_ascii=False))
        parameters.append(experience["experience_id"])
        self.connection.execute(
            f"""
            UPDATE experience_memory SET
                intents_link_json = ?,
                segment_ids_json = ?,
                summary_json = ?,
                updated_at = ?,
                version = ?,
                last_summarized_segment_count = ?,
                last_summarized_child_revision = ?,
                history_experience_json = ?,
                status = ?
                {legacy_assignment}
            WHERE experience_id = ?
            """,
            parameters,
        )

    def update_experience_activity(self, experience: dict[str, Any]) -> None:
        """Update hierarchy links while preserving async summary/history fields."""
        self.connection.execute(
            """
            UPDATE experience_memory SET
                intents_link_json = ?,
                segment_ids_json = ?,
                updated_at = ?,
                status = ?
            WHERE experience_id = ?
            """,
            (
                json.dumps(experience.get("intents_link") or [], ensure_ascii=False),
                json.dumps(experience.get("segment_ids") or [], ensure_ascii=False),
                format_timestamp(experience["updated_at"]),
                self._normalize_status(experience.get("status")),
                experience["experience_id"],
            ),
        )

    def reconcile_experience(
        self,
        *,
        experience_id: str,
        desired_status: str = "open",
        updated_at: str,
        summary: dict[str, Any] | None = None,
        summarized_segment_count: int | None = None,
        summarized_child_revision: int | None = None,
    ) -> None:
        """Apply one aggregate Experience update from normalized Segment data."""
        segment_rows = self.connection.execute(
            """
            SELECT segment_id, intent, updated_at FROM segment_memory
            WHERE experience_id = ? AND status != 'deleted'
            ORDER BY created_at, segment_id
            """,
            (experience_id,),
        ).fetchall()
        segment_ids = [str(row["segment_id"]) for row in segment_rows]
        intents = list(
            dict.fromkeys(
                str(row["intent"]).strip()
                for row in segment_rows
                if str(row["intent"] or "").strip()
            )
        )
        latest_segment_at = max(
            (str(row["updated_at"] or "") for row in segment_rows),
            default="",
        )
        effective_updated_at = max(
            format_timestamp(updated_at),
            format_timestamp(latest_segment_at) if latest_segment_at else "",
        )
        normalized_status = self._normalize_status(desired_status)
        summary_assignment = ""
        parameters: list[Any] = [
            json.dumps(intents, ensure_ascii=False),
            json.dumps(segment_ids, ensure_ascii=False),
        ]
        if summary is not None:
            summary_assignment = (
                ", summary_json = ?, "
                "last_summarized_segment_count = MAX("
                "last_summarized_segment_count, ?), "
                "last_summarized_child_revision = MAX("
                "last_summarized_child_revision, ?)"
            )
            parameters.extend(
                [
                    json.dumps(
                        self._experience_summary_payload(summary),
                        ensure_ascii=False,
                    ),
                    max(0, int(summarized_segment_count or 0)),
                    max(0, int(summarized_child_revision or 0)),
                ]
            )
        parameters.extend(
            [
                effective_updated_at,
                normalized_status,
                normalized_status,
                experience_id,
            ]
        )
        self.connection.execute(
            f"""
            UPDATE experience_memory SET
                intents_link_json = ?,
                segment_ids_json = ?
                {summary_assignment},
                updated_at = MAX(updated_at, ?),
                status = CASE
                    WHEN status = 'deleted' THEN 'deleted'
                    WHEN ? = 'deleted' THEN 'deleted'
                    WHEN status = 'completed' OR ? = 'completed' THEN 'completed'
                    ELSE 'open'
                END,
                version = version + 1
            WHERE experience_id = ?
            """,
            parameters,
        )
        if "state_json" in self._table_columns("experience_memory"):
            row = self.connection.execute(
                "SELECT status FROM experience_memory WHERE experience_id = ?",
                (experience_id,),
            ).fetchone()
            if row:
                self.connection.execute(
                    "UPDATE experience_memory SET state_json = ? "
                    "WHERE experience_id = ?",
                    (
                        json.dumps({"status": row["status"]}, ensure_ascii=False),
                        experience_id,
                    ),
                )

    def update_experience_summary(
        self,
        *,
        experience_id: str,
        summary: dict[str, Any],
        last_summarized_segment_count: int,
        status: str,
        updated_at: str,
    ) -> None:
        """Update derived Experience fields without overwriting live hierarchy links."""
        summary_payload = self._experience_summary_payload(summary)
        self.connection.execute(
            """
            UPDATE experience_memory SET
                summary_json = ?,
                last_summarized_segment_count = MAX(
                    last_summarized_segment_count, ?
                ),
                status = CASE
                    WHEN status = 'deleted' THEN 'deleted'
                    WHEN ? = 'completed' THEN 'completed'
                    ELSE status
                END,
                updated_at = ?,
                version = version + 1
            WHERE experience_id = ?
            """,
            (
                json.dumps(summary_payload, ensure_ascii=False),
                max(0, int(last_summarized_segment_count)),
                self._normalize_status(status),
                format_timestamp(updated_at),
                experience_id,
            ),
        )

    def update_experience_history(
        self,
        *,
        experience_id: str,
        history_experience: dict[str, Any] | str,
        updated_at: str,
    ) -> None:
        """Update recalled history without overwriting concurrent Experience changes."""
        self.connection.execute(
            """
            UPDATE experience_memory SET
                history_experience_json = ?,
                updated_at = ?,
                version = version + 1
            WHERE experience_id = ? AND status != 'deleted'
            """,
            (
                json.dumps(history_experience or {}, ensure_ascii=False),
                format_timestamp(updated_at),
                experience_id,
            ),
        )

    def list_segments_by_experience_ids(
        self, experience_ids: list[str]
    ) -> list[dict[str, Any]]:
        if not experience_ids:
            return []
        placeholders = ", ".join("?" for _ in experience_ids)
        rows = self.connection.execute(
            f"""
            SELECT * FROM segment_memory
            WHERE experience_id IN ({placeholders}) AND status != 'deleted'
            ORDER BY updated_at DESC, created_at DESC
            """,
            experience_ids,
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def list_qas_by_segment_ids(
        self, segment_ids: list[str]
    ) -> list[dict[str, Any]]:
        if not segment_ids:
            return []
        placeholders = ", ".join("?" for _ in segment_ids)
        rows = self.connection.execute(
            f"""
            SELECT * FROM qa_memory
            WHERE segment_id IN ({placeholders}) AND status = 'open'
            """,
            segment_ids,
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def list_latest_segments(
        self, experience_id: str, limit: int = 2
    ) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT * FROM segment_memory
            WHERE experience_id = ? AND status != 'deleted'
            ORDER BY updated_at DESC, created_at DESC
            LIMIT ?
            """,
            (experience_id, max(1, int(limit))),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def list_latest_qas(
        self, segment_ids: list[str], limit: int = 5
    ) -> list[dict[str, Any]]:
        if not segment_ids:
            return []
        placeholders = ", ".join("?" for _ in segment_ids)
        rows = self.connection.execute(
            f"""
            SELECT * FROM qa_memory
            WHERE segment_id IN ({placeholders}) AND status = 'open'
            ORDER BY timestamp DESC, qa_id DESC
            LIMIT ?
            """,
            (*segment_ids, max(1, int(limit))),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def count_rows(self, table: str) -> int:
        if table not in {
            "qa_memory",
            "segment_memory",
            "experience_memory",
            "runtime_state",
            "chat_session",
            "memory_outbox",
        }:
            raise ValueError(f"Unsupported table: {table}")
        return int(self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
