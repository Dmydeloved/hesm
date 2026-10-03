"""Durable Chat/Retriever/Add and memory distillation trajectories."""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from .time_utils import format_timestamp


TRAJECTORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat (
    chat_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    retriever_id TEXT NOT NULL UNIQUE,
    add_id TEXT NOT NULL UNIQUE,
    user_input TEXT NOT NULL,
    assistant_output TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed', 'interrupted')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    elapsed_ms REAL NOT NULL DEFAULT 0,
    model TEXT NOT NULL DEFAULT '',
    error_message TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    prompt_text TEXT NOT NULL DEFAULT '',
    timing_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (session_id) REFERENCES chat_session(session_id) DEFERRABLE INITIALLY DEFERRED,
    FOREIGN KEY (retriever_id) REFERENCES retriever(retriever_id) DEFERRABLE INITIALLY DEFERRED,
    FOREIGN KEY (add_id) REFERENCES "add"(add_id) DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE IF NOT EXISTS retriever (
    retriever_id TEXT PRIMARY KEY,
    chat_id TEXT,
    question TEXT NOT NULL DEFAULT '',
    topic_extract_json TEXT NOT NULL DEFAULT '{}',
    query_candidates_json TEXT NOT NULL DEFAULT '[]',
    experiences_json TEXT NOT NULL DEFAULT '[]',
    segments_json TEXT NOT NULL DEFAULT '[]',
    qas_json TEXT NOT NULL DEFAULT '[]',
    context_text TEXT NOT NULL DEFAULT '',
    topic_extract_time_ms REAL NOT NULL DEFAULT 0,
    retrieval_time_ms REAL NOT NULL DEFAULT 0,
    elapsed_ms REAL NOT NULL DEFAULT 0,
    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    error_message TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (chat_id) REFERENCES chat(chat_id) DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE IF NOT EXISTS "add" (
    add_id TEXT PRIMARY KEY,
    chat_id TEXT,
    topic_extract_json TEXT NOT NULL DEFAULT '{}',
    experience_id TEXT NOT NULL DEFAULT '',
    segment_id TEXT NOT NULL DEFAULT '',
    experience_ids_json TEXT NOT NULL DEFAULT '[]',
    segment_ids_json TEXT NOT NULL DEFAULT '[]',
    qa_ids_json TEXT NOT NULL DEFAULT '[]',
    memory_outbox_id TEXT,
    memory_outbox_ids_json TEXT NOT NULL DEFAULT '[]',
    user_input TEXT NOT NULL DEFAULT '',
    assistant_output TEXT NOT NULL DEFAULT '',
    action TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL CHECK (status IN ('pending', 'completed', 'failed')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    elapsed_ms REAL NOT NULL DEFAULT 0,
    error_message TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (chat_id) REFERENCES chat(chat_id) DEFERRABLE INITIALLY DEFERRED,
    FOREIGN KEY (memory_outbox_id) REFERENCES memory_outbox(job_id) DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE IF NOT EXISTS segment_update (
    segment_update_id TEXT PRIMARY KEY,
    add_id TEXT,
    memory_outbox_id TEXT,
    segment_id TEXT NOT NULL,
    old_segment_json TEXT NOT NULL DEFAULT '{}',
    distill_input_json TEXT NOT NULL DEFAULT '{}',
    prompt_text TEXT NOT NULL DEFAULT '',
    output_text TEXT NOT NULL DEFAULT '',
    new_segment_json TEXT NOT NULL DEFAULT '{}',
    before_version INTEGER NOT NULL DEFAULT 0,
    after_version INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'skipped', 'failed')),
    skip_reason TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    started_at TEXT NOT NULL,
    finished_at TEXT,
    elapsed_ms REAL NOT NULL DEFAULT 0,
    error_message TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY (add_id) REFERENCES "add"(add_id),
    FOREIGN KEY (memory_outbox_id) REFERENCES memory_outbox(job_id),
    FOREIGN KEY (segment_id) REFERENCES segment_memory(segment_id)
);

CREATE TABLE IF NOT EXISTS experience_update (
    experience_update_id TEXT PRIMARY KEY,
    add_id TEXT,
    memory_outbox_id TEXT,
    segment_update_id TEXT,
    experience_id TEXT NOT NULL,
    old_experience_json TEXT NOT NULL DEFAULT '{}',
    distill_input_json TEXT NOT NULL DEFAULT '{}',
    prompt_text TEXT NOT NULL DEFAULT '',
    output_text TEXT NOT NULL DEFAULT '',
    new_experience_json TEXT NOT NULL DEFAULT '{}',
    before_version INTEGER NOT NULL DEFAULT 0,
    after_version INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'skipped', 'failed')),
    skip_reason TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    started_at TEXT NOT NULL,
    finished_at TEXT,
    elapsed_ms REAL NOT NULL DEFAULT 0,
    error_message TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY (add_id) REFERENCES "add"(add_id),
    FOREIGN KEY (memory_outbox_id) REFERENCES memory_outbox(job_id),
    FOREIGN KEY (segment_update_id) REFERENCES segment_update(segment_update_id),
    FOREIGN KEY (experience_id) REFERENCES experience_memory(experience_id)
);

CREATE INDEX IF NOT EXISTS idx_chat_session_update_at ON chat_session(update_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_session_chat_ids ON chat(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_retriever_chat ON retriever(chat_id, created_at);
CREATE INDEX IF NOT EXISTS idx_add_chat ON "add"(chat_id, created_at);
CREATE INDEX IF NOT EXISTS idx_add_outbox ON "add"(memory_outbox_id);
CREATE INDEX IF NOT EXISTS idx_segment_update_target ON segment_update(segment_id, created_at);
CREATE INDEX IF NOT EXISTS idx_segment_update_add ON segment_update(add_id, created_at);
CREATE INDEX IF NOT EXISTS idx_segment_update_outbox ON segment_update(memory_outbox_id);
CREATE INDEX IF NOT EXISTS idx_experience_update_target ON experience_update(experience_id, created_at);
CREATE INDEX IF NOT EXISTS idx_experience_update_add ON experience_update(add_id, created_at);
CREATE INDEX IF NOT EXISTS idx_experience_update_outbox ON experience_update(memory_outbox_id);
"""


def _dump(value: Any, default: Any) -> str:
    return json.dumps(default if value is None else value, ensure_ascii=False, default=str)


def _loads(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    try:
        return json.loads(value) if isinstance(value, str) else value
    except (json.JSONDecodeError, TypeError):
        return default


class TraceRepository:
    """Repository for the normalized trajectory model."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        with self._connect() as connection:
            connection.executescript(TRAJECTORY_SCHEMA)
            chat_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(chat)").fetchall()
            }
            if "prompt_text" not in chat_columns:
                connection.execute(
                    "ALTER TABLE chat ADD COLUMN prompt_text TEXT NOT NULL DEFAULT ''"
                )
            if "timing_json" not in chat_columns:
                connection.execute(
                    "ALTER TABLE chat ADD COLUMN timing_json TEXT NOT NULL DEFAULT '{}'"
                )
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 15000")
        return connection

    @staticmethod
    def _id(prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex}"

    def start(self, session_id: str, user_input: str) -> dict[str, str]:
        chat_id = self._id("chat")
        retriever_id = self._id("retriever")
        add_id = self._id("add")
        now = format_timestamp()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT chats FROM chat_session WHERE session_id=?", (session_id,)
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO chat_session(session_id,status,create_at,update_at,chats) VALUES(?,?,?,?,?)",
                    (session_id, "active", now, now, "[]"),
                )
                chats: list[str] = []
            else:
                chats = _loads(row["chats"], [])
                chats = chats if isinstance(chats, list) else []
            chats.append(chat_id)
            connection.execute(
                "INSERT INTO retriever(retriever_id,chat_id,question,status,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                (retriever_id, chat_id, str(user_input), "running", now, now),
            )
            connection.execute(
                'INSERT INTO "add"(add_id,chat_id,status,created_at,updated_at) VALUES(?,?,?,?,?)',
                (add_id, chat_id, "pending", now, now),
            )
            connection.execute(
                "INSERT INTO chat(chat_id,session_id,retriever_id,add_id,user_input,status,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)",
                (chat_id, session_id, retriever_id, add_id, str(user_input), "running", now, now),
            )
            connection.execute(
                "UPDATE chat_session SET chats=?,update_at=? WHERE session_id=?",
                (_dump(chats, []), now, session_id),
            )
        return {"chat_id": chat_id, "retriever_id": retriever_id, "add_id": add_id}

    def start_retriever(self, question: str) -> str:
        retriever_id = self._id("retriever")
        now = format_timestamp()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO retriever(retriever_id,question,status,created_at,updated_at) VALUES(?,?,?,?,?)",
                (retriever_id, str(question), "running", now, now),
            )
        return retriever_id

    def create_add(self, *, chat_id: str | None = None, user_input: str = "", assistant_output: str = "") -> str:
        add_id = self._id("add")
        now = format_timestamp()
        with self._connect() as connection:
            connection.execute(
                'INSERT INTO "add"(add_id,chat_id,user_input,assistant_output,status,created_at,updated_at) VALUES(?,?,?,?,?,?,?)',
                (add_id, chat_id, user_input, assistant_output, "pending", now, now),
            )
        return add_id

    def update_retriever(self, retriever_id: str, **values: Any) -> None:
        allowed = {
            "question", "topic_extract_json", "query_candidates_json", "experiences_json",
            "segments_json", "qas_json", "context_text", "topic_extract_time_ms",
            "retrieval_time_ms", "elapsed_ms", "status", "error_message",
        }
        self._update("retriever", "retriever_id", retriever_id, allowed, values)

    def update_add(self, add_id: str, **values: Any) -> None:
        allowed = {
            "topic_extract_json", "experience_id", "segment_id", "experience_ids_json",
            "segment_ids_json", "qa_ids_json", "memory_outbox_id", "memory_outbox_ids_json",
            "user_input", "assistant_output", "action", "status", "elapsed_ms", "error_message",
        }
        self._update('"add"', "add_id", add_id, allowed, values)

    def update_chat(self, chat_id: str, **values: Any) -> None:
        self._update("chat", "chat_id", chat_id, {"prompt_text", "timing_json"}, values)

    def finish_chat(
        self, chat_id: str, *, status: str, assistant_output: str = "",
        elapsed_ms: float = 0, model: str = "", error_message: str = "",
        timing: dict[str, Any] | None = None,
    ) -> None:
        now = format_timestamp()
        with self._connect() as connection:
            if timing is None:
                connection.execute(
                    "UPDATE chat SET status=?,assistant_output=?,updated_at=?,elapsed_ms=?,model=?,error_message=? WHERE chat_id=?",
                    (status, assistant_output, now, float(elapsed_ms), model, error_message, chat_id),
                )
            else:
                connection.execute(
                    "UPDATE chat SET status=?,assistant_output=?,updated_at=?,elapsed_ms=?,model=?,error_message=?,timing_json=? WHERE chat_id=?",
                    (status, assistant_output, now, float(elapsed_ms), model, error_message,
                     _dump(timing, {}), chat_id),
                )
            connection.execute(
                "UPDATE chat_session SET update_at=? WHERE session_id=(SELECT session_id FROM chat WHERE chat_id=?)",
                (now, chat_id),
            )

    def _update(
        self, table: str, key: str, identifier: str, allowed: set[str], values: dict[str, Any]
    ) -> None:
        changes = {name: value for name, value in values.items() if name in allowed}
        if not changes:
            return
        changes["updated_at"] = format_timestamp()
        assignments = ", ".join(f"{name}=?" for name in changes)
        with self._connect() as connection:
            connection.execute(
                f"UPDATE {table} SET {assignments} WHERE {key}=?",
                (*changes.values(), identifier),
            )

    def start_update(
        self, job: dict[str, Any], *, before: dict[str, Any],
        input_payload: dict[str, Any], model: str = "",
    ) -> str:
        memory_type = str(job.get("memory_type") or "")
        update_id = self._id(f"{memory_type}_update")
        now = format_timestamp()
        payload = job.get("payload") or {}
        add_ids = payload.get("trace_add_ids") or []
        add_id = str(add_ids[0]) if add_ids else None
        values = (
            update_id, add_id, str(job.get("job_id") or "") or None, str(job.get("memory_id") or ""),
            _dump(before, {}), _dump(input_payload, {}), int(before.get("version") or 0), model,
            "running", now, now,
        )
        with self._connect() as connection:
            if memory_type == "segment":
                connection.execute(
                    "INSERT INTO segment_update(segment_update_id,add_id,memory_outbox_id,segment_id,old_segment_json,distill_input_json,before_version,model,status,started_at,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    values,
                )
            elif memory_type == "experience":
                connection.execute(
                    "INSERT INTO experience_update(experience_update_id,add_id,memory_outbox_id,experience_id,old_experience_json,distill_input_json,before_version,model,status,started_at,created_at,segment_update_id) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (*values, payload.get("parent_segment_update_id")),
                )
            else:
                raise ValueError(f"Unsupported trajectory memory type: {memory_type}")
        return update_id

    def finish_update(
        self, memory_type: str, update_id: str, *, after: dict[str, Any],
        prompt: str = "", output: str = "", elapsed_ms: float = 0,
        status: str = "completed", skip_reason: str = "", error_message: str = "",
        connection: sqlite3.Connection | None = None,
    ) -> None:
        table = "segment_update" if memory_type == "segment" else "experience_update"
        key = "segment_update_id" if memory_type == "segment" else "experience_update_id"
        snapshot = "new_segment_json" if memory_type == "segment" else "new_experience_json"
        now = format_timestamp()
        values = (
            _dump(after, {}), prompt, output, float(elapsed_ms), status, skip_reason,
            error_message, now, int(after.get("version") or 0), update_id,
        )
        if connection is not None:
            connection.execute(
                f"UPDATE {table} SET {snapshot}=?,prompt_text=?,output_text=?,elapsed_ms=?,status=?,skip_reason=?,error_message=?,finished_at=?,after_version=? WHERE {key}=?",
                values,
            )
            return
        with self._connect() as trace_connection:
            trace_connection.execute(
                f"UPDATE {table} SET {snapshot}=?,prompt_text=?,output_text=?,elapsed_ms=?,status=?,skip_reason=?,error_message=?,finished_at=?,after_version=? WHERE {key}=?",
                values,
            )

    def get(self, session_id: str, chat_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            chat = connection.execute(
                "SELECT * FROM chat WHERE chat_id=? AND session_id=?", (chat_id, session_id)
            ).fetchone()
            if chat is None:
                return None
            result: dict[str, Any] = {"chat": self._decode_row(chat)}
            retriever = connection.execute(
                "SELECT * FROM retriever WHERE retriever_id=?", (chat["retriever_id"],)
            ).fetchone()
            add = connection.execute(
                'SELECT * FROM "add" WHERE add_id=?', (chat["add_id"],)
            ).fetchone()
            result["retriever"] = self._decode_row(retriever)
            result["add"] = self._decode_row(add)
            result["segment_updates"] = self._list_updates(connection, "segment_update", "add_id", chat["add_id"])
            result["experience_updates"] = self._list_updates(connection, "experience_update", "add_id", chat["add_id"])
            outbox_ids = self._json_value(add["memory_outbox_ids_json"] if add else "[]", [])
            result["outbox"] = []
            for job_id in outbox_ids if isinstance(outbox_ids, list) else []:
                row = connection.execute("SELECT * FROM memory_outbox WHERE job_id=?", (str(job_id),)).fetchone()
                if row:
                    result["outbox"].append(self._decode_row(row))
            return result

    def list_operations(self, *, operation_type: str = "", status: str = "", limit: int = 100) -> list[dict[str, Any]]:
        queries = {
            "chat": "SELECT chat_id AS id, 'chat' AS type, status, created_at, updated_at, elapsed_ms FROM chat",
            "retriever": "SELECT retriever_id AS id, 'retriever' AS type, status, created_at, updated_at, elapsed_ms FROM retriever",
            "add": 'SELECT add_id AS id, \'add\' AS type, status, created_at, updated_at, elapsed_ms FROM "add"',
            "segment_update": "SELECT segment_update_id AS id, 'segment_update' AS type, status, created_at, finished_at AS updated_at, elapsed_ms FROM segment_update",
            "experience_update": "SELECT experience_update_id AS id, 'experience_update' AS type, status, created_at, finished_at AS updated_at, elapsed_ms FROM experience_update",
        }
        selected = [operation_type] if operation_type in queries else list(queries)
        with self._connect() as connection:
            rows: list[dict[str, Any]] = []
            for kind in selected:
                sql = queries[kind]
                params: list[Any] = []
                if status:
                    sql += " WHERE status=?"
                    params.append(status)
                sql += " ORDER BY created_at DESC LIMIT ?"
                params.append(max(1, int(limit)))
                rows.extend(dict(row) for row in connection.execute(sql, params).fetchall())
        rows.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return rows[: max(1, int(limit))]

    def list_memory_updates(self, memory_type: str, memory_id: str) -> list[dict[str, Any]]:
        """Return every distillation attempt for one memory, oldest first."""
        targets = {
            "segment": ("segment_update", "segment_id"),
            "experience": ("experience_update", "experience_id"),
        }
        target = targets.get(memory_type)
        if target is None:
            raise ValueError(f"Unsupported distillation memory type: {memory_type}")
        table, key = target
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM {table} WHERE {key}=? "
                "ORDER BY COALESCE(started_at, created_at) ASC, created_at ASC, rowid ASC",
                (memory_id,),
            ).fetchall()
        return [self._decode_row(row) for row in rows]

    def list_experience_distillation_history(self, experience_id: str) -> list[dict[str, Any]]:
        """Return Experience and descendant Segment attempts in one timeline."""
        with self._connect() as connection:
            segment_ids = [
                str(row["segment_id"])
                for row in connection.execute(
                    "SELECT segment_id FROM segment_memory WHERE experience_id=?",
                    (experience_id,),
                ).fetchall()
            ]
            experience_rows = connection.execute(
                "SELECT * FROM experience_update WHERE experience_id=?",
                (experience_id,),
            ).fetchall()
            segment_rows = []
            if segment_ids:
                placeholders = ",".join("?" for _ in segment_ids)
                segment_rows = connection.execute(
                    f"SELECT * FROM segment_update WHERE segment_id IN ({placeholders})",
                    segment_ids,
                ).fetchall()
        history = []
        for row in experience_rows:
            item = self._decode_row(row)
            item["record_type"] = "experience"
            history.append(item)
        for row in segment_rows:
            item = self._decode_row(row)
            item["record_type"] = "segment"
            history.append(item)
        history.sort(
            key=lambda item: (
                str(item.get("started_at") or item.get("created_at") or ""),
                str(item.get("created_at") or ""),
                str(item.get("segment_update_id") or item.get("experience_update_id") or ""),
            )
        )
        return history

    def get_operation(self, operation_type: str, operation_id: str) -> dict[str, Any] | None:
        table_by_type = {
            "chat": "chat",
            "retriever": "retriever",
            "add": '"add"',
            "segment_update": "segment_update",
            "experience_update": "experience_update",
        }
        key_by_type = {
            "chat": "chat_id",
            "retriever": "retriever_id",
            "add": "add_id",
            "segment_update": "segment_update_id",
            "experience_update": "experience_update_id",
        }
        table = table_by_type.get(operation_type)
        key = key_by_type.get(operation_type)
        if not table or not key:
            return None
        with self._connect() as connection:
            row = connection.execute(f"SELECT * FROM {table} WHERE {key}=?", (operation_id,)).fetchone()
            if row is None:
                return None
            result: dict[str, Any] = {"type": operation_type, "operation": self._decode_row(row)}
            if operation_type == "chat":
                chat = dict(row)
                result.update(self.get(str(chat["session_id"]), operation_id) or {})
            elif operation_type == "add":
                result["segment_updates"] = self._list_updates(connection, "segment_update", "add_id", operation_id)
                result["experience_updates"] = self._list_updates(connection, "experience_update", "add_id", operation_id)
            return result

    @staticmethod
    def _json_value(value: Any, fallback: Any) -> Any:
        return _loads(value, fallback)

    @classmethod
    def _decode_row(cls, row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        item = dict(row)
        for key in list(item):
            if key.endswith("_json"):
                item[key[:-5]] = cls._json_value(item.pop(key), {})
        return item

    @classmethod
    def _list_updates(cls, connection: sqlite3.Connection, table: str, key: str, value: str) -> list[dict[str, Any]]:
        rows = connection.execute(f"SELECT * FROM {table} WHERE {key}=? ORDER BY created_at ASC", (value,)).fetchall()
        return [cls._decode_row(row) for row in rows]


TrajectoryRepository = TraceRepository

__all__ = ["TRAJECTORY_SCHEMA", "TraceRepository", "TrajectoryRepository"]
