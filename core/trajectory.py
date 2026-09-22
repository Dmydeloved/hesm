"""Durable per-turn chat trajectories and asynchronous derivation events."""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from .time_utils import format_timestamp


TRACE_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_trace (
    turn_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    user_input TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chat_trace_event (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (turn_id) REFERENCES chat_trace(turn_id)
);
CREATE TABLE IF NOT EXISTS chat_trace_memory (
    turn_id TEXT NOT NULL,
    qa_id TEXT NOT NULL,
    segment_id TEXT NOT NULL,
    experience_id TEXT NOT NULL,
    PRIMARY KEY (turn_id, qa_id),
    FOREIGN KEY (turn_id) REFERENCES chat_trace(turn_id)
);
CREATE INDEX IF NOT EXISTS idx_chat_trace_session ON chat_trace(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_chat_trace_event_turn ON chat_trace_event(turn_id, event_id);
CREATE INDEX IF NOT EXISTS idx_chat_trace_memory_qa ON chat_trace_memory(qa_id);
CREATE INDEX IF NOT EXISTS idx_chat_trace_memory_segment ON chat_trace_memory(segment_id);
"""


class TraceRepository:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        with self._connect() as connection:
            connection.executescript(TRACE_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def start(self, session_id: str, user_input: str) -> str:
        turn_id = f"turn_{uuid.uuid4().hex}"
        now = format_timestamp()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO chat_trace VALUES (?,?,?,?,?,?)",
                (turn_id, session_id, user_input, "running", now, now),
            )
        return turn_id

    def append(self, turn_id: str, stage: str, status: str, payload: dict[str, Any]) -> None:
        now = format_timestamp()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO chat_trace_event (turn_id,stage,status,payload_json,created_at) "
                "VALUES (?,?,?,?,?)",
                (turn_id, stage, status, json.dumps(payload, ensure_ascii=False, default=str), now),
            )
            connection.execute(
                "UPDATE chat_trace SET updated_at=? WHERE turn_id=?", (now, turn_id)
            )

    def finish(self, turn_id: str, status: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE chat_trace SET status=?,updated_at=? WHERE turn_id=?",
                (status, format_timestamp(), turn_id),
            )

    def link_memories(self, turn_id: str, memories: list[dict[str, Any]]) -> None:
        with self._connect() as connection:
            connection.executemany(
                "INSERT OR IGNORE INTO chat_trace_memory "
                "(turn_id,qa_id,segment_id,experience_id) VALUES (?,?,?,?)",
                [
                    (turn_id, str(item["qa_id"]), str(item["segment_id"]),
                     str(item["experience_id"]))
                    for item in memories
                ],
            )

    def turns_for_qas(self, qa_ids: list[str]) -> list[str]:
        if not qa_ids:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT DISTINCT turn_id FROM chat_trace_memory WHERE qa_id IN ("
                + ",".join("?" for _ in qa_ids) + ")", qa_ids,
            ).fetchall()
        return [str(row["turn_id"]) for row in rows]

    def turns_for_segments(self, segment_ids: list[str]) -> list[str]:
        if not segment_ids:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT DISTINCT turn_id FROM chat_trace_memory WHERE segment_id IN ("
                + ",".join("?" for _ in segment_ids) + ")", segment_ids,
            ).fetchall()
        return [str(row["turn_id"]) for row in rows]

    def get(self, session_id: str, turn_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM chat_trace WHERE turn_id=? AND session_id=?",
                (turn_id, session_id),
            ).fetchone()
            if row is None:
                return None
            events = connection.execute(
                "SELECT event_id,stage,status,payload_json,created_at FROM chat_trace_event "
                "WHERE turn_id=? ORDER BY event_id", (turn_id,),
            ).fetchall()
            jobs = connection.execute(
                "SELECT job_id,job_type,memory_type,memory_id,status,retry_count,last_error,"
                "payload_json,created_at,updated_at FROM memory_outbox "
                "WHERE payload_json LIKE ?",
                (f'%"{turn_id}"%',),
            ).fetchall()
        result = dict(row)
        result["events"] = [
            {**dict(event), "payload": json.loads(event["payload_json"])}
            for event in events
        ]
        for event in result["events"]:
            event.pop("payload_json")
        result["jobs"] = []
        for job in jobs:
            try:
                payload = json.loads(job["payload_json"] or "{}")
            except (ValueError, TypeError):
                payload = {}
            if turn_id in payload.get("trace_turn_ids", []):
                item = dict(job)
                item.pop("payload_json")
                result["jobs"].append(item)
        return result


__all__ = ["TRACE_SCHEMA", "TraceRepository"]
