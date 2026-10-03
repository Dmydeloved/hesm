"""Chat-session navigation over the normalized chat trajectory table."""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from .time_utils import format_timestamp


SESSION_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_session (
    session_id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'active',
    create_at TEXT NOT NULL,
    update_at TEXT NOT NULL,
    chats TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(chats))
);
CREATE INDEX IF NOT EXISTS idx_chat_session_update_at ON chat_session(update_at DESC);
"""


def _loads(value: Any, fallback: Any) -> Any:
    try:
        return json.loads(value) if isinstance(value, str) else value
    except (json.JSONDecodeError, TypeError):
        return fallback


class SessionManager:
    """Manage session metadata; messages live in the chat table."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(SESSION_SCHEMA)
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 15000")
        return connection

    @staticmethod
    def _now() -> str:
        return format_timestamp()

    def create(self, *, title: str = "", session_id: str | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        identifier = str(session_id or f"session_{uuid.uuid4().hex[:16]}")
        now = self._now()
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO chat_session(session_id,status,create_at,update_at,chats) VALUES(?,?,?,?,?)",
                (identifier, "active", now, now, "[]"),
            )
        return self.get(identifier)

    def ensure(self, session_id: str, title: str = "") -> dict[str, Any]:
        return self.get(session_id, required=False) or self.create(session_id=session_id, title=title)

    def get(self, session_id: str, *, required: bool = True) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM chat_session WHERE session_id=?", (session_id,)
            ).fetchone()
            if row is None:
                if required:
                    raise KeyError(f"Session not found: {session_id}")
                return None
            item = dict(row)
            chat_ids = _loads(item.get("chats"), [])
            item["chats"] = chat_ids if isinstance(chat_ids, list) else []
            chats = self._load_chats(connection, item["chats"])
        return self._with_view_fields(item, chats)

    def list(self, *, limit: int = 50, include_archived: bool = False) -> list[dict[str, Any]]:
        where = "" if include_archived else "WHERE status='active'"
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM chat_session {where} ORDER BY update_at DESC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
            items = []
            for row in rows:
                item = dict(row)
                chat_ids = _loads(item.get("chats"), [])
                chat_ids = chat_ids if isinstance(chat_ids, list) else []
                items.append(self._with_view_fields(item, self._load_chats(connection, chat_ids)))
        return items

    def _load_chats(self, connection: sqlite3.Connection, chat_ids: list[Any]) -> list[dict[str, Any]]:
        if not chat_ids:
            return []
        placeholders = ",".join("?" for _ in chat_ids)
        rows = connection.execute(
            f"SELECT * FROM chat WHERE chat_id IN ({placeholders}) ORDER BY created_at ASC",
            [str(value) for value in chat_ids],
        ).fetchall()
        by_id = {str(row["chat_id"]): dict(row) for row in rows}
        return [by_id[str(identifier)] for identifier in chat_ids if str(identifier) in by_id]

    @staticmethod
    def _with_view_fields(item: dict[str, Any], chats: list[dict[str, Any]]) -> dict[str, Any]:
        result = dict(item)
        result["title"] = str(chats[0].get("user_input") or "智能对话")[:60] if chats else "智能对话"
        result["preview"] = str(chats[-1].get("assistant_output") or chats[-1].get("user_input") or "")[:120] if chats else "暂无消息"
        result["turn_count"] = len(chats)
        messages: list[dict[str, Any]] = []
        for chat in chats:
            metadata = {"chat_id": chat.get("chat_id"), "retriever_id": chat.get("retriever_id"), "add_id": chat.get("add_id")}
            messages.extend([
                {"role": "user", "content": chat.get("user_input", ""), "created_at": chat.get("created_at"), "metadata": metadata},
                {"role": "assistant", "content": chat.get("assistant_output", ""), "created_at": chat.get("updated_at"), "metadata": metadata},
            ])
        result["messages"] = messages
        return result

    def archive(self, session_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE chat_session SET status='archived',update_at=? WHERE session_id=?",
                (self._now(), session_id),
            )
            if not cursor.rowcount:
                raise KeyError(f"Session not found: {session_id}")
        return self.get(session_id)


__all__ = ["SESSION_SCHEMA", "SessionManager"]
