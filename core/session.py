"""Independent chat-session history management stored in the HESM database."""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from pathlib import Path
from threading import RLock
from typing import Any

from .time_utils import format_timestamp


logger = logging.getLogger(__name__)


SESSION_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_session (
    session_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    messages_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_chat_session_updated_at
ON chat_session(updated_at DESC);
"""


class SessionManager:
    """Manage chat histories without changing Experience/Segment/QA behavior."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        with self._connect() as connection:
            connection.executescript(SESSION_SCHEMA)
            connection.commit()
        logger.info("Session manager initialized database=%s", self.database_path.resolve())

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=15)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _now() -> str:
        return format_timestamp()

    @staticmethod
    def _loads(value: str, fallback: Any) -> Any:
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return fallback

    @staticmethod
    def _table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
        """检查兼容数据源表是否已经完成初始化。"""
        row = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
            (table_name,),
        ).fetchone()
        return row is not None

    def create(
        self,
        *,
        title: str = "新会话",
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        identifier = str(session_id or f"session_{uuid.uuid4().hex[:16]}")
        now = self._now()
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO chat_session "
                "(session_id,title,messages_json,status,created_at,updated_at,metadata_json) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    identifier, str(title or "新会话")[:120], "[]", "active",
                    now, now, json.dumps(metadata or {}, ensure_ascii=False),
                ),
            )
            connection.commit()
        logger.info("Chat session created session_id=%s title=%s", identifier, title)
        return self.get(identifier)

    def ensure(self, session_id: str, title: str = "新会话") -> dict[str, Any]:
        existing = self.get(session_id, required=False)
        return existing or self.create(session_id=session_id, title=title)

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
        item["messages"] = self._loads(item.pop("messages_json"), [])
        item["metadata"] = self._loads(item.pop("metadata_json"), {})
        item["turn_count"] = sum(
            message.get("role") == "assistant" for message in item["messages"]
        )
        return item

    def list(self, *, limit: int = 50, include_archived: bool = False) -> list[dict[str, Any]]:
        where = "" if include_archived else "WHERE status='active'"
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM chat_session {where} ORDER BY updated_at DESC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
        items = []
        for row in rows:
            item = dict(row)
            messages = self._loads(item.pop("messages_json"), [])
            item.pop("metadata_json", None)
            item["turn_count"] = sum(
                message.get("role") == "assistant" for message in messages
            )
            item["preview"] = next(
                (str(message.get("content") or "")[:100] for message in reversed(messages)),
                "暂无消息",
            )
            items.append(item)
        return items

    def append_turn(
        self,
        session_id: str,
        *,
        user_content: str,
        assistant_content: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT title,messages_json,metadata_json FROM chat_session WHERE session_id=?",
                (session_id,),
            ).fetchone()
            if row is None:
                self.create(session_id=session_id, title=user_content[:60] or "新会话")
                row = connection.execute(
                    "SELECT title,messages_json,metadata_json FROM chat_session WHERE session_id=?",
                    (session_id,),
                ).fetchone()
            messages = self._loads(row["messages_json"], [])
            now = self._now()
            turn_metadata = dict(metadata or {})
            messages.extend([
                {
                    "role": "user",
                    "content": user_content,
                    "created_at": now,
                    "metadata": turn_metadata,
                },
                {
                    "role": "assistant",
                    "content": assistant_content,
                    "created_at": now,
                    "metadata": turn_metadata,
                },
            ])
            session_metadata = self._loads(row["metadata_json"], {})
            session_metadata.update(turn_metadata)
            title = row["title"]
            if not messages[:-2] and title == "新会话":
                title = user_content[:60] or title
            connection.execute(
                "UPDATE chat_session SET title=?,messages_json=?,updated_at=?,metadata_json=? "
                "WHERE session_id=?",
                (
                    title, json.dumps(messages, ensure_ascii=False), now,
                    json.dumps(session_metadata, ensure_ascii=False), session_id,
                ),
            )
            connection.commit()
        logger.info(
            "Chat turn persisted session_id=%s user_content=%s assistant_content=%s",
            session_id,
            user_content,
            assistant_content,
        )
        return self.get(session_id)

    def rename(self, session_id: str, title: str) -> dict[str, Any]:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE chat_session SET title=?,updated_at=? WHERE session_id=?",
                (str(title).strip()[:120] or "新会话", self._now(), session_id),
            )
            if not cursor.rowcount:
                raise KeyError(f"Session not found: {session_id}")
            connection.commit()
        logger.info("Chat session renamed session_id=%s title=%s", session_id, title)
        return self.get(session_id)

    def archive(self, session_id: str) -> dict[str, Any]:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE chat_session SET status='archived',updated_at=? WHERE session_id=?",
                (self._now(), session_id),
            )
            if not cursor.rowcount:
                raise KeyError(f"Session not found: {session_id}")
            connection.commit()
        logger.info("Chat session archived session_id=%s", session_id)
        return self.get(session_id)

    def import_legacy_chat_turns(self) -> int:
        """单向导入旧版存放在 QA 工具调用中的聊天记录。"""
        imported = 0
        with self._connect() as connection:
            # 新数据库首次启动时 QA 表可能尚未由 MemoryStorage 创建，此时无需迁移。
            if not self._table_exists(connection, "qa_memory"):
                return 0
            rows = connection.execute(
                "SELECT qa_id,user_input,assistant_output,tools_json FROM qa_memory "
                "ORDER BY rowid"
            ).fetchall()
        for row in rows:
            try:
                tools = json.loads(row["tools_json"] or "[]")
            except (json.JSONDecodeError, TypeError):
                continue
            trace = next(
                (
                    item for item in tools
                    if isinstance(item, dict) and item.get("type") == "hesm_chat_turn"
                ),
                None,
            )
            if trace is None:
                continue
            if trace.get("session_managed") is True:
                continue
            session_id = str(trace.get("session_id") or trace.get("state_key") or "web_chat")
            session = self.ensure(session_id, str(row["user_input"] or "新会话")[:60])
            metadata = session.get("metadata") or {}
            imported_ids = set(metadata.get("legacy_qa_ids") or [])
            if row["qa_id"] in imported_ids:
                continue
            imported_ids.add(row["qa_id"])
            self.append_turn(
                session_id,
                user_content=str(row["user_input"] or ""),
                assistant_content=str(row["assistant_output"] or ""),
                metadata={"legacy_qa_ids": sorted(imported_ids)},
            )
            imported += 1
        logger.info("Legacy chat session import completed imported=%s", imported)
        return imported

    def remove_chat_traces_from_qa_tools(self) -> int:
        """清除废弃的聊天诊断记录，同时保留真实的工具调用。"""
        changed = 0
        with self._lock, self._connect() as connection:
            # 兼容全新数据库：QA 表不存在时没有需要清理的数据。
            if not self._table_exists(connection, "qa_memory"):
                return 0
            rows = connection.execute(
                "SELECT qa_id,tools_json FROM qa_memory WHERE tools_json != '[]'"
            ).fetchall()
            for row in rows:
                tools = self._loads(row["tools_json"], [])
                if not isinstance(tools, list):
                    tools = []
                retained = [
                    item for item in tools
                    if not (
                        isinstance(item, dict)
                        and item.get("type") == "hesm_chat_turn"
                    )
                ]
                if retained == tools:
                    continue
                connection.execute(
                    "UPDATE qa_memory SET tools_json=? WHERE qa_id=?",
                    (json.dumps(retained, ensure_ascii=False), row["qa_id"]),
                )
                changed += 1
            connection.commit()
        logger.info("Legacy QA chat traces cleanup completed changed=%s", changed)
        return changed


__all__ = ["SESSION_SCHEMA", "SessionManager"]
