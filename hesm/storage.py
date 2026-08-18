from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS qa_memory (
    qa_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    user_input TEXT NOT NULL,
    assistant_output TEXT NOT NULL,
    tools_json TEXT NOT NULL,
    topic TEXT NOT NULL,
    intent TEXT NOT NULL,
    core_entity TEXT NOT NULL,
    entities_json TEXT NOT NULL,
    segment_id TEXT NOT NULL,
    status TEXT NOT NULL,
    confidence REAL NOT NULL,
    reasoning TEXT NOT NULL,
    FOREIGN KEY (segment_id) REFERENCES segment_memory(segment_id)
);

CREATE TABLE IF NOT EXISTS segment_memory (
    segment_id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    intent TEXT NOT NULL,
    core_entity TEXT NOT NULL,
    qa_ids_json TEXT NOT NULL,
    status TEXT NOT NULL,
    summary TEXT NOT NULL,
    experience_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    version INTEGER NOT NULL,
    last_summarized_qa_count INTEGER NOT NULL,
    FOREIGN KEY (experience_id) REFERENCES experience_memory(experience_id)
);

CREATE TABLE IF NOT EXISTS experience_memory (
    experience_id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    core_entity TEXT NOT NULL,
    intents_link_json TEXT NOT NULL,
    segment_ids_json TEXT NOT NULL,
    summary_json TEXT NOT NULL,
    state_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    version INTEGER NOT NULL,
    last_summarized_segment_count INTEGER NOT NULL,
    history_experience_json TEXT NOT NULL DEFAULT '{}'
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
"""


JSON_FIELDS = {
    "tools_json",
    "entities_json",
    "qa_ids_json",
    "intents_link_json",
    "segment_ids_json",
    "state_json",
    "summary_json",
    "history_experience_json",
    "retrieval_cache_json",
    "vector_json",
    "messages_json",
    "metadata_json",
}

JSON_DEFAULTS: dict[str, Any] = {
    "tools_json": [],
    "entities_json": [],
    "qa_ids_json": [],
    "intents_link_json": [],
    "segment_ids_json": [],
    "state_json": {},
    "history_experience_json": {},
    "retrieval_cache_json": {},
    "vector_json": [],
    "messages_json": [],
    "metadata_json": {},
}


class MemoryStorage:
    """Small SQLite repository for QA, Segment, Experience and runtime state."""

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
        if not read_only:
            self.connection.executescript(SCHEMA)
            self._ensure_runtime_state_columns()
            self._ensure_experience_columns()
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

    def _ensure_experience_columns(self) -> None:
        """Migrate existing databases without rebuilding their memory tables."""
        columns = {
            row["name"]
            for row in self.connection.execute(
                "PRAGMA table_info(experience_memory)"
            ).fetchall()
        }
        if "history_experience_json" not in columns:
            self.connection.execute(
                "ALTER TABLE experience_memory "
                "ADD COLUMN history_experience_json TEXT NOT NULL DEFAULT '{}'"
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
            if key == 'summary_json':
                if not value:
                    data['summary'] = ''
                else:
                    try:
                        parsed = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        parsed = value
                    if isinstance(parsed, str):
                        data['summary'] = parsed
                    elif isinstance(parsed, dict):
                        data['summary'] = (
                            parsed.get('summary')
                            or parsed.get('long')
                            or parsed.get('short')
                            or ''
                        )
                    else:
                        data['summary'] = ''
                continue
            if key in JSON_FIELDS:
                default = JSON_DEFAULTS[key]
                try:
                    data[key[:-5]] = json.loads(value) if value else default.copy()
                except (json.JSONDecodeError, TypeError):
                    data[key[:-5]] = default.copy()
            else:
                data[key] = value
        return data

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

    def insert_qa(self, qa: dict[str, Any]) -> None:
        self.connection.execute(
            """
            INSERT INTO qa_memory (
                qa_id, timestamp, user_input, assistant_output, tools_json,
                topic, intent, core_entity, entities_json, segment_id, status,
                confidence, reasoning
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                qa["qa_id"],
                qa["timestamp"],
                qa["user_input"],
                qa["assistant_output"],
                json.dumps(qa["tools"], ensure_ascii=False),
                qa["topic"],
                qa["intent"],
                qa["core_entity"],
                json.dumps(qa["entities"], ensure_ascii=False),
                qa["segment_id"],
                qa["status"],
                qa["confidence"],
                qa["reasoning"],
            ),
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
        self.connection.execute(
            """
            INSERT INTO segment_memory (
                segment_id, topic, intent, core_entity, qa_ids_json, status,
                summary, experience_id, created_at, updated_at, version,
                last_summarized_qa_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                segment["segment_id"],
                segment["topic"],
                segment["intent"],
                segment["core_entity"],
                json.dumps(segment["qa_ids"], ensure_ascii=False),
                segment["status"],
                segment["summary"],
                segment["experience_id"],
                segment["created_at"],
                segment["updated_at"],
                segment["version"],
                segment["last_summarized_qa_count"],
            ),
        )

    def update_segment(self, segment: dict[str, Any]) -> None:
        self.connection.execute(
            """
            UPDATE segment_memory SET
                qa_ids_json = ?,
                status = ?,
                summary = ?,
                updated_at = ?,
                version = ?,
                last_summarized_qa_count = ?
            WHERE segment_id = ?
            """,
            (
                json.dumps(segment["qa_ids"], ensure_ascii=False),
                segment["status"],
                segment["summary"],
                segment["updated_at"],
                segment["version"],
                segment["last_summarized_qa_count"],
                segment["segment_id"],
            ),
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
            WHERE topic = ?
              AND core_entity = ?
              AND json_extract(state_json, '$.status') = 'in_progress'
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
            WHERE json_extract(state_json, '$.status') = 'completed'
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
        keywords: list[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Broad relational QA recall used only for low-confidence queries."""
        conditions: list[str] = []
        parameters: list[Any] = []
        topic = str(topic or "").strip()
        core_entity = str(core_entity or "").strip()
        normalized_entities = sorted({
            str(value).strip().casefold()
            for value in entities
            if str(value).strip()
        })
        normalized_keywords = [
            str(value).strip() for value in keywords if str(value).strip()
        ]

        if topic:
            conditions.append("q.topic = ?")
            parameters.append(topic)
        if core_entity:
            conditions.append("q.core_entity = ?")
            parameters.append(core_entity)
        if normalized_entities:
            placeholders = ", ".join("?" for _ in normalized_entities)
            # JSON1 membership avoids substring matches in serialized entity data.
            conditions.append(
                "EXISTS ("
                "SELECT 1 FROM json_each("
                "CASE WHEN json_valid(q.entities_json) THEN q.entities_json ELSE '[]' END"
                ") AS entity WHERE lower(trim(CAST(entity.value AS TEXT))) "
                f"IN ({placeholders})"
                ")"
            )
            parameters.extend(normalized_entities)
        searchable = (
            "COALESCE(q.user_input, '') || ' ' || "
            "COALESCE(q.assistant_output, '') || ' ' || "
            "COALESCE(q.intent, '')"
        )
        for keyword in normalized_keywords:
            conditions.append(f"({searchable}) LIKE ?")
            parameters.append(f"%{keyword}%")
        if not conditions:
            return []

        rows = self.connection.execute(
            f"""
            SELECT q.*
            FROM qa_memory AS q
            WHERE q.status = 'active' AND ({' OR '.join(conditions)})
            ORDER BY q.timestamp DESC
            LIMIT ?
            """,
            (*parameters, max(1, int(limit))),
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

    def insert_experience(self, experience: dict[str, Any]) -> None:
        self.connection.execute(
            """
            INSERT INTO experience_memory (
                experience_id, topic, core_entity, intents_link_json,
                segment_ids_json, summary_json, state_json, created_at,
                updated_at, version, last_summarized_segment_count,
                history_experience_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experience["experience_id"],
                experience["topic"],
                experience["core_entity"],
                json.dumps(experience["intents_link"], ensure_ascii=False),
                json.dumps(experience["segment_ids"], ensure_ascii=False),
                json.dumps(experience["summary"], ensure_ascii=False),
                json.dumps(experience["state"], ensure_ascii=False),
                experience["created_at"],
                experience["updated_at"],
                experience["version"],
                experience["last_summarized_segment_count"],
                json.dumps(
                    experience.get("history_experience") or {},
                    ensure_ascii=False,
                ),
            ),
        )

    def update_experience(self, experience: dict[str, Any]) -> None:
        self.connection.execute(
            """
            UPDATE experience_memory SET
                intents_link_json = ?,
                segment_ids_json = ?,
                summary_json = ?,
                state_json = ?,
                updated_at = ?,
                version = ?,
                last_summarized_segment_count = ?,
                history_experience_json = ?
            WHERE experience_id = ?
            """,
            (
                json.dumps(experience["intents_link"], ensure_ascii=False),
                json.dumps(experience["segment_ids"], ensure_ascii=False),
                json.dumps(experience["summary"], ensure_ascii=False),
                json.dumps(experience["state"], ensure_ascii=False),
                experience["updated_at"],
                experience["version"],
                experience["last_summarized_segment_count"],
                json.dumps(
                    experience.get("history_experience") or {},
                    ensure_ascii=False,
                ),
                experience["experience_id"],
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
            WHERE segment_id IN ({placeholders}) AND status = 'active'
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
            WHERE segment_id IN ({placeholders}) AND status = 'active'
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
        }:
            raise ValueError(f"Unsupported table: {table}")
        return int(self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
