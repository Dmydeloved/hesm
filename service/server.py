"""HESM web server: live memory management, retrieval and static frontend."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from threading import Lock, RLock
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from hesm.config import PROJECT_ROOT, config_path
from hesm.logging_config import configure_logging
from hesm.session import SessionManager
from hesm.storage import MemoryStorage
from service.hesm_service import HESMService


configure_logging()
LOGGER = logging.getLogger("hesm.service")
FRONTEND_DIR = PROJECT_ROOT / "frontend"
DATABASE_PATH = config_path("paths", "memory_db")
CHROMA_PATH = config_path("paths", "chroma")


class NoCacheStaticFiles(StaticFiles):
    """Always return current development assets instead of conditional 304s."""

    def is_not_modified(self, response_headers: Any, request_headers: Any) -> bool:
        return False


def _json(value: Any, fallback: Any) -> Any:
    if value in (None, ""):
        return fallback
    try:
        return json.loads(value) if isinstance(value, str) else value
    except (json.JSONDecodeError, TypeError):
        return fallback


class MemoryRepository:
    """Small web-facing repository that keeps management operations auditable."""

    TABLES = {
        "experience": ("experience_memory", "experience_id"),
        "segment": ("segment_memory", "segment_id"),
        "qa": ("qa_memory", "qa_id"),
    }

    def __init__(self, database: Path) -> None:
        self.database = database
        self._lock = RLock()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _ensure_database(self) -> None:
        if not self.database.exists():
            raise HTTPException(status_code=503, detail="HESM 记忆数据库尚未创建")

    @staticmethod
    def _experience(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["intents"] = _json(item.pop("intents_link_json", "[]"), [])
        item["segment_ids"] = _json(item.pop("segment_ids_json", "[]"), [])
        item["summary"] = _json(item.pop("summary_json", "{}"), {})
        item["state"] = {"status": item.get("status", "open")}
        item["history_experience"] = _json(
            item.pop("history_experience_json", "{}"), {}
        )
        item["segment_count"] = int(item.get("segment_count") or len(item["segment_ids"]))
        item["qa_count"] = int(item.get("qa_count") or 0)
        return item

    @staticmethod
    def _segment(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["qa_ids"] = _json(item.pop("qa_ids_json", "[]"), [])
        item["summary"] = _json(item.pop("summary_json", "{}"), {})
        item["qa_count"] = int(item.get("qa_count") or len(item["qa_ids"]))
        return item

    @staticmethod
    def _qa(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["tools"] = _json(item.pop("tools_json", "[]"), [])
        item["entities"] = _json(item.pop("entities_json", "[]"), [])
        item["reasoning"] = item.get("reason", "")
        return item

    def stats(self) -> dict[str, Any]:
        self._ensure_database()
        with self._connect() as connection:
            counts = {
                level: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for level, (table, _) in self.TABLES.items()
            }
            runtime_rows = connection.execute(
                "SELECT state_key, current_experience_id, current_segment_id, updated_at "
                "FROM runtime_state ORDER BY updated_at DESC"
            ).fetchall()
        return {"counts": counts, "runtime": [dict(row) for row in runtime_rows]}

    def list_memories(
        self,
        level: Literal["experience", "segment", "qa"],
        *,
        search: str = "",
        status: str = "",
        parent_id: str = "",
        experience_id: str = "",
        page: int = 1,
        page_size: int = 30,
    ) -> dict[str, Any]:
        self._ensure_database()
        table, id_column = self.TABLES[level]
        aliases = {"experience": "e", "segment": "s", "qa": "q"}
        alias = aliases[level]
        if level == "experience":
            select = (
                "e.*, COUNT(DISTINCT s.segment_id) AS segment_count, "
                "COUNT(DISTINCT q.qa_id) AS qa_count"
            )
            joins = (
                " LEFT JOIN segment_memory s ON s.experience_id=e.experience_id"
                " LEFT JOIN qa_memory q ON q.segment_id=s.segment_id"
            )
            search_columns = ["e.experience_id", "e.topic", "e.core_entity", "e.summary_json"]
            group = " GROUP BY e.experience_id"
        elif level == "segment":
            select = "s.*, COUNT(q.qa_id) AS qa_count"
            joins = " LEFT JOIN qa_memory q ON q.segment_id=s.segment_id"
            search_columns = ["s.segment_id", "s.topic", "s.intent", "s.core_entity", "s.summary_json"]
            group = " GROUP BY s.segment_id"
        else:
            select = "q.*, s.experience_id"
            joins = " LEFT JOIN segment_memory s ON s.segment_id=q.segment_id"
            search_columns = [
                "q.qa_id", "q.topic", "q.intent", "q.core_entity",
                "q.user_input", "q.assistant_output", "q.entities_json",
                "q.source_id",
            ]
            group = ""

        conditions: list[str] = []
        parameters: list[Any] = []
        if search.strip():
            conditions.append("(" + " OR ".join(f"{column} LIKE ?" for column in search_columns) + ")")
            parameters.extend([f"%{search.strip()}%"] * len(search_columns))
        if status.strip():
            conditions.append(f"{alias}.status = ?")
            parameters.append(status.strip())
        if parent_id.strip():
            parent_column = "s.experience_id" if level == "segment" else "q.segment_id"
            if level == "experience":
                raise HTTPException(status_code=400, detail="Experience 不支持 parent_id")
            conditions.append(f"{parent_column} = ?")
            parameters.append(parent_id.strip())
        if experience_id.strip():
            if level != "qa":
                raise HTTPException(
                    status_code=400,
                    detail="experience_id 仅支持查询 QA",
                )
            conditions.append("s.experience_id = ?")
            parameters.append(experience_id.strip())

        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        count_sql = (
            f"SELECT COUNT(DISTINCT {alias}.{id_column}) "
            f"FROM {table} {alias}{joins}{where}"
        )
        offset = (page - 1) * page_size
        if level == "segment" and parent_id.strip():
            order_by = "s.created_at ASC, s.rowid ASC"
        elif level == "qa" and (parent_id.strip() or experience_id.strip()):
            order_by = "q.timestamp ASC, q.rowid ASC"
        else:
            order_column = "updated_at" if level != "qa" else "timestamp"
            order_by = f"{alias}.{order_column} DESC, {alias}.rowid DESC"
        sql = (
            f"SELECT {select} FROM {table} {alias}{joins}{where}{group} "
            f"ORDER BY {order_by} LIMIT ? OFFSET ?"
        )
        with self._connect() as connection:
            total = int(connection.execute(count_sql, parameters).fetchone()[0])
            rows = connection.execute(sql, (*parameters, page_size, offset)).fetchall()
        serializer = {
            "experience": self._experience,
            "segment": self._segment,
            "qa": self._qa,
        }[level]
        items = []
        for index, row in enumerate(rows, start=offset + 1):
            item = serializer(row)
            item["sequence"] = index
            items.append(item)
        return {
            "items": items,
            "page": page,
            "page_size": page_size,
            "total": total,
            "pages": max(1, (total + page_size - 1) // page_size),
        }

    def detail(self, level: Literal["experience", "segment", "qa"], memory_id: str) -> dict[str, Any]:
        table, id_column = self.TABLES[level]
        self._ensure_database()
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT * FROM {table} WHERE {id_column} = ?", (memory_id,)
            ).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail=f"未找到 {level}: {memory_id}")
            if level == "experience":
                item = self._experience(row)
                segments = []
                segment_rows = connection.execute(
                    "SELECT *, (SELECT COUNT(*) FROM qa_memory q "
                    "WHERE q.segment_id=s.segment_id) qa_count "
                    "FROM segment_memory s WHERE experience_id=? "
                    "ORDER BY created_at ASC, rowid ASC",
                    (memory_id,),
                ).fetchall()
                for segment_index, value in enumerate(segment_rows, start=1):
                    segment = self._segment(value)
                    segment["sequence"] = segment_index
                    qa_rows = connection.execute(
                        "SELECT * FROM qa_memory WHERE segment_id=? "
                        "ORDER BY timestamp ASC, rowid ASC",
                        (segment["segment_id"],),
                    ).fetchall()
                    segment["qas"] = []
                    for qa_index, qa_row in enumerate(qa_rows, start=1):
                        qa = self._qa(qa_row)
                        qa["sequence"] = qa_index
                        segment["qas"].append(qa)
                    segments.append(segment)
                item["segments"] = segments
            elif level == "segment":
                item = self._segment(row)
                item["qas"] = []
                qa_rows = connection.execute(
                    "SELECT * FROM qa_memory WHERE segment_id=? "
                    "ORDER BY timestamp ASC, rowid ASC",
                    (memory_id,),
                ).fetchall()
                for qa_index, value in enumerate(qa_rows, start=1):
                    qa = self._qa(value)
                    qa["sequence"] = qa_index
                    item["qas"].append(qa)
            else:
                item = self._qa(row)
        return item

    def set_status(self, level: Literal["experience", "segment", "qa"], memory_id: str, status: str) -> dict[str, Any]:
        allowed = {
            "experience": {"open", "completed", "deleted"},
            "segment": {"open", "completed", "deleted"},
            "qa": {"open", "deleted"},
        }[level]
        if status not in allowed:
            raise HTTPException(status_code=422, detail=f"不支持的状态: {status}")
        table, id_column = self.TABLES[level]
        with self._lock, self._connect() as connection:
            row = connection.execute(
                f"SELECT * FROM {table} WHERE {id_column}=?", (memory_id,)
            ).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail=f"未找到 {level}: {memory_id}")
            connection.execute(
                f"UPDATE {table} SET status=? WHERE {id_column}=?",
                (status, memory_id),
            )
            connection.commit()
        return self.detail(level, memory_id)

    def chat_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return persisted chat sessions, newest first."""
        self._ensure_database()
        sessions: dict[str, dict[str, Any]] = {}
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT qa_id, timestamp, user_input, assistant_output, tools_json "
                "FROM qa_memory ORDER BY rowid DESC LIMIT 1000"
            ).fetchall()
        for row in rows:
            tools = _json(row["tools_json"], [])
            trace = next(
                (
                    item for item in tools
                    if isinstance(item, dict) and item.get("type") == "hesm_chat_turn"
                ),
                None,
            )
            if trace is None:
                continue
            state_key = str(trace.get("state_key") or "web_chat")
            if state_key not in sessions:
                sessions[state_key] = {
                    "state_key": state_key,
                    "title": str(row["user_input"] or "新会话")[:60],
                    "updated_at": row["timestamp"],
                    "turn_count": 0,
                    "last_answer": str(row["assistant_output"] or "")[:120],
                }
            sessions[state_key]["turn_count"] += 1
        return list(sessions.values())[:limit]

    def chat_history(self, state_key: str, limit: int = 50) -> dict[str, Any]:
        """Restore user/assistant messages from chat turns persisted in QA memory."""
        self._ensure_database()
        matched: list[dict[str, Any]] = []
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT qa_id, timestamp, user_input, assistant_output, tools_json "
                "FROM qa_memory ORDER BY rowid DESC LIMIT 2000"
            ).fetchall()
        for row in rows:
            tools = _json(row["tools_json"], [])
            trace = next(
                (
                    item for item in tools
                    if isinstance(item, dict) and item.get("type") == "hesm_chat_turn"
                ),
                None,
            )
            if trace is None:
                continue
            trace_state_key = str(trace.get("state_key") or "web_chat")
            if trace_state_key != state_key:
                continue
            matched.append({
                "qa_id": row["qa_id"],
                "timestamp": row["timestamp"],
                "user_input": row["user_input"],
                "assistant_output": row["assistant_output"],
                "model": (trace.get("generation") or {}).get("model", ""),
                "timing": (trace.get("generation") or {}).get("elapsed_ms", 0),
            })
            if len(matched) >= limit:
                break
        matched.reverse()
        messages: list[dict[str, str]] = []
        for turn in matched:
            messages.extend([
                {"role": "user", "content": str(turn["user_input"] or "")},
                {"role": "assistant", "content": str(turn["assistant_output"] or "")},
            ])
        return {"state_key": state_key, "turns": matched, "messages": messages}


class MemoryAddRequest(BaseModel):
    user_input: str = Field(min_length=1, max_length=20_000)
    source_id: str | None = Field(default=None, max_length=500)
    assistant_output: str = Field(default="", max_length=20_000)
    context: str = Field(default="", max_length=50_000)
    topic_result: dict[str, Any] | list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] | None = None
    timestamp: str | None = None
    state_key: str = Field(default="default", min_length=1, max_length=200)


class RetrievalRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2_000)


class StatusRequest(BaseModel):
    status: str = Field(min_length=1, max_length=32)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=20_000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)
    state_key: str = Field(default="web_chat", min_length=1, max_length=200)
    session_id: str | None = Field(default=None, min_length=1, max_length=200)


app = FastAPI(title="HESM Memory Console API", version="3.0.0")

# 管理接口会在 HESMService 惰性创建之前访问数据库，因此启动时先补齐完整记忆表结构。
database_initializer = MemoryStorage(DATABASE_PATH)
database_initializer.close()
repository = MemoryRepository(DATABASE_PATH)
session_manager = SessionManager(DATABASE_PATH)
session_manager.import_legacy_chat_turns()
session_manager.remove_chat_traces_from_qa_tools()
_service: HESMService | None = None
_service_lock = Lock()


class SessionCreateRequest(BaseModel):
    title: str = Field(default="新会话", max_length=120)


class SessionRenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=120)


@app.middleware("http")
async def disable_browser_cache(request: Request, call_next: Any) -> Response:
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


def get_service() -> HESMService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = HESMService()
    return _service


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse("/index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> RedirectResponse:
    return RedirectResponse("/favicon.svg?v=1")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ready" if DATABASE_PATH.exists() else "degraded",
        "database": str(DATABASE_PATH),
        "database_exists": DATABASE_PATH.exists(),
        "chroma": str(CHROMA_PATH),
        "chroma_exists": CHROMA_PATH.exists(),
    }


@app.get("/api/stats")
def stats() -> dict[str, Any]:
    return repository.stats()


@app.get("/api/sessions")
def list_sessions(limit: int = Query(default=50, ge=1, le=100)) -> dict[str, Any]:
    return {"items": session_manager.list(limit=limit)}


@app.post("/api/sessions", status_code=201)
def create_session(request: SessionCreateRequest) -> dict[str, Any]:
    return session_manager.create(title=request.title)


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    try:
        return session_manager.get(session_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.patch("/api/sessions/{session_id}")
def rename_session(session_id: str, request: SessionRenameRequest) -> dict[str, Any]:
    try:
        return session_manager.rename(session_id, request.title)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.delete("/api/sessions/{session_id}")
def archive_session(session_id: str) -> dict[str, Any]:
    try:
        return session_manager.archive(session_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


# Backward-compatible aliases keep already-open v4 pages from producing 404s.
@app.get("/api/chat/sessions", deprecated=True)
def legacy_chat_sessions(limit: int = Query(default=50, ge=1, le=100)) -> dict[str, Any]:
    items = session_manager.list(limit=limit)
    return {"items": [
        {**item, "state_key": item["session_id"], "last_answer": item["preview"]}
        for item in items
    ]}


@app.get("/api/chat/history", deprecated=True)
def legacy_chat_history(
    state_key: str = Query(default="web_chat", min_length=1, max_length=200),
    limit: int = Query(default=50, ge=1, le=100),
) -> dict[str, Any]:
    session = session_manager.get(state_key, required=False)
    if session is None:
        return {"state_key": state_key, "messages": [], "turns": []}
    messages = session["messages"][-limit * 2:]
    return {"state_key": state_key, "messages": messages, "turns": []}


@app.get("/api/{level}")
def list_memories(
    level: Literal["experience", "segment", "qa"],
    q: str = Query(default="", max_length=500),
    status: str = Query(default="", max_length=32),
    parent_id: str = Query(default="", max_length=200),
    experience_id: str = Query(default="", max_length=200),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=30, ge=1, le=100),
) -> dict[str, Any]:
    return repository.list_memories(
        level, search=q, status=status, parent_id=parent_id,
        experience_id=experience_id,
        page=page, page_size=page_size,
    )


@app.get("/api/{level}/{memory_id}")
def memory_detail(
    level: Literal["experience", "segment", "qa"], memory_id: str
) -> dict[str, Any]:
    return repository.detail(level, memory_id)


@app.patch("/api/{level}/{memory_id}/status")
def update_status(
    level: Literal["experience", "segment", "qa"],
    memory_id: str,
    request: StatusRequest,
) -> dict[str, Any]:
    return repository.set_status(level, memory_id, request.status)


@app.post("/api/memories")
def add_memory(request: MemoryAddRequest) -> dict[str, Any]:
    try:
        return get_service().add_memory(**request.model_dump())
    except Exception as error:
        LOGGER.exception("HESM memory ingestion failed")
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/api/retrieve")
def retrieve(request: RetrievalRequest) -> dict[str, Any]:
    try:
        return get_service().retrieve(**request.model_dump())
    except Exception as error:
        LOGGER.exception("HESM retrieval failed")
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    try:
        payload = request.model_dump()
        payload["history"] = [item for item in payload["history"]]
        return get_service().chat(**payload)
    except Exception as error:
        LOGGER.exception("HESM chat failed")
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/api/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    def stream() -> Any:
        try:
            payload = request.model_dump()
            payload["history"] = [item for item in payload["history"]]
            for event in get_service().chat_events(**payload):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception as error:
            LOGGER.exception("HESM streaming chat failed")
            yield json.dumps(
                {"event": "error", "message": str(error)},
                ensure_ascii=False,
            ) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.on_event("shutdown")
def close_service() -> None:
    global _service
    if _service is not None:
        _service.close()
        _service = None


app.mount("/", NoCacheStaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("service.server:app", host="0.0.0.0", port=8080, reload=False)
