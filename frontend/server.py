"""HTTP and static-file server for the production HESM service."""

from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from hesm.config import config_path
from hesm.service import HESMService


LOGGER = logging.getLogger("hesm.frontend")
FRONTEND_DIR = Path(__file__).resolve().parent
DATABASE_PATH = config_path("paths", "memory_db")
CHROMA_PATH = config_path("paths", "chroma")


class MemoryAddRequest(BaseModel):
    user_input: str = Field(min_length=1, max_length=20_000)
    assistant_output: str = Field(default="", max_length=20_000)
    context: str = Field(default="", max_length=50_000)
    topic_result: dict[str, Any] | list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] | None = None
    timestamp: str | None = None
    state_key: str = Field(default="default", min_length=1, max_length=200)


class RetrievalRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2_000)
    top_experience: int | None = Field(default=None, ge=1, le=10)
    top_segment: int | None = Field(default=None, ge=1, le=20)
    top_qa: int | None = Field(default=None, ge=1, le=50)


app = FastAPI(title="HESM Memory API", version="2.0.0")
_service: HESMService | None = None
_service_lock = Lock()


def get_service() -> HESMService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = HESMService()
    return _service


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ready",
        "database": str(DATABASE_PATH),
        "database_exists": DATABASE_PATH.exists(),
        "chroma": str(CHROMA_PATH),
        "chroma_exists": CHROMA_PATH.exists(),
    }


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


@app.on_event("shutdown")
def close_service() -> None:
    global _service
    if _service is not None:
        _service.close()
        _service = None


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("frontend.server:app", host="127.0.0.1", port=8080, reload=False)
