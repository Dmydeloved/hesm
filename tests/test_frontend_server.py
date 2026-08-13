from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from frontend.server import MemoryRepository
from hesm.storage import MemoryStorage


def seed_database(path: Path) -> None:
    storage = MemoryStorage(path)
    connection = storage.connection
    connection.execute(
        "INSERT INTO experience_memory VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "exp_1", "旅行规划", "Alice", json.dumps(["行程询问"]),
            json.dumps(["seg_1"]), json.dumps("Alice 的旅行记忆"),
            json.dumps({"status": "in_progress", "current_segment_id": "seg_1"}),
            "2026-01-01", "2026-01-02", 1, 1,
        ),
    )
    connection.execute(
        "INSERT INTO segment_memory VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "seg_1", "旅行规划", "行程询问", "Alice", json.dumps(["qa_1"]),
            "open", "巴黎行程", "exp_1", "2026-01-01", "2026-01-02", 1, 1,
        ),
    )
    connection.execute(
        "INSERT INTO qa_memory VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "qa_1", "2026-01-02", "Alice 去了哪里？", "巴黎", "[]",
            "旅行规划", "行程询问", "Alice", '["Alice", "巴黎"]',
            "seg_1", "active", 0.96, "明确询问目的地",
        ),
    )
    connection.commit()
    storage.close()


def test_repository_lists_and_hydrates_all_memory_levels(tmp_path: Path) -> None:
    database = tmp_path / "memory.sqlite3"
    seed_database(database)
    repository = MemoryRepository(database)

    assert repository.stats()["counts"] == {
        "experience": 1,
        "segment": 1,
        "qa": 1,
    }
    experience = repository.list_memories("experience")["items"][0]
    assert experience["segment_count"] == 1
    assert experience["qa_count"] == 1
    assert experience["state"]["status"] == "in_progress"

    segment = repository.detail("segment", "seg_1")
    assert segment["qas"][0]["assistant_output"] == "巴黎"


def test_repository_filters_children_and_updates_soft_status(tmp_path: Path) -> None:
    database = tmp_path / "memory.sqlite3"
    seed_database(database)
    repository = MemoryRepository(database)

    result = repository.list_memories("qa", search="巴黎", parent_id="seg_1")
    assert result["total"] == 1
    updated = repository.set_status("qa", "qa_1", "archived")
    assert updated["status"] == "archived"
    assert repository.list_memories("qa", status="active")["total"] == 0


def test_repository_restores_persisted_chat_sessions(tmp_path: Path) -> None:
    database = tmp_path / "memory.sqlite3"
    seed_database(database)
    trace = {
        "type": "hesm_chat_turn",
        "state_key": "web_chat_saved",
        "generation": {"model": "test-model", "elapsed_ms": 12.5},
    }
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE qa_memory SET tools_json=? WHERE qa_id='qa_1'",
            (json.dumps([trace], ensure_ascii=False),),
        )
        connection.commit()

    repository = MemoryRepository(database)
    sessions = repository.chat_sessions()
    assert sessions[0]["state_key"] == "web_chat_saved"
    assert sessions[0]["turn_count"] == 1

    history = repository.chat_history("web_chat_saved")
    assert history["messages"] == [
        {"role": "user", "content": "Alice 去了哪里？"},
        {"role": "assistant", "content": "巴黎"},
    ]
    assert history["turns"][0]["model"] == "test-model"
