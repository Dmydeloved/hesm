from __future__ import annotations

import json
from pathlib import Path

from hesm.session import SessionManager
from hesm.storage import MemoryStorage


def test_session_manager_crud_does_not_touch_memory_tables(tmp_path: Path) -> None:
    database = tmp_path / "memory.sqlite3"
    storage = MemoryStorage(database)
    storage.close()
    manager = SessionManager(database)

    created = manager.create(title="投资讨论")
    assert created["messages"] == []
    session_id = created["session_id"]

    updated = manager.append_turn(
        session_id,
        user_content="比较两家公司",
        assistant_content="可以从盈利质量开始。",
        metadata={"model": "test-model"},
    )
    assert updated["turn_count"] == 1
    assert updated["messages"][0]["role"] == "user"
    assert manager.list()[0]["preview"] == "可以从盈利质量开始。"

    renamed = manager.rename(session_id, "公司比较")
    assert renamed["title"] == "公司比较"
    manager.archive(session_id)
    assert manager.list() == []

    storage = MemoryStorage(database, read_only=True)
    assert storage.count_rows("experience_memory") == 0
    assert storage.count_rows("segment_memory") == 0
    assert storage.count_rows("qa_memory") == 0
    storage.close()


def test_session_manager_imports_legacy_chat_once(tmp_path: Path) -> None:
    database = tmp_path / "memory.sqlite3"
    storage = MemoryStorage(database)
    connection = storage.connection
    connection.execute(
        "INSERT INTO experience_memory VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        ("e1", "主题", "用户", "[]", '["s1"]', '""', '{}', "1", "1", 1, 0),
    )
    connection.execute(
        "INSERT INTO segment_memory VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        ("s1", "主题", "提问", "用户", '["q1"]', "open", "", "e1", "1", "1", 1, 0),
    )
    trace = {"type": "hesm_chat_turn", "state_key": "legacy-chat"}
    connection.execute(
        "INSERT INTO qa_memory VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("q1", "1", "旧问题", "旧回答", json.dumps([trace]), "主题", "提问", "用户", "[]", "s1", "active", 1.0, "测试"),
    )
    connection.commit()
    storage.close()

    manager = SessionManager(database)
    assert manager.import_legacy_chat_turns() == 1
    assert manager.import_legacy_chat_turns() == 0
    assert manager.get("legacy-chat")["messages"][1]["content"] == "旧回答"


def test_chat_trace_cleanup_preserves_real_tool_calls(tmp_path: Path) -> None:
    database = tmp_path / "memory.sqlite3"
    storage = MemoryStorage(database)
    connection = storage.connection
    connection.execute(
        "INSERT INTO experience_memory VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        ("e1", "主题", "用户", "[]", '["s1"]', '""', '{}', "1", "1", 1, 0),
    )
    connection.execute(
        "INSERT INTO segment_memory VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        ("s1", "主题", "提问", "用户", '["q1"]', "open", "", "e1", "1", "1", 1, 0),
    )
    tools = [
        {"type": "hesm_chat_turn", "prompt": "obsolete diagnostics"},
        {"name": "web_search", "arguments": {"query": "test"}},
    ]
    connection.execute(
        "INSERT INTO qa_memory VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("q1", "1", "问题", "回答", json.dumps(tools), "主题", "提问", "用户", "[]", "s1", "active", 1.0, "测试"),
    )
    connection.commit()
    storage.close()

    manager = SessionManager(database)
    assert manager.remove_chat_traces_from_qa_tools() == 1
    storage = MemoryStorage(database, read_only=True)
    assert storage.get_qa("q1")["tools"] == [tools[1]]
    storage.close()
