from __future__ import annotations

from threading import RLock
from types import SimpleNamespace

from core.session import SessionManager
from core.storage import MemoryStorage
from core.trajectory import TraceRepository
from service.hesm_service import HESMService


def test_chat_turn_records_all_stages_and_can_be_reopened(tmp_path):
    database = tmp_path / "memory.sqlite3"
    storage = MemoryStorage(database)
    service = HESMService.__new__(HESMService)
    service._lock = RLock()
    service.sessions = SessionManager(database)
    service.traces = TraceRepository(database)
    service.extractor = SimpleNamespace(extract=lambda **_: {
        "topic": "旅行", "core_entity": "上海", "intent": "咨询",
        "entities": ["上海"], "confidence": 0.9,
    })
    service.retriever = SimpleNamespace(retriever=lambda **_: {
        "context": "上海旅行记忆", "experiences": [], "segments": [], "qas": [],
    })
    service.answerer = SimpleNamespace(answer=lambda prompt: "建议去外滩")
    service.chat_model = "test-model"
    service.manager = SimpleNamespace(trace_turn_ids=[])
    observed = []

    def add_memory(**kwargs):
        observed.append((kwargs, list(service.manager.trace_turn_ids)))
        return {"memories": [{
            "qa_id": "qa-1", "segment_id": "seg-1", "experience_id": "exp-1",
        }]}

    service.add_memory = add_memory
    result = service.chat(message="上海怎么玩？", session_id="session-1")
    trace = service.traces.get("session-1", result["turn_id"])

    assert trace is not None
    assert trace["status"] == "completed"
    assert trace["user_input"] == "上海怎么玩？"
    assert [event["stage"] for event in trace["events"] if event["status"] == "stage_completed"] == [
        "extract", "retrieve", "prompt", "generate", "store",
    ]
    assert trace["events"][-1]["payload"]["result"]["answer"] == "建议去外滩"
    assert observed[0][1] == [result["turn_id"]]
    assert service.manager.trace_turn_ids == []
    assert all(
        message["metadata"]["turn_id"] == result["turn_id"]
        for message in service.sessions.get("session-1")["messages"]
    )
    assert service.traces.get("another-session", result["turn_id"]) is None
    storage.close()
