from __future__ import annotations

import json
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
    service.storage = storage
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


def test_topic_extraction_history_is_independent_and_limited_to_five_turns(tmp_path):
    storage = MemoryStorage(tmp_path / "memory.sqlite3")
    service = HESMService.__new__(HESMService)
    service._lock = RLock()
    service.storage = storage
    contexts = []

    def extract(**kwargs):
        contexts.append(kwargs["context"])
        return {
            "topic": "旅行",
            "core_entity": "上海",
            "intent": "规划",
            "entities": ["上海"],
            "confidence": 0.9,
        }

    counter = {"value": 0}

    def add_qa(**kwargs):
        counter["value"] += 1
        suffix = counter["value"]
        return {
            "qa_id": f"qa-{suffix}",
            "segment_id": "seg-1",
            "experience_id": "exp-1",
            "action": "append_segment",
        }

    service.extractor = SimpleNamespace(extract=extract)
    service.manager = SimpleNamespace(add_qa=add_qa)
    service.derivation_worker = None

    for index in range(1, 7):
        service.add_memory(
            user_input=f"问题{index}",
            assistant_output=f"回答{index}",
            state_key="session-1",
            timestamp=f"2026-01-0{index} 00:00:00",
        )

    recent = storage.list_recent_topic_extractions("session-1", limit=5)
    assert [item["user_input"] for item in recent] == [
        "问题2", "问题3", "问题4", "问题5", "问题6",
    ]
    sixth_context = json.loads(contexts[-1])
    assert [item["user_input"] for item in sixth_context["recent_topic_history"]] == [
        "问题1", "问题2", "问题3", "问题4", "问题5",
    ]
    assert storage.count_rows("topic_extraction_history") == 6
    assert storage.count_rows("chat_session") == 0
    storage.close()
