from __future__ import annotations

from threading import RLock
from typing import Any

from hesm.service import HESMService


class FakeExtractor:
    def extract(self, **_: Any) -> dict[str, Any]:
        return {
            "topic": "travel",
            "core_entity": "Alice",
            "intent": "query",
            "entities": ["Alice"],
            "confidence": 0.9,
        }


class FakeManager:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def add_qa(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {
            "qa_id": "q1",
            "segment_id": "s1",
            "experience_id": "e1",
            "action": "new_experience",
        }


class FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def recall(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {
            "experiences": [],
            "segments": [],
            "qas": [],
            "candidate_tree": [],
            "context_text": "memory context",
            "debug": {},
        }


def fake_service() -> HESMService:
    service = object.__new__(HESMService)
    service._lock = RLock()
    service.extractor = FakeExtractor()
    service.manager = FakeManager()
    service.retriever = FakeRetriever()
    service.api_config = {
        "top_experience": 2,
        "top_segment": 3,
        "top_qa": 8,
    }
    return service


def test_add_memory_is_a_public_ingestion_interface() -> None:
    service = fake_service()

    result = service.add_memory(
        user_input="Alice went to Paris.",
        assistant_output="Noted.",
        state_key="alice",
    )

    assert result["state_key"] == "alice"
    assert result["memories"][0]["qa_id"] == "q1"
    assert service.manager.calls[0]["topic_result"]["topic"] == "travel"


def test_retrieve_is_a_public_query_interface() -> None:
    service = fake_service()

    result = service.retrieve(question="Where did Alice go?", top_qa=4)

    assert result["context_text"] == "memory context"
    assert result["limits"] == {
        "top_experience": 2,
        "top_segment": 3,
        "top_qa": 4,
    }
    assert service.retriever.calls[0]["query_confidence"] == 0.9
