from __future__ import annotations

from threading import RLock
from typing import Any

from hesm.chat import build_chat_prompt
from hesm.service import HESMService


class FakeExtractor:
    def extract(self, **_: Any) -> dict[str, Any]:
        return {
            "topic": "旅行规划",
            "core_entity": "Alice",
            "intent": "地点询问",
            "entities": ["Alice", "巴黎"],
            "confidence": 0.95,
            "reasoning": "用户询问 Alice 的旅行地点",
        }


class FakeRetriever:
    def recall(self, **_: Any) -> dict[str, Any]:
        return {
            "experiences": [{"experience_id": "exp_old", "topic": "旅行规划"}],
            "segments": [],
            "qas": [{"qa_id": "qa_old", "user_input": "Alice 去了巴黎"}],
            "context_text": "Alice 曾经去过巴黎。",
            "debug": {},
        }


class FakeAnswerer:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def answer(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "Alice 去了巴黎。"


class FakeManager:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def add_qa(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {
            "qa_id": "qa_new",
            "segment_id": "seg_new",
            "experience_id": "exp_new",
            "action": "new_experience",
        }


class FakeSessions:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []

    def ensure(self, session_id: str) -> dict[str, Any]:
        return {"session_id": session_id, "messages": list(self.messages)}

    def append_turn(self, session_id: str, *, user_content: str, assistant_content: str, metadata: dict[str, Any]) -> dict[str, Any]:
        self.messages.extend([
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ])
        return {"session_id": session_id, "messages": list(self.messages)}


def fake_chat_service() -> HESMService:
    service = object.__new__(HESMService)
    service._lock = RLock()
    service.extractor = FakeExtractor()
    service.retriever = FakeRetriever()
    service.answerer = FakeAnswerer()
    service.manager = FakeManager()
    service.sessions = FakeSessions()
    service.chat_model = "fake-chat-model"
    service.api_config = {"top_experience": 2, "top_segment": 3, "top_qa": 8}
    return service


def test_chat_prompt_contains_memory_history_and_current_question() -> None:
    prompt = build_chat_prompt(
        question="Alice 去了哪里？",
        extraction={"topic": "旅行规划"},
        memory_context="Alice 曾经去过巴黎。",
        history=[{"role": "user", "content": "我们聊聊 Alice"}],
    )
    assert "Alice 曾经去过巴黎" in prompt
    assert "我们聊聊 Alice" in prompt
    assert "Alice 去了哪里" in prompt


def test_chat_retrieves_answers_and_keeps_tools_empty_without_tool_calls() -> None:
    service = fake_chat_service()

    result = service.chat(
        message="Alice 去了哪里？",
        history=[{"role": "user", "content": "请回忆旅行"}],
        state_key="web_chat",
    )

    assert result["answer"] == "Alice 去了巴黎。"
    assert result["stored"]["memories"][0]["qa_id"] == "qa_new"
    assert result["session_id"] == "web_chat"
    assert result["timing"]["generation_ms"] >= 0
    stored_call = service.manager.calls[0]
    assert stored_call["assistant_output"] == "Alice 去了巴黎。"
    assert stored_call["topic_result"]["topic"] == "旅行规划"
    assert stored_call["tools"] == []
    assert result["history"][0]["content"] == "请回忆旅行"


def test_chat_does_not_persist_when_answer_generation_fails() -> None:
    service = fake_chat_service()

    class BrokenAnswerer:
        def answer(self, _: str) -> str:
            raise RuntimeError("model unavailable")

    service.answerer = BrokenAnswerer()
    try:
        service.chat(message="Alice 去了哪里？")
    except RuntimeError as error:
        assert "model unavailable" in str(error)
    else:
        raise AssertionError("chat should propagate answer generation failure")

    assert service.manager.calls == []
