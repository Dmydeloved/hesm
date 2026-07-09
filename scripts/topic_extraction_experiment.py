#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory.extractor import (
    TopicResult,
    TopicExtractor,
    add_result_metadata,
)
from scripts.sample_topics import get_increase_topic, get_shift_topic, get_single_topic


class FixedArray:
    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self.array: list[object] = []

    def add(self, item: object) -> None:
        self.array.append(item)
        if len(self.array) > self.max_size:
            self.array.pop(0)

    def to_json_string(self) -> str:
        return json.dumps(self.array, ensure_ascii=False, indent=2)

    def to_list(self) -> list[object]:
        return self.array.copy()


def context_to_json(context: dict[str, object]) -> str:
    data = {
        "current_topic_state": context["current_topic_state"],
        "recent_semantic_history": context["recent_semantic_history"].to_list(),
    }
    return json.dumps(data, ensure_ascii=False, indent=2)


def output_path(name: str) -> Path:
    path = Path(__file__).resolve().parents[1] / "results" / "topic_extraction" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def print_and_save(name: str, payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    output_path(name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_single_rounds(extractor: TopicExtractor) -> None:
    result = {
        "single_topic_data": [],
        "increase_topic_data": [],
        "shift_topic_data": [],
    }
    for group_name, messages in (
        ("single_topic_data", get_single_topic()),
        ("increase_topic_data", get_increase_topic()),
        ("shift_topic_data", get_shift_topic()),
    ):
        for index, message in messages.items():
            extracted = extractor.extract(user_input=message)
            result[group_name].append(extracted)
            print(f"{group_name}:{index}")
            print(json.dumps(extracted, ensure_ascii=False, indent=2))
    print_and_save("first_result.json", result)


def run_history_rounds(extractor: TopicExtractor) -> None:
    result = {
        "single_topic_data": [],
        "increase_topic_data": [],
        "shift_topic_data": [],
    }
    history = FixedArray(5)
    for group_name, messages in (
        ("single_topic_data", get_single_topic()),
        ("increase_topic_data", get_increase_topic()),
        ("shift_topic_data", get_shift_topic()),
    ):
        for index, message in messages.items():
            extracted = extractor.extract(
                user_input=message,
                context=history.to_json_string(),
            )
            history.add(message)
            result[group_name].append(extracted)
            print(f"{group_name}:{index}")
            print(json.dumps(extracted, ensure_ascii=False, indent=2))
    print_and_save("second_result.json", result)


def run_semantic_state_rounds(extractor: TopicExtractor) -> None:
    result = {
        "single_topic_data": [],
        "increase_topic_data": [],
        "shift_topic_data": [],
    }
    conversation_context = {
        "current_topic_state": {},
        "recent_semantic_history": FixedArray(5),
    }
    for group_name, messages in (
        ("single_topic_data", get_single_topic()),
        ("increase_topic_data", get_increase_topic()),
        ("shift_topic_data", get_shift_topic()),
    ):
        for index, message in messages.items():
            extracted = extractor.extract(
                user_input=message,
                context=context_to_json(conversation_context),
            )
            annotated = add_result_metadata(extracted, message)
            result[group_name].append(annotated)
            conversation_context["recent_semantic_history"].add(annotated)
            conversation_context["current_topic_state"] = annotated
            print(f"{group_name}:{index}")
            print(json.dumps(annotated, ensure_ascii=False, indent=2))
    print_and_save("third_result.json", result)


def main() -> int:
    extractor = TopicExtractor()
    run_single_rounds(extractor)
    run_history_rounds(extractor)
    run_semantic_state_rounds(extractor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
