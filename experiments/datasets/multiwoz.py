from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import ExperimentExample


def topic_items(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def next_system_output(turns: list[dict[str, Any]], turn_index: int) -> str:
    for turn in turns[turn_index + 1 :]:
        if turn.get("role") == "system":
            return str(turn.get("content") or "")
        if turn.get("role") == "user":
            break
    return ""


def same_experience(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        str(left.get("topic") or "") == str(right.get("topic") or "")
        and str(left.get("core_entity") or "") == str(right.get("core_entity") or "")
    )


def case_type(previous: dict[str, Any] | None, current: dict[str, Any]) -> str:
    if not current:
        return "unknown"
    if previous is None:
        return "cold_start"
    if same_experience(previous, current) and str(previous.get("intent") or "") == str(
        current.get("intent") or ""
    ):
        return "same_experience_same_intent"
    if same_experience(previous, current):
        return "same_experience_new_intent"
    return "topic_switch"


def load_multiwoz(path: Path, limit: int = 0) -> list[ExperimentExample]:
    examples: list[ExperimentExample] = []
    with path.open(encoding="utf-8") as source:
        for dialogue_index, line in enumerate(source):
            if not line.strip():
                continue
            dialogue = json.loads(line)
            turns = dialogue.get("dialogue") or []
            history: list[dict[str, Any]] = []
            previous_topic: dict[str, Any] | None = None
            user_turn_index = 0
            for turn_index, turn in enumerate(turns):
                role = turn.get("role")
                if role != "user":
                    history.append(turn)
                    continue
                items = topic_items(turn.get("topic_extraction"))
                current = items[0] if items else {}
                evidence = [
                    str(item.get("content") or "")
                    for item in history
                    if item.get("role") == "user"
                    and current
                    and same_experience(
                        (topic_items(item.get("topic_extraction")) or [{}])[0],
                        current,
                    )
                ]
                answer = next_system_output(turns, turn_index)
                examples.append(
                    ExperimentExample(
                        case_id=f"multiwoz-d{dialogue_index:05d}-u{user_turn_index:03d}",
                        dialogue_id=f"multiwoz-d{dialogue_index:05d}",
                        turn_index=turn_index,
                        user_input=str(turn.get("content") or ""),
                        assistant_output=answer,
                        topic_result=current,
                        history=list(history),
                        reference_answer=answer,
                        evidence_user_inputs=evidence,
                        question_type=case_type(previous_topic, current),
                        metadata={
                            "scene": dialogue.get("scene") or [],
                            "all_topic_results": items,
                            "has_precomputed_topic": bool(items),
                            "user_turn_index": user_turn_index,
                        },
                    )
                )
                if current:
                    previous_topic = current
                user_turn_index += 1
                history.append(turn)
                if limit and len(examples) >= limit:
                    return examples
    if not examples:
        raise ValueError(f"No MultiWOZ examples loaded from {path}")
    return examples
