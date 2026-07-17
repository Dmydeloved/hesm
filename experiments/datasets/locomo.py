from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import ExperimentExample


def _topic_from_item(item: dict[str, Any]) -> dict[str, Any]:
    topic = item.get("topic_result")
    if isinstance(topic, dict):
        return topic
    return {
        "topic": str(item.get("topic") or item.get("category") or "locomo"),
        "core_entity": str(item.get("core_entity") or item.get("speaker") or "conversation"),
        "intent": str(item.get("intent") or item.get("question_type") or "memory_qa"),
        "entities": item.get("entities") or [],
        "confidence": float(item.get("confidence", 1.0)),
        "reasoning": "provided by dataset adapter",
    }


def load_locomo(path: Path, limit: int = 0) -> list[ExperimentExample]:
    return load_generic_memory_jsonl(path, dataset_name="locomo", limit=limit)


def load_generic_memory_jsonl(
    path: Path, dataset_name: str, limit: int = 0
) -> list[ExperimentExample]:
    examples: list[ExperimentExample] = []
    with path.open(encoding="utf-8") as source:
        for index, line in enumerate(source):
            if not line.strip():
                continue
            item = json.loads(line)
            dialogue_id = str(item.get("dialogue_id") or item.get("conversation_id") or "default")
            examples.append(
                ExperimentExample(
                    case_id=str(item.get("case_id") or item.get("question_id") or f"{dataset_name}-{index:06d}"),
                    dialogue_id=f"{dataset_name}-{dialogue_id}",
                    turn_index=int(item.get("turn_index") or index),
                    user_input=str(item.get("question") or item.get("user_input") or ""),
                    assistant_output=str(item.get("answer") or item.get("assistant_output") or ""),
                    topic_result=_topic_from_item(item),
                    history=item.get("history") if isinstance(item.get("history"), list) else [],
                    reference_answer=str(item.get("answer") or item.get("reference_answer") or ""),
                    evidence_user_inputs=[
                        str(value)
                        for value in item.get("evidence_user_inputs", [])
                        if str(value).strip()
                    ],
                    question_type=str(item.get("question_type") or "unknown"),
                    metadata={key: value for key, value in item.items() if key != "history"},
                )
            )
            if limit and len(examples) >= limit:
                break
    if not examples:
        raise ValueError(f"No {dataset_name} examples loaded from {path}")
    return examples

