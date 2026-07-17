from __future__ import annotations

import math
from typing import Any

from memory.embedder import tokenize


def memory_text(example: Any) -> str:
    topic = example.topic_result
    return "\n".join(
        [
            f"topic: {topic.get('topic', '')}",
            f"core_entity: {topic.get('core_entity', '')}",
            f"intent: {topic.get('intent', '')}",
            f"user: {example.user_input}",
            f"assistant: {example.assistant_output}",
        ]
    )


def cosine(left: list[float], right: list[float]) -> float:
    dot = sum(l_value * r_value for l_value, r_value in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


def bm25_score(query: str, document: str) -> float:
    query_tokens = tokenize(query)
    doc_tokens = tokenize(document)
    if not query_tokens or not doc_tokens:
        return 0.0
    doc_counts: dict[str, int] = {}
    for token in doc_tokens:
        doc_counts[token] = doc_counts.get(token, 0) + 1
    score = 0.0
    for token in query_tokens:
        freq = doc_counts.get(token, 0)
        if freq:
            score += (freq * 2.2) / (freq + 1.2)
    return score

