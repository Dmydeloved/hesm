from __future__ import annotations

import re
from typing import Any


TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+")


def tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text or "")]


def clipped_overlap(candidate: list[str], reference: list[str]) -> int:
    counts: dict[str, int] = {}
    for token in reference:
        counts[token] = counts.get(token, 0) + 1
    overlap = 0
    for token in candidate:
        count = counts.get(token, 0)
        if count <= 0:
            continue
        overlap += 1
        counts[token] = count - 1
    return overlap


def token_f1(candidate_text: str, reference_text: str) -> float:
    candidate = tokens(candidate_text)
    reference = tokens(reference_text)
    if not candidate and not reference:
        return 1.0
    if not candidate or not reference:
        return 0.0
    overlap = clipped_overlap(candidate, reference)
    if overlap == 0:
        return 0.0
    precision = overlap / len(candidate)
    recall = overlap / len(reference)
    return 2 * precision * recall / (precision + recall)


def evidence_recall(retrieved_items: list[dict[str, Any]], evidence: list[str]) -> float:
    gold = {" ".join(tokens(item)) for item in evidence if tokens(item)}
    if not gold:
        return 1.0
    retrieved = {
        " ".join(tokens(str(item.get("user_input") or "")))
        for item in retrieved_items
        if tokens(str(item.get("user_input") or ""))
    }
    return len(gold & retrieved) / len(gold)


def reciprocal_rank(retrieved_items: list[dict[str, Any]], evidence: list[str]) -> float:
    gold = {" ".join(tokens(item)) for item in evidence if tokens(item)}
    if not gold:
        return 0.0
    for index, item in enumerate(retrieved_items, 1):
        normalized = " ".join(tokens(str(item.get("user_input") or "")))
        if normalized in gold:
            return 1.0 / index
    return 0.0


def approx_tokens(text: str) -> int:
    return max(0, len(tokens(text)))


def row_metrics(
    answer: str,
    reference_answer: str,
    retrieved_items: list[dict[str, Any]],
    evidence_user_inputs: list[str],
    context_text: str,
) -> dict[str, float]:
    return {
        "answer_f1": token_f1(answer, reference_answer),
        "evidence_recall": evidence_recall(retrieved_items, evidence_user_inputs),
        "mrr": reciprocal_rank(retrieved_items, evidence_user_inputs),
        "retrieved_tokens": float(approx_tokens(context_text)),
    }

