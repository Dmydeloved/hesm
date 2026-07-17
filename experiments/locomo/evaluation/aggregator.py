"""
Aggregates per-question metrics into method-level statistics.

Input:  list of QARecord dicts (one per answered question)
Output: MethodMetrics dataclass with all averaged scores

QARecord schema (from runner/checkpoint.py):
{
  "question": str,
  "ground_truth": str,
  "prediction": str,
  "retrieved_ids": list[str],
  "retrieved_context": str,
  "retrieved_tokens": int,
  "evidence": list[str],
  "category": int,
  "f1": float,
  "f1_precision": float,
  "f1_recall": float,
  "judge_score": int,              # 0/1/2 or -1 if failed
  "retrieval_metrics": {           # keyed by str(K)
    "1": {"recall": ..., "precision": ..., "f1": ..., "accuracy": ...},
    "3": {...},
    "5": {...},
  },
  "total_conversation_tokens": int,
  "compression_ratio": float,
}
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class KMetrics:
    """Averaged retrieval metrics at a specific K."""
    k: int
    recall: float = 0.0
    precision: float = 0.0
    f1: float = 0.0
    accuracy: float = 0.0


@dataclass
class MethodMetrics:
    """All aggregated metrics for one method on one benchmark run."""

    method_name: str
    num_questions: int = 0

    # Answer quality
    avg_f1: float = 0.0
    avg_precision: float = 0.0
    avg_recall: float = 0.0
    avg_judge_score: float = 0.0

    # Retrieval quality @ K
    retrieval: dict[int, KMetrics] = field(default_factory=dict)

    # Context efficiency
    avg_retrieved_tokens: float = 0.0
    avg_compression_ratio: float = 0.0

    # Per-category breakdown (category 1/2/3)
    category_f1: dict[int, float] = field(default_factory=dict)
    category_judge: dict[int, float] = field(default_factory=dict)
    category_count: dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Flatten retrieval for JSON serialisation
        flat_retrieval: dict[str, Any] = {}
        for k, km in self.retrieval.items():
            flat_retrieval[f"recall@{k}"] = km.recall
            flat_retrieval[f"precision@{k}"] = km.precision
            flat_retrieval[f"f1@{k}"] = km.f1
            flat_retrieval[f"accuracy@{k}"] = km.accuracy
        d["retrieval_flat"] = flat_retrieval
        return d


def aggregate(
    method_name: str,
    qa_records: list[dict[str, Any]],
    k_values: list[int] | None = None,
) -> MethodMetrics:
    """
    Compute macro-averaged metrics from a list of per-question QA records.

    Skips judge_score == -1 entries when averaging the judge score.
    """
    if k_values is None:
        k_values = [1, 3, 5]

    m = MethodMetrics(method_name=method_name, num_questions=len(qa_records))
    if not qa_records:
        return m

    # ── Answer quality ─────────────────────────────────────────
    f1_scores = [r["f1"] for r in qa_records]
    precision_scores = [r["f1_precision"] for r in qa_records]
    recall_scores = [r["f1_recall"] for r in qa_records]
    judge_scores = [r["judge_score"] for r in qa_records if r.get("judge_score", -1) >= 0]

    m.avg_f1 = _mean(f1_scores)
    m.avg_precision = _mean(precision_scores)
    m.avg_recall = _mean(recall_scores)
    m.avg_judge_score = _mean(judge_scores) if judge_scores else 0.0

    # ── Retrieval quality ──────────────────────────────────────
    for k in k_values:
        sk = str(k)
        recalls, precisions, f1s, accuracies = [], [], [], []
        for r in qa_records:
            rm = r.get("retrieval_metrics", {})
            if sk in rm:
                recalls.append(rm[sk]["recall"])
                precisions.append(rm[sk]["precision"])
                f1s.append(rm[sk]["f1"])
                accuracies.append(rm[sk]["accuracy"])
        m.retrieval[k] = KMetrics(
            k=k,
            recall=_mean(recalls),
            precision=_mean(precisions),
            f1=_mean(f1s),
            accuracy=_mean(accuracies),
        )

    # ── Context efficiency ─────────────────────────────────────
    token_counts = [r.get("retrieved_tokens", 0) for r in qa_records]
    compression_ratios = [r.get("compression_ratio", 1.0) for r in qa_records]
    m.avg_retrieved_tokens = _mean(token_counts)
    m.avg_compression_ratio = _mean(compression_ratios)

    # ── Per-category breakdown ─────────────────────────────────
    from collections import defaultdict
    cat_f1: dict[int, list[float]] = defaultdict(list)
    cat_judge: dict[int, list[float]] = defaultdict(list)
    cat_count: dict[int, int] = defaultdict(int)

    for r in qa_records:
        cat = r.get("category", 0)
        cat_f1[cat].append(r["f1"])
        cat_count[cat] += 1
        js = r.get("judge_score", -1)
        if js >= 0:
            cat_judge[cat].append(float(js))

    m.category_f1 = {cat: _mean(vals) for cat, vals in cat_f1.items()}
    m.category_judge = {cat: _mean(vals) for cat, vals in cat_judge.items()}
    m.category_count = dict(cat_count)

    return m


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
