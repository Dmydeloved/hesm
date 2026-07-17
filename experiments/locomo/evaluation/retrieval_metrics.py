"""
Retrieval quality metrics for the LoCoMo benchmark.

Computes Evidence Recall@K, Evidence Precision@K, Evidence F1@K and
Retrieval Accuracy@K for K in {1, 3, 5}.

Evidence matching is exact string comparison on dia_ids (e.g. "D1:3").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class RetrievalMetrics:
    """Per-question retrieval metrics for a single K value."""

    k: int
    recall: float       # |retrieved ∩ evidence| / |evidence|
    precision: float    # |retrieved ∩ evidence| / k
    f1: float           # harmonic mean of recall and precision
    accuracy: float     # 1.0 if at least one hit, else 0.0


def compute_retrieval_metrics(
    retrieved_ids: Sequence[str],
    evidence_ids: Sequence[str],
    k_values: Sequence[int] = (1, 3, 5),
) -> dict[int, RetrievalMetrics]:
    """
    Compute retrieval metrics for all requested K values.

    Args:
        retrieved_ids: ordered list of retrieved dia_ids (ranked by score)
        evidence_ids:  ground-truth evidence dia_ids from the QA item
        k_values:      K values to evaluate (default: 1, 3, 5)

    Returns:
        dict mapping K → RetrievalMetrics
    """
    evidence_set = set(evidence_ids)
    results: dict[int, RetrievalMetrics] = {}

    for k in k_values:
        top_k = list(retrieved_ids[:k])
        top_k_set = set(top_k)
        hits = top_k_set & evidence_set
        num_hits = len(hits)

        # Recall: what fraction of evidence did we find?
        recall = num_hits / len(evidence_set) if evidence_set else 0.0

        # Precision: what fraction of retrieved items are in evidence?
        precision = num_hits / k if k > 0 else 0.0

        # F1: harmonic mean
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        # Accuracy: did we hit at least one evidence item?
        accuracy = 1.0 if num_hits > 0 else 0.0

        results[k] = RetrievalMetrics(
            k=k,
            recall=recall,
            precision=precision,
            f1=f1,
            accuracy=accuracy,
        )

    return results


def average_retrieval_metrics(
    per_question: list[dict[int, RetrievalMetrics]],
    k_values: Sequence[int] = (1, 3, 5),
) -> dict[int, RetrievalMetrics]:
    """
    Macro-average RetrievalMetrics across a list of per-question results.

    Args:
        per_question: list of {k: RetrievalMetrics} dicts, one per question
        k_values:     K values to aggregate

    Returns:
        dict mapping K → averaged RetrievalMetrics
    """
    if not per_question:
        return {k: RetrievalMetrics(k=k, recall=0.0, precision=0.0, f1=0.0, accuracy=0.0)
                for k in k_values}

    averages: dict[int, RetrievalMetrics] = {}
    for k in k_values:
        recalls = [pq[k].recall for pq in per_question if k in pq]
        precisions = [pq[k].precision for pq in per_question if k in pq]
        f1s = [pq[k].f1 for pq in per_question if k in pq]
        accuracies = [pq[k].accuracy for pq in per_question if k in pq]

        n = len(recalls) or 1
        averages[k] = RetrievalMetrics(
            k=k,
            recall=sum(recalls) / n,
            precision=sum(precisions) / n,
            f1=sum(f1s) / n,
            accuracy=sum(accuracies) / n,
        )

    return averages
