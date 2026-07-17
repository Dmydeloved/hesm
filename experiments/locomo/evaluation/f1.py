"""
Token-level F1 score computation.

Standard NLP evaluation metric: computes precision, recall and F1 over
word-level tokens between a predicted answer and a ground-truth answer.
Follows the SQuAD / LoCoMo evaluation convention.
"""

from __future__ import annotations

import re
import string
from collections import Counter


def normalize_text(text: str) -> str:
    """
    Lowercase, remove punctuation and extra whitespace.
    Mirrors the normalization used in SQuAD / LoCoMo official evaluation.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    return normalize_text(text).split()


def compute_f1(prediction: str, ground_truth: str) -> dict[str, float]:
    """
    Compute token-level Precision, Recall and F1.

    Returns a dict with keys: "precision", "recall", "f1".
    All values are in [0.0, 1.0].
    """
    pred_tokens = tokenize(prediction)
    gold_tokens = tokenize(ground_truth)

    if not pred_tokens and not gold_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_tokens or not gold_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)

    # Intersection: minimum count of each shared token
    common: Counter = pred_counter & gold_counter
    num_common = sum(common.values())

    if num_common == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}
