"""
Extract test questions from the first LoCoMo conversation.

Selects up to MAX_PER_CATEGORY (default=5) questions per category and
writes them to tests/locomo/questions.json.

Usage (from d:/code/hesm):
    python -m tests.locomo.questions_builder
    python -m tests.locomo.questions_builder --max-per-category 3

Output:
    tests/locomo/questions.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_DEFAULT_DATASET = _PROJECT_ROOT / "data" / "locomo10.json"
_OUTPUT_PATH = Path(__file__).parent / "questions.json"
_MAX_PER_CATEGORY = 5


def build_questions(
    dataset_path: str | Path | None = None,
    max_per_category: int = _MAX_PER_CATEGORY,
    output_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """
    Load the first LoCoMo conversation and extract up to max_per_category
    questions per category.

    Returns a list of question dicts, each containing:
      - conv_id: conversation identifier
      - question: question text
      - answer: ground-truth answer string
      - evidence: list of dia_ids (e.g. ["D1:3"])
      - category: integer category label (1-5)
      - q_index: original index in the QA list
    """
    from experiments.locomo.data.loader import LoCoMoLoader

    path = Path(dataset_path or _DEFAULT_DATASET)
    loader = LoCoMoLoader(path)
    conversations = loader.load(max_conversations=1)  # only first conv

    if not conversations:
        raise RuntimeError(f"No conversations loaded from {path}")

    conv = conversations[0]
    logger.info(
        "First conversation: %s | speakers: %s & %s | %d QA pairs",
        conv.conv_id,
        conv.speaker_a,
        conv.speaker_b,
        len(conv.questions),
    )

    # Group by category
    by_category: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for idx, qa in enumerate(conv.questions):
        by_category[qa.category].append({
            "conv_id": conv.conv_id,
            "question": qa.question,
            "answer": qa.answer_str(),
            "evidence": qa.evidence,
            "category": qa.category,
            "q_index": idx,
        })

    # Sample up to max_per_category per category
    selected: list[dict[str, Any]] = []
    for cat in sorted(by_category.keys()):
        items = by_category[cat]
        chosen = items[:max_per_category]
        selected.extend(chosen)
        logger.info(
            "  Category %d: %d available → %d selected", cat, len(items), len(chosen)
        )

    logger.info("Total selected: %d questions", len(selected))

    # Save
    out = Path(output_path or _OUTPUT_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "conv_id": conv.conv_id,
                "speaker_a": conv.speaker_a,
                "speaker_b": conv.speaker_b,
                "max_per_category": max_per_category,
                "questions": selected,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info("Questions saved → %s", out)
    return selected


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build LoCoMo test question set")
    p.add_argument("--dataset", default=None, help="Path to locomo10.json")
    p.add_argument("--max-per-category", type=int, default=_MAX_PER_CATEGORY)
    p.add_argument("--output", default=None, help="Output path for questions.json")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_questions(
        dataset_path=args.dataset,
        max_per_category=args.max_per_category,
        output_path=args.output,
    )
