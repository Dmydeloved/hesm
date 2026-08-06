"""Recalculate avg_retrieved_tokens for the four current LoCoMo methods.

The script reads ``retrieved_context`` from the per-conversation answer files
and counts every method with the same encoder in one process.  It writes
nothing; results are printed to stdout.

Run from the project root::

    python -B tests/recalculate_avg_retrieved_tokens.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.locomo.evaluation.token_metrics import _get_encoder, count_tokens


METHODS = ("hesm", "mem0", "vector_rag", "amem")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recalculate avg_retrieved_tokens from the retrieved_context "
            "stored in the four methods' answer files."
        )
    )
    parser.add_argument(
        "--answers-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "locomo" / "answers",
        help="Directory containing <method>_<conv-id>.json answer files.",
    )
    parser.add_argument(
        "--encoding",
        default="cl100k_base",
        help="tiktoken encoding name (default: cl100k_base).",
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)

    raw_records = payload.get("records", {})
    if isinstance(raw_records, dict):
        def record_key(item: tuple[str, Any]) -> tuple[int, str]:
            key = str(item[0])
            return (int(key), key) if key.isdigit() else (sys.maxsize, key)

        return [record for _, record in sorted(raw_records.items(), key=record_key)]
    if isinstance(raw_records, list):
        return raw_records
    raise ValueError(f"Unsupported records format in {path}")


def encoder_name(encoder: Any, requested_encoding: str) -> str:
    name = getattr(encoder, "name", None)
    if name:
        return f"tiktoken:{name}"
    return f"fallback:whitespace (requested {requested_encoding})"


def main() -> int:
    args = parse_args()
    answers_dir = args.answers_dir.resolve()
    if not answers_dir.is_dir():
        print(f"ERROR: answers directory does not exist: {answers_dir}", file=sys.stderr)
        return 1

    # Resolve the encoder once so every subsequent count_tokens() call uses one
    # identical tokenizer backend, including the project's whitespace fallback.
    encoder = _get_encoder(args.encoding)

    print(f"Answers directory : {answers_dir}")
    print(f"Tokenizer backend : {encoder_name(encoder, args.encoding)}")
    print()
    print(
        f"{'Method':<12} {'Files':>5} {'Records':>8} "
        f"{'Recalculated avg':>18} {'Stored avg':>12} {'Delta':>12}"
    )
    print("-" * 73)

    had_error = False
    for method in METHODS:
        paths = sorted(answers_dir.glob(f"{method}_*.json"))
        if not paths:
            print(f"{method:<12} {'0':>5} {'0':>8} {'MISSING':>18}")
            had_error = True
            continue

        token_counts: list[int] = []
        stored_counts: list[float] = []
        for path in paths:
            for record in load_records(path):
                context = record.get("retrieved_context")
                if not isinstance(context, str):
                    raise ValueError(
                        f"Missing string retrieved_context in {path.name}"
                    )
                token_counts.append(count_tokens(context, args.encoding))

                stored = record.get("retrieved_tokens")
                if isinstance(stored, (int, float)) and not isinstance(stored, bool):
                    stored_counts.append(float(stored))

        if not token_counts:
            print(f"{method:<12} {len(paths):>5} {'0':>8} {'NO RECORDS':>18}")
            had_error = True
            continue

        recalculated_avg = sum(token_counts) / len(token_counts)
        stored_avg = (
            sum(stored_counts) / len(stored_counts)
            if len(stored_counts) == len(token_counts)
            else None
        )
        delta = recalculated_avg - stored_avg if stored_avg is not None else None

        stored_text = f"{stored_avg:.2f}" if stored_avg is not None else "N/A"
        delta_text = f"{delta:+.2f}" if delta is not None else "N/A"
        print(
            f"{method:<12} {len(paths):>5} {len(token_counts):>8} "
            f"{recalculated_avg:>18.2f} {stored_text:>12} {delta_text:>12}"
        )

    return 1 if had_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
