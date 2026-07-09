#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory.config import config_path


DEFAULT_INPUT = config_path("paths", "topic_output")
DEFAULT_OUTPUT = config_path("paths", "benchmark")
logger = logging.getLogger("build_topic_benchmark")


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


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


def same_segment(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return same_experience(left, right) and str(left.get("intent") or "") == str(
        right.get("intent") or ""
    )


def case_type(previous: dict[str, Any] | None, current: dict[str, Any]) -> str:
    if previous is None:
        return "cold_start"
    if same_segment(previous, current):
        return "same_experience_same_intent"
    if same_experience(previous, current):
        return "same_experience_new_intent"
    return "topic_switch"


def iter_cases(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for dialogue_index, line in enumerate(source):
            if not line.strip():
                continue
            dialogue = json.loads(line)
            turns = dialogue.get("dialogue") or []
            previous_topic: dict[str, Any] | None = None
            user_turn_index = 0
            for turn_index, turn in enumerate(turns):
                if turn.get("role") != "user":
                    continue
                items = topic_items(turn.get("topic_extraction"))
                if not items:
                    continue
                current = items[0]
                kind = case_type(previous_topic, current)
                case_id = f"d{dialogue_index:05d}_u{user_turn_index:03d}"
                yield {
                    "case_id": case_id,
                    "dialogue_index": dialogue_index,
                    "turn_index": turn_index,
                    "user_turn_index": user_turn_index,
                    "case_type": kind,
                    "query": str(turn.get("content") or ""),
                    "answer": next_system_output(turns, turn_index),
                    "scene": dialogue.get("scene") or [],
                    "topic_result": current,
                    "all_topic_results": items,
                    "gold": {
                        "topic": current.get("topic", ""),
                        "core_entity": current.get("core_entity", ""),
                        "intent": current.get("intent", ""),
                        "entities": current.get("entities") or [],
                        "user_input": str(turn.get("content") or ""),
                    },
                    "expected_cache": {
                        "experience": previous_topic is not None
                        and same_experience(previous_topic, current),
                        "segment": previous_topic is not None
                        and same_segment(previous_topic, current),
                    },
                }
                previous_topic = current
                user_turn_index += 1


def build_benchmark(input_path: Path, output_path: Path, limit: int = 0) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as target:
        for case in iter_cases(input_path):
            if limit and written >= limit:
                break
            target.write(json.dumps(case, ensure_ascii=False) + "\n")
            written += 1
    if written == 0:
        raise ValueError(f"No benchmark cases were built from {input_path}.")
    logger.info("benchmark built cases=%s output=%s", written, output_path.resolve())
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    build_benchmark(args.input, args.output, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
