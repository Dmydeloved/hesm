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
from memory.embedder import BailianEmbedder
from memory.manager import MemoryManager
from memory.storage import MemoryStorage
from memory.vector_store import ChromaVectorStore


DEFAULT_INPUT = config_path("paths", "topic_output")
DEFAULT_DB = config_path("paths", "memory_db")
DEFAULT_CHROMA = config_path("paths", "chroma")


logger = logging.getLogger("build_memory_from_multiwoz_topics")


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def iter_topic_qas(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for dialogue_index, line in enumerate(source):
            if not line.strip():
                continue
            dialogue = json.loads(line)
            turns = dialogue.get("dialogue") or []
            for turn_index, turn in enumerate(turns):
                if turn.get("role") != "user" or "topic_extraction" not in turn:
                    continue
                assistant_output = next_system_output(turns, turn_index)
                topic_items = turn["topic_extraction"]
                if isinstance(topic_items, dict):
                    topic_items = [topic_items]
                if not isinstance(topic_items, list):
                    logger.warning(
                        "跳过非法 topic_extraction dialogue=%s turn=%s",
                        dialogue_index,
                        turn_index,
                    )
                    continue
                for topic_result in topic_items:
                    if isinstance(topic_result, dict):
                        yield {
                            "dialogue_index": dialogue_index,
                            "turn_index": turn_index,
                            "topic_result": topic_result,
                            "user_input": turn.get("content") or "",
                            "assistant_output": assistant_output,
                        }


def next_system_output(turns: list[dict[str, Any]], turn_index: int) -> str:
    for next_turn in turns[turn_index + 1 :]:
        if next_turn.get("role") == "system":
            return str(next_turn.get("content") or "")
        if next_turn.get("role") == "user":
            break
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--chroma-path", type=Path, default=DEFAULT_CHROMA)
    parser.add_argument("--limit", type=int, default=0, help="Maximum topic QAs to import. 0 means all.")
    parser.add_argument("--state-key", default="default")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    storage = MemoryStorage(args.db)
    manager = MemoryManager(
        storage,
        vector_store=ChromaVectorStore(args.chroma_path),
        embedder=BailianEmbedder(),
    )
    imported = 0
    try:
        for item in iter_topic_qas(args.input):
            if args.limit and imported >= args.limit:
                break
            result = manager.add_qa(
                topic_result=item["topic_result"],
                user_input=item["user_input"],
                assistant_output=item["assistant_output"],
                tools=[],
                state_key=args.state_key,
            )
            imported += 1
            logger.info(
                "导入完成 count=%s dialogue=%s turn=%s qa_id=%s segment_id=%s experience_id=%s action=%s",
                imported,
                item["dialogue_index"],
                item["turn_index"],
                result["qa_id"],
                result["segment_id"],
                result["experience_id"],
                result["action"],
            )
    finally:
        storage.close()
    logger.info("导入结束 imported=%s db=%s", imported, args.db.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
