#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory.config import config_path, config_value
from memory.embedder import BailianEmbedder
from memory.retriever import HybridRetriever
from memory.storage import MemoryStorage
from memory.vector_store import ChromaVectorStore


DEFAULT_BENCHMARK = config_path("paths", "benchmark")
DEFAULT_DB = config_path("paths", "memory_db")
DEFAULT_CHROMA = config_path("paths", "chroma")
DEFAULT_DETAILS = config_path("paths", "retrieval_details")
DEFAULT_SUMMARY = config_path("paths", "retrieval_summary")
logger = logging.getLogger("evaluate_retrieval")


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def exact_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def approx_tokens(text: str) -> int:
    return max(1, int(len(text or "") / 4))


def qa_rank(qas: list[dict[str, Any]], gold: dict[str, Any]) -> int | None:
    gold_input = exact_text(gold.get("user_input"))
    gold_topic = str(gold.get("topic") or "")
    gold_entity = str(gold.get("core_entity") or "")
    for index, qa in enumerate(qas, 1):
        if (
            exact_text(qa.get("user_input")) == gold_input
            and str(qa.get("topic") or "") == gold_topic
            and str(qa.get("core_entity") or "") == gold_entity
        ):
            return index
    return None


def experience_hit(experiences: list[dict[str, Any]], gold: dict[str, Any]) -> bool:
    return any(
        item.get("topic") == gold.get("topic")
        and item.get("core_entity") == gold.get("core_entity")
        for item in experiences
    )


def segment_hit(segments: list[dict[str, Any]], gold: dict[str, Any]) -> bool:
    return any(
        item.get("topic") == gold.get("topic")
        and item.get("core_entity") == gold.get("core_entity")
        and item.get("intent") == gold.get("intent")
        for item in segments
    )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    groups["overall"] = rows
    for row in rows:
        groups[row["case_type"]].append(row)

    summary: dict[str, Any] = {}
    for name, items in groups.items():
        if not items:
            continue
        latencies = [item["latency_ms"] for item in items]
        p95_index = max(0, math.ceil(len(latencies) * 0.95) - 1)
        summary[name] = {
            "cases": len(items),
            "experience_hit": statistics.fmean(item["experience_hit"] for item in items),
            "segment_hit": statistics.fmean(item["segment_hit"] for item in items),
            "qa_hit": statistics.fmean(item["qa_hit"] for item in items),
            "qa_mrr": statistics.fmean(item["qa_rr"] for item in items),
            "experience_cache_hit": statistics.fmean(
                item["experience_cache_hit"] for item in items
            ),
            "segment_cache_hit": statistics.fmean(item["segment_cache_hit"] for item in items),
            "retrieved_tokens": statistics.fmean(item["retrieved_tokens"] for item in items),
            "latency_p50_ms": statistics.median(latencies),
            "latency_p95_ms": sorted(latencies)[p95_index],
        }
    return summary


def evaluate(
    benchmark_path: Path,
    db_path: Path,
    chroma_path: Path,
    details_path: Path,
    summary_path: Path,
    state_prefix: str,
    top_experience: int,
    top_segment: int,
    top_qa: int,
    limit: int = 0,
) -> dict[str, Any]:
    storage = MemoryStorage(db_path)
    retriever = HybridRetriever(
        storage=storage,
        vector_store=ChromaVectorStore(chroma_path),
        embedder=BailianEmbedder(),
    )
    details_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    try:
        with details_path.open("w", encoding="utf-8") as details_file:
            for index, case in enumerate(iter_jsonl(benchmark_path), 1):
                if limit and index > limit:
                    break
                topic = case["topic_result"]
                state_key = f"{state_prefix}-{case['dialogue_index']}"
                started = time.perf_counter()
                result = retriever.recall(
                    topic=topic["topic"],
                    core_entity=topic["core_entity"],
                    intent=topic.get("intent"),
                    entities=topic.get("entities") or [],
                    top_experience=top_experience,
                    top_segment=top_segment,
                    top_qa=top_qa,
                    state_key=state_key,
                )
                latency_ms = (time.perf_counter() - started) * 1000.0
                rank = qa_rank(result["qas"], case["gold"])
                row = {
                    "case_id": case["case_id"],
                    "case_type": case["case_type"],
                    "dialogue_index": case["dialogue_index"],
                    "experience_hit": experience_hit(result["experiences"], case["gold"]),
                    "segment_hit": segment_hit(result["segments"], case["gold"]),
                    "qa_hit": rank is not None,
                    "qa_rank": rank,
                    "qa_rr": 1.0 / rank if rank else 0.0,
                    "experience_cache_hit": bool(
                        result["debug"].get("experience_cache_hit")
                    ),
                    "segment_cache_hit": bool(result["debug"].get("segment_cache_hit")),
                    "retrieved_tokens": approx_tokens(result.get("context_text", "")),
                    "latency_ms": latency_ms,
                    "debug": result["debug"],
                }
                rows.append(row)
                details_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        storage.close()

    summary = summarize(rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("retrieval evaluation complete cases=%s summary=%s", len(rows), summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--chroma-path", type=Path, default=DEFAULT_CHROMA)
    parser.add_argument("--details", type=Path, default=DEFAULT_DETAILS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--state-prefix", default="retrieval-eval")
    parser.add_argument("--top-experience", type=int, default=int(config_value("retrieval", "top_experience", 3)))
    parser.add_argument("--top-segment", type=int, default=int(config_value("retrieval", "top_segment", 5)))
    parser.add_argument("--top-qa", type=int, default=int(config_value("retrieval", "top_qa", 8)))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    summary = evaluate(
        benchmark_path=args.benchmark,
        db_path=args.db,
        chroma_path=args.chroma_path,
        details_path=args.details,
        summary_path=args.summary,
        state_prefix=args.state_prefix,
        top_experience=args.top_experience,
        top_segment=args.top_segment,
        top_qa=args.top_qa,
        limit=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
