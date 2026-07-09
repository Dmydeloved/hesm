#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory.config import config_path, config_value
from memory.embedder import BailianEmbedder
from memory.extractor import TopicExtractor, TopicRecord, TopicResult
from memory.manager import MemoryManager
from memory.retriever import HybridRetriever
from memory.storage import MemoryStorage
from memory.vector_store import ChromaVectorStore


DEFAULT_DB = config_path("paths", "memory_db")
DEFAULT_CHROMA = config_path("paths", "chroma")
logger = logging.getLogger("topic_memory_flow")


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def topic_records(topic_result: TopicResult) -> list[TopicRecord]:
    return topic_result if isinstance(topic_result, list) else [topic_result]


class TopicMemoryFlow:
    """Extract topic, write memory in background, and retrieve in foreground."""

    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB,
        chroma_path: str | Path = DEFAULT_CHROMA,
        state_key: str = "default",
        max_workers: int = 1,
        topic_extractor: TopicExtractor | None = None,
        embedder: BailianEmbedder | None = None,
        retriever: HybridRetriever | None = None,
        writer_vector_store_factory: Callable[[], ChromaVectorStore] | None = None,
        retrieval_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.chroma_path = Path(chroma_path)
        self.state_key = state_key
        self.topic_extractor = topic_extractor or TopicExtractor()
        self.embedder = embedder or BailianEmbedder()
        self.writer_vector_store_factory = writer_vector_store_factory
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        if retriever is None:
            self.retrieval_storage = MemoryStorage(self.db_path)
            retriever = HybridRetriever(
                storage=self.retrieval_storage,
                vector_store=ChromaVectorStore(self.chroma_path),
                embedder=self.embedder,
                **(retrieval_kwargs or {}),
            )
        else:
            self.retrieval_storage = None
        self.retriever = retriever

    def close(self, wait: bool = True) -> None:
        self.executor.shutdown(wait=wait)
        if self.retrieval_storage is not None:
            self.retrieval_storage.close()

    def run(
        self,
        user_input: str,
        assistant_output: str = "",
        context: str = "",
        domain_knowledge: str = "",
        tools: list[dict[str, Any]] | None = None,
        top_experience: int = 3,
        top_segment: int = 5,
        top_qa: int = 8,
    ) -> dict[str, Any]:
        topic_result = self.topic_extractor.extract(
            user_input=user_input,
            context=context,
            domain_knowledge=domain_knowledge,
        )
        records = topic_records(topic_result)
        if not records:
            raise ValueError("Topic extraction returned no records.")

        write_future = self.store_async(
            topic_records=records,
            user_input=user_input,
            assistant_output=assistant_output,
            tools=tools or [],
        )

        primary_topic = records[0]
        retrieval = self.retriever.recall(
            topic=primary_topic["topic"],
            core_entity=primary_topic["core_entity"],
            intent=primary_topic.get("intent"),
            entities=primary_topic.get("entities") or [],
            top_experience=top_experience,
            top_segment=top_segment,
            top_qa=top_qa,
            state_key=self.state_key,
        )

        return {
            "user_input": user_input,
            "topic_result": topic_result,
            "retrieval": retrieval,
            "memory_write_future": write_future,
        }

    def store_async(
        self,
        topic_records: list[TopicRecord],
        user_input: str,
        assistant_output: str = "",
        tools: list[dict[str, Any]] | None = None,
    ) -> Future[list[dict[str, Any]]]:
        return self.executor.submit(
            self._store_records,
            topic_records,
            user_input,
            assistant_output,
            tools or [],
        )

    def _store_records(
        self,
        topic_records: list[TopicRecord],
        user_input: str,
        assistant_output: str,
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        storage = MemoryStorage(self.db_path)
        vector_store = (
            self.writer_vector_store_factory()
            if self.writer_vector_store_factory is not None
            else ChromaVectorStore(self.chroma_path)
        )
        manager = MemoryManager(
            storage=storage,
            vector_store=vector_store,
            embedder=self.embedder,
        )
        results: list[dict[str, Any]] = []
        try:
            for topic_record in topic_records:
                results.append(
                    manager.add_qa(
                        topic_result=topic_record,
                        user_input=user_input,
                        assistant_output=assistant_output,
                        tools=tools,
                        state_key=self.state_key,
                    )
                )
            return results
        finally:
            storage.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("user_input", nargs="?", help="User input to process.")
    parser.add_argument("--context", default="")
    parser.add_argument("--domain-knowledge", default="")
    parser.add_argument("--assistant-output", default="")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--chroma-path", type=Path, default=DEFAULT_CHROMA)
    parser.add_argument("--state-key", default="default")
    parser.add_argument("--top-experience", type=int, default=int(config_value("retrieval", "top_experience", 3)))
    parser.add_argument("--top-segment", type=int, default=int(config_value("retrieval", "top_segment", 5)))
    parser.add_argument("--top-qa", type=int, default=int(config_value("retrieval", "top_qa", 8)))
    parser.add_argument("--wait-memory-write", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    user_input = args.user_input or input("user_input> ").strip()
    if not user_input:
        raise ValueError("user_input is required.")

    flow = TopicMemoryFlow(
        db_path=args.db,
        chroma_path=args.chroma_path,
        state_key=args.state_key,
    )
    try:
        result = flow.run(
            user_input=user_input,
            assistant_output=args.assistant_output,
            context=args.context,
            domain_knowledge=args.domain_knowledge,
            top_experience=args.top_experience,
            top_segment=args.top_segment,
            top_qa=args.top_qa,
        )
        write_future = result.pop("memory_write_future")
        if args.wait_memory_write:
            result["memory_write"] = write_future.result()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        flow.close(wait=args.wait_memory_write)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
