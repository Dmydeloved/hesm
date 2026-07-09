#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory.config import config_path, config_value, get as config_get
from memory.embedder import BailianEmbedder
from memory.retriever import HybridRetriever, strip_markdown_code_fence
from memory.storage import MemoryStorage
from memory.vector_store import ChromaVectorStore


DEFAULT_BENCHMARK = config_path("paths", "benchmark")
DEFAULT_DB = config_path("paths", "memory_db")
DEFAULT_CHROMA = config_path("paths", "chroma")
DEFAULT_OUTPUT = config_path("paths", "end_to_end_details")
DEFAULT_SUMMARY = config_path("paths", "end_to_end_summary")
DEFAULT_MODEL = str(config_value("evaluation", "model", "qwen-plus"))
DEFAULT_BASE_URL = str(config_get("evaluation", "base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
logger = logging.getLogger("evaluate_end_to_end")


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


class OpenAICompatibleChat:
    """Strict OpenAI-compatible chat client. No local or rule fallback."""

    def __init__(
        self,
        api_key: str | None,
        model: str,
        base_url: str,
        temperature: float = 0.0,
    ) -> None:
        from openai import OpenAI

        api_key = api_key or config_get("evaluation", "api_key")
        if not api_key:
            raise ValueError("Set evaluation.api_key in configs/config.yaml.")
        model = model or config_get("evaluation", "model", DEFAULT_MODEL)
        base_url = base_url or config_get("evaluation", "base_url", DEFAULT_BASE_URL)
        self.client = OpenAI(api_key=str(api_key), base_url=str(base_url))
        self.model = str(model)
        self.temperature = temperature

    def complete(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("LLM returned empty content.")
        return content.strip()


def answer_prompt(question: str, context_text: str) -> str:
    return f"""你是一个长期记忆问答助手。请只根据给定记忆上下文回答问题。

【记忆上下文】
{context_text}

【问题】
{question}

请输出简洁答案，不要编造上下文中没有的信息。"""


def judge_prompt(question: str, reference_answer: str, model_answer: str) -> str:
    return f"""你是严格的问答评审。请比较模型答案和参考答案是否语义一致。

只输出 JSON：
{{"score": 0.0到1.0之间的数字, "reason": "简短理由"}}

【问题】
{question}

【参考答案】
{reference_answer}

【模型答案】
{model_answer}
"""


def parse_judge_response(content: str) -> dict[str, Any]:
    payload = json.loads(strip_markdown_code_fence(content))
    if not isinstance(payload, dict):
        raise ValueError("Judge response must be a JSON object.")
    score = float(payload["score"])
    if not 0.0 <= score <= 1.0:
        raise ValueError("Judge score must be between 0 and 1.")
    return {
        "score": score,
        "reason": str(payload.get("reason") or ""),
    }


TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+")


def metric_tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text or "")]


def clipped_unigram_overlap(candidate: list[str], reference: list[str]) -> int:
    reference_counts: dict[str, int] = {}
    for token in reference:
        reference_counts[token] = reference_counts.get(token, 0) + 1

    overlap = 0
    for token in candidate:
        count = reference_counts.get(token, 0)
        if count <= 0:
            continue
        overlap += 1
        reference_counts[token] = count - 1
    return overlap


def bleu1(candidate_text: str, reference_text: str) -> float:
    candidate = metric_tokens(candidate_text)
    reference = metric_tokens(reference_text)
    if not candidate:
        return 1.0 if not reference else 0.0
    if not reference:
        return 0.0

    precision = clipped_unigram_overlap(candidate, reference) / len(candidate)
    if precision == 0.0:
        return 0.0
    brevity_penalty = 1.0 if len(candidate) > len(reference) else math.exp(1 - len(reference) / len(candidate))
    return brevity_penalty * precision


def token_f1(candidate_text: str, reference_text: str) -> float:
    candidate = metric_tokens(candidate_text)
    reference = metric_tokens(reference_text)
    if not candidate and not reference:
        return 1.0
    if not candidate or not reference:
        return 0.0

    overlap = clipped_unigram_overlap(candidate, reference)
    if overlap == 0:
        return 0.0
    precision = overlap / len(candidate)
    recall = overlap / len(reference)
    return 2 * precision * recall / (precision + recall)


def text_metrics(candidate_text: str, reference_text: str) -> dict[str, float]:
    return {
        "bleu1": bleu1(candidate_text, reference_text),
        "f1": token_f1(candidate_text, reference_text),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"cases": 0}
    scores = [float(row["judge"]["score"]) for row in rows]
    bleu1_scores = [float(row["metrics"]["bleu1"]) for row in rows]
    f1_scores = [float(row["metrics"]["f1"]) for row in rows]
    latencies = [float(row["latency_ms"]) for row in rows]
    p95_index = max(0, math.ceil(len(latencies) * 0.95) - 1)
    return {
        "cases": len(rows),
        "judge_score": statistics.fmean(scores),
        "bleu1": statistics.fmean(bleu1_scores),
        "f1": statistics.fmean(f1_scores),
        "latency_p50_ms": statistics.median(latencies),
        "latency_p95_ms": sorted(latencies)[p95_index],
    }


def evaluate(
    benchmark_path: Path,
    db_path: Path,
    chroma_path: Path,
    output_path: Path,
    summary_path: Path,
    chat: OpenAICompatibleChat,
    judge: OpenAICompatibleChat,
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
    rows: list[dict[str, Any]] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("w", encoding="utf-8") as output:
            for index, case in enumerate(iter_jsonl(benchmark_path), 1):
                if limit and index > limit:
                    break
                topic = case["topic_result"]
                started = time.perf_counter()
                retrieval = retriever.recall(
                    topic=topic["topic"],
                    core_entity=topic["core_entity"],
                    intent=topic.get("intent"),
                    entities=topic.get("entities") or [],
                    top_experience=top_experience,
                    top_segment=top_segment,
                    top_qa=top_qa,
                    state_key=f"{state_prefix}-{case['dialogue_index']}",
                )
                model_answer = chat.complete(
                    answer_prompt(case["query"], retrieval["context_text"])
                )
                judge_result = parse_judge_response(
                    judge.complete(judge_prompt(case["query"], case["answer"], model_answer))
                )
                metrics = text_metrics(model_answer, case["answer"])
                row = {
                    "case_id": case["case_id"],
                    "case_type": case["case_type"],
                    "query": case["query"],
                    "reference_answer": case["answer"],
                    "model_answer": model_answer,
                    "judge": judge_result,
                    "metrics": metrics,
                    "latency_ms": (time.perf_counter() - started) * 1000.0,
                    "retrieval_debug": retrieval["debug"],
                }
                rows.append(row)
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        storage.close()

    summary = summarize(rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("end-to-end evaluation complete cases=%s summary=%s", len(rows), summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--chroma-path", type=Path, default=DEFAULT_CHROMA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--state-prefix", default="e2e-eval")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--judge-model", default=str(config_value("evaluation", "judge_model", DEFAULT_MODEL)))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--judge-base-url", default=str(config_value("evaluation", "judge_base_url", "")))
    parser.add_argument("--api-key", default=str(config_value("evaluation", "api_key", "")))
    parser.add_argument("--judge-api-key", default=str(config_value("evaluation", "judge_api_key", "")))
    parser.add_argument("--top-experience", type=int, default=int(config_value("retrieval", "top_experience", 3)))
    parser.add_argument("--top-segment", type=int, default=int(config_value("retrieval", "top_segment", 5)))
    parser.add_argument("--top-qa", type=int, default=int(config_value("retrieval", "top_qa", 8)))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    chat = OpenAICompatibleChat(
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
    )
    judge = OpenAICompatibleChat(
        api_key=args.judge_api_key or args.api_key,
        model=args.judge_model,
        base_url=args.judge_base_url or args.base_url,
    )
    summary = evaluate(
        benchmark_path=args.benchmark,
        db_path=args.db,
        chroma_path=args.chroma_path,
        output_path=args.output,
        summary_path=args.summary,
        chat=chat,
        judge=judge,
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
