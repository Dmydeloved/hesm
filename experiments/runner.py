#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import platform
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.datasets import load_dataset
from experiments.llm import (
    OpenAICompatibleChat,
    answer_prompt,
    judge_prompt,
    parse_judge_response,
)
from experiments.metrics import row_metrics
from experiments.methods import build_method
from experiments.statistics import bootstrap_ci, mean, percentile
from memory.config import get as config_get
from memory.extractor import TopicExtractor


logger = logging.getLogger("experiments.runner")


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def truncate(value: Any, limit: int = 240) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def scrub_config(value: Any) -> Any:
    if isinstance(value, dict):
        scrubbed = {}
        for key, item in value.items():
            lower = str(key).lower()
            if "api_key" in lower or lower in {"key", "token", "secret"}:
                scrubbed[key] = "***"
            else:
                scrubbed[key] = scrub_config(item)
        return scrubbed
    if isinstance(value, list):
        return [scrub_config(item) for item in value]
    return value


def configure_logging(output_dir: Path, verbose: bool = False) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.propagate = False


def close_logging() -> None:
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def write_node_input(handle, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-c", "safe.directory=D:/code/hesm", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    groups["overall"] = rows
    for row in rows:
        groups[row["question_type"]].append(row)
    summary: dict[str, Any] = {}
    for name, items in groups.items():
        latencies = [float(item["latency_ms"]) for item in items]
        recalls = [float(item["metrics"]["evidence_recall"]) for item in items]
        answer_f1 = [float(item["metrics"]["answer_f1"]) for item in items]
        mrr = [float(item["metrics"]["mrr"]) for item in items]
        retrieved_tokens = [float(item["metrics"]["retrieved_tokens"]) for item in items]
        judge_scores = [
            float(item["judge"]["score"])
            for item in items
            if isinstance(item.get("judge"), dict) and "score" in item["judge"]
        ]
        low, high = bootstrap_ci(recalls)
        group_summary = {
            "cases": len(items),
            "answer_f1": mean(answer_f1),
            "evidence_recall": mean(recalls),
            "evidence_recall_ci95": [low, high],
            "mrr": mean(mrr),
            "retrieved_tokens": mean(retrieved_tokens),
            "latency_p50_ms": percentile(latencies, 0.50),
            "latency_p95_ms": percentile(latencies, 0.95),
        }
        if judge_scores:
            group_summary["judge_score"] = mean(judge_scores)
        summary[name] = group_summary
    return summary


def build_chat(config: dict[str, Any], section: str) -> OpenAICompatibleChat | None:
    if not bool(config.get("enabled", False)):
        return None
    return OpenAICompatibleChat(
        section=section,
        api_key=config.get("api_key"),
        api_key_env=config.get("api_key_env"),
        model=config.get("model"),
        base_url=config.get("base_url"),
        temperature=float(config.get("temperature", 0.0)),
    )


def build_topic_extractor(config: dict[str, Any]) -> TopicExtractor | None:
    if not bool(config.get("enabled", False)):
        return None
    api_key = config.get("api_key")
    api_key_env = config.get("api_key_env")
    if not api_key and api_key_env:
        import os

        api_key = os.getenv(str(api_key_env))
    return TopicExtractor(
        api_key=str(api_key) if api_key else None,
        model=config.get("model") or config_get("topic_extraction", "model"),
        base_url=config.get("base_url") or config_get("topic_extraction", "base_url"),
        max_retries=config.get("max_retries"),
        retry_delay=config.get("retry_delay"),
    )


def build_topic_context(history: list[dict[str, Any]], history_size: int) -> str:
    recent = history[-history_size:] if history_size > 0 else []
    return json.dumps(
        {
            "history_size": history_size,
            "recent_turns": recent,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def primary_topic_result(value: Any) -> dict[str, Any]:
    if isinstance(value, list):
        if not value:
            raise ValueError("Topic extraction returned an empty list.")
        value = value[0]
    if not isinstance(value, dict):
        raise ValueError("Topic extraction must return a dict or non-empty list.")
    return value


def same_experience(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        str(left.get("topic") or "") == str(right.get("topic") or "")
        and str(left.get("core_entity") or "") == str(right.get("core_entity") or "")
    )


def classify_question_type(
    previous_topic: dict[str, Any] | None,
    current_topic: dict[str, Any],
) -> str:
    if not current_topic:
        return "unknown"
    if previous_topic is None:
        return "cold_start"
    if same_experience(previous_topic, current_topic) and str(previous_topic.get("intent") or "") == str(
        current_topic.get("intent") or ""
    ):
        return "same_experience_same_intent"
    if same_experience(previous_topic, current_topic):
        return "same_experience_new_intent"
    return "topic_switch"


def online_evidence_user_inputs(
    history: list[dict[str, Any]],
    current_topic: dict[str, Any],
) -> list[str]:
    evidence: list[str] = []
    if not current_topic:
        return evidence
    for item in history:
        topic_value = item.get("topic_extraction")
        topic = primary_topic_result(topic_value) if topic_value else {}
        if topic and same_experience(topic, current_topic):
            evidence.append(str(item.get("user_input") or ""))
    return evidence


def dialogue_groups(examples: list[Any]) -> list[tuple[str, list[Any]]]:
    groups: list[tuple[str, list[Any]]] = []
    current_id = None
    current_items: list[Any] = []
    for example in examples:
        if example.dialogue_id != current_id:
            if current_items:
                groups.append((str(current_id), current_items))
            current_id = example.dialogue_id
            current_items = []
        current_items.append(example)
    if current_items:
        groups.append((str(current_id), current_items))
    return groups


def completed_dialogue_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            dialogue_id = payload.get("dialogue_id")
            if dialogue_id:
                completed.add(str(dialogue_id))
    return completed


def load_completed_rows(path: Path, completed: set[str]) -> list[dict[str, Any]]:
    if not path.exists() or not completed:
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(row.get("dialogue_id")) in completed:
                rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(config_path: Path, reset_output: bool = False) -> dict[str, Any]:
    config = load_config(config_path)
    seed = int(config.get("run", {}).get("seed", 42))
    random.seed(seed)
    dataset_config = config.get("dataset", {})
    method_config = config.get("method", {})
    run_config = config.get("run", {})
    generation_config = config.get("generation", {})
    judge_config = config.get("judge", {})
    topic_config = config.get("topic_extraction", {})
    output_dir = resolve_path(run_config.get("output_dir", "results/experiments/run"))
    reset_output = reset_output or bool(run_config.get("reset_output", False))
    if reset_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir, verbose=bool(run_config.get("verbose", False)))
    progress_interval = max(1, int(run_config.get("progress_interval", 10)))
    log_node_inputs = bool(run_config.get("log_node_inputs", True))
    logger.info("run_start config=%s output_dir=%s", config_path, output_dir)
    logger.info("config=%s", json.dumps(scrub_config(config), ensure_ascii=False))

    examples = load_dataset(
        name=str(dataset_config.get("name")),
        path=resolve_path(dataset_config.get("path")),
        limit=int(dataset_config.get("limit", 0)),
        dialogue_limit=int(dataset_config.get("dialogue_limit", 0)),
    )
    groups = dialogue_groups(examples)
    logger.info(
        "dataset_loaded name=%s path=%s examples=%s dialogues=%s limit=%s dialogue_limit=%s",
        dataset_config.get("name"),
        resolve_path(dataset_config.get("path")),
        len(examples),
        len(groups),
        dataset_config.get("limit", 0),
        dataset_config.get("dialogue_limit", 0),
    )
    method = build_method(
        name=str(method_config.get("name")),
        output_dir=output_dir,
        config=method_config,
    )
    answerer = build_chat(generation_config, "evaluation")
    judge = build_chat(judge_config, "evaluation")
    topic_extractor = build_topic_extractor(topic_config)
    topic_history_size = int(topic_config.get("history_size", 5))
    topic_domain_knowledge = str(topic_config.get("domain_knowledge", ""))
    logger.info(
        "method_ready name=%s topic_extraction=%s topic_history_size=%s generation=%s judge=%s",
        method_config.get("name"),
        topic_extractor is not None,
        topic_history_size,
        answerer is not None,
        judge is not None,
    )
    predictions_path = output_dir / "predictions.jsonl"
    node_inputs_path = output_dir / "node_inputs.jsonl"
    completed_path = output_dir / "completed_dialogues.jsonl"
    completed = completed_dialogue_ids(completed_path)
    rows: list[dict[str, Any]] = load_completed_rows(predictions_path, completed)
    write_jsonl(predictions_path, rows)
    if completed:
        logger.info(
            "resume_enabled completed_dialogues=%s completed_rows=%s incomplete_dialogue_will_rerun=true",
            len(completed),
            len(rows),
        )

    with predictions_path.open("a", encoding="utf-8") as target, node_inputs_path.open(
        "w", encoding="utf-8"
    ) as node_target:
        case_index = len(rows)
        for dialogue_index, (dialogue_id, dialogue_examples) in enumerate(groups, 1):
            if dialogue_id in completed:
                logger.info(
                    "dialogue_skip_completed id=%s dialogue_index=%s cases=%s",
                    dialogue_id,
                    dialogue_index,
                    len(dialogue_examples),
                )
                continue
            method.reset_dialogue(dialogue_id)
            topic_history: list[dict[str, Any]] = []
            previous_online_topic: dict[str, Any] | None = None
            logger.info(
                "dialogue_start id=%s dialogue_index=%s cases=%s",
                dialogue_id,
                dialogue_index,
                len(dialogue_examples),
            )
            for example in dialogue_examples:
                case_index += 1
                if topic_extractor is not None:
                    recent_topic_history = (
                        topic_history[-topic_history_size:] if topic_history_size > 0 else []
                    )
                    topic_context = build_topic_context(topic_history, topic_history_size)
                    if log_node_inputs:
                        write_node_input(
                            node_target,
                            {
                                "node": "topic_extraction_input",
                                "case_id": example.case_id,
                                "dialogue_id": example.dialogue_id,
                                "turn_index": example.turn_index,
                                "user_input": example.user_input,
                                "history_turns": len(recent_topic_history),
                                "context": json.loads(topic_context),
                            },
                        )
                    raw_topic_result = topic_extractor.extract(
                        user_input=example.user_input,
                        context=topic_context,
                        domain_knowledge=topic_domain_knowledge,
                    )
                    example.metadata["precomputed_topic_result"] = example.topic_result
                    example.metadata["online_topic_result"] = raw_topic_result
                    example.topic_result = primary_topic_result(raw_topic_result)
                    example.question_type = classify_question_type(
                        previous_online_topic,
                        example.topic_result,
                    )
                    example.evidence_user_inputs = online_evidence_user_inputs(
                        topic_history,
                        example.topic_result,
                    )
                    previous_online_topic = example.topic_result
                    if log_node_inputs:
                        write_node_input(
                            node_target,
                            {
                                "node": "topic_extraction_output",
                                "case_id": example.case_id,
                                "dialogue_id": example.dialogue_id,
                                "topic_result": raw_topic_result,
                            },
                        )
                    logger.info(
                        "topic_extracted case_id=%s history_turns=%s topic=%s core_entity=%s intent=%s",
                        example.case_id,
                        len(recent_topic_history),
                        example.topic_result.get("topic", ""),
                        example.topic_result.get("core_entity", ""),
                        example.topic_result.get("intent", ""),
                    )

                topic = example.topic_result
                if log_node_inputs:
                    write_node_input(
                        node_target,
                        {
                            "node": "retrieve_input",
                            "case_id": example.case_id,
                            "dialogue_id": example.dialogue_id,
                            "turn_index": example.turn_index,
                            "question_type": example.question_type,
                            "user_input": truncate(example.user_input),
                            "topic": topic.get("topic", ""),
                            "core_entity": topic.get("core_entity", ""),
                            "intent": topic.get("intent", ""),
                            "entities": topic.get("entities") or [],
                            "evidence_count": len(example.evidence_user_inputs),
                        },
                    )
                started = time.perf_counter()
                result = method.predict(example)
                model_answer = result.answer
                if answerer is not None:
                    if log_node_inputs:
                        write_node_input(
                            node_target,
                            {
                                "node": "generation_input",
                                "case_id": example.case_id,
                                "user_input": truncate(example.user_input),
                                "context_chars": len(result.context_text),
                                "context_preview": truncate(result.context_text),
                                "retrieved_ids": [
                                    item.get("id") for item in result.retrieved_items
                                ],
                            },
                        )
                    model_answer = answerer.complete(
                        answer_prompt(example.user_input, result.context_text)
                    )
                judge_result = None
                if judge is not None:
                    if log_node_inputs:
                        write_node_input(
                            node_target,
                            {
                                "node": "judge_input",
                                "case_id": example.case_id,
                                "question": truncate(example.user_input),
                                "reference_answer": truncate(example.reference_answer),
                                "model_answer": truncate(model_answer),
                            },
                        )
                    judge_result = parse_judge_response(
                        judge.complete(
                            judge_prompt(
                                question=example.user_input,
                                reference_answer=example.reference_answer,
                                model_answer=model_answer,
                            )
                        )
                    )
                    if log_node_inputs:
                        write_node_input(
                            node_target,
                            {
                                "node": "judge_output",
                                "case_id": example.case_id,
                                "judge": judge_result,
                            },
                        )
                latency_ms = (time.perf_counter() - started) * 1000.0
                metrics = row_metrics(
                    answer=model_answer,
                    reference_answer=example.reference_answer,
                    retrieved_items=result.retrieved_items,
                    evidence_user_inputs=example.evidence_user_inputs,
                    context_text=result.context_text,
                )
                method.observe(example)
                topic_history.append(
                    {
                        "case_id": example.case_id,
                        "turn_index": example.turn_index,
                        "user_input": example.user_input,
                        "assistant_output": example.assistant_output,
                        "topic_extraction": example.metadata.get(
                            "online_topic_result", example.topic_result
                        ),
                    }
                )
                if topic_history_size > 0 and len(topic_history) > topic_history_size:
                    topic_history = topic_history[-topic_history_size:]
                row = {
                    "case_id": example.case_id,
                    "dialogue_id": example.dialogue_id,
                    "turn_index": example.turn_index,
                    "question_type": example.question_type,
                    "topic_result": example.topic_result,
                    "latency_ms": latency_ms,
                    "metrics": metrics,
                    "answer": model_answer,
                    "reference_answer": example.reference_answer,
                    "judge": judge_result,
                    "retrieved_items": result.retrieved_items,
                    "debug": result.debug,
                }
                rows.append(row)
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
                target.flush()
                if log_node_inputs:
                    write_node_input(
                        node_target,
                        {
                            "node": "case_output",
                            "case_id": example.case_id,
                            "latency_ms": latency_ms,
                            "metrics": metrics,
                            "retrieved_count": len(result.retrieved_items),
                            "judge_score": (
                                judge_result.get("score")
                                if isinstance(judge_result, dict)
                                else None
                            ),
                        },
                    )
                if case_index == 1 or case_index % progress_interval == 0 or case_index == len(examples):
                    logger.info(
                        "progress %s/%s case_id=%s dialogue_id=%s qtype=%s latency_ms=%.2f retrieved=%s evidence_recall=%.3f mrr=%.3f judge=%s",
                        case_index,
                        len(examples),
                        example.case_id,
                        example.dialogue_id,
                        example.question_type,
                        latency_ms,
                        len(result.retrieved_items),
                        metrics["evidence_recall"],
                        metrics["mrr"],
                        judge_result.get("score") if isinstance(judge_result, dict) else "NA",
                    )
            completed_record = {
                "dialogue_id": dialogue_id,
                "cases": len(dialogue_examples),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            with completed_path.open("a", encoding="utf-8") as completed_target:
                completed_target.write(json.dumps(completed_record, ensure_ascii=False) + "\n")
            completed.add(dialogue_id)
            logger.info(
                "dialogue_complete id=%s cases=%s checkpoint=%s",
                dialogue_id,
                len(dialogue_examples),
                completed_path,
            )
    close = getattr(method, "close", None)
    if callable(close):
        close()
    summary = summarize_rows(rows)
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("metrics_written path=%s", output_dir / "metrics.json")
    shutil.copy2(config_path, output_dir / "config.yaml")
    manifest = {
        "config": str(config_path),
        "examples": len(examples),
        "git_commit": git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "seed": seed,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "run_complete examples=%s predictions=%s node_inputs=%s",
        len(examples),
        predictions_path,
        node_inputs_path,
    )
    close_logging()
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--reset-output",
        action="store_true",
        help="删除配置中的 output_dir 后重新开始；中断恢复时不要加这个参数。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run(args.config, reset_output=args.reset_output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
