"""
Part 2 — Ablation Study: evaluate 5 HESM retrieval variants.

Variants:
  flat_memory  — raw turn vectors, no TopicExtractor (independent ChromaDB)
  qa_only      — HESM QA vector layer only
  qa_segment   — HESM QA + Segment layers merged
  full_hesm_no_reranker — full hierarchy without LLM reranking
  full_hesm    — full HybridRetriever.recall() (3-layer)

The last 4 variants reuse HESM storage built in Part 1. When it is missing,
run_ablation builds the shared storage automatically.

Usage (from d:/code/hesm):
    python -m experiments.locomo.run_ablation
    python -m experiments.locomo.run_ablation --variants qa_only qa_segment full_hesm
    python -m experiments.locomo.run_ablation --max-conversations 2

Outputs:
    outputs/locomo/answers/ablation_{variant}_{conv_id}.json
    outputs/locomo/metrics/ablation_{variant}_metrics.json
    outputs/locomo/tables/ablation_results.{md,csv,json}
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import yaml

from experiments.locomo.data.loader import LoCoMoLoader
from experiments.locomo.evaluation.aggregator import MethodMetrics
from experiments.locomo.evaluation.judge import LLMJudge
from experiments.locomo.methods.base import LLMAnswerGenerator
from experiments.locomo.methods.hesm_adapter import HESMAblationMemory
from experiments.locomo.reporting.table_generator import TableGenerator
from experiments.locomo.runner.qa_runner import QARunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_ALL_VARIANTS = [
    "flat_memory",
    "qa_only",
    "qa_segment",
    "full_hesm_no_reranker",
    "full_hesm",
]


def run_ablation(
    config_path: str | Path | None = None,
    enabled_variants: list[str] | None = None,
    max_conversations: int | None = None,
) -> list[MethodMetrics]:
    if config_path is None:
        config_path = _PROJECT_ROOT / "experiments" / "locomo" / "config" / "experiment.yaml"

    with open(config_path, encoding="utf-8") as f:
        exp_cfg: dict[str, Any] = yaml.safe_load(f)
    with open(_PROJECT_ROOT / "configs" / "config.yaml", encoding="utf-8") as f:
        hesm_cfg: dict[str, Any] = yaml.safe_load(f)

    random.seed(exp_cfg.get("experiment", {}).get("seed", 42))

    answers_dir = _PROJECT_ROOT / exp_cfg["output"]["answers"]
    metrics_dir = _PROJECT_ROOT / exp_cfg["output"]["metrics"]
    tables_dir  = _PROJECT_ROOT / exp_cfg["output"]["tables"]
    logs_dir = _PROJECT_ROOT / exp_cfg["output"].get("logs", "outputs/locomo/logs")
    memory_root = _PROJECT_ROOT / exp_cfg["output"]["memory"]
    for d in (answers_dir, metrics_dir, tables_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    max_conv = max_conversations or exp_cfg.get("experiment", {}).get("max_conversations")
    loader = LoCoMoLoader(_PROJECT_ROOT / exp_cfg["dataset"]["path"])
    conversations = loader.load(max_conversations=max_conv)

    answer_generator = LLMAnswerGenerator(hesm_cfg)
    judge = LLMJudge(hesm_cfg)
    top_k_values: list[int] = exp_cfg.get("retrieval", {}).get("top_k_values", [1, 3, 5])
    token_encoding: str = exp_cfg.get("token_counter", {}).get("encoding", "cl100k_base")

    ablation_cfg: dict[str, Any] = exp_cfg.get("ablation", {}).get("variants", {})
    hesm_section = exp_cfg.get("hesm", {})
    variants = enabled_variants or _ALL_VARIANTS

    all_metrics: list[MethodMetrics] = []
    for variant in variants:
        if variant not in ablation_cfg:
            logger.warning("Variant %r not found in config, skipping", variant)
            continue

        logger.info("=== Ablation variant: %s ===", variant)
        method = HESMAblationMemory(
            variant=variant,
            memory_root=memory_root,
            hesm_cfg=hesm_section,
            variant_cfg=ablation_cfg[variant],
        )
        runner = QARunner(
            method=method,
            answer_generator=answer_generator,
            judge=judge,
            output_dir=answers_dir,
            metrics_dir=metrics_dir,
            logs_dir=logs_dir,
            top_k_values=top_k_values,
            token_encoding=token_encoding,
        )
        metrics = runner.run(conversations)
        all_metrics.append(metrics)

    tg = TableGenerator(tables_dir)
    tg.generate_ablation_results(all_metrics)
    logger.info("=== Part 2 complete. Tables → %s ===", tables_dir)
    return all_metrics


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoCoMo Benchmark — Ablation Study (Part 2)")
    p.add_argument("--config", default=None)
    p.add_argument(
        "--variants", nargs="+",
        choices=_ALL_VARIANTS, default=None,
        help="Ablation variants to run",
    )
    p.add_argument("--max-conversations", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_ablation(
        config_path=args.config,
        enabled_variants=args.variants,
        max_conversations=args.max_conversations,
    )
