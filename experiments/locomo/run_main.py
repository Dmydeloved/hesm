"""
Part 1 — Main Experiment: evaluate all 5 memory systems on LoCoMo.

Usage (from d:/code/hesm):
    python -m experiments.locomo.run_main
    python -m experiments.locomo.run_main --config experiments/locomo/config/experiment.yaml
    python -m experiments.locomo.run_main --methods full_context vector_rag hesm
    python -m experiments.locomo.run_main --max-conversations 2  # quick test

Outputs:
    outputs/locomo/answers/{method}_{conv_id}.json  — per-question QA records
    outputs/locomo/metrics/{method}_metrics.json    — aggregated metrics
    outputs/locomo/tables/main_results.{md,csv,json}
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path

# Ensure the HESM project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import yaml

from experiments.locomo.data.loader import LoCoMoLoader
from experiments.locomo.evaluation.aggregator import MethodMetrics
from experiments.locomo.evaluation.judge import LLMJudge
from experiments.locomo.methods.amem_adapter import AMEMMemory
from experiments.locomo.methods.base import LLMAnswerGenerator, MemorySystem
from experiments.locomo.methods.full_context import FullContextMemory
from experiments.locomo.methods.hesm_adapter import HESMMemory
from experiments.locomo.methods.mem0_adapter import Mem0Memory
from experiments.locomo.methods.vector_rag import VectorRAGMemory
from experiments.locomo.reporting.table_generator import TableGenerator
from experiments.locomo.runner.qa_runner import QARunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_experiment_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_hesm_config() -> dict:
    """Load the main HESM configs/config.yaml."""
    cfg_path = _PROJECT_ROOT / "configs" / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_methods(
    exp_cfg: dict,
    hesm_cfg: dict,
    enabled_methods: list[str] | None,
) -> list[MemorySystem]:
    """Construct all enabled MemorySystem instances."""
    methods_cfg = exp_cfg.get("methods", {})
    memory_root = _PROJECT_ROOT / exp_cfg["output"]["memory"]
    hesm_section = exp_cfg.get("hesm", {})
    amem_section = exp_cfg.get("amem", {})
    mem0_section = exp_cfg.get("mem0", {})

    all_methods: list[tuple[str, MemorySystem]] = [
        ("full_context", FullContextMemory()),
        ("vector_rag",   VectorRAGMemory(memory_root=memory_root)),
        ("mem0",         Mem0Memory(
            memory_root=memory_root,
            hesm_config=hesm_cfg,
            collection_prefix=mem0_section.get("collection_prefix", "mem0_locomo"),
        )),
        ("amem",         AMEMMemory(
            memory_root=memory_root,
            hesm_config=hesm_cfg,
            amem_cfg=amem_section,
        )),
        ("hesm",         HESMMemory(
            memory_root=memory_root,
            hesm_cfg=hesm_section,
            use_llm_summarizer=hesm_section.get("use_llm_summarizer", True),
            use_llm_reranker=hesm_section.get("use_llm_reranker", True),
        )),
    ]

    selected: list[MemorySystem] = []
    for name, method in all_methods:
        if enabled_methods and name not in enabled_methods:
            continue
        if not methods_cfg.get(name, {}).get("enabled", True):
            continue
        selected.append(method)
        logger.info("Enabled method: %s", name)

    return selected


def run_main(
    config_path: str | Path | None = None,
    enabled_methods: list[str] | None = None,
    max_conversations: int | None = None,
) -> list[MethodMetrics]:
    """
    Main entry point — usable both as a script and as a library call.
    Returns list of MethodMetrics (one per method).
    """
    if config_path is None:
        config_path = _PROJECT_ROOT / "experiments" / "locomo" / "config" / "experiment.yaml"

    exp_cfg = load_experiment_config(config_path)
    hesm_cfg = load_hesm_config()

    # Fixed random seed
    seed = exp_cfg.get("experiment", {}).get("seed", 42)
    random.seed(seed)

    # Resolve output paths
    root = _PROJECT_ROOT / exp_cfg["output"]["root"]
    answers_dir = _PROJECT_ROOT / exp_cfg["output"]["answers"]
    metrics_dir = _PROJECT_ROOT / exp_cfg["output"]["metrics"]
    tables_dir  = _PROJECT_ROOT / exp_cfg["output"]["tables"]
    logs_dir = _PROJECT_ROOT / exp_cfg["output"].get("logs", "outputs/locomo/logs")
    for d in (answers_dir, metrics_dir, tables_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Load dataset
    max_conv = max_conversations or exp_cfg.get("experiment", {}).get("max_conversations")
    loader = LoCoMoLoader(_PROJECT_ROOT / exp_cfg["dataset"]["path"])
    conversations = loader.load(max_conversations=max_conv)
    logger.info("Loaded %d conversations", len(conversations))

    # Shared evaluators
    answer_generator = LLMAnswerGenerator(hesm_cfg)
    judge = LLMJudge(hesm_cfg)
    top_k_values: list[int] = exp_cfg.get("retrieval", {}).get("top_k_values", [1, 3, 5])
    token_encoding: str = exp_cfg.get("token_counter", {}).get("encoding", "cl100k_base")

    methods = build_methods(exp_cfg, hesm_cfg, enabled_methods)
    if not methods:
        logger.error("No methods enabled — check experiment.yaml")
        return []

    # Run each method
    all_metrics: list[MethodMetrics] = []
    for method in methods:
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

    # Generate result tables
    tg = TableGenerator(tables_dir)
    tg.generate_main_results(all_metrics, filename_stem="main_results")

    logger.info("=== Part 1 complete. Tables → %s ===", tables_dir)
    return all_metrics


# ─── CLI entry point ─────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoCoMo Benchmark — Main Experiment (Part 1)")
    p.add_argument("--config", default=None, help="Path to experiment.yaml")
    p.add_argument(
        "--methods",
        nargs="+",
        choices=["full_context", "vector_rag", "mem0", "amem", "hesm"],
        default=None,
        help="Subset of methods to run (default: all enabled in config)",
    )
    p.add_argument(
        "--max-conversations",
        type=int,
        default=None,
        help="Limit number of conversations (for quick testing)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_main(
        config_path=args.config,
        enabled_methods=args.methods,
        max_conversations=args.max_conversations,
    )
