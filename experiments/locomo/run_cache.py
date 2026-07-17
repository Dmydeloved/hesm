"""
Part 3 — Cache Evaluation: compare HESM with cache ON vs OFF.

Measures:
  - P50 / P90 / P99 / Average retrieval latency
  - Cache hit rate (embedding cache)
  - LLM call reduction (memory cache, via use_cache flag)

Requires HESM memory to have been built in Part 1 (run_main.py with 'hesm' enabled).

Usage (from d:/code/hesm):
    python -m experiments.locomo.run_cache
    python -m experiments.locomo.run_cache --num-samples 100

Outputs:
    outputs/locomo/tables/cache_results.{md,csv,json}
    outputs/locomo/metrics/cache_raw.json
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
from experiments.locomo.methods.hesm_adapter import HESMMemory
from experiments.locomo.reporting.table_generator import TableGenerator
from experiments.locomo.runner.qa_runner import CacheEvalRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_cache_eval(
    config_path: str | Path | None = None,
    num_samples: int | None = None,
) -> dict[str, Any]:
    if config_path is None:
        config_path = _PROJECT_ROOT / "experiments" / "locomo" / "config" / "experiment.yaml"

    with open(config_path, encoding="utf-8") as f:
        exp_cfg: dict[str, Any] = yaml.safe_load(f)
    with open(_PROJECT_ROOT / "configs" / "config.yaml", encoding="utf-8") as f:
        hesm_cfg: dict[str, Any] = yaml.safe_load(f)

    random.seed(exp_cfg.get("experiment", {}).get("seed", 42))

    metrics_dir = _PROJECT_ROOT / exp_cfg["output"]["metrics"]
    tables_dir  = _PROJECT_ROOT / exp_cfg["output"]["tables"]
    memory_root = _PROJECT_ROOT / exp_cfg["output"]["memory"]
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    cache_cfg = exp_cfg.get("cache_eval", {})
    n_samples = num_samples or cache_cfg.get("num_latency_samples", 50)
    percentiles: list[int] = cache_cfg.get("percentiles", [50, 90, 99])

    max_conv = exp_cfg.get("experiment", {}).get("max_conversations")
    loader = LoCoMoLoader(_PROJECT_ROOT / exp_cfg["dataset"]["path"])
    conversations = loader.load(max_conversations=max_conv)

    hesm_section = exp_cfg.get("hesm", {})
    hesm_memory = HESMMemory(
        memory_root=memory_root,
        hesm_cfg=hesm_section,
        use_llm_summarizer=hesm_section.get("use_llm_summarizer", True),
        use_llm_reranker=hesm_section.get("use_llm_reranker", True),
    )

    runner = CacheEvalRunner(
        hesm_memory=hesm_memory,
        output_dir=metrics_dir,
        percentiles=percentiles,
        num_samples=n_samples,
    )
    cache_data = runner.run(conversations)

    tg = TableGenerator(tables_dir)
    tg.generate_cache_results(cache_data)
    logger.info("=== Part 3 complete. Tables → %s ===", tables_dir)
    return cache_data


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoCoMo Benchmark — Cache Evaluation (Part 3)")
    p.add_argument("--config", default=None)
    p.add_argument(
        "--num-samples", type=int, default=None,
        help="Number of questions to sample for latency measurement",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_cache_eval(
        config_path=args.config,
        num_samples=args.num_samples,
    )
