"""
One-click runner — executes all three experiment parts in sequence.

Part 1: Main experiment  (Vector RAG, Mem0, A-MEM, HESM)
Part 2: Ablation study   (5 HESM variants)
Part 3: Cache evaluation (cache ON/OFF)

Usage (from d:/code/hesm):
    python -m experiments.locomo.run_all
    python -m experiments.locomo.run_all --skip-parts ablation cache
    python -m experiments.locomo.run_all --max-conversations 2  # quick test
    python -m experiments.locomo.run_all --methods hesm vector_rag  # subset
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_RUN_ALL_METHODS = ["vector_rag", "mem0", "amem", "hesm"]


def run_all(
    config_path: str | Path | None = None,
    skip_parts: list[str] | None = None,
    enabled_methods: list[str] | None = None,
    max_conversations: int | None = None,
) -> None:
    skip = set(skip_parts or [])

    if "main" not in skip:
        logger.info("=" * 60)
        logger.info("PART 1: Main Experiment")
        logger.info("=" * 60)
        from experiments.locomo.run_main import run_main
        requested_methods = (
            list(enabled_methods) if enabled_methods is not None else _RUN_ALL_METHODS
        )
        main_methods = [
            method for method in requested_methods if method in _RUN_ALL_METHODS
        ]
        removed_methods = sorted(set(requested_methods) - set(main_methods))
        if removed_methods:
            logger.warning(
                "run_all ignores unsupported methods: %s", removed_methods
            )
        if main_methods:
            run_main(
                config_path=config_path,
                # Full Context remains available from run_main, but is
                # intentionally excluded from this one-click suite.
                enabled_methods=main_methods,
                max_conversations=max_conversations,
            )
        else:
            logger.warning("Skipping Part 1: no supported run_all methods selected")
    else:
        logger.info("Skipping Part 1 (main)")

    if "ablation" not in skip:
        logger.info("=" * 60)
        logger.info("PART 2: Ablation Study")
        logger.info("=" * 60)
        from experiments.locomo.run_ablation import run_ablation
        run_ablation(
            config_path=config_path,
            max_conversations=max_conversations,
        )
    else:
        logger.info("Skipping Part 2 (ablation)")

    if "cache" not in skip:
        logger.info("=" * 60)
        logger.info("PART 3: Cache Evaluation")
        logger.info("=" * 60)
        from experiments.locomo.run_cache import run_cache_eval
        run_cache_eval(config_path=config_path)
    else:
        logger.info("Skipping Part 3 (cache)")

    logger.info("=" * 60)
    logger.info("ALL EXPERIMENTS COMPLETE")
    logger.info("Results: %s", _PROJECT_ROOT / "outputs" / "locomo" / "tables")
    logger.info("=" * 60)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoCoMo Benchmark — Full Experiment Suite")
    p.add_argument("--config", default=None, help="Path to experiment.yaml")
    p.add_argument(
        "--skip-parts", nargs="+",
        choices=["main", "ablation", "cache"], default=None,
        help="Parts to skip",
    )
    p.add_argument(
        "--methods", nargs="+",
        choices=_RUN_ALL_METHODS,
        default=None,
        help="Methods to run in Part 1 (default: all four run_all methods)",
    )
    p.add_argument("--max-conversations", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_all(
        config_path=args.config,
        skip_parts=args.skip_parts,
        enabled_methods=args.methods,
        max_conversations=args.max_conversations,
    )
