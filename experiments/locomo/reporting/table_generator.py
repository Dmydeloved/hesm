"""
Report generator — produces paper-ready result tables in Markdown, CSV and JSON.

Three report types:
  - Main results table  (5 methods × all metrics)
  - Ablation table      (5 HESM variants × all metrics)
  - Cache eval table    (cache ON/OFF × latency metrics)
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from experiments.locomo.evaluation.aggregator import MethodMetrics

logger = logging.getLogger(__name__)


# ─── Column definitions ───────────────────────────────────────────────────────

# Ordered column list for the main / ablation result table
_MAIN_COLUMNS = [
    ("method",           "Method"),
    ("avg_f1",           "F1"),
    ("avg_judge_score",  "Judge"),
    ("accuracy@1",       "Acc@1"),
    ("accuracy@3",       "Acc@3"),
    ("accuracy@5",       "Acc@5"),
    ("recall@1",         "Rec@1"),
    ("recall@3",         "Rec@3"),
    ("recall@5",         "Rec@5"),
    ("precision@1",      "Pre@1"),
    ("precision@3",      "Pre@3"),
    ("precision@5",      "Pre@5"),
    ("f1@1",             "EvF1@1"),
    ("f1@3",             "EvF1@3"),
    ("f1@5",             "EvF1@5"),
    ("avg_retrieved_tokens",  "Tokens"),
    ("avg_compression_ratio", "Compression"),
]

_CACHE_COLUMNS = [
    ("mode",             "Mode"),
    ("avg_latency_ms",   "Avg Latency (ms)"),
    ("p50_ms",           "P50 (ms)"),
    ("p90_ms",           "P90 (ms)"),
    ("p99_ms",           "P99 (ms)"),
    ("cache_hit_rate",   "Hit Rate"),
    ("latency_reduction","Latency Reduction"),
]


# ─── Public API ───────────────────────────────────────────────────────────────

class TableGenerator:
    """Generates MD / CSV / JSON result tables from MethodMetrics objects."""

    def __init__(self, tables_dir: str | Path) -> None:
        self.tables_dir = Path(tables_dir)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

    # ── Main / Ablation ───────────────────────────────────────────────────────

    def generate_main_results(
        self,
        metrics_list: list[MethodMetrics],
        filename_stem: str = "main_results",
    ) -> None:
        """
        Write main_results.md / .csv / .json for a list of MethodMetrics.
        Called for both Part 1 (main experiment) and Part 2 (ablation).
        """
        rows = [self._metrics_to_row(m) for m in metrics_list]
        self._write_all(rows, _MAIN_COLUMNS, filename_stem)
        logger.info("Tables written: %s.{md,csv,json}", filename_stem)

    def generate_ablation_results(self, metrics_list: list[MethodMetrics]) -> None:
        self.generate_main_results(metrics_list, filename_stem="ablation_results")

    # ── Cache eval ────────────────────────────────────────────────────────────

    def generate_cache_results(self, cache_data: dict[str, Any]) -> None:
        """
        Write cache_results.md / .csv / .json from the CacheEvalRunner output dict.
        """
        rows: list[dict[str, Any]] = []
        for mode in ("cache_off", "cache_on"):
            d = cache_data.get(mode, {})
            percs = d.get("latency_percentiles", {})
            rows.append({
                "mode": mode,
                "avg_latency_ms": round(d.get("avg_latency_ms", 0.0), 2),
                "p50_ms": round(percs.get("p50", 0.0), 2),
                "p90_ms": round(percs.get("p90", 0.0), 2),
                "p99_ms": round(percs.get("p99", 0.0), 2),
                "cache_hit_rate": round(cache_data.get("cache_hit_rate", 0.0), 4),
                "latency_reduction": _pct(cache_data.get("latency_reduction", 0.0)),
            })
        self._write_all(rows, _CACHE_COLUMNS, "cache_results")
        logger.info("Tables written: cache_results.{md,csv,json}")

    # ─── Internal: row extraction ─────────────────────────────────────────────

    @staticmethod
    def _metrics_to_row(m: MethodMetrics) -> dict[str, Any]:
        row: dict[str, Any] = {
            "method": m.method_name,
            "avg_f1": _pct(m.avg_f1),
            "avg_judge_score": round(m.avg_judge_score, 4),
            "avg_retrieved_tokens": round(m.avg_retrieved_tokens, 1),
            "avg_compression_ratio": round(m.avg_compression_ratio, 2),
        }
        for k, km in m.retrieval.items():
            row[f"recall@{k}"] = _pct(km.recall)
            row[f"precision@{k}"] = _pct(km.precision)
            row[f"f1@{k}"] = _pct(km.f1)
            row[f"accuracy@{k}"] = _pct(km.accuracy)
        return row

    # ─── Internal: writers ────────────────────────────────────────────────────

    def _write_all(
        self,
        rows: list[dict[str, Any]],
        columns: list[tuple[str, str]],
        stem: str,
    ) -> None:
        self._write_json(rows, stem)
        self._write_csv(rows, columns, stem)
        self._write_markdown(rows, columns, stem)

    def _write_json(self, rows: list[dict[str, Any]], stem: str) -> None:
        path = self.tables_dir / f"{stem}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

    def _write_csv(
        self,
        rows: list[dict[str, Any]],
        columns: list[tuple[str, str]],
        stem: str,
    ) -> None:
        path = self.tables_dir / f"{stem}.csv"
        keys = [k for k, _ in columns]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    def _write_markdown(
        self,
        rows: list[dict[str, Any]],
        columns: list[tuple[str, str]],
        stem: str,
    ) -> None:
        path = self.tables_dir / f"{stem}.md"
        keys = [k for k, _ in columns]
        headers = [h for _, h in columns]

        lines: list[str] = []
        # Header row
        lines.append("| " + " | ".join(headers) + " |")
        # Separator row (right-align numeric columns)
        seps: list[str] = []
        for k, _ in columns:
            if k == "method" or k == "mode":
                seps.append(":---")
            else:
                seps.append("---:")
        lines.append("| " + " | ".join(seps) + " |")
        # Data rows
        for row in rows:
            cells = [str(row.get(k, "")) for k in keys]
            lines.append("| " + " | ".join(cells) + " |")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


# ─── Helper ───────────────────────────────────────────────────────────────────

def _pct(v: float) -> str:
    """Format a 0-1 float as a percentage string, e.g. 0.8234 → '82.34%'."""
    return f"{v * 100:.2f}%"
