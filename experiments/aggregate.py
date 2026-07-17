#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def iter_metric_files(root: Path):
    yield from root.glob("**/metrics.json")


def flatten(run_name: str, summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for group, metrics in summary.items():
        row = {"run": run_name, "group": group}
        row.update(metrics)
        rows.append(row)
    return rows


def aggregate(root: Path, output: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in iter_metric_files(root):
        summary = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.extend(flatten(str(metrics_path.parent.relative_to(root)), summary))
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".json":
        output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        fieldnames = sorted({key for row in rows for key in row})
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/experiments"))
    parser.add_argument("--output", type=Path, default=Path("results/experiments/summary.csv"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = aggregate(args.root, args.output)
    print(json.dumps({"runs": len({row["run"] for row in rows}), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

