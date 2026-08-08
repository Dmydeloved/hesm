"""Per-method, four-stage experiment logging.

Each method writes to ``outputs/locomo/logs/<method>.log``.  Every line is a
single JSON object so logs remain both human-readable and easy to aggregate.
The ``stage`` field is one of BUILD, RETRIEVAL, ANSWER, or JUDGE.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class MethodStageLogger:
    """Append structured stage events to one log file per method."""

    _write_lock = threading.Lock()
    _stages = ("BUILD", "RETRIEVAL", "ANSWER", "JUDGE")

    def __init__(self, logs_dir: str | Path, method_name: str) -> None:
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        safe_name = "".join(
            char if char.isalnum() or char in {"-", "_"} else "_"
            for char in method_name
        )
        self.path = self.logs_dir / f"{safe_name}.log"
        self.method_name = method_name
        self.event(
            "BUILD",
            "RUN_START",
            message="Four-stage method run started",
            sections=list(self._stages),
        )

    def event(
        self,
        stage: str,
        status: str,
        *,
        message: str = "",
        **details: Any,
    ) -> None:
        normalized_stage = stage.upper()
        if normalized_stage not in self._stages:
            raise ValueError(f"Unknown experiment stage: {stage}")

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": self.method_name,
            "stage": normalized_stage,
            "status": status.upper(),
            "message": message,
            **details,
        }
        line = json.dumps(payload, ensure_ascii=False, default=str)
        with self._write_lock:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def skipped_query(
        self,
        *,
        conv_id: str,
        query_index: int,
        question: str,
        reason: str,
    ) -> None:
        """Record that all query stages were skipped from a successful checkpoint."""
        common = {
            "conv_id": conv_id,
            "query_index": query_index,
            "question": question,
            "reason": reason,
        }
        for stage in ("RETRIEVAL", "ANSWER", "JUDGE"):
            self.event(stage, "SKIPPED", message="Query already successful", **common)

