"""
Checkpoint system for the LoCoMo experiment runner.

Enables resuming interrupted experiments without re-running completed questions.
One checkpoint file per (method, conv_id) pair.

Checkpoint file schema:
{
  "method": "hesm",
  "conv_id": "conv-26",
  "status": "partial" | "complete",
  "total_questions": 199,
  "completed_indices": [0, 1, 2, ...],
  "records": [<QARecord>, ...]   # indexed by question position
}

QARecord schema (written by runner, read by aggregator):
{
  "question": str,
  "ground_truth": str,
  "prediction": str,
  "retrieved_ids": list[str],
  "retrieved_context": str,
  "retrieved_tokens": int,
  "evidence": list[str],
  "category": int,
  "f1": float,
  "f1_precision": float,
  "f1_recall": float,
  "judge_score": int,            # 0/1 or -1 on failure
  "query_status": str,            # success | failed
  "stage_status": {               # retrieval / answer / judge details
    "retrieval": {"status": "success" | "failed", ...},
    "answer": {"status": "success" | "failed" | "skipped", ...},
    "judge": {"status": "success" | "failed" | "skipped", ...}
  },
  "retrieval_metrics": {         # keyed by str(K)
    "1": {"recall": ..., "precision": ..., "f1": ..., "accuracy": ...},
    "3": {...},
    "5": {...},
  },
  "total_conversation_tokens": int,
  "compression_ratio": float,
  "latency_ms": float,           # retrieval latency in milliseconds
}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Checkpoint:
    """
    Manages per-(method, conv_id) checkpoint files.

    Usage:
        ckpt = Checkpoint(output_dir, method_name, conv_id)
        if ckpt.is_complete():
            records = ckpt.load_records()
        else:
            for i, qa in enumerate(questions):
                if ckpt.is_done(i):
                    continue
                record = ...  # run QA
                ckpt.save_record(i, record, total_questions=len(questions))
    """

    def __init__(self, output_dir: str | Path, method_name: str, conv_id: str) -> None:
        self.output_dir = Path(output_dir)
        self.method_name = method_name
        self.conv_id = conv_id
        self._path = self.output_dir / f"{method_name}_{conv_id}.json"
        self._state: dict[str, Any] = self._load()

    # ─── Public API ───────────────────────────────────────────────────────────

    def is_complete(self) -> bool:
        return self._state.get("status") == "complete"

    def is_done(self, question_index: int) -> bool:
        return question_index in self._state.get("completed_indices_set", set())

    def is_answered(
        self,
        question_index: int,
        question: str | None = None,
    ) -> bool:
        """Return whether all required stages succeeded for this question.

        New checkpoints carry an explicit ``query_status``. Older checkpoint
        files are accepted only when they contain a non-empty prediction and a
        successful Judge score. This deliberately retries legacy empty-answer
        and ``judge_score == -1`` records.
        """
        if not self.is_done(question_index):
            return False
        record = self._state.get("records", {}).get(question_index)
        if not isinstance(record, dict):
            return False
        if question is not None:
            saved_question = str(record.get("question") or "").strip()
            if saved_question != str(question).strip():
                return False
        query_status = record.get("query_status")
        if query_status is not None:
            return query_status == "success"

        prediction = str(record.get("prediction") or "").strip()
        judge_score = record.get("judge_score", -1)
        return bool(prediction) and isinstance(judge_score, (int, float)) and judge_score >= 0

    def load_records(self) -> list[dict[str, Any]]:
        """Return all completed QA records in question order."""
        return list(self._state.get("records", {}).values())

    def load_all_records_ordered(self, total_questions: int) -> list[dict[str, Any]]:
        """Return records in index order (index 0..total_questions-1)."""
        records_map: dict[int, dict] = self._state.get("records", {})
        return [records_map[i] for i in range(total_questions) if i in records_map]

    def save_record(
        self,
        question_index: int,
        record: dict[str, Any],
        total_questions: int,
        successful: bool = True,
    ) -> None:
        """Persist a record and mark it complete only when successful.

        Failed records remain available for diagnostics and metrics, but their
        indices are omitted from ``completed_indices`` so they are retried.
        """
        records: dict[int, dict] = self._state.setdefault("records", {})
        records[question_index] = record

        completed: list[int] = self._state.setdefault("completed_indices", [])
        if successful and question_index not in completed:
            completed.append(question_index)
        elif not successful and question_index in completed:
            completed.remove(question_index)
        completed.sort()
        self._state["completed_indices_set"] = set(completed)

        self._state["total_questions"] = total_questions
        self._state["method"] = self.method_name
        self._state["conv_id"] = self.conv_id

        if len(completed) >= total_questions:
            self._state["status"] = "complete"
        else:
            self._state["status"] = "partial"

        self._persist()

    def num_completed(self) -> int:
        return len(self._state.get("completed_indices", []))

    # ─── Internal ─────────────────────────────────────────────────────────────

    def _load(self) -> dict[str, Any]:
        if self._path.exists():
            try:
                with open(self._path, encoding="utf-8") as f:
                    state = json.load(f)
                # Rebuild set for fast lookup (sets are not JSON-serialisable)
                state["completed_indices_set"] = set(
                    state.get("completed_indices", [])
                )
                # Convert string keys back to int (JSON forces string keys)
                raw_records = state.get("records", {})
                state["records"] = {int(k): v for k, v in raw_records.items()}
                logger.info(
                    "Resumed checkpoint %s: %d/%s questions done",
                    self._path.name,
                    len(state["completed_indices"]),
                    state.get("total_questions", "?"),
                )
                return state
            except Exception as exc:
                logger.warning("Corrupt checkpoint %s, starting fresh: %s", self._path, exc)
        return {}

    def _persist(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Serialise: convert int-keyed records and set to list
        serialisable = dict(self._state)
        serialisable.pop("completed_indices_set", None)
        serialisable["records"] = {
            str(k): v for k, v in self._state.get("records", {}).items()
        }
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(serialisable, f, ensure_ascii=False, indent=2)
