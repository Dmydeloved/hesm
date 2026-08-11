"""Persistent build progress helpers for resumable memory construction."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.locomo.data.loader import Session


def expected_dia_ids(sessions: list[Session]) -> set[str]:
    """Return traceable, non-empty dialogue IDs expected in a memory store."""
    return {
        turn.dia_id
        for session in sessions
        for turn in session.turns
        if turn.dia_id and turn.text.strip()
    }


def load_completed_dia_ids(path: str | Path) -> set[str]:
    state_path = Path(path)
    if not state_path.exists():
        return set()
    try:
        data: dict[str, Any] = json.loads(state_path.read_text(encoding="utf-8"))
        return {
            str(value)
            for value in data.get("completed_dia_ids", [])
            if str(value).strip()
        }
    except (OSError, TypeError, ValueError):
        return set()


def save_build_state(
    path: str | Path,
    *,
    method: str,
    conv_id: str,
    expected: set[str],
    completed: set[str],
) -> None:
    """Atomically save per-turn build progress."""
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    completed_expected = expected & completed
    payload = {
        "method": method,
        "conv_id": conv_id,
        "status": "complete" if expected <= completed else "partial",
        "expected_count": len(expected),
        "completed_count": len(completed_expected),
        "completed_dia_ids": sorted(completed_expected),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    temporary_path = state_path.with_suffix(state_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(state_path)

