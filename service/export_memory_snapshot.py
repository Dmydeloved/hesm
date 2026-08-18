"""Export a read-only HESM SQLite snapshot for the static frontend."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


def parse_json(value: str, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return fallback


def export_snapshot(database: Path, output: Path) -> None:
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row

    runtime_row = connection.execute(
        "SELECT current_experience_id, current_segment_id, updated_at "
        "FROM runtime_state WHERE state_key = 'default'"
    ).fetchone()

    qa_by_segment: dict[str, list[dict[str, Any]]] = {}
    for row in connection.execute("SELECT * FROM qa_memory ORDER BY rowid"):
        qa = {
            "id": row["qa_id"],
            "timestamp": row["timestamp"],
            "userInput": row["user_input"],
            "assistantOutput": row["assistant_output"],
            "tools": parse_json(row["tools_json"], []),
            "topic": row["topic"],
            "intent": row["intent"],
            "coreEntity": row["core_entity"],
            "entities": parse_json(row["entities_json"], []),
            "status": row["status"],
            "confidence": row["confidence"],
            "reasoning": row["reasoning"],
        }
        qa_by_segment.setdefault(row["segment_id"], []).append(qa)

    segments_by_experience: dict[str, list[dict[str, Any]]] = {}
    for row in connection.execute("SELECT * FROM segment_memory ORDER BY rowid"):
        segment = {
            "id": row["segment_id"],
            "topic": row["topic"],
            "intent": row["intent"],
            "coreEntity": row["core_entity"],
            "status": row["status"],
            "summary": row["summary"],
            "createdAt": row["created_at"],
            "updatedAt": row["updated_at"],
            "version": row["version"],
            "lastSummarizedQaCount": row["last_summarized_qa_count"],
            "qas": qa_by_segment.get(row["segment_id"], []),
        }
        segments_by_experience.setdefault(row["experience_id"], []).append(segment)

    experiences: list[dict[str, Any]] = []
    for row in connection.execute("SELECT * FROM experience_memory ORDER BY rowid"):
        experiences.append(
            {
                "id": row["experience_id"],
                "topic": row["topic"],
                "coreEntity": row["core_entity"],
                "intents": parse_json(row["intents_link_json"], []),
                "summary": parse_json(row["summary_json"], ""),
                "state": parse_json(row["state_json"], {}),
                "historyExperience": parse_json(
                    row["history_experience_json"], {}
                ),
                "createdAt": row["created_at"],
                "updatedAt": row["updated_at"],
                "version": row["version"],
                "lastSummarizedSegmentCount": row["last_summarized_segment_count"],
                "segments": segments_by_experience.get(row["experience_id"], []),
            }
        )

    snapshot = {
        "meta": {
            "source": str(database).replace("\\", "/"),
            "experienceCount": len(experiences),
            "segmentCount": sum(len(item["segments"]) for item in experiences),
            "qaCount": sum(
                len(segment["qas"])
                for item in experiences
                for segment in item["segments"]
            ),
            "runtime": dict(runtime_row) if runtime_row else {},
        },
        "experiences": experiences,
    }
    connection.close()

    payload = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))
    output.write_text(
        "/* Generated from the read-only HESM SQLite snapshot. */\n"
        f"window.HESM_MEMORY_DATA={payload};\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export_snapshot(args.database.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
