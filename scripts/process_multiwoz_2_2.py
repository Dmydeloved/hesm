#!/usr/bin/env python3
"""Extract only scenes and dialogue text from the original MultiWOZ 2.2 data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory.config import config_path


DEFAULT_SOURCE = config_path("paths", "raw_multiwoz")
DEFAULT_OUTPUT = config_path("paths", "processed_dialogues")
SPLITS = ("train", "dev", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Original MultiWOZ 2.2 directory containing train/dev/test.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10_000,
        help="Maximum dialogues to extract. Use 0 to process all dialogues.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output file if it already exists.",
    )
    return parser.parse_args()


def dialogue_files(source: Path) -> list[Path]:
    files = [
        path
        for split in SPLITS
        for path in sorted((source / split).glob("dialogues_*.json"))
    ]
    if not files:
        raise FileNotFoundError(
            f"No MultiWOZ 2.2 dialogue files found under {source.resolve()}."
        )
    return files


def original_dialogues(files: list[Path]) -> Iterator[dict[str, Any]]:
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Expected a dialogue list in {path}.")
        for dialogue in data:
            if not isinstance(dialogue, dict):
                raise ValueError(f"Expected dialogue objects in {path}.")
            yield dialogue


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def infer_scenes(dialogue: dict[str, Any]) -> list[str]:
    scenes = []
    for turn in dialogue.get("turns", []):
        for frame in turn.get("frames", []):
            state = frame.get("state") or {}
            is_active = (
                state.get("active_intent") not in (None, "NONE")
                or bool(frame.get("actions"))
                or bool(frame.get("slots"))
            )
            service = clean_text(frame.get("service")).lower()
            if is_active and service and service not in scenes:
                scenes.append(service)
    return scenes


def extract_dialogue(dialogue: dict[str, Any]) -> dict[str, Any]:
    scenes = [
        clean_text(service).lower()
        for service in dialogue.get("services", [])
        if clean_text(service)
    ]
    if not scenes:
        scenes = infer_scenes(dialogue)

    turns = []
    for turn in dialogue.get("turns", []):
        text = clean_text(turn.get("utterance"))
        speaker = clean_text(turn.get("speaker")).lower()
        if text and speaker in {"user", "system"}:
            turns.append({"role": speaker, "content": text})

    if not scenes:
        raise ValueError(f"Dialogue {dialogue.get('dialogue_id')} has no scene.")
    if not turns:
        raise ValueError(f"Dialogue {dialogue.get('dialogue_id')} has no usable turns.")
    return {"scene": scenes, "dialogue": turns}


def process(source: Path, output: Path, limit: int, overwrite: bool) -> int:
    if limit < 0:
        raise ValueError("--limit must be zero or greater.")
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} already exists; use --overwrite.")

    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output.open("w", encoding="utf-8", newline="\n") as destination:
        for dialogue in original_dialogues(dialogue_files(source)):
            if limit and written >= limit:
                break
            destination.write(
                json.dumps(extract_dialogue(dialogue), ensure_ascii=False) + "\n"
            )
            written += 1

    if written == 0:
        output.unlink(missing_ok=True)
        raise ValueError("No dialogues were extracted.")
    return written


def main() -> int:
    args = parse_args()
    try:
        written = process(args.source, args.output, args.limit, args.overwrite)
    except (FileNotFoundError, FileExistsError, ValueError, OSError, json.JSONDecodeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Created: {args.output.resolve()}")
    print(f"Dialogues: {written}")
    print("Fields: scene, dialogue")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
