#!/usr/bin/env python3
"""Extract topics for MultiWOZ user turns and write them back into the dialogue JSONL."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Protocol

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory.config import config_path, config_value
from memory.extractor import (
    TopicRecord,
    TopicExtractor,
    TopicResult,
    validate_topic_result,
)


logger = logging.getLogger("memory.multiwoz_topic_pipeline")

DEFAULT_INPUT = config_path("paths", "processed_dialogues")
DEFAULT_OUTPUT = config_path("paths", "topic_output")
DEFAULT_CHECKPOINT = config_path("paths", "topic_checkpoint")
DEFAULT_FAILURES = config_path("paths", "topic_failures")


class ExtractorProtocol(Protocol):
    def extract(self, user_input: str, context: str) -> TopicResult:
        """Extract one or more structured topic results."""


@dataclass
class SemanticContext:
    history_size: int = 5
    current_topic_state: TopicResult = field(default_factory=dict)
    recent_semantic_history: deque[TopicResult] = field(init=False)

    def __post_init__(self) -> None:
        if self.history_size <= 0:
            raise ValueError("history_size must be greater than zero.")
        self.recent_semantic_history = deque(maxlen=self.history_size)

    def to_prompt_json(self) -> str:
        return json.dumps(
            {
                "current_topic_state": self.current_topic_state,
                "recent_semantic_history": list(self.recent_semantic_history),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def update(self, semantic_result: TopicResult) -> None:
        state = copy.deepcopy(semantic_result)
        self.recent_semantic_history.append(state)
        self.current_topic_state = state

    def to_dict(self) -> dict[str, Any]:
        return {
            "history_size": self.history_size,
            "current_topic_state": self.current_topic_state,
            "recent_semantic_history": list(self.recent_semantic_history),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SemanticContext":
        context = cls(history_size=int(value.get("history_size", 5)))
        context.current_topic_state = copy.deepcopy(value.get("current_topic_state") or {})
        context.recent_semantic_history.extend(
            copy.deepcopy(value.get("recent_semantic_history") or [])
        )
        return context


@dataclass
class Checkpoint:
    input_path: str
    next_dialogue_index: int = 0
    next_turn_index: int = 0
    processed_user_turns: int = 0
    context: dict[str, Any] = field(default_factory=dict)
    current_dialogue: dict[str, Any] | None = None
    output_has_partial_dialogue: bool = False


def iter_dialogues(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as source:
        for dialogue_index, line in enumerate(source):
            if line.strip():
                dialogue = json.loads(line)
                if not isinstance(dialogue, dict):
                    raise ValueError(f"Dialogue {dialogue_index} is not a JSON object.")
                yield dialogue_index, dialogue


def input_stats(path: Path) -> tuple[int, int]:
    dialogues = 0
    user_turns = 0
    for _, dialogue in iter_dialogues(path):
        dialogues += 1
        user_turns += sum(
            turn.get("role") == "user" and bool(str(turn.get("content") or "").strip())
            for turn in dialogue.get("dialogue") or []
        )
    return dialogues, user_turns


def semantic_state(
    result: TopicResult, user_input: str, turn_index: int
) -> TopicResult:
    timestamp = datetime.now(timezone.utc).isoformat()
    if isinstance(result, list):
        return [
            {
                **item,
                "user_input": user_input,
                "turn_index": turn_index,
                "timestamp": timestamp,
            }
            for item in result
        ]
    return {
        **result,
        "user_input": user_input,
        "turn_index": turn_index,
        "timestamp": timestamp,
    }


def primary_topic_record(result: TopicResult) -> TopicRecord:
    return result[0] if isinstance(result, list) else result


def append_json_line(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as destination:
        destination.write(json.dumps(value, ensure_ascii=False) + "\n")
        destination.flush()


def remove_failure_entry(path: Path, dialogue_index: int, turn_index: int) -> None:
    if not path.exists():
        return

    remaining_lines = []
    removed = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            remaining_lines.append(line)
            continue

        if (
            record.get("dialogue_index") == dialogue_index
            and record.get("turn_index") == turn_index
        ):
            removed = True
            continue
        remaining_lines.append(line)

    if not removed:
        return

    if not remaining_lines:
        path.unlink(missing_ok=True)
        return

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(remaining_lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def remove_last_json_line(path: Path) -> None:
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    temporary = path.with_suffix(path.suffix + ".tmp")
    content = "\n".join(lines[:-1])
    temporary.write_text(content + ("\n" if content else ""), encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(path: Path, checkpoint: Checkpoint) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(asdict(checkpoint), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def checkpoint_matches_input(checkpoint_input: str, input_path: Path) -> bool:
    current = str(input_path.resolve())
    if checkpoint_input == current:
        return True
    try:
        previous_path = Path(checkpoint_input)
    except (OSError, ValueError):
        return False
    return previous_path.name == input_path.name


def load_checkpoint(path: Path, input_path: Path) -> Checkpoint:
    if not path.exists():
        return Checkpoint(input_path=str(input_path.resolve()))
    checkpoint = Checkpoint(**json.loads(path.read_text(encoding="utf-8")))
    current_input = str(input_path.resolve())
    if not checkpoint_matches_input(checkpoint.input_path, input_path):
        raise ValueError("Checkpoint belongs to a different input file.")
    if checkpoint.input_path != current_input:
        logger.warning(
            "Checkpoint input path moved from %s to %s; continuing from saved progress.",
            checkpoint.input_path,
            current_input,
        )
        checkpoint.input_path = current_input
    return checkpoint


def prepare_resume(checkpoint: Checkpoint, output_path: Path, checkpoint_path: Path) -> None:
    if checkpoint.output_has_partial_dialogue:
        logger.info("检测到部分对话输出，续跑前移除最后一行以原位补充")
        remove_last_json_line(output_path)
        checkpoint.output_has_partial_dialogue = False
        save_checkpoint(checkpoint_path, checkpoint)


def persist_partial_dialogue(
    checkpoint: Checkpoint, output_path: Path, checkpoint_path: Path
) -> None:
    if checkpoint.current_dialogue is not None:
        append_json_line(output_path, checkpoint.current_dialogue)
        checkpoint.output_has_partial_dialogue = True
    save_checkpoint(checkpoint_path, checkpoint)


def process_dialogues(
    extractor: ExtractorProtocol,
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    failures_path: Path,
    max_user_turns: int = 1000,
    history_size: int = 5,
    resume: bool = True,
    continue_on_error: bool = False,
    progress_every: int = 10,
) -> int:
    if max_user_turns < 0:
        raise ValueError("max_user_turns must be zero or greater.")
    if progress_every <= 0:
        raise ValueError("progress_every must be greater than zero.")
    if not resume:
        for path in (output_path, checkpoint_path, failures_path):
            path.unlink(missing_ok=True)
    checkpoint = (
        load_checkpoint(checkpoint_path, input_path)
        if resume
        else Checkpoint(input_path=str(input_path.resolve()))
    )
    total_dialogues, total_user_turns = input_stats(input_path)
    started_at = time.monotonic()
    run_limit = max_user_turns or total_user_turns
    logger.info(
        "开始主题提取 input=%s dialogues=%s user_turns=%s run_limit=%s resume=%s",
        input_path.resolve(),
        total_dialogues,
        total_user_turns,
        run_limit,
        resume,
    )
    logger.info(
        "当前断点 dialogue=%s turn=%s processed_total=%s",
        checkpoint.next_dialogue_index,
        checkpoint.next_turn_index,
        checkpoint.processed_user_turns,
    )
    prepare_resume(checkpoint, output_path, checkpoint_path)

    processed_this_run = 0
    for dialogue_index, source_dialogue in iter_dialogues(input_path):
        if dialogue_index < checkpoint.next_dialogue_index:
            continue

        is_resumed_dialogue = (
            dialogue_index == checkpoint.next_dialogue_index
            and checkpoint.current_dialogue is not None
        )
        dialogue = (
            copy.deepcopy(checkpoint.current_dialogue)
            if is_resumed_dialogue
            else copy.deepcopy(source_dialogue)
        )
        context = (
            SemanticContext.from_dict(checkpoint.context)
            if is_resumed_dialogue
            else SemanticContext(history_size=history_size)
        )
        start_turn_index = checkpoint.next_turn_index if is_resumed_dialogue else 0

        for turn_index, turn in enumerate(dialogue.get("dialogue") or []):
            if turn_index < start_turn_index or turn.get("role") != "user":
                continue
            user_input = " ".join(str(turn.get("content") or "").split())
            if not user_input:
                continue
            if max_user_turns and processed_this_run >= max_user_turns:
                persist_partial_dialogue(checkpoint, output_path, checkpoint_path)
                elapsed = time.monotonic() - started_at
                logger.info(
                    "达到本次处理上限，已保存部分对话 processed_run=%s processed_total=%s elapsed=%.1fs",
                    processed_this_run,
                    checkpoint.processed_user_turns,
                    elapsed,
                )
                return processed_this_run

            try:
                result = validate_topic_result(
                    extractor.extract(user_input=user_input, context=context.to_prompt_json())
                )
                turn["topic_extraction"] = result
                context.update(semantic_state(result, user_input, turn_index))
                remove_failure_entry(failures_path, dialogue_index, turn_index)
                processed_this_run += 1
                checkpoint.processed_user_turns += 1
                if processed_this_run == 1 or processed_this_run % progress_every == 0:
                    elapsed = time.monotonic() - started_at
                    primary_result = primary_topic_record(result)
                    topic_count = len(result) if isinstance(result, list) else 1
                    overall_percentage = (
                        checkpoint.processed_user_turns / total_user_turns * 100
                        if total_user_turns
                        else 0.0
                    )
                    logger.info(
                        "进度 run=%s/%s total=%s/%s(%.2f%%) dialogue=%s turn=%s "
                        "topics=%s primary_topic=%s primary_core_entity=%s elapsed=%.1fs",
                        processed_this_run,
                        run_limit,
                        checkpoint.processed_user_turns,
                        total_user_turns,
                        overall_percentage,
                        dialogue_index,
                        turn_index,
                        topic_count,
                        primary_result["topic"],
                        primary_result["core_entity"],
                        elapsed,
                    )
            except Exception as error:
                logger.error(
                    "主题提取失败 dialogue=%s turn=%s user_input=%r error=%s",
                    dialogue_index,
                    turn_index,
                    user_input,
                    error,
                )
                append_json_line(
                    failures_path,
                    {
                        "dialogue_index": dialogue_index,
                        "turn_index": turn_index,
                        "user_input": user_input,
                        "error": str(error),
                    },
                )
                if not continue_on_error:
                    checkpoint.next_dialogue_index = dialogue_index
                    checkpoint.next_turn_index = turn_index
                    checkpoint.context = context.to_dict()
                    checkpoint.current_dialogue = dialogue
                    persist_partial_dialogue(checkpoint, output_path, checkpoint_path)
                    raise

            checkpoint.next_dialogue_index = dialogue_index
            checkpoint.next_turn_index = turn_index + 1
            checkpoint.context = context.to_dict()
            checkpoint.current_dialogue = dialogue
            save_checkpoint(checkpoint_path, checkpoint)

        append_json_line(output_path, dialogue)
        checkpoint.next_dialogue_index = dialogue_index + 1
        checkpoint.next_turn_index = 0
        checkpoint.context = {}
        checkpoint.current_dialogue = None
        checkpoint.output_has_partial_dialogue = False
        save_checkpoint(checkpoint_path, checkpoint)

    elapsed = time.monotonic() - started_at
    logger.info(
        "全部输入处理完成 processed_run=%s processed_total=%s elapsed=%.1fs output=%s",
        processed_this_run,
        checkpoint.processed_user_turns,
        elapsed,
        output_path.resolve(),
    )
    return processed_this_run


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--failures", type=Path, default=DEFAULT_FAILURES)
    parser.add_argument(
        "--max-user-turns",
        type=int,
        default=1000,
        help="Maximum user turns for this run. Use 0 for no limit.",
    )
    parser.add_argument("--history-size", type=int, default=5)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--max-retries", type=int, default=int(config_value("topic_extraction", "max_retries", 3)))
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress after this many successfully processed user turns.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    try:
        processed = process_dialogues(
            extractor=TopicExtractor(max_retries=args.max_retries),
            input_path=args.input,
            output_path=args.output,
            checkpoint_path=args.checkpoint,
            failures_path=args.failures,
            max_user_turns=args.max_user_turns,
            history_size=args.history_size,
            resume=not args.no_resume,
            continue_on_error=args.continue_on_error,
            progress_every=args.progress_every,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(f"Processed user turns: {processed}")
    print(f"Output: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
