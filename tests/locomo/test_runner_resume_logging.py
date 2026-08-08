from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.locomo.data.loader import (
    LoCoMoConversation,
    QAItem,
    Session,
    Turn,
)
from experiments.locomo.methods.base import RetrievalResult
from experiments.locomo.runner.checkpoint import Checkpoint
from experiments.locomo.runner.qa_runner import QARunner


class _Method:
    method_name = "fake_method"

    def __init__(self) -> None:
        self.build_calls = 0
        self.retrieve_calls = 0

    def reset(self) -> None:
        return None

    def build_memory(self, conv_id, sessions, speaker_a, speaker_b) -> None:
        self.build_calls += 1

    def retrieve(self, question: str, top_k: int = 5) -> RetrievalResult:
        self.retrieve_calls += 1
        return RetrievalResult("[A]: answer", ["D1:1"], 3, {})


class _Answer:
    last_error = None

    def generate(self, **kwargs) -> str:
        return "answer"


class _BuildFailMethod(_Method):
    method_name = "build_fail_method"

    def build_memory(self, conv_id, sessions, speaker_a, speaker_b) -> None:
        raise ValueError("broken source data")


class _Judge:
    def __init__(self, score: int) -> None:
        self.score = score
        self.last_error = "judge unavailable" if score < 0 else None

    def judge(self, **kwargs) -> int:
        return self.score


def _conversation() -> LoCoMoConversation:
    turn = Turn("A", "answer", "D1:1", 1, "2026-01-01")
    return LoCoMoConversation(
        conv_id="conv-test",
        speaker_a="A",
        speaker_b="B",
        sessions=[Session(1, "2026-01-01", [turn])],
        questions=[QAItem("What is the answer?", "answer", ["D1:1"], 4)],
        _turn_index={"D1:1": turn},
    )


def _runner(tmp_path, method, judge) -> QARunner:
    return QARunner(
        method=method,
        answer_generator=_Answer(),
        judge=judge,
        output_dir=tmp_path / "answers",
        metrics_dir=tmp_path / "metrics",
        logs_dir=tmp_path / "logs",
    )


class RunnerResumeLoggingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_successful_query_is_skipped_on_resume(self) -> None:
        conv = _conversation()
        first_method = _Method()
        _runner(self.root, first_method, _Judge(1)).run([conv])
        self.assertEqual(first_method.build_calls, 1)
        self.assertEqual(first_method.retrieve_calls, 1)

        resumed_method = _Method()
        _runner(self.root, resumed_method, _Judge(1)).run([conv])
        self.assertEqual(resumed_method.build_calls, 0)
        self.assertEqual(resumed_method.retrieve_calls, 0)

        log_path = self.root / "logs" / "fake_method.log"
        events = [json.loads(line) for line in log_path.read_text("utf-8").splitlines()]
        self.assertEqual(
            {event["stage"] for event in events},
            {"BUILD", "RETRIEVAL", "ANSWER", "JUDGE"},
        )
        self.assertTrue(any(event["status"] == "SKIPPED" for event in events))

    def test_failed_judge_is_retried_and_not_marked_complete(self) -> None:
        conv = _conversation()
        failed_method = _Method()
        _runner(self.root, failed_method, _Judge(-1)).run([conv])

        failed_checkpoint = Checkpoint(
            self.root / "answers", "fake_method", "conv-test"
        )
        self.assertFalse(failed_checkpoint.is_complete())
        self.assertFalse(
            failed_checkpoint.is_answered(0, "What is the answer?")
        )
        failed_record = failed_checkpoint.load_all_records_ordered(1)[0]
        self.assertEqual(failed_record["query_status"], "failed")
        self.assertEqual(failed_record["stage_status"]["judge"]["status"], "failed")
        log_events = [
            json.loads(line)
            for line in (self.root / "logs" / "fake_method.log")
            .read_text("utf-8")
            .splitlines()
        ]
        self.assertTrue(
            any(
                event["stage"] == "JUDGE"
                and event["status"] == "FAILED"
                and "judge unavailable" in event["error"]
                for event in log_events
            )
        )

        retry_method = _Method()
        _runner(self.root, retry_method, _Judge(1)).run([conv])
        self.assertEqual(retry_method.retrieve_calls, 1)

        completed_checkpoint = Checkpoint(
            self.root / "answers", "fake_method", "conv-test"
        )
        self.assertTrue(completed_checkpoint.is_complete())
        self.assertTrue(
            completed_checkpoint.is_answered(0, "What is the answer?")
        )

    def test_legacy_checkpoint_requires_nonempty_answer_and_valid_judge(self) -> None:
        answers_dir = self.root / "answers"
        checkpoint = Checkpoint(answers_dir, "legacy", "conv-test")
        checkpoint.save_record(
            0,
            {"question": "q", "prediction": "", "judge_score": -1},
            total_questions=1,
        )

        reloaded = Checkpoint(answers_dir, "legacy", "conv-test")
        self.assertFalse(reloaded.is_answered(0, "q"))

    def test_build_failure_logs_reason_and_skips_downstream_stages(self) -> None:
        _runner(self.root, _BuildFailMethod(), _Judge(1)).run([_conversation()])
        events = [
            json.loads(line)
            for line in (self.root / "logs" / "build_fail_method.log")
            .read_text("utf-8")
            .splitlines()
        ]
        self.assertTrue(
            any(
                event["stage"] == "BUILD"
                and event["status"] == "FAILED"
                and "broken source data" in event["error"]
                for event in events
            )
        )
        self.assertEqual(
            {
                event["stage"]
                for event in events
                if event["status"] == "SKIPPED"
            },
            {"RETRIEVAL", "ANSWER", "JUDGE"},
        )


if __name__ == "__main__":
    unittest.main()
