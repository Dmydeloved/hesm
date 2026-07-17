import json
import tempfile
import unittest
from pathlib import Path

from experiments.datasets import load_dataset
from experiments.datasets.multiwoz import load_multiwoz
from experiments.metrics import evidence_recall, token_f1
from experiments.methods import build_method
from experiments.runner import build_topic_context, primary_topic_result, run


class ExperimentHarnessTests(unittest.TestCase):
    def test_multiwoz_loader_keeps_only_past_turns_as_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "multiwoz.jsonl"
            path.write_text(json.dumps(sample_dialogue(), ensure_ascii=False) + "\n", encoding="utf-8")
            examples = load_multiwoz(path)
        self.assertEqual(2, len(examples))
        self.assertEqual([], examples[0].evidence_user_inputs)
        self.assertEqual(["I need a cheap hotel."], examples[1].evidence_user_inputs)

    def test_multiwoz_loader_keeps_user_turns_without_precomputed_topics(self):
        dialogue = sample_dialogue()
        dialogue["dialogue"][2]["topic_extraction"] = None
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "multiwoz.jsonl"
            path.write_text(json.dumps(dialogue, ensure_ascii=False) + "\n", encoding="utf-8")
            examples = load_multiwoz(path)
        self.assertEqual(2, len(examples))
        self.assertEqual({}, examples[1].topic_result)
        self.assertFalse(examples[1].metadata["has_precomputed_topic"])

    def test_metrics_cover_text_and_evidence_retrieval(self):
        self.assertGreater(token_f1("cheap hotel", "cheap hotel please"), 0.0)
        recall = evidence_recall(
            [{"user_input": "I need a cheap hotel."}],
            ["I need a cheap hotel."],
        )
        self.assertEqual(1.0, recall)

    def test_runner_writes_reproducible_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "multiwoz.jsonl"
            data_path.write_text(json.dumps(sample_dialogue(), ensure_ascii=False) + "\n", encoding="utf-8")
            output_dir = root / "run"
            config_path = root / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "dataset:",
                        "  name: multiwoz",
                        f"  path: {data_path.as_posix()}",
                        "  limit: 0",
                        "method:",
                        "  name: bm25",
                        "  top_k: 3",
                        "run:",
                        f"  output_dir: {output_dir.as_posix()}",
                        "  seed: 7",
                    ]
                ),
                encoding="utf-8",
            )
            summary = run(config_path)
            self.assertTrue((output_dir / "predictions.jsonl").exists())
            self.assertTrue((output_dir / "metrics.json").exists())
            self.assertTrue((output_dir / "run.log").exists())
            self.assertTrue((output_dir / "node_inputs.jsonl").exists())
            self.assertEqual(2, summary["overall"]["cases"])

    def test_method_factory_builds_offline_baselines(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual("bm25", build_method("bm25", root, {}).name)
            self.assertEqual("dense_rag", build_method("dense_rag", root, {}).name)
            self.assertEqual("full_context", build_method("full_context", root, {}).name)

    def test_dataset_can_limit_by_dialogue(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "multiwoz.jsonl"
            path.write_text(
                json.dumps(sample_dialogue(), ensure_ascii=False)
                + "\n"
                + json.dumps(sample_dialogue(), ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
            examples = load_dataset("multiwoz", path, dialogue_limit=1)
        self.assertEqual(2, len(examples))
        self.assertEqual({"multiwoz-d00000"}, {item.dialogue_id for item in examples})

    def test_topic_context_keeps_recent_raw_io_and_topic_results(self):
        history = [
            {
                "case_id": str(index),
                "user_input": f"user {index}",
                "assistant_output": f"assistant {index}",
                "topic_extraction": {"topic": f"topic {index}"},
            }
            for index in range(6)
        ]
        payload = json.loads(build_topic_context(history, 5))
        self.assertEqual(5, len(payload["recent_turns"]))
        self.assertEqual("1", payload["recent_turns"][0]["case_id"])
        self.assertEqual("user 5", payload["recent_turns"][-1]["user_input"])
        self.assertEqual(
            {"topic": "topic 5"},
            payload["recent_turns"][-1]["topic_extraction"],
        )

    def test_primary_topic_result_accepts_single_or_list_payload(self):
        self.assertEqual({"topic": "x"}, primary_topic_result({"topic": "x"}))
        self.assertEqual({"topic": "x"}, primary_topic_result([{"topic": "x"}]))


def sample_dialogue():
    first_topic = {
        "topic": "hotel booking",
        "core_entity": "hotel",
        "intent": "find cheap hotel",
        "entities": ["hotel"],
        "confidence": 1.0,
        "reasoning": "test",
    }
    second_topic = {
        "topic": "hotel booking",
        "core_entity": "hotel",
        "intent": "find cheap hotel",
        "entities": ["hotel"],
        "confidence": 1.0,
        "reasoning": "test",
    }
    return {
        "scene": ["hotel"],
        "dialogue": [
            {"role": "user", "content": "I need a cheap hotel.", "topic_extraction": first_topic},
            {"role": "system", "content": "I can help find one."},
            {"role": "user", "content": "Does it need parking?", "topic_extraction": second_topic},
            {"role": "system", "content": "Parking is available."},
        ],
    }


if __name__ == "__main__":
    unittest.main()
