import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_topic_benchmark import build_benchmark
from scripts.evaluate_end_to_end import parse_judge_response, summarize as summarize_e2e, text_metrics
from scripts.evaluate_retrieval import summarize


def topic(topic_name, entity, intent):
    return {
        "topic": topic_name,
        "core_entity": entity,
        "intent": intent,
        "entities": [entity],
        "confidence": 0.9,
        "reasoning": "test",
    }


class EvaluationScriptTests(unittest.TestCase):
    def test_build_topic_benchmark_labels_switch_types(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "topics.jsonl"
            output_path = Path(directory) / "benchmark.jsonl"
            dialogue = {
                "scene": ["restaurant"],
                "dialogue": [
                    {
                        "role": "user",
                        "content": "first",
                        "topic_extraction": topic("restaurant", "restaurant", "query"),
                    },
                    {"role": "system", "content": "first answer"},
                    {
                        "role": "user",
                        "content": "second",
                        "topic_extraction": topic("restaurant", "restaurant", "query"),
                    },
                    {"role": "system", "content": "second answer"},
                    {
                        "role": "user",
                        "content": "third",
                        "topic_extraction": topic("restaurant", "restaurant", "book"),
                    },
                    {"role": "system", "content": "third answer"},
                    {
                        "role": "user",
                        "content": "fourth",
                        "topic_extraction": topic("hotel", "hotel", "query"),
                    },
                ],
            }
            input_path.write_text(
                json.dumps(dialogue, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            self.assertEqual(4, build_benchmark(input_path, output_path))
            cases = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [
                    "cold_start",
                    "same_experience_same_intent",
                    "same_experience_new_intent",
                    "topic_switch",
                ],
                [case["case_type"] for case in cases],
            )
            self.assertEqual("second answer", cases[1]["answer"])
            self.assertTrue(cases[1]["expected_cache"]["segment"])
            self.assertTrue(cases[2]["expected_cache"]["experience"])
            self.assertFalse(cases[2]["expected_cache"]["segment"])

    def test_retrieval_summary_computes_mem0_style_metrics(self):
        rows = [
            {
                "case_type": "cold_start",
                "experience_hit": True,
                "segment_hit": True,
                "qa_hit": True,
                "qa_rr": 1.0,
                "experience_cache_hit": False,
                "segment_cache_hit": False,
                "retrieved_tokens": 100,
                "latency_ms": 10.0,
            },
            {
                "case_type": "cold_start",
                "experience_hit": True,
                "segment_hit": False,
                "qa_hit": False,
                "qa_rr": 0.0,
                "experience_cache_hit": True,
                "segment_cache_hit": False,
                "retrieved_tokens": 200,
                "latency_ms": 20.0,
            },
        ]
        summary = summarize(rows)
        self.assertEqual(2, summary["overall"]["cases"])
        self.assertEqual(1.0, summary["overall"]["experience_hit"])
        self.assertEqual(0.5, summary["overall"]["qa_hit"])
        self.assertEqual(150.0, summary["overall"]["retrieved_tokens"])

    def test_text_metrics_compute_bleu1_and_f1(self):
        exact = text_metrics("The Golden Curry", "The Golden Curry")
        self.assertEqual(1.0, exact["bleu1"])
        self.assertEqual(1.0, exact["f1"])

        partial = text_metrics("Golden Curry phone", "Golden Curry address")
        self.assertGreater(partial["bleu1"], 0.0)
        self.assertGreater(partial["f1"], 0.0)
        self.assertLess(partial["f1"], 1.0)

        missing = text_metrics("", "Golden Curry")
        self.assertEqual(0.0, missing["bleu1"])
        self.assertEqual(0.0, missing["f1"])

    def test_end_to_end_summary_includes_text_metrics(self):
        summary = summarize_e2e([
            {
                "judge": {"score": 0.5},
                "metrics": {"bleu1": 0.25, "f1": 0.5},
                "latency_ms": 10.0,
            },
            {
                "judge": {"score": 1.0},
                "metrics": {"bleu1": 0.75, "f1": 1.0},
                "latency_ms": 20.0,
            },
        ])
        self.assertEqual(0.5, summary["bleu1"])
        self.assertEqual(0.75, summary["f1"])

    def test_parse_judge_response_requires_json_score(self):
        parsed = parse_judge_response('{"score": 0.75, "reason": "mostly correct"}')
        self.assertEqual(0.75, parsed["score"])
        with self.assertRaises(ValueError):
            parse_judge_response('{"score": 2, "reason": "bad"}')


if __name__ == "__main__":
    unittest.main()
