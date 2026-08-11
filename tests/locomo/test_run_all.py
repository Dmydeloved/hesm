from __future__ import annotations

import unittest
from unittest.mock import patch

from experiments.locomo.run_ablation import _ALL_VARIANTS
from experiments.locomo.run_all import _RUN_ALL_METHODS, run_all


class RunAllSelectionTests(unittest.TestCase):
    def test_full_context_is_not_a_run_all_method(self) -> None:
        self.assertNotIn("full_context", _RUN_ALL_METHODS)

    @patch("experiments.locomo.run_main.run_main")
    def test_programmatic_full_context_request_is_filtered(self, run_main) -> None:
        run_all(
            enabled_methods=["full_context", "hesm"],
            skip_parts=["ablation", "cache"],
            max_conversations=1,
        )
        run_main.assert_called_once_with(
            config_path=None,
            enabled_methods=["hesm"],
            max_conversations=1,
            method_workers=None,
            qa_workers=None,
        )

    def test_complete_ablation_set_contains_no_reranker_variant(self) -> None:
        self.assertEqual(
            _ALL_VARIANTS,
            [
                "flat_memory",
                "qa_only",
                "qa_segment",
                "full_hesm_no_reranker",
                "full_hesm",
            ],
        )


if __name__ == "__main__":
    unittest.main()
