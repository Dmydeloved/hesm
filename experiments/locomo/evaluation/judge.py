"""
LLM-as-a-Judge evaluation.

Uses the topic_extraction LLM to score each predicted answer against the
ground truth on a 0 / 1 / 2 scale:
  0 = Wrong          (completely incorrect or irrelevant)
  1 = Partially Correct (some correct info but incomplete or with errors)
  2 = Correct        (accurate and complete)
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any

import openai

logger = logging.getLogger(__name__)

_JUDGE_PROMPT = """\
You are an expert evaluator assessing the quality of an AI assistant's answer.

Question: {question}
Ground Truth Answer: {ground_truth}
Predicted Answer: {prediction}

Score the predicted answer on the following scale:
0 = Wrong: The answer is completely incorrect, irrelevant, or says "Unknown" when the answer is in the ground truth.
1 = Partially Correct: The answer contains some correct information but is incomplete, imprecise, or contains errors.
2 = Correct: The answer is accurate and sufficiently complete relative to the ground truth.

Respond with ONLY the integer score (0, 1, or 2). Do not include any explanation."""


class LLMJudge:
    """
    Calls the LLM to score (question, ground_truth, prediction) triples.
    Reads API settings from the topic_extraction section of configs/config.yaml.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        te: dict[str, Any] = config.get("topic_extraction", {})
        api_key = te.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
        base_url = te.get("base_url", "https://api.openai.com/v1/")
        self.model: str = te.get("model", "gpt-4")
        self.max_retries: int = int(te.get("max_retries", 3))
        self.retry_delay: float = float(te.get("retry_delay", 2.0))
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def judge(
        self,
        question: str,
        ground_truth: str,
        prediction: str,
    ) -> int:
        """
        Score a single (question, ground_truth, prediction) triple.

        Returns 0, 1, or 2. Returns -1 on total failure (excluded from averages
        by the aggregator).
        """
        prompt = _JUDGE_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
            prediction=prediction,
        )
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=8,
                )
                raw = resp.choices[0].message.content.strip()
                score = self._parse_score(raw)
                if score is not None:
                    return score
                logger.warning("Judge returned unparseable response: %r", raw)
            except Exception as exc:
                logger.warning(
                    "Judge attempt %d/%d failed: %s", attempt, self.max_retries, exc
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)
        return -1

    @staticmethod
    def _parse_score(text: str) -> int | None:
        """Extract 0/1/2 from LLM response; returns None if not parseable."""
        m = re.search(r"\b([012])\b", text)
        if m:
            return int(m.group(1))
        return None
