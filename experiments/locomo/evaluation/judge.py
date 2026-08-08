"""
LLM-as-a-Judge evaluation.

Uses the evaluation LLM to score each predicted answer against the ground
truth on a binary 0 / 1 scale:
  0 = WRONG   (factually incorrect, missing key info, extra unsupported facts,
               wrong temporal/entity info, or unjustified inferences)
  1 = CORRECT (factually accurate, all essential facts present, paraphrases ok)

The LLM returns a JSON object:
  {"reason": "...", "label": "CORRECT" | "WRONG"}
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

import openai

logger = logging.getLogger(__name__)

_JUDGE_PROMPT = """\
You are an impartial evaluator for long-term memory question answering.

You will receive:

1. Question
2. Ground Truth
3. Predicted Answer

Evaluation Rules:

- Compare the predicted answer with the ground truth based on factual correctness.
- Ignore wording differences and paraphrases.
- The predicted answer must contain all essential facts required to answer the question.
- Missing key information should be judged as WRONG.
- Additional unsupported facts should be judged as WRONG.
- Temporal information and named entities must be correct.
- Do not infer facts that are not explicitly stated.

Question: {question}
Ground Truth: {ground_truth}
Predicted Answer: {prediction}

Output JSON only:

{{
  "reason": "...",
  "label": "CORRECT" | "WRONG"
}}"""


class LLMJudge:
    """
    Calls the LLM to score (question, ground_truth, prediction) triples.
    Reads API settings from the evaluation section of configs/config.yaml.

    Missing evaluation fields fall back to the legacy judge-prefixed keys and
    then topic_extraction for compatibility with older configuration files.

    Returns 1 (CORRECT) or 0 (WRONG). Returns -1 on total failure (excluded
    from averages by the aggregator).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        te: dict[str, Any] = config.get("topic_extraction", {})
        evaluation: dict[str, Any] = config.get("evaluation", {})
        api_key = (
            evaluation.get("api_key")
            or evaluation.get("judge_api_key")
            or te.get("api_key")
            or os.environ.get("OPENAI_API_KEY", "")
        )
        base_url = evaluation.get(
            "base_url",
            evaluation.get(
                "judge_base_url",
                te.get("base_url", "https://api.openai.com/v1/"),
            ),
        )
        self.model: str = evaluation.get(
            "model", evaluation.get("judge_model", te.get("model", "gpt-4"))
        )
        self.max_retries: int = int(
            evaluation.get(
                "max_retries",
                evaluation.get("judge_max_retries", te.get("max_retries", 3)),
            )
        )
        self.retry_delay: float = float(
            evaluation.get(
                "retry_delay",
                evaluation.get("judge_retry_delay", te.get("retry_delay", 2.0)),
            )
        )
        self.last_error: str | None = None
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def judge(
        self,
        question: str,
        ground_truth: str,
        prediction: str,
    ) -> int:
        """
        Score a single (question, ground_truth, prediction) triple.

        Returns 1 (CORRECT) or 0 (WRONG).
        Returns -1 on total failure (excluded from averages by the aggregator).
        """
        self.last_error = None
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
                    max_tokens=200,
                )
                raw = resp.choices[0].message.content.strip()
                score = self._parse_score(raw)
                if score is not None:
                    self.last_error = None
                    return score
                self.last_error = f"Unparseable Judge response: {raw[:500]}"
                logger.warning("Judge returned unparseable response: %r", raw)
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "Judge attempt %d/%d failed: %s", attempt, self.max_retries, exc
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)
        if self.last_error is None:
            self.last_error = "Judge failed without an error message"
        return -1

    @staticmethod
    def _parse_score(text: str) -> int | None:
        """
        Parse CORRECT/WRONG label from LLM JSON response.

        Expected format:  {"reason": "...", "label": "CORRECT"}
        Fallback:         plain text containing CORRECT or WRONG keyword.

        Returns 1 for CORRECT, 0 for WRONG, None if unparseable.
        """
        # Primary: parse as JSON and extract "label"
        try:
            # Strip markdown code fence if present
            cleaned = re.sub(r"^```[a-z]*\n?", "", text.strip(), flags=re.IGNORECASE)
            cleaned = re.sub(r"\n?```$", "", cleaned.strip())
            obj = json.loads(cleaned)
            label = str(obj.get("label", "")).strip().upper()
            if label == "CORRECT":
                return 1
            if label == "WRONG":
                return 0
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass

        # Fallback: keyword scan in raw text
        upper = text.upper()
        if "CORRECT" in upper:
            return 1
        if "WRONG" in upper:
            return 0
        return None
