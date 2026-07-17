from __future__ import annotations

import json
import os
from typing import Any

from memory.config import get as config_get
from memory.retriever import strip_markdown_code_fence


class OpenAICompatibleChat:
    def __init__(
        self,
        section: str,
        api_key: str | None = None,
        api_key_env: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        from openai import OpenAI

        resolved_key = api_key or (os.getenv(api_key_env) if api_key_env else None)
        resolved_key = resolved_key or config_get(section, "api_key")
        if not resolved_key:
            raise ValueError(
                f"Set {section}.api_key in configs/config.yaml or provide api_key_env."
            )
        self.model = str(model or config_get(section, "model"))
        if not self.model:
            raise ValueError(f"Set {section}.model in configs/config.yaml or config.")
        self.base_url = str(base_url or config_get(section, "base_url"))
        if not self.base_url:
            raise ValueError(f"Set {section}.base_url in configs/config.yaml or config.")
        self.temperature = float(temperature)
        self.client = OpenAI(api_key=str(resolved_key), base_url=self.base_url)

    def complete(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("LLM returned empty content.")
        return content.strip()


def answer_prompt(question: str, context_text: str) -> str:
    return f"""You are a long-term memory QA assistant. Answer only from the memory context.

[Memory Context]
{context_text}

[Question]
{question}

Return a concise answer. If the context is insufficient, say that the memory does not contain enough information."""


def judge_prompt(question: str, reference_answer: str, model_answer: str) -> str:
    return f"""You are a strict QA evaluator. Compare the model answer with the reference answer.

Return JSON only:
{{"score": a number from 0.0 to 1.0, "reason": "brief reason"}}

[Question]
{question}

[Reference Answer]
{reference_answer}

[Model Answer]
{model_answer}
"""


def parse_judge_response(content: str) -> dict[str, Any]:
    payload = json.loads(strip_markdown_code_fence(content))
    if not isinstance(payload, dict):
        raise ValueError("Judge response must be a JSON object.")
    score = float(payload["score"])
    if not 0.0 <= score <= 1.0:
        raise ValueError("Judge score must be between 0 and 1.")
    return {
        "score": score,
        "reason": str(payload.get("reason") or ""),
    }
