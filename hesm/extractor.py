from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .prompts.topic_memory import (
    DEFAULT_EXTRACTOR_PROMPT_PATH,
    build_extractor_prompt as prompt_build_extractor_prompt,
    load_extractor_prompt_template,
)

from .config import get as config_get, get_config_path
from .time_utils import format_timestamp


RESULT_FIELDS = (
    "topic",
    "core_entity",
    "intent",
    "entities",
    "confidence",
    "reasoning",
)
REQUIRED_RESULT_FIELDS = set(RESULT_FIELDS)
DEFAULT_PROMPT_PATH = DEFAULT_EXTRACTOR_PROMPT_PATH

TopicRecord = dict[str, Any]
TopicResult = TopicRecord | list[TopicRecord]


def load_prompt_template(path: str | Path | None = None) -> str:
    return load_extractor_prompt_template(path or DEFAULT_PROMPT_PATH)


def build_extractor_prompt(
    user_input: str,
    conversation_context: str = "",
    domain_knowledge: str = "",
    *,
    prompt_path: str | Path | None = None,
) -> str:
    return prompt_build_extractor_prompt(
        user_input=user_input,
        conversation_context=conversation_context,
        domain_knowledge=domain_knowledge,
        prompt_path=prompt_path or DEFAULT_PROMPT_PATH,
    )


def strip_markdown_code_fence(content: str) -> str:
    text = content.strip()
    if not text.startswith("```"):
        return text

    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_topic_response(content: str) -> TopicResult:
    payload = json.loads(strip_markdown_code_fence(content))

    if isinstance(payload, list):
        if not payload:
            raise ValueError("Topic response array must not be empty.")
        if not all(isinstance(item, dict) for item in payload):
            raise ValueError(
                "Every topic record in the response array must be a JSON object."
            )
        return payload[0] if len(payload) == 1 else payload

    if not isinstance(payload, dict):
        raise ValueError("Topic response must be a JSON object or a JSON array.")

    return payload


def validate_topic_record(value: Any) -> TopicRecord:
    if not isinstance(value, dict):
        raise ValueError("Topic result must be a JSON object.")
    missing = REQUIRED_RESULT_FIELDS - value.keys()
    if missing:
        raise ValueError(f"Topic result is missing fields: {sorted(missing)}")

    result = {key: value[key] for key in RESULT_FIELDS}
    for key in ("topic", "core_entity", "intent", "reasoning"):
        if not isinstance(result[key], str) or not result[key].strip():
            raise ValueError(f"{key} must be a non-empty string.")
        result[key] = result[key].strip()

    entities = result["entities"]
    if not isinstance(entities, list) or not entities:
        raise ValueError("entities must be a non-empty list.")
    result["entities"] = [str(entity).strip() for entity in entities if str(entity).strip()]
    if not result["entities"]:
        raise ValueError("entities must contain at least one non-empty value.")

    confidence = result["confidence"]
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise ValueError("confidence must be numeric.")
    if not 0.0 <= float(confidence) <= 1.0:
        raise ValueError("confidence must be between 0.0 and 1.0.")
    result["confidence"] = float(confidence)
    return result


def validate_topic_result(value: Any) -> TopicResult:
    if isinstance(value, list):
        if not value:
            raise ValueError("Topic result list must not be empty.")
        return [validate_topic_record(item) for item in value]
    return validate_topic_record(value)


def add_result_metadata(topic_result: TopicResult, user_input: str) -> TopicResult:
    timestamp = format_timestamp()
    if isinstance(topic_result, list):
        return [
            {
                **item,
                "user_input": user_input,
                "timestamp": timestamp,
            }
            for item in topic_result
        ]

    return {
        **topic_result,
        "user_input": user_input,
        "timestamp": timestamp,
    }


class TopicExtractor:
    """OpenAI-compatible topic extractor."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        max_retries: int | None = None,
        retry_delay: float | None = None,
        prompt_path: str | Path | None = None,
    ) -> None:
        from openai import OpenAI

        api_key = api_key or config_get("topic_extraction", "api_key")
        if not api_key:
            raise ValueError(f"Set topic_extraction.api_key in {get_config_path()}.")

        model = model or config_get("topic_extraction", "model")
        base_url = base_url or config_get("topic_extraction", "base_url")
        if not model:
            raise ValueError(f"Set topic_extraction.model in {get_config_path()}.")
        if not base_url:
            raise ValueError(f"Set topic_extraction.base_url in {get_config_path()}.")
        max_retries = max_retries if max_retries is not None else int(
            config_get("topic_extraction", "max_retries", 3)
        )
        retry_delay = retry_delay if retry_delay is not None else float(
            config_get("topic_extraction", "retry_delay", 2.0)
        )

        self.client = OpenAI(api_key=str(api_key), base_url=str(base_url))
        self.model = str(model)
        self.max_retries = int(max_retries)
        self.retry_delay = float(retry_delay)
        self.prompt_path = Path(prompt_path).expanduser() if prompt_path else DEFAULT_PROMPT_PATH

    def extract(
        self,
        user_input: str,
        context: str = "",
        domain_knowledge: str = "",
    ) -> TopicResult:
        prompt = build_extractor_prompt(
            user_input,
            context,
            domain_knowledge,
            prompt_path=self.prompt_path,
        )
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                content = (response.choices[0].message.content or "").strip()
                return validate_topic_result(parse_topic_response(content))
            except Exception as error:
                last_error = error
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)
        raise RuntimeError(
            f"Topic extraction failed after {self.max_retries} attempts: {last_error}"
        ) from last_error


def run_entity_extract(
    user_input: str,
    ctx: str = "",
    kg: str = "",
    *,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> TopicResult:
    extractor = TopicExtractor(
        api_key=api_key,
        model=model,
        base_url=base_url,
    )
    return extractor.extract(user_input=user_input, context=ctx, domain_knowledge=kg)


__all__ = [
    "TopicExtractor",
    "TopicRecord",
    "TopicResult",
    "add_result_metadata",
    "build_extractor_prompt",
    "load_prompt_template",
    "parse_topic_response",
    "run_entity_extract",
    "strip_markdown_code_fence",
    "validate_topic_record",
    "validate_topic_result",
]
