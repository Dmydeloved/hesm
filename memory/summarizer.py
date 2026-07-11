from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Protocol

from prompts.topic_memory import (
    DEFAULT_EXPERIENCE_SUMMARY_PROMPT_PATH,
    DEFAULT_SEGMENT_SUMMARY_PROMPT_PATH,
    build_experience_summary_prompt,
    build_segment_summary_prompt,
)

from .config import get as config_get

from collections import Counter


class SummarizerProtocol(Protocol):
    def summarize_segment(self, segment: dict[str, Any], qa_items: list[dict[str, Any]]) -> str:
        """Summarize one segment from its QA evidence."""

    def summarize_experience(
        self, experience: dict[str, Any], segments: list[dict[str, Any]]
    ) -> str:
        """Summarize one experience from its segment evidence."""


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


def parse_summary_response(content: str) -> str:
    text = strip_markdown_code_fence(content).strip()
    if not text:
        return ""

    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text

    if isinstance(payload, dict):
        for key in ("summary", "result", "content"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(payload, str):
        return payload.strip()
    return text


class TemplateSummarizer:
    """Deterministic summaries for the MVP memory system.

    这里先不调用 LLM，保证结构化记忆库可离线、可重复、低成本运行。
    后续可以把这两个方法替换成模型总结。
    """

    def summarize_segment(self, segment: dict[str, Any], qa_items: list[dict[str, Any]]) -> str:
        entities = []
        for qa in qa_items:
            entities.extend(qa.get("entities") or [])
        top_entities = [name for name, _ in Counter(entities).most_common(8)]
        return (
            f"该片段围绕 {segment['topic']} / {segment['core_entity']} 展开，"
            f"意图为 {segment['intent']}，累计 {len(qa_items)} 条 QA，"
            f"涉及实体：{', '.join(top_entities) if top_entities else '无'}。"
        )

    def summarize_experience(
        self, experience: dict[str, Any], segments: list[dict[str, Any]]
    ) -> str:
        intents = ", ".join(experience.get("intents_link") or [])
        latest_summary = segments[-1]["summary"] if segments and segments[-1].get("summary") else ""
        summary = f"用户持续围绕 {experience['topic']} / {experience['core_entity']} 进行交互。"
        summary += (
            f"该 Experience 当前累计 {len(segments)} 个 Segment，"
            f"涉及意图：{intents or '无'}。"
            f"{'最近片段：' + latest_summary if latest_summary else ''}"
        )
        return summary


class LLMSummarizer:
    """OpenAI-compatible summarizer for Segment and Experience memory."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        max_retries: int | None = None,
        retry_delay: float | None = None,
        client: Any | None = None,
        segment_prompt_path: str | Path | None = None,
        experience_prompt_path: str | Path | None = None,
    ) -> None:
        api_key = (
            api_key
            or config_get("summarization", "api_key")
        )
        model = (
            model
            or config_get("summarization", "model")
        )
        base_url = (
            base_url
            or config_get("summarization", "base_url")
        )
        max_retries = max_retries if max_retries is not None else int(
            config_get(
                "summarization",
                "max_retries",
                3,
            )
        )
        retry_delay = retry_delay if retry_delay is not None else float(
            config_get(
                "summarization",
                "retry_delay",
                2.0,
            )
        )

        if client is None:
            from openai import OpenAI

            if not api_key:
                raise ValueError("Set summarization.api_key in configs/config.yaml.")
            if not model:
                raise ValueError("Set summarization.model in configs/config.yaml.")
            if not base_url:
                raise ValueError("Set summarization.base_url in configs/config.yaml.")
            client = OpenAI(api_key=str(api_key), base_url=str(base_url))

        self.client = client
        self.model = str(model)
        self.max_retries = int(max_retries)
        self.retry_delay = float(retry_delay)
        self.segment_prompt_path = (
            Path(segment_prompt_path).expanduser()
            if segment_prompt_path
            else DEFAULT_SEGMENT_SUMMARY_PROMPT_PATH
        )
        self.experience_prompt_path = (
            Path(experience_prompt_path).expanduser()
            if experience_prompt_path
            else DEFAULT_EXPERIENCE_SUMMARY_PROMPT_PATH
        )

    def summarize_segment(self, segment: dict[str, Any], qa_items: list[dict[str, Any]]) -> str:
        prompt = build_segment_summary_prompt(
            segment,
            self._normalized_qas(qa_items),
            prompt_path=self.segment_prompt_path,
        )
        return self._generate_summary(prompt, fallback=TemplateSummarizer().summarize_segment(segment, qa_items))

    def summarize_experience(
        self, experience: dict[str, Any], segments: list[dict[str, Any]]
    ) -> str:
        prompt = build_experience_summary_prompt(
            experience,
            self._normalized_segments(segments),
            prompt_path=self.experience_prompt_path,
        )
        return self._generate_summary(
            prompt,
            fallback=TemplateSummarizer().summarize_experience(experience, segments),
        )

    def _generate_summary(self, prompt: str, fallback: str) -> str:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                content = (response.choices[0].message.content or "").strip()
                summary = parse_summary_response(content)
                if summary:
                    return summary
            except Exception as error:
                last_error = error
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)

        if last_error is not None:
            return fallback
        return fallback

    def _normalized_qas(self, qa_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for item in qa_items[-20:]:
            normalized.append(
                {
                    "qa_id": item.get("qa_id", ""),
                    "timestamp": item.get("timestamp", ""),
                    "topic": item.get("topic", ""),
                    "core_entity": item.get("core_entity", ""),
                    "intent": item.get("intent", ""),
                    "entities": item.get("entities", []),
                    "user_input": str(item.get("user_input") or "")[:1200],
                    "assistant_output": str(item.get("assistant_output") or "")[:1200],
                    "confidence": item.get("confidence", 0.0),
                    "reasoning": str(item.get("reasoning") or "")[:800],
                }
            )
        return normalized

    def _normalized_segments(self, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for item in segments[-20:]:
            normalized.append(
                {
                    "segment_id": item.get("segment_id", ""),
                    "topic": item.get("topic", ""),
                    "core_entity": item.get("core_entity", ""),
                    "intent": item.get("intent", ""),
                    "status": item.get("status", ""),
                    "summary": str(item.get("summary") or "")[:1500],
                    "created_at": item.get("created_at", ""),
                    "updated_at": item.get("updated_at", ""),
                }
            )
        return normalized
