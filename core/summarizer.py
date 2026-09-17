from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Protocol

from .prompts.topic_memory import (
    DEFAULT_EXPERIENCE_SUMMARY_PROMPT_PATH,
    DEFAULT_SEGMENT_SUMMARY_PROMPT_PATH,
    build_experience_summary_prompt,
    build_segment_summary_prompt,
)

from .config import get as config_get

from collections import Counter


logger = logging.getLogger(__name__)


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
        conclusion = (
            f"该阶段围绕 {segment['topic']} / {segment['core_entity']} 推进"
            f"{segment['intent']}，已累计 {len(qa_items)} 条 QA 证据。"
        )
        payload = {
            "goal": f"完成与“{segment['intent']}”相关的阶段任务",
            "key_facts": [
                {
                    "fact": f"相关实体包括：{'、'.join(top_entities)}",
                    "source_qa_ids": [
                        str(item.get("qa_id") or "") for item in qa_items
                    ],
                }
            ] if top_entities else [],
            "state_changes": [],
            "state": {"status": "ongoing", "current_conclusion": conclusion},
        }
        return json.dumps(payload, ensure_ascii=False)

    def summarize_experience(
        self, experience: dict[str, Any], segments: list[dict[str, Any]]
    ) -> str:
        intents = ", ".join(experience.get("intents_link") or [])
        payload = {
            "goal": f"完成 {experience['topic']} / {experience['core_entity']} 的长期任务",
            "stage_trajectory": [
                {
                    "intent": str(item.get("intent") or ""),
                    "result": "该阶段已形成结构化状态记忆",
                }
                for item in segments
                if item.get("intent")
            ],
            "stable_facts": [],
            "current_state": {
                "status": "ongoing",
                "summary": (
                    f"当前累计 {len(segments)} 个阶段，涉及意图：{intents or '无'}。"
                ),
            },
        }
        return json.dumps(payload, ensure_ascii=False)


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
                raise ValueError("Set summarization.api_key in config/hesm.yaml.")
            if not model:
                raise ValueError("Set summarization.model in config/hesm.yaml.")
            if not base_url:
                raise ValueError("Set summarization.base_url in config/hesm.yaml.")
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
        logger.info(
            "Segment summary started segment_id=%s qa_count=%s",
            segment.get("segment_id", ""),
            len(qa_items),
        )
        return self._generate_summary(
            prompt,
            fallback=TemplateSummarizer().summarize_segment(segment, qa_items),
            task="segment_summary",
        )

    def summarize_experience(
        self, experience: dict[str, Any], segments: list[dict[str, Any]]
    ) -> str:
        prompt = build_experience_summary_prompt(
            experience,
            self._normalized_segments(segments),
            prompt_path=self.experience_prompt_path,
        )
        logger.info(
            "Experience summary started experience_id=%s segment_count=%s",
            experience.get("experience_id", ""),
            len(segments),
        )
        return self._generate_summary(
            prompt,
            fallback=TemplateSummarizer().summarize_experience(experience, segments),
            task="experience_summary",
        )

    def _generate_summary(
        self,
        prompt: str,
        fallback: str,
        task: str = "summary",
    ) -> str:
        logger.info(
            "Summary LLM prompt task=%s model=%s prompt=%s",
            task,
            self.model,
            prompt,
        )
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    "Summary LLM request task=%s model=%s attempt=%s",
                    task,
                    self.model,
                    attempt,
                )
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                content = (response.choices[0].message.content or "").strip()
                logger.info(
                    "Summary LLM response task=%s model=%s attempt=%s content=%s",
                    task,
                    self.model,
                    attempt,
                    content,
                )
                summary = parse_summary_response(content)
                if summary:
                    logger.info(
                        "Summary parsed task=%s model=%s summary=%s",
                        task,
                        self.model,
                        summary,
                    )
                    return summary
                logger.warning(
                    "Summary LLM returned empty parsed summary task=%s model=%s attempt=%s",
                    task,
                    self.model,
                    attempt,
                )
            except Exception as error:
                last_error = error
                logger.warning(
                    "Summary LLM attempt failed task=%s model=%s attempt=%s/%s",
                    task,
                    self.model,
                    attempt,
                    self.max_retries,
                    exc_info=True,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)

        if last_error is not None:
            logger.warning(
                "Summary generation failed; using fallback task=%s model=%s error=%s fallback=%s",
                task,
                self.model,
                last_error,
                fallback,
            )
            return fallback
        logger.info(
            "Summary generation empty after retries; using fallback task=%s model=%s fallback=%s",
            task,
            self.model,
            fallback,
        )
        return fallback

    def _normalized_qas(self, qa_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for item in qa_items:
            normalized.append(
                {
                    "qa_id": item.get("qa_id", ""),
                    "timestamp": item.get("timestamp", ""),
                    "topic": item.get("topic", ""),
                    "core_entity": item.get("core_entity", ""),
                    "intent": item.get("intent", ""),
                    "entities": item.get("entities", []),
                    "user_input": str(item.get("user_input") or ""),
                    "assistant_output": str(item.get("assistant_output") or "")[:1200],
                    "tools": item.get("tools") or [],
                    "confidence": item.get("confidence", 0.0),
                    "reason": str(
                        item.get("reason") or item.get("reasoning") or ""
                    )[:800],
                }
            )
        return normalized

    def _normalized_segments(self, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for item in segments:
            normalized.append(
                {
                    "segment_id": item.get("segment_id", ""),
                    "topic": item.get("topic", ""),
                    "core_entity": item.get("core_entity", ""),
                    "intent": item.get("intent", ""),
                    "status": item.get("status", ""),
                    "summary_json": item.get("summary") or {},
                    "created_at": item.get("created_at", ""),
                    "updated_at": item.get("updated_at", ""),
                }
            )
        return normalized
