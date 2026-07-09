from __future__ import annotations

from collections import Counter
from typing import Any


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
