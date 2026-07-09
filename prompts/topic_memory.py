from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROMPTS_DIR = Path(__file__).resolve().parent
DEFAULT_RETRIEVAL_PROMPT_PATH = PROMPTS_DIR / "retrieval_prompt.txt"
DEFAULT_EXTRACTOR_PROMPT_PATH = PROMPTS_DIR / "extractor_prompt.txt"


EXPERIENCE_RETRIEVAL_CRITERIA = """* Decide whether topic belongs to the same long-term discussion domain.
* Decide whether core_entity is the same object or an acceptable alias.
* Decide whether intents cover the current user intent.
* Use summary and state as additional relevance evidence.
* vector_recalled means the candidate also appeared in vector retrieval; it is only a recall hint, not a relevance score."""

SEGMENT_RETRIEVAL_CRITERIA = """* Decide whether the segment intent matches the query intent.
* Decide whether summary contains information that should be continued or reused.
* topic and core_entity should remain consistent with the selected Experience layer.
* vector_recalled means the candidate also appeared in vector retrieval; it is only a recall hint, not a relevance score."""

QA_RETRIEVAL_CRITERIA = """* Decide whether user_input and assistant_output can directly help answer the current query.
* Decide whether intent, entities, topic, and core_entity match the current query.
* Prefer QA records that are complete, traceable, and higher confidence.
* vector_recalled means the candidate also appeared in vector retrieval; confidence and timestamp are context only, not relevance scores."""


def load_prompt_template(path: str | Path) -> str:
    return Path(path).expanduser().read_text(encoding="utf-8")


def load_retrieval_prompt_template(path: str | Path | None = None) -> str:
    return load_prompt_template(path or DEFAULT_RETRIEVAL_PROMPT_PATH)


def load_extractor_prompt_template(path: str | Path | None = None) -> str:
    return load_prompt_template(path or DEFAULT_EXTRACTOR_PROMPT_PATH)


def build_retrieval_prompt(
    *,
    layer_name: str,
    query_text: str,
    candidates: list[dict[str, Any]],
    limit: int,
    criteria: str,
    prompt_path: str | Path | None = None,
) -> str:
    candidates_json = json.dumps(candidates, ensure_ascii=False, indent=2)
    template = load_retrieval_prompt_template(prompt_path)
    return (
        template.replace("{layer_name}", layer_name)
        .replace("{query_text}", query_text)
        .replace("{candidates_json}", candidates_json)
        .replace("{criteria}", criteria)
        .replace("{limit}", str(limit))
    )


def build_extractor_prompt(
    user_input: str,
    conversation_context: str = "",
    domain_knowledge: str = "",
    *,
    prompt_path: str | Path | None = None,
) -> str:
    template = load_extractor_prompt_template(prompt_path)
    return (
        template.replace("{user_input}", user_input)
        .replace("{conversation_context}", conversation_context)
        .replace("{domain_knowledge}", domain_knowledge)
    )


def experience_retrieval_prompt(
    query_text: str,
    candidates: list[dict[str, Any]],
    limit: int,
) -> str:
    return build_retrieval_prompt(
        layer_name="Experience",
        query_text=query_text,
        candidates=candidates,
        limit=limit,
        criteria=EXPERIENCE_RETRIEVAL_CRITERIA,
    )


def segment_retrieval_prompt(
    query_text: str,
    candidates: list[dict[str, Any]],
    limit: int,
) -> str:
    return build_retrieval_prompt(
        layer_name="Segment",
        query_text=query_text,
        candidates=candidates,
        limit=limit,
        criteria=SEGMENT_RETRIEVAL_CRITERIA,
    )


def qa_retrieval_prompt(
    query_text: str,
    candidates: list[dict[str, Any]],
    limit: int,
) -> str:
    return build_retrieval_prompt(
        layer_name="QA",
        query_text=query_text,
        candidates=candidates,
        limit=limit,
        criteria=QA_RETRIEVAL_CRITERIA,
    )


__all__ = [
    "DEFAULT_EXTRACTOR_PROMPT_PATH",
    "DEFAULT_RETRIEVAL_PROMPT_PATH",
    "EXPERIENCE_RETRIEVAL_CRITERIA",
    "PROMPTS_DIR",
    "QA_RETRIEVAL_CRITERIA",
    "SEGMENT_RETRIEVAL_CRITERIA",
    "build_extractor_prompt",
    "build_retrieval_prompt",
    "experience_retrieval_prompt",
    "load_extractor_prompt_template",
    "load_prompt_template",
    "load_retrieval_prompt_template",
    "qa_retrieval_prompt",
    "segment_retrieval_prompt",
]
