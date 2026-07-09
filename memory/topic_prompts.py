"""Backward-compatible imports for retrieval prompts.

Prompt templates and builders live in the prompts package. New code should import
from prompts.topic_memory directly.
"""

from prompts.topic_memory import (
    DEFAULT_RETRIEVAL_PROMPT_PATH,
    EXPERIENCE_RETRIEVAL_CRITERIA,
    QA_RETRIEVAL_CRITERIA,
    SEGMENT_RETRIEVAL_CRITERIA,
    build_retrieval_prompt,
    experience_retrieval_prompt,
    load_retrieval_prompt_template,
    qa_retrieval_prompt,
    segment_retrieval_prompt,
)

__all__ = [
    "DEFAULT_RETRIEVAL_PROMPT_PATH",
    "EXPERIENCE_RETRIEVAL_CRITERIA",
    "QA_RETRIEVAL_CRITERIA",
    "SEGMENT_RETRIEVAL_CRITERIA",
    "build_retrieval_prompt",
    "experience_retrieval_prompt",
    "load_retrieval_prompt_template",
    "qa_retrieval_prompt",
    "segment_retrieval_prompt",
]
