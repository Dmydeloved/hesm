"""Backward-compatible imports for retrieval prompts.

Prompt templates and builders live in the prompts package. New code should import
from prompts.topic_memory directly.
"""

from prompts.topic_memory import (
    DEFAULT_EXPERIENCE_SUMMARY_PROMPT_PATH,
    DEFAULT_RETRIEVAL_PROMPT_PATH,
    DEFAULT_SEGMENT_SUMMARY_PROMPT_PATH,
    EXPERIENCE_RETRIEVAL_CRITERIA,
    QA_RETRIEVAL_CRITERIA,
    SEGMENT_RETRIEVAL_CRITERIA,
    build_experience_summary_prompt,
    build_retrieval_prompt,
    build_segment_summary_prompt,
    experience_retrieval_prompt,
    load_experience_summary_prompt_template,
    load_retrieval_prompt_template,
    load_segment_summary_prompt_template,
    qa_retrieval_prompt,
    segment_retrieval_prompt,
)

__all__ = [
    "DEFAULT_EXPERIENCE_SUMMARY_PROMPT_PATH",
    "DEFAULT_RETRIEVAL_PROMPT_PATH",
    "DEFAULT_SEGMENT_SUMMARY_PROMPT_PATH",
    "EXPERIENCE_RETRIEVAL_CRITERIA",
    "QA_RETRIEVAL_CRITERIA",
    "SEGMENT_RETRIEVAL_CRITERIA",
    "build_experience_summary_prompt",
    "build_retrieval_prompt",
    "build_segment_summary_prompt",
    "experience_retrieval_prompt",
    "load_experience_summary_prompt_template",
    "load_retrieval_prompt_template",
    "load_segment_summary_prompt_template",
    "qa_retrieval_prompt",
    "segment_retrieval_prompt",
]
