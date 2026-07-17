from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExperimentExample:
    case_id: str
    dialogue_id: str
    turn_index: int
    user_input: str
    assistant_output: str
    topic_result: dict[str, Any]
    history: list[dict[str, Any]] = field(default_factory=list)
    reference_answer: str = ""
    evidence_user_inputs: list[str] = field(default_factory=list)
    question_type: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

