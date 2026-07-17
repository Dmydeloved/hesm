"""
LoCoMo dataset loader.

Parses locomo10.json into structured Python objects.
Each conversation has sessions (in temporal order), turns, and QA pairs.
Evidence IDs use the format "D{session_num}:{turn_num}" matching dia_id in turns.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Turn:
    """A single dialog turn in a LoCoMo session."""

    speaker: str
    text: str
    dia_id: str          # e.g. "D1:3" — session 1, turn 3
    session_num: int     # 1-based session index
    timestamp: str       # session date string, e.g. "1:56 pm on 8 May, 2023"

    def to_text(self) -> str:
        return f"[{self.speaker}]: {self.text}"


@dataclass
class QAItem:
    """A single QA pair with evidence pointers."""

    question: str
    answer: str          # can be str or int in the raw data
    evidence: list[str]  # list of dia_ids like ["D1:3", "D2:8"]
    category: int        # 1=Factual, 2=Temporal, 3=Inferential

    def answer_str(self) -> str:
        return str(self.answer)


@dataclass
class Session:
    """One conversation session (a single date's conversation)."""

    session_num: int
    timestamp: str
    turns: list[Turn] = field(default_factory=list)


@dataclass
class LoCoMoConversation:
    """One LoCoMo conversation with all sessions and QA pairs."""

    conv_id: str
    speaker_a: str
    speaker_b: str
    sessions: list[Session] = field(default_factory=list)  # ordered by time
    questions: list[QAItem] = field(default_factory=list)
    # Pre-built turn index: dia_id → Turn (for evidence lookup)
    _turn_index: dict[str, Turn] = field(default_factory=dict, repr=False)

    @property
    def all_turns(self) -> list[Turn]:
        """All turns across all sessions, in temporal order."""
        turns: list[Turn] = []
        for s in self.sessions:
            turns.extend(s.turns)
        return turns

    def get_turn(self, dia_id: str) -> Turn | None:
        return self._turn_index.get(dia_id)

    def total_turn_text(self) -> str:
        """Concatenated text of all turns — used for compression ratio baseline."""
        return "\n".join(t.to_text() for t in self.all_turns)


class LoCoMoLoader:
    """
    Loads and parses the LoCoMo benchmark dataset.

    The JSON format:
    [
      {
        "sample_id": "conv-26",
        "qa": [{"question": ..., "answer": ..., "evidence": ["D1:3"], "category": 1}],
        "conversation": {
          "speaker_a": "Caroline",
          "speaker_b": "Melanie",
          "session_1_date_time": "1:56 pm on 8 May, 2023",
          "session_1": [{"speaker": ..., "dia_id": "D1:1", "text": ...}, ...],
          ...
        },
        ...  # event_summary, observation, session_summary (not used in experiment)
      },
      ...
    ]
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self, max_conversations: int | None = None) -> list[LoCoMoConversation]:
        """
        Parse and return all (or up to max_conversations) conversations.

        Memory building must respect temporal order: sessions are returned
        in ascending session_num order — callers must not reorder them.
        """
        with open(self.path, encoding="utf-8") as f:
            raw: list[dict[str, Any]] = json.load(f)

        if max_conversations is not None:
            raw = raw[:max_conversations]

        conversations: list[LoCoMoConversation] = []
        for i, item in enumerate(raw):
            try:
                conv = self._parse_conversation(item)
                conversations.append(conv)
            except Exception as exc:
                logger.warning("Failed to parse conversation index %d: %s", i, exc)

        logger.info("Loaded %d conversations from %s", len(conversations), self.path)
        return conversations

    # ─── Private helpers ──────────────────────────────────────────────────────

    def _parse_conversation(self, item: dict[str, Any]) -> LoCoMoConversation:
        conv_id: str = item.get("sample_id", f"conv_{id(item)}")
        conversation_raw: dict[str, Any] = item.get("conversation", {})

        speaker_a: str = conversation_raw.get("speaker_a", "Speaker A")
        speaker_b: str = conversation_raw.get("speaker_b", "Speaker B")

        # Collect sessions in numeric order
        session_nums = sorted(self._find_session_nums(conversation_raw))
        sessions: list[Session] = []
        turn_index: dict[str, Turn] = {}

        for num in session_nums:
            session_key = f"session_{num}"
            date_key = f"session_{num}_date_time"
            turns_raw: list[dict] = conversation_raw.get(session_key, [])
            if not turns_raw:
                # Session exists only as a date entry (empty session) — skip
                continue

            timestamp: str = conversation_raw.get(date_key, "")
            turns: list[Turn] = []
            for t in turns_raw:
                turn = Turn(
                    speaker=t.get("speaker", ""),
                    text=t.get("text", ""),
                    dia_id=t.get("dia_id", ""),
                    session_num=num,
                    timestamp=timestamp,
                )
                turns.append(turn)
                if turn.dia_id:
                    turn_index[turn.dia_id] = turn

            sessions.append(Session(session_num=num, timestamp=timestamp, turns=turns))

        # Parse QA pairs
        questions: list[QAItem] = []
        for qa_raw in item.get("qa", []):
            answer_raw = qa_raw.get("answer", "")
            questions.append(
                QAItem(
                    question=str(qa_raw.get("question", "")),
                    answer=answer_raw,
                    evidence=list(qa_raw.get("evidence", [])),
                    category=int(qa_raw.get("category", 1)),
                )
            )

        conv = LoCoMoConversation(
            conv_id=conv_id,
            speaker_a=speaker_a,
            speaker_b=speaker_b,
            sessions=sessions,
            questions=questions,
            _turn_index=turn_index,
        )

        logger.debug(
            "Parsed conv %s: %d sessions, %d total turns, %d QA pairs",
            conv_id,
            len(sessions),
            len(conv.all_turns),
            len(questions),
        )
        return conv

    @staticmethod
    def _find_session_nums(conversation_raw: dict[str, Any]) -> list[int]:
        """Extract all session numbers from keys like 'session_1', 'session_2', ..."""
        nums: list[int] = []
        pattern = re.compile(r"^session_(\d+)$")
        for key in conversation_raw:
            m = pattern.match(key)
            if m:
                nums.append(int(m.group(1)))
        return nums
