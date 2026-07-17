"""
A-MEM memory system — implemented from scratch based on the paper:
"A-MEM: Agentic Memory for LLM Agents" (Xu et al., 2024)

A-MEM organises memories as linked notes with:
  - content:   original turn text
  - context:   LLM-generated contextual summary
  - keywords:  LLM-extracted key terms for sparse retrieval
  - links:     ids of related notes (bidirectional graph)
  - timestamp: insertion order

Retrieval: keyword overlap + dense vector similarity → graph expansion
           → top-K nodes ranked by combined score.

Storage: SQLite for note graph, ChromaDB for dense vectors.
All LLM calls use the topic_extraction config (same as the rest of the project).
All embeddings use the BailianEmbedder (embedding config).
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

import openai

from experiments.locomo.data.loader import Session
from experiments.locomo.evaluation.token_metrics import count_tokens
from experiments.locomo.methods.base import MemorySystem, RetrievalResult

logger = logging.getLogger(__name__)

# ─── LLM prompts ─────────────────────────────────────────────────────────────

_KEYWORD_PROMPT = """\
Extract 3-5 concise keywords from the following text that best capture its \
key facts, entities, and topics. Return ONLY a JSON array of strings.

Text: {text}

Keywords (JSON array):"""

_CONTEXT_PROMPT = """\
Write a single concise sentence (≤30 words) summarising the key fact or \
event in the following text for future retrieval.

Text: {text}

Summary:"""

_RERANK_PROMPT = """\
Given the question and retrieved memory notes, rank the top {k} most \
relevant notes by returning their IDs as a JSON array (most relevant first).

Question: {question}

Notes:
{notes}

Return ONLY a JSON array of note IDs:"""


class AMEMMemory(MemorySystem):
    """
    A-MEM: graph-structured agentic memory with keyword + vector retrieval.

    Implementation faithful to the paper algorithm:
    1. add_note(text):
       a. extract keywords via LLM
       b. generate context summary via LLM
       c. embed (context + keywords) for dense retrieval
       d. find related notes by vector similarity (top_related)
       e. link bidirectionally if similarity >= link_threshold
       f. persist to SQLite + Chroma

    2. search(query, top_k):
       a. extract keywords from query
       b. keyword-overlap scoring over all notes
       c. dense vector recall (top_k * 3 candidates)
       d. merge + score = α*keyword_score + (1-α)*vector_score
       e. optional graph expansion: include linked notes
       f. optional LLM reranking
       g. return top_k notes
    """

    def __init__(
        self,
        memory_root: str | Path,
        hesm_config: dict[str, Any],
        amem_cfg: dict[str, Any],
    ) -> None:
        self._memory_root = Path(memory_root)
        self._hesm_config = hesm_config
        self._top_related: int = int(amem_cfg.get("top_related", 5))
        self._link_threshold: float = float(amem_cfg.get("link_threshold", 0.7))
        self._alpha: float = 0.4  # keyword vs vector weight

        self._db: sqlite3.Connection | None = None
        self._vector_store: Any = None
        self._embedder: Any = None
        self._llm_client: openai.OpenAI | None = None
        self._llm_model: str = "gpt-4"
        self._max_retries: int = 3
        self._retry_delay: float = 2.0
        self._conv_id: str = ""

    @property
    def method_name(self) -> str:
        return "amem"

    def reset(self) -> None:
        if self._db is not None:
            try:
                self._db.close()
            except Exception:
                pass
        self._db = None
        self._vector_store = None
        self._embedder = None
        self._llm_client = None
        self._conv_id = ""

    def build_memory(
        self,
        conv_id: str,
        sessions: list[Session],
        speaker_a: str,
        speaker_b: str,
    ) -> None:
        self._conv_id = conv_id
        self._setup_components(conv_id)

        total = 0
        for session in sessions:
            for turn in session.turns:
                if not turn.text.strip():
                    continue
                text = f"[{turn.speaker}]: {turn.text}"
                try:
                    self._add_note(
                        text=text,
                        dia_id=turn.dia_id,
                        timestamp=turn.timestamp,
                    )
                    total += 1
                except Exception as exc:
                    logger.warning("[AMEM] turn %s failed: %s", turn.dia_id, exc)

        logger.info("[AMEM] %s: added %d notes", conv_id, total)

    def retrieve(self, question: str, top_k: int = 5) -> RetrievalResult:
        if self._db is None:
            return RetrievalResult("", [], 0, {"error": "amem not initialised"})

        try:
            top_notes = self._search(question, top_k)
        except Exception as exc:
            logger.warning("[AMEM] search failed: %s", exc)
            return RetrievalResult("", [], 0, {"error": str(exc)})

        lines: list[str] = []
        retrieved_ids: list[str] = []
        for note in top_notes:
            content = note.get("content", "")
            if content:
                lines.append(content)
            dia_id = note.get("dia_id", "")
            if dia_id:
                retrieved_ids.append(dia_id)

        context_text = "\n".join(lines)
        return RetrievalResult(
            context_text=context_text,
            retrieved_ids=retrieved_ids,
            token_count=count_tokens(context_text),
            raw_result={"num_notes": len(top_notes)},
        )

    # ─── Internal: storage setup ──────────────────────────────────────────────

    def _setup_components(self, conv_id: str) -> None:
        from memory.embedder import BailianEmbedder
        from memory.vector_store import ChromaVectorStore

        base = self._memory_root / f"amem_{conv_id}"
        base.mkdir(parents=True, exist_ok=True)

        # SQLite for note graph
        self._db = sqlite3.connect(str(base / "amem_notes.sqlite3"))
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                note_id     TEXT PRIMARY KEY,
                dia_id      TEXT,
                content     TEXT,
                context     TEXT,
                keywords    TEXT,    -- JSON list of strings
                links       TEXT,    -- JSON list of note_ids
                timestamp   TEXT,
                created_at  REAL
            )
        """)
        self._db.commit()

        # Chroma for dense vectors
        self._vector_store = ChromaVectorStore(
            persist_path=str(base / "chroma")
        )
        self._embedder = BailianEmbedder()

        # LLM client (topic_extraction config)
        te = self._hesm_config.get("topic_extraction", {})
        api_key = te.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
        base_url = te.get("base_url", "https://api.openai.com/v1/")
        self._llm_model = te.get("model", "gpt-4")
        self._max_retries = int(te.get("max_retries", 3))
        self._retry_delay = float(te.get("retry_delay", 2.0))
        self._llm_client = openai.OpenAI(api_key=api_key, base_url=base_url)

    # ─── Internal: add note ───────────────────────────────────────────────────

    def _add_note(self, text: str, dia_id: str, timestamp: str) -> str:
        note_id = str(uuid.uuid4())

        # Step 1: Extract keywords
        keywords = self._extract_keywords(text)

        # Step 2: Generate contextual summary
        context = self._generate_context(text)

        # Step 3: Embed (context + keywords) for retrieval
        embed_text = f"{context} {' '.join(keywords)}"
        embedding = self._embedder.embed(embed_text)

        # Step 4: Find related notes by vector similarity
        try:
            candidates = self._vector_store.query(
                query_embedding=embedding,
                memory_type="qa",
                top_k=self._top_related,
            )
        except Exception:
            candidates = []

        # Step 5: Link notes above similarity threshold
        links: list[str] = []
        for cand in candidates:
            sim = cand.get("similarity", 0.0)
            if sim >= self._link_threshold:
                related_id = cand.get("metadata", {}).get("note_id", "")
                if related_id and related_id != note_id:
                    links.append(related_id)
                    # Bidirectional: update the related note's links
                    self._add_link(related_id, note_id)

        # Step 6: Persist
        self._db.execute(
            """INSERT INTO notes
               (note_id, dia_id, content, context, keywords, links, timestamp, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                note_id, dia_id, text, context,
                json.dumps(keywords, ensure_ascii=False),
                json.dumps(links),
                timestamp,
                time.time(),
            ),
        )
        self._db.commit()

        self._vector_store.upsert(
            memory_type="qa",
            memory_id=note_id,
            text=embed_text,
            embedding=embedding,
            updated_at=timestamp,
            metadata={"note_id": note_id, "dia_id": dia_id},
        )
        return note_id

    def _add_link(self, note_id: str, new_link_id: str) -> None:
        """Add new_link_id to note_id's links list (bidirectional)."""
        row = self._db.execute(
            "SELECT links FROM notes WHERE note_id = ?", (note_id,)
        ).fetchone()
        if row is None:
            return
        links: list[str] = json.loads(row[0] or "[]")
        if new_link_id not in links:
            links.append(new_link_id)
            self._db.execute(
                "UPDATE notes SET links = ? WHERE note_id = ?",
                (json.dumps(links), note_id),
            )
            self._db.commit()

    # ─── Internal: search ─────────────────────────────────────────────────────

    def _search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        # Step 1: Extract query keywords
        q_keywords = set(self._extract_keywords(query))

        # Step 2: Get all notes for keyword scoring
        rows = self._db.execute(
            "SELECT note_id, dia_id, content, context, keywords, links FROM notes"
        ).fetchall()

        keyword_scores: dict[str, float] = {}
        for row in rows:
            note_id, dia_id, content, ctx, kw_json, links_json = row
            kw_list: list[str] = json.loads(kw_json or "[]")
            note_kw_set = set(w.lower() for w in kw_list)
            q_kw_lower = set(w.lower() for w in q_keywords)
            overlap = len(q_kw_lower & note_kw_set)
            keyword_scores[note_id] = overlap / max(len(q_kw_lower), 1)

        # Step 3: Dense vector recall
        try:
            q_context = self._generate_context(query)
            q_embed_text = f"{q_context} {' '.join(q_keywords)}"
            q_embedding = self._embedder.embed(q_embed_text)
            vector_results = self._vector_store.query(
                query_embedding=q_embedding,
                memory_type="qa",
                top_k=min(top_k * 3, 20),
            )
        except Exception as exc:
            logger.warning("[AMEM] vector search failed: %s", exc)
            vector_results = []

        vector_scores: dict[str, float] = {}
        for r in vector_results:
            nid = r.get("metadata", {}).get("note_id", "")
            if nid:
                vector_scores[nid] = r.get("similarity", 0.0)

        # Step 4: Merge scores
        all_note_ids = set(keyword_scores) | set(vector_scores)
        combined: list[tuple[float, str]] = []
        for nid in all_note_ids:
            kw_s = keyword_scores.get(nid, 0.0)
            vec_s = vector_scores.get(nid, 0.0)
            score = self._alpha * kw_s + (1 - self._alpha) * vec_s
            combined.append((score, nid))
        combined.sort(key=lambda x: x[0], reverse=True)

        # Step 5: Graph expansion — include linked notes of top candidates
        primary_ids = [nid for _, nid in combined[:top_k]]
        expanded_ids: set[str] = set(primary_ids)
        for nid in primary_ids:
            row = self._db.execute(
                "SELECT links FROM notes WHERE note_id = ?", (nid,)
            ).fetchone()
            if row:
                for link_id in json.loads(row[0] or "[]"):
                    expanded_ids.add(link_id)

        # Step 6: Fetch and return top-K notes
        result_notes: list[dict[str, Any]] = []
        id_to_score = {nid: s for s, nid in combined}
        sorted_expanded = sorted(
            expanded_ids,
            key=lambda nid: id_to_score.get(nid, 0.0),
            reverse=True,
        )[:top_k]

        for nid in sorted_expanded:
            row = self._db.execute(
                "SELECT note_id, dia_id, content, context, keywords, links FROM notes "
                "WHERE note_id = ?",
                (nid,),
            ).fetchone()
            if row:
                result_notes.append({
                    "note_id": row[0],
                    "dia_id": row[1],
                    "content": row[2],
                    "context": row[3],
                    "keywords": json.loads(row[4] or "[]"),
                    "links": json.loads(row[5] or "[]"),
                    "score": id_to_score.get(nid, 0.0),
                })

        return result_notes

    # ─── Internal: LLM helpers ────────────────────────────────────────────────

    def _llm_call(self, prompt: str) -> str:
        """Retry-wrapped LLM call. Returns "" on failure."""
        if self._llm_client is None:
            return ""
        for attempt in range(1, self._max_retries + 1):
            try:
                resp = self._llm_client.chat.completions.create(
                    model=self._llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                return resp.choices[0].message.content.strip()
            except Exception as exc:
                if attempt < self._max_retries:
                    time.sleep(self._retry_delay * attempt)
        return ""

    def _extract_keywords(self, text: str) -> list[str]:
        raw = self._llm_call(_KEYWORD_PROMPT.format(text=text[:800]))
        try:
            # Strip markdown fence if present
            raw = re.sub(r"```(?:json)?|```", "", raw).strip()
            kws = json.loads(raw)
            if isinstance(kws, list):
                return [str(k).strip() for k in kws if k][:8]
        except Exception:
            pass
        # Fallback: split words
        return re.findall(r"\b[a-zA-Z一-鿿]{2,}\b", text)[:5]

    def _generate_context(self, text: str) -> str:
        ctx = self._llm_call(_CONTEXT_PROMPT.format(text=text[:800]))
        return ctx if ctx else text[:100]
