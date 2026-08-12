"""
HESM memory system adapter.

Wraps the existing HESM Memory Pipeline (TopicExtractor → MemoryManager →
HybridRetriever) without modifying any source files.

Key design decisions:
- Per-conversation isolation: each conv_id gets its own SQLite DB + Chroma dir
- dia_id tracking: stored in the 'tools' field of each QA as {"dia_id": "D1:3"},
  then recovered from qa["tools"][0]["dia_id"] at retrieval time
- Question topics: TopicExtractor is also called on the question to get
  topic/core_entity/intent/entities for hierarchical retrieval

Also provides HESMAblationMemory (subclass) that exposes the underlying
storage/vector_store for ablation variants to use directly.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any
import json
from experiments.locomo.data.loader import Session
from experiments.locomo.evaluation.token_metrics import count_tokens
from experiments.locomo.methods.base import MemorySystem, RetrievalResult

logger = logging.getLogger(__name__)

# Context window passed to TopicExtractor when processing turns
_CONTEXT_WINDOW = 5  # number of preceding turns to include as context


def _hesm_model_config(experiment_config: dict[str, Any]) -> dict[str, Any]:
    method_config = experiment_config.get("memory_methods", {}).get("hesm", {})
    required_sections = (
        "embedding",
        "topic_extraction",
        "summarization",
        "retrieval",
    )
    missing = [
        name
        for name in required_sections
        if not isinstance(method_config.get(name), dict) or not method_config[name]
    ]
    if missing:
        raise ValueError(
            "experiments/config/locomo.yaml is missing HESM experiment settings: "
            + ", ".join(f"memory_methods.hesm.{name}" for name in missing)
        )
    return method_config


class HESMMemory(MemorySystem):
    """
    Calls the HESM Memory Pipeline for memory building and retrieval.

    Build:  TopicExtractor.extract(turn_text, context=recent_turns)
            → MemoryManager.add_qa(topic_result, user_input, tools=[{dia_id}])

    Retrieve: TopicExtractor.extract(question)
              → HybridRetriever.recall(topic, core_entity, intent, entities)
              → extract dia_ids from qa["tools"][0]["dia_id"]
    """

    def __init__(
        self,
        memory_root: str | Path,
        hesm_cfg: dict[str, Any],
        experiment_config: dict[str, Any],
        use_llm_summarizer: bool = True,
        use_llm_reranker: bool = True,
        use_cache: bool = True,
    ) -> None:
        """
        Args:
            memory_root:        base dir for per-conv SQLite + Chroma
            hesm_cfg:           hesm section from experiments/config/locomo.yaml
            use_llm_summarizer: if False, use TemplateSummarizer (faster, offline)
            use_llm_reranker:   if False, skip LLM reranking (vector-only retrieval)
            experiment_config:  experiment-owned model and method settings
            use_cache:          benchmark switch retained by the adapter
        """
        self._memory_root = Path(memory_root)
        self._cfg = hesm_cfg
        self._use_llm_summarizer = use_llm_summarizer
        self._use_llm_reranker = use_llm_reranker
        del use_cache
        self._experiment_config = experiment_config

        # Populated by build_memory(); reset by reset()
        self._storage: Any = None
        self._vector_store: Any = None
        self._embedder: Any = None
        self._extractor: Any = None
        self._manager: Any = None
        self._retriever: Any = None
        self._conv_id: str = ""

    @property
    def method_name(self) -> str:
        return "hesm"

    def reset(self) -> None:
        self._storage = None
        self._vector_store = None
        self._embedder = None
        self._extractor = None
        self._manager = None
        self._retriever = None
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

        if self._has_existing_memory(sessions):
            return

        completed_dia_ids = self._stored_dia_ids()

        total_turns = sum(len(s.turns) for s in sessions)
        processed = 0
        recent_turns: list[str] = []  # rolling window for extractor context
        count = 0
        for session in sessions:
            for turn in session.turns:
                if not turn.text.strip():
                    continue
                user_input = f"[{turn.speaker}]: {turn.text}"
                if turn.dia_id and turn.dia_id in completed_dia_ids:
                    recent_turns.append(
                        {"user_input": user_input, "extractor_result": {}}
                    )
                    continue
                window_turns = recent_turns[-_CONTEXT_WINDOW:]
                str_list = [json.dumps(turn, ensure_ascii=False) for turn in window_turns]
                context = "\n".join(str_list)

                topic_result: Any = {}
                try:
                    topic_result = self._extractor.extract(
                        user_input=user_input,
                        context=context,
                    )
                    # Normalise: TopicExtractor can return list or dict
                    if isinstance(topic_result, list):
                        for item in topic_result:
                            self._manager.add_qa(
                                topic_result=item,
                                user_input=user_input,
                                assistant_output="",
                                # Store dia_id in tools field for later retrieval
                                tools=[{"dia_id": turn.dia_id}] if turn.dia_id else [],
                                timestamp=turn.timestamp,
                                state_key="default",
                            )
                    else:
                        self._manager.add_qa(
                            topic_result=topic_result,
                            user_input=user_input,
                            assistant_output="",
                            # Store dia_id in tools field for later retrieval
                            tools=[{"dia_id": turn.dia_id}] if turn.dia_id else [],
                            timestamp=turn.timestamp,
                            state_key="default",
                        )
                    processed += 1
                    if turn.dia_id:
                        completed_dia_ids.add(turn.dia_id)
                except Exception as exc:
                    logger.warning(
                        "[HESM] %s turn %s failed: %s", conv_id, turn.dia_id, exc
                    )
                tmp_turn = {
                    "user_input": user_input,
                    "extractor_result": topic_result
                }
                recent_turns.append(tmp_turn)
                count += 1
                print(f"========================count : {count}")

        logger.info(
            "[HESM] %s: processed %d/%d turns", conv_id, processed, total_turns
        )

    def _has_existing_memory(self, sessions: list[Session]) -> bool:
        """
        Return True when the per-conversation HESM store already has usable data.

        The runner calls build_memory() before QA, so this check lets us attach the
        existing SQLite/Chroma components and skip re-ingesting turns.
        """
        if self._storage is None or self._vector_store is None:
            return False

        try:
            qa_count = int(
                self._storage.connection.execute(
                    "SELECT COUNT(*) FROM qa_memory"
                ).fetchone()[0]
            )
            vector_count = int(self._vector_store.count())
        except Exception as exc:
            logger.warning(
                "[HESM] %s: failed to inspect existing memory, rebuilding: %s",
                self._conv_id,
                exc,
            )
            return False

        if qa_count <= 0 or vector_count <= 0:
            return False

        expected_dia_ids = {
            turn.dia_id
            for session in sessions
            for turn in session.turns
            if turn.text.strip() and turn.dia_id
        }
        stored_dia_ids = self._stored_dia_ids()
        missing_count = len(expected_dia_ids - stored_dia_ids)
        if missing_count:
            logger.warning(
                "[HESM] %s: existing memory has %d missing dia_ids; "
                "only missing turns will be built (qa=%d, vectors=%d)",
                self._conv_id,
                missing_count,
                qa_count,
                vector_count,
            )
            return False
        else:
            logger.info(
                "[HESM] %s: existing memory found (qa=%d, vectors=%d), skipping build",
                self._conv_id,
                qa_count,
                vector_count,
            )
            return True

    def _stored_dia_ids(self) -> set[str]:
        if self._storage is None:
            return set()

        rows = self._storage.connection.execute(
            "SELECT tools_json FROM qa_memory"
        ).fetchall()
        dia_ids: set[str] = set()
        for row in rows:
            try:
                tools = json.loads(row["tools_json"] or "[]")
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(tools, list):
                continue
            for tool in tools:
                if isinstance(tool, dict) and tool.get("dia_id"):
                    dia_ids.add(str(tool["dia_id"]))
        return dia_ids

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        use_cache: bool | None = None,
    ) -> RetrievalResult:
        # The shared benchmark interface still supplies top_k. HESM gets its
        # three explicit limits from configuration.
        # use_cache belongs to the cache experiment contract. The current
        # HybridRetriever.recall() does not accept it.
        del top_k, use_cache
        if self._retriever is None or self._extractor is None:
            return RetrievalResult("", [], 0, {"error": "memory not built"})

        cfg = self._cfg

        try:
            # Extract topic/entity/intent from the question
            q_topic = self._extractor.extract(user_input=question)
            if isinstance(q_topic, list):
                q_topic = q_topic[0] if q_topic else {}

            topic = q_topic.get("topic", "")
            core_entity = q_topic.get("core_entity", "")
            intent = q_topic.get("intent")
            entities = q_topic.get("entities", [])

            result = self._retriever.recall(
                topic=topic,
                core_entity=core_entity,
                intent=intent,
                entities=entities,
                query=question,
                query_confidence=float(q_topic.get("confidence", 0.0)),
                top_experience=int(cfg.get("top_experience", 3)),
                top_segment=int(cfg.get("top_segment", 5)),
                top_qa=int(cfg.get("top_qa", 8)),
            )
        except Exception as exc:
            logger.warning("[HESM] retrieve failed for %r: %s", question[:60], exc)
            return RetrievalResult("", [], 0, {"error": str(exc)})

        # Extract dia_ids from returned QAs
        retrieved_ids: list[str] = []
        for qa in result.get("qas", []):
            tools = qa.get("tools") or []
            if tools and isinstance(tools, list) and isinstance(tools[0], dict):
                dia_id = tools[0].get("dia_id", "")
                if dia_id:
                    retrieved_ids.append(dia_id)

        context_text = result.get("context_text", "")
        return RetrievalResult(
            context_text=context_text,
            retrieved_ids=retrieved_ids,
            token_count=count_tokens(context_text),
            raw_result=result,
        )

    # ─── Internal setup ───────────────────────────────────────────────────────

    def _setup_components(self, conv_id: str) -> None:
        """Create isolated HESM components for this conversation."""
        from hesm.embedder import BailianEmbedder
        from hesm.extractor import TopicExtractor
        from hesm.manager import MemoryManager
        from hesm.retriever import HybridRetriever
        from hesm.storage import MemoryStorage
        from hesm.summarizer import LLMSummarizer, TemplateSummarizer
        from hesm.vector_store import ChromaVectorStore

        base = self._memory_root / f"hesm_{conv_id}"
        base.mkdir(parents=True, exist_ok=True)

        db_path = base / "memory.sqlite3"
        chroma_path = base / "chroma"

        self._storage = MemoryStorage(db_path=str(db_path))
        self._vector_store = ChromaVectorStore(persist_path=str(chroma_path))
        method_models = _hesm_model_config(self._experiment_config)
        embedding_cfg = method_models["embedding"]
        topic_cfg = method_models["topic_extraction"]
        summarization_cfg = method_models["summarization"]
        retrieval_cfg = method_models["retrieval"]

        self._embedder = BailianEmbedder(
            api_key=embedding_cfg.get("api_key"),
            model=embedding_cfg.get("model"),
            base_url=embedding_cfg.get("base_url"),
        )
        self._extractor = TopicExtractor(
            api_key=topic_cfg.get("api_key"),
            model=topic_cfg.get("model"),
            base_url=topic_cfg.get("base_url"),
            max_retries=topic_cfg.get("max_retries"),
            retry_delay=topic_cfg.get("retry_delay"),
        )

        summarizer = (
            LLMSummarizer(
                api_key=summarization_cfg.get("api_key"),
                model=summarization_cfg.get("model"),
                base_url=summarization_cfg.get("base_url"),
                max_retries=summarization_cfg.get("max_retries"),
                retry_delay=summarization_cfg.get("retry_delay"),
            )
            if self._use_llm_summarizer
            else TemplateSummarizer()
        )

        self._manager = MemoryManager(
            storage=self._storage,
            vector_store=self._vector_store,
            embedder=self._embedder,
            summarizer=summarizer,
            segment_summary_qa_threshold=int(
                self._cfg.get("segment_qa_threshold", 5)
            ),
            experience_summary_segment_threshold=int(
                self._cfg.get("experience_segment_threshold", 5)
            ),
            experience_similarity_threshold=float(
                self._cfg.get("experience_similarity_threshold", 0.82)
            ),
            min_segment_qas=int(
                self._cfg.get("min_segment_qas", 2)
            ),
        )

        if self._use_llm_reranker:
            self._retriever = HybridRetriever(
                storage=self._storage,
                vector_store=self._vector_store,
                embedder=self._embedder,
                rerank_with_llm=True,
                retrieval_api_key=retrieval_cfg.get("api_key"),
                retrieval_model=retrieval_cfg.get("model"),
                retrieval_base_url=retrieval_cfg.get("base_url"),
                retrieval_max_retries=retrieval_cfg.get("max_retries"),
                retrieval_retry_delay=retrieval_cfg.get("retry_delay"),
                retrieval_config=self._cfg,
            )
        else:
            self._retriever = HybridRetriever(
                storage=self._storage,
                vector_store=self._vector_store,
                embedder=self._embedder,
                rerank_with_llm=False,
                retrieval_config=self._cfg,
            )

    # ─── Expose internals for ablation ────────────────────────────────────────

    def get_storage(self) -> Any:
        return self._storage

    def get_vector_store(self) -> Any:
        return self._vector_store

    def get_embedder(self) -> Any:
        return self._embedder

    def get_extractor(self) -> Any:
        return self._extractor

    def get_memory_root(self) -> Path:
        return self._memory_root

    def get_conv_id(self) -> str:
        return self._conv_id


# ─────────────────────────────────────────────────────────────────────────────
# Ablation variants — share HESM storage, differ only in retrieval
# ─────────────────────────────────────────────────────────────────────────────

class HESMAblationMemory(MemorySystem):
    """
    Ablation-study memory system that reuses HESM's built storage but applies
    a configurable retrieval strategy:

      flat_memory  — raw turn vectors, no TopicExtractor, flat Chroma query
      qa_only      — Chroma query on QA vectors only
      qa_segment   — merged Chroma query on QA + Segment vectors
      full_hesm_no_reranker — full hierarchy with local selection only
      full_hesm    — full HybridRetriever.recall() (same as HESMMemory)

    For "flat_memory" the memory is built fresh (raw turns → Chroma, no HESM).
    Other variants reuse HESM storage. If it is absent, the first variant builds
    it automatically so run_ablation.py also works standalone.
    """

    def __init__(
        self,
        variant: str,
        memory_root: str | Path,
        hesm_cfg: dict[str, Any],
        variant_cfg: dict[str, Any],
        experiment_config: dict[str, Any],
    ) -> None:
        """
        Args:
            variant:      one of "flat_memory", "qa_only", "qa_segment", "full_hesm"
            memory_root:  same root used by HESMMemory (to locate its storage)
            hesm_cfg:     hesm section from experiments/config/locomo.yaml
            variant_cfg:  ablation.variants.<variant> section from experiments/config/locomo.yaml
        """
        self._variant = variant
        self._memory_root = Path(memory_root)
        self._hesm_cfg = hesm_cfg
        self._variant_cfg = variant_cfg
        self._experiment_config = experiment_config

        self._storage: Any = None
        self._vector_store: Any = None       # HESM vector store (QA/Segment/Exp vectors)
        self._flat_vector_store: Any = None  # flat_memory only: raw turn vectors
        self._embedder: Any = None
        self._extractor: Any = None
        self._retriever: Any = None          # full_hesm only
        self._conv_id: str = ""

    @property
    def method_name(self) -> str:
        return f"ablation_{self._variant}"

    def reset(self) -> None:
        self._storage = None
        self._vector_store = None
        self._flat_vector_store = None
        self._embedder = None
        self._extractor = None
        self._retriever = None
        self._conv_id = ""

    def build_memory(
        self,
        conv_id: str,
        sessions: list[Session],
        speaker_a: str,
        speaker_b: str,
    ) -> None:
        """
        For flat_memory: build a raw turn vector store.
        For others: attach to the HESM storage built by the main experiment.
        """
        from hesm.embedder import BailianEmbedder

        self._conv_id = conv_id
        method_models = _hesm_model_config(self._experiment_config)
        embedding_cfg = method_models["embedding"]
        self._embedder = BailianEmbedder(
            api_key=embedding_cfg.get("api_key"),
            model=embedding_cfg.get("model"),
            base_url=embedding_cfg.get("base_url"),
        )

        if self._variant == "flat_memory":
            self._build_flat_memory(conv_id, sessions)
        else:
            self._ensure_hesm_storage(
                conv_id, sessions, speaker_a=speaker_a, speaker_b=speaker_b
            )
            self._attach_hesm_storage(conv_id)

    def retrieve(self, question: str, top_k: int = 5) -> RetrievalResult:
        layers = self._variant_cfg.get("retrieval_layers", ["qa"])

        if self._variant == "flat_memory":
            return self._retrieve_flat(question, top_k)
        elif layers == ["qa"]:
            return self._retrieve_qa_only(question, top_k)
        elif set(layers) == {"qa", "segment"}:
            return self._retrieve_qa_segment(question, top_k)
        else:  # full_hesm
            return self._retrieve_full_hesm(question, top_k)

    # ─── Build helpers ────────────────────────────────────────────────────────

    def _build_flat_memory(self, conv_id: str, sessions: list[Session]) -> None:
        """Build a flat vector store of raw turns (no topic extraction)."""
        from hesm.vector_store import ChromaVectorStore

        flat_path = self._memory_root / f"ablation_flat_{conv_id}" / "chroma"
        flat_path.mkdir(parents=True, exist_ok=True)
        self._flat_vector_store = ChromaVectorStore(persist_path=str(flat_path))

        expected_count = sum(
            1
            for session in sessions
            for turn in session.turns
            if turn.dia_id and turn.text.strip()
        )
        if expected_count and self._flat_vector_store.count() >= expected_count:
            logger.info(
                "[AblationFlat] %s: memory already complete (%d turns), skipping build",
                conv_id,
                expected_count,
            )
            return

        total = 0
        for session in sessions:
            for turn in session.turns:
                if not turn.text.strip() or not turn.dia_id:
                    continue
                text = f"[{turn.speaker}]: {turn.text}"
                try:
                    embedding = self._embedder.embed(text)
                    safe_id = turn.dia_id.replace(":", "_")
                    self._flat_vector_store.upsert(
                        memory_type="qa",
                        memory_id=safe_id,
                        text=text,
                        embedding=embedding,
                        updated_at=turn.timestamp,
                        metadata={"dia_id": turn.dia_id},
                    )
                    total += 1
                except Exception as exc:
                    logger.warning("[AblationFlat] turn %s failed: %s", turn.dia_id, exc)

        logger.info("[AblationFlat] %s: indexed %d turns", conv_id, total)

    def _attach_hesm_storage(self, conv_id: str) -> None:
        """Open HESM storage built during Part 1 (read-only attachment)."""
        from hesm.extractor import TopicExtractor
        from hesm.retriever import HybridRetriever
        from hesm.storage import MemoryStorage
        from hesm.vector_store import ChromaVectorStore

        base = self._memory_root / f"hesm_{conv_id}"
        if not base.exists():
            raise FileNotFoundError(
                f"HESM storage not found at {base}. "
                "Run the main experiment (run_main.py) before running ablation."
            )

        self._storage = MemoryStorage(db_path=str(base / "memory.sqlite3"))
        self._vector_store = ChromaVectorStore(persist_path=str(base / "chroma"))
        method_models = _hesm_model_config(self._experiment_config)
        topic_cfg = method_models["topic_extraction"]
        retrieval_cfg = method_models["retrieval"]
        self._extractor = TopicExtractor(
            api_key=topic_cfg.get("api_key"),
            model=topic_cfg.get("model"),
            base_url=topic_cfg.get("base_url"),
            max_retries=topic_cfg.get("max_retries"),
            retry_delay=topic_cfg.get("retry_delay"),
        )

        if self._variant in {"full_hesm", "full_hesm_no_reranker"}:
            self._retriever = HybridRetriever(
                storage=self._storage,
                vector_store=self._vector_store,
                embedder=self._embedder,
                rerank_with_llm=bool(
                    self._variant_cfg.get("use_llm_reranker", True)
                ),
                retrieval_api_key=retrieval_cfg.get("api_key"),
                retrieval_model=retrieval_cfg.get("model"),
                retrieval_base_url=retrieval_cfg.get("base_url"),
                retrieval_max_retries=retrieval_cfg.get("max_retries"),
                retrieval_retry_delay=retrieval_cfg.get("retry_delay"),
                retrieval_config=self._cfg,
            )

    def _ensure_hesm_storage(
        self,
        conv_id: str,
        sessions: list[Session],
        speaker_a: str,
        speaker_b: str,
    ) -> None:
        """Build the shared HESM memory when ablation is run standalone."""
        base = self._memory_root / f"hesm_{conv_id}"
        db_path = base / "memory.sqlite3"
        chroma_path = base / "chroma"
        if db_path.exists() and chroma_path.exists():
            return

        logger.info(
            "[Ablation] HESM storage for %s is missing; building it now", conv_id
        )
        builder = HESMMemory(
            memory_root=self._memory_root,
            hesm_cfg=self._hesm_cfg,
            use_llm_summarizer=bool(
                self._hesm_cfg.get("use_llm_summarizer", True)
            ),
            use_llm_reranker=False,
            experiment_config=self._experiment_config,
        )
        builder.build_memory(conv_id, sessions, speaker_a, speaker_b)
        if not db_path.exists() or not chroma_path.exists():
            raise RuntimeError(f"Failed to build reusable HESM storage at {base}")

    # ─── Retrieval helpers ────────────────────────────────────────────────────

    def _retrieve_flat(self, question: str, top_k: int) -> RetrievalResult:
        if self._flat_vector_store is None:
            return RetrievalResult("", [], 0, {"error": "flat memory not built"})
        try:
            q_emb = self._embedder.embed(question)
            results = self._flat_vector_store.query(q_emb, memory_type="qa", top_k=top_k)
        except Exception as exc:
            return RetrievalResult("", [], 0, {"error": str(exc)})

        return self._pack_chroma_results(results)

    def _retrieve_qa_only(self, question: str, top_k: int) -> RetrievalResult:
        if self._vector_store is None:
            return RetrievalResult("", [], 0, {"error": "HESM storage not attached"})
        try:
            q_emb = self._embedder.embed(question)
            results = self._vector_store.query(q_emb, memory_type="qa", top_k=top_k)
        except Exception as exc:
            return RetrievalResult("", [], 0, {"error": str(exc)})

        # QA vectors store the turn text; extract dia_id from metadata
        return self._pack_chroma_results_with_storage(results)

    def _retrieve_qa_segment(self, question: str, top_k: int) -> RetrievalResult:
        if self._vector_store is None:
            return RetrievalResult("", [], 0, {"error": "HESM storage not attached"})
        try:
            q_emb = self._embedder.embed(question)
            qa_results = self._vector_store.query(q_emb, memory_type="qa", top_k=top_k)
            seg_results = self._vector_store.query(
                q_emb, memory_type="segment", top_k=top_k
            )
        except Exception as exc:
            return RetrievalResult("", [], 0, {"error": str(exc)})

        # Merge QA and Segment results; QA ids carry dia_ids via storage lookup
        qa_part = self._pack_chroma_results_with_storage(qa_results)
        seg_text = "\n".join(
            r.get("document", "") for r in seg_results if r.get("document")
        )
        merged_context = "\n\n".join(
            filter(None, [qa_part.context_text, seg_text])
        )
        return RetrievalResult(
            context_text=merged_context,
            retrieved_ids=qa_part.retrieved_ids,
            token_count=count_tokens(merged_context),
            raw_result={"qa_results": len(qa_results), "seg_results": len(seg_results)},
        )

    def _retrieve_full_hesm(self, question: str, top_k: int) -> RetrievalResult:
        del top_k
        if self._retriever is None or self._extractor is None:
            return RetrievalResult("", [], 0, {"error": "retriever not initialised"})
        cfg = self._hesm_cfg
        try:
            q_topic = self._extractor.extract(user_input=question)
            if isinstance(q_topic, list):
                q_topic = q_topic[0] if q_topic else {}

            result = self._retriever.recall(
                topic=q_topic.get("topic", ""),
                core_entity=q_topic.get("core_entity", ""),
                intent=q_topic.get("intent"),
                entities=q_topic.get("entities", []),
                query=question,
                query_confidence=float(q_topic.get("confidence", 0.0)),
                top_experience=int(cfg.get("top_experience", 3)),
                top_segment=int(cfg.get("top_segment", 5)),
                top_qa=int(cfg.get("top_qa", 8)),
            )
        except Exception as exc:
            return RetrievalResult("", [], 0, {"error": str(exc)})

        retrieved_ids: list[str] = []
        for qa in result.get("qas", []):
            tools = qa.get("tools") or []
            if tools and isinstance(tools, list) and isinstance(tools[0], dict):
                dia_id = tools[0].get("dia_id", "")
                if dia_id:
                    retrieved_ids.append(dia_id)

        ctx = result.get("context_text", "")
        return RetrievalResult(
            context_text=ctx,
            retrieved_ids=retrieved_ids,
            token_count=count_tokens(ctx),
            raw_result=result,
        )

    def _pack_chroma_results(self, results: list[dict]) -> RetrievalResult:
        """Pack flat vector store results (metadata has dia_id directly)."""
        ids: list[str] = []
        lines: list[str] = []
        for r in results:
            dia_id = r.get("metadata", {}).get("dia_id", "")
            if dia_id:
                ids.append(dia_id)
            doc = r.get("document", "")
            if doc:
                lines.append(doc)
        ctx = "\n".join(lines)
        return RetrievalResult(ctx, ids, count_tokens(ctx), {"num_results": len(results)})

    def _pack_chroma_results_with_storage(self, results: list[dict]) -> RetrievalResult:
        """
        Pack HESM QA vector results.
        Chroma ID format: "qa:{safe_dia_id}" — recover dia_id from storage.
        """
        ids: list[str] = []
        lines: list[str] = []
        for r in results:
            doc = r.get("document", "")
            if doc:
                lines.append(doc)
            # Chroma ID is "qa:{safe_id}" where safe_id = dia_id.replace(":", "_")
            chroma_id: str = r.get("chroma_id", "")
            if chroma_id.startswith("qa:"):
                safe_id = chroma_id[3:]  # strip "qa:" prefix
                # Recover original dia_id: "D1_3" → "D1:3"
                # Safe IDs replace ":" with "_" — but this is reversible only when
                # exactly one "_" separates Dnum from turn. Prefer storage lookup.
                if self._storage is not None:
                    qa = self._storage.get_qa(safe_id)
                    if qa:
                        tools = qa.get("tools") or []
                        if tools and isinstance(tools[0], dict):
                            dia_id = tools[0].get("dia_id", "")
                            if dia_id:
                                ids.append(dia_id)
                                continue
                # Fallback: reconstruct from safe_id if D{n}_{t} pattern
                import re
                m = re.match(r"D(\d+)_(\d+)", safe_id)
                if m:
                    ids.append(f"D{m.group(1)}:{m.group(2)}")

        ctx = "\n".join(lines)
        return RetrievalResult(ctx, ids, count_tokens(ctx), {"num_results": len(results)})
