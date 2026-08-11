"""
QA Runner — the central execution engine for LoCoMo benchmark experiments.

Orchestrates the complete pipeline for one method across all conversations:
  1. Build memory (session by session, temporal order)
  2. For each question: retrieve → generate answer → evaluate → checkpoint
  3. Aggregate metrics across all conversations
  4. Save per-method metrics to disk

Supports:
  - Checkpoint / resume  (skips already-completed questions)
  - Latency measurement  (per-question retrieval latency)
  - Pluggable top_k_values (for Recall/Precision/Accuracy@K)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from experiments.locomo.data.loader import LoCoMoConversation
from experiments.locomo.evaluation.aggregator import MethodMetrics, aggregate
from experiments.locomo.evaluation.f1 import compute_f1
from experiments.locomo.evaluation.retrieval_metrics import (
    RetrievalMetrics,
    average_retrieval_metrics,
    compute_retrieval_metrics,
)
from experiments.locomo.evaluation.token_metrics import (
    compute_compression_ratio,
    count_tokens,
)
from experiments.locomo.methods.base import LLMAnswerGenerator, MemorySystem, RetrievalResult
from experiments.locomo.runner.checkpoint import Checkpoint
from experiments.locomo.runner.stage_logger import MethodStageLogger

logger = logging.getLogger(__name__)


class _WarningCollector(logging.Handler):
    """Collect warnings emitted by methods that tolerate per-item failures."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []
        self.thread_id = threading.get_ident()

    def emit(self, record: logging.LogRecord) -> None:
        if record.thread != self.thread_id:
            return
        try:
            self.messages.append(self.format(record))
        except Exception:
            self.messages.append(record.getMessage())


class _UnavailableMemory:
    """Query adapter used to route worker setup failures through normal logging."""

    def __init__(self, error: str) -> None:
        self.error = error

    def retrieve(self, question: str, top_k: int = 5) -> RetrievalResult:
        return RetrievalResult("", [], 0, {"error": self.error})


class QARunner:
    """
    Runs the full QA pipeline for one memory system across all conversations.

    Args:
        method:          MemorySystem instance to evaluate
        answer_generator: shared LLMAnswerGenerator (same for all methods)
        judge:           LLMJudge instance (pass None to skip judge scoring)
        output_dir:      directory for checkpoint files (answers/)
        metrics_dir:     directory for aggregated metric files (metrics/)
        logs_dir:        directory for per-method four-stage logs
        qa_workers:      concurrent query workers; 1 keeps serial behavior
        method_factory:  creates a thread-local memory reader for parallel QA
        top_k_values:    K values for Recall/Precision/Accuracy@K
        token_encoding:  tiktoken encoding name
    """

    def __init__(
        self,
        method: MemorySystem,
        answer_generator: LLMAnswerGenerator,
        judge: Any,   # LLMJudge | None
        output_dir: str | Path,
        metrics_dir: str | Path,
        logs_dir: str | Path | None = None,
        qa_workers: int = 1,
        method_factory: Callable[[], MemorySystem] | None = None,
        answer_generator_factory: Callable[[], Any] | None = None,
        judge_factory: Callable[[], Any] | None = None,
        top_k_values: list[int] | None = None,
        token_encoding: str = "cl100k_base",
    ) -> None:
        self.method = method
        self.answer_generator = answer_generator
        self.judge = judge
        self.output_dir = Path(output_dir)
        self.metrics_dir = Path(metrics_dir)
        self.logs_dir = Path(logs_dir) if logs_dir is not None else self.output_dir.parent / "logs"
        self.qa_workers = max(1, int(qa_workers))
        self.method_factory = method_factory
        self.answer_generator_factory = answer_generator_factory
        self.judge_factory = judge_factory
        self._worker_local = threading.local()
        self.top_k_values = top_k_values or [1, 3, 5]
        self.token_encoding = token_encoding

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.stage_logger = MethodStageLogger(self.logs_dir, self.method.method_name)
        if self.qa_workers > 1 and (
            self.method_factory is None or self.answer_generator_factory is None
            or (self.judge is not None and self.judge_factory is None)
        ):
            logger.warning(
                "[%s] qa_workers=%d requested without worker factories; "
                "falling back to serial QA",
                self.method.method_name,
                self.qa_workers,
            )
            self.qa_workers = 1

    # ─── Public: run all conversations ───────────────────────────────────────

    def run(self, conversations: list[LoCoMoConversation]) -> MethodMetrics:
        """
        Run the method on all conversations and return aggregated MethodMetrics.
        Already-completed conversations (checkpoint = 'complete') are skipped.
        """
        method_name = self.method.method_name
        logger.info("=== [%s] Starting evaluation on %d conversations ===",
                    method_name, len(conversations))

        all_records: list[dict[str, Any]] = []

        for conv in conversations:
            conv_records = self._run_conversation(conv)
            all_records.extend(conv_records)
            logger.info(
                "[%s] %s done: %d questions processed",
                method_name, conv.conv_id, len(conv_records),
            )

        # Aggregate and save
        metrics = aggregate(method_name, all_records, self.top_k_values)
        self._save_metrics(metrics)
        logger.info(
            "=== [%s] Evaluation complete: F1=%.4f Judge=%.2f ===",
            method_name, metrics.avg_f1, metrics.avg_judge_score,
        )
        return metrics

    # ─── Internal: single conversation ────────────────────────────────────────

    def _run_conversation(
        self, conv: LoCoMoConversation
    ) -> list[dict[str, Any]]:
        """Process one conversation; returns list of QARecord dicts."""
        method_name = self.method.method_name
        ckpt = Checkpoint(self.output_dir, method_name, conv.conv_id)

        if ckpt.is_complete():
            all_answered = all(
                ckpt.is_answered(index, qa_item.question)
                for index, qa_item in enumerate(conv.questions)
            )
            if all_answered:
                self.stage_logger.event(
                    "BUILD",
                    "SKIPPED",
                    message="Conversation already has successful query checkpoints",
                    conv_id=conv.conv_id,
                    questions=len(conv.questions),
                )
                for index, qa_item in enumerate(conv.questions):
                    self.stage_logger.skipped_query(
                        conv_id=conv.conv_id,
                        query_index=index,
                        question=qa_item.question,
                        reason="successful checkpoint",
                    )
                logger.info(
                    "[%s] %s: answers file already contains all %d answers, skipping",
                    method_name,
                    conv.conv_id,
                    len(conv.questions),
                )
                return ckpt.load_all_records_ordered(len(conv.questions))
            logger.warning(
                "[%s] %s: checkpoint is marked complete but contains missing or "
                "mismatched answers; validating questions individually",
                method_name,
                conv.conv_id,
            )

        completed_count = ckpt.num_completed()
        if completed_count > 0:
            logger.info(
                "[%s] %s: resuming from question %d/%d",
                method_name, conv.conv_id, completed_count, len(conv.questions),
            )

        # Build memory before retrying unfinished queries. Persistent methods may
        # attach an existing store and return quickly.
        build_started = time.perf_counter()
        build_warnings = _WarningCollector()
        logging.getLogger().addHandler(build_warnings)
        self.stage_logger.event(
            "BUILD",
            "STARTED",
            message="Building or attaching conversation memory",
            conv_id=conv.conv_id,
            sessions=len(conv.sessions),
            turns=len(conv.all_turns),
            pending_questions=len(conv.questions) - completed_count,
        )
        try:
            self.method.reset()
            self.method.build_memory(
                conv_id=conv.conv_id,
                sessions=conv.sessions,
                speaker_a=conv.speaker_a,
                speaker_b=conv.speaker_b,
            )
        except Exception as exc:
            logging.getLogger().removeHandler(build_warnings)
            elapsed_ms = (time.perf_counter() - build_started) * 1000
            reason = f"{type(exc).__name__}: {exc}"
            self.stage_logger.event(
                "BUILD",
                "FAILED",
                message="Memory build failed; unfinished queries will be retried",
                conv_id=conv.conv_id,
                elapsed_ms=round(elapsed_ms, 3),
                error=reason,
            )
            for index, qa_item in enumerate(conv.questions):
                if ckpt.is_answered(index, qa_item.question):
                    continue
                for stage in ("RETRIEVAL", "ANSWER", "JUDGE"):
                    self.stage_logger.event(
                        stage,
                        "SKIPPED",
                        message="Memory build failed",
                        conv_id=conv.conv_id,
                        query_index=index,
                        question=qa_item.question,
                        reason=reason,
                    )
            logger.exception("[%s] %s build failed", method_name, conv.conv_id)
            return ckpt.load_all_records_ordered(len(conv.questions))

        logging.getLogger().removeHandler(build_warnings)

        build_elapsed_ms = (time.perf_counter() - build_started) * 1000
        self.stage_logger.event(
            "BUILD",
            "PARTIAL" if build_warnings.messages else "SUCCESS",
            message=(
                "Conversation memory is ready with item-level failures"
                if build_warnings.messages
                else "Conversation memory is ready"
            ),
            conv_id=conv.conv_id,
            elapsed_ms=round(build_elapsed_ms, 3),
            sessions=len(conv.sessions),
            turns=len(conv.all_turns),
            failure_reasons=build_warnings.messages[:100],
            suppressed_failure_count=max(0, len(build_warnings.messages) - 100),
        )

        # Pre-compute total conversation tokens for compression ratio
        total_conv_tokens = count_tokens(
            conv.total_turn_text(), self.token_encoding
        )

        # Default top_k for retrieval (used during memory building)
        default_top_k = max(self.top_k_values)

        pending_queries: list[tuple[int, Any]] = []
        for i, qa_item in enumerate(conv.questions):
            if ckpt.is_answered(i, qa_item.question):
                self.stage_logger.skipped_query(
                    conv_id=conv.conv_id,
                    query_index=i,
                    question=qa_item.question,
                    reason="successful checkpoint",
                )
                logger.info(
                    "[%s] %s: question %d/%d already answered, skipping: %s",
                    method_name,
                    conv.conv_id,
                    i + 1,
                    len(conv.questions),
                    qa_item.question[:100],
                )
                continue
            if ckpt.is_done(i):
                logger.warning(
                    "[%s] %s: question %d/%d has an invalid or mismatched saved "
                    "answer; running again: %s",
                    method_name,
                    conv.conv_id,
                    i + 1,
                    len(conv.questions),
                    qa_item.question[:100],
                )
            pending_queries.append((i, qa_item))

        effective_workers = 1 if build_warnings.messages else self.qa_workers
        if build_warnings.messages and self.qa_workers > 1:
            logger.warning(
                "[%s] %s: memory build was partial; using serial QA for safety",
                method_name,
                conv.conv_id,
            )

        if effective_workers <= 1 or len(pending_queries) <= 1:
            for i, qa_item in pending_queries:
                record = self._run_single_qa(
                    query_index=i,
                    qa_item=qa_item,
                    conv=conv,
                    default_top_k=default_top_k,
                    total_conv_tokens=total_conv_tokens,
                )
                ckpt.save_record(
                    i,
                    record,
                    total_questions=len(conv.questions),
                    successful=record.get("query_status") == "success",
                )
        else:
            logger.info(
                "[%s] %s: running %d pending queries with %d workers",
                method_name,
                conv.conv_id,
                len(pending_queries),
                effective_workers,
            )
            with ThreadPoolExecutor(
                max_workers=effective_workers,
                thread_name_prefix=f"{method_name}-{conv.conv_id}",
            ) as executor:
                futures: dict[Future[dict[str, Any]], int] = {
                    executor.submit(
                        self._run_parallel_query,
                        query_index=i,
                        qa_item=qa_item,
                        conv=conv,
                        default_top_k=default_top_k,
                        total_conv_tokens=total_conv_tokens,
                    ): i
                    for i, qa_item in pending_queries
                }
                for future in as_completed(futures):
                    i = futures[future]
                    record = future.result()
                    # Only the owning conversation thread writes checkpoints.
                    ckpt.save_record(
                        i,
                        record,
                        total_questions=len(conv.questions),
                        successful=record.get("query_status") == "success",
                    )

        return ckpt.load_all_records_ordered(len(conv.questions))

    # ─── Internal: single QA ──────────────────────────────────────────────────

    def _run_parallel_query(
        self,
        *,
        query_index: int,
        qa_item: Any,
        conv: LoCoMoConversation,
        default_top_k: int,
        total_conv_tokens: int,
    ) -> dict[str, Any]:
        """Run one query with resources owned by the current worker thread."""
        try:
            components = self._get_worker_components(conv)
        except Exception as exc:
            reason = f"Worker initialisation failed: {type(exc).__name__}: {exc}"
            components = (_UnavailableMemory(reason), self.answer_generator, None)
        return self._run_single_qa(
            query_index=query_index,
            qa_item=qa_item,
            conv=conv,
            default_top_k=default_top_k,
            total_conv_tokens=total_conv_tokens,
            components=components,
        )

    def _get_worker_components(
        self,
        conv: LoCoMoConversation,
    ) -> tuple[Any, Any, Any]:
        """Create one memory reader, answer client, and Judge per worker thread."""
        state = getattr(self._worker_local, "state", None)
        if state is not None and state["conv_id"] == conv.conv_id:
            return state["components"]

        if self.method_factory is None or self.answer_generator_factory is None:
            raise RuntimeError("parallel QA worker factories are not configured")

        worker_method = self.method_factory()
        worker_method.reset()
        worker_method.build_memory(
            conv_id=conv.conv_id,
            sessions=conv.sessions,
            speaker_a=conv.speaker_a,
            speaker_b=conv.speaker_b,
        )
        worker_answer_generator = self.answer_generator_factory()
        worker_judge = (
            self.judge_factory()
            if self.judge is not None and self.judge_factory is not None
            else None
        )
        components = (worker_method, worker_answer_generator, worker_judge)
        self._worker_local.state = {
            "conv_id": conv.conv_id,
            "components": components,
        }
        return components

    def _run_single_qa(
        self,
        query_index: int,
        qa_item: Any,   # QAItem
        conv: LoCoMoConversation,
        default_top_k: int,
        total_conv_tokens: int,
        components: tuple[Any, Any, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute and log retrieval, answer, and Judge stages for one query."""

        active_method, active_answer_generator, active_judge = components or (
            self.method,
            self.answer_generator,
            self.judge,
        )

        common = {
            "conv_id": conv.conv_id,
            "query_index": query_index,
            "question": qa_item.question,
        }
        stage_status: dict[str, dict[str, Any]] = {}
        retrieval = RetrievalResult("", [], 0, {})
        prediction = ""
        judge_score = -1
        answer_latency_ms = 0.0
        judge_latency_ms = 0.0

        # ── Part 2/4: Retrieval ────────────────────────────────────────────────
        self.stage_logger.event(
            "RETRIEVAL", "STARTED", message="Retrieving memory", **common
        )
        retrieval_started = time.perf_counter()
        try:
            retrieval = active_method.retrieve(
                question=qa_item.question, top_k=default_top_k
            )
            retrieval_latency_ms = (time.perf_counter() - retrieval_started) * 1000
            retrieval_error = retrieval.raw_result.get("error")
            if retrieval_error:
                raise RuntimeError(str(retrieval_error))
            stage_status["retrieval"] = {
                "status": "success",
                "elapsed_ms": retrieval_latency_ms,
            }
            self.stage_logger.event(
                "RETRIEVAL",
                "SUCCESS",
                message="Retrieval completed",
                **common,
                elapsed_ms=round(retrieval_latency_ms, 3),
                retrieved_ids=retrieval.retrieved_ids,
                retrieved_count=len(retrieval.retrieved_ids),
                retrieved_tokens=retrieval.token_count,
                context_preview=retrieval.context_text[:1000],
            )
        except Exception as exc:
            retrieval_latency_ms = (time.perf_counter() - retrieval_started) * 1000
            reason = f"{type(exc).__name__}: {exc}"
            stage_status["retrieval"] = {
                "status": "failed",
                "elapsed_ms": retrieval_latency_ms,
                "error": reason,
            }
            stage_status["answer"] = {
                "status": "skipped",
                "reason": "retrieval failed",
            }
            stage_status["judge"] = {
                "status": "skipped",
                "reason": "retrieval failed",
            }
            self.stage_logger.event(
                "RETRIEVAL",
                "FAILED",
                message="Retrieval failed",
                **common,
                elapsed_ms=round(retrieval_latency_ms, 3),
                error=reason,
            )
            self.stage_logger.event(
                "ANSWER", "SKIPPED", message="Retrieval failed", **common, reason=reason
            )
            self.stage_logger.event(
                "JUDGE", "SKIPPED", message="Retrieval failed", **common, reason=reason
            )

        # ── Part 3/4: LLM answer generation ──────────────────────────────────
        if stage_status["retrieval"]["status"] == "success":
            self.stage_logger.event(
                "ANSWER", "STARTED", message="Generating LLM answer", **common
            )
            answer_started = time.perf_counter()
            try:
                prediction = active_answer_generator.generate(
                    question=qa_item.question,
                    context=retrieval.context_text,
                    speaker_a=conv.speaker_a,
                    speaker_b=conv.speaker_b,
                )
                answer_latency_ms = (time.perf_counter() - answer_started) * 1000
                if not prediction.strip():
                    error = getattr(active_answer_generator, "last_error", None)
                    raise RuntimeError(error or "LLM returned an empty answer")
                stage_status["answer"] = {
                    "status": "success",
                    "elapsed_ms": answer_latency_ms,
                }
                self.stage_logger.event(
                    "ANSWER",
                    "SUCCESS",
                    message="LLM answer generated",
                    **common,
                    elapsed_ms=round(answer_latency_ms, 3),
                    prediction=prediction,
                )
            except Exception as exc:
                answer_latency_ms = (time.perf_counter() - answer_started) * 1000
                reason = f"{type(exc).__name__}: {exc}"
                stage_status["answer"] = {
                    "status": "failed",
                    "elapsed_ms": answer_latency_ms,
                    "error": reason,
                }
                stage_status["judge"] = {
                    "status": "skipped",
                    "reason": "answer generation failed",
                }
                self.stage_logger.event(
                    "ANSWER",
                    "FAILED",
                    message="LLM answer generation failed",
                    **common,
                    elapsed_ms=round(answer_latency_ms, 3),
                    error=reason,
                )
                self.stage_logger.event(
                    "JUDGE",
                    "SKIPPED",
                    message="Answer generation failed",
                    **common,
                    reason=reason,
                )

        ground_truth = qa_item.answer_str()

        # ── Part 4/4: Judge ───────────────────────────────────────────────────
        if stage_status.get("answer", {}).get("status") == "success":
            if active_judge is None:
                stage_status["judge"] = {
                    "status": "skipped",
                    "reason": "Judge disabled",
                }
                self.stage_logger.event(
                    "JUDGE", "SKIPPED", message="Judge disabled", **common
                )
            else:
                self.stage_logger.event(
                    "JUDGE", "STARTED", message="Evaluating answer", **common
                )
                judge_started = time.perf_counter()
                try:
                    judge_score = active_judge.judge(
                        question=qa_item.question,
                        ground_truth=ground_truth,
                        prediction=prediction,
                    )
                    judge_latency_ms = (time.perf_counter() - judge_started) * 1000
                    if judge_score < 0:
                        error = getattr(active_judge, "last_error", None)
                        raise RuntimeError(error or "Judge returned failure score -1")
                    stage_status["judge"] = {
                        "status": "success",
                        "elapsed_ms": judge_latency_ms,
                        "score": judge_score,
                    }
                    self.stage_logger.event(
                        "JUDGE",
                        "SUCCESS",
                        message="Judge evaluation completed",
                        **common,
                        elapsed_ms=round(judge_latency_ms, 3),
                        ground_truth=ground_truth,
                        prediction=prediction,
                        judge_score=judge_score,
                    )
                except Exception as exc:
                    judge_latency_ms = (time.perf_counter() - judge_started) * 1000
                    reason = f"{type(exc).__name__}: {exc}"
                    judge_score = -1
                    stage_status["judge"] = {
                        "status": "failed",
                        "elapsed_ms": judge_latency_ms,
                        "error": reason,
                    }
                    self.stage_logger.event(
                        "JUDGE",
                        "FAILED",
                        message="Judge evaluation failed",
                        **common,
                        elapsed_ms=round(judge_latency_ms, 3),
                        error=reason,
                        ground_truth=ground_truth,
                        prediction=prediction,
                    )

        query_success = (
            stage_status.get("retrieval", {}).get("status") == "success"
            and stage_status.get("answer", {}).get("status") == "success"
            and stage_status.get("judge", {}).get("status") in {"success", "skipped"}
        )

        # ── Metrics ───────────────────────────────────────────────────────────
        f1_scores = compute_f1(prediction, ground_truth)

        retrieval_metrics_per_k = compute_retrieval_metrics(
            retrieved_ids=retrieval.retrieved_ids,
            evidence_ids=qa_item.evidence,
            k_values=self.top_k_values,
        )
        retrieval_metrics_dict = {
            str(k): {
                "recall": m.recall,
                "precision": m.precision,
                "f1": m.f1,
                "accuracy": m.accuracy,
            }
            for k, m in retrieval_metrics_per_k.items()
        }

        compression_ratio = compute_compression_ratio(
            retrieved_tokens=retrieval.token_count,
            total_conversation_tokens=total_conv_tokens,
        )

        return {
            "question": qa_item.question,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "retrieved_ids": retrieval.retrieved_ids,
            "retrieved_context": retrieval.context_text,
            "retrieved_tokens": retrieval.token_count,
            "evidence": qa_item.evidence,
            "category": qa_item.category,
            "f1": f1_scores["f1"],
            "f1_precision": f1_scores["precision"],
            "f1_recall": f1_scores["recall"],
            "judge_score": judge_score,
            "retrieval_metrics": retrieval_metrics_dict,
            "total_conversation_tokens": total_conv_tokens,
            "compression_ratio": compression_ratio,
            "latency_ms": retrieval_latency_ms,
            "answer_latency_ms": answer_latency_ms,
            "judge_latency_ms": judge_latency_ms,
            "query_status": "success" if query_success else "failed",
            "stage_status": stage_status,
        }

    # ─── Internal: persist metrics ─────────────────────────────────────────────

    def _save_metrics(self, metrics: MethodMetrics) -> None:
        path = self.metrics_dir / f"{metrics.method_name}_metrics.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info("Saved metrics → %s", path)


# ─── Cache-aware runner for Part 3 ───────────────────────────────────────────

class CachedEmbedder:
    """
    In-memory embedding cache wrapper around BailianEmbedder.
    Tracks hit/miss counts for cache hit rate computation.
    """

    def __init__(self, embedder: Any) -> None:
        self._inner = embedder
        self._cache: dict[str, list[float]] = {}
        self.hits = 0
        self.misses = 0

    def embed(self, text: str) -> list[float]:
        if text in self._cache:
            self.hits += 1
            return self._cache[text]
        emb = self._inner.embed(text)
        self._cache[text] = emb
        self.misses += 1
        return emb

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def reset_stats(self) -> None:
        self.hits = 0
        self.misses = 0


class CacheEvalRunner:
    """
    Measures latency and cache effectiveness for HESM with cache ON vs OFF.

    Runs retrieval on a sample of questions, twice:
      1. cache_off: use_cache=False for HybridRetriever, no embedding cache
      2. cache_on:  use_cache=True  for HybridRetriever, CachedEmbedder wrapper

    Computes P50/P90/P99/Average latency, cache hit rate, and LLM call reduction.
    """

    def __init__(
        self,
        hesm_memory: Any,  # HESMMemory
        output_dir: str | Path,
        percentiles: list[int] | None = None,
        num_samples: int = 50,
    ) -> None:
        self.hesm = hesm_memory
        self.output_dir = Path(output_dir)
        self.percentiles = percentiles or [50, 90, 99]
        self.num_samples = num_samples
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        conversations: list[LoCoMoConversation],
    ) -> dict[str, Any]:
        """Run cache ON/OFF comparison. Returns result dict."""
        import statistics

        # Collect sample questions (up to num_samples across conversations)
        samples: list[tuple[str, LoCoMoConversation]] = []
        for conv in conversations:
            for qa in conv.questions:
                samples.append((qa.question, conv))
                if len(samples) >= self.num_samples:
                    break
            if len(samples) >= self.num_samples:
                break

        results: dict[str, Any] = {}

        for mode in ("cache_off", "cache_on"):
            latencies: list[float] = []
            use_cache = mode == "cache_on"

            # Build memory for the first conversation (or use pre-built)
            conv = conversations[0] if conversations else None
            if conv is None:
                continue

            self.hesm.reset()
            self.hesm.build_memory(conv.conv_id, conv.sessions, conv.speaker_a, conv.speaker_b)

            for question, _ in samples:
                t0 = time.perf_counter()
                self.hesm.retrieve(question=question, top_k=5, use_cache=use_cache)
                latencies.append((time.perf_counter() - t0) * 1000)

            def _percentile(data: list[float], p: int) -> float:
                if not data:
                    return 0.0
                sorted_d = sorted(data)
                idx = max(0, int(len(sorted_d) * p / 100) - 1)
                return sorted_d[idx]

            results[mode] = {
                "avg_latency_ms": statistics.mean(latencies) if latencies else 0.0,
                "latency_percentiles": {
                    f"p{p}": _percentile(latencies, p) for p in self.percentiles
                },
                "num_samples": len(latencies),
            }

        # Compute reduction ratios
        off = results.get("cache_off", {})
        on = results.get("cache_on", {})
        avg_off = off.get("avg_latency_ms", 1.0)
        avg_on = on.get("avg_latency_ms", avg_off)
        results["latency_reduction"] = (avg_off - avg_on) / avg_off if avg_off > 0 else 0.0

        # Save
        out_path = self.output_dir / "cache_raw.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info("Cache eval raw results → %s", out_path)

        return results
