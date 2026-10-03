"""Persistent background processing for summaries and parent-memory vectors."""

from __future__ import annotations

import logging
from threading import Event, RLock, Thread
from typing import Any

from .manager import MemoryManager
from .time_utils import format_timestamp
from .trajectory import TraceRepository


logger = logging.getLogger(__name__)


class MemoryDerivationWorker:
    """Drain idempotent outbox jobs without extending request latency."""

    def __init__(
        self,
        manager: MemoryManager,
        lock: RLock,
        *,
        traces: TraceRepository | None = None,
        poll_interval: float = 0.5,
        batch_size: int = 8,
        max_retries: int = 3,
    ) -> None:
        self.manager = manager
        self.storage = manager.storage
        self.lock = lock
        self.traces = traces
        self.poll_interval = max(0.05, float(poll_interval))
        self.batch_size = max(1, int(batch_size))
        self.max_retries = max(1, int(max_retries))
        self._wake = Event()
        self._stop = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        with self.lock:
            recovered = self.storage.recover_processing_memory_jobs()
            self.storage.commit()
        if recovered:
            logger.info("Recovered interrupted memory jobs count=%s", recovered)
        self._thread = Thread(
            target=self._run,
            name="hesm-memory-derivation",
            daemon=True,
        )
        self._thread.start()

    def notify(self) -> None:
        self._wake.set()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=max(0.0, float(timeout)))

    def _run(self) -> None:
        while not self._stop.is_set():
            processed = self.drain_once()
            if processed:
                self._wake.wait(self.poll_interval)
                self._wake.clear()
                continue
            self._wake.wait(self.poll_interval)
            self._wake.clear()

    def drain_once(self) -> int:
        """Process one snapshot of pending jobs and return the attempted count."""
        with self.lock:
            jobs = self.storage.list_pending_memory_jobs(self.batch_size)
        for job in jobs:
            if self._stop.is_set():
                break
            self._process_job(job)
        return len(jobs)

    def _process_job(self, job: dict[str, Any]) -> None:
        job_id = str(job["job_id"])
        now = format_timestamp()
        update_id: str | None = None
        memory_type = str(job.get("memory_type") or "")
        is_update = str(job.get("job_type") or "") in {"update_segment", "update_experience"}
        before: dict[str, Any] = {}
        claimed = False
        claimed_job = job
        try:
            with self.lock:
                claimed = self.storage.mark_memory_job_processing(job_id, now)
                self.storage.commit()
                if not claimed:
                    return
                claimed_job = self.storage.get_memory_job(job_id) or job
                job_payload = claimed_job.get("payload") or {}
                self.manager.trace_add_ids = list(job_payload.get("trace_add_ids") or [])
                self.manager.trace_chat_ids = list(job_payload.get("trace_chat_ids") or [])
                before = self._snapshot(claimed_job)
                should_distill, input_payload = self._distill_input(claimed_job) if is_update else (False, {})
                if is_update and memory_type in {"segment", "experience"}:
                    update_id = self.traces.start_update(
                        claimed_job,
                        before=before,
                        input_payload=input_payload,
                        model=str(getattr(self.manager.summarizer, "model", "")),
                    ) if self.traces else None
                    self.manager.trace_parent_segment_update_id = (
                        update_id if memory_type == "segment" else None
                    )
                self._derive(claimed_job, now)
                after = self._snapshot(claimed_job)
                if update_id and self.traces:
                    trace = getattr(self.manager.summarizer, "last_trace", {}) or {}
                    self.traces.finish_update(
                        memory_type,
                        update_id,
                        after=after,
                        prompt=str(trace.get("prompt") or "") if should_distill else "",
                        output=str(trace.get("output") or "") if should_distill else "",
                        elapsed_ms=float(trace.get("elapsed_ms") or 0) if should_distill else 0,
                        status="completed" if should_distill else "skipped",
                        skip_reason="threshold_not_reached" if not should_distill else "",
                        # Keep the trace update in the same SQLite transaction as
                        # the derived memory changes and outbox completion.
                        connection=self.storage.connection,
                    )
                self.storage.mark_memory_job_completed(
                    job_id,
                    int(claimed_job.get("target_version") or 0),
                    format_timestamp(),
                )
                self.storage.commit()
        except Exception as error:
            with self.lock:
                self.storage.rollback()
                self.storage.mark_memory_job_failed(
                    job_id,
                    str(error),
                    format_timestamp(),
                    max_retries=self.max_retries,
                )
                self.storage.commit()
            # If preparation failed before start_update(), still persist a
            # failed operation row for this claimed Segment/Experience update.
            if (
                claimed
                and is_update
                and memory_type in {"segment", "experience"}
                and self.traces
                and not update_id
            ):
                try:
                    update_id = self.traces.start_update(
                        claimed_job,
                        before=before,
                        input_payload={},
                        model=str(getattr(self.manager.summarizer, "model", "")),
                    )
                except Exception:
                    logger.exception(
                        "Could not record failed memory update job_id=%s", job_id
                    )
            if update_id and self.traces:
                self.traces.finish_update(
                    memory_type,
                    update_id,
                    after=before,
                    status="failed",
                    error_message=str(error),
                )
            logger.exception(
                "Memory derivation job failed job_id=%s job_type=%s memory_id=%s",
                job_id,
                job.get("job_type"),
                job.get("memory_id"),
            )
        finally:
            self.manager.trace_add_ids = []
            self.manager.trace_chat_ids = []
            self.manager.trace_parent_segment_update_id = None

    def _snapshot(self, job: dict[str, Any]) -> dict[str, Any]:
        kind = str(job.get("memory_type") or "")
        memory_id = str(job.get("memory_id") or "")
        if kind == "segment":
            item = self.storage.get_segment(memory_id)
        elif kind == "experience":
            item = self.storage.get_experience(memory_id)
        else:
            item = None
        if not item:
            return {}
        return dict(item)

    def _distill_input(self, job: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        memory_type = str(job.get("memory_type") or "")
        memory_id = str(job.get("memory_id") or "")
        payload = job.get("payload") or {}
        if memory_type == "segment":
            segment = self.storage.get_segment(memory_id) or {}
            delta = self.storage.list_unupdated_qas(segment) if segment else []
            should = bool(delta) and (
                bool(payload.get("force_summary"))
                or len(delta) >= self.manager.segment_summary_qa_threshold
            )
            return should, {"segment": segment, "qa_items": delta}
        if memory_type == "experience":
            experience = self.storage.get_experience(memory_id) or {}
            segments = self.storage.list_segments_by_experience_ids([memory_id])
            child_revision = sum(max(0, int(item.get("summary_version") or 0)) for item in segments)
            pending = max(0, child_revision - int(experience.get("last_summarized_child_revision") or 0))
            delta = [
                item for item in self.storage.list_unupdated_segments(experience)
                if int(item.get("summary_version") or 0) > 0
            ]
            should = pending >= self.manager.experience_summary_segment_threshold
            return should, {"experience": experience, "segments": delta, "all_segments": segments}
        return False, {}

    def _derive(self, job: dict[str, Any], now: str) -> None:
        job_type = str(job["job_type"])
        memory_id = str(job["memory_id"])
        payload = job.get("payload") or {}

        if job_type == "update_segment":
            segment = self.storage.get_segment(memory_id)
            if not segment or segment.get("status") == "deleted":
                return
            qa_items = self.storage.list_qas_by_segment(memory_id)
            qa_count = len(qa_items)
            delta_qas = self.storage.list_unupdated_qas(segment)
            should_summarize = bool(delta_qas) and (
                bool(payload.get("force_summary"))
                or len(delta_qas) >= self.manager.segment_summary_qa_threshold
            )
            summary = None
            desired_status = str(payload.get("desired_status") or "open")
            if should_summarize:
                summary_input = {
                    **segment,
                    "qa_ids": [str(qa["qa_id"]) for qa in qa_items],
                }
                summary = self.manager._summary_object(
                    self.manager.summarizer.summarize_segment(
                        summary_input,
                        delta_qas,
                    )
                )
            previous_qa_ids = list(segment.get("qa_ids") or [])
            previous_status = str(segment.get("status") or "open")
            self.storage.reconcile_segment(
                segment_id=memory_id,
                desired_status=desired_status,
                updated_at=now,
                summary=summary,
                summarized_qa_count=qa_count if summary is not None else None,
                last_updated_qa_id=(
                    str(delta_qas[-1]["qa_id"]) if summary is not None else None
                ),
            )
            refreshed = self.storage.get_segment(memory_id) or segment
            if (
                not previous_qa_ids
                or summary is not None
                or str(refreshed.get("status") or "") != previous_status
            ):
                self.manager._enqueue_job(
                    "embed_segment",
                    "segment",
                    memory_id,
                    int(refreshed.get("version") or 0),
                    now,
                )
            experience_id = str(refreshed.get("experience_id") or "")
            parent_changed = (
                not previous_qa_ids
                or summary is not None
                or str(refreshed.get("status") or "") != previous_status
            )
            if experience_id and parent_changed:
                segments = self.storage.list_segments_by_experience_ids(
                    [experience_id]
                )
                child_revision = sum(
                    max(0, int(item.get("summary_version") or 0))
                    for item in segments
                )
                self.manager._enqueue_job(
                    "update_experience",
                    "experience",
                    experience_id,
                    child_revision,
                    now,
                    payload={
                        "desired_status": "open",
                        "allow_summary_completion": bool(
                            payload.get("allow_experience_completion")
                        ),
                    },
                )
            return

        if job_type == "update_experience":
            experience = self.storage.get_experience(memory_id)
            if not experience or experience.get("status") == "deleted":
                return
            segments = sorted(
                self.storage.list_segments_by_experience_ids([memory_id]),
                key=lambda item: (
                    str(item.get("created_at") or ""),
                    str(item.get("segment_id") or ""),
                ),
            )
            segment_count = len(segments)
            child_revision = sum(
                max(0, int(segment.get("summary_version") or 0))
                for segment in segments
            )
            delta_segments = [
                segment
                for segment in self.storage.list_unupdated_segments(experience)
                if int(segment.get("summary_version") or 0) > 0
            ]
            pending_segment_updates = max(
                0,
                child_revision
                - int(experience.get("last_summarized_child_revision") or 0),
            )
            should_summarize = (
                pending_segment_updates
                >= self.manager.experience_summary_segment_threshold
            )
            summary = None
            desired_status = str(payload.get("desired_status") or "open")
            if should_summarize:
                summary_input = {
                    **experience,
                    "segment_ids": [
                        str(segment["segment_id"]) for segment in segments
                    ],
                    "intents_link": list(
                        dict.fromkeys(
                            str(segment.get("intent") or "").strip()
                            for segment in segments
                            if str(segment.get("intent") or "").strip()
                        )
                    ),
                }
                summary = self.manager._summary_object(
                    self.manager.summarizer.summarize_experience(
                        summary_input,
                        segments,
                    )
                )
                if (
                    bool(payload.get("allow_summary_completion"))
                    and (summary.get("current_state") or {}).get("status")
                    == "completed"
                ):
                    desired_status = self._merge_status(
                        desired_status,
                        "completed",
                    )
            previous_segment_ids = list(experience.get("segment_ids") or [])
            previous_intents = list(experience.get("intents_link") or [])
            previous_status = str(experience.get("status") or "open")
            marker_segment = (
                delta_segments[-1]
                if delta_segments
                else max(
                    segments,
                    key=lambda item: (
                        str(item.get("updated_at") or ""),
                        str(item.get("segment_id") or ""),
                    ),
                    default=None,
                )
            )
            self.storage.reconcile_experience(
                experience_id=memory_id,
                desired_status=desired_status,
                updated_at=now,
                summary=summary,
                summarized_segment_count=(
                    segment_count if summary is not None else None
                ),
                summarized_child_revision=(
                    child_revision if summary is not None else None
                ),
                last_updated_segment_id=(
                    str(marker_segment["segment_id"])
                    if summary is not None and marker_segment
                    else None
                ),
                last_updated_segment_at=(
                    str(marker_segment.get("updated_at") or "")
                    if summary is not None and marker_segment
                    else None
                ),
            )
            refreshed = self.storage.get_experience(memory_id) or experience
            if (
                summary is not None
                or list(refreshed.get("segment_ids") or []) != previous_segment_ids
                or list(refreshed.get("intents_link") or []) != previous_intents
                or str(refreshed.get("status") or "") != previous_status
            ):
                self.manager._enqueue_job(
                    "embed_experience",
                    "experience",
                    memory_id,
                    int(refreshed.get("version") or 0),
                    now,
                )
            return

        if job_type == "recall_experience_history":
            experience = self.storage.get_experience(memory_id)
            if not experience or experience.get("status") == "deleted":
                return
            recaller = self.manager.experience_recaller
            if recaller is None:
                raise RuntimeError("experience_recaller is not configured")
            recalled = recaller.recall(
                topic=str(payload.get("topic") or experience.get("topic") or ""),
                core_entity=str(
                    payload.get("core_entity")
                    or experience.get("core_entity")
                    or ""
                ),
                query=str(payload.get("query") or ""),
                intent=str(payload.get("intent") or ""),
            )
            experience["history_experience"] = (
                recalled.get("history_experience", {})
                if isinstance(recalled, dict)
                else {}
            )
            self.storage.update_experience_history(
                experience_id=memory_id,
                history_experience=experience["history_experience"],
                updated_at=now,
            )
            return

        if job_type == "embed_segment":
            if self.storage.get_segment(memory_id):
                self.manager.upsert_segment_vector(memory_id)
            return

        if job_type == "embed_experience":
            if self.storage.get_experience(memory_id):
                self.manager.upsert_experience_content_vector(memory_id)
            return

        if job_type == "create_experience_route_vector":
            experience = self.storage.get_experience(memory_id)
            if experience and experience.get("status") != "deleted":
                self.manager.create_experience_route_vector(memory_id, payload)
            return

        if job_type == "embed_experience_route":
            logger.info(
                "Ignored legacy Experience route refresh job memory_id=%s",
                memory_id,
            )
            return

        raise ValueError(f"Unsupported memory derivation job: {job_type}")

    @staticmethod
    def _merge_status(*statuses: str) -> str:
        priority = {"open": 0, "completed": 1, "deleted": 2}
        return max(
            (str(status or "open") for status in statuses),
            key=lambda status: priority.get(status, 0),
        )


__all__ = ["MemoryDerivationWorker"]
