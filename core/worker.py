"""Persistent background processing for summaries and parent-memory vectors."""

from __future__ import annotations

import logging
from threading import Event, RLock, Thread
from typing import Any

from .manager import MemoryManager
from .time_utils import format_timestamp


logger = logging.getLogger(__name__)


class MemoryDerivationWorker:
    """Drain idempotent outbox jobs without extending request latency."""

    def __init__(
        self,
        manager: MemoryManager,
        lock: RLock,
        *,
        poll_interval: float = 0.5,
        batch_size: int = 8,
        max_retries: int = 3,
    ) -> None:
        self.manager = manager
        self.storage = manager.storage
        self.lock = lock
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
        try:
            with self.lock:
                claimed = self.storage.mark_memory_job_processing(job_id, now)
                self.storage.commit()
                if not claimed:
                    return
                claimed_job = self.storage.get_memory_job(job_id) or job
                self._derive(claimed_job, now)
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
            logger.exception(
                "Memory derivation job failed job_id=%s job_type=%s memory_id=%s",
                job_id,
                job.get("job_type"),
                job.get("memory_id"),
            )

    def _derive(self, job: dict[str, Any], now: str) -> None:
        job_type = str(job["job_type"])
        memory_id = str(job["memory_id"])
        payload = job.get("payload") or {}

        # Drain jobs created by the previous split-task design through the new
        # aggregate handlers instead of failing an in-place upgrade.
        if job_type in {"sync_segment", "summarize_segment"}:
            payload = {
                **payload,
                "force_summary": (
                    bool(payload.get("force_summary"))
                    or job_type == "summarize_segment"
                ),
            }
            job_type = "update_segment"
        elif job_type in {"sync_experience", "summarize_experience"}:
            payload = {
                **payload,
                "force_summary": (
                    bool(payload.get("force_summary"))
                    or job_type == "summarize_experience"
                ),
            }
            job_type = "update_experience"

        if job_type == "update_segment":
            segment = self.storage.get_segment(memory_id)
            if not segment or segment.get("status") == "deleted":
                return
            qa_items = self.storage.list_qas_by_segment(memory_id)
            qa_count = len(qa_items)
            should_summarize = bool(qa_items) and (
                bool(payload.get("force_summary"))
                or qa_count - int(segment.get("last_summarized_qa_count") or 0)
                >= self.manager.segment_summary_qa_threshold
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
                        qa_items,
                    )
                )
                summary_state = summary.get("state") or {}
                if summary_state.get("status") == "completed":
                    desired_status = self._merge_status(
                        desired_status,
                        "completed",
                    )
            self.storage.reconcile_segment(
                segment_id=memory_id,
                desired_status=desired_status,
                updated_at=now,
                summary=summary,
                summarized_qa_count=qa_count if summary is not None else None,
            )
            refreshed = self.storage.get_segment(memory_id) or segment
            self.manager._enqueue_job(
                "embed_segment",
                "segment",
                memory_id,
                int(refreshed.get("version") or 0),
                now,
            )
            experience_id = str(refreshed.get("experience_id") or "")
            if experience_id:
                segments = self.storage.list_segments_by_experience_ids(
                    [experience_id]
                )
                child_revision = sum(
                    max(1, int(item.get("version") or 0)) for item in segments
                )
                self.manager._enqueue_job(
                    "update_experience",
                    "experience",
                    experience_id,
                    child_revision,
                    now,
                    payload={
                        "desired_status": "open",
                        "force_summary": (
                            summary is not None
                            or bool(payload.get("force_experience_summary"))
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
            should_summarize = (
                bool(payload.get("force_summary"))
                or segment_count
                - int(experience.get("last_summarized_segment_count") or 0)
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
                current_state = summary.get("current_state") or {}
                if current_state.get("status") == "completed":
                    desired_status = self._merge_status(
                        desired_status,
                        "completed",
                    )
            self.storage.reconcile_experience(
                experience_id=memory_id,
                desired_status=desired_status,
                updated_at=now,
                summary=summary,
                summarized_segment_count=(
                    segment_count if summary is not None else None
                ),
            )
            refreshed = self.storage.get_experience(memory_id) or experience
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
