from __future__ import annotations

import sqlite3
from threading import RLock

from core.embedder import HashingEmbedder
from core.manager import MemoryManager
from core.retriever import HybridRetriever
from core.storage import MemoryStorage
from core.summarizer import TemplateSummarizer
from core.vector_store import ChromaVectorStore
from core.worker import MemoryDerivationWorker


def _experience():
    return {
        "experience_id": "exp-1",
        "topic": "旅行",
        "core_entity": "上海",
        "intents_link": ["回忆"],
        "segment_ids": ["seg-1"],
        "summary": {},
        "status": "open",
        "created_at": "2026-01-01 00:00:00",
        "updated_at": "2026-01-01 00:00:00",
        "version": 1,
        "last_summarized_segment_count": 1,
        "history_experience": {},
    }


def _segment():
    return {
        "segment_id": "seg-1",
        "topic": "旅行",
        "intent": "回忆",
        "core_entity": "上海",
        "qa_ids": [],
        "summary": {},
        "experience_id": "exp-1",
        "created_at": "2026-01-01 00:00:00",
        "updated_at": "2026-01-01 00:00:00",
        "version": 1,
        "last_summarized_qa_count": 0,
        "status": "open",
    }


class RecordingManager(MemoryManager):
    def __init__(self, storage):
        self.storage = storage
        self.segment_summary_qa_threshold = 1
        self.experience_summary_segment_threshold = 99
        self.min_segment_qas = 2
        self.qa_vector_updates = []

    def route_experience(self, **kwargs):
        return self.storage.get_experience("exp-1"), self.storage.get_segment("seg-1")

    def upsert_qa_vector(self, qa_id):
        self.qa_vector_updates.append(qa_id)

    def upsert_segment_vector(self, segment_id):
        raise AssertionError("Segment vectors must not run in the request path")

    def upsert_experience_vector(self, experience_id):
        raise AssertionError("Experience vectors must not run in the request path")


class EmptyHistoryRecaller:
    def __init__(self):
        self.calls = []

    def recall(self, **kwargs):
        self.calls.append(kwargs)
        return {"experiences": {}, "history_experience": {}}


class CountingSummarizer(TemplateSummarizer):
    def __init__(self):
        self.segment_calls = 0
        self.experience_calls = 0

    def summarize_segment(self, segment, qa_items):
        self.segment_calls += 1
        return super().summarize_segment(segment, qa_items)

    def summarize_experience(self, experience, segments):
        self.experience_calls += 1
        return super().summarize_experience(experience, segments)


class CapturingSummarizer(CountingSummarizer):
    def __init__(self):
        super().__init__()
        self.segment_batches = []
        self.experience_batches = []

    def summarize_segment(self, segment, qa_items):
        self.segment_batches.append([item["qa_id"] for item in qa_items])
        return super().summarize_segment(segment, qa_items)

    def summarize_experience(self, experience, segments):
        self.experience_batches.append(
            [item["segment_id"] for item in segments]
        )
        return super().summarize_experience(experience, segments)


class CompletingSummarizer(TemplateSummarizer):
    def summarize_segment(self, segment, qa_items):
        return {
            "goal": "test",
            "key_facts": [],
            "state_changes": [],
            "state": {"status": "completed", "current_conclusion": "done"},
        }

    def summarize_experience(self, experience, segments):
        return {
            "goal": "test",
            "stage_trajectory": [],
            "stable_facts": [],
            "current_state": {"status": "completed", "summary": "done"},
        }


def test_write_path_only_updates_qa_vector_and_enqueues_parent_work(tmp_path):
    storage = MemoryStorage(tmp_path / "memory.sqlite3")
    storage.insert_experience(_experience())
    storage.insert_segment(_segment())
    storage.commit()
    manager = RecordingManager(storage)

    result = manager.add_qa(
        topic_result={
            "topic": "旅行",
            "core_entity": "上海",
            "intent": "回忆",
            "entities": ["外滩"],
            "confidence": 0.9,
        },
        user_input="去年去了上海外滩",
        assistant_output="你很喜欢外滩夜景",
        timestamp="2026-01-02 00:00:00",
    )

    assert manager.qa_vector_updates == [result["qa_id"]]
    jobs = storage.list_pending_memory_jobs()
    assert {(job["job_type"], job["memory_id"]) for job in jobs} == {
        ("update_segment", "seg-1"),
    }
    assert storage.get_segment("seg-1")["qa_ids"] == []
    assert storage.get_experience("exp-1")["segment_ids"] == ["seg-1"]
    storage.close()


def test_outbox_coalesces_newer_versions(tmp_path):
    storage = MemoryStorage(tmp_path / "memory.sqlite3")
    for version in (1, 3, 2):
        storage.enqueue_memory_job(
            job_id=f"job-{version}",
            job_type="embed_segment",
            memory_type="segment",
            memory_id="seg-1",
            target_version=version,
            timestamp=f"2026-01-0{version} 00:00:00",
        )
    storage.commit()

    jobs = storage.list_pending_memory_jobs()
    assert len(jobs) == 1
    assert jobs[0]["target_version"] == 3
    storage.close()


def test_outbox_preserves_force_summary_and_terminal_status_when_coalescing(tmp_path):
    storage = MemoryStorage(tmp_path / "memory.sqlite3")
    storage.enqueue_memory_job(
        job_id="job-close",
        job_type="update_segment",
        memory_type="segment",
        memory_id="seg-1",
        target_version=1,
        timestamp="2026-01-01 00:00:00",
        payload={
            "desired_status": "completed",
            "force_summary": True,
            "force_experience_summary": True,
        },
    )
    storage.enqueue_memory_job(
        job_id="job-later",
        job_type="update_segment",
        memory_type="segment",
        memory_id="seg-1",
        target_version=2,
        timestamp="2026-01-02 00:00:00",
        payload={
            "desired_status": "open",
            "force_summary": False,
            "force_experience_summary": False,
        },
    )
    storage.commit()

    job = storage.list_pending_memory_jobs()[0]
    assert job["target_version"] == 2
    assert job["payload"]["desired_status"] == "completed"
    assert job["payload"]["force_summary"] is True
    assert job["payload"]["force_experience_summary"] is True
    storage.close()


def test_outbox_does_not_overwrite_new_work_while_old_version_runs(tmp_path):
    database = tmp_path / "memory.sqlite3"
    worker_storage = MemoryStorage(database)
    request_storage = MemoryStorage(database)
    worker_storage.enqueue_memory_job(
        job_id="job-1",
        job_type="embed_segment",
        memory_type="segment",
        memory_id="seg-1",
        target_version=1,
        timestamp="2026-01-01 00:00:00",
    )
    worker_storage.commit()
    job = worker_storage.list_pending_memory_jobs()[0]
    assert worker_storage.mark_memory_job_processing(
        job["job_id"], "2026-01-01 00:00:01"
    )
    worker_storage.commit()

    request_storage.enqueue_memory_job(
        job_id="job-2",
        job_type="embed_segment",
        memory_type="segment",
        memory_id="seg-1",
        target_version=2,
        timestamp="2026-01-01 00:00:02",
    )
    request_storage.commit()

    assert not worker_storage.mark_memory_job_completed(
        job["job_id"], 1, "2026-01-01 00:00:03"
    )
    worker_storage.commit()
    pending = worker_storage.list_pending_memory_jobs()
    assert len(pending) == 1
    assert pending[0]["target_version"] == 2
    request_storage.close()
    worker_storage.close()


def test_storage_removes_legacy_qa_full_text_table(tmp_path):
    db_path = tmp_path / "memory.sqlite3"
    storage = MemoryStorage(db_path)
    storage.connection.execute(
        "CREATE VIRTUAL TABLE qa_memory_fts USING fts5(qa_id, search_text)"
    )
    storage.commit()
    storage.close()


def test_storage_migrates_summary_revision_watermarks(tmp_path):
    db_path = tmp_path / "memory.sqlite3"
    storage = MemoryStorage(db_path)
    storage.insert_experience(_experience())
    segment = _segment()
    segment["qa_ids"] = ["qa-1", "qa-2"]
    segment["last_summarized_qa_count"] = 1
    storage.insert_segment(segment)
    storage.commit()
    storage.close()

    connection = sqlite3.connect(db_path)
    connection.execute("ALTER TABLE segment_memory DROP COLUMN summary_version")
    connection.execute(
        "ALTER TABLE segment_memory DROP COLUMN summarized_qa_ids_json"
    )
    connection.execute(
        "ALTER TABLE experience_memory "
        "DROP COLUMN last_summarized_child_revision"
    )
    connection.commit()
    connection.close()

    storage = MemoryStorage(db_path)
    migrated_segment = storage.get_segment("seg-1")
    migrated_experience = storage.get_experience("exp-1")

    assert migrated_segment["summary_version"] == 1
    assert migrated_segment["summarized_qa_ids"] == ["qa-1"]
    assert migrated_experience["last_summarized_child_revision"] == 0
    storage.close()

    storage = MemoryStorage(db_path)
    legacy_tables = storage.connection.execute(
        "SELECT name FROM sqlite_master WHERE name LIKE 'qa_memory_fts%'"
    ).fetchall()

    assert legacy_tables == []
    storage.close()


def test_sqlite_keyword_recall_uses_only_stored_structured_fields(tmp_path):
    storage = MemoryStorage(tmp_path / "memory.sqlite3")
    storage.insert_experience(_experience())
    storage.insert_segment(_segment())
    storage.insert_qa(
        {
            "qa_id": "qa-keyword",
            "source_id": None,
            "timestamp": "2026-01-02 00:00:00",
            "user_input": "正文里没有检索词",
            "assistant_output": "普通回答",
            "tools": [],
            "topic": "旅行",
            "intent": "回忆",
            "core_entity": "上海",
            "entities": ["外滩"],
            "segment_id": "seg-1",
            "status": "open",
            "confidence": 0.9,
            "reason": "",
        }
    )
    storage.commit()

    matches = storage.search_qas(
        topic="不匹配主题",
        core_entity="不匹配实体",
        intent="不匹配意图",
        entities=["外滩"],
        keywords=["正文里没有检索词"],
        limit=5,
    )

    assert [item["qa_id"] for item in matches] == ["qa-keyword"]
    storage.close()


def test_worker_summarizes_then_embeds_parent_memory(tmp_path):
    storage = MemoryStorage(tmp_path / "memory.sqlite3")
    storage.insert_experience(_experience())
    segment = _segment()
    segment["qa_ids"] = ["qa-1"]
    storage.insert_segment(segment)
    storage.insert_qa(
        {
            "qa_id": "qa-1",
            "source_id": None,
            "timestamp": "2026-01-02 00:00:00",
            "user_input": "去年去了上海外滩",
            "assistant_output": "你很喜欢外滩夜景",
            "tools": [],
            "topic": "旅行",
            "intent": "回忆",
            "core_entity": "上海",
            "entities": ["外滩"],
            "segment_id": "seg-1",
            "status": "open",
            "confidence": 0.9,
            "reason": "",
        }
    )
    storage.enqueue_memory_job(
        job_id="job-summary",
        job_type="update_segment",
        memory_type="segment",
        memory_id="seg-1",
        target_version=1,
        timestamp="2026-01-02 00:00:00",
        payload={"desired_status": "open", "force_summary": True},
    )
    storage.commit()
    worker_storage = MemoryStorage(tmp_path / "memory.sqlite3")
    manager = MemoryManager(
        storage=worker_storage,
        vector_store=ChromaVectorStore(ephemeral=True),
        embedder=HashingEmbedder(),
        summarizer=TemplateSummarizer(),
    )
    worker = MemoryDerivationWorker(manager, RLock())

    assert worker.drain_once() == 1
    assert storage.get_segment("seg-1")["last_summarized_qa_count"] == 1
    assert {job["job_type"] for job in storage.list_pending_memory_jobs()} == {
        "embed_segment",
        "update_experience",
    }
    assert worker.drain_once() == 2
    assert manager.vector_store.count() == 1
    worker_storage.close()
    storage.close()


def test_aggregate_segment_update_summarizes_only_at_threshold(tmp_path):
    storage = MemoryStorage(tmp_path / "memory.sqlite3")
    storage.insert_experience(_experience())
    storage.insert_segment(_segment())
    summarizer = CountingSummarizer()
    manager = MemoryManager(
        storage=storage,
        vector_store=ChromaVectorStore(ephemeral=True),
        embedder=HashingEmbedder(),
        summarizer=summarizer,
        segment_summary_qa_threshold=2,
        experience_summary_segment_threshold=99,
    )
    worker = MemoryDerivationWorker(manager, RLock(), batch_size=20)

    for index in (1, 2):
        storage.insert_qa(
            {
                "qa_id": f"qa-{index}",
                "source_id": None,
                "timestamp": f"2026-01-0{index} 00:00:00",
                "user_input": f"问题 {index}",
                "assistant_output": f"回答 {index}",
                "tools": [],
                "topic": "旅行",
                "intent": "回忆",
                "core_entity": "上海",
                "entities": ["上海"],
                "segment_id": "seg-1",
                "status": "open",
                "confidence": 0.9,
                "reason": "",
            }
        )
        storage.enqueue_memory_job(
            job_id=f"job-update-{index}",
            job_type="update_segment",
            memory_type="segment",
            memory_id="seg-1",
            target_version=index,
            timestamp=f"2026-01-0{index} 00:00:00",
            payload={"desired_status": "open", "force_summary": False},
        )
        storage.commit()
        worker.drain_once()
        if index == 1:
            assert summarizer.segment_calls == 0
            assert storage.get_segment("seg-1")["last_summarized_qa_count"] == 0

    assert summarizer.segment_calls == 1
    assert storage.get_segment("seg-1")["last_summarized_qa_count"] == 2
    while worker.drain_once():
        pass
    assert summarizer.experience_calls == 0
    assert storage.get_experience("exp-1")["last_summarized_child_revision"] == 0
    storage.close()


def test_one_updated_segment_does_not_trigger_experience_summary(tmp_path):
    storage = MemoryStorage(tmp_path / "memory.sqlite3")
    storage.insert_experience(_experience())
    segment = _segment()
    segment["summary_version"] = 1
    storage.insert_segment(segment)
    storage.enqueue_memory_job(
        job_id="job-exp-summary",
        job_type="update_experience",
        memory_type="experience",
        memory_id="exp-1",
        target_version=2,
        timestamp="2026-01-02 00:00:00",
        payload={"desired_status": "open", "force_summary": True},
    )
    storage.commit()
    manager = MemoryManager(
        storage=storage,
        vector_store=ChromaVectorStore(ephemeral=True),
        embedder=HashingEmbedder(),
        summarizer=TemplateSummarizer(),
    )
    worker = MemoryDerivationWorker(manager, RLock())

    assert worker.drain_once() == 1
    assert storage.list_pending_memory_jobs() == []
    storage.close()


def test_async_summary_update_preserves_concurrently_appended_qa(tmp_path):
    storage = MemoryStorage(tmp_path / "memory.sqlite3")
    storage.insert_experience(_experience())
    segment = _segment()
    segment["qa_ids"] = ["qa-1"]
    storage.insert_segment(segment)
    storage.commit()

    request_view = storage.get_segment("seg-1")
    request_view["qa_ids"].append("qa-2")
    request_view["updated_at"] = "2026-01-02 00:00:00"
    storage.update_segment_activity(request_view)
    storage.update_segment_summary(
        segment_id="seg-1",
        summary={
            "goal": "回忆旅行",
            "key_facts": ["去过外滩"],
            "state_changes": [],
            "state": {"status": "ongoing", "current_conclusion": "喜欢夜景"},
        },
        last_summarized_qa_count=1,
        status="open",
        updated_at="2026-01-02 00:00:01",
    )
    storage.commit()

    updated = storage.get_segment("seg-1")
    assert updated["qa_ids"] == ["qa-1", "qa-2"]
    assert updated["summary"]["goal"] == "回忆旅行"
    assert updated["last_summarized_qa_count"] == 1
    storage.close()


def test_new_experience_request_only_embeds_qa_synchronously(tmp_path):
    storage = MemoryStorage(tmp_path / "memory.sqlite3")
    vector_store = ChromaVectorStore(ephemeral=True)
    recaller = EmptyHistoryRecaller()
    manager = MemoryManager(
        storage=storage,
        vector_store=vector_store,
        embedder=HashingEmbedder(),
        summarizer=TemplateSummarizer(),
        experience_recaller=recaller,
    )

    result = manager.add_qa(
        topic_result={
            "topic": "旅行",
            "core_entity": "上海",
            "intent": "规划",
            "entities": ["上海"],
            "confidence": 0.9,
        },
        user_input="计划去上海旅行",
        timestamp="2026-01-02 00:00:00",
    )

    assert result["action"] == "new_experience"
    assert vector_store.count() == 1
    assert recaller.calls == []
    assert {job["job_type"] for job in storage.list_pending_memory_jobs()} == {
        "recall_experience_history",
        "create_experience_route_vector",
        "update_segment",
    }
    worker = MemoryDerivationWorker(manager, RLock())
    while worker.drain_once():
        pass
    assert len(recaller.calls) == 1
    assert vector_store.count() == 4
    assert storage.get_segment(result["segment_id"])["qa_ids"] == [result["qa_id"]]
    assert storage.get_experience(result["experience_id"])["segment_ids"] == [
        result["segment_id"]
    ]
    storage.close()


def test_low_qa_count_does_not_merge_an_incompatible_segment(tmp_path):
    storage = MemoryStorage(tmp_path / "memory.sqlite3")
    manager = MemoryManager(
        storage=storage,
        vector_store=ChromaVectorStore(ephemeral=True),
        embedder=HashingEmbedder(),
        summarizer=TemplateSummarizer(),
        experience_recaller=EmptyHistoryRecaller(),
    )
    first = manager.add_qa(
        {
            "topic": "travel",
            "core_entity": "Shanghai",
            "intent": "plan itinerary",
            "entities": ["Shanghai"],
            "confidence": 0.9,
        },
        "plan a trip",
    )
    second = manager.add_qa(
        {
            "topic": "travel",
            "core_entity": "Shanghai",
            "intent": "review restaurant",
            "entities": ["Shanghai"],
            "confidence": 0.9,
        },
        "review a restaurant",
    )
    third = manager.add_qa(
        {
            "topic": "travel",
            "core_entity": "Shanghai",
            "intent": "plan itinerary",
            "entities": ["Shanghai"],
            "confidence": 0.9,
        },
        "continue planning before the worker drains",
    )

    assert first["experience_id"] == second["experience_id"]
    assert first["segment_id"] != second["segment_id"]
    assert third["segment_id"] not in {first["segment_id"], second["segment_id"]}
    worker = MemoryDerivationWorker(manager, RLock(), batch_size=20)
    while worker.drain_once():
        pass
    assert storage.get_segment(first["segment_id"])["status"] == "completed"
    storage.close()


def test_segment_summary_consumes_only_unsummarized_qas(tmp_path):
    storage = MemoryStorage(tmp_path / "memory.sqlite3")
    storage.insert_experience(_experience())
    storage.insert_segment(_segment())
    summarizer = CapturingSummarizer()
    manager = MemoryManager(
        storage=storage,
        vector_store=ChromaVectorStore(ephemeral=True),
        embedder=HashingEmbedder(),
        summarizer=summarizer,
        segment_summary_qa_threshold=2,
        experience_summary_segment_threshold=99,
    )
    worker = MemoryDerivationWorker(manager, RLock(), batch_size=20)

    for index in (1, 2, 3):
        storage.insert_qa(
            {
                "qa_id": f"qa-{index}",
                "source_id": None,
                "timestamp": f"2026-01-0{index}",
                "user_input": f"question {index}",
                "assistant_output": f"answer {index}",
                "tools": [{"name": "calendar"}],
                "topic": "travel",
                "intent": "plan",
                "core_entity": "Shanghai",
                "entities": ["Shanghai"],
                "segment_id": "seg-1",
                "status": "open",
                "confidence": 0.9,
                "reason": "",
            }
        )
        storage.enqueue_memory_job(
            job_id=f"job-{index}",
            job_type="update_segment",
            memory_type="segment",
            memory_id="seg-1",
            target_version=index,
            timestamp=f"2026-01-0{index}",
            payload={
                "desired_status": "completed" if index == 3 else "open",
                "force_summary": index == 3,
            },
        )
        storage.commit()
        worker.drain_once()

    assert summarizer.segment_batches == [["qa-1", "qa-2"], ["qa-3"]]
    assert storage.get_segment("seg-1")["last_updated_qa_id"] == "qa-3"
    storage.close()


def test_experience_summary_receives_every_current_segment(tmp_path):
    storage = MemoryStorage(tmp_path / "memory.sqlite3")
    experience = _experience()
    experience["segment_ids"] = []
    experience["last_summarized_segment_count"] = 0
    storage.insert_experience(experience)
    expected_ids = []
    for index in range(3):
        segment = _segment()
        segment["segment_id"] = f"seg-{index}"
        segment["intent"] = f"stage-{index}"
        segment["summary"] = {
            "goal": f"stage-{index}",
            "key_facts": [],
            "state_changes": [],
            "state": {"status": "ongoing", "current_conclusion": ""},
        }
        segment["summary_version"] = 1
        storage.insert_segment(segment)
        expected_ids.append(segment["segment_id"])
    storage.enqueue_memory_job(
        job_id="job-exp",
        job_type="update_experience",
        memory_type="experience",
        memory_id="exp-1",
        target_version=3,
        timestamp="2026-01-05",
        payload={"desired_status": "open", "force_summary": True},
    )
    storage.commit()
    summarizer = CapturingSummarizer()
    manager = MemoryManager(
        storage=storage,
        vector_store=ChromaVectorStore(ephemeral=True),
        embedder=HashingEmbedder(),
        summarizer=summarizer,
        experience_summary_segment_threshold=2,
    )

    MemoryDerivationWorker(manager, RLock()).drain_once()

    assert summarizer.experience_batches == [expected_ids]
    assert storage.get_experience("exp-1")["last_summarized_child_revision"] == 3
    assert storage.get_experience("exp-1")["last_updated_segment_id"] == "seg-2"
    storage.close()


def test_summary_suggested_completion_does_not_close_memory(tmp_path):
    storage = MemoryStorage(tmp_path / "memory.sqlite3")
    storage.insert_experience(_experience())
    storage.insert_segment(_segment())
    storage.insert_qa(
        {
            "qa_id": "qa-1",
            "source_id": None,
            "timestamp": "2026-01-02",
            "user_input": "question",
            "assistant_output": "answer",
            "tools": [],
            "topic": "travel",
            "intent": "plan",
            "core_entity": "Shanghai",
            "entities": ["Shanghai"],
            "segment_id": "seg-1",
            "status": "open",
            "confidence": 0.9,
            "reason": "",
        }
    )
    storage.enqueue_memory_job(
        job_id="job-segment",
        job_type="update_segment",
        memory_type="segment",
        memory_id="seg-1",
        target_version=1,
        timestamp="2026-01-02",
        payload={"desired_status": "open", "force_summary": True},
    )
    storage.commit()
    manager = MemoryManager(
        storage=storage,
        vector_store=ChromaVectorStore(ephemeral=True),
        embedder=HashingEmbedder(),
        summarizer=CompletingSummarizer(),
    )

    MemoryDerivationWorker(manager, RLock()).drain_once()

    segment = storage.get_segment("seg-1")
    assert segment["summary"]["state"]["status"] == "completed"
    assert segment["status"] == "open"
    storage.close()


def test_experience_still_waits_for_two_segments_at_route_boundary(tmp_path):
    storage = MemoryStorage(tmp_path / "memory.sqlite3")
    storage.insert_experience(_experience())
    storage.insert_segment(_segment())
    storage.insert_qa(
        {
            "qa_id": "qa-1",
            "source_id": None,
            "timestamp": "2026-01-02",
            "user_input": "question",
            "assistant_output": "answer",
            "tools": [],
            "topic": "travel",
            "intent": "plan",
            "core_entity": "Shanghai",
            "entities": ["Shanghai"],
            "segment_id": "seg-1",
            "status": "open",
            "confidence": 0.9,
            "reason": "",
        }
    )
    storage.enqueue_memory_job(
        job_id="job-boundary",
        job_type="update_segment",
        memory_type="segment",
        memory_id="seg-1",
        target_version=1,
        timestamp="2026-01-02",
        payload={
            "desired_status": "completed",
            "force_summary": True,
            "allow_experience_completion": True,
        },
    )
    storage.commit()
    manager = MemoryManager(
        storage=storage,
        vector_store=ChromaVectorStore(ephemeral=True),
        embedder=HashingEmbedder(),
        summarizer=CompletingSummarizer(),
    )
    worker = MemoryDerivationWorker(manager, RLock(), batch_size=20)

    while worker.drain_once():
        pass

    assert storage.get_segment("seg-1")["status"] == "completed"
    assert storage.get_experience("exp-1")["status"] == "open"
    storage.close()


def test_hierarchy_retrieval_returns_only_content_after_update_markers(tmp_path):
    storage = MemoryStorage(tmp_path / "memory.sqlite3")
    experience = _experience()
    experience["last_updated_segment_id"] = "seg-1"
    experience["last_updated_segment_at"] = "2026-01-02 00:00:00"
    storage.insert_experience(experience)

    for index in (1, 2, 3):
        segment = _segment()
        segment["segment_id"] = f"seg-{index}"
        segment["intent"] = f"stage-{index}"
        segment["created_at"] = f"2026-01-0{index + 1} 00:00:00"
        segment["updated_at"] = segment["created_at"]
        segment["summary_version"] = 1
        if index == 3:
            segment["last_updated_qa_id"] = "qa-31"
        storage.insert_segment(segment)

    for qa_id, timestamp in (
        ("qa-31", "2026-01-04 00:00:01"),
        ("qa-32", "2026-01-04 00:00:02"),
        ("qa-33", "2026-01-04 00:00:03"),
    ):
        storage.insert_qa(
            {
                "qa_id": qa_id,
                "source_id": None,
                "timestamp": timestamp,
                "user_input": qa_id,
                "assistant_output": "answer",
                "tools": [],
                "topic": experience["topic"],
                "intent": "stage-3",
                "core_entity": experience["core_entity"],
                "entities": [],
                "segment_id": "seg-3",
                "status": "open",
                "confidence": 1.0,
                "reason": "",
            }
        )
    storage.upsert_runtime_state(
        state_key="session-1",
        current_experience_id="exp-1",
        current_segment_id="seg-3",
        updated_at="2026-01-04 00:00:03",
    )
    storage.commit()
    manager = MemoryManager(
        storage=storage,
        vector_store=ChromaVectorStore(ephemeral=True),
        embedder=HashingEmbedder(),
    )

    result = HybridRetriever(manager).retriever(
        topic=experience["topic"],
        core_entity=experience["core_entity"],
        query="continue",
        state_key="session-1",
    )

    assert [item["segment_id"] for item in result["segments"]] == ["seg-2", "seg-3"]
    assert [item["qa_id"] for item in result["qas"]] == ["qa-32", "qa-33"]
    storage.close()


def test_close_routes_use_goal_then_recency(tmp_path):
    class EqualRouteVectorStore:
        def query(self, *args, **kwargs):
            return [
                {"metadata": {"experience_id": "exp-old"}, "similarity": 0.9},
                {"metadata": {"experience_id": "exp-new"}, "similarity": 0.89},
            ]

    storage = MemoryStorage(tmp_path / "memory.sqlite3")
    for experience_id, goal, updated_at in (
        ("exp-old", "book hotel", "2026-01-02 00:00:00"),
        ("exp-new", "visit museum", "2026-01-03 00:00:00"),
    ):
        experience = _experience()
        experience["experience_id"] = experience_id
        experience["summary"] = {"goal": goal}
        experience["updated_at"] = updated_at
        storage.insert_experience(experience)
    manager = MemoryManager(
        storage=storage,
        vector_store=EqualRouteVectorStore(),
        embedder=HashingEmbedder(),
        experience_similarity_threshold=0.8,
        experience_route_margin=0.05,
    )

    routed = manager._find_experience_by_vector(
        "travel", "Shanghai", "book hotel"
    )
    assert routed["experience_id"] == "exp-old"

    candidates = [
        ({"segment_id": "seg-old", "summary": {}, "updated_at": "2026-01-02"}, 0.9),
        ({"segment_id": "seg-new", "summary": {}, "updated_at": "2026-01-03"}, 0.89),
    ]
    assert manager._select_ambiguous_by_goal(candidates, "anything")["segment_id"] == "seg-new"
    storage.close()


def test_coalesced_parent_jobs_rebuild_all_links_from_relations(tmp_path):
    storage = MemoryStorage(tmp_path / "memory.sqlite3")
    manager = MemoryManager(
        storage=storage,
        vector_store=ChromaVectorStore(ephemeral=True),
        embedder=HashingEmbedder(),
        summarizer=TemplateSummarizer(),
        segment_summary_qa_threshold=99,
        experience_summary_segment_threshold=99,
        experience_recaller=EmptyHistoryRecaller(),
    )
    topic_result = {
        "topic": "旅行",
        "core_entity": "上海",
        "intent": "规划",
        "entities": ["上海"],
        "confidence": 0.9,
    }

    first = manager.add_qa(topic_result, "规划上海第一天", timestamp="2026-01-01")
    second = manager.add_qa(topic_result, "规划上海第二天", timestamp="2026-01-02")

    assert first["segment_id"] == second["segment_id"]
    assert storage.get_segment(first["segment_id"])["qa_ids"] == []
    jobs = storage.list_pending_memory_jobs(20)
    assert len([job for job in jobs if job["job_type"] == "update_segment"]) == 1
    worker = MemoryDerivationWorker(manager, RLock(), batch_size=20)
    while worker.drain_once():
        pass

    assert storage.get_segment(first["segment_id"])["qa_ids"] == [
        first["qa_id"],
        second["qa_id"],
    ]
    assert storage.get_experience(first["experience_id"])["segment_ids"] == [
        first["segment_id"]
    ]
    storage.close()
