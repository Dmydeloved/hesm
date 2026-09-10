import hashlib
import sqlite3
from pathlib import Path

import experiments.backend as backend_module


class FakeHESMService:
    instances = []

    def __init__(
        self, config_path=None, *, config_data=None, storage_root, retriever_class
    ):
        self.config_data = config_data
        self.storage_root = Path(storage_root)
        self.retriever_class = retriever_class
        self.added = []
        self.closed = False
        self.instances.append(self)

    def add_memory(self, **kwargs):
        self.added.append(kwargs)
        return {"memories": [{"qa_id": f"qa_{len(self.added)}"}]}

    def retrieve(self, **kwargs):
        return {
            "context": f"native context from {self.storage_root.name}",
            "route_status": "runtime",
            "query_extraction": {
                "topic": "topic", "core_entity": "entity", "intent": "query"
            },
            "experiences": [{"experience_id": "exp_1"}],
            "segments": [{"segment_id": "seg_1"}],
            "qas": [{"qa_id": "qa_1"}],
        }

    def close(self):
        self.closed = True


def make_backend(monkeypatch, tmp_path):
    FakeHESMService.instances = []
    monkeypatch.setattr(backend_module, "HESMService", FakeHESMService)
    settings = {
        "data_root": str(tmp_path),
        "native_config": {"embedding": {"model": "text-embedding-v4"}},
        "max_top_k": 100,
        "profile": "test",
        "config_fingerprint": "fingerprint",
    }
    return backend_module.EvaluationBackend(settings, "locomo_fixture")


def user_hash(user_id):
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()


def request_rows(backend):
    with sqlite3.connect(backend.request_db_path) as connection:
        connection.row_factory = sqlite3.Row
        return connection.execute(
            "SELECT * FROM request_log ORDER BY rowid"
        ).fetchall()


def test_add_isolates_native_hesm_by_user_and_is_idempotent(monkeypatch, tmp_path):
    backend = make_backend(monkeypatch, tmp_path)
    messages = [
        {"role": "user", "content": "remember this", "chat_time": "2024-01-01"}
    ]

    first = backend.add("user-a", messages, "session-a")
    duplicate = backend.add("user-a", messages, "session-b")
    other = backend.add("user-b", messages, "session-a")

    assert first["idempotent"] is False
    assert duplicate["idempotent"] is True
    assert other["idempotent"] is False
    assert len(FakeHESMService.instances) == 2
    assert {instance.storage_root for instance in FakeHESMService.instances} == {
        backend.users_root / user_hash("user-a"),
        backend.users_root / user_hash("user-b"),
    }
    assert sum(len(instance.added) for instance in FakeHESMService.instances) == 2
    assert all(
        item["state_key"] == "evaluation"
        for instance in FakeHESMService.instances
        for item in instance.added
    )

    rows = request_rows(backend)
    assert [row["status"] for row in rows] == ["success", "success", "success"]
    assert [row["idempotent_hit"] for row in rows] == [0, 1, 0]
    assert rows[0]["request_hash"] == rows[1]["request_hash"]
    assert rows[0]["user_hash"] != rows[2]["user_hash"]


def test_search_uses_only_requested_user_store_and_records_status(monkeypatch, tmp_path):
    backend = make_backend(monkeypatch, tmp_path)
    result_a = backend.search("user-a", "question", 20, "2024-01-02")
    result_b = backend.search("user-b", "question", 20, "2024-01-02")

    assert user_hash("user-a") in result_a["context"]
    assert user_hash("user-b") in result_b["context"]
    assert result_a["candidate_counts"] == {
        "experiences": 1, "segments": 1, "qas": 1
    }
    rows = request_rows(backend)
    assert [row["operation"] for row in rows] == ["search", "search"]
    assert all(row["status"] == "success" for row in rows)


def test_write_locks_are_stable_per_user_and_distinct_between_users(
    monkeypatch, tmp_path
):
    backend = make_backend(monkeypatch, tmp_path)
    lock_a = backend._write_lock(user_hash("user-a"))
    assert lock_a is backend._write_lock(user_hash("user-a"))
    assert lock_a is not backend._write_lock(user_hash("user-b"))


def test_failed_add_records_hash_without_request_content(monkeypatch, tmp_path):
    backend = make_backend(monkeypatch, tmp_path)

    def fail(**kwargs):
        raise RuntimeError("provider failed")

    with backend._write_lock(user_hash("user-a")):
        service = backend._service(user_hash("user-a"))
    service.add_memory = fail

    try:
        backend.add("user-a", [{"content": "private request text"}])
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected add failure")

    row = request_rows(backend)[0]
    assert row["status"] == "failed"
    assert row["error_type"] == "RuntimeError"
    assert "private request text" not in str(dict(row))


def test_validation_failure_is_also_recorded(monkeypatch, tmp_path):
    backend = make_backend(monkeypatch, tmp_path)
    try:
        backend.search("user-a", "", 20)
    except ValueError:
        pass
    else:
        raise AssertionError("expected validation failure")

    row = request_rows(backend)[0]
    assert row["operation"] == "search"
    assert row["status"] == "failed"
    assert row["error_type"] == "ValueError"
