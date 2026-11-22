from threading import Barrier, Event
from types import SimpleNamespace

import pytest

from core.retriever import HybridRetriever, build_context_text
from core.storage import MemoryStorage


@pytest.fixture
def storage(tmp_path):
    database = MemoryStorage(tmp_path / "memory.db")
    rows = [
        ("topic", "户外", "采购", "背包需要防水", 1, "open"),
        ("intent", "摄影", "排查", "检查镜头密封圈", 2, "open"),
        ("both", "户外", "排查", "外壳需要防水", 3, "open"),
        ("miss", "户外", "计划", "周末骑行", 4, "open"),
        ("outside", "旅行", "计划", "防水", 5, "open"),
        ("recent", "维修", "清洁", "没有防水涂层", 6, "open"),
        ("deleted", "户外", "排查", "防水", 7, "deleted"),
    ]
    for key, topic, intent, question, day, status in rows:
        timestamp = f"2025-01-{day:02}T10:00:00Z"
        fields = {
            "topic": topic,
            "intent": intent,
            "core_entity": key,
            "experience_id": f"e-{key}",
            "segment_id": f"s-{key}",
            "qa_id": f"q-{key}",
            "created_at": timestamp,
            "updated_at": timestamp,
            "version": 1,
        }
        database.insert_experience({
            **fields,
            "status": status,
            "intents_link": [],
            "segment_ids": [fields["segment_id"]],
            "last_summarized_segment_count": 0,
        })
        database.insert_segment({
            **fields, "qa_ids": [fields["qa_id"]], "last_summarized_qa_count": 0
        })
        database.insert_qa({
            **fields,
            "timestamp": timestamp,
            "user_input": question,
            "assistant_output": "",
            "entities": [],
            "tools": [],
            "confidence": 1,
        })
    database.commit()
    try:
        yield database
    finally:
        database.close()


class VectorSearch:
    def __init__(self, segment_hits=()):
        self.candidate_ids = None
        self.segment_hits = segment_hits
        self.segment_queries = []

    def query(self, embedding, *, memory_type, top_k, metadata_filter):
        assert embedding == [1.0]
        if memory_type == "segment":
            self.segment_queries.append(metadata_filter)
            return [
                {"metadata": {"segment_id": segment_id}, "similarity": similarity}
                for segment_id, similarity in self.segment_hits[:top_k]
            ]
        assert memory_type == "qa"
        self.candidate_ids = metadata_filter["qa_id"]
        assert top_k <= len(self.candidate_ids)
        return [
            {"metadata": {"qa_id": qa_id}, "similarity": similarity}
            for qa_id, similarity in [
                ("q-outside", 1.0),
                ("q-intent", 0.99),
                ("q-both", 0.95),
                ("q-intent", 0.90),
                ("q-miss", 0.0),
            ]
        ]


def make_retriever(storage, vectors):
    manager = SimpleNamespace(
        storage=storage,
        vector_store=vectors,
        embedder=SimpleNamespace(embed=lambda _: [1.0]),
    )
    return HybridRetriever(manager, recent_qa_limit=1)


@pytest.mark.parametrize(
    ("mode", "excluded"),
    [
        ("default", set()),
        ("no_topic", {"q-topic", "q-miss"}),
        ("no_intent", {"q-intent"}),
        ("no_time", {"q-recent"}),
    ],
)
def test_modes_restrict_candidate_pool_and_context(storage, mode, excluded):
    vectors = VectorSearch()
    result = make_retriever(storage, vectors).retriever("户外", "topic", "怎么防水", "排查", [], mode)
    debug = result["debug"]
    expected_pool = {"q-topic", "q-intent", "q-both", "q-miss", "q-recent"} - excluded

    assert set(result) == {"model", "experiences", "segments", "qas", "context", "debug"}
    assert result["model"] == mode
    assert set(debug["topic_experience_ids"]) == (
        set() if mode == "no_topic" else {"e-topic", "e-both", "e-miss"}
    )
    assert set(debug["intent_segment_ids"]) == (
        set() if mode == "no_intent" else {"s-intent", "s-both"}
    )
    assert debug["recent_qa_ids"] == ([] if mode == "no_time" else ["q-recent"])
    assert {qa["qa_id"] for qa in debug["candidate_qa_pool"]} == expected_pool
    assert len(debug["candidate_qa_pool"]) == len(expected_pool)
    assert set(vectors.candidate_ids) == expected_pool
    assert set(debug["keyword_qa_ids"]) == expected_pool - {"q-intent", "q-miss"}
    assert set(debug["semantic_qa_ids"]) == expected_pool & {"q-intent", "q-both"}

    assert result["qas"][0]["qa_id"] == "q-both"
    assert {qa["qa_id"] for qa in result["qas"]} == expected_pool - {"q-miss"}
    assert {qa["segment_id"] for qa in result["qas"]} == {
        segment["segment_id"] for segment in result["segments"]
    }
    assert {segment["experience_id"] for segment in result["segments"]} == {
        experience["experience_id"] for experience in result["experiences"]
    }
    assert result["context"] == build_context_text(
        result["experiences"], result["segments"], result["qas"]
    )


def test_rankers_wait_until_all_three_recall_paths_complete(storage, monkeypatch):
    barrier = Barrier(3, timeout=5)
    retriever = make_retriever(storage, VectorSearch())
    completed = []
    for name in ("_recall_topic", "_recall_intent", "_recall_recent"):
        done = Event()
        completed.append(done)
        recall = getattr(retriever, name)

        def synchronized(*args, recall=recall, done=done):
            barrier.wait()
            rows = recall(*args)
            done.set()
            return rows

        monkeypatch.setattr(retriever, name, synchronized)

    def rank_keywords(*args):
        assert all(done.is_set() for done in completed)
        return []

    monkeypatch.setattr("core.retriever._rank_keywords", rank_keywords)
    result = retriever.retriever("户外", "topic", "怎么防水", "排查", [])
    assert result["model"] == "default"


def test_recall_reads_committed_sqlite_memory(storage):
    retriever = make_retriever(storage, VectorSearch())
    storage.connection.execute("UPDATE qa_memory SET status = 'deleted' WHERE qa_id = 'q-recent'")

    result = retriever.retriever("户外", "topic", "怎么防水", "排查", [])
    assert result["debug"]["recent_qa_ids"] == ["q-recent"]
    storage.commit()

    result = retriever.retriever("户外", "topic", "怎么防水", "排查", [])
    assert result["debug"]["recent_qa_ids"] == ["q-outside"]
    assert "q-recent" not in result["debug"]["candidate_qa_ids"]


def test_intent_recall_combines_labels_with_input_semantics(storage):
    vectors = VectorSearch([
        ("s-both", 1.0),
        ("s-outside", 0.9),
        ("s-outside", 0.8),
        ("s-deleted", 0.7),
        ("s-missing", 0.6),
        ("s-recent", 0.0),
    ])
    retriever = make_retriever(storage, vectors)
    embedded_texts = []

    def embed(text):
        embedded_texts.append(text)
        return [1.0]

    retriever.manager.embedder.embed = embed
    result = retriever.retriever("户外", "topic", "怎么防水", "排查", [], model="no_time")

    assert embedded_texts == ["意图：排查\n用户输入：怎么防水", "怎么防水"]
    assert vectors.segment_queries == [{"status": ["open", "completed"]}]
    assert set(result["debug"]["intent_segment_ids"]) == {"s-intent", "s-both", "s-outside"}
    assert len(result["debug"]["intent_segment_ids"]) == 3
    assert "q-outside" in result["debug"]["candidate_qa_ids"]
    assert "q-deleted" not in result["debug"]["candidate_qa_ids"]
    assert "q-recent" not in result["debug"]["candidate_qa_ids"]

    vectors.segment_queries.clear()
    embedded_texts.clear()
    result = retriever.retriever("户外", "topic", "怎么防水", "排查", [], model="no_intent")
    assert not vectors.segment_queries
    assert embedded_texts == ["怎么防水"]
    assert result["debug"]["intent_segment_ids"] == []
    assert "q-outside" not in result["debug"]["candidate_qa_ids"]


def test_intent_recall_keeps_label_matches_on_vector_failure(storage, monkeypatch):
    vectors = VectorSearch()
    retriever = make_retriever(storage, vectors)
    query_vectors = vectors.query

    def query(embedding, *, memory_type, **kwargs):
        if memory_type == "segment":
            raise RuntimeError("Segment index unavailable")
        return query_vectors(embedding, memory_type=memory_type, **kwargs)

    monkeypatch.setattr(vectors, "query", query)
    result = retriever.retriever("", "", "怎么防水", "排查", [], model="no_time")
    assert set(result["debug"]["intent_segment_ids"]) == {"s-intent", "s-both"}
    assert set(result["debug"]["candidate_qa_ids"]) == {"q-intent", "q-both"}
