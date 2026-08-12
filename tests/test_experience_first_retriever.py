from __future__ import annotations

import inspect
import tempfile
from pathlib import Path
from typing import Any

from hesm.retriever import HybridRetriever
from hesm.storage import MemoryStorage
from hesm.vector_store import ChromaVectorStore


def _experience(memory_id: str, topic: str, entity: str) -> dict[str, Any]:
    return {
        "experience_id": memory_id,
        "topic": topic,
        "core_entity": entity,
        "intents_link": ["query"],
        "summary": f"summary {memory_id}",
        "state": {},
        "updated_at": "2026-01-01",
    }


def _segment(
    memory_id: str, experience_id: str, topic: str, entity: str
) -> dict[str, Any]:
    return {
        "segment_id": memory_id,
        "experience_id": experience_id,
        "topic": topic,
        "core_entity": entity,
        "intent": "query",
        "summary": f"summary {memory_id}",
        "status": "active",
        "updated_at": "2026-01-01",
    }


def _qa(
    memory_id: str, segment_id: str, topic: str, entity: str
) -> dict[str, Any]:
    return {
        "qa_id": memory_id,
        "segment_id": segment_id,
        "topic": topic,
        "core_entity": entity,
        "intent": "query",
        "entities": [entity],
        "user_input": f"question {memory_id}",
        "assistant_output": f"answer {memory_id}",
        "tools": [],
        "timestamp": "2026-01-01",
        "status": "active",
        "confidence": 0.95,
        "reasoning": "test",
    }


class FakeStorage:
    def __init__(self) -> None:
        self.experiences = {
            "e1": _experience("e1", "travel", "Alice"),
            "e2": _experience("e2", "work", "Bob"),
        }
        self.segments = {
            "s1": _segment("s1", "e1", "travel", "Alice"),
            "s2": _segment("s2", "e2", "work", "Bob"),
            "s2-sibling": _segment("s2-sibling", "e2", "work", "Bob"),
        }
        self.qas = {
            "q1": _qa("q1", "s1", "travel", "Alice"),
            "q2": _qa("q2", "s2", "work", "Bob"),
            "q2-sibling": _qa("q2-sibling", "s2-sibling", "work", "Bob"),
        }
        self.qa_search_calls = 0
        self.descendant_loads: list[list[str]] = []

    def search_experiences(
        self, topic: str, core_entity: str, limit: int
    ) -> list[dict[str, Any]]:
        del topic, core_entity, limit
        return [{**self.experiences["e1"], "relation_score": 2.0}]

    def search_qas(self, **_: Any) -> list[dict[str, Any]]:
        self.qa_search_calls += 1
        return [self.qas["q2"]]

    def get_experiences(self, ids: list[str]) -> list[dict[str, Any]]:
        return [self.experiences[value] for value in ids if value in self.experiences]

    def get_segments(self, ids: list[str]) -> list[dict[str, Any]]:
        return [self.segments[value] for value in ids if value in self.segments]

    def get_qas(self, ids: list[str]) -> list[dict[str, Any]]:
        return [self.qas[value] for value in ids if value in self.qas]

    def list_segments_by_experience_ids(
        self, ids: list[str]
    ) -> list[dict[str, Any]]:
        self.descendant_loads.append(list(ids))
        return [
            row for row in self.segments.values()
            if row["experience_id"] in ids
        ]

    def list_qas_by_segment_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        return [row for row in self.qas.values() if row["segment_id"] in ids]


class FakeVectorStore:
    def __init__(self) -> None:
        self.global_qa_calls = 0

    def query(
        self,
        query_embedding: list[float],
        memory_type: str,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        del query_embedding, top_k
        if memory_type != "qa":
            return []
        if metadata_filter:
            allowed_segments = metadata_filter.get("segment_id", [])
            return [] if "s2" not in allowed_segments else [{
                "metadata": {"memory_id": "q2", "qa_id": "q2"},
                "similarity": 0.95,
            }]
        self.global_qa_calls += 1
        return [{
            "metadata": {"memory_id": "q2", "qa_id": "q2"},
            "similarity": 0.95,
        }]

    def count(self) -> int:
        return 3


class FakeEmbedder:
    def embed(self, text: str) -> list[float]:
        del text
        return [1.0, 0.0]


def _retriever(storage: FakeStorage) -> HybridRetriever:
    return HybridRetriever(
        storage=storage,
        vector_store=FakeVectorStore(),
        embedder=FakeEmbedder(),
        rerank_with_llm=False,
    )


def test_high_confidence_uses_only_initial_experience_tree() -> None:
    storage = FakeStorage()
    result = _retriever(storage).recall(
        topic="travel",
        core_entity="Alice",
        intent="query",
        entities=["Alice"],
        query="Where did Alice travel?",
        query_confidence=0.9,
        top_experience=3,
        top_segment=5,
        top_qa=8,
    )

    assert storage.qa_search_calls == 0
    assert storage.descendant_loads == [["e1"]]
    assert result["debug"]["low_confidence_qa_rescue"] is False
    assert result["debug"]["total_retrieval_ms"] >= 0
    assert result["debug"]["embedding"]["vector_dimensions"] == 2
    collection_timing = result["debug"]["candidate_collection"]
    assert set(collection_timing) >= {
        "sql_recall",
        "chroma_vector_recall",
        "ranking_processing",
    }
    assert result["debug"]["candidate_trees"]["original"]
    assert result["debug"]["candidate_trees"]["pruned"]
    assert result["debug"]["constraints"]["requested_top_k"] == {
        "experience": 3,
        "segment": 5,
        "qa": 8,
    }
    assert [node["id"] for node in result["candidate_tree"]] == ["e1"]
    assert result["candidate_tree"][0]["segments"][0]["qas"][0]["id"] == "q1"


def test_low_confidence_adds_only_rescued_qa_ancestor_path() -> None:
    storage = FakeStorage()
    result = _retriever(storage).recall(
        topic="travel",
        core_entity="Alice",
        intent="query",
        entities=["Bob"],
        query="What happened to Bob?",
        query_confidence=0.4,
        top_experience=3,
        top_segment=5,
        top_qa=8,
    )

    assert storage.qa_search_calls == 1
    # e2 is never passed to the bulk descendant loader.
    assert storage.descendant_loads == [["e1"]]
    assert result["debug"]["low_confidence_qa_rescue"] is True
    assert "qa_rescue" in result["debug"]["candidate_collection"][
        "chroma_vector_recall"
    ]["by_layer"]
    assert "qa_rescue" in result["debug"]["candidate_collection"][
        "sql_recall"
    ]["by_layer"]
    tree_by_id = {node["id"]: node for node in result["candidate_tree"]}
    assert set(tree_by_id) == {"e1", "e2"}
    rescued_segments = tree_by_id["e2"]["segments"]
    assert [node["id"] for node in rescued_segments] == ["s2"]
    assert [node["id"] for node in rescued_segments[0]["qas"]] == ["q2"]
    assert "global_qa_vector" in (
        rescued_segments[0]["qas"][0]["retrieval_sources"]
    )


def test_recall_interface_has_no_top_k_parameter() -> None:
    assert "top_k" not in inspect.signature(HybridRetriever.recall).parameters


def test_minimum_tree_uses_two_summary_compression_rounds() -> None:
    retriever = _retriever(FakeStorage())
    retriever.max_context_tokens = 300
    retriever.min_retained_experiences = 1
    retriever.min_retained_segments = 1
    tree = [{
        "id": "e1",
        "summary": "E" * 2000,
        "local_score": 0.9,
        "segments": [{
            "id": "s1",
            "summary": "S" * 1500,
            "local_score": 0.9,
            "qas": [{
                "id": "q1",
                "user_input": "U" * 500,
                "assistant_output": "A" * 500,
                "local_score": 0.9,
            }],
        }],
    }]

    fitted, debug = retriever._fit_candidate_tree_to_context(tree)

    retriever._validate_candidate_tree(fitted)
    assert debug["removed_qa_ids"] == []
    assert debug["removed_segment_ids"] == []
    assert debug["removed_experience_ids"] == []
    assert debug["normal_summary_compression"]
    assert debug["extreme_summary_compression"]
    assert not debug["context_overflow_unresolved"]
    assert len(fitted[0]["summary"]) <= retriever.summary_extreme_experience_chars + 3
    assert len(fitted[0]["segments"][0]["summary"]) <= (
        retriever.summary_extreme_segment_chars + 3
    )


def test_storage_relational_recall_uses_expected_fields() -> None:
    with tempfile.TemporaryDirectory() as directory:
        storage = MemoryStorage(Path(directory) / "memory.sqlite3")
        try:
            storage.insert_experience({
                **_experience("e1", "travel", "Alice"),
                "segment_ids": ["s1"],
                "version": 1,
                "created_at": "2026-01-01",
                "last_summarized_segment_count": 0,
            })
            storage.insert_segment({
                **_segment("s1", "e1", "travel", "Alice"),
                "qa_ids": ["q1"],
                "version": 1,
                "created_at": "2026-01-01",
                "last_summarized_qa_count": 0,
            })
            storage.insert_qa(_qa("q1", "s1", "travel", "Alice"))
            storage.commit()

            experiences = storage.search_experiences("travel", "Alice", 10)
            qas = storage.search_qas(
                topic="unknown",
                core_entity="unknown",
                entities=["alice"],
                keywords=[],
                limit=10,
            )
            assert [row["experience_id"] for row in experiences] == ["e1"]
            assert experiences[0]["relation_score"] == 2
            assert [row["qa_id"] for row in qas] == ["q1"]
        finally:
            storage.close()


def test_context_pruning_runs_qa_segment_experience_stages_in_order() -> None:
    retriever = _retriever(FakeStorage())
    retriever.max_context_tokens = 220
    retriever.min_retained_experiences = 1
    retriever.min_retained_segments = 1

    def qa_node(memory_id: str, score: float) -> dict[str, Any]:
        return {
            "id": memory_id,
            "user_input": "question words " * 80,
            "assistant_output": "answer words " * 80,
            "local_score": score,
        }

    tree = [
        {
            "id": "e1",
            "summary": "experience summary " * 80,
            "local_score": 0.9,
            "segments": [
                {
                    "id": "s1",
                    "summary": "segment summary " * 60,
                    "local_score": 0.9,
                    "qas": [qa_node("q1", 0.9), qa_node("q1-low", 0.05)],
                },
                {
                    "id": "s1-low",
                    "summary": "segment summary " * 60,
                    "local_score": 0.2,
                    "qas": [qa_node("q1-branch", 0.2)],
                },
            ],
        },
        {
            "id": "e2",
            "summary": "experience summary " * 80,
            "local_score": 0.1,
            "segments": [{
                "id": "s2",
                "summary": "segment summary " * 60,
                "local_score": 0.1,
                "qas": [qa_node("q2", 0.1)],
            }],
        },
    ]

    fitted, debug = retriever._fit_candidate_tree_to_context(tree)

    retriever._validate_candidate_tree(fitted)
    assert "q1-low" in debug["removed_qa_ids"]
    assert "s1-low" in debug["removed_segment_ids"]
    assert "e2" in debug["removed_experience_ids"]
    assert len(fitted) == 1
    assert len(fitted[0]["segments"]) == 1
    assert len(fitted[0]["segments"][0]["qas"]) == 1


def test_vector_store_metadata_filter_limits_scoped_qa_search() -> None:
    vector_store = ChromaVectorStore(ephemeral=True)
    vector_store.upsert(
        "qa", "q1", "one", [1.0, 0.0], "2026-01-01",
        metadata={"segment_id": "s1"},
    )
    vector_store.upsert(
        "qa", "q2", "two", [0.9, 0.1], "2026-01-01",
        metadata={"segment_id": "s2"},
    )

    results = vector_store.query(
        [1.0, 0.0],
        memory_type="qa",
        top_k=10,
        metadata_filter={"segment_id": ["s1"]},
    )

    assert [item["metadata"]["memory_id"] for item in results] == ["q1"]
