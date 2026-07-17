import unittest

from memory.retriever import HybridRetriever


class CountingEmbedder:
    def __init__(self) -> None:
        self.calls = 0
        self.vector = [0.25, 0.75]

    def embed(self, text: str) -> list[float]:
        self.calls += 1
        return self.vector


class RecordingVectorStore:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        memory_type = kwargs["memory_type"]
        memory_ids = {
            "experience": "exp_1",
            "segment": "seg_1",
            "qa": "qa_1",
        }
        return [{"metadata": {"memory_id": memory_ids[memory_type]}}]


class FakeStorage:
    def find_experiences(self, topic: str, core_entity: str, limit: int):
        return [
            {
                "experience_id": "exp_1",
                "topic": topic,
                "core_entity": core_entity,
                "intents_link": '["intent"]',
                "summary": "experience summary",
                "state": "{}",
                "updated_at": "2026-01-01T00:00:00+00:00",
            }
        ]

    def list_segments_by_experience_ids(self, experience_ids: list[str]):
        return [
            {
                "segment_id": "seg_1",
                "experience_id": experience_ids[0],
                "topic": "topic",
                "core_entity": "entity",
                "intent": "intent",
                "status": "active",
                "summary": "segment summary",
                "updated_at": "2026-01-01T00:00:00+00:00",
            }
        ]

    def list_qas_by_segment_ids(self, segment_ids: list[str]):
        return [
            {
                "qa_id": "qa_1",
                "segment_id": segment_ids[0],
                "timestamp": "2026-01-01T00:00:00+00:00",
                "user_input": "question",
                "assistant_output": "answer",
                "topic": "topic",
                "core_entity": "entity",
                "intent": "intent",
                "entities": "[]",
                "confidence": 1.0,
                "reasoning": "",
            }
        ]


class PassThroughReranker:
    def rerank(self, layer, query_text, candidates, limit):
        return [
            {"id": candidate["id"], "score": 1.0}
            for candidate in candidates[:limit]
        ]


class RetrieverEmbeddingTests(unittest.TestCase):
    def test_recall_reuses_one_query_embedding_for_all_memory_layers(self):
        embedder = CountingEmbedder()
        vector_store = RecordingVectorStore()
        retriever = HybridRetriever(
            storage=FakeStorage(),
            vector_store=vector_store,
            embedder=embedder,
            reranker=PassThroughReranker(),
        )

        result = retriever.recall(
            topic="topic",
            core_entity="entity",
            intent="intent",
            entities=["related"],
            use_cache=False,
        )

        self.assertEqual(1, embedder.calls)
        self.assertEqual(
            ["experience", "segment", "qa"],
            [call["memory_type"] for call in vector_store.calls],
        )
        self.assertTrue(
            all(
                call["query_embedding"] is embedder.vector
                for call in vector_store.calls
            )
        )
        self.assertEqual(1, len(result["experiences"]))
        self.assertEqual(1, len(result["segments"]))
        self.assertEqual(1, len(result["qas"]))


if __name__ == "__main__":
    unittest.main()
