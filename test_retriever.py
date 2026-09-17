from __future__ import annotations

from core.retriever import HybridRetriever, ReadOnlyHybridRetriever


class FakeStorage:
    def __init__(self, *, runtime=None, experience=None, segments=None, qas=None):
        self.runtime = runtime
        self.experience = experience
        self.segments = segments or []
        self.qas = qas or []

    def get_runtime_state(self, state_key):
        return self.runtime

    def get_experience(self, experience_id):
        if self.experience and experience_id == self.experience["experience_id"]:
            return self.experience
        return None

    def find_active_experience(self, topic, core_entity):
        return None

    def list_latest_segments(self, experience_id, limit):
        return self.segments[-limit:]

    def list_latest_qas(self, segment_ids, limit):
        return self.qas[-limit:]


class FakeRecaller:
    def __init__(self):
        self.calls = []

    def recall(self, **kwargs):
        self.calls.append(kwargs)
        return {"history_experience": {"prior_context": "历史上下文"}}


class FakeManager:
    def __init__(self, storage, *, vector_match=None):
        self.storage = storage
        self.vector_match = vector_match
        self.experience_recaller = FakeRecaller()

    @staticmethod
    def _same_experience(experience, topic, core_entity):
        return bool(
            experience
            and experience.get("status") == "open"
            and experience.get("topic") == topic
            and experience.get("core_entity") == core_entity
        )

    def _find_experience_by_vector(self, topic, core_entity):
        return self.vector_match

    def route_experience(self, **kwargs):
        raise AssertionError("retrieval must not call the write-capable route_experience")

    def create_experience(self, **kwargs):
        raise AssertionError("retrieval must not create an Experience")


def test_hybrid_retriever_miss_goes_directly_to_qa_fallback():
    recaller = ConfigurableRecaller({"unused": {"status": "completed"}})
    manager = FallbackManager(FallbackStorage([], [], []), FakeVectorStore(), recaller)

    result = HybridRetriever(manager).retriever(
        topic="旅行",
        core_entity="上海",
        query="我之前去过哪里？",
        intent="历史查询",
        state_key="session-1",
    )

    assert result["route_status"] == "qa_fallback"
    assert result["experiences"] == []
    assert result["segments"] == []
    assert result["qas"] == []
    assert result["history_experience"] == {}
    assert manager.experience_recaller.calls == []


def test_hybrid_retriever_runtime_hit_reads_existing_hierarchy():
    experience = {
        "experience_id": "exp-1",
        "topic": "旅行",
        "core_entity": "上海",
        "status": "open",
        "summary": "上海旅行",
        "history_experience": {},
    }
    segments = [
        {
            "segment_id": "seg-1",
            "intent": "规划",
            "summary": "制定路线",
            "created_at": "2026-01-01",
            "updated_at": "2026-01-01",
        }
    ]
    qas = [
        {
            "qa_id": "qa-1",
            "user_input": "想去上海",
            "assistant_output": "可以规划三天行程",
            "timestamp": "2026-01-01",
        }
    ]
    storage = FakeStorage(
        runtime={"current_experience_id": "exp-1"},
        experience=experience,
        segments=segments,
        qas=qas,
    )
    manager = FakeManager(storage)

    result = HybridRetriever(manager).retriever(
        topic="旅行",
        core_entity="上海",
        query="继续规划",
        state_key="session-1",
    )

    assert result["route_status"] == "runtime"
    assert result["experiences"] == [experience]
    assert result["segments"] == segments
    assert result["qas"] == qas
    assert manager.experience_recaller.calls == []


def test_read_only_compatibility_name_uses_same_implementation():
    assert issubclass(ReadOnlyHybridRetriever, HybridRetriever)


class FallbackStorage:
    def __init__(
        self,
        experiences,
        segments,
        qas,
        relational_qas=None,
    ):
        self.experiences = {item["experience_id"]: item for item in experiences}
        self.segments = {item["segment_id"]: item for item in segments}
        self.qas = {item["qa_id"]: item for item in qas}
        self.relational_qas = relational_qas or []

    def get_runtime_state(self, state_key):
        return None

    def find_active_experience(self, topic, core_entity):
        return None

    def get_experience(self, experience_id):
        return self.experiences.get(experience_id)

    def get_segment(self, segment_id):
        return self.segments.get(segment_id)

    def get_qa(self, qa_id):
        return self.qas.get(qa_id)

    def get_qas(self, qa_ids):
        return [self.qas[qa_id] for qa_id in qa_ids if qa_id in self.qas]

    def search_qas(self, **kwargs):
        return self.relational_qas[: kwargs["limit"]]


class FakeEmbedder:
    def embed(self, text):
        return [1.0, 0.0]


class FakeVectorStore:
    def __init__(
        self,
        *,
        route_items=None,
        qa_items=None,
        segment_items=None,
        experience_items=None,
    ):
        self.route_items = route_items or []
        self.qa_items = qa_items or []
        self.segment_items = segment_items or []
        self.experience_items = experience_items or []
        self.queries = []

    def query(self, embedding, *, memory_type, top_k, metadata_filter):
        self.queries.append(memory_type)
        return {
            "experience_route": self.route_items,
            "qa": self.qa_items,
            "segment": self.segment_items,
            "experience": self.experience_items,
        }[memory_type]


class ConfigurableRecaller(FakeRecaller):
    def __init__(self, experiences):
        super().__init__()
        self.experiences = experiences

    def recall(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "experiences": self.experiences,
            "history_experience": {},
        }


class FallbackManager(FakeManager):
    def __init__(self, storage, vector_store, recaller):
        super().__init__(storage)
        self.embedder = FakeEmbedder()
        self.vector_store = vector_store
        self.experience_recaller = recaller
        self.experience_similarity_threshold = 0.82


def _qa_hierarchy():
    experience = {
        "experience_id": "exp-hit",
        "topic": "旅行",
        "core_entity": "上海",
        "status": "open",
        "summary": {},
    }
    segment = {
        "segment_id": "seg-hit",
        "experience_id": "exp-hit",
        "intent": "回忆",
        "status": "open",
        "summary": {},
    }
    qa = {
        "qa_id": "qa-hit",
        "segment_id": "seg-hit",
        "topic": "旅行",
        "core_entity": "上海",
        "intent": "回忆",
        "status": "open",
        "user_input": "去年去了上海外滩",
        "assistant_output": "你很喜欢夜景",
        "timestamp": "2026-01-01",
    }
    return experience, segment, qa


def test_low_confidence_experience_route_falls_back_to_qa_before_history():
    experience, segment, qa = _qa_hierarchy()
    low_confidence_route = {
        "metadata": {"experience_id": "exp-hit"},
        "similarity": 0.5,
    }
    qa_match = {"metadata": {"qa_id": "qa-hit"}, "similarity": 0.91}
    vector_store = FakeVectorStore(
        route_items=[low_confidence_route],
        qa_items=[qa_match],
    )
    recaller = ConfigurableRecaller({"completed": {}})
    manager = FallbackManager(
        FallbackStorage([experience], [segment], [qa]),
        vector_store,
        recaller,
    )

    result = HybridRetriever(manager).retriever(
        topic="旅行",
        core_entity="上海",
        query="我去上海哪里玩过？",
        intent="回忆",
    )

    assert result["route_status"] == "qa_fallback"
    assert result["fallback_reason"] == "low_confidence"
    assert result["qas"] == [qa]
    assert recaller.calls == []


def test_route_vector_status_is_revalidated_by_sqlite():
    experience, segment, qa = _qa_hierarchy()
    experience["status"] = "completed"
    vector_store = FakeVectorStore(
        route_items=[
            {"metadata": {"experience_id": "exp-hit"}, "similarity": 0.99}
        ],
        qa_items=[{"metadata": {"qa_id": "qa-hit"}, "similarity": 0.91}],
    )
    recaller = ConfigurableRecaller({})
    manager = FallbackManager(
        FallbackStorage([experience], [segment], [qa]),
        vector_store,
        recaller,
    )

    result = HybridRetriever(manager).retriever(
        topic="旅行",
        core_entity="上海",
        query="我去上海哪里玩过？",
        intent="回忆",
    )

    assert result["route_status"] == "qa_fallback"
    assert result["fallback_reason"] == "no_vector_candidate"
    assert result["qas"] == [qa]
    assert recaller.calls == []


def test_no_experience_candidate_falls_back_to_qa_without_history_recall():
    experience, segment, qa = _qa_hierarchy()
    qa_match = {"metadata": {"qa_id": "qa-hit"}, "similarity": 0.91}
    vector_store = FakeVectorStore(qa_items=[qa_match])
    recaller = ConfigurableRecaller({})
    manager = FallbackManager(
        FallbackStorage([experience], [segment], [qa]),
        vector_store,
        recaller,
    )

    result = HybridRetriever(manager).retriever(
        topic="旅行",
        core_entity="上海",
        query="我去上海哪里玩过？",
        intent="回忆",
    )

    assert result["route_status"] == "qa_fallback"
    assert result["fallback_reason"] == "no_vector_candidate"
    assert result["qas"] == [qa]
    assert recaller.calls == []


def test_completed_history_recaller_is_not_used_by_retrieval_miss():
    vector_store = FakeVectorStore(
        qa_items=[{"metadata": {"qa_id": "unused"}, "similarity": 1.0}]
    )
    recaller = ConfigurableRecaller({"exp-old": {"status": "completed"}})
    manager = FallbackManager(FallbackStorage([], [], []), vector_store, recaller)

    result = HybridRetriever(manager).retriever(
        topic="旅行",
        core_entity="上海",
        query="以前的旅行怎么样？",
        intent="回忆",
    )

    assert result["route_status"] == "qa_fallback"
    assert result["qas"] == []
    assert recaller.calls == []
    assert vector_store.queries == [
        "experience_route",
        "qa",
        "segment",
        "experience",
    ]


def test_sqlite_keyword_candidate_can_reconstruct_hierarchy_without_dense_match():
    experience, segment, qa = _qa_hierarchy()
    vector_store = FakeVectorStore()
    recaller = ConfigurableRecaller({})
    storage = FallbackStorage(
        [experience],
        [segment],
        [qa],
        relational_qas=[qa],
    )
    manager = FallbackManager(storage, vector_store, recaller)
    qa["keyword_score"] = 10

    result = HybridRetriever(manager).retriever(
        topic="旅行",
        core_entity="上海",
        query="外滩夜景",
        intent="回忆",
    )

    assert result["route_status"] == "qa_fallback"
    assert result["fallback_reason"] == "no_vector_candidate"
    assert result["retrieval_channels"] == ["keyword"]
    assert result["experiences"] == [experience]
    assert result["segments"] == [segment]


def test_weak_structured_match_cannot_bypass_dense_threshold():
    experience, segment, qa = _qa_hierarchy()
    qa["keyword_score"] = 2
    storage = FallbackStorage(
        [experience],
        [segment],
        [qa],
        relational_qas=[qa],
    )
    manager = FallbackManager(storage, FakeVectorStore(), ConfigurableRecaller({}))

    result = HybridRetriever(manager).retriever(
        topic="different",
        core_entity="different",
        query="generic question",
        intent=qa["intent"],
    )

    assert result["qas"] == []


def test_segment_summary_provenance_expands_qa_fallback_candidates():
    experience, segment, qa = _qa_hierarchy()
    segment["summary"] = {
        "goal": "remember the trip",
        "key_facts": [
            {"fact": "visited the Bund", "source_qa_ids": [qa["qa_id"]]}
        ],
        "state_changes": [],
        "state": {"status": "ongoing", "current_conclusion": ""},
    }
    vector_store = FakeVectorStore(
        segment_items=[
            {
                "metadata": {"segment_id": segment["segment_id"]},
                "similarity": 0.91,
            }
        ]
    )
    manager = FallbackManager(
        FallbackStorage([experience], [segment], [qa]),
        vector_store,
        ConfigurableRecaller({}),
    )

    result = HybridRetriever(manager).retriever(
        topic="unmatched",
        core_entity="unmatched",
        query="where did I visit",
        intent="recall",
    )

    assert result["qas"] == [qa]
    assert result["retrieval_channels"] == ["summary"]
    assert result["qa_matches"][0]["parent_similarity"] == 0.91
