from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

from experiments.locomo.data.loader import Session, Turn
from experiments.locomo.methods.amem_adapter import AMEMMemory
from experiments.locomo.methods.hesm_adapter import HESMAblationMemory, HESMMemory
from experiments.locomo.methods.mem0_adapter import Mem0Memory


def _sessions() -> list[Session]:
    return [
        Session(
            1,
            "2026-01-01",
            [
                Turn("A", "first", "D1:1", 1, "2026-01-01"),
                Turn("B", "second", "D1:2", 1, "2026-01-01"),
            ],
        )
    ]


class _FakeMem0Backend:
    def __init__(self) -> None:
        self.add_calls: list[str] = []

    def add(self, text: str, **kwargs) -> None:
        self.add_calls.append(kwargs["metadata"]["dia_id"])


class _TestMem0Memory(Mem0Memory):
    def __init__(self, memory_root: Path) -> None:
        super().__init__(memory_root, {})
        self.backend = _FakeMem0Backend()

    def _create_mem0(self, conv_id: str) -> _FakeMem0Backend:
        return self.backend


class _TestAMEMMemory(AMEMMemory):
    def __init__(self, memory_root: Path) -> None:
        super().__init__(memory_root, {}, {})
        self.add_calls: list[str] = []
        self.vector_dia_ids: set[str] = set()

    def _setup_components(self, conv_id: str) -> None:
        if self._db is not None:
            self._db.close()
        base = self._memory_root / f"amem_{conv_id}"
        base.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(base / "amem_notes.sqlite3"))
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS notes "
            "(note_id TEXT PRIMARY KEY, dia_id TEXT)"
        )
        self._db.commit()

    def _add_note(self, text: str, dia_id: str, timestamp: str) -> str:
        assert self._db is not None
        self.add_calls.append(dia_id)
        self._db.execute(
            "INSERT OR REPLACE INTO notes (note_id, dia_id) VALUES (?, ?)",
            (f"note-{dia_id}", dia_id),
        )
        self._db.commit()
        self.vector_dia_ids.add(dia_id)
        return f"note-{dia_id}"

    def _stored_vector_dia_ids(self) -> set[str]:
        return set(self.vector_dia_ids)


class _FakeHESMVectorStore:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def count(self) -> int:
        return int(
            self.connection.execute("SELECT COUNT(*) FROM qa_memory").fetchone()[0]
        )


class _FakeHESMManager:
    def __init__(self, connection: sqlite3.Connection, calls: list[str]) -> None:
        self.connection = connection
        self.calls = calls

    def add_qa(self, **kwargs) -> None:
        tools = kwargs.get("tools") or []
        dia_id = str(tools[0]["dia_id"])
        self.calls.append(dia_id)
        self.connection.execute(
            "INSERT INTO qa_memory (tools_json) VALUES (?)",
            (json.dumps(tools),),
        )
        self.connection.commit()


class _TestHESMMemory(HESMMemory):
    def __init__(self, memory_root: Path) -> None:
        super().__init__(
            memory_root=memory_root,
            hesm_cfg={},
            use_llm_summarizer=False,
            use_llm_reranker=False,
        )
        self.add_calls: list[str] = []

    def _setup_components(self, conv_id: str) -> None:
        old_storage = self._storage
        if old_storage is not None:
            old_storage.connection.close()
        base = self._memory_root / f"hesm_{conv_id}"
        base.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(base / "memory.sqlite3"))
        connection.row_factory = sqlite3.Row
        connection.execute(
            "CREATE TABLE IF NOT EXISTS qa_memory (tools_json TEXT NOT NULL)"
        )
        connection.commit()
        self._storage = SimpleNamespace(connection=connection)
        self._vector_store = _FakeHESMVectorStore(connection)
        self._extractor = SimpleNamespace(
            extract=lambda **kwargs: {
                "topic": "topic",
                "core_entity": "entity",
                "intent": "intent",
            }
        )
        self._manager = _FakeHESMManager(connection, self.add_calls)
        self._retriever = SimpleNamespace()


class MemoryBuildResumeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_mem0_skips_add_when_manifest_is_complete(self) -> None:
        method = _TestMem0Memory(self.root)
        method.build_memory("conv-1", _sessions(), "A", "B")
        self.assertEqual(method.backend.add_calls, ["D1:1", "D1:2"])

        method.reset()
        method.build_memory("conv-1", _sessions(), "A", "B")
        self.assertEqual(method.backend.add_calls, ["D1:1", "D1:2"])
        self.assertTrue(
            (self.root / "mem0_conv-1" / "build_state.json").exists()
        )

    def test_amem_skips_add_when_sqlite_memory_is_complete(self) -> None:
        method = _TestAMEMMemory(self.root)
        method.build_memory("conv-1", _sessions(), "A", "B")
        self.assertEqual(method.add_calls, ["D1:1", "D1:2"])

        method.reset()
        method.build_memory("conv-1", _sessions(), "A", "B")
        self.assertEqual(method.add_calls, ["D1:1", "D1:2"])
        self.assertTrue(
            (self.root / "amem_conv-1" / "build_state.json").exists()
        )

    def test_amem_does_not_treat_sqlite_only_note_as_complete(self) -> None:
        method = _TestAMEMMemory(self.root)
        method._setup_components("conv-1")
        assert method._db is not None
        method._db.execute(
            "INSERT INTO notes (note_id, dia_id) VALUES (?, ?)",
            ("dangling-note", "D1:1"),
        )
        method._db.commit()

        method.build_memory("conv-1", _sessions(), "A", "B")
        self.assertEqual(method.add_calls, ["D1:1", "D1:2"])

    def test_hesm_skips_complete_memory_on_retry(self) -> None:
        method = _TestHESMMemory(self.root)
        method.build_memory("conv-1", _sessions(), "A", "B")
        self.assertEqual(method.add_calls, ["D1:1", "D1:2"])

        method.reset()
        method.build_memory("conv-1", _sessions(), "A", "B")
        self.assertEqual(method.add_calls, ["D1:1", "D1:2"])

    def test_hesm_partial_memory_builds_only_missing_turns(self) -> None:
        method = _TestHESMMemory(self.root)
        method._setup_components("conv-1")
        method._storage.connection.execute(
            "INSERT INTO qa_memory (tools_json) VALUES (?)",
            (json.dumps([{"dia_id": "D1:1"}]),),
        )
        method._storage.connection.commit()

        method.build_memory("conv-1", _sessions(), "A", "B")
        self.assertEqual(method.add_calls, ["D1:2"])

    def test_hesm_retrieval_does_not_forward_legacy_use_cache(self) -> None:
        class RetrieverWithoutCacheArgument:
            def recall(
                self,
                topic,
                core_entity,
                intent=None,
                entities=None,
                query=None,
                *,
                query_confidence,
                top_experience=3,
                top_segment=5,
                top_qa=8,
            ):
                return {"qas": [], "context_text": "ready"}

        extractor = SimpleNamespace(
            extract=lambda **kwargs: {
                "topic": "topic",
                "core_entity": "entity",
                "intent": "intent",
                "confidence": 1.0,
            }
        )
        retriever = RetrieverWithoutCacheArgument()

        method = HESMMemory(self.root, {}, use_cache=True)
        method._extractor = extractor
        method._retriever = retriever
        result = method.retrieve("question", use_cache=True)
        self.assertEqual(result.context_text, "ready")
        self.assertNotIn("error", result.raw_result)

        ablation = HESMAblationMemory(
            variant="full_hesm",
            memory_root=self.root,
            hesm_cfg={},
            variant_cfg={"retrieval_layers": ["experience", "segment", "qa"]},
        )
        ablation._extractor = extractor
        ablation._retriever = retriever
        result = ablation.retrieve("question")
        self.assertEqual(result.context_text, "ready")
        self.assertNotIn("error", result.raw_result)

    def test_memory_method_llms_have_complete_isolated_config(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        config = yaml.safe_load(
            (project_root / "configs" / "config.yaml").read_text("utf-8")
        )
        methods = config["memory_methods"]
        profiles = config["_llm_profiles"]
        for profile in profiles.values():
            self.assertTrue(profile["model"])
            self.assertTrue(profile["base_url"])
            self.assertGreaterEqual(int(profile["max_retries"]), 1)
        for method in ("mem0", "amem"):
            self.assertTrue(methods[method]["llm"]["model"])
            self.assertTrue(methods[method]["llm"]["base_url"])
            self.assertTrue(methods[method]["embedding"]["model"])
        for role in ("topic_extraction", "summarization", "retrieval"):
            self.assertTrue(methods["hesm"][role]["model"])
            self.assertTrue(methods["hesm"][role]["base_url"])


if __name__ == "__main__":
    unittest.main()
