import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory.embedder import BailianEmbedder
from memory.manager import MemoryManager
from memory.storage import MemoryStorage
from memory.vector_store import ChromaVectorStore


def topic_result():
    return {
        "topic": "restaurant",
        "core_entity": "center expensive restaurant",
        "intent": "find_restaurant",
        "entities": ["restaurant", "center", "expensive"],
        "confidence": 0.92,
        "reasoning": "The user wants an expensive restaurant in the center.",
    }


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        storage = MemoryStorage(Path(directory) / "topic_memory.sqlite3")
        vector_store = ChromaVectorStore(ephemeral=True)
        manager = MemoryManager(
            storage=storage,
            vector_store=vector_store,
            embedder=BailianEmbedder(),
        )

        result = manager.add_qa(
            topic_result=topic_result(),
            user_input="i need a place to dine in the center thats expensive",
            assistant_output="I can help you find an expensive restaurant in the center.",
            state_key="test_manager",
        )

        print("memory build result:")
        print(json.dumps(result, ensure_ascii=False, indent=4))
        print("row counts:")
        print(
            json.dumps(
                {
                    "qa_memory": storage.count_rows("qa_memory"),
                    "segment_memory": storage.count_rows("segment_memory"),
                    "experience_memory": storage.count_rows("experience_memory"),
                    "runtime_state": storage.count_rows("runtime_state"),
                    "vector_memory": vector_store.count(),
                },
                ensure_ascii=False,
                indent=4,
            )
        )
        storage.close()
