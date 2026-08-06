import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from memory.config import config_path
from memory.retriever import HybridRetriever
from memory.storage import MemoryStorage
from memory.embedder import BailianEmbedder




def load_query(storage):
    row = storage.connection.execute(
        """
        SELECT topic, core_entity, intent, entities_json
        FROM qa_memory
        WHERE status = 'active'
        ORDER BY timestamp
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        raise RuntimeError("No active QA memory found. Build memory before retrieval.")

    return {
        "topic": row["topic"],
        "core_entity": row["core_entity"],
        "intent": row["intent"],
        "entities": json.loads(row["entities_json"] or "[]"),
    }


if __name__ == "__main__":
    storage = MemoryStorage(config_path("paths", "memory_db"))
    try:
        # query = load_query(storage)
        retriever = HybridRetriever(
            storage=storage,
            embedder=BailianEmbedder()
        )

        start = time.perf_counter()
        result = retriever.recall(
            topic='人物经历',
            core_entity='Caroline',
            intent='时间查询',
            entities='',
            query='When did Caroline go to the LGBTQ support group?',
            top_experience=1,
            top_segment=1,
            top_qa=1,
            state_key="test_retriever",
            use_cache=False,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        print("retrieval result:")
        print(
            json.dumps(
                {
                    "elapsed_ms": round(elapsed_ms, 3),
                    "result": result,
                },
                ensure_ascii=False,
                indent=4,
            )
        )
    finally:
        storage.close()
