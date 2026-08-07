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
from memory.vector_store import ChromaVectorStore

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)


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
        vector_store = ChromaVectorStore(
            persist_path=config_path("paths", "chroma")
        )
        retriever = HybridRetriever(
            storage=storage,
            vector_store=vector_store,
            embedder=BailianEmbedder()
        )

        start = time.perf_counter()
        result = retriever.recall(
            topic='跨性别经历与LGBTQ倡导',
            core_entity='Caroline',
            intent='查询',
            entities= [
                        "Caroline",
                        "4年前",
                        "搬迁",
                        "来源地"
                    ],
            query='Where did Caroline move from 4 years ago?',
            query_confidence=0.7,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        output_data = {
            "elapsed_ms": round(elapsed_ms, 3),
            "result": result,
        }

        # 输出到控制台
        print("retrieval result:")
        print(json.dumps(output_data, ensure_ascii=False, indent=4))

        # 写入json文件
        with open("retrieval_result.json", "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
    finally:
        storage.close()
