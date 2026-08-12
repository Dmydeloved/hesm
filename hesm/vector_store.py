from __future__ import annotations

import json
import math
from pathlib import Path
import pickle
import sqlite3
import struct
from typing import Any
import uuid

from .config import config_path

try:
    import chromadb
except ImportError:  # pragma: no cover - exercised in environments without chromadb
    chromadb = None


class _InMemoryCollection:
    def __init__(self) -> None:
        self._items: dict[str, dict[str, Any]] = {}

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        for item_id, document, embedding, metadata in zip(
            ids, documents, embeddings, metadatas
        ):
            self._items[item_id] = {
                "id": item_id,
                "document": document,
                "embedding": embedding,
                "metadata": metadata,
            }

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int,
        where: dict[str, Any] | None = None,
        include: list[str] | None = None,
    ) -> dict[str, list[list[Any]]]:
        del include
        query_embedding = query_embeddings[0]
        candidates = [
            item for item in self._items.values()
            if self._matches_where(item["metadata"], where)
        ]
        ranked = sorted(
            candidates,
            key=lambda item: self._cosine_distance(query_embedding, item["embedding"]),
        )[:n_results]
        return {
            "ids": [[item["id"] for item in ranked]],
            "documents": [[item["document"] for item in ranked]],
            "metadatas": [[item["metadata"] for item in ranked]],
            "distances": [[self._cosine_distance(query_embedding, item["embedding"]) for item in ranked]],
        }

    def _matches_where(
        self, metadata: dict[str, Any], where: dict[str, Any] | None
    ) -> bool:
        """Small Chroma-compatible subset used by the offline fallback."""
        if not where:
            return True
        if "$and" in where:
            return all(
                self._matches_where(metadata, condition)
                for condition in where["$and"]
            )
        for key, condition in where.items():
            actual = metadata.get(key)
            if isinstance(condition, dict):
                if "$in" in condition and actual not in condition["$in"]:
                    return False
                if "$eq" in condition and actual != condition["$eq"]:
                    return False
            elif actual != condition:
                return False
        return True

    def count(self) -> int:
        return len(self._items)

    def _cosine_distance(
        self, left: list[float], right: list[float]
    ) -> float:
        dot = sum(left_value * right_value for left_value, right_value in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if not left_norm or not right_norm:
            return 1.0
        similarity = dot / (left_norm * right_norm)
        similarity = max(-1.0, min(1.0, similarity))
        return 1.0 - similarity


class _PersistentQueueBackedCollection(_InMemoryCollection):
    """Read-only fallback backed by Chroma's sqlite queue snapshots."""

    def __init__(self, sqlite_path: Path) -> None:
        super().__init__()
        self.sqlite_path = sqlite_path
        if not self.sqlite_path.exists():
            raise FileNotFoundError(f"Chroma sqlite file not found: {self.sqlite_path}")
        self._load_items()

    def _load_items(self) -> None:
        connection = sqlite3.connect(self.sqlite_path)
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(
                """
                SELECT seq_id, id, vector, metadata
                FROM embeddings_queue
                WHERE vector IS NOT NULL AND metadata IS NOT NULL
                ORDER BY seq_id DESC
                """
            ).fetchall()
        finally:
            connection.close()

        seen_ids: set[str] = set()
        for row in rows:
            item_id = str(row["id"])
            if item_id in seen_ids:
                continue
            metadata = json.loads(row["metadata"])
            self._items[item_id] = {
                "id": item_id,
                "document": metadata.get("chroma:document", ""),
                "embedding": self._decode_float32_vector(row["vector"]),
                "metadata": metadata,
            }
            seen_ids.add(item_id)

    def _decode_float32_vector(self, blob: bytes) -> list[float]:
        if not blob:
            return []
        return [value[0] for value in struct.iter_unpack("<f", blob)]


class _PersistentReadOnlyCollection(_PersistentQueueBackedCollection):
    """Read all persisted HNSW vectors and overlay the unapplied queue entries."""

    def _load_items(self) -> None:
        connection = sqlite3.connect(self.sqlite_path)
        connection.row_factory = sqlite3.Row
        try:
            vector_segment = connection.execute(
                "SELECT id FROM segments WHERE scope = 'VECTOR' LIMIT 1"
            ).fetchone()
            collection = connection.execute(
                "SELECT dimension FROM collections LIMIT 1"
            ).fetchone()
            if vector_segment is None or collection is None:
                super()._load_items()
                return
            dimension = int(collection["dimension"] or 0)
            index_dir = self.sqlite_path.parent / str(vector_segment["id"])
            metadata_path = index_dir / "index_metadata.pickle"
            data_path = index_dir / "data_level0.bin"
            if not dimension or not metadata_path.exists() or not data_path.exists():
                super()._load_items()
                return

            with metadata_path.open("rb") as handle:
                index_metadata = pickle.load(handle)
            id_to_label = index_metadata.get("id_to_label") or {}
            total_elements = int(index_metadata.get("total_elements_added") or 0)
            raw_vectors = data_path.read_bytes()
            if not total_elements or not id_to_label:
                super()._load_items()
                return
            record_size = len(raw_vectors) // total_elements
            vector_size = dimension * 4
            # hnswlib stores level-0 links, then float32 vector, then uint64 label.
            vector_offset = record_size - vector_size - 8
            if vector_offset < 0 or record_size * total_elements != len(raw_vectors):
                raise ValueError("Unsupported persisted HNSW level-0 layout")

            metadata_by_id: dict[str, dict[str, Any]] = {}
            rows = connection.execute(
                """
                SELECT e.embedding_id, m.key, m.string_value, m.int_value,
                       m.float_value, m.bool_value
                FROM embeddings e
                JOIN embedding_metadata m ON m.id = e.id
                """
            ).fetchall()
            for row in rows:
                value: Any
                if row["string_value"] is not None:
                    value = row["string_value"]
                elif row["int_value"] is not None:
                    value = row["int_value"]
                elif row["float_value"] is not None:
                    value = row["float_value"]
                else:
                    value = bool(row["bool_value"])
                metadata_by_id.setdefault(str(row["embedding_id"]), {})[
                    str(row["key"])
                ] = value

            for item_id, label in id_to_label.items():
                start = int(label) * record_size + vector_offset
                end = start + vector_size
                if start < 0 or end > len(raw_vectors):
                    continue
                metadata = dict(metadata_by_id.get(str(item_id), {}))
                document = str(metadata.pop("chroma:document", ""))
                self._items[str(item_id)] = {
                    "id": str(item_id),
                    "document": document,
                    "embedding": list(struct.unpack(f"<{dimension}f", raw_vectors[start:end])),
                    "metadata": metadata,
                }

            queue_rows = connection.execute(
                """
                SELECT seq_id, id, vector, metadata
                FROM embeddings_queue
                WHERE vector IS NOT NULL AND metadata IS NOT NULL
                ORDER BY seq_id
                """
            ).fetchall()
            for row in queue_rows:
                metadata = json.loads(row["metadata"])
                item_id = str(row["id"])
                self._items[item_id] = {
                    "id": item_id,
                    "document": metadata.pop("chroma:document", ""),
                    "embedding": self._decode_float32_vector(row["vector"]),
                    "metadata": metadata,
                }
        finally:
            connection.close()


class ChromaVectorStore:
    """Persistent Chroma collection for all memory-layer vectors."""

    def __init__(
        self,
        persist_path: str | Path | None = None,
        collection_name: str = "topic_memory",
        ephemeral: bool = False,
        read_only: bool = False,
    ) -> None:
        self.persist_path = Path(persist_path) if persist_path is not None else config_path("paths", "chroma")
        if read_only:
            self.client = None
            self.collection = _PersistentReadOnlyCollection(
                self.persist_path / "chroma.sqlite3"
            )
            return
        if not ephemeral:
            self.persist_path.mkdir(parents=True, exist_ok=True)
        if chromadb is None:
            self.client = None
            self.collection = (
                _InMemoryCollection()
                if ephemeral
                else _PersistentQueueBackedCollection(self.persist_path / "chroma.sqlite3")
            )
            return
        self.client = (
            chromadb.EphemeralClient()
            if ephemeral
            else chromadb.PersistentClient(path=str(self.persist_path))
        )
        if ephemeral:
            collection_name = f"{collection_name}_{uuid.uuid4().hex}"
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def upsert(
        self,
        memory_type: str,
        memory_id: str,
        text: str,
        embedding: list[float],
        updated_at: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        vector_metadata: dict[str, Any] = {
            "memory_type": memory_type,
            "memory_id": memory_id,
            "updated_at": updated_at,
        }
        for key, value in (metadata or {}).items():
            if value is None:
                continue
            vector_metadata[key] = (
                json.dumps(value, ensure_ascii=False)
                if isinstance(value, (list, dict))
                else value
            )
        self.collection.upsert(
            ids=[f"{memory_type}:{memory_id}"],
            documents=[text],
            embeddings=[embedding],
            metadatas=[vector_metadata],
        )

    def query(
        self,
        query_embedding: list[float],
        memory_type: str = "qa",
        top_k: int = 20,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if self.collection.count() == 0:
            return []
        filters: list[dict[str, Any]] = [{"memory_type": memory_type}]
        for key, value in (metadata_filter or {}).items():
            filters.append({
                key: {"$in": value} if isinstance(value, list) else value
            })
        where = filters[0] if len(filters) == 1 else {"$and": filters}
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        items = []
        for index, chroma_id in enumerate(result["ids"][0]):
            distance = float(result["distances"][0][index])
            items.append({
                "chroma_id": chroma_id,
                "document": result["documents"][0][index],
                "metadata": result["metadatas"][0][index],
                "distance": distance,
                "similarity": max(0.0, min(1.0, 1.0 - distance)),
            })
        return items

    def count(self) -> int:
        return self.collection.count()
