from __future__ import annotations

import json
import logging
import math
from pathlib import Path
import pickle
import sqlite3
import struct
from typing import Any
import uuid

from .config import config_path
from .time_utils import format_timestamp


logger = logging.getLogger(__name__)

try:
    import chromadb
except ImportError:  # pragma: no cover - 在未安装 Chroma 的环境中使用本地回退实现
    chromadb = None


MEMORY_TYPES = {"qa", "segment", "experience", "experience_route"}
TIME_METADATA_FIELDS = {"timestamp", "created_at", "updated_at"}


def _metadata_timestamp(value: Any) -> str:
    """规范向量元数据中的非空时间字段。"""
    text = str(value or "").strip()
    return format_timestamp(text) if text else ""


def _normalize_chroma_sqlite_timestamps(sqlite_path: Path) -> int:
    """在 Chroma 客户端启动前规范其 SQLite 元数据中的历史时间。"""
    if not sqlite_path.exists():
        return 0
    connection = sqlite3.connect(sqlite_path)
    connection.row_factory = sqlite3.Row
    changed = 0
    try:
        tables = {
            str(row["name"])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        if "embedding_metadata" in tables:
            placeholders = ", ".join("?" for _ in TIME_METADATA_FIELDS)
            rows = connection.execute(
                "SELECT id, key, string_value FROM embedding_metadata "
                f"WHERE key IN ({placeholders}) AND string_value IS NOT NULL",
                tuple(TIME_METADATA_FIELDS),
            ).fetchall()
            for row in rows:
                original = str(row["string_value"] or "").strip()
                if not original:
                    continue
                try:
                    normalized = format_timestamp(original)
                except ValueError:
                    continue
                if normalized == original:
                    continue
                connection.execute(
                    "UPDATE embedding_metadata SET string_value = ? "
                    "WHERE id = ? AND key = ?",
                    (normalized, row["id"], row["key"]),
                )
                changed += 1

        # 队列中也保存了一份完整元数据，必须同步更新以免重放后恢复旧格式。
        if "embeddings_queue" in tables:
            rows = connection.execute(
                "SELECT seq_id, metadata FROM embeddings_queue "
                "WHERE metadata IS NOT NULL"
            ).fetchall()
            for row in rows:
                try:
                    metadata = json.loads(row["metadata"])
                except (json.JSONDecodeError, TypeError):
                    continue
                if not isinstance(metadata, dict):
                    continue
                metadata_changed = False
                for field in TIME_METADATA_FIELDS:
                    original = str(metadata.get(field) or "").strip()
                    if not original:
                        continue
                    try:
                        normalized = format_timestamp(original)
                    except ValueError:
                        continue
                    if normalized != original:
                        metadata[field] = normalized
                        metadata_changed = True
                if metadata_changed:
                    connection.execute(
                        "UPDATE embeddings_queue SET metadata = ? WHERE seq_id = ?",
                        (json.dumps(metadata, ensure_ascii=False), row["seq_id"]),
                    )
                    changed += 1
        connection.commit()
        if changed:
            logger.info(
                "Normalized Chroma SQLite timestamp metadata path=%s changed=%s",
                sqlite_path,
                changed,
            )
        return changed
    finally:
        connection.close()


def _json_object(value: Any) -> dict[str, Any]:
    """把结构化摘要统一转换为字典。"""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _text_list(value: Any) -> list[str]:
    """提取非空文本列表，并兼容包含事实对象的列表。"""
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if isinstance(item, dict):
            text = str(item.get("fact") or item.get("result") or "").strip()
        else:
            text = str(item or "").strip()
        if text:
            result.append(text)
    return result


def _state_change_text(change: Any) -> str:
    """把状态变化对象转换为适合嵌入的自然语言，避免直接序列化 JSON。"""
    if not isinstance(change, dict):
        return str(change or "").strip()
    old_state = str(change.get("from") or "").strip()
    new_state = str(change.get("to") or "").strip()
    reason = str(change.get("reason") or "").strip()
    if not old_state and not new_state:
        return reason
    text = f"状态从“{old_state or '未说明'}”变为“{new_state or '未说明'}”"
    return f"{text}，原因是{reason}。" if reason else f"{text}。"


def build_vector_document(memory_type: str, memory: dict[str, Any]) -> str:
    """按设计文档构造四类 Chroma Document。"""
    if memory_type not in MEMORY_TYPES:
        raise ValueError(f"Unsupported memory type: {memory_type}")

    topic = str(memory.get("topic") or "").strip()
    core_entity = str(memory.get("core_entity") or "").strip()
    if memory_type == "qa":
        return "\n".join(
            [
                f"主题：{topic}",
                f"核心实体：{core_entity}",
                f"意图：{memory.get('intent', '')}",
                f"实体：{'、'.join(_text_list(memory.get('entities')))}",
                f"用户输入：{memory.get('user_input', '')}",
                f"助手回答：{memory.get('assistant_output', '')}",
            ]
        )

    summary = _json_object(memory.get("summary"))
    if memory_type == "segment":
        state = _json_object(summary.get("state"))
        changes = [
            text
            for text in (
                _state_change_text(item)
                for item in summary.get("state_changes", [])
            )
            if text
        ]
        return "\n".join(
            [
                f"主题：{topic}",
                f"核心实体：{core_entity}",
                f"意图：{memory.get('intent', '')}",
                f"阶段目标：{summary.get('goal', '')}",
                f"关键事实：{'；'.join(_text_list(summary.get('key_facts')))}",
                f"状态变化：{'；'.join(changes)}",
                f"当前结论：{state.get('current_conclusion', '')}",
            ]
        )

    goal = str(summary.get("goal") or "").strip()
    if memory_type == "experience_route":
        return "\n".join(
            [f"主题：{topic}", f"核心实体：{core_entity}", f"目标：{goal}"]
        )

    current_state = _json_object(summary.get("current_state"))
    trajectory = [
        f"{item.get('intent', '')}：{item.get('result', '')}"
        for item in summary.get("stage_trajectory", [])
        if isinstance(item, dict)
    ]
    return "\n".join(
        [
            f"主题：{topic}",
            f"核心实体：{core_entity}",
            f"长期目标：{goal}",
            f"阶段轨迹：{'；'.join(trajectory)}",
            f"稳定事实：{'；'.join(_text_list(summary.get('stable_facts')))}",
            f"当前结论：{current_state.get('summary', '')}",
        ]
    )


def build_vector_metadata(
    memory_type: str,
    memory: dict[str, Any],
) -> dict[str, Any]:
    """按四类向量的字段定义构造 Chroma Metadata。"""
    if memory_type not in MEMORY_TYPES:
        raise ValueError(f"Unsupported memory type: {memory_type}")
    common = {
        "topic": str(memory.get("topic") or ""),
        "core_entity": str(memory.get("core_entity") or ""),
        "status": str(memory.get("status") or ""),
    }
    if memory_type == "qa":
        return {
            **common,
            "qa_id": str(memory.get("qa_id") or ""),
            "segment_id": str(memory.get("segment_id") or ""),
            "intent": str(memory.get("intent") or ""),
            "timestamp": _metadata_timestamp(memory.get("timestamp")),
        }
    if memory_type == "segment":
        return {
            **common,
            "segment_id": str(memory.get("segment_id") or ""),
            "experience_id": str(memory.get("experience_id") or ""),
            "intent": str(memory.get("intent") or ""),
            "created_at": _metadata_timestamp(memory.get("created_at")),
            "updated_at": _metadata_timestamp(memory.get("updated_at")),
        }
    return {
        **common,
        "experience_id": str(memory.get("experience_id") or ""),
        "created_at": _metadata_timestamp(memory.get("created_at")),
        "updated_at": _metadata_timestamp(memory.get("updated_at")),
    }


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
        """实现离线回退所需的最小 Chroma 过滤语义。"""
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
    """从 Chroma SQLite 队列快照加载数据的只读回退集合。"""

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
    """读取持久化 HNSW 向量，并覆盖尚未应用的队列记录。"""

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
            # hnswlib 依次保存零层链接、float32 向量和 uint64 标签。
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
    """统一管理四类 HESM 记忆向量的持久化 Chroma 集合。"""

    def __init__(
        self,
        persist_path: str | Path | None = None,
        collection_name: str = "topic_memory",
        ephemeral: bool = False,
        read_only: bool = False,
    ) -> None:
        self.persist_path = Path(persist_path) if persist_path is not None else config_path("paths", "chroma")
        logger.info(
            "Initializing vector store persist_path=%s collection=%s ephemeral=%s read_only=%s",
            self.persist_path,
            collection_name,
            ephemeral,
            read_only,
        )
        if read_only:
            self.client = None
            self.collection = _PersistentReadOnlyCollection(
                self.persist_path / "chroma.sqlite3"
            )
            logger.info(
                "Vector store initialized in read-only mode count=%s",
                self.collection.count(),
            )
            return
        if not ephemeral:
            self.persist_path.mkdir(parents=True, exist_ok=True)
            _normalize_chroma_sqlite_timestamps(
                self.persist_path / "chroma.sqlite3"
            )
        if chromadb is None:
            self.client = None
            self.collection = (
                _InMemoryCollection()
                if ephemeral
                else _PersistentQueueBackedCollection(self.persist_path / "chroma.sqlite3")
            )
            logger.info(
                "Chroma unavailable; vector store switched to fallback collection ephemeral=%s count=%s",
                ephemeral,
                self.collection.count(),
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
        logger.info(
            "Vector store initialized collection=%s count=%s",
            collection_name,
            self.collection.count(),
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
        if memory_type not in MEMORY_TYPES:
            raise ValueError(f"Unsupported memory type: {memory_type}")
        if not str(text or "").strip():
            raise ValueError("Vector document cannot be empty")
        if not embedding:
            raise ValueError("Vector embedding cannot be empty")
        vector_metadata: dict[str, Any] = {
            "memory_type": memory_type,
            "memory_id": memory_id,
            "updated_at": format_timestamp(updated_at),
        }
        for key, value in (metadata or {}).items():
            if value is None:
                continue
            if key in TIME_METADATA_FIELDS:
                vector_metadata[key] = _metadata_timestamp(value)
            else:
                vector_metadata[key] = (
                    json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, dict))
                    else value
                )
        logger.info(
            "Vector upsert started memory_type=%s memory_id=%s text_length=%s metadata=%s",
            memory_type,
            memory_id,
            len(text),
            json.dumps(vector_metadata, ensure_ascii=False),
        )
        self.collection.upsert(
            ids=[f"{memory_type}:{memory_id}"],
            documents=[text],
            embeddings=[embedding],
            metadatas=[vector_metadata],
        )
        logger.info(
            "Vector upsert completed memory_type=%s memory_id=%s",
            memory_type,
            memory_id,
        )

    def normalize_timestamps(self) -> int:
        """将已有 Chroma 记录中的时间元数据迁移为标准格式。"""
        if not hasattr(self.collection, "get") or not hasattr(self.collection, "update"):
            return 0
        result = self.collection.get(include=["metadatas"])
        ids = result.get("ids") or []
        metadatas = result.get("metadatas") or []
        changed_ids: list[str] = []
        changed_metadatas: list[dict[str, Any]] = []
        for item_id, metadata in zip(ids, metadatas):
            if not isinstance(metadata, dict):
                continue
            normalized_metadata = dict(metadata)
            changed = False
            for field in TIME_METADATA_FIELDS:
                original = str(normalized_metadata.get(field) or "").strip()
                if not original:
                    continue
                try:
                    normalized = format_timestamp(original)
                except ValueError:
                    continue
                if normalized != original:
                    normalized_metadata[field] = normalized
                    changed = True
            if changed:
                changed_ids.append(str(item_id))
                changed_metadatas.append(normalized_metadata)
        if changed_ids:
            self.collection.update(
                ids=changed_ids,
                metadatas=changed_metadatas,
            )
        logger.info("Vector timestamp normalization completed changed=%s", len(changed_ids))
        return len(changed_ids)

    def query(
        self,
        query_embedding: list[float],
        memory_type: str = "qa",
        top_k: int = 20,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if memory_type not in MEMORY_TYPES:
            raise ValueError(f"Unsupported memory type: {memory_type}")
        if self.collection.count() == 0:
            logger.info("Vector query skipped because collection is empty memory_type=%s", memory_type)
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
        logger.info(
            "Vector query completed memory_type=%s top_k=%s metadata_filter=%s result_count=%s",
            memory_type,
            top_k,
            json.dumps(metadata_filter or {}, ensure_ascii=False),
            len(items),
        )
        return items

    def count(self) -> int:
        return self.collection.count()


__all__ = [
    "ChromaVectorStore",
    "MEMORY_TYPES",
    "build_vector_document",
    "build_vector_metadata",
]
