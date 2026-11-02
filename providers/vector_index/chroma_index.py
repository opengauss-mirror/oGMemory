"""ChromaDB-backed Vector Index with persistence.

Data survives server restarts. Uses ChromaDB's embedding-free mode - vectors
come from the Embedder, not from ChromaDB's built-in models.
"""

import hashlib
import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from core.interfaces import VectorIndex
from core.memory_names import memory_directory
from core.models import IndexRecord, SeedHit
from core.resource_identity import CODE_INDEX_METADATA_FIELDS

logger = logging.getLogger(__name__)

DEFAULT_CHROMA_BATCH_SIZE = 5_000
_CHROMA_LOCKS: dict[str, threading.RLock] = {}
_CHROMA_LOCKS_GUARD = threading.Lock()
_WINDOWS_LOCK_RETRY_SECONDS = 0.05


def _get_process_lock(lock_path: str) -> threading.RLock:
    with _CHROMA_LOCKS_GUARD:
        lock = _CHROMA_LOCKS.get(lock_path)
        if lock is None:
            lock = threading.RLock()
            _CHROMA_LOCKS[lock_path] = lock
        return lock


def _is_windows_lock_conflict(exc: OSError) -> bool:
    winerror = getattr(exc, "winerror", None)
    if winerror is not None:
        return winerror in {33, 36}
    return getattr(exc, "errno", None) in {11, 13}


def _lock_windows_file(fh) -> None:
    import msvcrt

    while True:
        try:
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            return
        except OSError as exc:
            if not _is_windows_lock_conflict(exc):
                raise
            time.sleep(_WINDOWS_LOCK_RETRY_SECONDS)


def _unlock_windows_file(fh) -> None:
    import msvcrt

    fh.seek(0)
    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)


@contextmanager
def _chroma_directory_lock(lock_path: str):
    process_lock = _get_process_lock(lock_path)
    with process_lock:
        with open(lock_path, "a+b") as fh:
            if os.name == "nt":
                _lock_windows_file(fh)
                try:
                    yield
                finally:
                    _unlock_windows_file(fh)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


class ChromaVectorIndex(VectorIndex):
    """Persistent vector index backed by ChromaDB."""

    def __init__(
        self,
        collection_name: str = "contextengine",
        persist_directory: str = ".chroma_data",
        dimension: int = 1536,
    ):
        import chromadb

        self._dimension = dimension
        self._persist_directory = str(Path(persist_directory))
        Path(self._persist_directory).mkdir(parents=True, exist_ok=True)
        self._lock_path = str(Path(self._persist_directory) / ".ogmemory-chroma.lock")
        with self._locked():
            self._client = chromadb.PersistentClient(path=self._persist_directory)
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            existing = self._collection.count()
        logger.info(
            "ChromaVectorIndex ready: collection=%s, dir=%s, existing=%d records",
            collection_name, self._persist_directory, existing,
        )

    def _locked(self):
        return _chroma_directory_lock(self._lock_path)

    def upsert(self, records: list[IndexRecord]) -> None:
        if not records:
            return
        ids = [r.id for r in records]
        documents = [r.text for r in records]
        metadatas = [self._record_to_meta(r) for r in records]
        embeddings = []
        for r in records:
            real_emb = (r.metadata or {}).get("_embedding")
            if real_emb and isinstance(real_emb, list):
                embeddings.append(real_emb)
            else:
                embeddings.append(self._stable_vector(r.text))
        batch_size = self._provider_batch_size()
        with self._locked():
            for start in range(0, len(ids), batch_size):
                end = start + batch_size
                self._collection.upsert(
                    ids=ids[start:end],
                    embeddings=embeddings[start:end],
                    documents=documents[start:end],
                    metadatas=metadatas[start:end],
                )

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        batch_size = self._provider_batch_size()
        with self._locked():
            for start in range(0, len(ids), batch_size):
                self._collection.delete(ids=ids[start : start + batch_size])

    def search_by_vector(
        self,
        query_vector: list[float],
        filters: dict[str, Any],
        top_k: int,
    ) -> list[SeedHit]:
        chroma_where = self._build_where(filters)
        with self._locked():
            if "memory_directory" in filters:
                # Derive type constraints from persisted URIs before nearest-neighbor truncation.
                uris = set()
                offset = 0
                while True:
                    page = self._collection.get(
                        where=chroma_where, include=["metadatas"], limit=1000, offset=offset,
                    )
                    for meta in page["metadatas"]:
                        uri = (meta or {}).get("uri", "")
                        if memory_directory(uri) in filters["memory_directory"]:
                            uris.add(uri)
                    if len(page["ids"]) < 1000:
                        break
                    offset += len(page["ids"])
                if not uris:
                    return []
                chroma_where = self._build_where({**filters, "uri": sorted(uris)})
            count = self._collection.count()
            if count == 0:
                return []
            results = self._collection.query(
                query_embeddings=[query_vector],
                n_results=min(top_k, count),
                where=chroma_where if chroma_where else None,
                include=["documents", "metadatas", "distances"],
            )
        hits: list[SeedHit] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return hits
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            meta = dict(meta)
            if "agent_memory_grounding_json" in meta:
                meta["grounding"] = json.loads(meta.pop("agent_memory_grounding_json"))
            doc = results["documents"][0][i] if results["documents"] else ""
            dist = results["distances"][0][i] if results["distances"] else 1.0
            score = max(0.0, 1.0 - dist)
            hits.append(SeedHit(
                uri=meta.get("uri", ""),
                score=score,
                level=int(meta.get("level", 2)),
                parent_uri=meta.get("parent_uri"),
                context_type=meta.get("context_type", ""),
                category=meta.get("category", ""),
                owner_space=meta.get("owner_space", ""),
                abstract=meta.get("abstract", doc[:200] if doc else ""),
                has_overview=bool(meta.get("has_overview")),
                has_content=bool(meta.get("has_content")),
                active_count=int(meta.get("active_count", 0)),
                updated_at=meta.get("updated_at"),
                metadata=dict(meta),
            ))
        return hits

    def get_embeddings(
        self,
        uris: list[str],
        filters: dict[str, Any],
        level: int = 0,
        *, expected_texts: dict[str, str] | None = None,
    ) -> dict[str, list[float]]:
        if not uris:
            return {}
        where = self._build_where({**filters, "level": level, "uri": uris})
        with self._locked():
            result = self._collection.get(
                where=where if where else None,
                include=["embeddings", "metadatas"] + (["documents"] if expected_texts is not None else []),
            )
        output: dict[str, list[float]] = {}
        embeddings = result.get("embeddings")
        metadatas = result.get("metadatas")
        if embeddings is None:
            embeddings = []
        if metadatas is None:
            metadatas = []
        wanted = set(uris)
        documents = result.get("documents")
        for position, (embedding, metadata) in enumerate(zip(embeddings, metadatas)):
            uri = str((metadata or {}).get("uri", ""))
            if expected_texts is not None and (documents is None or position >= len(documents)
                                               or documents[position] != expected_texts.get(uri)):
                continue
            if uri in wanted:
                output[uri] = [float(value) for value in embedding]
        return output

    def search_children(
        self,
        parent_uri: str,
        query_vector: list[float],
        filters: dict[str, Any],
        top_k: int,
    ) -> list[SeedHit]:
        enhanced = dict(filters)
        enhanced["parent_uri"] = parent_uri
        return self.search_by_vector(query_vector, enhanced, top_k)

    def delete_account_data(self, account_id: str) -> int:
        return self._delete_where({"account_id": account_id})

    def delete_by_owner_space(self, account_id: str, owner_space: str) -> int:
        return self._delete_where(
            {
                "$and": [
                    {"account_id": account_id},
                    {"owner_space": owner_space},
                ]
            }
        )

    def _delete_where(self, where: dict[str, Any]) -> int:
        deleted = 0
        batch_size = self._provider_batch_size()
        with self._locked():
            while True:
                existing = self._collection.get(
                    where=where,
                    limit=batch_size,
                    include=[],
                )
                ids = list(existing.get("ids", []))
                if not ids:
                    return deleted
                self._collection.delete(ids=ids)
                deleted += len(ids)

    # -- Helpers ---------------------------------------------------------------

    @staticmethod
    def _record_to_meta(record: IndexRecord) -> dict[str, Any]:
        meta = {
            "uri": record.uri,
            "level": record.level,
            "account_id": record.filters.get("account_id", ""),
            "owner_space": record.filters.get("owner_space", ""),
        }
        for key in (
            "abstract",
            "context_type",
            "category",
            "parent_uri",
            "has_overview",
            "has_content",
            "active_count",
            "updated_at",
            "when",
            "who",
            "where",
            "routing_key",
            "unit_key",
            "agent_memory_unit_key",
            "unit_revision",
            *CODE_INDEX_METADATA_FIELDS,
        ):
            val = record.metadata.get(key) if record.metadata else None
            if val is not None:
                meta[key] = val
        if isinstance((record.metadata or {}).get("grounding"), list):
            meta["agent_memory_grounding_json"] = json.dumps(record.metadata["grounding"], ensure_ascii=False)
        sanitized: dict[str, str | int | float | bool] = {}
        for k, v in meta.items():
            if isinstance(v, bool):
                sanitized[k] = v
            elif isinstance(v, (int, float, str)):
                sanitized[k] = v
        return sanitized

    @staticmethod
    def _build_where(filters: dict[str, Any]) -> dict | None:
        conditions: list[dict] = []
        for key, expected in filters.items():
            if key == "level":
                if isinstance(expected, list):
                    conditions.append({"level": {"$in": expected}})
                else:
                    conditions.append({"level": expected})
            elif key in ("account_id", "owner_space", "resource_type", "uri"):
                if isinstance(expected, list):
                    conditions.append({key: {"$in": expected}})
                else:
                    conditions.append({key: expected})
            elif key in ("context_type", "category"):
                if isinstance(expected, list):
                    conditions.append({key: {"$in": expected}})
                else:
                    conditions.append({key: expected})
            elif key == "parent_uri":
                conditions.append({key: expected})
        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def _provider_batch_size(self) -> int:
        getter = getattr(self._client, "get_max_batch_size", None)
        if callable(getter):
            provider_limit = int(getter())
        else:
            provider_limit = int(
                getattr(self._client, "max_batch_size", DEFAULT_CHROMA_BATCH_SIZE)
            )
        return max(1, min(DEFAULT_CHROMA_BATCH_SIZE, provider_limit))

    def _stable_vector(self, text: str) -> list[float]:
        import numpy as np
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16)
        rng = np.random.RandomState(seed % (2**31))
        vec = rng.randn(self._dimension).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def count(self, filters: dict[str, Any] | None = None) -> int:
        with self._locked():
            if not filters:
                return self._collection.count()
            result = self._collection.get(where=self._build_where(filters), include=[])
            return len(result.get("ids") or [])

    def get_record(self, record_id: str) -> IndexRecord | None:
        """Fetch one :class:`IndexRecord` by id.

        Returns ``None`` when no row matches. Mirrors the
        :class:`OpenGaussVectorIndex` implementation: ``text`` comes
        back as the full stored document, and ``metadata`` carries
        the per-record scope columns (``extracted_node_id``,
        ``video_id``, ``clip_id``, ``before_clip``).

        Chroma's ``collection.get`` returns documents by id in one
        round-trip; we then rehydrate the ``IndexRecord`` shape so
        callers don't need to know which backend they're on.
        """
        if not record_id:
            return None
        result = self._collection.get(
            ids=[record_id], include=["documents", "metadatas", "embeddings"]
        )
        ids = (result or {}).get("ids") or []
        if not ids:
            return None
        meta = ((result.get("metadatas") or [{}])[0]) or {}
        doc = ((result.get("documents") or [""])[0]) or ""
        emb = ((result.get("embeddings") or [[]])[0]) or []
        # Mirror the OpenGauss path: surface a stable ``filters``
        # shape with account_id / owner_space so callers can re-derive
        # the scope without inspecting raw metadata.
        filters = {
            "account_id": meta.get("account_id", ""),
            "owner_space": meta.get("owner_space", ""),
        }
        metadata = dict(meta)
        if emb and "_embedding" not in metadata:
            metadata["_embedding"] = [float(x) for x in emb]
        return IndexRecord(
            id=str(ids[0]),
            uri=str(meta.get("uri", "")),
            level=int(meta.get("level", 2)),
            text=doc,
            filters=filters,
            metadata=metadata,
        )
