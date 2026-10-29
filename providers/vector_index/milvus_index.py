"""Milvus vector provider. Embeddings are supplied by oGMemory's Embedder.

Tenant and hierarchy constraints run inside Milvus before nearest-neighbor
truncation. Strong reads make outbox acknowledgements observable across workers.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from core.interfaces import VectorIndex
from core.memory_names import memory_directory
from core.models import IndexRecord, SeedHit

_SCOPE_FIELDS = (
    "uri",
    "account_id",
    "owner_space",
    "parent_uri",
    "context_type",
    "category",
    "memory_directory",
    "resource_type",
    "space_id",
    "video_id",
    "extractor_type",
)
_JSON_FILTERS = {"node_status", "extracted_node_id", "clip_id"}
_BATCH_SIZE = 128


class MilvusVectorIndex(VectorIndex):
    """Persistent COSINE/HNSW index using the synchronous MilvusClient API."""

    def __init__(
        self,
        uri: str = "http://localhost:19530",
        token: str | None = None,
        collection_name: str = "contextengine",
        dimension: int = 1024,
        db_name: str = "default",
        timeout: float = 30.0,
    ):
        from pymilvus import DataType, MilvusClient

        if dimension <= 0 or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Milvus dimension and timeout must be positive")
        self._dimension = dimension
        self._timeout = timeout
        self._collection_name = collection_name
        self._client = MilvusClient(uri=uri, token=token or "", db_name=db_name, timeout=timeout)
        try:
            if not self._rpc("has_collection", collection_name=collection_name):
                schema = self._client.create_schema(auto_id=False, enable_dynamic_field=False)
                schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=512)
                schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dimension)
                schema.add_field("text", DataType.VARCHAR, max_length=65535)
                schema.add_field("metadata", DataType.JSON)
                schema.add_field("level", DataType.INT64)
                for field in _SCOPE_FIELDS:
                    schema.add_field(field, DataType.VARCHAR, max_length=8192)
                self._create_if_absent(
                    "create_collection",
                    collection_name=collection_name,
                    schema=schema,
                    consistency_level="Strong",
                )
            self._validate_schema(DataType)
            self._ensure_index()
            self._rpc("load_collection", collection_name=collection_name)
        except Exception:
            self._client.close()
            raise

    def _rpc(self, method: str, **kwargs):
        """Pass a deadline to every network operation, including iterator creation."""
        return getattr(self._client, method)(timeout=self._timeout, **kwargs)

    def _create_if_absent(self, method: str, **kwargs) -> None:
        from pymilvus import MilvusException

        try:
            self._rpc(method, **kwargs)
        except MilvusException as exc:
            # Only a concurrent duplicate is safe to verify and reuse. Preserve
            # permissions, transport and resource errors rather than masking them.
            if not re.search(r"already\s+exist", str(exc), re.IGNORECASE):
                raise

    def _validate_schema(self, data_type) -> None:
        schema = self._rpc("describe_collection", collection_name=self._collection_name)
        fields = {field["name"]: field for field in schema["fields"]}
        expected = {
            "id": data_type.VARCHAR,
            "embedding": data_type.FLOAT_VECTOR,
            "text": data_type.VARCHAR,
            "metadata": data_type.JSON,
            "level": data_type.INT64,
            **dict.fromkeys(_SCOPE_FIELDS, data_type.VARCHAR),
        }
        if (
            schema.get("auto_id")
            or not fields.get("id", {}).get("is_primary")
            or any(fields.get(key, {}).get("type") != dtype for key, dtype in expected.items())
            or int(fields.get("embedding", {}).get("params", {}).get("dim", 0)) != self._dimension
        ):
            raise ValueError(
                "Existing Milvus collection has an incompatible oGMemory schema/dimension"
            )

    def _embedding_index(self) -> dict[str, Any] | None:
        for name in self._rpc("list_indexes", collection_name=self._collection_name):
            info = self._rpc(
                "describe_index", collection_name=self._collection_name, index_name=name
            )
            if info.get("field_name") == "embedding":
                return dict(info)
        return None

    def _ensure_index(self) -> None:
        # A previous startup may have created the schema but failed before its
        # index was created. Retrying startup must finish that initialization.
        info = self._embedding_index()
        if info is None:
            indexes = self._client.prepare_index_params()
            indexes.add_index(
                field_name="embedding",
                index_name="embedding",
                index_type="HNSW",
                metric_type="COSINE",
                params={"M": 16, "efConstruction": 200},
            )
            self._create_if_absent(
                "create_index", collection_name=self._collection_name, index_params=indexes
            )
            info = self._embedding_index()
        if info is None or info.get("metric_type") != "COSINE":
            raise ValueError("Existing Milvus collection must have a COSINE embedding index")

    def _vector(self, values) -> list[float]:
        if values is None or len(values) != self._dimension:
            raise ValueError(f"Milvus requires an embedding of dimension {self._dimension}")
        vector = [float(value) for value in values]
        if not all(math.isfinite(value) for value in vector) or not any(vector):
            raise ValueError("Milvus COSINE embeddings must be finite and non-zero")
        return vector

    @staticmethod
    def _build_filter(filters: dict[str, Any]) -> str:
        clauses = []
        for key, value in filters.items():
            if key in _SCOPE_FIELDS or key == "level":
                field = key
            elif key in _JSON_FILTERS:
                field = f'metadata["{key}"]'
            elif key == "before_clip":
                clauses.append('metadata["before_clip"] <= ' + json.dumps(value, allow_nan=False))
                continue
            elif key == "before_sec":
                from core.temporal import seconds

                bound = json.dumps(seconds(value), allow_nan=False)
                clauses.extend(
                    f'metadata["{name}"] <= {bound}'
                    for name in ("source_start_sec", "source_duration_sec", "source_end_sec")
                )
                continue
            else:
                # Never silently discard a scope constraint.
                raise ValueError(f"Unsupported Milvus filter: {key}")
            encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
            clauses.append(
                f"{field} {'in' if isinstance(value, list) else '=='} {encoded}"
                if value != []
                else "id in []"
            )
        return " and ".join(f"({clause})" for clause in clauses)

    def upsert(self, records: list[IndexRecord]) -> None:
        rows = []
        for record in records:
            if not record.filters.get("account_id") or not record.filters.get("owner_space"):
                raise ValueError("Milvus records require account_id and owner_space")
            metadata = {
                "abstract": record.text[:200],
                **{key: value for key, value in record.metadata.items() if key != "_embedding"},
            }
            row: dict[str, Any] = {
                key: str(record.filters.get(key, metadata.get(key, "")) or "")
                for key in _SCOPE_FIELDS
            }
            row.update(
                id=record.id,
                uri=record.uri,
                level=record.level,
                text=record.text,
                memory_directory=memory_directory(record.uri),
                metadata=metadata,
                embedding=self._vector(record.metadata.get("_embedding")),
            )
            # Validate locally before any batch is written.
            for name, limit in {
                "id": 512,
                "text": 65535,
                **dict.fromkeys(_SCOPE_FIELDS, 8192),
            }.items():
                if len(row[name].encode("utf-8")) > limit:
                    raise ValueError(f"Milvus {name} exceeds {limit} UTF-8 bytes")
            if (
                len(json.dumps(metadata, ensure_ascii=False, allow_nan=False).encode("utf-8"))
                > 65536
            ):
                raise ValueError("Milvus metadata exceeds 65536 UTF-8 bytes")
            json.dumps(row, allow_nan=False)
            rows.append(row)
        for start in range(0, len(rows), _BATCH_SIZE):
            self._rpc(
                "upsert",
                collection_name=self._collection_name,
                data=rows[start : start + _BATCH_SIZE],
            )

    def delete(self, ids: list[str]) -> None:
        for start in range(0, len(ids), _BATCH_SIZE):
            self._rpc(
                "delete",
                collection_name=self._collection_name,
                ids=ids[start : start + _BATCH_SIZE],
            )

    def search_by_vector(
        self, query_vector: list[float], filters: dict[str, Any], top_k: int
    ) -> list[SeedHit]:
        if top_k <= 0:
            return []
        if top_k > 16384:
            raise ValueError("Milvus top_k must not exceed 16384")
        result = self._rpc(
            "search",
            collection_name=self._collection_name,
            data=[self._vector(query_vector)],
            anns_field="embedding",
            filter=self._build_filter(filters),
            limit=top_k,
            output_fields=["metadata", "level", *_SCOPE_FIELDS],
            search_params={"metric_type": "COSINE", "params": {"ef": max(64, top_k)}},
            consistency_level="Strong",
        )
        hits = []
        for match in result[0]:
            row = match["entity"]
            meta = {
                **row.get("metadata", {}),
                **{key: row.get(key, "") for key in _SCOPE_FIELDS},
                "level": row["level"],
            }
            hits.append(
                SeedHit(
                    uri=row["uri"],
                    score=max(0.0, min(1.0, float(match["distance"]))),
                    level=int(row["level"]),
                    parent_uri=row.get("parent_uri") or None,
                    context_type=row.get("context_type", ""),
                    category=row.get("category", ""),
                    owner_space=row.get("owner_space", ""),
                    abstract=meta.get("abstract", row.get("text", "")[:200]),
                    has_overview=bool(meta.get("has_overview")),
                    has_content=bool(meta.get("has_content")),
                    active_count=int(meta.get("active_count", 0)),
                    updated_at=meta.get("updated_at"),
                    metadata=meta,
                )
            )
        return hits

    def search_children(
        self, parent_uri: str, query_vector: list[float], filters: dict[str, Any], top_k: int
    ) -> list[SeedHit]:
        return self.search_by_vector(query_vector, {**filters, "parent_uri": parent_uri}, top_k)

    def _query_rows(self, filters: dict[str, Any], output_fields: list[str]):
        iterator = self._rpc(
            "query_iterator",
            collection_name=self._collection_name,
            filter=self._build_filter(filters),
            output_fields=output_fields,
            batch_size=1000,
            consistency_level="Strong",
        )
        try:
            while True:
                batch = iterator.next()
                if not batch:
                    break
                yield from batch
        finally:
            iterator.close()

    def get_embeddings(
        self,
        uris: list[str],
        filters: dict[str, Any],
        level: int = 0,
        *,
        expected_texts: dict[str, str] | None = None,
    ) -> dict[str, list[float]]:
        if not uris:
            return {}
        return {
            row["uri"]: [float(value) for value in row["embedding"]]
            for row in self._query_rows(
                {**filters, "uri": uris, "level": level}, ["uri", "embedding", "text"]
            )
            if expected_texts is None or row["text"] == expected_texts.get(row["uri"])
        }

    def get_record(self, record_id: str) -> IndexRecord | None:
        if not record_id:
            return None
        rows = self._rpc(
            "get",
            collection_name=self._collection_name,
            ids=[record_id],
            output_fields=["*"],
            consistency_level="Strong",
        )
        if not rows:
            return None
        row = rows[0]
        return IndexRecord(
            id=row["id"],
            uri=row["uri"],
            level=row["level"],
            text=row["text"],
            filters={key: row[key] for key in ("account_id", "owner_space")},
            metadata={**row["metadata"], "_embedding": list(row["embedding"])},
        )

    def count(self, filters: dict[str, Any] | None = None) -> int:
        rows = self._rpc(
            "query",
            collection_name=self._collection_name,
            filter=self._build_filter(filters or {}),
            output_fields=["count(*)"],
            consistency_level="Strong",
        )
        return int(rows[0]["count(*)"])

    def _delete_where(self, filters: dict[str, Any]) -> int:
        # Count the visible snapshot for the protocol's return value. Some Milvus
        # versions do not report an accurate logical-row count for expression deletes.
        count = self.count(filters)
        if count:
            self._rpc(
                "delete", collection_name=self._collection_name, filter=self._build_filter(filters)
            )
        return count

    def delete_account_data(self, account_id: str) -> int:
        return self._delete_where({"account_id": account_id})

    def delete_by_owner_space(self, account_id: str, owner_space: str) -> int:
        return self._delete_where({"account_id": account_id, "owner_space": owner_space})

    def close(self) -> None:
        self._client.close()
