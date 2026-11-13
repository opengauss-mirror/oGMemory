"""Milvus vector provider. Embeddings are supplied by oGMemory's Embedder.

Tenant and hierarchy constraints run inside Milvus before nearest-neighbor
truncation. Strong reads make outbox acknowledgements observable across workers.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from array import array
from itertools import islice
from typing import Any

from core.interfaces import VectorIndex
from core.memory_names import memory_directory
from core.models import IndexRecord, SeedHit
from core.projection_ownership import check_projection_owner

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
    "extractor_type",
)
_JSON_FILTERS = {"node_status", "extracted_node_id"}
_BATCH_SIZE = 32
_URI_BATCH_SIZE = 128
_QUERY_PAGE_SIZE = 256
_KEY_PAGE_SIZE = 1024
_QUERY_RESPONSE_BUDGET = 48 * 1024 * 1024
_TEXT_FIELDS = ("text", "text_1", "text_2", "text_3", "text_4")
_VARCHAR_LIMITS = {"id": 512, "text": 65535, **dict.fromkeys(_SCOPE_FIELDS, 8192)}


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
                schema.add_field(
                    "id", DataType.VARCHAR, is_primary=True, max_length=_VARCHAR_LIMITS["id"]
                )
                schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dimension)
                schema.add_field("text", DataType.VARCHAR, max_length=_VARCHAR_LIMITS["text"])
                schema.add_field("logical_id", DataType.VARCHAR, max_length=512, nullable=True)
                for name in _TEXT_FIELDS[1:]:
                    schema.add_field(name, DataType.VARCHAR, max_length=65535, nullable=True)
                schema.add_field("metadata", DataType.JSON)
                schema.add_field("level", DataType.INT64)
                for field in _SCOPE_FIELDS:
                    schema.add_field(field, DataType.VARCHAR, max_length=_VARCHAR_LIMITS[field])
                self._create_if_absent(
                    "create_collection",
                    collection_name=collection_name,
                    schema=schema,
                    consistency_level="Strong",
                )
            self._validate_schema(DataType)
            self._ensure_text_fields(DataType)
            self._ensure_index()
            self._ensure_scalar_indexes()
            self._rpc("load_collection", collection_name=collection_name)
        except Exception:
            self._client.close()
            raise

    def _rpc(self, method: str, **kwargs):
        """Pass a deadline to every network operation, including schema and index changes."""
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
            or any(
                int(fields.get(name, {}).get("params", {}).get("max_length", 0)) < limit
                for name, limit in _VARCHAR_LIMITS.items()
            )
        ):
            raise ValueError(
                "Existing Milvus collection has an incompatible oGMemory schema/dimension"
            )

    def _ensure_text_fields(self, data_type) -> None:
        # Nullable additive fields preserve collections created before long-text
        # support. Old rows read as empty suffixes; no data rewrite is required.
        schema = self._rpc("describe_collection", collection_name=self._collection_name)
        fields = {field["name"]: field for field in schema["fields"]}
        for name, limit in [("logical_id", 512), *((key, 65535) for key in _TEXT_FIELDS[1:])]:
            if name not in fields:
                self._create_if_absent(
                    "add_collection_field", collection_name=self._collection_name,
                    field_name=name, data_type=data_type.VARCHAR,
                    max_length=limit, nullable=True,
                )
            elif (fields[name]["type"] != data_type.VARCHAR
                  or int(fields[name].get("params", {}).get("max_length", 0)) < limit):
                raise ValueError("Existing Milvus collection has incompatible text suffix fields")

    @staticmethod
    def _text_parts(text: str) -> dict[str, str]:
        encoded = text.encode("utf-8")
        if len(encoded) > 65535 * len(_TEXT_FIELDS):
            raise ValueError("Milvus text exceeds 327675 UTF-8 bytes")
        parts = {}
        for name in _TEXT_FIELDS:
            # Only drop the incomplete codepoint at this boundary. Advance by
            # the bytes actually consumed so it is retained in the next part.
            part = encoded[:65535].decode("utf-8", errors="ignore")
            parts[name] = part
            encoded = encoded[len(part.encode("utf-8")):]
        if encoded:
            raise ValueError("Milvus text exceeds available UTF-8 chunk capacity")
        return parts

    @staticmethod
    def _text(row: dict) -> str:
        return "".join(row.get(name) or "" for name in _TEXT_FIELDS)

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

    def _ensure_scalar_indexes(self) -> None:
        indexed = set()
        for name in self._rpc("list_indexes", collection_name=self._collection_name):
            info = self._rpc("describe_index", collection_name=self._collection_name, index_name=name)
            indexed.add(info.get("field_name"))
        # Preserve existing scalar indexes regardless of their implementation.
        for field in ("account_id", "owner_space", "level", "parent_uri", "uri", "logical_id"):
            if field not in indexed:
                params = self._client.prepare_index_params()
                params.add_index(field_name=field, index_name=f"scope_{field}", index_type="AUTOINDEX")
                self._create_if_absent(
                    "create_index", collection_name=self._collection_name, index_params=params,
                )

    def _vector(self, values) -> list[float]:
        if values is None or len(values) != self._dimension:
            raise ValueError(f"Milvus requires an embedding of dimension {self._dimension}")
        # Validate what Milvus actually stores, not just the Python FLOAT64 input.
        # Finite values can overflow to infinity or all underflow to zero in FLOAT32.
        try:
            vector = array("f", (float(value) for value in values))
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError("Milvus COSINE embeddings must be finite and non-zero") from exc
        if not all(math.isfinite(value) for value in vector) or not any(vector):
            raise ValueError("Milvus COSINE embeddings must be finite and non-zero")
        # Milvus calculates norms in FLOAT32. Finite components alone do not
        # prevent squared-norm overflow or underflow (including flushed subnormals).
        squared_norm = math.fsum(value * value for value in vector)
        if not 2.0**-126 <= squared_norm <= (2.0 - 2.0**-23) * 2.0**127:
            raise ValueError("Milvus COSINE embedding norm is outside the stable FLOAT32 range")
        return vector.tolist()

    @staticmethod
    def _build_filter(filters: dict[str, Any]) -> str:
        clauses = []
        for key, value in filters.items():
            if key in _SCOPE_FIELDS or key == "level":
                field = key
            elif key in _JSON_FILTERS:
                field = f'metadata["{key}"]'
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
        rows = {}
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
                **self._text_parts(record.text),
                memory_directory=memory_directory(record.uri),
                metadata=metadata,
                embedding=self._vector(record.metadata.get("_embedding")),
            )
            # Validate locally before any batch is written.
            for name, limit in _VARCHAR_LIMITS.items():
                if len(row[name].encode("utf-8")) > limit:
                    raise ValueError(f"Milvus {name} exceeds {limit} UTF-8 bytes")
            if (
                len(json.dumps(metadata, ensure_ascii=False, allow_nan=False).encode("utf-8"))
                > 65536
            ):
                raise ValueError("Milvus metadata exceeds 65536 UTF-8 bytes")
            json.dumps(row, allow_nan=False)
            rows[record.id] = row
        self._publish(list(rows.values()))

    def _epoch(self):
        epoch = self._client._get_connection().alloc_timestamp(timeout=self._timeout)
        if not epoch:
            raise RuntimeError("Milvus must support timestamp allocation for projection writes")
        check_projection_owner()
        return epoch

    def _publish(self, rows: list[dict], *, deleted: bool = False) -> None:
        if not rows:
            return
        epoch = self._epoch()
        for row in rows:
            logical_id = row["id"]
            digest = hashlib.sha256(logical_id.encode()).hexdigest()
            row["logical_id"] = logical_id
            row["id"] = f"ogv_{epoch:020d}_{int(deleted)}_{digest}"
        for start in range(0, len(rows), _BATCH_SIZE):
            self._rpc(
                "upsert",
                collection_name=self._collection_name,
                data=rows[start : start + _BATCH_SIZE],
            )
        self._prune_versions(rows)

    def _prune_versions(self, writes):
        # Keep the maximum generation, including a deletion marker. A worker
        # racing this cleanup can only remove immutable, superseded identities.
        for start in range(0, len(writes), _URI_BATCH_SIZE):
            batch = writes[start : start + _URI_BATCH_SIZE]
            expr = self._version_filter([row["logical_id"] for row in batch])
            accounts = {row["account_id"] for row in batch if row["account_id"]}
            if accounts:
                expr = f"({expr}) or ({self._marker_filter(accounts)})"
            cutoffs = {}
            latest = {}
            obsolete = []
            for row in self._key_rows(expr, ["logical_id", "level", "account_id", "owner_space"]):
                if row["level"] == -2:
                    self._add_marker(cutoffs, row)
                    continue
                key = row.get("logical_id") or row["id"]
                rank = row["id"] if row.get("logical_id") else ""
                current = rank, row["id"]
                if key in latest:
                    loser, current = sorted((latest[key], current))
                    obsolete.append(loser[1])
                latest[key] = current
            retired = [row for row in batch if self._is_live(row) and not self._current_in_scope(row, cutoffs)]
            for offset in range(0, len(retired), _BATCH_SIZE):
                self._retire_payloads(retired[offset : offset + _BATCH_SIZE])
            for offset in range(0, len(obsolete), _BATCH_SIZE):
                self._rpc(
                    "delete",
                    collection_name=self._collection_name,
                    ids=obsolete[offset : offset + _BATCH_SIZE],
                )

    @staticmethod
    def _version_filter(keys):
        literals = json.dumps(keys, ensure_ascii=False)
        return f"logical_id in {literals} or (logical_id is null and id in {literals})"

    @staticmethod
    def _is_live(row):
        return not row.get("logical_id") or row["id"].split("_", 3)[2] == "0"

    @staticmethod
    def _marker_filter(accounts):
        expression = 'level == -2 and id like "ogscope_%"'
        if accounts:
            expression += " and account_id in " + json.dumps(sorted(set(accounts)))
        return expression

    @staticmethod
    def _add_marker(cutoffs, row):
        scope = row["account_id"], row["owner_space"]
        epoch = row["id"].split("_", 2)[1]
        cutoffs[scope] = max(epoch, cutoffs.get(scope, ""))

    @staticmethod
    def _current_in_scope(row, cutoffs):
        account = row["account_id"]
        bound = max(cutoffs.get((account, ""), ""), cutoffs.get((account, row["owner_space"]), ""))
        return not bound or (row.get("logical_id") and row["id"].split("_", 2)[1] > bound)

    def _heads(self, ids):
        heads = {}
        keys = list(dict.fromkeys(ids))
        for start in range(0, len(keys), _URI_BATCH_SIZE):
            expression = self._version_filter(keys[start : start + _URI_BATCH_SIZE])
            for row in self._key_rows(
                expression, ["logical_id", "account_id", "owner_space", "level"]
            ):
                key = row.get("logical_id") or row["id"]
                rank = row["id"] if row.get("logical_id") else ""
                if key not in heads or rank > heads[key][0]:
                    heads[key] = (rank, row)
        live = {
            key: row for key, (_, row) in heads.items() if row["level"] != -2 and self._is_live(row)
        }
        cutoffs = {}
        if live:
            expression = self._marker_filter([row["account_id"] for row in live.values()])
            for row in self._key_rows(expression, ["account_id", "owner_space"]):
                self._add_marker(cutoffs, row)
        return {key: row["id"] for key, row in live.items() if self._current_in_scope(row, cutoffs)}

    def _marker(self, key, level):
        return {
            **dict.fromkeys(_SCOPE_FIELDS, ""),
            **dict.fromkeys(_TEXT_FIELDS, ""),
            "id": key,
            "level": level,
            "metadata": {},
            "embedding": [1.0] + [0.0] * (self._dimension - 1),
        }

    def delete(self, ids: list[str]) -> None:
        self._publish([self._marker(key, -1) for key in dict.fromkeys(ids)], deleted=True)

    def search_by_vector(
        self, query_vector: list[float], filters: dict[str, Any], top_k: int
    ) -> list[SeedHit]:
        if top_k <= 0:
            return []
        if top_k > 16384:
            raise ValueError("Milvus top_k must not exceed 16384")
        requested = self._build_filter(filters)
        expression = f"({requested}) and level >= 0" if requested else "level >= 0"
        excluded = set()
        scope = expression
        vector = self._vector(query_vector)
        while True:
            result = self._rpc(
                "search",
                collection_name=self._collection_name,
                data=[vector],
                anns_field="embedding",
                filter=expression,
                limit=top_k,
                output_fields=["logical_id", "metadata", "level", *_SCOPE_FIELDS],
                search_params={"metric_type": "COSINE", "params": {"ef": max(64, top_k)}},
                consistency_level="Strong",
            )
            matches = result[0]
            keys = [m["entity"].get("logical_id") or m["id"] for m in matches]
            heads = self._heads(keys)
            current = [m for key, m in zip(keys, matches) if heads.get(key) == m["id"]]
            if len(current) == len(matches) or len(matches) < top_k:
                break
            excluded.update(m["id"] for key, m in zip(keys, matches) if heads.get(key) != m["id"])
            omit = "id not in " + json.dumps(sorted(excluded))
            expression = f"({scope}) and ({omit})" if scope else omit
        hits = []
        for match in current:
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

    def _query_rows(
        self, filters: dict[str, Any], output_fields: list[str], *, page_size: int = _QUERY_PAGE_SIZE
    ):
        # The server iterator flag orders by primary key. Use bounded keyset
        # pages directly: pymilvus 2.6.17 QueryIterator performs an unbounded
        # describe_collection during construction, even with a timeout.
        yield from self._pages(self._build_filter(filters), output_fields, page_size)

    def _pages(self, expression, output_fields, page_size=_QUERY_PAGE_SIZE):
        cursor = None
        snapshot = {}
        while True:
            page_filter = expression
            if cursor is not None:
                bound = "id > " + json.dumps(cursor, ensure_ascii=False)
                page_filter = f"({expression}) and ({bound})" if expression else bound
            batch = self._rpc(
                "query", collection_name=self._collection_name, filter=page_filter,
                output_fields=list(dict.fromkeys(["id", *output_fields])),
                limit=page_size, iterator="True", consistency_level="Strong",
                **snapshot,
            )
            if not batch:
                break
            ids = [row["id"] for row in batch]
            if ids != sorted(ids) or (cursor is not None and ids[0] <= cursor):
                raise RuntimeError("Milvus returned an unordered iterator page")
            cursor = ids[-1]
            timestamp = getattr(batch, "extra", {}).get("iterator_session_ts")
            if timestamp:
                snapshot["guarantee_timestamp"] = timestamp
            yield from batch
            if len(batch) < page_size:
                break

    def _key_rows(self, expression, fields):
        output = list(dict.fromkeys(["id", *fields]))
        batch = self._rpc("query", collection_name=self._collection_name, filter=expression,
                          output_fields=output, limit=_KEY_PAGE_SIZE, consistency_level="Strong")
        if len(batch) < _KEY_PAGE_SIZE:
            yield from batch
        else:
            yield from self._pages(expression, fields, _KEY_PAGE_SIZE)

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
        output = {}
        wanted = list(dict.fromkeys(uris))
        fields = ["uri", "embedding"]
        if expected_texts is not None:
            fields.extend(_TEXT_FIELDS)
        # Budget the worst-case row, including all text suffixes and vectors.
        # Scene checks expected_texts too; it should not inherit 32-row write batches.
        row_bytes = (self._dimension * 4 + 2 * _VARCHAR_LIMITS["id"] + _VARCHAR_LIMITS["uri"]
                     + _VARCHAR_LIMITS["account_id"] + _VARCHAR_LIMITS["owner_space"] + 1024)
        if expected_texts is not None:
            row_bytes += len(_TEXT_FIELDS) * _VARCHAR_LIMITS["text"]
        page_size = min(_QUERY_PAGE_SIZE, max(1, _QUERY_RESPONSE_BUDGET // row_bytes))
        for start in range(0, len(wanted), _URI_BATCH_SIZE):
            batch = wanted[start : start + _URI_BATCH_SIZE]
            scope = {**filters, "uri": batch, "level": level}
            if filters.keys() <= {"account_id", "owner_space"}:
                rows = self._embedding_rows(batch, scope, fields, page_size)
            else:
                rows = self._visible_rows(scope, fields, page_size)
            for row in rows:
                if expected_texts is None or self._text(row) == expected_texts.get(row["uri"]):
                    output[row["uri"]] = [float(value) for value in row["embedding"]]
        return output

    def _embedding_rows(self, uris, scope, fields, page_size):
        # Include the complete history of canonical URI/level IDs in the same
        # bounded query. Sibling records with custom IDs resolve independently.
        expected_ids = {IndexRecord.generate_id(uri, scope["level"]) for uri in uris}
        expression = (
            f"({self._build_filter(scope)}) or ({self._version_filter(sorted(expected_ids))})"
        )
        account = scope.get("account_id", [])
        markers = self._marker_filter(account if isinstance(account, list) else [account])
        newest, cutoffs = {}, {}
        rows = self._pages(
            f"({expression}) or ({markers})",
            ["logical_id", "account_id", "owner_space", *scope, *fields],
            page_size,
        )
        while batch := list(islice(rows, page_size)):
            siblings = []
            for row in batch:
                if row["level"] == -2:
                    self._add_marker(cutoffs, row)
                    continue
                key = row.get("logical_id") or row["id"]
                if key not in expected_ids:
                    siblings.append(row)
                    continue
                rank = row["id"] if row.get("logical_id") else ""
                if key not in newest or rank > newest[key][0]:
                    newest[key] = rank, row
            yield from self._visible_batch(siblings)
        for _, row in newest.values():
            if (
                self._is_live(row)
                and self._current_in_scope(row, cutoffs)
                and all(
                    row.get(field) in value if isinstance(value, list) else row.get(field) == value
                    for field, value in scope.items()
                )
            ):
                yield row

    def _visible_rows(self, filters, fields, page_size=_QUERY_PAGE_SIZE):
        rows = self._query_rows(filters, ["logical_id", *fields], page_size=page_size)
        while batch := list(islice(rows, page_size)):
            yield from self._visible_batch(batch)

    def _visible_batch(self, batch):
        keys = [row.get("logical_id") or row["id"] for row in batch]
        heads = self._heads(keys)
        for key, row in zip(keys, batch):
            if heads.get(key) == row["id"]:
                yield row

    def get_record(self, record_id: str) -> IndexRecord | None:
        if not record_id:
            return None
        deadline = time.monotonic() + self._timeout
        while True:
            physical = self._heads([record_id]).get(record_id)
            if physical is None:
                return None
            rows = self._rpc(
                "get",
                collection_name=self._collection_name,
                ids=[physical],
                output_fields=["*"],
                consistency_level="Strong",
            )
            if rows:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Milvus record changed throughout the lookup")
        row = rows[0]
        scope = {key: row.get(key, "") for key in _SCOPE_FIELDS}
        return IndexRecord(
            id=record_id,
            uri=row["uri"],
            level=row["level"],
            text=self._text(row),
            filters={
                key: value
                for key, value in scope.items()
                if key not in {"uri", "memory_directory"}
                and (value or key in {"account_id", "owner_space"})
            },
            metadata={
                # Restore scalar scope in filters only. Copying it into JSON
                # would grow a valid near-limit payload during read/rewrite.
                **row["metadata"],
                "_embedding": [float(value) for value in row["embedding"]],
            },
        )

    def probe(self, account_id: str) -> None:
        self._rpc("query", collection_name=self._collection_name,
                  filter=self._build_filter({"account_id": account_id}),
                  output_fields=["id"], limit=1, consistency_level="Strong")

    def count(self, filters: dict[str, Any] | None = None) -> int:
        rows = self._key_rows(self._build_filter(filters or {}), ["logical_id"])
        total = 0
        while batch := list(islice(rows, _KEY_PAGE_SIZE)):
            total += sum(1 for _ in self._visible_batch(batch))
        return total

    def _delete_where(self, filters: dict[str, Any]) -> int:
        count = self.count(filters)
        epoch = self._epoch()
        digest = hashlib.sha256(json.dumps(filters, sort_keys=True).encode()).hexdigest()
        marker = self._marker(f"ogscope_{epoch:020d}_{digest}", -2)
        marker.update(filters)
        self._rpc("upsert", collection_name=self._collection_name, data=[marker])
        scope = self._build_filter(filters)
        prefix = f"ogv_{epoch:020d}_"
        retiring = f'({scope}) and level >= 0 and (logical_id is null or id < "{prefix}")'
        rows = self._key_rows(retiring, ["logical_id"])
        while batch := list(islice(rows, _BATCH_SIZE)):
            self._retire_payloads(batch)
        return count

    def _retire_payloads(self, batch):
        tombstones = []
        for row in batch:
            key = row.get("logical_id") or row["id"]
            version = row["id"].split("_", 2)[1] if row.get("logical_id") else "0" * 20
            digest = hashlib.sha256(key.encode()).hexdigest()
            tombstone = self._marker(f"ogv_{version}_1_{digest}", -1)
            tombstone["logical_id"] = key
            tombstones.append(tombstone)
        self._rpc("upsert", collection_name=self._collection_name, data=tombstones)
        self._rpc("delete", collection_name=self._collection_name, ids=[row["id"] for row in batch])

    def delete_account_data(self, account_id: str) -> int:
        return self._delete_where({"account_id": account_id})

    def delete_by_owner_space(self, account_id: str, owner_space: str) -> int:
        return self._delete_where({"account_id": account_id, "owner_space": owner_space})

    def close(self) -> None:
        self._client.close()
