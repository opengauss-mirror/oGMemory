"""In-Memory Vector Index for testing.

Simple in-memory implementation of VectorIndex protocol.
Supports filter matching for cross-tenant isolation testing.

Also implements ``search_by_vector`` / ``search_children``
for the retrieval pipeline.
"""

import math
from dataclasses import replace
from typing import Any

from core.clip_lookup import clip_ordinal, validate_clip_lookup
from core.interfaces import VectorIndex
from core.memory_names import memory_directory
from core.models import IndexRecord, SeedHit
from core.temporal import contains_time, seconds, within_scope
from core.transcript_lexical import indexed_metadata, transcript_score, validate_transcript_request


class InMemoryVectorIndex(VectorIndex):
    """In-memory vector index for testing.

    Stores records in a dict and performs cosine similarity search.
    Filters are applied to enforce tenant isolation.

    This is a TEST implementation using simple cosine similarity.
    Production should use a proper vector database (Qdrant, Milvus, etc.).
    """

    def __init__(self, dimension: int = 384):
        """Initialize an empty in-memory index.

        Args:
            dimension: Embedding dimension for new records
        """
        self._dimension = dimension
        self._records: dict[str, IndexRecord] = {}
        self._vectors: dict[str, list[float]] = {}

    def upsert(self, records: list[IndexRecord]) -> None:
        """Add or update records in the index.

        Args:
            records: List of IndexRecords to upsert

        Note:
            Idempotent: duplicate IndexRecord.id overwrites existing record.
        """
        for record in records:
            record = replace(record, metadata=indexed_metadata(record.metadata))
            self._records[record.id] = record
            # Use real embedding if provided in metadata, otherwise fall back to mock
            real_embedding = record.metadata.get("_embedding")
            if real_embedding:
                self._vectors[record.id] = real_embedding
            else:
                self._vectors[record.id] = self._mock_vector(record.text)

    def delete(self, ids: list[str]) -> None:
        """Remove records from the index.

        Args:
            ids: List of record IDs to delete

        Note:
            No-op for non-existent IDs.
        """
        for record_id in ids:
            self._records.pop(record_id, None)
            self._vectors.pop(record_id, None)

    def _mock_vector(self, text: str) -> list[float]:
        """Generate a mock embedding vector for testing.

        Uses a hash-based approach for deterministic but varied vectors.
        Real implementation should use an actual Embedder.

        Args:
            text: Text to embed

        Returns:
            Mock vector of self._dimension dimensions
        """
        import hashlib

        # Create a deterministic seed from text
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16)

        # Generate pseudo-random vector using simple LCG
        vector = []
        for i in range(self._dimension):
            # Linear congruential generator
            seed = (1103515245 * seed + 12345) & 0x7fffffff
            # Map to [-1, 1] range
            value = (seed / 0x7fffffff) * 2 - 1
            vector.append(value)

        return vector

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            a: First vector
            b: Second vector

        Returns:
            Similarity score in [0, 1] range
        """
        if len(a) != len(b):
            raise ValueError("Vectors must have same dimension")

        dot_product = sum(x * y for x, y in zip(a, b))
        magnitude_a = math.sqrt(sum(x * x for x in a))
        magnitude_b = math.sqrt(sum(y * y for y in b))

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (magnitude_a * magnitude_b)

    # -- Retrieval-chain extensions ----------------------------------------

    def search_by_vector(
        self,
        query_vector: list[float],
        filters: dict[str, Any],
        top_k: int,
    ) -> list[SeedHit]:
        hits: list[SeedHit] = []
        for record_id, record in self._records.items():
            if not self._matches_raw_filters(record, filters):
                continue
            similarity = self._cosine_similarity(query_vector, self._vectors[record_id])
            hits.append(self._record_to_vector_hit(record, similarity))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def search_by_transcript(self, query: str, filters: dict[str, Any], top_k: int) -> list[SeedHit]:
        """Independent ASR recall; never uses vectors or full description text."""
        if not validate_transcript_request(query, filters, top_k):
            return []
        hits = []
        for record in self._records.values():
            if not self._matches_raw_filters(record, filters):
                continue
            score = transcript_score(query, record.metadata)
            if score > 0:
                hits.append(self._record_to_vector_hit(record, score))
        hits.sort(key=lambda hit: (-hit.score, hit.uri))
        return hits[:top_k]

    def get_embeddings(
        self,
        uris: list[str],
        filters: dict[str, Any],
        level: int = 0,
        *, expected_texts: dict[str, str] | None = None,
    ) -> dict[str, list[float]]:
        wanted = set(uris)
        output: dict[str, list[float]] = {}
        for record_id, record in self._records.items():
            if record.uri not in wanted or record.level != level:
                continue
            if expected_texts is not None and record.text != expected_texts.get(record.uri):
                continue
            if self._matches_raw_filters(record, filters):
                output[record.uri] = list(self._vectors[record_id])
        return output

    def search_children(
        self,
        parent_uri: str,
        query_vector: list[float],
        filters: dict[str, Any],
        top_k: int,
    ) -> list[SeedHit]:
        # Ensure parent_uri has trailing slash for consistent matching
        normalized_parent = parent_uri if parent_uri.endswith("/") else parent_uri + "/"

        hits: list[SeedHit] = []
        for record_id, record in self._records.items():
            if not record.uri.startswith(normalized_parent):
                continue
            # Check depth: count slashes after parent to ensure direct children
            remaining = record.uri[len(normalized_parent):]
            if "/" in remaining:
                continue  # Not a direct child
            if not self._matches_raw_filters(record, filters):
                continue
            similarity = self._cosine_similarity(query_vector, self._vectors[record_id])
            hits.append(self._record_to_vector_hit(record, similarity))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def _matches_raw_filters(self, record: IndexRecord, filters: dict[str, Any]) -> bool:
        if not within_scope(
            {**record.filters, **record.metadata},
            filters.get("video_id"), filters.get("before_clip"), filters.get("before_sec"),
        ):
            return False
        for key, expected in filters.items():
            if key == "memory_directory":
                if memory_directory(record.uri) not in expected:
                    return False
            elif key == "level":
                if isinstance(expected, list):
                    if record.level not in expected:
                        return False
                elif record.level != expected:
                    return False
            elif key in ("account_id", "owner_space", "extractor_type", "resource_type"):
                actual = record.filters.get(key)
                if isinstance(expected, list):
                    if actual not in expected:
                        return False
                elif actual != expected:
                    return False
            elif key in ("context_type", "category"):
                actual = record.metadata.get(key)
                if isinstance(expected, list):
                    if actual not in expected:
                        return False
                elif actual != expected:
                    return False
        return True

    @staticmethod
    def _record_to_vector_hit(record: IndexRecord, score: float) -> SeedHit:
        meta = record.metadata or {}
        return SeedHit(
            metadata=dict(meta),
            uri=record.uri,
            score=score,
            level=record.level,
            parent_uri=meta.get("parent_uri"),
            context_type=meta.get("context_type", ""),
            category=meta.get("category", ""),
            owner_space=record.filters.get("owner_space", ""),
            abstract=meta.get("abstract", record.text[:200]),
            has_overview=meta.get("has_overview", False),
            has_content=meta.get("has_content", False),
            active_count=meta.get("active_count", 0),
            updated_at=meta.get("updated_at"),
        )

    def get_records_by_clips(self, filters: dict[str, Any], clip_ids: list[int]) -> list[IndexRecord]:
        clips = set(validate_clip_lookup(filters, clip_ids))
        matches = []
        for record in self._records.values():
            clip = clip_ordinal(record.metadata.get("clip_id", record.metadata.get("before_clip")))
            if clip in clips and self._matches_raw_filters(record, filters):
                matches.append(record)
        return sorted(matches, key=lambda record: record.uri)

    def get_records_at_time(self, filters: dict[str, Any], timestamp: float) -> list[IndexRecord]:
        validate_clip_lookup(filters, [])
        timestamp = seconds(timestamp)
        return sorted(
            (record for record in self._records.values()
             if self._matches_raw_filters(record, filters)
             and contains_time(record.metadata, timestamp)),
            key=lambda record: record.uri,
        )

    def get_record(self, record_id: str) -> IndexRecord | None:
        """Get a record by ID (for testing).

        Args:
            record_id: Record ID to fetch

        Returns:
            IndexRecord if found, None otherwise
        """
        return self._records.get(record_id)

    def count(self, filters: dict[str, Any] | None = None) -> int:
        """Get record count, optionally constrained by filters.

        Returns:
            Number of records in the index
        """
        if not filters:
            return len(self._records)

        def matches(record: IndexRecord) -> bool:
            for key, expected in filters.items():
                actual = record.filters.get(key)
                if isinstance(expected, (list, tuple, set)):
                    if actual not in expected:
                        return False
                elif actual != expected:
                    return False
            return True

        return sum(matches(record) for record in self._records.values())

    def delete_account_data(self, account_id: str) -> int:
        """Delete all index records for an account.

        Args:
            account_id: Account ID to delete

        Returns:
            Count of deleted records
        """
        to_delete = [
            record_id for record_id, record in self._records.items()
            if record.filters.get("account_id") == account_id
        ]
        for record_id in to_delete:
            self._records.pop(record_id, None)
            self._vectors.pop(record_id, None)
        return len(to_delete)

    def delete_by_owner_space(
        self, account_id: str, owner_space: str
    ) -> int:
        """Delete all records matching account_id + owner_space.

        Args:
            account_id: Account ID to filter
            owner_space: Owner space in colon format "user:{id}" or "agent:{id}"

        Returns:
            Count of deleted records
        """
        to_delete = [
            record_id for record_id, record in self._records.items()
            if record.filters.get("account_id") == account_id
            and record.filters.get("owner_space") == owner_space
        ]
        for record_id in to_delete:
            self._records.pop(record_id, None)
            self._vectors.pop(record_id, None)
        return len(to_delete)

    def clear(self) -> None:
        """Clear all records (for testing)."""
        self._records.clear()
        self._vectors.clear()

    # -- Backward compatibility convenience method ------------------------------

    def search(self, query, top_k: int | None = None) -> list[SeedHit]:
        """Convenience method for backward compatibility with TypedQuery API.

        Converts TypedQuery to search_by_vector() call.
        This maintains compatibility with older tests while using the new API.

        Args:
            query: TypedQuery object with text, account_id, owner_space, etc.
            top_k: Override for top_k (defaults to query.top_k)

        Returns:
            List of SeedHit results
        """
        from core.models import TypedQuery

        if not isinstance(query, TypedQuery):
            raise TypeError(f"Expected TypedQuery, got {type(query)}")

        # Build filters from TypedQuery
        filters: dict[str, Any] = {}
        if query.account_id:
            filters["account_id"] = query.account_id
        if query.owner_space:
            filters["owner_space"] = query.owner_space
        if query.context_type:
            filters["context_type"] = query.context_type
        if query.categories:
            filters["category"] = query.categories
        if query.resource_type:
            filters["resource_type"] = query.resource_type

        # Generate mock vector from query text
        query_vector = self._mock_vector(query.text)

        return self.search_by_vector(
            query_vector=query_vector,
            filters=filters,
            top_k=top_k or query.top_k,
        )
