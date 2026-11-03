"""Application-level BM25 index for keyword retrieval.

Built from ChromaDB documents at startup, kept in sync via upsert/delete.
Uses rank_bm25 library with English tokenization (NLTK).
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

from core.memory_names import memory_directory
from core.models import IndexRecord
from retrieval.code_ordering import code_source_key

logger = logging.getLogger(__name__)

# Lazy-import NLTK to avoid startup cost
_nltk_ready = False


def _ensure_nltk():
    global _nltk_ready
    if _nltk_ready:
        return
    import nltk
    for res in ("punkt_tab", "stopwords"):
        try:
            nltk.data.find(f"tokenizers/{res}" if "punkt" in res else f"corpora/{res}")
        except LookupError:
            nltk.download(res, quiet=True)
    _nltk_ready = True


def _tokenize(text: str) -> list[str]:
    """Tokenize English text: lowercase, stem, remove stopwords."""
    _ensure_nltk()
    import nltk
    from nltk.stem import PorterStemmer

    stemmer = PorterStemmer()
    stops = set(nltk.corpus.stopwords.words("english"))
    tokens = nltk.word_tokenize(text.lower())
    return [stemmer.stem(t) for t in tokens if t.isalnum() and t not in stops]


def _tokenize_code_field(text: str) -> list[str]:
    """Keep whole identifiers and expose camel-case components on both sides."""
    words = re.findall(r"[A-Za-z0-9]+", str(text or ""))
    expanded = []
    for word in words:
        parts = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", word)
        parts = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", parts)
        expanded.append(word)
        if parts != word:
            expanded.append(parts)
    return _tokenize(" ".join(expanded))


@dataclass
class Bm25Hit:
    """A single BM25 search result."""
    id: str
    text: str
    score: float
    metadata: dict
    fields: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Bm25Document:
    """A document accepted by the shared BM25 lifecycle."""

    id: str
    text: str
    metadata: dict
    # Optional field payload used by the code RESOURCE route.  Generic BM25
    # callers continue to provide only id/text/metadata.
    fields: dict[str, str] = field(default_factory=dict)


_CODE_LEXICAL_FIELDS = (
    "relative_path",
    "source_extension",
    "language",
    "symbol",
    "symbol_kind",
    "signature",
)
_CODE_FIELD_WEIGHTS = {
    "symbol": 4.5,
    "signature": 2.6,
    "path": 1.2,
    "kind": 0.8,
    "body": 0.45,
}


def build_bm25_document(record: IndexRecord) -> Bm25Document:
    """Preserve canonical node metadata when indexing an IndexRecord."""
    return Bm25Document(
        id=record.id,
        text=record.text,
        metadata={
            **dict(record.metadata or {}),
            **dict(record.filters or {}),
            "uri": record.uri,
            "level": record.level,
        },
    )


def build_code_bm25_document(record: IndexRecord) -> Bm25Document | None:
    """Adapt a code IndexRecord to the shared lexical index contract."""

    # Validated IndexRecord filters are authoritative.  In particular, a
    # stale/conflicting metadata value must never move a document into another
    # account or resource subtype.
    metadata = build_bm25_document(record).metadata
    if (
        metadata.get("context_type") != "RESOURCE"
        or metadata.get("resource_type") != "code"
    ):
        return None
    fields = {
        "symbol": str(metadata.get("symbol") or ""),
        "signature": str(metadata.get("signature") or ""),
        "path": str(metadata.get("relative_path") or ""),
        "extension": str(metadata.get("source_extension") or ""),
        "language": str(metadata.get("language") or ""),
        "kind": str(metadata.get("symbol_kind") or "code"),
        "body": str(record.text or ""),
    }
    lexical_text = " ".join(
        [fields["body"]]
        + [str(metadata.get(field) or "") for field in _CODE_LEXICAL_FIELDS]
    )
    return Bm25Document(
        id=record.id,
        text=re.sub(r"[^A-Za-z0-9]+", " ", lexical_text).strip(),
        metadata=metadata,
        fields=fields,
    )


class Bm25Index:
    """Thread-safe application-level BM25 index.

    Lifecycle:
    1. build_from_chroma() — initial load at startup
    2. upsert() / delete() — incremental updates from outbox worker
    3. search() — query-time retrieval
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ids: list[str] = []       # parallel to _texts
        self._texts: list[str] = []
        self._metas: list[dict] = []
        self._fields: list[dict[str, str]] = []
        self._tokenized: list[list[str]] = []
        self._field_tokenized: list[dict[str, list[str]]] = []
        self._positions: dict[str, int] = {}
        self._bm25: BM25Okapi | None = None
        self._dirty = False
        self._rebuild_count = 0

    # -- Build / Rebuild -----------------------------------------------------

    def build_from_chroma(self, collection) -> None:
        """Load all documents from ChromaDB collection and build index."""
        count = collection.count()
        if count == 0:
            logger.info("Bm25Index: empty collection, nothing to index")
            return

        # Fetch all documents in batches
        batch_size = 1000
        all_ids: list[str] = []
        all_texts: list[str] = []
        all_metas: list[dict] = []
        for offset in range(0, count, batch_size):
            results = collection.get(
                include=["documents", "metadatas"],
                limit=batch_size,
                offset=offset,
            )
            all_ids.extend(results["ids"])
            all_texts.extend(results["documents"] or [])
            all_metas.extend(results["metadatas"] or [])

        with self._lock:
            self._ids = all_ids
            self._texts = all_texts
            self._metas = [dict(m or {}) for m in all_metas]
            for metadata in self._metas:
                if "agent_memory_grounding_json" in metadata:
                    metadata["grounding"] = json.loads(metadata.pop("agent_memory_grounding_json"))
            self._fields = [{} for _ in all_ids]
            self._tokenized = [_tokenize(t) for t in all_texts]
            self._field_tokenized = [{} for _ in all_ids]
            self._positions = {
                doc_id: position for position, doc_id in enumerate(all_ids)
            }
            self._rebuild()

        logger.info("Bm25Index: built from ChromaDB, %d documents indexed", len(all_ids))

    def _rebuild(self) -> None:
        """Rebuild BM25 index from current data (caller must hold lock)."""
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None
        self._rebuild_count += 1
        self._dirty = False

    def _ensure_built(self) -> None:
        """Build once after one or more mutations (caller must hold lock)."""
        if self._dirty:
            self._rebuild()

    # -- Incremental Updates -------------------------------------------------

    def upsert(self, doc_id: str, text: str, metadata: dict) -> None:
        """Insert or update a single document."""
        self.bulk_upsert((Bm25Document(doc_id, text, metadata or {}),))

    def bulk_upsert(self, documents: Iterable[Bm25Document]) -> None:
        """Insert or update documents without rebuilding per document."""

        prepared = [
            Bm25Document(
                document.id,
                document.text,
                dict(document.metadata or {}),
                dict(document.fields or {}),
            )
            for document in documents
        ]
        if not prepared:
            return

        tokenized = [_tokenize(document.text) for document in prepared]
        with self._lock:
            for document, tokens in zip(prepared, tokenized):
                position = self._positions.get(document.id)
                if position is None:
                    self._positions[document.id] = len(self._ids)
                    self._ids.append(document.id)
                    self._texts.append(document.text)
                    self._metas.append(document.metadata)
                    self._fields.append(document.fields)
                    self._tokenized.append(tokens)
                    self._field_tokenized.append(
                        {name: _tokenize_code_field(value) for name, value in document.fields.items()}
                    )
                    continue
                self._texts[position] = document.text
                self._metas[position] = document.metadata
                self._fields[position] = document.fields
                self._tokenized[position] = tokens
                self._field_tokenized[position] = {
                    name: _tokenize_code_field(value) for name, value in document.fields.items()
                }
            self._dirty = True

    def replace_where(
        self,
        filters: dict,
        documents: Iterable[Bm25Document],
    ) -> None:
        """Replace one filtered corpus slice and defer the next rebuild."""

        prepared = [
            Bm25Document(
                document.id,
                document.text,
                dict(document.metadata or {}),
                dict(document.fields or {}),
            )
            for document in documents
        ]
        tokenized = [_tokenize(document.text) for document in prepared]
        with self._lock:
            if len(self._fields) != len(self._ids):
                self._fields = [{} for _ in self._ids]
            if len(self._field_tokenized) != len(self._ids):
                self._field_tokenized = [{} for _ in self._ids]
            keep = [
                (doc_id, text, metadata, fields, tokens, field_tokens)
                for doc_id, text, metadata, fields, tokens, field_tokens in zip(
                    self._ids,
                    self._texts,
                    self._metas,
                    self._fields,
                    self._tokenized,
                    self._field_tokenized,
                )
                if not _matches_filters(metadata, filters)
            ]
            self._ids = [item[0] for item in keep]
            self._texts = [item[1] for item in keep]
            self._metas = [item[2] for item in keep]
            self._fields = [item[3] for item in keep]
            self._tokenized = [item[4] for item in keep]
            self._field_tokenized = [item[5] for item in keep]
            self._positions = {
                doc_id: position for position, doc_id in enumerate(self._ids)
            }
            for document, tokens in zip(prepared, tokenized):
                position = self._positions.get(document.id)
                if position is None:
                    self._positions[document.id] = len(self._ids)
                    self._ids.append(document.id)
                    self._texts.append(document.text)
                    self._metas.append(document.metadata)
                    self._fields.append(document.fields)
                    self._tokenized.append(tokens)
                    self._field_tokenized.append(
                        {name: _tokenize_code_field(value) for name, value in document.fields.items()}
                    )
                    continue
                self._texts[position] = document.text
                self._metas[position] = document.metadata
                self._fields[position] = document.fields
                self._tokenized[position] = tokens
                self._field_tokenized[position] = {
                    name: _tokenize_code_field(value) for name, value in document.fields.items()
                }
            self._dirty = True

    def delete(self, doc_ids: list[str]) -> None:
        """Remove documents by ID."""
        id_set = set(doc_ids)
        with self._lock:
            if len(self._fields) != len(self._ids):
                self._fields = [{} for _ in self._ids]
            if len(self._field_tokenized) != len(self._ids):
                self._field_tokenized = [{} for _ in self._ids]
            keep = [
                (doc_id, text, metadata, fields, tokens, field_tokens)
                for doc_id, text, metadata, fields, tokens, field_tokens in zip(
                    self._ids,
                    self._texts,
                    self._metas,
                    self._fields,
                    self._tokenized,
                    self._field_tokenized,
                )
                if doc_id not in id_set
            ]
            if len(keep) == len(self._ids):
                return  # nothing to delete
            self._ids = [k[0] for k in keep]
            self._texts = [k[1] for k in keep]
            self._metas = [k[2] for k in keep]
            self._fields = [k[3] for k in keep]
            self._tokenized = [k[4] for k in keep]
            self._field_tokenized = [k[5] for k in keep]
            self._positions = {
                doc_id: position for position, doc_id in enumerate(self._ids)
            }
            self._dirty = True

    # -- Search --------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 30,
        filters: dict | None = None,
        *,
        include_nonpositive_matches: bool = False,
    ) -> list[Bm25Hit]:
        """Search BM25 index, optionally filtering by metadata.

        Args:
            query: Search query text
            top_k: Number of results to return
            filters: Optional metadata filters (account_id, owner_space, etc.)

        Returns:
            List of Bm25Hit sorted by BM25 score descending
        """
        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return []

        with self._lock:
            self._ensure_built()
            if self._bm25 is None or not self._ids:
                return []
            if len(self._fields) != len(self._ids):
                self._fields = [{} for _ in self._ids]
            scores = self._bm25.get_scores(tokenized_query)
            documents = [
                (doc_id, text, dict(metadata), fields, score)
                for doc_id, text, metadata, fields, score in zip(
                    self._ids,
                    self._texts,
                    self._metas,
                    self._fields,
                    scores,
                )
            ]

        # Build results with filtering
        results: list[Bm25Hit] = []
        for doc_id, text, meta, fields, score in documents:
            if score <= 0 and not include_nonpositive_matches:
                continue
            if filters and not _matches_filters(meta, filters):
                continue
            results.append(Bm25Hit(
                id=doc_id,
                text=text,
                score=float(score),
                metadata=meta,
                fields=dict(fields),
            ))

        results.sort(key=lambda h: h.score, reverse=True)
        return results[:top_k]

    def search_code(
        self,
        query: str,
        top_k: int = 30,
        filters: dict | None = None,
        *,
        include_nonpositive_matches: bool = False,
        allowed_ids: set[str] | None = None,
    ) -> list[Bm25Hit]:
        """Search the field-aware code corpus with deterministic BM25F-like scores.

        Field data is derived at mutation time and uses the same record IDs and
        metadata filters as generic BM25.  Rebuild of aggregate statistics is
        deferred until this method (or :meth:`search`) is called.
        """

        query_tokens = _tokenize_code_field(query)
        if not query_tokens:
            return []
        with self._lock:
            self._ensure_built()
            if not self._ids:
                return []
            if len(self._fields) != len(self._ids):
                self._fields = [{} for _ in self._ids]
            if len(self._field_tokenized) != len(self._ids):
                self._field_tokenized = [{} for _ in self._ids]
            all_docs = list(
                zip(
                    self._ids,
                    self._texts,
                    self._metas,
                    self._fields,
                    self._field_tokenized,
                )
            )
            docs = [
                document
                for document in all_docs
                if (not filters or _matches_filters(document[2], filters))
                and (allowed_ids is None or document[0] in allowed_ids)
            ]
            if not docs:
                return []
            # Snapshot field statistics under the index lock.  Scoring itself
            # is pure, so the lock is not held while sorting/filtering.
            def field_tokens_for(
                tokens: dict[str, list[str]], field_name: str,
            ) -> list[str]:
                # Extension and language are part of the searchable path
                # facet, while remaining separately visible in hit.fields.
                if field_name == "path":
                    return (
                        list(tokens.get("path", ()))
                        + list(tokens.get("extension", ()))
                        + list(tokens.get("language", ()))
                    )
                return list(tokens.get(field_name, ()))

            field_stats: dict[str, tuple[int, dict[str, int], float]] = {}
            for field_name in _CODE_FIELD_WEIGHTS:
                lengths = [
                    len(field_tokens_for(tokens, field_name))
                    for *_, tokens in docs
                ]
                doc_freq: dict[str, int] = {}
                for *_, tokens in docs:
                    for term in set(field_tokens_for(tokens, field_name)):
                        doc_freq[term] = doc_freq.get(term, 0) + 1
                avg_len = (sum(lengths) / len(lengths)) if lengths else 0.0
                field_stats[field_name] = (len(docs), doc_freq, avg_len)

        scored: list[Bm25Hit] = []
        k1 = 1.2
        b = 0.75
        for doc_id, text, metadata, fields, field_tokens in docs:
            score = 0.0
            for field_name, weight in _CODE_FIELD_WEIGHTS.items():
                if field_name == "path":
                    field_tokens_for_doc = (
                        list(field_tokens.get("path", ()))
                        + list(field_tokens.get("extension", ()))
                        + list(field_tokens.get("language", ()))
                    )
                else:
                    field_tokens_for_doc = list(field_tokens.get(field_name, ()))
                length = len(field_tokens_for_doc)
                if not length:
                    continue
                _, doc_freq, avg_len = field_stats[field_name]
                term_counts: dict[str, int] = {}
                for token in field_tokens_for_doc:
                    term_counts[token] = term_counts.get(token, 0) + 1
                for term in query_tokens:
                    tf = term_counts.get(term, 0)
                    if not tf:
                        continue
                    df = doc_freq.get(term, 0)
                    idf = math.log(1.0 + (len(docs) - df + 0.5) / (df + 0.5))
                    norm = tf * (k1 + 1.0) / (
                        tf + k1 * (1.0 - b + b * length / max(avg_len, 1.0))
                    )
                    score += weight * idf * norm
            if score <= 0.0 and not include_nonpositive_matches:
                continue
            scored.append(
                Bm25Hit(
                    id=doc_id,
                    text=text,
                    score=float(score),
                    metadata=dict(metadata),
                    fields=dict(fields),
                )
            )
        scored.sort(key=lambda hit: (-hit.score, code_source_key({"id": hit.id, "metadata": hit.metadata})))
        return scored[: max(0, top_k)]

    @property
    def doc_count(self) -> int:
        with self._lock:
            return len(self._ids)

    @property
    def rebuild_count(self) -> int:
        """Number of BM25 builds, useful for lifecycle diagnostics/tests."""
        with self._lock:
            return self._rebuild_count


def _matches_filters(metadata: dict, filters: dict) -> bool:
    """Check if metadata matches all filter criteria."""
    for key, expected in filters.items():
        val = memory_directory(metadata.get("uri", "")) if key == "memory_directory" else metadata.get(key)
        if val is None:
            return False
        if isinstance(expected, list):
            if val not in expected:
                return False
        elif val != expected:
            return False
    return True
