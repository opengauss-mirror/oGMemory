"""Application-level BM25 index for keyword retrieval.

Built from ChromaDB documents at startup, kept in sync via upsert/delete.
Uses rank_bm25 library with English tokenization (NLTK).
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from collections.abc import Iterable

from rank_bm25 import BM25Okapi

from core.memory_names import memory_directory
from core.models import IndexRecord
from core.projection_ownership import check_projection_owner

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


@dataclass
class Bm25Hit:
    """A single BM25 search result."""
    id: str
    text: str
    score: float
    metadata: dict


@dataclass(frozen=True)
class Bm25Document:
    """Canonical document for lexical indexing."""

    id: str
    text: str
    metadata: dict


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
        self._tokenized: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    # -- Build / Rebuild -----------------------------------------------------

    def build_from_chroma(self, collection) -> None:
        """Load all documents from ChromaDB collection and build index."""
        count = collection.count()
        if count == 0:
            logger.info("Bm25Index: empty collection, nothing to index")
            return

        # Fetch all documents in batches
        batch_size = 1000
        all_ids, all_texts, all_metas = [], [], []
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
            check_projection_owner()
            self._ids = all_ids
            self._texts = all_texts
            self._metas = [dict(m or {}) for m in all_metas]
            for metadata in self._metas:
                if "agent_memory_grounding_json" in metadata:
                    metadata["grounding"] = json.loads(metadata.pop("agent_memory_grounding_json"))
            self._tokenized = [_tokenize(t) for t in all_texts]
            self._bm25 = BM25Okapi(self._tokenized)

        logger.info("Bm25Index: built from ChromaDB, %d documents indexed", len(all_ids))

    def _rebuild(self) -> None:
        """Rebuild BM25 index from current data (caller must hold lock)."""
        self._tokenized = [_tokenize(t) for t in self._texts]
        self._bm25 = BM25Okapi(self._tokenized)

    # -- Incremental Updates -------------------------------------------------

    def upsert(self, doc_id: str, text: str, metadata: dict) -> None:
        """Insert or update a single document."""
        with self._lock:
            check_projection_owner()
            # Check if ID exists
            for i, existing_id in enumerate(self._ids):
                if existing_id == doc_id:
                    self._texts[i] = text
                    self._metas[i] = metadata or {}
                    self._tokenized[i] = _tokenize(text)
                    self._bm25 = BM25Okapi(self._tokenized)
                    return
            # New document
            self._ids.append(doc_id)
            self._texts.append(text)
            self._metas.append(metadata or {})
            self._tokenized.append(_tokenize(text))
            self._bm25 = BM25Okapi(self._tokenized)

    def bulk_upsert(self, documents: Iterable[Bm25Document]) -> None:
        """Insert or update a batch with one index rebuild."""
        prepared = list(documents)
        if not prepared:
            return
        with self._lock:
            check_projection_owner()
            positions = {doc_id: i for i, doc_id in enumerate(self._ids)}
            for document in prepared:
                position = positions.get(document.id)
                if position is None:
                    positions[document.id] = len(self._ids)
                    self._ids.append(document.id)
                    self._texts.append(document.text)
                    self._metas.append(dict(document.metadata or {}))
                else:
                    self._texts[position] = document.text
                    self._metas[position] = dict(document.metadata or {})
            self._rebuild()

    def delete(self, doc_ids: list[str]) -> None:
        """Remove documents by ID."""
        id_set = set(doc_ids)
        with self._lock:
            check_projection_owner()
            keep = [(i, t, m) for i, t, m in zip(self._ids, self._texts, self._metas)
                    if i not in id_set]
            if len(keep) == len(self._ids):
                return  # nothing to delete
            self._ids = [k[0] for k in keep]
            self._texts = [k[1] for k in keep]
            self._metas = [k[2] for k in keep]
            self._rebuild()

    # -- Search --------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 30,
        filters: dict | None = None,
    ) -> list[Bm25Hit]:
        """Search BM25 index, optionally filtering by metadata.

        Args:
            query: Search query text
            top_k: Number of results to return
            filters: Optional metadata filters (account_id, owner_space, etc.)

        Returns:
            List of Bm25Hit sorted by BM25 score descending
        """
        with self._lock:
            check_projection_owner()
            if self._bm25 is None or not self._ids:
                return []

            tokenized_query = _tokenize(query)
            if not tokenized_query:
                return []

            scores = self._bm25.get_scores(tokenized_query)

        # Build results with filtering
        results: list[Bm25Hit] = []
        for i, score in enumerate(scores):
            if score <= 0:
                continue
            meta = self._metas[i]
            if filters and not _matches_filters(meta, filters):
                continue
            results.append(Bm25Hit(
                id=self._ids[i],
                text=self._texts[i],
                score=float(score),
                metadata=meta,
            ))

        results.sort(key=lambda h: h.score, reverse=True)
        return results[:top_k]

    @property
    def doc_count(self) -> int:
        return len(self._ids)


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
