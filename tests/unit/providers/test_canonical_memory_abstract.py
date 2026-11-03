"""Memory summaries survive vector and lexical index round trips."""

from contextlib import nullcontext
from unittest.mock import Mock

import pytest

from core.models import ContextNode, RequestContext, TypedQuery
from index.index_record_builder import build_index_records
from providers.vector_index.chroma_index import ChromaVectorIndex
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from providers.vector_index.opengauss_index import OpenGaussVectorIndex
from retrieval.bm25_index import Bm25Document, Bm25Index, build_bm25_document
from retrieval.seed_retriever import SeedRetriever


@pytest.fixture
def memory():
    return ContextNode(
        uri="ctx://tenant/users/reader/memories/entities/observatory",
        context_type="MEMORY",
        category="entity",
        level=3,
        owner_space="user:reader",
        abstract="An observatory used by the astronomy club.",
        overview="The club has access to research instruments.",
        content="# Facilities\n" + "Instrument maintenance. " * 20
        + "\nThe spectrograph measures stellar spectra.",
    )


@pytest.mark.parametrize("level", [0, 2])
@pytest.mark.parametrize("provider", ["memory", "opengauss", "chroma"])
def test_index_round_trip_preserves_node_summary(memory, level, provider):
    record = next(record for record in build_index_records(memory) if record.level == level)
    if provider == "memory":
        hit = InMemoryVectorIndex._record_to_vector_hit(record, 0.9)
    elif provider == "opengauss":
        hit = OpenGaussVectorIndex._row_to_hit({
            "uri": record.uri,
            "level": record.level,
            "text": record.text,
            "metadata": record.metadata,
            "filters": record.filters,
            "score": 0.9,
        })
    else:
        index = ChromaVectorIndex.__new__(ChromaVectorIndex)
        index._locked = nullcontext
        index._collection = Mock()
        index._collection.count.return_value = 1
        index._collection.query.return_value = {
            "ids": [[record.id]],
            "metadatas": [[index._record_to_meta(record)]],
            "documents": [[record.text]],
            "distances": [[0.1]],
        }
        hit = index.search_by_vector([1.0], record.filters, 1)[0]

    assert hit.abstract == memory.abstract
    assert hit.uri == record.uri
    assert hit.metadata["abstract"] == memory.abstract
    assert record.text == (memory.abstract if level == 0 else memory.content)


def test_lexical_hit_carries_summary_and_full_searchable_body(memory, monkeypatch):
    monkeypatch.setattr("retrieval.bm25_index._tokenize", lambda text: text.lower().split())
    records = build_index_records(memory)
    index = Bm25Index()
    index.bulk_upsert([
        *(build_bm25_document(record) for record in records),
        *(Bm25Document(str(i), text, {}) for i, text in enumerate([
            "orchard apples", "river kayaking", "ceramic workshop", "forest trails",
        ])),
    ])
    query = TypedQuery(text="spectrograph", context_type="MEMORY", categories=[])
    context = RequestContext("tenant", "reader", "assistant", "session", "trace")
    hits = SeedRetriever(Mock(), Mock(), bm25_index=index)._bm25_search(query, context)

    assert len(hits) == 1
    assert hits[0].level == 2
    assert hits[0].abstract == memory.abstract
    assert hits[0].metadata["account_id"] == "tenant"
    assert hits[0].metadata["owner_space"] == "user:reader"
    assert hits[0].parent_uri == memory.uri
