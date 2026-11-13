"""Exact filtered ranking against persisted Chroma vectors."""

import pytest

from core.models import IndexRecord
from providers.vector_index.chroma_index import ChromaVectorIndex


@pytest.mark.parametrize("size", [2, 1003])
def test_exact_filtered_ranking_across_pages(tmp_path, size):
    pytest.importorskip("chromadb")
    index = ChromaVectorIndex(persist_directory=str(tmp_path), dimension=2)
    records = [
        IndexRecord(str(i), f"ctx://acct/items/{i}", 2, f"item {i}",
                    {"account_id": "acct", "owner_space": "agent:agent"},
                    {"_embedding": [1.0, float(size - i)]})
        for i in range(size)
    ]
    records.append(IndexRecord("other", "ctx://other/item", 2, "other tenant",
                               {"account_id": "other", "owner_space": "agent:agent"},
                               {"_embedding": [1.0, 0.0]}))
    index.upsert(records)
    where = index._build_where({"account_id": "acct", "owner_space": "agent:agent", "level": 2})
    with index._locked():
        result = index._exact_filtered_query([1.0, 0.0], where, 2)
    assert result["ids"] == [[str(size - 1), str(size - 2)]]
    assert result["distances"][0] == pytest.approx([1 - 1 / 2**0.5, 1 - 1 / 5**0.5])
    assert result["documents"] == [[f"item {size - 1}", f"item {size - 2}"]]
    assert [row["account_id"] for row in result["metadatas"][0]] == ["acct", "acct"]
