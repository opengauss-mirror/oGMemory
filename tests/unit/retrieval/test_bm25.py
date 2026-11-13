"""Unit tests for Bm25Index and Vector-Anchored Fusion.

NLTK data (stopwords, punkt_tab) is not available in CI, so _tokenize and
_ensure_nltk are mocked at the module level.  The mocks use a simple regex
tokenizer + built-in stopword list + Porter stemmer (no data needed).
"""

import re
from unittest.mock import patch

import pytest
from nltk.stem import PorterStemmer

from retrieval.bm25_index import Bm25Index, Bm25Hit, _matches_filters
from retrieval.seed_retriever import SeedRetriever
from core.models import SeedHit

# ---------------------------------------------------------------------------
# Mock tokenizer (no NLTK data required)
# ---------------------------------------------------------------------------

_MOCK_STOPS = frozenset(
    "a an the and or but is are was were be been being have has had do does did "
    "will would could should may might shall can to of in for on with at by from "
    "as into through during before after above below between out off over under "
    "again further then once here there when where why how all both each few more "
    "most other some such no nor not only own same so than too very it its he she "
    "they them we you i me my his her our your this that these those what which who"
    .split()
)

_stemmer = PorterStemmer()


def _mock_tokenize(text: str) -> list[str]:
    """Simple regex tokenization + Porter stemming + stop filtering."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [_stemmer.stem(t) for t in tokens if t not in _MOCK_STOPS]


# Apply module-level mocks for the entire file
@pytest.fixture(autouse=True)
def _patch_nltk():
    with patch("retrieval.bm25_index._ensure_nltk"), \
         patch("retrieval.bm25_index._tokenize", _mock_tokenize):
        yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_basic_english(self):
        from retrieval.bm25_index import _tokenize
        tokens = _tokenize("Caroline is researching adoption agencies")
        assert len(tokens) > 0
        assert "is" not in tokens

    def test_empty_string(self):
        from retrieval.bm25_index import _tokenize
        assert _tokenize("") == []

    def test_special_chars_removed(self):
        from retrieval.bm25_index import _tokenize
        tokens = _tokenize("Hello! @World #123")
        assert "@" not in tokens
        assert "#" not in tokens


class TestMatchesFilters:
    def test_match_single_filter(self):
        assert _matches_filters({"account_id": "acme"}, {"account_id": "acme"})

    def test_no_match(self):
        assert not _matches_filters({"account_id": "acme"}, {"account_id": "other"})

    def test_match_list_filter(self):
        assert _matches_filters({"level": 2}, {"level": [0, 1, 2]})

    def test_missing_key(self):
        assert not _matches_filters({"account_id": "acme"}, {"owner_space": "user:alice"})

    def test_empty_filters(self):
        assert _matches_filters({"a": 1}, {})


class TestBm25Index:
    def _make_index(self):
        idx = Bm25Index()
        idx.upsert("doc1", "Caroline is researching adoption agencies", {"account_id": "acme", "category": "event"})
        idx.upsert("doc2", "Melanie participated in a charity race for mental health", {"account_id": "acme", "category": "event"})
        idx.upsert("doc3", "Jon started a dance studio after losing his job", {"account_id": "bigco", "category": "event"})
        return idx

    def test_search_basic(self):
        idx = self._make_index()
        hits = idx.search("adoption research")
        assert len(hits) > 0
        assert hits[0].id == "doc1"

    def test_search_with_filter(self):
        idx = self._make_index()
        hits = idx.search("dance studio", filters={"account_id": "bigco"})
        assert len(hits) > 0
        assert all(h.metadata.get("account_id") == "bigco" for h in hits)

    def test_search_filter_excludes(self):
        idx = self._make_index()
        hits = idx.search("adoption", filters={"account_id": "bigco"})
        assert len(hits) == 0

    def test_search_empty_index(self):
        idx = Bm25Index()
        assert idx.search("anything") == []

    def test_upsert_updates_existing(self):
        idx = self._make_index()
        idx.upsert("doc1", "Caroline adopted a child successfully", {"account_id": "acme"})
        hits = idx.search("adopted child")
        assert len(hits) > 0
        assert hits[0].id == "doc1"
        assert "child" in hits[0].text.lower() or "adopt" in hits[0].text.lower()

    def test_delete(self):
        idx = self._make_index()
        idx.delete(["doc1"])
        hits = idx.search("adoption")
        assert all(h.id != "doc1" for h in hits)

    def test_doc_count(self):
        idx = self._make_index()
        assert idx.doc_count == 3




class TestVectorAnchoredFusion:
    def test_fuse_pure_vector(self):
        """No BM25 hits -> scores unchanged."""
        vec_hits = [SeedHit(uri="a", score=0.9, level=2)]
        result = SeedRetriever._fuse_results(vec_hits, [])
        assert len(result) == 1
        assert result[0].score == pytest.approx(0.9 * 0.7, abs=0.01)

    def test_fuse_pure_bm25(self):
        """No vector hits -> BM25 scores with floor default."""
        bm25_hits = [SeedHit(
            uri="b", score=0.0, level=2,
            metadata={"_bm25_raw_score": 10.0},
        )]
        result = SeedRetriever._fuse_results([], bm25_hits)
        assert len(result) == 1
        assert result[0].score > 0

    def test_fuse_both_sources(self):
        """Both sources -> weighted combination."""
        vec_hits = [SeedHit(uri="a", score=0.8, level=2)]
        bm25_hits = [SeedHit(
            uri="a", score=0.0, level=2,
            metadata={"_bm25_raw_score": 10.0},
        )]
        result = SeedRetriever._fuse_results(vec_hits, bm25_hits)
        assert len(result) == 1
        assert result[0].score > 0.7

    def test_fuse_deduplicates_by_uri(self):
        """Same URI from both sources -> merged into one."""
        vec_hits = [SeedHit(uri="x", score=0.9, level=2)]
        bm25_hits = [SeedHit(
            uri="x", score=0.0, level=2,
            metadata={"_bm25_raw_score": 5.0},
        )]
        result = SeedRetriever._fuse_results(vec_hits, bm25_hits)
        assert len(result) == 1

    def test_saturation_function(self):
        """Higher BM25 score -> higher but saturating contribution."""
        bm25_low = [SeedHit(uri="a", score=0.0, level=2, metadata={"_bm25_raw_score": 1.0})]
        bm25_high = [SeedHit(uri="b", score=0.0, level=2, metadata={"_bm25_raw_score": 100.0})]
        result = SeedRetriever._fuse_results([], bm25_low + bm25_high)
        scores = {h.uri: h.score for h in result}
        assert scores["b"] > scores["a"]
