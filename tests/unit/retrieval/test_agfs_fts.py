import time
from dataclasses import replace

import pytest

from core.lexical import LexicalHit
from core.models import ContextNode, RequestContext, RetrievalConfig
from retrieval.agfs_retriever import AGFSRetrieval


_CTX = RequestContext("acme", "alice", "", "session", "trace")
_ROOT = "ctx://acme/users/alice/memories"
_ENTITIES = f"{_ROOT}/entities"


class SearchFS:
    def __init__(self):
        self.nodes = {}

    def add(self, slug, content):
        uri = f"{_ENTITIES}/{slug}"
        self.nodes[uri] = ContextNode(uri, "MEMORY", "entity", 2, "user:alice", "", "", content)
        return uri

    def list_children_including_inactive(self, uri, ctx):
        return [_ENTITIES] if uri == _ROOT else list(self.nodes)

    def read_search_node(self, uri, ctx, max_field_bytes):
        return self.nodes[uri]


def _warm(pipeline, query, ctx=_CTX):
    until = time.monotonic() + 5
    while time.monotonic() < until:
        result = pipeline.run(query, ctx, context_type="MEMORY")
        if result.trace.stages[0].metrics.get("fts") == "ready":
            return result
        time.sleep(0.01)
    raise AssertionError("FTS snapshot did not become ready")


def test_fts_finds_selective_match_past_scan_budget():
    fs = SearchFS()
    for i in range(520):
        fs.add(f"node{i}", "unrelated text")
    expected = fs.add("match", "northern salamander")
    pipeline = AGFSRetrieval(fs, RetrievalConfig())
    try:
        result = _warm(pipeline, "northern salamander")
        assert [hit.uri for hit in result.hits] == [expected]
        assert result.trace.stages[0].input_count == 1
    finally:
        pipeline.fts.close(5)


def test_fts_reports_partial_when_the_only_match_is_beyond_the_field_budget():
    class BoundedSearchFS(SearchFS):
        def read_search_node(self, uri, ctx, max_field_bytes):
            node = super().read_search_node(uri, ctx, max_field_bytes)
            return replace(node, content=node.content[:max_field_bytes],
                           metadata={"_search_truncated": len(node.content) > max_field_bytes})

    fs = BoundedSearchFS()
    fs.add("long-document", "x" * 20_000 + " tailonlyzafran")
    pipeline = AGFSRetrieval(fs, RetrievalConfig())
    try:
        result = _warm(pipeline, "tailonlyzafran")
        assert result.partial
        assert result.trace.warnings == ["agfs_field_budget"]
    finally:
        pipeline.fts.close(5)


@pytest.mark.parametrize("ranker", ["literal", "sqlite_bm25"])
def test_fts_reports_partial_when_candidates_exceed_the_scan_budget(ranker):
    fs = SearchFS()
    for i in range(520):
        fs.add(f"node{i:03d}", "ubiquitous text")
    pipeline = AGFSRetrieval(fs, RetrievalConfig(), fts_ranker=ranker)
    try:
        result = _warm(pipeline, "ubiquitous")
        assert len(result.hits) == pipeline._config.default_top_k
        assert result.partial
        assert result.trace.warnings == ["agfs_scan_budget"]
        assert result.trace.stages[0].input_count == 512
    finally:
        pipeline.fts.close(5)


def test_sqlite_bm25_ranks_matched_nodes_by_term_frequency():
    fs = SearchFS()
    first = fs.add("a-first", "salamander and other animals")
    frequent = fs.add("z-frequent", "salamander salamander salamander")
    pipeline = AGFSRetrieval(fs, RetrievalConfig(), fts_ranker="sqlite_bm25")
    try:
        result = _warm(pipeline, "salamander")
        assert [hit.uri for hit in result.hits] == [frequent, first]
    finally:
        pipeline.fts.close(5)


def test_fts_finds_chinese_bigrams_and_newly_published_nodes():
    fs = SearchFS()
    expected = fs.add("outdoors", "孩子喜欢恐龙和大自然")
    pipeline = AGFSRetrieval(fs, RetrievalConfig())
    try:
        assert [hit.uri for hit in _warm(pipeline, "恐龙").hits] == [expected]
        published = fs.add("museum", "新的恐龙博物馆")
        pipeline.fts.changed(published)
        result = pipeline.run("恐龙", _CTX, context_type="MEMORY")
        assert {hit.uri for hit in result.hits} == {expected, published}
    finally:
        pipeline.fts.close(5)


def test_fts_reuses_memory_snapshot_for_agent_skill_search():
    ctx = RequestContext("acme", "alice", "agent", "session", "trace")
    skills = "ctx://acme/agents/agent/skills"
    generated = f"{skills}/generated"
    skill = f"{generated}/salamander"

    class SharedSearchFS(SearchFS):
        def list_children_including_inactive(self, uri, ctx):
            if uri == skills:
                return [generated]
            if uri == generated:
                return [skill]
            if uri == _ENTITIES:
                return [u for u in self.nodes if u.startswith(_ENTITIES + "/")]
            return [_ENTITIES] if uri == _ROOT else []

    fs = SharedSearchFS()
    memory = fs.add("match", "northern salamander")
    fs.nodes[skill] = ContextNode(skill, "SKILL", "skill", 2, "agent:agent", "", "", "northern salamander")
    pipeline = AGFSRetrieval(fs, RetrievalConfig())
    try:
        result = _warm(pipeline, "salamander", ctx)
        assert [hit.uri for hit in result.hits] == [memory]
        skill_result = pipeline.run("salamander", ctx, context_type="SKILL")
        assert skill_result.trace.stages[0].metrics["fts"] == "ready"
        assert [hit.uri for hit in skill_result.hits] == [skill]
    finally:
        pipeline.fts.close(5)


def test_opengauss_bm25_reads_canonical_agfs_nodes_in_native_rank_order():
    fs = SearchFS()
    first = fs.add("a-first", "fresh canonical text")
    second = fs.add("z-second", "another canonical passage")

    class NativeIndex:
        def search(self, query, top_k, filters):
            assert query == "salamander"
            assert top_k == 50
            assert filters == {
                "account_id": "acme", "owner_space": ["user:alice", "user_alice"],
                "context_type": "MEMORY", "category": ["entity"],
                "memory_directory": ["entities"], "level": [0, 1, 2],
            }
            return [
                LexicalHit("second", 4.0, {"uri": f"{second}/content.md", "account_id": "acme", "owner_space": "user:alice"}),
                LexicalHit("first", 2.0, {"uri": f"{first}/content.md", "account_id": "acme", "owner_space": "user:alice"}),
            ]

    class Backfill:
        def ensure_ready(self, ctx, scopes):
            assert ctx == _CTX
            assert scopes == ["user:alice", "user_alice"]
            return True

    pipeline = AGFSRetrieval(fs, RetrievalConfig(), lexical_index=NativeIndex(), lexical_backfill=Backfill())
    result = pipeline.run("salamander", _CTX, context_type="MEMORY", categories=["entity"],
                          memory_directories=["entities"], fill_content_for_top_k=2)

    assert [(hit.uri, hit.content_excerpt, hit.match_reason) for hit in result.hits] == [
        (second, "another canonical passage", "openGauss BM25 match"),
        (first, "fresh canonical text", "openGauss BM25 match"),
    ]
    assert result.trace.stages[0].metrics["lexical"] == "ready"


def test_native_retrieval_resolves_distinct_winners_and_score_ties_before_stopping():
    fs = SearchFS()
    first_tie = fs.add("a-tie", "first matching canonical passage")
    second_tie = fs.add("b-tie", "second matching canonical passage")
    winner = fs.add("winner", "best matching canonical passage")
    lower = fs.add("lower", "another canonical passage")

    class NativeIndex:
        def search(self, query, top_k, filters):
            return [
                LexicalHit(str(i), score, {"uri": uri, "account_id": "acme", "owner_space": "user:alice"})
                for i, (uri, score) in enumerate((
                    (lower, 1.0), (second_tie, 5.0), (winner, 9.0),
                    (f"{winner}/content.md", 10.0), (first_tie, 5.0),
                ))
            ]

    pipeline = AGFSRetrieval(fs, RetrievalConfig(), lexical_index=NativeIndex())
    result = pipeline.run("salamander", _CTX, context_type="MEMORY", top_k=2)

    assert [hit.uri for hit in result.hits] == [winner, first_tie]
    stage = result.trace.stages[0]
    assert stage.metrics["native_candidate_count"] == 4
    assert stage.input_count == 3
    assert result.partial is False
