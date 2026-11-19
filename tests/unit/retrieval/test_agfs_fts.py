import time
from dataclasses import replace

import pytest

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
