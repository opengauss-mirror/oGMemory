from __future__ import annotations

import pytest

from core.models import ContextNode, RequestContext, SeedHit
from extraction.prefetch import MemoryPrefetcher
from extraction.schemas.models import FieldType, MemoryTypeSchema, SchemaField
from extraction.schemas.registry import SchemaRegistry


def _ctx() -> RequestContext:
    return RequestContext(
        account_id="acct",
        user_id="user",
        agent_id="agent",
        session_id="sess",
        trace_id="trace",
    )


def _node(uri: str, category: str = "preference") -> ContextNode:
    return ContextNode(
        uri=uri,
        context_type="MEMORY",
        category=category,
        level=2,
        owner_space="user:user",
        abstract=f"{category} abstract",
        overview=f"{category} overview",
        content=f"{category} content",
    )


class FakeFS:
    def __init__(self):
        self.nodes: dict[str, ContextNode] = {}
        self.children: dict[str, list[str]] = {}

    def exists(self, uri, ctx):
        return uri in self.nodes

    def read_node(self, uri, ctx):
        return self.nodes[uri]

    def list_children(self, uri, ctx):
        return self.children.get(uri, [])


class FailingFS(FakeFS):
    def __init__(self, *, fail_exists=False, fail_read=False, fail_list=False):
        super().__init__()
        self.fail_exists = fail_exists
        self.fail_read = fail_read
        self.fail_list = fail_list

    def exists(self, uri, ctx):
        if self.fail_exists:
            raise RuntimeError("exists failed")
        return super().exists(uri, ctx)

    def read_node(self, uri, ctx):
        if self.fail_read:
            raise RuntimeError("read failed")
        return super().read_node(uri, ctx)

    def list_children(self, uri, ctx):
        if self.fail_list:
            raise RuntimeError("list failed")
        return super().list_children(uri, ctx)


class FakeEmbedder:
    def __init__(self):
        self.queries: list[str] = []

    def embed_texts(self, texts):
        self.queries.extend(texts)
        return [[1.0, 0.0, 0.0] for _ in texts]


class FakeVectorIndex:
    def __init__(self):
        self.filters = None
        self.hits = []

    def search_by_vector(self, query_vector, filters, top_k):
        self.filters = filters
        return self.hits


class FailingVectorIndex(FakeVectorIndex):
    def search_by_vector(self, query_vector, filters, top_k):
        raise RuntimeError("search failed")


def _prefetcher(fs: FakeFS, vector_index: FakeVectorIndex, embedder: FakeEmbedder):
    registry = SchemaRegistry()
    from core.uri_resolver import URIResolver

    return MemoryPrefetcher(fs, vector_index, embedder, registry, URIResolver(registry))


def test_prefetch_single_file_profile_reads_existing_node():
    fs = FakeFS()
    vector = FakeVectorIndex()
    embedder = FakeEmbedder()
    uri = "ctx://acct/users/user/memories/profile"
    fs.nodes[uri] = _node(uri, "profile")

    result = _prefetcher(fs, vector, embedder)._prefetch("profile", _ctx())

    assert uri in result.listed_uris
    assert uri in result.read_uris
    assert "Existing profile memory" in result.messages[0]


def test_prefetch_multi_file_uses_conversation_query_and_owner_filter():
    fs = FakeFS()
    vector = FakeVectorIndex()
    embedder = FakeEmbedder()
    hit_uri = "ctx://acct/users/user/memories/preferences/coffee/content.md"
    node_uri = "ctx://acct/users/user/memories/preferences/coffee"
    fs.nodes[node_uri] = _node(node_uri, "preference")
    vector.hits = [
        SeedHit(uri=hit_uri, score=0.91, category="preference", abstract="coffee", level=2)
    ]

    result = _prefetcher(fs, vector, embedder)._prefetch(
        "preference",
        _ctx(),
        conversation_text="coffee preference",
    )

    assert embedder.queries == ["coffee preference"]
    assert vector.filters == {
        "category": "preference",
        "account_id": "acct",
        "owner_space": "user:user",
    }
    assert node_uri in result.read_uris
    assert "similarity: 0.91" in result.messages[0]


def test_prefetch_add_only_lists_recent_entries_newest_first():
    fs = FakeFS()
    vector = FakeVectorIndex()
    embedder = FakeEmbedder()
    dir_uri = "ctx://acct/users/user/memories/events"
    old_uri = f"{dir_uri}/20240501000000_old"
    new_uri = f"{dir_uri}/20240601000000_new"
    fs.children[dir_uri] = [old_uri, new_uri]
    fs.nodes[old_uri] = _node(old_uri, "event")
    fs.nodes[new_uri] = _node(new_uri, "event")

    result = _prefetcher(fs, vector, embedder)._prefetch("event", _ctx())

    assert result.read_uris == {old_uri, new_uri}
    assert f"URI: {new_uri}" in result.messages[0]


def test_prefetch_ignores_incompatible_schema_versions():
    fs = FakeFS()
    vector = FakeVectorIndex()
    embedder = FakeEmbedder()
    registry = SchemaRegistry(schemas_dir="/path/that/does/not/exist")
    registry.register(
        MemoryTypeSchema(
            memory_type="future",
            description="Future schema",
            directory="future",
            filename_template="{{ routing_key }}.md",
            operation_mode="upsert",
            version="2.0",
            fields=[
                SchemaField(
                    name="routing_key",
                    field_type=FieldType.STRING,
                    required=True,
                )
            ],
        )
    )
    from core.uri_resolver import URIResolver

    result = MemoryPrefetcher(fs, vector, embedder, registry, URIResolver(registry))._prefetch(
        "future",
        _ctx(),
        conversation_text="future",
    )

    assert result.messages == []
    assert embedder.queries == []
    assert vector.filters is None


def test_prefetch_unknown_and_unknown_operation_modes_return_empty():
    fs = FakeFS()
    vector = FakeVectorIndex()
    embedder = FakeEmbedder()
    registry = SchemaRegistry(schemas_dir="/path/that/does/not/exist")
    registry.register(
        MemoryTypeSchema(
            memory_type="weird",
            description="Weird schema",
            directory="weird",
            filename_template="{{ routing_key }}.md",
            operation_mode="replace",
            fields=[],
        )
    )
    from core.uri_resolver import URIResolver

    prefetcher = MemoryPrefetcher(fs, vector, embedder, registry, URIResolver(registry))

    assert prefetcher._prefetch("missing", _ctx()).messages == []
    assert prefetcher._prefetch("weird", _ctx()).messages == []


def test_prefetch_for_span_combines_results_and_continues_after_error(monkeypatch):
    fs = FakeFS()
    vector = FakeVectorIndex()
    embedder = FakeEmbedder()
    prefetcher = _prefetcher(fs, vector, embedder)
    original_prefetch = prefetcher._prefetch

    def flaky_prefetch(category, ctx, *, conversation_text=None, lazy=False):
        if category == "broken":
            raise RuntimeError("broken")
        return original_prefetch(category, ctx, conversation_text=conversation_text, lazy=lazy)

    uri = "ctx://acct/users/user/memories/profile"
    fs.nodes[uri] = _node(uri, "profile")
    monkeypatch.setattr(prefetcher, "_prefetch", flaky_prefetch)

    result = prefetcher.prefetch_for_span(["broken", "profile"], _ctx())

    assert uri in result.read_uris
    assert result.messages


def test_prefetch_single_file_handles_missing_and_errors():
    vector = FakeVectorIndex()
    embedder = FakeEmbedder()

    missing = _prefetcher(FakeFS(), vector, embedder)._prefetch("profile", _ctx())
    failed = _prefetcher(FailingFS(fail_exists=True), vector, embedder)._prefetch("profile", _ctx())

    assert missing.messages == []
    assert failed.messages == []


def test_prefetch_multi_file_handles_no_hits_search_failure_and_read_failure():
    fs = FakeFS()
    vector = FakeVectorIndex()
    embedder = FakeEmbedder()

    no_hits = _prefetcher(fs, vector, embedder)._prefetch("preference", _ctx())

    failing_search = _prefetcher(fs, FailingVectorIndex(), FakeEmbedder())._prefetch("preference", _ctx())

    hit_uri = "ctx://acct/users/user/memories/preferences/coffee/content.md"
    read_fail_vector = FakeVectorIndex()
    read_fail_vector.hits = [SeedHit(uri=hit_uri, score=0.91, category="preference", abstract="coffee", level=2)]
    read_failed = _prefetcher(FailingFS(fail_read=True), read_fail_vector, FakeEmbedder())._prefetch(
        "preference",
        _ctx(),
    )

    assert no_hits.messages == []
    assert failing_search.messages == []
    assert "read failed" in read_failed.messages[0]


def test_prefetch_multi_file_uses_agent_owner_filter():
    fs = FakeFS()
    vector = FakeVectorIndex()
    embedder = FakeEmbedder()

    _prefetcher(fs, vector, embedder)._prefetch("tool", _ctx(), conversation_text="tool usage")

    assert vector.filters["owner_space"] == "agent:agent"


def test_prefetch_add_only_handles_empty_read_failure_and_list_failure():
    fs = FakeFS()
    vector = FakeVectorIndex()
    embedder = FakeEmbedder()
    empty = _prefetcher(fs, vector, embedder)._prefetch("event", _ctx())

    dir_uri = "ctx://acct/users/user/memories/events"
    event_uri = f"{dir_uri}/20240501000000_event"
    read_fail_fs = FailingFS(fail_read=True)
    read_fail_fs.children[dir_uri] = [event_uri]
    read_failed = _prefetcher(read_fail_fs, vector, embedder)._prefetch("event", _ctx())

    list_failed = _prefetcher(FailingFS(fail_list=True), vector, embedder)._prefetch("event", _ctx())

    assert empty.messages == []
    assert read_failed.messages == []
    assert list_failed.messages == []

@pytest.mark.parametrize('category,owner,directory', [
    ('event', 'user:user', 'ctx://acct/users/user/memories/events'),
    ('case', 'agent:agent', 'ctx://acct/agents/agent/memories/cases'),
])
@pytest.mark.parametrize('lazy', [False, True])
def test_occurrence_prefetch_recalls_old_semantic_hit_before_recent_entries(category, owner, directory, lazy):
    fs, vector, embedder = FakeFS(), FakeVectorIndex(), FakeEmbedder()
    old = directory + '/a_old_inspection'
    recent = [directory + f'/z_recent_{index:02}' for index in range(12)]
    fs.children[directory] = [old, *recent]
    for uri in fs.children[directory]:
        node = _node(uri, category)
        node.owner_space = owner
        node.metadata = {'version': 3, 'scope_ref': 'Project A', 'when': '2026-09-01', 'who': 'Technician'}
        fs.nodes[uri] = node
    vector.hits = [SeedHit(uri=old + '/content.md', score=.9, category=category, abstract='old inspection', level=2)]
    result = _prefetcher(fs, vector, embedder).prefetch_for_span([category], _ctx(), conversation_text='the earlier inspection', lazy=lazy)
    assert old in result.listed_uris
    assert len(result.listed_uris) == 10
    assert vector.filters == {'category': category, 'account_id': 'acct', 'owner_space': owner}
    assert embedder.queries == ['the earlier inspection']
    if lazy:
        assert result.uri_hints[0]['uri'] == old
        assert result.read_uris == set()
        assert result.read_versions == {}
    else:
        assert old in result.read_uris
        assert result.read_versions[old] == 3
        assert 'Content: ' + fs.nodes[old].content in result.messages[0]
        assert 'Project A' in result.messages[0]


def test_occurrence_prefetch_falls_back_when_search_fails_and_rejects_foreign_hits():
    fs, vector, embedder = FakeFS(), FakeVectorIndex(), FakeEmbedder()
    directory = 'ctx://acct/users/user/memories/events'
    uri = directory + '/inspection'
    fs.children[directory] = [uri]
    fs.nodes[uri] = _node(uri, 'event')
    vector.hits = [SeedHit(uri='ctx://other/users/user/memories/events/foreign', score=1, category='event', abstract='foreign', level=2)]
    for index in (vector, FailingVectorIndex()):
        result = _prefetcher(fs, index, embedder).prefetch_for_span(['event'], _ctx(), conversation_text='inspection')
        assert result.read_uris == {uri}
        assert result.listed_uris == {uri}


def test_occurrence_prefetch_uses_valid_semantic_hits_when_listing_fails():
    fs, vector = FailingFS(fail_list=True), FakeVectorIndex()
    uri = 'ctx://acct/users/user/memories/events/inspection'
    fs.nodes[uri] = _node(uri, 'event')
    vector.hits = [SeedHit(uri=uri, score=.9, category='event', abstract='inspection', level=2)]
    result = _prefetcher(fs, vector, FakeEmbedder()).prefetch_for_span(['event'], _ctx(), conversation_text='inspection')
    assert result.read_uris == {uri}


def test_occurrence_prefetch_stale_hits_do_not_consume_eager_slots():
    fs, vector, embedder = FakeFS(), FakeVectorIndex(), FakeEmbedder()
    directory = 'ctx://acct/users/user/memories/events'
    uris = [directory + f'/inspection_{i:02}' for i in range(12)]
    fs.children[directory] = uris
    for uri in uris:
        fs.nodes[uri] = _node(uri, 'event')
    vector.hits = [SeedHit(uri=directory+'/removed', score=1, category='event', abstract='old', level=2)]
    result = _prefetcher(fs, vector, embedder).prefetch_for_span(['event', 'case'], _ctx(), conversation_text='inspection')
    assert len(result.read_uris) == 10
    assert directory+'/removed' not in result.listed_uris
    assert embedder.queries == ['inspection']
