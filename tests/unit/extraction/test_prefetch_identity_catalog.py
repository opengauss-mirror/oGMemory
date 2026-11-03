"""Identity hints complement semantic hits without authorizing unread updates."""
from tests.unit.extraction.test_prefetch import (
    FakeEmbedder,
    FakeFS,
    FakeVectorIndex,
    _ctx,
    _node,
    _prefetcher,
)


def test_lazy_messages_include_catalog_alongside_vector_hints():
    from extraction.extraction_react_loop import ExtractionTaskContext
    from extraction.prefetch import PrefetchResult
    from tests.unit.extraction.test_react_loop import _make_loop

    result = PrefetchResult(is_lazy=True, messages=['Identity catalog: Atlas in Project A'],
                            uri_hints=[{'uri': 'ctx://acct/users/user/memories/entities/atlas', 'abstract': 'Atlas', 'score': .9}])
    task = ExtractionTaskContext(conversation_text='Update Atlas', prefetch_result=result)
    messages = _make_loop()._build_messages(_ctx(), task)
    assert any('Identity catalog: Atlas in Project A' in m['content'] for m in messages)
    assert any('Available Memory URIs' in m['content'] for m in messages)


class SummaryFS(FakeFS):
    def __init__(self):
        super().__init__()
        self.pages = []

    def list_node_summaries_page(self, directory, ctx, *, limit, after_uri=None):
        self.pages.append(after_uri)
        return sorted([n for n in self.nodes.values() if n.uri.rsplit('/', 1)[0] == directory
                       and (after_uri is None or n.uri > after_uri)], key=lambda n: n.uri)[:limit]


def test_catalog_uses_backend_order_for_encoded_names():
    from urllib.parse import unquote

    class PhysicalOrderFS(SummaryFS):
        list_node_order_key = staticmethod(unquote)

        def list_node_summaries_page(self, directory, ctx, *, limit, after_uri=None):
            return sorted([n for n in self.nodes.values()
                           if after_uri is None or unquote(n.uri) > after_uri],
                          key=lambda n: unquote(n.uri))[:limit]

    fs = PhysicalOrderFS()
    for i in range(101):
        node = _node(f'ctx://acct/users/user/memories/entities/x%20{i:03}', 'entity')
        node.abstract = 'Atlas'
        fs.nodes[node.uri] = node
    result = _prefetcher(fs, FakeVectorIndex(), FakeEmbedder())._prefetch('entity', _ctx())
    assert len(result.listed_uris) == 101
    assert 'Catalog status: complete' in ''.join(result.messages)


def test_catalog_exposes_scope_aliases_but_does_not_mark_nodes_read():
    fs = SummaryFS()
    uri = 'ctx://acct/users/user/memories/entities/atlas'
    node = _node(uri, 'entity')
    node.metadata = {'scope_ref': '北区项目', 'object_name': 'Atlas', 'routing_key_aliases': ['old_atlas']}
    fs.nodes[uri] = node
    result = _prefetcher(fs, FakeVectorIndex(), FakeEmbedder())._prefetch('entity', _ctx(), lazy=True)
    text = '\n'.join(result.messages)
    assert '北区项目' in text and 'Atlas' in text and 'old_atlas' in text
    assert 'Catalog status: complete' in text
    assert result.listed_uris == {uri}
    assert result.read_uris == set() and result.read_versions == {}


def test_catalog_is_character_bounded_and_reports_partial():
    fs = SummaryFS()
    for i in range(150):
        node = _node(f'ctx://acct/users/user/memories/entities/item_{i:03}', 'entity')
        node.abstract = 'description ' * 100
        fs.nodes[node.uri] = node
    result = _prefetcher(fs, FakeVectorIndex(), FakeEmbedder())._prefetch('entity', _ctx())
    assert len(''.join(result.messages)) <= 16000
    assert 'Catalog status: partial' in ''.join(result.messages)
    assert 0 < len(result.listed_uris) < 150


def test_catalog_paginates_and_skips_other_owner_and_nested_nodes():
    fs = SummaryFS()
    for i in range(102):
        node = _node(f'ctx://acct/users/user/memories/entities/n_{i:03}', 'entity')
        node.abstract = 'Atlas'
        fs.nodes[node.uri] = node
    other = _node('ctx://acct/users/other/memories/entities/private', 'entity')
    fs.nodes[other.uri] = other
    result = _prefetcher(fs, FakeVectorIndex(), FakeEmbedder())._prefetch('entity', _ctx())
    assert len(result.listed_uris) == 102
    assert len(fs.pages) == 2
    assert other.uri not in result.listed_uris


def test_duplicate_vector_levels_are_read_once_and_keep_identity_metadata():
    from core.models import SeedHit
    fs, vector = SummaryFS(), FakeVectorIndex()
    uri = 'ctx://acct/users/user/memories/entities/atlas'
    node = _node(uri, 'entity')
    node.metadata = {'version': 3, 'scope_ref': '北区项目', 'object_name': 'Atlas'}
    fs.nodes[uri] = node
    vector.hits = [SeedHit(uri=uri, score=.8, category='entity', abstract='Atlas', level=0),
                   SeedHit(uri=uri+'/content.md', score=.9, category='entity', abstract='Atlas', level=2)]
    result = _prefetcher(fs, vector, FakeEmbedder())._prefetch('entity', _ctx())
    assert sum('Content: entity content' in m for m in result.messages) == 1
    assert 'Scope: 北区项目' in ''.join(result.messages)
    assert result.read_versions == {uri: 3}


def test_long_query_is_bounded_covers_tail_and_is_shared_across_categories():
    embedder = FakeEmbedder()
    embedder.max_input_chars = 200
    text = 'Start ' + ('middle ' * 1500) + 'Tail atlas'
    _prefetcher(SummaryFS(), FakeVectorIndex(), embedder).prefetch_for_span(
        ['entity', 'preference', 'entity'], _ctx(), conversation_text=text)
    assert len(embedder.queries) <= 4
    assert all(len(q) <= 200 for q in embedder.queries)
    assert embedder.queries[-1].endswith('Tail atlas')
