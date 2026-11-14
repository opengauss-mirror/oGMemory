"""Shared retrieval ranking and page traversal with authorized catalog fixtures."""

from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

from flask import Flask

from core.models import RequestContext
from server.space_management import create_shared_search_blueprint
from server.space_repository import SpaceRepository
from service.shared_memory import SharedMemoryRetrievalService


def item(name, content=None, kind='experience'):
    return {'item_id': name, 'space_id': 'ops', 'uri': f'ctx://acme/spaces/ops/memories/{name}',
            'kind': kind, 'content': content or f'连接池排障步骤 {name}', 'abstract': f'摘要 {name}',
            'rule_id': 'rule', 'rule_rev': 1, 'version': 1, 'updated_at': '2026-11-12T00:00:00Z'}


class Catalog(SpaceRepository):
    def __init__(self, rows, keywords):
        self.rows = {row['uri']: row for row in rows}
        self.keywords = keywords
        self._cursor_key = b'unit-search-cursor'
        self.content_store = object()
        self.hydrated = []

    @contextmanager
    def transaction(self, account_id, **kwargs):
        yield Mock(fetchone=lambda: {'authz_rev': 4})

    def _identity(self, cur, ctx):
        return {'role': 'user', 'status': 'active'}

    def visible_items(self, ctx, spaces, *, uris=None, query=None, kind=None, limit=100, hydrate=True):
        selected = self.keywords if query else (list(self.rows) if uris is None else uris)
        rows = [deepcopy(self.rows[uri]) for uri in selected if uri in self.rows
                and (kind is None or self.rows[uri]['kind'] == kind)][:limit]
        if hydrate:
            self.hydrated.append([row['item_id'] for row in rows])
        return rows


def context():
    return RequestContext('acme', 'alice', '', '', 'shared-search-test')


def test_hybrid_rrf_merges_keyword_and_semantic_candidates():
    rows = [item(name) for name in ('both', 'keyword', 'semantic')]
    catalog = Catalog(rows, [rows[0]['uri'], rows[1]['uri']])
    index = Mock(search_by_vector=Mock(return_value=[
        SimpleNamespace(uri=rows[2]['uri'], score=0.9),
        SimpleNamespace(uri=rows[0]['uri'], score=0.8),
        SimpleNamespace(uri=rows[0]['uri'] + '/content.md', score=0.7),
    ]))
    embedder = Mock(embed_texts=Mock(return_value=[[0.1, 0.2]]))
    search = SharedMemoryRetrievalService(catalog, index, embedder)
    result = search.search(context(), '连接池', ['ops'], mode='hybrid', limit=3)
    assert [row['item_id'] for row in result['items']] == ['both', 'semantic', 'keyword']
    assert result['total'] == 3
    assert result['mode'] == 'hybrid'
    assert result['items'][0]['score'] > result['items'][1]['score']
    assert catalog.hydrated == [['both', 'semantic', 'keyword']]
    assert index.search_by_vector.call_args.kwargs['filters'] == {
        'account_id': 'acme', 'owner_space': ['space:ops'], 'category': 'experience'}


def test_semantic_search_keeps_vector_order_and_best_level_score():
    rows = [item(name) for name in ('a', 'b')]
    catalog = Catalog(rows, [rows[0]['uri']])
    index = Mock(search_by_vector=Mock(return_value=[
        SimpleNamespace(uri=rows[0]['uri'], score=0.5),
        SimpleNamespace(uri=rows[1]['uri'], score=0.9),
        SimpleNamespace(uri=rows[1]['uri'] + '/content.md', score=0.8),
    ]))
    search = SharedMemoryRetrievalService(catalog, index, Mock(embed_texts=Mock(return_value=[[1.0]])))
    result = search.search(context(), '连接池', ['ops'], mode='semantic')
    assert [(row['item_id'], row['score']) for row in result['items']] == [('b', 0.9), ('a', 0.5)]
    assert result['next_cursor'] is None


def test_keyword_pages_hydrate_the_selected_page_and_traverse_budget():
    rows = [item(str(index), content='排查连接池。' * 30) for index in range(3)]
    catalog = Catalog(rows, [row['uri'] for row in rows])
    search = SharedMemoryRetrievalService(catalog)
    arguments = {'query': '连接池', 'space_ids': ['ops'], 'mode': 'keyword',
                 'top_k': 100, 'limit': 2, 'max_context_tokens': 128}
    pages = []
    cursor = None
    for _ in rows:
        page = search.search(context(), **arguments, cursor=cursor)
        pages.append(page)
        cursor = page['next_cursor']
    assert [[row['item_id'] for row in page['items']] for page in pages] == [['0'], ['1'], ['2']]
    assert [page['total'] for page in pages] == [3, 3, 3]
    assert pages[-1]['next_cursor'] is None
    assert catalog.hydrated == [['0', '1'], ['1', '2'], ['2']]
    assert all(sum(len(row['content'].encode()) for row in page['items']) <= 128 for page in pages)


def test_http_shared_search_accepts_modes_and_signed_page_cursor():
    rows = [item(str(index)) for index in range(3)]
    retrieval = SharedMemoryRetrievalService(Catalog(rows, [row['uri'] for row in rows]))
    auth = Mock(role_control_active=lambda: True, resolve_identity=lambda headers: object())
    service = Mock(get_auth_service=lambda: auth, build_context=lambda params, identity: context(),
                   search_shared_memory=retrieval.search)
    app = Flask(__name__)
    app.register_blueprint(create_shared_search_blueprint(lambda: service, lambda exc: ({'error': str(exc)}, 500)))
    client = app.test_client()
    body = {'query': '连接池', 'space_ids': ['ops'], 'mode': 'keyword', 'top_k': 100, 'limit': 2}
    first = client.post('/api/spaces/search', json=body)
    assert first.status_code == 200
    first = first.get_json()
    second = client.post('/api/spaces/search', json={**body, 'cursor': first['next_cursor']})
    assert second.status_code == 200
    second = second.get_json()
    assert first['mode'] == second['mode'] == 'keyword'
    assert first['total'] == second['total'] == 3
    assert [row['item_id'] for row in first['items'] + second['items']] == ['0', '1', '2']
    assert second['next_cursor'] is None
