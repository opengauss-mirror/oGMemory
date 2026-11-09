"""Request catalogs traverse Chinese AGFS identities in physical order."""

import json
import os
from types import SimpleNamespace
from urllib.parse import quote
from uuid import uuid4

import pytest
from pyagfs import AGFSClient

from core.models import ContextNode, RequestContext
from extraction.node_processing import NodeProcessor
from fs.agfs_adapter.agfs_context_fs import AGFSContextFS


def test_chinese_identity_catalog_traverses_three_pages(monkeypatch):
    address = os.environ.get("OGMEM_AGFS_TEST_URL")
    if not address:
        pytest.skip("set OGMEM_AGFS_TEST_URL to a disposable AGFS server")
    client = AGFSClient(address)
    client.session.trust_env = False
    client.session.proxies.clear()
    fs = AGFSContextFS(client, mount_prefix=os.environ.get("OGMEM_AGFS_TEST_MOUNT", "/local"))
    ctx = RequestContext("catalog-" + uuid4().hex, "reader", "agent", "session", "trace")
    directory = f"ctx://{ctx.account_id}/users/{ctx.user_id}/memories/entities/"
    expected = []
    for index in range(49):
        name = f"项目{index:02d}"
        uri = directory + quote(name, safe="")
        fs.write_node(ContextNode(uri, "MEMORY", "entity", 1, "reader", name, name,
                                  f"# {name}\n\n## Identity\n- 项目使用 uv。",
                                  {"routing_key": name, "object_name": name}), ctx)
        expected.append(uri)
    pages = []
    original = AGFSContextFS.list_identity_node_summaries_page

    def record_page(self, target, request, *, limit, after_uri):
        result = original(self, target, request, limit=limit, after_uri=after_uri)
        pages.append({"cursor": after_uri, "uris": [node.uri for node in result]})
        return result

    monkeypatch.setattr(AGFSContextFS, "list_identity_node_summaries_page", record_page)
    processor = NodeProcessor(SimpleNamespace(fs=fs), None)
    catalog = list(processor._catalog(directory, ctx))
    actual = [node.uri for node in catalog]
    assert actual == expected
    assert len(set(actual)) == 49
    assert [len(page["uris"]) for page in pages] == [24, 24, 1]
    assert [page["cursor"] for page in pages] == [
        None, directory + "项目23", directory + "项目47",
    ]
    print("AGFS_CATALOG " + json.dumps({"pages": pages, "unique_nodes": len(set(actual))}, ensure_ascii=False))
