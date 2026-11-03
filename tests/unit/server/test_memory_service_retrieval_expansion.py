"""Real read-pipeline behavior for differently configured service instances."""

from __future__ import annotations

import pytest

from core.models import IndexRecord, RequestContext
from providers.embedder.mock_embedder import MockEmbedder
from providers.unified_config import OgMemConfig
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from server.memory_service import MemoryService

QUERY = "remember preferences"
DIRECTORY_URI = "ctx://acct/users/user/memories/preferences/"


class TrackingIndex(InMemoryVectorIndex):
    def __init__(self):
        super().__init__(dimension=8)
        self.child_search_count = 0

    def search_children(self, *args, **kwargs):
        self.child_search_count += 1
        return super().search_children(*args, **kwargs)


def create_read_api(config, monkeypatch):
    service = MemoryService(config=config)
    service._embedder = MockEmbedder(dimension=8)
    index = TrackingIndex()
    index.upsert([
        IndexRecord(
            id="directory-overview",
            uri=DIRECTORY_URI,
            level=1,
            text=QUERY,
            filters={"account_id": "acct", "owner_space": "user:user"},
            metadata={
                "context_type": "MEMORY",
                "has_overview": True,
                "_embedding": service._embedder.embed_texts([QUERY])[0],
            },
        ),
    ])
    service._vector_index = index
    # These providers need external storage. The retrieval components remain real.
    monkeypatch.setattr(service, "_get_context_fs", lambda: None)
    monkeypatch.setattr(service, "_get_relation_store", lambda: None)
    return service.get_read_api(start_outbox_worker=False), index


def search(read_api):
    return read_api.search_memory(
        QUERY,
        RequestContext(
            account_id="acct", user_id="user", agent_id="agent",
            session_id="session", trace_id="trace",
            visible_owner_spaces=("user:user", "agent:agent"),
        ),
        context_type="MEMORY",
    )


@pytest.mark.parametrize("configured", [None, True, False])
def test_service_hierarchy_setting_controls_real_read_pipeline(monkeypatch, configured):
    config = (
        OgMemConfig() if configured is None
        else OgMemConfig(retrieval_expand_hierarchy=configured)
    )
    read_api, index = create_read_api(config, monkeypatch)

    result = search(read_api)

    if configured is False:
        assert result.hits == []
        assert index.child_search_count == 0
    else:
        assert result.hits == []
        assert index.child_search_count > 0


def test_service_instances_keep_independent_hierarchy_settings(tmp_path, monkeypatch):
    config_path = tmp_path / "ogmem.yaml"
    config_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("OGMEM_RETRIEVAL_EXPAND_HIERARCHY", "false")
    direct_api, direct_index = create_read_api(OgMemConfig.load(str(config_path)), monkeypatch)
    monkeypatch.setenv("OGMEM_RETRIEVAL_EXPAND_HIERARCHY", "true")
    expanded_api, expanded_index = create_read_api(OgMemConfig.load(str(config_path)), monkeypatch)

    for _ in range(2):
        assert search(direct_api).hits == []
        assert search(expanded_api).hits == []
    assert direct_index.child_search_count == 0
    assert expanded_index.child_search_count > 0


def test_false_service_returns_existing_l2_without_child_search(monkeypatch):
    read_api, index = create_read_api(OgMemConfig(retrieval_expand_hierarchy=False), monkeypatch)
    index.upsert([IndexRecord(
        id="content", uri=DIRECTORY_URI + "coding/content.md", level=2, text=QUERY,
        filters={"account_id": "acct", "owner_space": "user:user"},
        metadata={"context_type": "MEMORY", "category": "preference", "_embedding": MockEmbedder(dimension=8).embed_texts([QUERY])[0]},
    )])
    assert [block.uri for block in search(read_api).hits] == [DIRECTORY_URI + "coding/content.md"]
    assert index.child_search_count == 0
