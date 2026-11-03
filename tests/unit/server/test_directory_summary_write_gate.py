"""Directory summary configuration across the synchronous write path."""

import json
from unittest.mock import Mock

import pytest

from commit.outbox_store import OutboxStore
from commit.sql_outbox_store import SQLOutboxStore
from core.models import CandidateMemory, ContextNode, OutboxEvent, RequestContext, WritePlan
from providers.llm import MockLLM
from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("backend", ["agfs", "sql"])
@pytest.mark.parametrize("atomic", [False, True])
@pytest.mark.parametrize("action", ["create", "merge"])
def test_directory_summary_write_path(monkeypatch, enabled, backend, atomic, action):
    ctx = RequestContext(
        account_id="acct", user_id="u1", agent_id="a1", session_id="session", trace_id="trace"
    )
    node = ContextNode(
        uri="ctx://acct/users/u1/memories/preferences/coffee",
        context_type="MEMORY",
        category="preference",
        level=0,
        owner_space="user:u1",
        abstract="Likes coffee",
        overview="Coffee preference",
        content="Prefers coffee in the morning",
    )
    candidate = CandidateMemory(
        category=node.category,
        owner_scope="user",
        routing_key="coffee",
        abstract=node.abstract,
        overview=node.overview,
        content=node.content,
        confidence=0.9,
    )
    events = []
    operations = []

    class AtomicFS(Mock):
        def write_node_with_outbox(self, written_node, request_ctx, event):
            operations.append("write_node")
            events.append(event)

    fs = AtomicFS() if atomic else Mock()
    fs.write_node.side_effect = lambda *args: operations.append("write_node")

    def list_children(*args):
        operations.append("list_children")
        return [node.uri] * 20

    def read_node(*args):
        operations.append("read_node")
        return node

    fs.list_children.side_effect = list_children
    fs.read_node.side_effect = read_node
    if backend == "agfs":
        client = Mock()
        client.write.side_effect = lambda path, data: events.append(OutboxEvent(**json.loads(data)))
        outbox = OutboxStore(client=client, fs=fs)
        monkeypatch.setattr(outbox, "_write_event", lambda event, uri: events.append(event))
    else:
        outbox = SQLOutboxStore.__new__(SQLOutboxStore)
        outbox._fs = fs
        monkeypatch.setattr(outbox, "_insert_event", lambda event, account: events.append(event))

    service = MemoryService(
        config=OgMemConfig(
            provider="mock",
            storage_backend=backend,
            vector_db_type="memory",
            directory_summary_enabled=enabled,
        )
    )
    monkeypatch.setattr(service, "_get_context_fs", lambda: fs)
    monkeypatch.setattr(service, "_get_outbox_store", lambda fs: outbox)
    monkeypatch.setattr(service, "get_llm", lambda: MockLLM())
    monkeypatch.setattr(service, "_get_shared_vector_index", lambda: None)
    monkeypatch.setattr(service, "_get_shared_embedder", lambda: None)
    monkeypatch.setattr(service, "_start_outbox_worker", lambda: None)
    api = service.get_write_api()
    api._writer._identity_resolver = None
    plan = WritePlan(action=action, target_uri=node.uri)
    api._writer._policy_router.plan = Mock(return_value=plan)
    api._writer._archive_builder.build = Mock(return_value=node)

    result = api.write_memory(candidate, ctx)

    assert result == {"action": action, "target_uri": node.uri, "merged_fields": {}}
    expected_operations = ["write_node"]
    expected_events = ["UPSERT_CONTEXT"]
    if enabled:
        expected_operations += ["list_children"] + ["read_node"] * 20
        expected_events.append("UPSERT_DIRECTORY")
        assert events[-1].payload["child_abstracts"] == [node.abstract] * 20
    assert operations == expected_operations
    assert [event.event_type for event in events] == expected_events
