"""Only paused graph tasks are excluded from ordinary index readiness."""
from unittest.mock import Mock

import pytest

from commit.sql_outbox_store import SQLOutboxStore
from core.models import RequestContext
from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService


@pytest.mark.parametrize("graph,pending,failed,idle", [
    (False, 0, 0, True), (True, 0, 0, False),
    (False, 1, 0, False), (False, 0, 1, False),
])
def test_wait_ignores_only_paused_graph_without_hiding_public_counts(monkeypatch, graph, pending, failed, idle):
    service = MemoryService(config=OgMemConfig(provider="mock", vector_db_type="memory"))
    outbox = SQLOutboxStore.__new__(SQLOutboxStore)
    outbox.count_unfinished = Mock(side_effect=lambda account_id, **kwargs: {
        "total": pending + (0 if kwargs.get("exclude_event_types") else 1),
        "failed": failed,
    })
    monkeypatch.setattr(service, "_get_context_fs", lambda: Mock())
    monkeypatch.setattr(service, "_get_outbox_store", lambda _: outbox)
    monkeypatch.setattr(service, "_graph_enabled", lambda: graph)
    ctx = RequestContext("account", "user", "agent", "session", "trace")
    assert service._wait_for_outbox_idle(ctx, timeout_seconds=0)["idle"] is idle
    outbox.count_unfinished.assert_called_once_with(
        account_id="account", **({"exclude_event_types": ["PROJECT_GRAPH"]} if not graph else {}))
    assert service._count_outbox_unfinished(ctx) == {"total": pending + 1, "failed": failed}
    assert outbox.count_unfinished.call_args.kwargs == {"account_id": "account"}
