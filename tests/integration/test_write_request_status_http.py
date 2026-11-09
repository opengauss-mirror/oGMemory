"""HTTP status reads retain the identity of a durably completed write."""

import pytest

from commit.outbox_scope import capture_outbox_events
from core.models import RequestContext
from providers.unified_config import OgMemConfig
from server.auth import AuthService
from server.memory_service import MemoryService
from service.api import MemoryWriteAPI
from service.extraction_checkpoint import ExtractionBatchJournal
from session.session_manager import SessionManager
from tests.unit.commit.test_request_outbox_events import make_runtime
from tests.unit.server.test_extraction_batch_recovery import NoModel


@pytest.mark.parametrize("identity_source,role_control", [
    ("headers", False), ("headers", True), ("camel", False), ("snake", False),
])
@capture_outbox_events
def test_completed_write_status_uses_http_identity_after_service_restart(monkeypatch, identity_source, role_control):
    import server.app as app_module

    _, fs, store = make_runtime()
    ctx = RequestContext(account_id="status-account", user_id="reader", agent_id="main",
                         session_id="status-session", trace_id="status-trace")
    manager = SessionManager(get_context_fs=lambda: fs, session_state_bridge_enabled=False)
    source = {"role": "user", "content": "The receiver was repaired."}
    manager.add_message(ctx.session_id, source["role"], source["content"], ctx, message_id="source-one")
    buffer = manager.get_or_create(ctx.session_id, ctx)
    journal = ExtractionBatchJournal(manager, buffer, ctx, list(buffer.messages), archive_id="status-archive",
        messages=[source], session_time=None, session_summary="", tool_stats_text="", client_request_id="write-one")
    writer = MemoryWriteAPI(fs, NoModel(), store, directory_summary_enabled=False)

    @capture_outbox_events
    def complete_write():
        archived = writer.write_raw_chunk([source], ctx, session_id=ctx.session_id,
                                         candidate_checkpoint=journal.checkpoint("raw:0"))
        return journal.finish(outcome={"source_status": "RAW_ARCHIVED", "semantic_status": "COMPLETE",
                                       "raw_source_uris": [archived["target_uri"]]})

    receipt = complete_write()
    for uri, event in store.list_pending_events([tuple(reference) for reference in receipt["outbox_events"]]):
        store.mark_done(event, uri)
    cfg = OgMemConfig(account_id="bootstrap-account", user_id="bootstrap", agent_id="",
                      session_state_bridge_enabled=False, role_control_enabled=role_control,
                      root_api_key="status-test-root")
    service = MemoryService(config=cfg)
    service._auth = AuthService(cfg, None)
    service._session_mgr = SessionManager(get_context_fs=lambda: fs, session_state_bridge_enabled=False)
    service._get_outbox_store = lambda _fs: store
    monkeypatch.setattr(app_module, "_get_service", lambda: service)
    headers, query = {}, {}
    if identity_source == "headers":
        headers = {"X-Account-ID": ctx.account_id, "X-User-ID": ctx.user_id, "X-Agent-ID": ctx.agent_id}
    else:
        names = ("accountId", "userId", "agentId") if identity_source == "camel" else ("account_id", "user_id", "agent_id")
        query = dict(zip(names, (ctx.account_id, ctx.user_id, ctx.agent_id)))
    if role_control:
        headers["X-API-Key"] = "status-test-root"
    response = app_module.app.test_client().get(
        f"/api/v1/sessions/{ctx.session_id}/write-requests/write-one", headers=headers, query_string=query)

    assert response.status_code == 200
    assert response.json["request_id"] == "write-one"
    assert response.json["request_status"] == "SUCCEEDED"
    assert response.json["projection_status"] == "DONE"
    assert response.json["raw_source_uris"] == receipt["raw_source_uris"]
