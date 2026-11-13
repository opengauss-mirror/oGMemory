"""Compaction usage failures remain queryable until explicit recovery."""

import json
from dataclasses import replace
from threading import Thread
from types import SimpleNamespace

import pytest
import requests
from werkzeug.serving import make_server

from providers.token_tracker import TokenTracker
from service.extraction_checkpoint import EXTENSION_KEY
from tests.integration.test_agfs_node_snapshot_reads import _runtime
from tests.integration.test_agfs_node_snapshot_reads import native_agfs as native_agfs
from tests.unit.server.test_extraction_batch_recovery import candidate
from tests.unit.server.test_lifecycle_batch_recovery import real_archives


@pytest.mark.parametrize("entry", ["compact", "prepare_compaction"])
def test_compaction_usage_failure_receipt_survives_restart(native_agfs, monkeypatch, entry):
    import server.app as app_module

    client, fs, ctx = _runtime(native_agfs[0], "compaction-usage")
    tracker = TokenTracker("local-fixture")
    usage = []

    def unavailable(event):
        raise OSError("fixture usage store unavailable")

    tracker.set_usage_store(SimpleNamespace(record=unavailable))

    def extract(**kwargs):
        tracker.record_llm(5, 2)
        return [replace(candidate(), provenance_ids=[f"prov:1:archive:{kwargs['archive_id']}:source-one"])]

    def service_for_request():
        service, _, manager = real_archives(fs, ctx, extract=extract)
        service._default_account_id = ctx.account_id
        service._default_user_id = ctx.user_id
        service._default_agent_id = ctx.agent_id
        service._agfs_base_url = native_agfs[0]
        service._cfg.after_turn_threshold = 100000
        monkeypatch.setattr(app_module, "_get_service", lambda: service)
        return service, manager

    service, manager = service_for_request()
    server = make_server("127.0.0.1", 0, app_module.app)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    http = requests.Session()
    http.trust_env = False

    def request(method, path, payload=None):
        response = http.request(method, f"http://127.0.0.1:{server.server_port}{path}",
                                json=payload, timeout=20)
        body = response.json()
        print("HTTP " + json.dumps({"method": method, "path": path, "request": payload,
                                   "http_status": response.status_code, "response": body}))
        return response.status_code, body

    session = {"sessionId": ctx.session_id}
    path = f"/api/v1/sessions/{ctx.session_id}/write-requests/usage-source"
    try:
        status, accepted = request("POST", "/api/v1/after_turn", {
            **session, "clientRequestId": "usage-source", "wait": True,
            "messages": [{"id": "source-one", "role": "user", "content": "I repaired the radio receiver."}],
        })
        assert (status, accepted["source_status"]) == (202, "BUFFERED")
        status, failure = request("POST", f"/api/v1/{entry}", session)
        assert (status, failure["request_status"]) == (500, "FAILED")
        assert failure["error_detail"]["code"] == "MODEL_USAGE_PERSISTENCE_FAILED"
        state_uri = manager._session_state.state_uri(ctx.account_id, ctx.session_id)
        batch = json.loads(fs.read_node(state_uri, ctx).content)["runtime_extensions"][EXTENSION_KEY]["batch"]
        print("PERSISTED " + json.dumps({"archive_id": batch["archive_id"], "lifecycle": batch["lifecycle"]}))
        service._extraction_retry.close(timeout=5)
        service, _ = service_for_request()
        status, observed = request("GET", path)
        assert (status, observed["request_status"], observed["retry_action"]) == (200, "FAILED", "manual_review")
        assert observed["error_detail"] == {**failure["error_detail"], "request_id": "usage-source"}
        assert observed["archive_id"] == failure["archive_id"] == batch["archive_id"]
        assert batch["lifecycle"]["last_error"]["status"] == "failed"

        tracker.set_usage_store(SimpleNamespace(record=usage.append))
        status, prepared = request("POST", "/api/v1/prepare_compaction", session)
        assert (status, prepared["prepared"]) == (200, True)
        status, resumed = request("GET", path)
        assert (status, resumed["request_status"], resumed["archive_id"]) == (200, "RUNNING", batch["archive_id"])
        status, compacted = request("POST", "/api/v1/compact", {**session, "prepareToken": prepared["prepareToken"]})
        assert (status, compacted["compacted"]) == (200, True)
        status, completed = request("GET", path)
        assert (status, completed["request_status"], completed["archive_id"]) == (200, "SUCCEEDED", batch["archive_id"])
        assert len(usage) == 1
    finally:
        service._extraction_retry.close(timeout=5)
        http.close()
        server.shutdown()
        server.server_close()
        thread.join(5)
