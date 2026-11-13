"""SQL relocation preserves terminal evidence for the original write request."""

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import psycopg2
import pytest
import requests
from werkzeug.serving import make_server

from core.models import RequestContext
from index.outbox_worker import OutboxWorker
from providers.embedder.mock_embedder import MockEmbedder
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from session.sql_archive_store import SQLSessionArchiveStore
from tests.integration.test_lifecycle_batch_recovery_sql import sql_service
from tests.integration.test_occurrence_replay_sql import close_stores, stores


@pytest.mark.parametrize("phase", ["pending", "processing"])
def test_moved_projection_terminates_original_receipt(monkeypatch, phase):
    import server.app as app_module

    fs, outbox = stores()
    ctx = RequestContext("move-receipt-" + uuid4().hex, "reader", "agent", "source-session", "move-proof")
    archives = SQLSessionArchiveStore(os.environ["OGMEM_IDENTITY_TEST_DSN"])
    service, _, _ = sql_service(fs, outbox, ctx, archive_store=archives)
    service._get_outbox_store = lambda _fs: outbox
    monkeypatch.setattr(app_module, "_get_service", lambda: service)
    monkeypatch.setattr(app_module, "_build_authenticated_context", lambda _params: ctx)
    server = make_server("127.0.0.1", 0, app_module.app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    http = requests.Session()
    http.trust_env = False
    entered, release = threading.Event(), threading.Event()
    index = InMemoryVectorIndex(dimension=16)

    class ObservedEmbedder(MockEmbedder):
        def embed_texts(self, texts):
            entered.set()
            assert release.wait(20)
            return super().embed_texts(texts)

    worker = OutboxWorker(index, ObservedEmbedder(16), fs=fs, directory_summary_enabled=False)

    def request(method, path, payload=None):
        response = http.request(method, f"http://127.0.0.1:{server.server_port}" + path,
                                json=payload, timeout=20)
        print("HTTP " + json.dumps({"method": method, "path": path, "request": payload,
              "http_status": response.status_code, "response": response.json()}))
        if method == "GET":
            assert response.status_code == 200
        return response.json()

    def rows():
        with psycopg2.connect(os.environ["OGMEM_IDENTITY_TEST_DSN"]) as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT event_id,event_type,uri,status,payload FROM outbox_events "
                               "WHERE account_id=%s ORDER BY seq", (ctx.account_id,))
                result = [{"id": row[0], "type": row[1], "uri": row[2], "status": row[3], "payload": row[4]}
                          for row in cursor.fetchall()]
        print("SQL_EVENTS " + json.dumps(result))
        return result

    try:
        source = {"sessionId": ctx.session_id, "clientRequestId": "move-request", "wait": True,
                  "messages": [{"id": "msg-one", "role": "user", "content": "I repaired the receiver."}]}
        accepted = request("POST", "/api/v1/after_turn", source)
        path = f"/api/v1/sessions/{ctx.session_id}/write-requests/move-request"
        initial = request("GET", path)
        old_uri, old_id = next((uri, event_id) for uri, event_id in initial["outbox_events"]
                              if "/memories/events/" in uri)
        old_event = outbox.get_event(ctx.account_id, old_id)
        new_uri = old_uri + "-moved"
        with ThreadPoolExecutor(max_workers=1) as executor:
            running = None
            if phase == "processing":
                running = executor.submit(worker.run_once, outbox, [ctx.account_id],
                                          "moving-worker", [(old_uri, old_event)])
                assert entered.wait(15)
            before_move = rows()
            print("MOVE_REQUEST " + json.dumps({"method": "SQLContextFS.move_node",
                                                "from_uri": old_uri, "to_uri": new_uri}))
            fs.move_node(old_uri, new_uri, ctx)
            print("MOVE_RESPONSE " + json.dumps({"uri": fs.read_node(new_uri, ctx).uri}))
            after_move = rows()
            moved = request("GET", path)
            release.set()
            if running is not None:
                print("OLD_WORKER " + json.dumps(running.result(timeout=20)))
        print("PROJECTION " + json.dumps(worker.run_once(outbox, [ctx.account_id])))
        outbox.register_once(old_event, ctx)
        print("REPLAY " + json.dumps(worker.run_once(outbox, [ctx.account_id],
                                                    pending_events=[(old_uri, old_event)])))
        final_rows = rows()
        completed = request("GET", path)
        assert accepted["source_status"] == "ARCHIVED"
        assert initial["request_status"] == "PROJECTING"
        assert next(row["status"] for row in before_move if row["id"] == old_id) == phase.upper()
        replacements = [row for row in final_rows if row["id"] != old_id and row["uri"] in (old_uri, new_uri)]
        assert replacements and all(row["status"] == "DONE" for row in replacements)
        assert any(record.uri == new_uri for record in index._records.values())
        assert all(record.uri not in (old_uri, old_uri + "/content.md") for record in index._records.values())
        assert moved["request_status"] == completed["request_status"] == "NEEDS_CONFIRMATION"
        assert completed["terminal"] is True
        assert completed["retry_action"] == "manual_review"
        assert completed["source_status"] == "ARCHIVED"
        assert completed["projection_status"] == "FAILED"
        assert next(row["status"] for row in after_move if row["id"] == old_id) == "CANCELLED"
        assert next(row["status"] for row in final_rows if row["id"] == old_id) == "CANCELLED"
        assert next(row["payload"] for row in final_rows if row["id"] == old_id) == {"cancel_reason": "node_moved"}
    finally:
        release.set()
        service._extraction_retry.close(timeout=5)
        server.shutdown()
        server.server_close()
        thread.join(5)
        http.close()
        close_stores(fs, outbox)
        for connection in archives._pool:
            connection.close()
