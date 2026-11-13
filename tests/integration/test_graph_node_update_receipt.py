"""Normal HTTP source update includes trigger-created graph projection receipts."""

import json
import os
import threading
from dataclasses import replace

import pytest
import requests
from werkzeug.serving import make_server

from commit.outbox_scope import capture_outbox_events, current_outbox_events
from core.models import CandidateMemory, RelationEdge
from index.outbox_worker import OutboxWorker
from service.api import MemoryWriteAPI
from tests.integration.test_semantic_context_graph import _graph_sql
from tests.integration.test_semantic_context_graph import graph_db as graph_db
from tests.unit.server.test_extraction_batch_recovery import NoModel
from tests.unit.server.test_lifecycle_batch_recovery import real_archives


def test_http_update_of_existing_relation_endpoint_tracks_trigger(graph_db, monkeypatch):
    import server.app as app_module

    ctx, fs, index, relations, projections, outbox, worker, _, profile = graph_db
    seed = MemoryWriteAPI(fs, NoModel(), outbox, directory_summary_enabled=False)
    source = CandidateMemory("entity", "user", "project_atlas", "Atlas uses Beacon.",
                             "Atlas uses Beacon.", "Atlas uses Beacon.", .95)
    target = CandidateMemory("entity", "user", "project_beacon", "Beacon stores backups locally.",
                             "Beacon stores backups locally.", "Beacon stores backups locally.", .95)
    source_uri = seed.write_memory(source, ctx)["target_uri"]
    target_uri = seed.write_memory(target, ctx)["target_uri"]
    relations.upsert_edges([RelationEdge(source_uri, target_uri, "depends on", 1.,
                                         "Atlas stores backups in Beacon.")], ctx)
    for _ in range(4):
        initial_drain = worker.run_once(outbox, [ctx.account_id])
        assert initial_drain["failed"] == 0
    before = _graph_sql(fs, ctx, "SELECT event_id,uri,status FROM outbox_events "
                        "WHERE account_id=%s AND event_type='PROJECT_GRAPH'", (ctx.account_id,))
    assert before and all(row[2] == "DONE" for row in before)
    old = fs.read_node(target_uri, ctx)
    message = "Beacon now stores encrypted backups in an offsite vault. Atlas still uses Beacon."
    updated = replace(target, abstract=message, overview=message, content=message,
                      target_uri=target_uri, expected_version=old.metadata["version"])
    service, writer, manager = real_archives(fs, ctx, entries=[updated], outbox=outbox)
    service._cfg.storage_backend = "sql"
    service._cfg.sql_connection_string = os.environ["GRAPH_TEST_DSN"]
    service._cfg.graph_backend = "sql"
    service._get_outbox_store = lambda _fs: outbox
    monkeypatch.setattr(app_module, "_get_service", lambda: service)
    monkeypatch.setattr(app_module, "_build_authenticated_context", lambda _params: ctx)
    server = make_server("127.0.0.1", 0, app_module.app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    http = requests.Session()
    http.trust_env = False

    def request(method, path, body=None):
        response = http.request(method, f"http://127.0.0.1:{server.server_port}" + path,
                                json=body, timeout=30)
        print("HTTP_EXCHANGE " + json.dumps({"method": method, "path": path, "request": body,
              "http_status": response.status_code, "response": response.json()}, default=str))
        return response

    try:
        body = {"sessionId": ctx.session_id, "clientRequestId": "beacon-location-update",
                "wait": True, "forceExtract": True,
                "messages": [{"id": "msg-one", "role": "user", "content": message}]}
        response = request("POST", "/api/v1/after_turn", body)
        receipt = response.json()
        graph = _graph_sql(fs, ctx, "SELECT event_id,uri,status,payload FROM outbox_events "
                           "WHERE account_id=%s AND event_type='PROJECT_GRAPH'", (ctx.account_id,))
        new_graph = [row for row in graph if row[0] not in {old_row[0] for old_row in before}]
        stored = fs.read_node(target_uri, ctx)
        print("SQL_TRIGGER " + json.dumps({"before": before, "after": graph,
              "node_uri": target_uri, "old_version": old.metadata["version"],
              "new_version": stored.metadata["version"], "new_content": stored.content,
              "receipt_refs": receipt.get("outbox_events")}, default=str))
        ordinary = OutboxWorker(index, worker._embedder, fs=fs, directory_summary_enabled=False)
        for _ in range(4):
            drained = ordinary.run_once(outbox, [ctx.account_id])
            print("ORDINARY_DRAIN " + json.dumps(drained))
        observed = request("GET", f"/api/v1/sessions/{ctx.session_id}/write-requests"
                           "?clientRequestId=beacon-location-update").json()
        final_graph = _graph_sql(fs, ctx, "SELECT event_id,status FROM outbox_events "
                                 "WHERE account_id=%s AND event_type='PROJECT_GRAPH'", (ctx.account_id,))
        print("GRAPH_PENDING " + json.dumps(final_graph))
        assert response.status_code == 202
        assert stored.content == message
        assert stored.metadata["version"] == old.metadata["version"] + 1
        assert len(new_graph) == 1
        assert {(row[1], row[0]) for row in new_graph} <= {tuple(ref) for ref in receipt["outbox_events"]}
        assert observed["request_status"] == "PROJECTING"
        assert observed["projection_status"] == "PENDING"
        graph_drain = worker.run_once(outbox, [ctx.account_id])
        print("GRAPH_DRAIN " + json.dumps(graph_drain))
        completed = request("GET", f"/api/v1/sessions/{ctx.session_id}/write-requests"
                            "?clientRequestId=beacon-location-update").json()
        assert graph_drain["failed"] == 0
        assert completed["request_status"] == "SUCCEEDED"
        assert completed["projection_status"] == "DONE"
        assert outbox.get_event_status(new_graph[0][1], new_graph[0][0]) == "DONE"
        assert _graph_sql(fs, ctx, "SELECT status FROM edge_projection_jobs WHERE account_id=%s",
                          (ctx.account_id,)) == [("COMPLETE",)]
    finally:
        service._extraction_retry.close(timeout=5)
        server.shutdown()
        server.server_close()
        thread.join(5)
        http.close()


@pytest.mark.parametrize("write_method", ["write_node", "write_node_with_outbox"])
@pytest.mark.parametrize("versioned", [True, False], ids=["cas", "upsert"])
def test_node_update_tracks_all_committed_graph_wakeups(graph_db, write_method, versioned):
    ctx, fs, _, relations, _, outbox, worker, nodes, _ = graph_db
    target = nodes[1]
    relations.upsert_edges([
        RelationEdge(source.uri, target.uri, "depends on", 1.0, "Project dependency")
        for source in (nodes[0], nodes[2])
    ], ctx)
    assert worker.run_once(outbox, [ctx.account_id])["failed"] == 0
    node = fs.read_node(target.uri, ctx)
    node.content = "The shared backup service now uses an offsite vault."
    if versioned:
        node.metadata["_expected_version"] = node.metadata["version"]
    else:
        node.metadata["version"] += 1

    @capture_outbox_events
    def update():
        if write_method == "write_node_with_outbox":
            fs.write_node_with_outbox(node, ctx, outbox.build_write_event(node))
        else:
            fs.write_node(node, ctx)
        return current_outbox_events(ctx.account_id)

    references = update()
    events = _graph_sql(fs, ctx, "SELECT uri,event_id FROM outbox_events WHERE account_id=%s "
                        "AND event_type='PROJECT_GRAPH' AND status='PENDING'", (ctx.account_id,))
    print("NODE_WRITE_RECEIPTS " + json.dumps({"method": write_method, "versioned": versioned,
          "references": references, "durable_events": events}))
    assert len(events) == 2
    assert set(references) == set(events)
    assert fs.read_node(target.uri, ctx).content == node.content
    for _ in range(2):
        assert worker.run_once(outbox, [ctx.account_id])["failed"] == 0
    assert all(outbox.get_event_status(uri, event_id) == "DONE" for uri, event_id in references)
    assert _graph_sql(fs, ctx, "SELECT status FROM edge_projection_jobs WHERE account_id=%s",
                      (ctx.account_id,)) == [("COMPLETE",), ("COMPLETE",)]


def test_trigger_receipt_uses_current_committed_transaction_on_reused_connection(graph_db):
    ctx, fs, _, relations, _, outbox, worker, nodes, _ = graph_db
    target = nodes[1]
    relations.upsert_edges([RelationEdge(nodes[0].uri, target.uri, "depends on", 1.0,
                                         "Project dependency")], ctx)
    assert worker.run_once(outbox, [ctx.account_id])["failed"] == 0
    conn = fs._get_connection()
    backend_pid = conn.get_backend_pid()
    try:
        fs.bind_tenant(conn, ctx.account_id)
        with conn.cursor() as cur:
            cur.execute("UPDATE context_nodes SET content=%s,version=version+1 WHERE uri=%s",
                        ("Draft backup location.", target.uri))
            cur.execute("SELECT current_setting('ogmemory.node_write_events')")
            draft = json.loads(cur.fetchone()[0])
        conn.rollback()
    finally:
        fs._return_connection(conn)

    @capture_outbox_events
    def update():
        node = fs.read_node(target.uri, ctx)
        assert node.content == target.content
        fs.write_node(node, ctx)
        clean_references = current_outbox_events(ctx.account_id)
        assert clean_references == []
        node.content = "Committed backup location."
        node.metadata["_expected_version"] = node.metadata["version"]
        fs.write_node(node, ctx)
        return current_outbox_events(ctx.account_id)

    references = update()
    conn = fs._get_connection()
    try:
        assert conn.get_backend_pid() == backend_pid
        with conn.cursor() as cur:
            cur.execute("SELECT current_setting('ogmemory.node_write_events')")
            assert cur.fetchone()[0] == ""
        conn.rollback()
    finally:
        fs._return_connection(conn)
    events = _graph_sql(fs, ctx, "SELECT uri,event_id FROM outbox_events WHERE account_id=%s "
                        "AND event_type='PROJECT_GRAPH' AND status='PENDING'", (ctx.account_id,))
    print("REUSED_TRANSACTION " + json.dumps({"rolled_back": draft, "references": references,
          "durable_events": events, "backend_pid": backend_pid}))
    assert len(events) == 1
    assert references == events
    assert draft["event_ids"][0] != references[0][1]
    assert worker.run_once(outbox, [ctx.account_id])["failed"] == 0
    assert outbox.get_event_status(*references[0]) == "DONE"
