"""SQL graph wakeups participate in the receipt of their source write."""

import json
import os
import re
from unittest.mock import Mock

import pytest

from commit.outbox_scope import capture_outbox_events, current_outbox_events
from core.models import RelationEdge
from fs.sql_adapter.session_mutation import active_session_guard
from index.outbox_worker import OutboxWorker
from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService
from service.api import MemoryWriteAPI
from service.extraction_checkpoint import EXTENSION_KEY
from tests.integration.test_semantic_context_graph import _graph_sql
from tests.integration.test_semantic_context_graph import graph_db as graph_db
from tests.unit.server.test_write_receipt_transitions import http_client, record


def graph_llm():
    llm = Mock()
    llm.complete.return_value = "Projectx depends on Projecty."
    llm.complete_with_tools.return_value = [
        {
            "tool": "extract_entity",
            "input": {
                "routing_key": name,
                "abstract": "Project " + name,
                "overview": "Organization project " + name,
                "content": "Details for project " + name,
                "confidence": 0.95,
                "update_activation": "Organization project " + name + " uses a database.",
            },
        }
        for name in ("projectx", "projecty")
    ]

    def response(prompt, schema):
        properties = schema.get("properties", {})
        if "spans" in properties:
            return {
                "spans": [
                    {"start": 0, "end": 0, "categories": ["entity"], "reason": "Project facts"}
                ]
            }
        if "groups" in properties:
            return {"groups": [[0], [1]]}
        if "relations" in properties:
            block = prompt.split("NEWLY EXTRACTED MEMORY NODES:", 1)[1].split(
                "POTENTIALLY RELEVANT EXISTING MEMORY NODES:", 1
            )[0]
            nodes = json.loads(re.search(r"^(\[.*\])$", block, re.MULTILINE).group(1))
            return {
                "relations": [
                    {
                        "source_node_ref": nodes[0]["node_ref"],
                        "target_node_ref": nodes[1]["node_ref"],
                        "relation": "depends on",
                        "evidence_ids": nodes[0]["source_ids"] + nodes[1]["source_ids"],
                    }
                ]
            }
        return {key: "Projectx depends on Projecty." for key in properties}

    llm.complete_json.side_effect = response
    return llm


def _graph_runtime(graph_db, monkeypatch, *, nodes_only=False):
    ctx, fs, index, relations, projections, outbox, graph_worker, _, profile = graph_db
    llm = graph_llm()
    service = MemoryService(
        OgMemConfig(
            storage_backend="sql",
            graph_backend="sql",
            sql_connection_string=os.environ["GRAPH_TEST_DSN"],
            graph_edge_embedding_profile="test",
            session_state_bridge_enabled=False,
            after_turn_threshold=1,
            scene_enabled=False,
            directory_summary_enabled=False,
            dual_pass_enabled=False,
        )
    )
    service._write_api = MemoryWriteAPI(
        fs,
        llm,
        outbox,
        vector_index=index,
        embedder=graph_worker._embedder,
        directory_summary_enabled=False,
        graph_enabled=True,
        relation_store=relations,
        dual_pass_enabled=False,
    )
    monkeypatch.setattr(service, "get_llm", lambda: llm)
    monkeypatch.setattr(service, "_get_context_fs", lambda: fs)
    monkeypatch.setattr(service, "_get_outbox_store", lambda _fs: outbox)
    monkeypatch.setattr(service, "_get_shared_projection_index", lambda: index)
    monkeypatch.setattr(service, "_get_shared_embedder", lambda: graph_worker._embedder)
    monkeypatch.setattr(service, "_graph_projection_components", lambda: (projections, profile))
    if nodes_only:
        node_worker = OutboxWorker(
            index, graph_worker._embedder, fs=fs, directory_summary_enabled=False
        )
        production_drain = service.drain_outbox_sync

        def drain_nodes(**kwargs):
            if active_session_guard() is not None:
                return production_drain(**kwargs)
            return node_worker.run_once(outbox, [ctx.account_id])

        monkeypatch.setattr(service, "drain_outbox_sync", drain_nodes)
    return service, http_client(monkeypatch, service, ctx)


@pytest.mark.parametrize("projection_phase", ["request_drain", "nodes_first"])
def test_sql_graph_event_is_part_of_the_public_write_receipt(
    graph_db, monkeypatch, projection_phase
):
    ctx, fs, _, _, _, outbox, graph_worker, _, _ = graph_db
    service, http = _graph_runtime(graph_db, monkeypatch, nodes_only=projection_phase == "nodes_first")
    request = {
        "sessionId": ctx.session_id,
        "clientRequestId": "project-dependency",
        "wait": True,
        "forceExtract": True,
        "messages": [{"id": "turn", "role": "user", "content": "Projectx depends on Projecty."}],
    }
    try:
        accepted = http.post("/api/v1/after_turn", json=request)
        record("POST", "/api/v1/after_turn", request, accepted)
        path = f"/api/v1/sessions/{ctx.session_id}/write-requests"
        observed = http.get(path, query_string={"clientRequestId": "project-dependency"})
        record("GET", path, {"clientRequestId": "project-dependency"}, observed)
        events = _graph_sql(
            fs,
            ctx,
            "SELECT uri,event_id,event_type,status,payload FROM outbox_events WHERE account_id=%s "
            "ORDER BY created_at,event_id",
            (ctx.account_id,),
        )
        jobs = _graph_sql(
            fs,
            ctx,
            "SELECT edge_id,status FROM edge_projection_jobs WHERE account_id=%s",
            (ctx.account_id,),
        )
        print("SQL_BEFORE_GRAPH_ACK " + json.dumps({"events": events, "jobs": jobs}, default=str))
        graph_events = [event for event in events if event[2] == "PROJECT_GRAPH"]
        if projection_phase == "nodes_first":
            service._cfg.graph_backend = "none"
            paused = http.get(path, query_string={"clientRequestId": "project-dependency"})
            print("GRAPH_BACKEND none")
            record("GET", path, {"clientRequestId": "project-dependency"}, paused)
            service._cfg.graph_backend = "sql"
            resumed = http.get(path, query_string={"clientRequestId": "project-dependency"})
            print("GRAPH_BACKEND sql")
            record("GET", path, {"clientRequestId": "project-dependency"}, resumed)
        graph_worker.run_once(outbox, [ctx.account_id])
        final = http.get(path, query_string={"clientRequestId": "project-dependency"})
        record("GET", path, {"clientRequestId": "project-dependency"}, final)
        assert accepted.json["ok"] is True
        assert len(graph_events) == 1
        assert all(event[3] == "DONE" for event in events if event[2] != "PROJECT_GRAPH")
        references = {tuple(reference) for reference in observed.json["outbox_events"]}
        assert {(event[0], event[1]) for event in graph_events} <= references
        assert observed.json["request_status"] == (
            "PROJECTING" if projection_phase == "nodes_first" else "SUCCEEDED"
        )
        if projection_phase == "nodes_first":
            assert paused.json["request_status"] == "SUCCEEDED"
            assert paused.json["paused_projection_count"] == 1
            assert resumed.json["request_status"] == "PROJECTING"
        assert final.json["request_status"] == "SUCCEEDED"
        assert final.json["projection_status"] == "DONE"
        assert outbox.get_event_status(graph_events[0][0], graph_events[0][1]) == "DONE"
        assert _graph_sql(
            fs,
            ctx,
            "SELECT status FROM edge_projection_jobs WHERE account_id=%s",
            (ctx.account_id,),
        ) == [("COMPLETE",)]
    finally:
        service.shutdown()


@pytest.mark.parametrize("legacy", [False, True], ids=["fenced", "legacy"])
def test_graph_receipt_survives_interruption_after_edge_commit(graph_db, monkeypatch, legacy):
    ctx, fs, _, relations, _, outbox, graph_worker, _, _ = graph_db
    service, http = _graph_runtime(graph_db, monkeypatch, nodes_only=True)
    state_uri = service.get_session_manager()._session_state.state_uri(ctx.account_id, ctx.session_id)
    request = {
        "sessionId": ctx.session_id, "clientRequestId": "graph-replay", "wait": True, "forceExtract": True,
        "messages": [{"id": "turn", "role": "user", "content": "Projectx depends on Projecty."}],
    }
    upsert_edges = relations.upsert_edges

    class Interrupted(BaseException):
        pass

    def commit_then_interrupt(edges, ctx):
        upsert_edges(edges, ctx)
        raise Interrupted()

    monkeypatch.setattr(relations, "upsert_edges", commit_then_interrupt)
    try:
        print("HTTP_INTERRUPTED " + json.dumps(request))
        with pytest.raises(Interrupted):
            http.post("/api/v1/after_turn", json=request)
        state_node = fs.read_node(state_uri, ctx)
        state = json.loads(state_node.content)
        batch = state["runtime_extensions"][EXTENSION_KEY]["batch"]
        if legacy:
            batch["outbox_tracking_version"] = 1
            for slot in batch["slots"].values():
                slot.pop("relation_tracking", None)
            state_node.content = json.dumps(state)
            fs.write_node(state_node, ctx)
        events = _graph_sql(fs, ctx, "SELECT uri,event_id FROM outbox_events "
                           "WHERE account_id=%s AND event_type='PROJECT_GRAPH'", (ctx.account_id,))
        assert len(events) == 1
        assert list(events[0]) not in batch["outbox_events"]
        assert all(op["status"] == "complete" for slot in batch["slots"].values()
                   for op in slot["operations"].values())
        print("SQL_INTERRUPTED " + json.dumps({
            "events": events, "tracking_version": batch.get("outbox_tracking_version"),
            "checkpoint_events": batch["outbox_events"],
            "operations": {name: [op["status"] for op in slot["operations"].values()]
                           for name, slot in batch["slots"].items()},
        }))
    finally:
        service.shutdown()

    monkeypatch.setattr(relations, "upsert_edges", upsert_edges)
    restarted, http = _graph_runtime(graph_db, monkeypatch, nodes_only=True)
    try:
        replay = http.post("/api/v1/after_turn", json=request)
        record("POST", "/api/v1/after_turn", request, replay)
        path = f"/api/v1/sessions/{ctx.session_id}/write-requests/graph-replay"
        receipt = http.get(path)
        record("GET", path, None, receipt)
        event_status = outbox.get_event_status(*events[0])
        print("SQL_REPLAY " + json.dumps({"event": events[0], "status": event_status}))
        assert event_status == "PENDING"
        assert _graph_sql(fs, ctx, "SELECT uri,event_id FROM outbox_events "
                          "WHERE account_id=%s AND event_type='PROJECT_GRAPH'", (ctx.account_id,)) == events
        assert receipt.json["source_status"] == "ARCHIVED"
        assert receipt.json["request_status"] == "NEEDS_CONFIRMATION"
        assert receipt.json["projection_status"] == "UNKNOWN"
        assert receipt.json["retry_action"] == "manual_review"
        assert receipt.json["outbox_tracking_complete"] is False
        graph_worker.run_once(outbox, [ctx.account_id])
        assert outbox.get_event_status(*events[0]) == "DONE"
    finally:
        restarted.shutdown()


@capture_outbox_events
def test_graph_receipt_follows_the_committed_edge_transaction(graph_db):
    ctx, fs, _, relations, _, _, _, nodes, _ = graph_db
    edge = RelationEdge(nodes[0].uri, nodes[1].uri, "depends on", 1.0, "Project dependency")
    conn = relations._get_connection()
    try:
        relations.bind_tenant(conn, ctx.account_id)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT uri FROM context_nodes WHERE account_id=%s AND uri=ANY(%s) "
                "ORDER BY uri FOR UPDATE",
                (ctx.account_id, sorted([edge.from_uri, edge.to_uri])),
            )
            relations._enqueue_edge_projection(cur, edge, ctx)
            cur.execute(
                "SELECT uri,event_id FROM outbox_events WHERE account_id=%s", (ctx.account_id,)
            )
            draft = cur.fetchall()
        conn.rollback()
    finally:
        relations._return_connection(conn)
    relations.upsert_edges([edge], ctx)
    committed = _graph_sql(
        fs, ctx, "SELECT uri,event_id FROM outbox_events WHERE account_id=%s", (ctx.account_id,)
    )
    references = current_outbox_events(ctx.account_id)
    print(
        "GRAPH_TRANSACTION "
        + json.dumps({"rolled_back": draft, "committed": committed, "receipt": references})
    )
    assert len(draft) == len(committed) == 1
    assert references == committed
    assert draft[0][1] != committed[0][1]
    relations.upsert_edges([edge], ctx)
    assert current_outbox_events(ctx.account_id) == committed
    assert (
        _graph_sql(
            fs, ctx, "SELECT uri,event_id FROM outbox_events WHERE account_id=%s", (ctx.account_id,)
        )
        == committed
    )
