"""Parallel summary generation settles on the SQL mutation owner."""

import threading

from core.operation_deadline import remaining_seconds
from fs.sql_adapter.model_usage_store import SQLModelUsageStore
from fs.sql_adapter.session_mutation import active_session_guard
from providers.token_tracker import TokenTracker
from tests.integration.test_session_mutation_sql import database, ctx, service_pair
from tests.integration.test_extraction_batch_recovery_sql import category_nodes
from tests.unit.extraction.test_node_processing import NodeLLM, card, operation
from tests.unit.server.test_extraction_batch_recovery import params


def test_parallel_summary_keeps_sql_writes_on_mutation_owner(database, ctx):
    _, make = database
    service, writer, manager, archive, _ = service_pair(make, ctx, extract=lambda **kwargs: [card()])
    del service._generate_summary_candidate
    writer._node_processing_enabled = writer._card_format_enabled = True
    service._cfg.node_processing_enabled = service._cfg.card_format_enabled = True
    summary_started, node_started = threading.Event(), threading.Event()
    threads = {}
    tracker = TokenTracker(model="parallel-summary-test")
    usage = SQLModelUsageStore(writer._fs._shared_pool)
    tracker.set_usage_store(usage)

    def decide(kwargs):
        assert summary_started.wait(5)
        threads["node"] = threading.get_ident()
        node_started.set()
        return [operation([0])]

    model = NodeLLM(decide)

    def summarize(prompt, schema):
        threads["summary"] = threading.get_ident()
        assert remaining_seconds() > 0
        summary_started.set()
        assert node_started.wait(5)
        tracker.record_llm(10, 2)
        return {"facts_and_details": "Caroline's grandmother gave her a necklace."}

    model.complete_json = summarize
    service.get_llm = lambda: model
    writer._llm = model
    writes = []
    write_node = writer._fs.write_node

    def save(node, request):
        active_session_guard().check_context(request)
        writes.append((node.category, threading.get_ident()))
        return write_node(node, request)

    writer._fs.write_node = save
    owner = threading.get_ident()
    result = service.after_turn({**params(ctx), "timeoutSeconds": 30})
    assert result["status"] == "completed"
    assert threads["node"] != owner
    assert threads["summary"] != owner
    assert all(ident == owner for category, ident in writes)
    assert len(category_nodes(writer, "entity", ctx)) == 1
    assert len(category_nodes(writer, "session_summary", ctx)) == 1
    assert len(archive.list_archives(ctx.session_id, ctx)) == 1
    assert manager.get_or_create(ctx.session_id).request_completed("original")
    conn = writer._fs._shared_pool.get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT SUM(input_tokens), SUM(output_tokens), SUM(llm_calls) "
                           "FROM model_usage_events WHERE account_id=%s", (ctx.account_id,))
            assert cursor.fetchone() == (10, 2, 1)
        conn.rollback()
    finally:
        writer._fs._shared_pool.return_connection(conn)
