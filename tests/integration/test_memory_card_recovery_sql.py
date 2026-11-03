from dataclasses import replace

from extraction.card_format import validate_memory_card
from tests.integration.test_session_mutation_sql import database, ctx, service_pair
from tests.integration.test_extraction_batch_recovery_sql import category_nodes
from tests.unit.server.test_extraction_batch_recovery import params
from tests.unit.server.test_extraction_retry_task import ready
from tests.unit.server.test_memory_card_recovery import card, enable


def test_card_queue_reconnect_commits_one_node_and_archive_on_sql(database, ctx):
    _, make = database
    service, writer, manager, archive, outbox = service_pair(
        make, ctx, extract=lambda **kwargs: [replace(card(), content="Nora is a radio engineer.")])
    enable(service, writer)
    result = service.after_turn(params(ctx))
    assert result["ok"] is True and result["status"] == "queued"
    assert writer._pipeline.extract.call_count == 1
    restarted, writer, manager, archive, outbox = service_pair(
        make, ctx, extract=lambda **kwargs: [card()])
    enable(restarted, writer)
    status = restarted.wait_until_idle({"sessionId": ctx.session_id, "_ctx": ctx})
    assert status["recovery_task"]["id"] == result["recovery_task"]["id"]
    with restarted._session_mutation_scope(ctx):
        ready(manager, ctx)
    restarted._extraction_retry.run_once(ctx, result["recovery_task"]["id"])
    saved = category_nodes(writer, "entity", ctx)
    assert len(saved) == 1
    validate_memory_card(saved[0])
    assert restarted.after_turn(params(ctx))["status"] == "duplicate"
    assert len(archive.list_archives(ctx.session_id, ctx)) == 1
