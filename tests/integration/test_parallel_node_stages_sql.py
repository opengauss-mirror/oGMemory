"""Independent model groups commit through their SQL connection owner."""

import threading

from service.api import MemoryWriteAPI
from tests.integration.test_session_mutation_sql import database, ctx
from tests.unit.extraction.test_node_processing import NodeLLM, card, node_payload, operation
from tests.unit.extraction.test_parallel_node_stages import record_storage_threads


def test_sql_owner_collects_independent_node_models(database, ctx):
    _, make = database
    fs, outbox, _ = make()
    rendezvous = threading.Barrier(2)
    model_threads = []

    def decide(call):
        payload = node_payload(call)
        model_threads.append(threading.get_ident())
        rendezvous.wait(5)
        return [operation([0], target=payload["nodes"][0]["uri"], action="update")]

    llm = NodeLLM(decide)
    api = MemoryWriteAPI(fs, llm, outbox, node_processing_enabled=True,
                         card_format_enabled=True, directory_summary_enabled=False)
    drafts = [card(scope_ref=scope) for scope in ("project_a", "project_b")]
    uris = [api.write_memory(draft, ctx)["target_uri"] for draft in drafts]
    api._pipeline.extract = lambda *args, **kwargs: drafts
    calls = record_storage_threads(fs)
    result = api.commit_session([{"role": "user", "content": "Both necklaces have heart charms."}], ctx)
    assert result["writes_completed"] == 2
    assert len(set(model_threads)) == 2
    assert {thread_id for _, thread_id in calls} == {threading.get_ident()}
    assert all(fs.read_node(uri, ctx).metadata["version"] == 2 for uri in uris)
