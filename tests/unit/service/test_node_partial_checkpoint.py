"""Independent node writes survive source recovery and service restarts."""

import json
from copy import deepcopy
from dataclasses import asdict, replace
from importlib import import_module
from unittest.mock import Mock

import pytest

from commit.write_receipts import RECEIPTS_KEY
from core.models import RequestContext
from service.api import MemoryWriteAPI
from service.extraction_checkpoint import ExtractionBatchJournal
from tests.unit.commit.test_memory_identity import MemoryFS, memory
from tests.unit.extraction.test_node_processing import NodeLLM, node_payload, operation
from tests.unit.server.test_extraction_batch_recovery import nodes, params, setup_service
from tests.unit.server.test_extraction_retry_task import ready
from tests.unit.server.test_memory_card_recovery import enable


pytestmark = pytest.mark.usefixtures("service_threading")

SOURCE = (
    "Caroline: I now mentor LGBTQ youth at the youth center. "
    "Melanie: I returned from a family camping trip."
)
OLD_FACTS = {
    "Caroline": "Caroline volunteers at a youth center.",
    "Melanie": "Melanie enjoys family outings.",
}
NEW_FACTS = {
    "Caroline": "Caroline mentors LGBTQ youth at the youth center.",
    "Melanie": "Melanie returned from a family camping trip.",
}


def person(name, fact):
    return replace(memory(
        name.lower(), object_name=name, who=name,
        content=f"# {name}\n\n## Activities\n- {fact}",
    ),
        abstract=fact, overview=f"- {fact}",
    )


@pytest.mark.parametrize("action", ["update", "noop"])
def test_bound_draft_recovers_against_current_identity(monkeypatch, action):
    ctx = RequestContext("bound-recovery", "reader", "locomo", "source", "proof")
    fs = MemoryFS()
    service, writer, manager = setup_service(fs, ctx)
    enable(service, writer)
    service._cfg.node_processing_enabled = writer._node_processing_enabled = True
    original = person("Nate", "Nate lives in Seattle.")
    target = writer.write_memory(original, ctx)["target_uri"]
    source = "Nate: I own three turtles. Joanna: He cares for them daily."
    draft = replace(person("He", "He owns three turtles."), target_uri=target,
                    expected_version=1, source_contexts=[source])
    writer._pipeline.extract.side_effect = lambda *args, **kw: [replace(
        draft, provenance_ids=[f"prov:1:archive:{kw['archive_id']}:msg-one"])]

    def decide(call):
        payload = node_payload(call)
        snapshot, = payload["nodes"]
        op = operation([0], target=target, action="update")
        op.update(routing_key="nate", object_name="Nate", abstract="Nate owns three turtles.",
                  overview=["Nate owns three turtles."], sections=[])
        if len(writer._llm.calls) == 1:
            op["edits"] = [{"search": "Nate owns two turtles.", "replacement": op["abstract"]}]
        else:
            assert snapshot["version"] == 2
            op["action"] = action
            op["scope_resolution"] = {"stored_scope_role": "self", "incoming_scope_role": "self",
                                      "source_quotes": [source]}
            if action == "update":
                op["sections"] = [{"heading": "Pets", "facts": [op["abstract"]]}]
        print("Node request:", json.dumps(payload))
        print("Node response:", json.dumps(op))
        return [op]

    writer._llm = NodeLLM(decide)
    module = import_module("server.app")
    monkeypatch.setattr(module, "_service", service)
    request = params(ctx, content=source)
    request.pop("_ctx")
    request.update(accountId=ctx.account_id, userId=ctx.user_id, agentId=ctx.agent_id)
    client = module.app.test_client()

    def submit():
        response = client.post("/api/v1/after_turn", json=request)
        print("POST /api/v1/after_turn", json.dumps(request),
              "HTTP", response.status_code, response.get_json())
        return response

    initial = submit()
    assert initial.status_code == 202
    peer_fact = "Nate lives in Portland." + (" Nate owns three turtles." if action == "noop" else "")
    peer = replace(person("Nate", peer_fact), target_uri=target, expected_version=1,
                   verified_object_aliases=["Nathan"], verified_routing_aliases=["nathan"])
    print("write_memory:", asdict(peer), writer.write_memory(peer, replace(ctx, session_id="peer")))
    ready(manager, ctx)
    service._extraction_retry.run_once(ctx, initial.get_json()["recovery_task"]["id"])
    final = submit()
    stored = fs.read_node(target, ctx)
    print("Stored node:", json.dumps(asdict(stored)))
    assert final.status_code == 200
    assert final.get_json()["request_status"] == "SUCCEEDED"
    assert "Nate lives in Portland." in stored.content
    assert "Nate owns three turtles." in stored.content
    assert stored.metadata["object_name"] == "Nate"
    assert set(stored.metadata["object_name_aliases"]) == {"Nate", "Nathan"}
    assert stored.metadata["version"] == 3
    assert writer._pipeline.extract.call_count == 1
    assert len(writer._llm.calls) == 2


@pytest.mark.parametrize("action", ["create", "update"])
def test_conflicting_entity_operations_preserve_independent_write(monkeypatch, action):
    ctx = RequestContext("node-partial", "reader", "locomo", f"duplicate-{action}", "proof")
    fs = MemoryFS()
    facts = ("Alice restores radios.", "Alice collects antique receivers.", "Bob repairs cameras.")
    drafts = [person(name, fact) for name, fact in zip(("Alice", "Alice", "Bob"), facts)]
    service, writer, manager = setup_service(fs, ctx, entries=drafts)
    enable(service, writer)
    service._cfg.node_processing_enabled = writer._node_processing_enabled = True
    target = ""
    if action == "update":
        target = writer.write_memory(person("Alice", "Alice restores electronics."), ctx)["target_uri"]

    def decide(call):
        names = [draft["object_name"] for draft in node_payload(call)["drafts"]]
        if len(names) == 3:
            groups = [([i], name, [facts[i]]) for i, name in enumerate(names)]
        else:
            assert names == ["Alice", "Alice"]
            groups = [([0, 1], "Alice", list(facts[:2]))]
        ops = []
        for indices, name, claims in groups:
            op = operation(indices, target=target, action=action) if name == "Alice" else operation(indices)
            op.update(object_name=name, routing_key=name.lower(), abstract=claims[0],
                      overview=claims, sections=[{"heading": "Activities", "facts": claims}])
            ops.append(op)
        print("Model response:", json.dumps(ops))
        return ops

    writer._llm = NodeLLM(decide)
    module = import_module("server.app")
    monkeypatch.setattr(module, "_service", service)
    request = params(ctx, content=" ".join(facts))
    request.pop("_ctx")
    request.update(accountId=ctx.account_id, userId=ctx.user_id, agentId=ctx.agent_id)
    client = module.app.test_client()

    def submit():
        print("POST /api/v1/after_turn", json.dumps(request))
        response = client.post("/api/v1/after_turn", json=request)
        print("HTTP", response.status_code, response.get_json(),
              "entity_cards", {node.metadata["object_name"]: node.metadata["version"]
                               for node in nodes(fs, "entity")})
        return response

    initial = submit()
    assert initial.status_code == 202
    assert initial.get_json()["request_status"] == "RECOVERING"
    bob, = [node for node in nodes(fs, "entity") if node.metadata["object_name"] == "Bob"]
    if target:
        assert fs.read_node(target, ctx).metadata["version"] == 1
    journal = ExtractionBatchJournal.resume(manager, manager.get_or_create(ctx.session_id), ctx)
    assert {(f["target"], tuple(f["draft_ids"])) for f in journal.unresolved_node_failures()} == {
        (target, (0, 1)),
    }

    ready(manager, ctx)
    service._extraction_retry.run_once(ctx, initial.get_json()["recovery_task"]["id"])
    final = submit()
    assert final.status_code == 200
    assert final.get_json()["request_status"] == "SUCCEEDED"
    assert final.get_json()["writes_completed"] == 2
    assert {node.metadata["object_name"] for node in nodes(fs, "entity")} == {"Alice", "Bob"}
    alice, = [node for node in nodes(fs, "entity") if node.metadata["object_name"] == "Alice"]
    assert alice.metadata["version"] == (2 if target else 1)
    assert all(fact in alice.content for fact in facts[:2])
    assert fs.read_node(bob.uri, ctx).metadata == bob.metadata
    assert writer._pipeline.extract.call_count == 1
    assert len(writer._llm.calls) == 2


def runtime(fs, ctx, targets, *, repaired):
    drafts = [person(name, fact) for name, fact in NEW_FACTS.items()]
    drafts.append(memory(
        "melanie_family_camping", category="event", who="Melanie",
        title="Family camping trip", when="2023-07-12",
        content=NEW_FACTS["Melanie"],
    ))
    service, writer, manager = setup_service(fs, ctx, entries=drafts)
    enable(service, writer)
    service._cfg.node_processing_enabled = writer._node_processing_enabled = True

    def decide(call):
        payload = node_payload(call)
        result = []
        for index, draft in enumerate(payload["drafts"]):
            name = draft["object_name"]
            item = operation([index], target=targets[name], action="update")
            item.update(routing_key=name.lower(), object_name=name,
                        abstract=NEW_FACTS[name], overview=[NEW_FACTS[name]], sections=[])
            item["edits"] = [{
                "search": ("Melanie stopped camping last year."
                           if name == "Melanie" and not repaired else OLD_FACTS[name]),
                "replacement": NEW_FACTS[name],
            }]
            result.append(item)
        return result

    writer._llm = NodeLLM(decide)
    return service, writer, manager


@pytest.mark.parametrize("repaired", [True, False], ids=["remaining-node-recovers", "remaining-node-archived"])
@pytest.mark.parametrize("concurrent_edit", [False, True], ids=["stable-version", "advanced-version"])
def test_partial_checkpoint_restores_independent_writes_and_finishes_source(repaired, concurrent_edit):
    ctx = RequestContext("node-partial", "reader", "locomo", "source", "proof")
    fs = MemoryFS()
    targets = {}
    service, writer, manager = runtime(fs, ctx, targets, repaired=False)
    for name, fact in OLD_FACTS.items():
        targets[name] = writer.write_memory(person(name, fact), ctx)["target_uri"]
    if concurrent_edit:
        write_candidate = writer._writer.write_candidate
        advanced = []

        def write_after_peer_update(entry, request_ctx):
            if entry.target_uri == targets["Caroline"] and not advanced:
                current = fs.read_node(entry.target_uri, request_ctx)
                current.content += "\n- Caroline has volunteered since 2021."
                fs.write_node(current, request_ctx)
                advanced.append(current.uri)
            return write_candidate(entry, request_ctx)

        writer._writer.write_candidate = write_after_peer_update

    request = params(ctx, content=SOURCE)
    response = service.after_turn(request)
    identity = response["recovery_task"]["id"]
    assert response["request_status"] == "RECOVERING"
    assert writer._pipeline.extract.call_count == 1
    assert len(writer._llm.calls) == 1 + int(concurrent_edit)
    assert fs.read_node(targets["Caroline"], ctx).metadata["version"] == 2 + int(concurrent_edit)
    assert NEW_FACTS["Caroline"] in fs.read_node(targets["Caroline"], ctx).content
    assert fs.read_node(targets["Melanie"], ctx).content == person("Melanie", OLD_FACTS["Melanie"]).content
    assert len(nodes(fs, "event")) == 1
    event = nodes(fs, "event")[0]
    completed = {uri: fs.read_node(uri, ctx) for uri in (targets["Caroline"], event.uri)}
    journal = ExtractionBatchJournal.resume(manager, manager.get_or_create(ctx.session_id), ctx)
    child = next(slot for slot in journal.batch["slots"].values() if slot.get("partial_parent"))
    parent = journal.checkpoint(child["partial_parent"])
    drafts = parent.load_drafts()
    stage_inputs = {"source": SOURCE, "archive_id": identity, "mode": "eager"}
    native_result = {"candidates": [asdict(draft) for draft in drafts], "reads": {}}
    parent.save_stage("native_extraction", stage_inputs, native_result)
    prepared_operation = operation([0], target=targets["Caroline"], action="noop")
    prepared_operation["sections"] = []
    parent.save_stage("node_processing", stage_inputs, [{"tool": "process_memory_nodes", "input": {
        "operations": [prepared_operation],
        "related_reconciliations": {},
    }}])
    parent.invalidate_unaccepted_preparation()
    assert parent.load_stage("native_extraction", stage_inputs) == native_result
    assert parent.load_stage("node_processing", stage_inputs) is None
    assert parent.load_drafts() == drafts
    assert [draft.node_draft_indices for draft in drafts] == [[0], [1], []]
    saved = deepcopy(journal.batch)
    assert {entry["object_name"] for entry in child["candidates"] if entry["category"] == "entity"} == {"Caroline"}
    assert {entry["category"] for entry in child["candidates"]} == {"entity", "event"}
    assert {operation["status"] for operation in child["operations"].values()} == {"complete"}
    assert journal.unresolved_node_failures()[0]["draft_ids"] == [1]
    if concurrent_edit:
        assert NEW_FACTS["Caroline"] in completed[targets["Caroline"]].content
        assert "Caroline has volunteered since 2021." in completed[targets["Caroline"]].content
        assert list(child["revisions"].values()) == [1]
        assert child["revision_history"][0]["candidate"]["node_draft_indices"] == [0]
        assert node_payload(writer._llm.calls[-1])["source"] == journal.checkpoint(
            child["partial_parent"]).draft_source()

    resumed, next_writer, next_manager = runtime(fs, ctx, targets, repaired=repaired)
    resumed.wait_until_idle({"_ctx": ctx, "sessionId": ctx.session_id, "timeoutSeconds": 0})
    restored = ExtractionBatchJournal.resume(next_manager, next_manager.get_or_create(ctx.session_id), ctx)
    assert restored.batch["slots"] == saved["slots"]
    assert restored.batch["snapshot"] == saved["snapshot"]
    assert restored.checkpoint(child["partial_parent"]).load_stage("native_extraction", stage_inputs) == native_result
    assert restored.checkpoint(child["partial_parent"]).load_drafts() == drafts
    for _ in range(1 if repaired else 3):
        ready(next_manager, ctx)
        resumed._extraction_retry.run_once(ctx, identity)

    final = resumed.after_turn(request)
    assert final["source_status"] == "ARCHIVED"
    assert final["semantic_status"] == ("COMPLETE" if repaired else "DEGRADED")
    assert final["writes_completed"] == (3 if repaired else 2)
    assert final["candidates_extracted"] == 3
    assert len(nodes(fs, "session_archive")) == 1
    assert SOURCE in nodes(fs, "session_archive")[0].content
    assert next_manager.get_or_create(ctx.session_id).request_completed("original")
    assert next_writer._pipeline.extract.call_count == 0
    assert len(next_writer._llm.calls) == (1 if repaired else 2)
    assert [[draft["object_name"] for draft in node_payload(call)["drafts"]]
            for call in next_writer._llm.calls] == [["Melanie"]] * len(next_writer._llm.calls)
    for uri, before in completed.items():
        after = fs.read_node(uri, ctx)
        assert after.content == before.content
        assert after.metadata["version"] == before.metadata["version"]
        assert after.metadata[RECEIPTS_KEY] == before.metadata[RECEIPTS_KEY]
    if repaired:
        assert fs.read_node(targets["Melanie"], ctx).metadata["version"] == 2
        assert NEW_FACTS["Melanie"] in fs.read_node(targets["Melanie"], ctx).content
    else:
        assert final["recovery_mode"] == "partial"
        assert [(failure["target"], failure["draft_ids"]) for failure in final["failed_nodes"]] == [
            (targets["Melanie"], [1])]
        assert "L2 edit search must occur once" in final["failed_nodes"][0]["error"]
        assert fs.read_node(targets["Melanie"], ctx).metadata["version"] == 1


def test_partial_first_chunk_preserves_healthy_later_chunk_and_source():
    ctx = RequestContext("partial-chunks", "reader", "locomo", "source", "proof")
    fs, targets = MemoryFS(), {}
    service, writer, manager = runtime(fs, ctx, targets, repaired=False)
    for name, fact in OLD_FACTS.items():
        targets[name] = writer.write_memory(person(name, fact), ctx)["target_uri"]
    service._cfg.chunking_enabled = True
    service._cfg.chunk_max_messages = 1
    service._cfg.chunk_min_tokens = 0
    service._cfg.chunk_llm_boundary = False
    successor_source = "Erin: I opened a new community garden on July 14."
    extract_first = writer._pipeline.extract.side_effect

    def extract(messages, request_ctx, **kwargs):
        if messages[0]["content"] == successor_source:
            return [replace(memory(
                "erin_community_garden_opening", category="event", who="Erin",
                title="Community garden opening", when="2023-07-14", content=successor_source,
            ), provenance_ids=[f"prov:1:archive:{kwargs['archive_id']}:garden-source"])]
        return extract_first(messages, request_ctx, **kwargs)

    writer._pipeline.extract.side_effect = extract
    request = params(ctx, content=SOURCE)
    request["messages"].append({"role": "user", "content": successor_source})

    response = service.after_turn(request)

    assert response["request_status"] == "RECOVERING"
    journal = ExtractionBatchJournal.resume(manager, manager.get_or_create(ctx.session_id), ctx)
    assert [[message["content"] for message in chunk] for chunk in journal.batch["chunks"]] == [
        [SOURCE], [successor_source]]
    assert journal.checkpoint("chunk:1").record["operations"]["0"]["status"] == "complete"
    assert len(nodes(fs, "event")) == 2
    assert writer._pipeline.extract.call_count == 2
    saved_events = {node.uri: deepcopy(node) for node in nodes(fs, "event")}
    for _ in range(3):
        ready(manager, ctx)
        service._extraction_retry.run_once(ctx, response["recovery_task"]["id"])

    final = service.after_turn(request)

    assert final["semantic_status"] == "DEGRADED"
    assert final["recovery_mode"] == "partial"
    assert final["writes_completed"] == 3
    assert final["candidates_extracted"] == 4
    assert final["failed_nodes"][0]["target"] == targets["Melanie"]
    assert writer._pipeline.extract.call_count == 2
    assert len(writer._llm.calls) == 3
    archives = nodes(fs, "session_archive")
    assert len(archives) == 2
    assert {archive.content for archive in archives} == {f"user: {SOURCE}", f"user: {successor_source}"}
    for uri, saved in saved_events.items():
        current = fs.read_node(uri, ctx)
        assert current.content == saved.content
        assert current.metadata["version"] == saved.metadata["version"]
        assert current.metadata[RECEIPTS_KEY] == saved.metadata[RECEIPTS_KEY]


def test_partial_relations_connect_successful_nodes_from_the_current_batch():
    ctx = RequestContext("partial-relations", "reader", "locomo", "source", "proof")
    fs, relations = MemoryFS(), Mock()
    llm = NodeLLM(lambda call: [])
    writer = MemoryWriteAPI(fs, llm, directory_summary_enabled=False,
                            graph_enabled=True, relation_store=relations)
    old_target = writer.write_memory(person("Melanie", OLD_FACTS["Melanie"]), ctx)["target_uri"]
    candidates = [person("Caroline", NEW_FACTS["Caroline"]),
                  person("Erin", "Erin also mentors LGBTQ youth.")]
    plans = writer._writer.write_candidates(candidates, ctx)
    proposals = [{"from_routing_key": "caroline", "from_category": "entity",
                  "to_routing_key": name, "to_category": "entity", "weight": .9,
                  "relation_type": "supports", "reason": "They mentor LGBTQ youth."}
                 for name in ("erin", "melanie")]

    result = writer.finish_partial_relations(candidates, plans, ctx, proposals=proposals)

    assert fs.read_node(old_target, ctx).metadata["version"] == 1
    assert result == {"native_relation_accepted": 1,
                      "native_relation_rejections": ["missing_endpoint"], "relation_persisted": 1}
    relations.upsert_edges.assert_called_once()
    saved, request_ctx = relations.upsert_edges.call_args.args
    assert request_ctx == ctx
    assert [(edge.from_uri, edge.to_uri) for edge in saved] == [
        (plans[0].target_uri, plans[1].target_uri)]
    assert len(llm.calls) == 0
