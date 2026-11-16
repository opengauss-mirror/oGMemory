"""Keep independent source-grounded decisions available during node repair."""

from dataclasses import replace

import pytest

from extraction.node_processing import NodeProcessingPartialError
from tests.unit.extraction.test_node_processing import card, node_payload, node_response, operation, setup
from tests.unit.extraction.test_node_reconciliation import SOURCE, ReconciliationLLM, atlas, prepare, primary, related


def decision(draft, ids, *, target="", action=None):
    result = operation(ids, target=target, action=action or ("update" if target else "create"))
    result.update(object_name=draft.object_name, routing_key=draft.routing_key,
                  abstract=draft.abstract, overview=[draft.abstract])
    if action == "noop":
        result["sections"] = []
    return result


def partial(processor, drafts, ctx, **kwargs):
    try:
        processor.process(drafts, [], ctx, **kwargs)
    except NodeProcessingPartialError as exc:
        return exc
    pytest.fail("Expected the remaining repair group")


@pytest.mark.parametrize("repair", ["schema", "edit"])
def test_healthy_card_and_reconciled_event_remain_available_during_peer_repair(repair):
    healthy = card("necklace", node_draft_indices=[11])
    repair_draft = card("bracelet", node_draft_indices=[23])
    event = card("visit", category="event", content="Caroline visited Melanie.")
    ctx, fs, llm, api = setup(lambda _: [])
    target = api.write_memory(repair_draft, ctx)["target_uri"]
    healthy_op = decision(healthy, [0])
    repair_op = decision(repair_draft, [1], target=target)
    if repair == "schema":
        repair_op["overview"] = 1
    else:
        repair_op["edits"] = [{"search": "The bracelet has a green charm.", "replacement": "It has a blue charm."}]
    llm.decide = lambda _: [healthy_op, repair_op]
    result = partial(api._node_processor(), [healthy, repair_draft, event], ctx, reconcile=lambda: [event])
    assert {candidate.routing_key for candidate in result.candidates} == {"necklace", "visit"}
    assert result.others_complete is True
    assert result.failures[0]["target"] == target
    assert result.failures[0]["draft_ids"] == [23]
    assert result.candidates[1].target_uri == ""
    assert len(llm.calls) == 1


@pytest.mark.parametrize("acknowledgement", ["complete", "repair"])
def test_same_target_decisions_remain_one_repair_group(acknowledgement):
    drafts = [card("necklace", node_draft_indices=[31]), card("necklace", node_draft_indices=[47]),
              card("bracelet", node_draft_indices=[59])]
    ctx, _, llm, api = setup(lambda _: [])
    target = api.write_memory(drafts[0], ctx)["target_uri"]
    update = decision(drafts[0], [0], target=target)
    update["edits"] = [{"search": "The necklace is silver.", "replacement": "The necklace is gold."}]
    noop = decision(drafts[1], [1], target=target, action="noop")
    if acknowledgement == "repair":
        noop["overview"] = 1
    llm.decide = lambda _: [update, noop, decision(drafts[2], [2])]
    result = partial(api._node_processor(), drafts, ctx)
    assert [candidate.routing_key for candidate in result.candidates] == ["bracelet"]
    assert result.failures[0]["draft_ids"] == [31, 47]


def test_new_identity_contributors_repair_together_while_an_independent_card_completes():
    drafts = [card("necklace", node_draft_indices=[37]), card("necklace", node_draft_indices=[43]),
              card("bracelet", node_draft_indices=[61])]
    ctx, _, llm, api = setup(lambda _: [])
    repair = decision(drafts[0], [0])
    del repair["action"]
    llm.decide = lambda _: [repair, decision(drafts[1], [1]), decision(drafts[2], [2])]
    result = partial(api._node_processor(), drafts, ctx)
    assert [candidate.routing_key for candidate in result.candidates] == ["bracelet"]
    assert result.failures[0]["draft_ids"] == [37, 43]


def test_explicit_related_correction_keeps_its_source_group_together():
    ctx, _, api, targets = prepare()
    source = replace(atlas(hour=21), source_contexts=[SOURCE], node_draft_indices=[67])
    healthy = card("necklace", scope_ref="Garden", node_draft_indices=[79])

    def decide(payload):
        if payload["drafts"][0]["routing_key"] == "necklace":
            return {"operations": [decision(healthy, [0])], "related_reconciliations": []}
        nodes = {node["uri"]: node for node in payload["nodes"]}
        correction = related(nodes[targets["preference"]])
        correction["edits"] = [{"search": "北京时间19点", "replacement": "北京时间21点"}]
        return {"operations": [primary([0], nodes[targets["entity"]])],
                "related_reconciliations": [correction]}

    api._llm = ReconciliationLLM(decide)
    result = partial(api._node_processor(), [source, healthy], ctx)
    assert [candidate.routing_key for candidate in result.candidates] == ["necklace"]
    assert {failure["target"] for failure in result.failures} == set(targets.values())
    assert all(failure["draft_ids"] == [67] for failure in result.failures)


def test_staged_new_node_preserves_all_contributors_for_repair():
    ctx, _, _, api = setup(lambda _: [])
    processor = api._node_processor()
    drafts = [card(node_draft_indices=[83]), card(node_draft_indices=[97])]
    generator = processor._process_chain([(drafts[:1], []), (drafts[1:], [])], [], ctx, None)
    prompt, tool = next(generator)
    first = node_response({"prompt": prompt}, [decision(drafts[0], [0])])
    prompt, tool = generator.send([{"tool": "process_memory_nodes", "input": first}])
    target = node_payload({"prompt": prompt})["nodes"][0]["uri"]
    update = decision(drafts[1], [0], target=target)
    update["edits"] = [{"search": "The chain is silver.", "replacement": "The chain is gold."}]
    try:
        generator.send([{"tool": "process_memory_nodes", "input": node_response({"prompt": prompt}, [update])}])
    except NodeProcessingPartialError as result:
        assert result.candidates == []
        assert result.failures[0]["draft_ids"] == [83, 97]
        assert result.failures[0]["target"] == target
    else:
        pytest.fail("Expected the complete staged node repair group")


def test_committed_target_is_read_only_while_a_new_node_completes():
    existing = card("necklace", node_draft_indices=[101])
    fresh = card("bracelet", node_draft_indices=[103])
    ctx, fs, llm, api = setup(lambda _: [])
    target = api.write_memory(existing, ctx)["target_uri"]
    llm.decide = lambda _: [decision(existing, [0], target=target, action="noop"), decision(fresh, [1])]
    result = api._node_processor().process([existing, fresh], [], ctx, protected_targets=[target])
    assert [candidate.routing_key for candidate in result] == ["bracelet"]
    assert target in llm.calls[0]["prompt"]
    assert fs.read_node(target, ctx).metadata["version"] == 1


def test_reconciled_event_survives_a_node_response_requiring_full_repair():
    ctx, _, llm, api = setup(lambda _: [])
    event = card("visit", category="event", content="Caroline visited Melanie.")
    drafts = [card(node_draft_indices=[107]), card("bracelet", node_draft_indices=[109]), event]
    llm.complete_with_tools = lambda **kwargs: []
    result = partial(api._node_processor(), drafts, ctx, reconcile=lambda: [event])
    assert result.candidates == [event]
    assert result.others_complete is True
    assert result.failures[0]["draft_ids"] == [107, 109]


def test_independent_chain_and_event_complete_while_a_peer_response_is_repaired():
    ctx, _, llm, api = setup(lambda _: [])
    healthy = card("necklace", scope_ref="Garden", node_draft_indices=[131])
    peers = [card("bracelet", scope_ref="Museum", node_draft_indices=[113]),
             card("watch", scope_ref="Museum", node_draft_indices=[127])]
    event = card("visit", category="event", content="Caroline visited Melanie.")
    scopes = []

    def complete_with_tools(**kwargs):
        scope = node_payload(kwargs)["drafts"][0]["scope_ref"]
        scopes.append(scope)
        payload = (node_response(kwargs, [decision(healthy, [0])])
                   if scope == "Garden" else '{"operations": [')
        return [{"tool": "process_memory_nodes", "input": payload}]

    llm.complete_with_tools = complete_with_tools
    result = partial(api._node_processor(), [healthy, *peers, event], ctx, reconcile=lambda: [event])
    assert {candidate.routing_key for candidate in result.candidates} == {"necklace", "visit"}
    assert result.others_complete is True
    assert [(failure["target"], failure["draft_ids"]) for failure in result.failures] == [("", [113, 127])]
    assert sorted(scopes) == ["Garden", "Museum"]


def test_related_answer_uses_authorized_sources_for_the_repair_group():
    import json

    ctx, fs, api, targets = prepare()
    source = replace(atlas(hour=21), source_contexts=[SOURCE])
    healthy = card("necklace", scope_ref="Atlas", source_contexts=["Caroline owns a necklace from her grandma."])
    api._pipeline.extract = lambda *args, **kwargs: [source, healthy]
    model_packets = []

    def decide(payload):
        nodes = {node["uri"]: node for node in payload["nodes"]}
        question = next(question for question in payload["related_questions"].values()
                        if question["target"] == targets["preference"])
        assert question["source_draft_ids"] == [0]
        correction = related(nodes[targets["preference"]])
        correction["source_draft_ids"] = [1]
        response = {"operations": [primary([0], nodes[targets["entity"]]), decision(healthy, [1])],
                    "related_reconciliations": [correction]}
        model_packets.append({"drafts": payload["drafts"], "question": question, "response": response})
        return response

    api._llm = ReconciliationLLM(decide)
    request = [{"role": "user", "content": SOURCE + " Caroline owns a necklace from her grandma."}]
    result = api.commit_session(request, ctx)
    print(json.dumps({"request": request, "model": model_packets, "result": result}, ensure_ascii=False))

    assert result["degraded"] is True
    assert result["writes_completed"] == 1
    assert result["plans"][0]["action"] == "create"
    healthy_node = fs.read_node(result["plans"][0]["target_uri"], ctx)
    assert healthy_node.metadata["object_name"] == "necklace"
    assert all(fs.read_node(target, ctx).metadata["version"] == 1 for target in targets.values())
    assert all(failure["draft_ids"] == [0] for failure in result["failed_nodes"])
    assert len(api._llm.calls) == 1
