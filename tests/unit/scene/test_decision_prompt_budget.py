"""Complete evidence survives bounded decision and quality prompts."""

import json
from copy import deepcopy

import pytest

from scene.adapters import _fingerprint
from scene.decision import SceneDecisionEngine
from scene.decision_calls import SceneModelCaller
from scene.decision_contracts import DecisionLimits
from scene.decision_prompts import DECISION_PROMPT, QUALITY_PROMPT
from tests.unit.scene.test_engine_refresh import EvidenceModel, source_case


def restore_references(value):
    payload = deepcopy(value)
    references = payload.pop("reference_values", [])
    payload.pop("reference_format", None)

    def restore(item):
        if isinstance(item, dict):
            if set(item) == {"$scene_ref"}:
                return references[item["$scene_ref"]]
            if set(item) == {"$scene_refs"}:
                return [references[index] for index in item["$scene_refs"]]
            return {key: restore(child) for key, child in item.items()}
        if isinstance(item, list):
            return [restore(child) for child in item]
        return item

    return restore(payload)


@pytest.mark.parametrize("stage", ["decision", "quality"])
def test_reference_catalog_preserves_128_complete_cards_and_ordered_members(stage):
    uris = [f"ctx://acct/users/alice/memories/events/{index}-" + "u" * 120 for index in range(128)]
    sources = ["prov:archive:" + "s" * 120 + str(index) for index in range(8)]
    memories = [{"uri": uri, "text": "完整来源 " * 16, "node_fingerprint": f"fingerprint-{index}",
                 "source_refs": [sources[index % 8], sources[(index + 1) % 8], sources[index % 8]],
                 "time": {"event_time": None}} for index, uri in enumerate(uris)]
    scenes = [{"scene_id": f"scene-{index}", "description": "完整场景证据", "member_uris": uris[index:index + 64],
               "score": index / 10} for index in range(8)]
    packet = {"memories": memories, "provenance_context": deepcopy(memories[:8]), "offered_member_uris": uris,
              "existing_scenes": deepcopy(scenes[:4]), "tool_evidence": [{"tool": "search_scene_cards", "results": scenes}]}
    payload = packet if stage == "decision" else {"decision": {"action": "CREATE", "draft": {"member_uris": uris[:64]},
                "evidence_uris": uris[:64], "unassigned_member_uris": uris[64:]}, "evidence": packet}
    original = deepcopy(payload)
    encoded = SceneModelCaller._reference_catalog(payload)
    assert len(SceneModelCaller.dump(encoded)) < len(SceneModelCaller.dump(payload))
    assert restore_references(encoded) == payload == original
    assert encoded["reference_values"] == list(dict.fromkeys(encoded["reference_values"]))
    assert "output full literal URIs" in encoded["reference_format"]


@pytest.mark.parametrize("literal", [{"$scene_ref": 0}, {"$scene_refs": [0, 0]}])
def test_reference_catalog_preserves_literal_marker_objects(literal):
    reference = "ctx://acct/users/alice/memories/events/" + "r" * 120
    payload = {"memories": [{"uri": reference, "source_refs": [reference], "time": literal}]}
    original = deepcopy(payload)
    assert SceneModelCaller._reference_catalog(payload) == payload == original


@pytest.mark.parametrize("key", ["reference_values", "reference_format"])
def test_reference_catalog_preserves_existing_format_fields(key):
    reference = "prov:archive:" + "s" * 120
    payload = {key: {"original": None}, "source_refs": [reference, reference]}
    original = deepcopy(payload)
    assert SceneModelCaller._reference_catalog(payload) == payload == original


@pytest.mark.parametrize("stage", ["decision", "quality"])
def test_columnar_evidence_roundtrip_preserves_complete_cards(stage):
    cards = [{"uri": f"ctx://acct/users/alice/memories/events/{index}", "context_type": "MEMORY",
              "category": "events", "owner_space": "user:alice", "text": "证据" * 32,
              "source_refs": [f"prov:archive:session-{index}", "prov:archive:shared"],
              "time": {"event_time": None}, "node_fingerprint": f"fingerprint-{index}"}
             for index in range(64)]
    cards[0]["optional"] = None
    cards[1]["optional"] = "完整来源"
    packet = {"memories": cards, "provenance_context": deepcopy(cards[:8]),
              "offered_member_uris": [card["uri"] for card in cards],
              "existing_scenes": [{"scene_id": "release", "member_uris": [cards[0]["uri"]]}],
              "tool_evidence": [{"tool": "search_related_nodes", "results": deepcopy(cards[:8])}]}
    prefix = DECISION_PROMPT if stage == "decision" else QUALITY_PROMPT
    payload = packet if stage == "decision" else {"decision": {"action": "CREATE"}, "evidence": packet}
    original = deepcopy(payload)
    caller = SceneModelCaller(None, DecisionLimits(max_prompt_chars=len(prefix + SceneModelCaller.dump(payload)) - 1))
    prompt = caller.budgeted_prompt(prefix, payload)
    assert len(prompt) <= caller.limits.max_prompt_chars
    fitted = json.loads(prompt.removeprefix(prefix))
    evidence = fitted.get("evidence", fitted)
    assert "output full URIs" in evidence.pop("card_table_format")
    for key in ("memories", "provenance_context"):
        table = evidence[key]
        evidence[key] = [dict(zip(table["columns"][shape], values)) for shape, values in table["rows"]]
    assert fitted == payload == original


@pytest.mark.parametrize("prefix", [DECISION_PROMPT, QUALITY_PROMPT])
def test_small_prompt_preserves_serialized_evidence(prefix):
    caller = SceneModelCaller(None)
    packet = {"tool_evidence": [{"results": [{"text": "完整来源", "source_refs": ["archive:one"]}]}]}
    original = deepcopy(packet)
    assert caller.budgeted_prompt(prefix, packet).encode() == (prefix + caller.dump(packet)).encode()
    assert packet == original


@pytest.mark.parametrize("max_node_chars", [32, 64])
def test_compact_card_configuration_preserves_its_text_limit(max_node_chars):
    caller = SceneModelCaller(None, DecisionLimits(max_node_chars=max_node_chars))
    packet = {"tool_evidence": [{"results": [{"text": "evidence " * 1000} for _ in range(8)]}]}
    prompt = caller.budgeted_prompt(DECISION_PROMPT, packet)
    fitted = json.loads(prompt.removeprefix(DECISION_PROMPT))
    assert len(prompt) <= caller.limits.max_prompt_chars
    assert {len(card["text"]) for card in fitted["tool_evidence"][0]["results"]} == {max_node_chars}
    assert all(card["original_text_chars"] == 9000 for card in fitted["tool_evidence"][0]["results"])


@pytest.mark.parametrize("stage", ["decision", "quality"])
def test_large_prompt_shares_text_budget_and_preserves_provenance(stage):
    caller = SceneModelCaller(None)
    cards = [{"uri": f"ctx://acct/users/alice/memories/event/{index}",
              "source_refs": [f"archive:{index}", "archive:shared"],
              "node_fingerprint": f"fingerprint-{index}", "text": "完整证据" * 200}
             for index in range(16)]
    cards.append({"source_ref": "archive:short", "text": "Short source"})
    packet = {"memories": [{"uri": card["uri"], "text": "context" * 100,
                            "source_refs": [f"archive:{index}:" + "s" * 1200 for index in range(2)]}
                           for card in cards[:-1]],
              "tool_evidence": [{"results": cards[:8]}, {"results": cards[8:]}]}
    prefix = DECISION_PROMPT if stage == "decision" else QUALITY_PROMPT
    payload = packet if stage == "decision" else {"decision": {"action": "CREATE"}, "evidence": packet}
    original = deepcopy(payload)
    assert len(prefix + caller.dump(payload)) > caller.limits.max_prompt_chars
    prompt = caller.budgeted_prompt(prefix, payload)
    assert len(prompt) <= caller.limits.max_prompt_chars
    fitted = json.loads(prompt.removeprefix(prefix))
    evidence = fitted.get("evidence", fitted)
    shortened = [card for result in evidence["tool_evidence"] for card in result["results"]]
    retained_lengths = {len(card["text"]) for card in shortened[:-1]}
    assert len(retained_lengths) == 1
    assert 64 <= retained_lengths.pop() < 800
    for before, after in zip(cards[:-1], shortened[:-1]):
        assert after == {**before, "text": before["text"][:len(after["text"])],
                         "text_truncated": True, "original_text_chars": len(before["text"])}
    assert shortened[-1] == cards[-1]
    for before, after in zip(packet["memories"], evidence["memories"]):
        assert after == {**before, "text": before["text"][:len(after["text"])],
                         "text_truncated": True, "original_text_chars": len(before["text"])}
    assert payload == original


@pytest.mark.parametrize("stage", ["decision", "quality"])
def test_expanded_evidence_retains_sources_and_members_with_budgeted_hints(stage):
    caller = SceneModelCaller(None)
    memories = [{"uri": f"ctx://acct/users/alice/memories/events/{index}-" + "u" * 80,
                 "text": "evidence" * 100, "node_fingerprint": f"fingerprint-{index}",
                 "source_refs": [f"prov:archive:session-{index}-{source}-" + "s" * 100 for source in range(2)]}
                for index in range(64)]
    uris = [card["uri"] for card in memories]
    scene = {"scene_id": "release", "description": "Complete release description. " * 30,
             "member_uris": uris[:8]}
    packet = {"memories": memories, "offered_member_uris": uris,
              "provenance_context": [{"uri": "ctx://acct/sessions/source", "text": "source" * 150}],
              "existing_scenes": [deepcopy(scene)], "tool_evidence": [{"results": [deepcopy(scene)]}],
              "candidate_groups": [{"basis": "semantic", "quality": 1 - index * 0.05,
                                    "member_uris": uris[:64 - index]} for index in range(8)]}
    prefix = DECISION_PROMPT if stage == "decision" else QUALITY_PROMPT
    payload = packet if stage == "decision" else {"decision": {"action": "CREATE"}, "evidence": packet}
    original = deepcopy(payload)
    prompt = caller.budgeted_prompt(prefix, payload)
    assert len(prompt) <= 64000
    fitted = json.loads(prompt.removeprefix(prefix))
    evidence = fitted.get("evidence", fitted)
    omitted = evidence["omitted_candidate_groups"]
    assert 0 < omitted <= len(packet["candidate_groups"])
    assert evidence["candidate_groups"] == packet["candidate_groups"][:-omitted]
    assert evidence["offered_member_uris"] == uris
    for before, after in zip(memories, evidence["memories"]):
        assert after == {**before, "text": before["text"][:64],
                         "text_truncated": True, "original_text_chars": len(before["text"])}
    assert evidence["provenance_context"][0] == {**packet["provenance_context"][0], "text": "source" * 10 + "sour",
        "text_truncated": True, "original_text_chars": 900}
    for after in (evidence["existing_scenes"][0], evidence["tool_evidence"][0]["results"][0]):
        assert after == {**scene, "description": scene["description"][:64],
                         "description_truncated": True, "original_description_chars": len(scene["description"])}
    assert payload == original


@pytest.mark.parametrize("action", ["CREATE", "UPDATE"])
def test_budgeted_tool_evidence_publishes_reviewed_scene_with_checked_cas(action):
    source, nodes, _, proposal = source_case(action)
    cards = [{**source.node_card(node), "node_fingerprint": _fingerprint(node),
              "text": "Complete release evidence. " * 800} for node in nodes]
    original = deepcopy(cards)
    source.search_related_nodes = lambda *args: deepcopy(cards)
    model = EvidenceModel(proposal)
    engine = SceneDecisionEngine(model, evidence_reader=source)
    audit = []
    fit = engine._proposal.caller.budgeted_prompt

    def observe(prefix, payload, **kwargs):
        prompt = fit(prefix, payload, **kwargs)
        audit.append((len(prefix + SceneModelCaller.dump(payload)), len(prompt)))
        return prompt

    engine._proposal.caller.budgeted_prompt = observe
    result = engine.refresh(source, "acct", "user:alice", node_uris=[node.uri for node in nodes])
    assert result.action == action
    assert result.calls == 3
    assert result.snapshot.revision == source.repository.snapshot.revision == 1
    assert result.snapshot.scenes[0].version == (1 if action == "CREATE" else 2)
    assert result.pending_member_uris == (nodes[-1].uri,)
    assert all(after <= 64000 for _, after in audit)
    assert any(before > 64000 >= after for before, after in audit)
    for (prompt, _), prefix in zip(model.requests, (DECISION_PROMPT, QUALITY_PROMPT)):
        payload = json.loads(prompt.removeprefix(prefix))
        evidence = payload.get("evidence", payload)
        for before, after in zip(cards, evidence["tool_evidence"][0]["results"]):
            assert after["uri"] == before["uri"]
            assert after["source_refs"] == before["source_refs"]
            assert after["node_fingerprint"] == before["node_fingerprint"]
            assert after["text"] == before["text"][:800]
            assert after["text_truncated"] is True
            assert after["original_text_chars"] == len(before["text"])
    assert cards == original
