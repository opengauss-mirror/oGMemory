"""Compact singleton references retain validated facts and stable occurrence identity."""

import json
import threading
from copy import deepcopy
from dataclasses import replace
from unittest.mock import Mock

import pytest
from jsonschema import validate

from commit.identity_resolver import occurrence_replay_id
from core.validation import validate_candidate
from extraction.event_reconciliation import reconcile_event_drafts
from extraction.source_occurrence_reconciliation import _SCHEMA, reconcile_source_occurrences
from extraction.tools import _Span
from core.model_stage import stage_collector_scope
from tests.unit.extraction.test_event_draft_boundaries import event
from tests.unit.extraction.test_event_occurrence_reconciliation import proposal
from tests.unit.extraction.test_source_occurrence_reconciliation import CTX, EARLY, LATE, TIME, SourceLLM


def compact_singletons(response, _payload):
    response = {"groups": [{"members": group["members"]} if len(group["members"]) == 1
                            else group for group in response["groups"]]}
    validate(response, _SCHEMA)
    return response


def source_entries(category="event"):
    texts = [EARLY, LATE, "I inspected the pump separately at 11:00. Its flow was 8 litres per minute."]
    bodies = ["Nora observed pressure of 5 bar.", "Nora observed a seal leak.",
              "Nora inspected the pump at 11:00. Flow was 8 litres per minute."]
    if category == "case":
        texts = ["I repaired the receiver once by replacing its cable.",
                 "In that same receiver repair I rebooted it and it worked.",
                 "I repaired a separate pump by replacing its fuse. Flow returned to normal."]
        bodies = ["Nora replaced the receiver cable.", "Nora rebooted the receiver and it worked.",
                  "Nora replaced the pump fuse. Flow returned to normal."]
    entries = []
    for index, (text, body) in enumerate(zip(texts, bodies)):
        message = {"id": f"message-{index}", "role": "user", "message_index": index, "content": text}
        source = {"known_origin": True, "start": index, "end": index,
                  "message_ids": [message["id"]], "selected_user_messages": [message],
                  "session_time": TIME.isoformat(),
                  "source_context": f"Session Time: {TIME.isoformat()}\nuser: {text}"}
        draft = validate_candidate(event(
            category=category, abstract=body, overview=body, content=body,
            provenance_ids=[f"archive:{index}"], evidence_quote=text,
        ))
        entries.append((_Span(index, index, occurrence_source=source), [draft]))
    return entries


@pytest.mark.parametrize("category", ["event", "case"])
def test_two_compact_source_singletons_reference_all_original_fields(category):
    entries = source_entries(category)[:2]
    original = deepcopy(entries)
    llm = SourceLLM(category=category, groups=[[0], [1]], mutate=compact_singletons)

    result = reconcile_source_occurrences(entries, CTX, "compact-archive", llm)

    assert len(result) == 2
    assert [replace(candidate, source_occurrence_id="") for candidate in result] == [
        batch[0] for _, batch in original]
    assert len({candidate.source_occurrence_id for candidate in result}) == 2
    assert [call[0] for call in llm.json_calls] == [{"groups"}]
    assert entries == original


@pytest.mark.parametrize("category", ["event", "case"])
def test_mixed_source_partition_verifies_complete_merge_and_retains_singleton(category):
    entries = source_entries(category)
    original = deepcopy(entries)
    llm = SourceLLM(category=category, groups=[[0, 1], [2]], mutate=compact_singletons)

    merged, singleton = reconcile_source_occurrences(entries, CTX, "compact-archive", llm)

    assert [call[0] for call in llm.json_calls] == [{"groups"}, {"valid"}]
    assert merged.provenance_ids == ["archive:0", "archive:1"]
    assert merged.evidence_quote == "\n".join(batch[0].evidence_quote for _, batch in entries[:2])
    facts = ["5 bar", "seal leak"] if category == "event" else ["cable", "rebooted", "worked"]
    assert all(fact in merged.content for fact in facts)
    assert replace(singleton, source_occurrence_id="") == original[2][1][0]
    verification = json.loads(llm.json_calls[1][1].split("\n\n", 1)[1])
    assert [draft["source_index"] for draft in verification["member_drafts"]] == [0, 1]
    assert [item["quote"] for item in verification["proposed_result"]["identity_evidence"]] == [
        batch[0].evidence_quote for _, batch in entries[:2]]
    assert entries == original


def test_compact_duplicate_replay_retains_stable_distinct_occurrence_ids():
    span, drafts = source_entries()[0]
    entries = [(span, [drafts[0], deepcopy(drafts[0])])]
    llm = SourceLLM(groups=[[0], [1]], mutate=compact_singletons)

    first = reconcile_source_occurrences(entries, CTX, "compact-archive", llm)
    second = reconcile_source_occurrences(entries, CTX, "compact-archive", llm)

    assert first == second
    assert len({candidate.source_occurrence_id for candidate in first}) == 2
    assert len({occurrence_replay_id(candidate) for candidate in first}) == 2
    assert [call[0] for call in llm.json_calls] == [{"groups"}, {"groups"}]


def test_legacy_complete_source_singletons_keep_original_candidates():
    entries = source_entries()[:2]

    def check_schema(response, _payload):
        validate(response, _SCHEMA)
        return response

    llm = SourceLLM(groups=[[0], [1]], mutate=check_schema)
    result = reconcile_source_occurrences(entries, CTX, "compact-archive", llm)

    assert [replace(candidate, source_occurrence_id="") for candidate in result] == [
        batch[0] for _, batch in entries]
    assert [call[0] for call in llm.json_calls] == [{"groups"}]


def test_legacy_repair_with_compact_neighbor_preserves_full_summary_and_facts():
    entries = source_entries()[:2]
    long_summary = "Nora observed pressure of 5 bar during the Atlas inspection. " * 5
    llm = Mock()
    legacy = SourceLLM(groups=[[0], [1]])
    responses = []

    def respond(prompt, schema):
        kind = set(schema["properties"])
        if kind == {"groups"}:
            response = legacy.complete_json(prompt, schema)
            response["groups"][0].update(abstract=long_summary, where="")
            response["groups"][1] = {"members": [1]}
            responses.append(deepcopy(response))
            return response
        if kind == {"repairs"}:
            return {"repairs": [{"group_index": 0, "abstract": "Atlas inspection pressure"}]}
        return {"valid": True}

    llm.complete_json.side_effect = respond
    repaired, singleton = reconcile_source_occurrences(entries, CTX, "compact-archive", llm)

    assert repaired.abstract == "Atlas inspection pressure"
    assert repaired.where is None
    assert long_summary in repaired.content
    assert repaired.provenance_ids == entries[0][1][0].provenance_ids
    assert repaired.evidence_quote == entries[0][1][0].evidence_quote
    assert replace(singleton, source_occurrence_id="") == entries[1][1][0]
    assert [set(call.kwargs["schema"]["properties"]) for call in llm.complete_json.call_args_list] == [
        {"groups"}, {"repairs"}, {"valid"}]
    verified = json.loads(llm.complete_json.call_args.args[0].split("\n\n", 1)[1])
    assert len(verified["member_drafts"]) == 1
    assert verified["proposed_result"]["content"] == repaired.content
    assert responses[0]["groups"][0]["abstract"] == long_summary


def test_event_compact_singletons_reference_validated_candidates():
    drafts = [validate_candidate(event()), validate_candidate(event(content="Nora inspected a pump."))]
    llm = Mock()
    response = {"groups": [{"members": [0]}, {"members": [1]}]}
    llm.complete_json.return_value = response

    result = reconcile_event_drafts(drafts, "Nora inspected Atlas and a separate pump.", llm)

    assert all(result[index] is draft for index, draft in enumerate(drafts))
    assert llm.complete_json.call_count == 1
    validate(response, llm.complete_json.call_args.kwargs["schema"])


def test_event_mixed_partition_verifies_merged_facts_and_keeps_original_singleton():
    drafts = [event(), event(content="The gauge measured 5 bar.", provenance_ids=["archive:2"]),
              event(content="Nora inspected a pump separately.")]
    response = proposal([[0, 1]])
    response["groups"].append({"members": [2]})
    llm = Mock()
    llm.complete_json.side_effect = [response, {"valid": True}]

    merged, singleton = reconcile_event_drafts(
        drafts, "Nora inspected Atlas once, measured 5 bar, then inspected a separate pump.", llm)

    assert merged.provenance_ids == ["archive:1", "archive:2"]
    assert merged.content == response["groups"][0]["content"]
    assert singleton is drafts[2]
    assert llm.complete_json.call_count == 2
    validate(response, llm.complete_json.call_args_list[0].kwargs["schema"])


class PartitionCheckpoint:
    def __init__(self):
        self.owner = threading.get_ident()
        self.records = {}

    def load_stage(self, name, inputs):
        assert threading.get_ident() == self.owner
        saved = self.records.get(name)
        if saved and saved["inputs"] == inputs:
            return deepcopy(saved["result"])
        return None

    def save_stage(self, name, inputs, result):
        assert threading.get_ident() == self.owner
        self.records[name] = json.loads(json.dumps({"inputs": inputs, "result": result}))


@pytest.mark.parametrize("groups", [[[0], [1], [2]], [[0, 1], [2]]])
def test_verified_source_partition_checkpoint_replays_on_owner_thread(groups):
    entries = source_entries()
    checkpoint = PartitionCheckpoint()
    llm = SourceLLM(groups=groups, mutate=compact_singletons)
    collections = []

    with stage_collector_scope(lambda: collections.append(threading.get_ident())):
        first = reconcile_source_occurrences(entries, CTX, "compact-archive", llm, checkpoint=checkpoint)
        first_calls = len(llm.json_calls)
        restored = reconcile_source_occurrences(entries, CTX, "compact-archive", llm, checkpoint=checkpoint)

    assert restored == first
    assert len(llm.json_calls) == first_calls == (2 if len(groups[0]) > 1 else 1)
    assert set(collections) == {checkpoint.owner}
    assert len(checkpoint.records) == 1
    saved = next(iter(checkpoint.records.values()))
    validate(saved["result"], _SCHEMA)
    payload = json.loads(saved["inputs"]["prompt"].split("\n\n", 1)[1])
    assert payload["sources"] == [span.occurrence_source for span, _ in entries]
    assert saved["inputs"]["schema"] == _SCHEMA


@pytest.mark.parametrize("change", ["source", "draft", "model"])
def test_source_checkpoint_keys_track_complete_generation_inputs(change):
    entries = source_entries()[:2]
    checkpoint = PartitionCheckpoint()
    llm = SourceLLM(groups=[[0], [1]], mutate=compact_singletons)
    reconcile_source_occurrences(entries, CTX, "compact-archive", llm, checkpoint=checkpoint)

    if change == "source":
        entries[0][0].occurrence_source["source_context"] += "\nuser: The inspection was at 09:00."
    elif change == "draft":
        entries[0][1][0] = replace(entries[0][1][0], title="Atlas pressure inspection")
    else:
        llm._model = "updated-model"

    result = reconcile_source_occurrences(entries, CTX, "compact-archive", llm, checkpoint=checkpoint)

    assert len(result) == 2
    assert [call[0] for call in llm.json_calls] == [{"groups"}, {"groups"}]
    assert len(checkpoint.records) == 2


@pytest.mark.parametrize("merged", [False, True])
def test_event_checkpoint_restores_verified_full_or_reference_partition(merged):
    drafts = [event(), event(content="The gauge measured 5 bar.")]
    response = proposal([[0, 1]]) if merged else {"groups": [{"members": [0]}, {"members": [1]}]}
    llm = Mock()
    llm.complete_json.side_effect = [response, {"valid": True}] if merged else [response]
    checkpoint = PartitionCheckpoint()

    first = reconcile_event_drafts(drafts, "Nora inspected Atlas once. Gauge: 5 bar.", llm,
                                   checkpoint=checkpoint)
    restored = reconcile_event_drafts(drafts, "Nora inspected Atlas once. Gauge: 5 bar.", llm,
                                      checkpoint=checkpoint)

    assert restored == first
    assert llm.complete_json.call_count == (2 if merged else 1)
    assert len(checkpoint.records) == 1


def test_repaired_singleton_checkpoint_restores_normalized_verified_facts():
    entries = source_entries()[:2]
    legacy = SourceLLM(groups=[[0], [1]])
    llm = Mock()

    def respond(prompt, schema):
        if set(schema["properties"]) == {"groups"}:
            response = legacy.complete_json(prompt, schema)
            response["groups"][0]["where"] = ""
            response["groups"][1] = {"members": [1]}
            return response
        return {"valid": True}

    llm.complete_json.side_effect = respond
    checkpoint = PartitionCheckpoint()
    first = reconcile_source_occurrences(entries, CTX, "compact-archive", llm, checkpoint=checkpoint)
    restored = reconcile_source_occurrences(entries, CTX, "compact-archive", llm, checkpoint=checkpoint)

    assert restored == first
    assert restored[0].where is None
    assert replace(restored[1], source_occurrence_id="") == entries[1][1][0]
    assert llm.complete_json.call_count == 2
