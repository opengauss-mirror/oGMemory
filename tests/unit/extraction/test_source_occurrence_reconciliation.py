"""Source-controlled fixtures exercise public extraction, writes and recovery."""

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

from commit.candidate_pipeline import CandidatePipeline
from commit.identity_resolver import occurrence_replay_id
from commit.write_receipts import RECEIPTS_KEY, WriteNotStartedError, operation_digest
from core.errors import ExtractionIncompleteError, LLMOutputTruncatedError, ValidationError
from core.models import ContextNode, RequestContext
from core.uri_resolver import URIResolver
from core.validation import validate_candidate
from extraction.schemas.registry import SchemaRegistry
from extraction.source_occurrence_reconciliation import reconcile_source_occurrences
from extraction.tools import Extractor, _Span
from providers.unified_config import OgMemConfig
from service.api import MemoryWriteAPI
from service.extraction_checkpoint import ExtractionBatchJournal
from session.session_manager import SessionManager
from tests.unit.commit.test_memory_identity import MemoryFS
from tests.unit.extraction.test_event_draft_boundaries import event

TIME = datetime(2026, 10, 4, tzinfo=UTC)
CTX = RequestContext("source-proof", "nora", "agent", "session", "trace")
EARLY = "I, Nora, inspected Atlas once at 09:00 on 2026-10-04 in Lab A for project Cedar. Pressure was 5 bar."
LATE = "Following up on that same single inspection: I also observed a seal leak. I did not inspect it again."


def messages(early=EARLY, late=LATE):
    return [{"id": "early", "role": "user", "content": early},
            *[{"id": f"bridge-{i}", "role": "assistant", "content": "Please continue."}
              for i in range(7)], {"id": "late", "role": "user", "content": late}]


class SourceLLM:
    """Controls model protocol only; actual parser, pipeline and writer run."""

    def __init__(self, *, category="event", identical=False, groups=None, verify=True,
                 same_span=False, mutate=None, outputs_per_run=(1, 1)):
        self.category, self.identical, self.groups = category, identical, groups
        self.verify, self.same_span, self.mutate = verify, same_span, mutate
        self.outputs_per_run = outputs_per_run
        self.json_calls, self.native_calls = [], []

    def complete_json(self, prompt, schema):
        properties = set(schema["properties"])
        self.json_calls.append((properties, prompt, schema))
        if properties == {"spans"}:
            return {"spans": ([{"start": 0, "end": 0, "categories": [self.category]}]
                              if self.same_span else [{"start": i, "end": i, "categories": [self.category]}
                                                      for i in (0, 8)])}
        payload = json.loads(prompt.split("\n\n", 1)[1])
        if properties == {"repair"}:
            return {"repair": None}
        if properties == {"valid"}:
            return {"valid": self.verify}
        assert properties == {"groups"}
        drafts = payload["drafts"]
        groups = self.groups or [list(range(len(drafts)))]
        result = {"groups": []}
        for members in groups:
            first = drafts[members[0]]["draft"]
            content = "Nora inspected Atlas once. Pressure was 5 bar. She also observed a seal leak."
            if self.category == "case":
                content = "Nora repaired the receiver once. She replaced the cable, rebooted it, and it worked."
            group = {"members": members, **{name: first[name] for name in
                     ("abstract", "overview", "content", "title", "when", "who", "where", "scope_ref")},
                     "identity_evidence": []}
            if len(members) > 1:
                group.update(abstract="Atlas inspection" if self.category == "event" else "Receiver repaired",
                             overview=content, content=content)
                for member in members:
                    source_index = drafts[member]["source_index"]
                    message = payload["sources"][source_index]["selected_user_messages"][0]
                    group["identity_evidence"].append({"member": member, "source_index": source_index,
                                                      "message_index": message["message_index"],
                                                      "quote": message["content"]})
            result["groups"].append(group)
        return self.mutate(result, payload) if self.mutate else result

    def complete_with_tools(self, *, prompt, tools, tool_choice, temperature=None):
        late = (LATE in prompt or "another Atlas" in prompt or "assistant suggested" in prompt) and not self.same_span
        self.native_calls.append((late, temperature))
        body = "Nora inspected Atlas."
        if not self.identical:
            body = "Nora observed a seal leak." if late else "Nora observed pressure of 5 bar."
            if "assistant suggested" in prompt:
                body = "Nora did not perform the suggested inspection."
        if self.category == "case":
            body = "Nora rebooted the receiver and it worked." if late else "Nora replaced the receiver cable."
        operation = {"tool": f"extract_{self.category}", "input": {
            "routing_key": "atlas_inspection", "abstract": body, "overview": body,
            "content": body, "confidence": .9, "title": "Atlas inspection",
            "when": "2026-10-04 09:00", "who": "Nora", "where": "Lab A",
            "scope_ref": "project Elm" if "another Atlas" in prompt else "project Cedar"}}
        return [deepcopy(operation) for _ in range(self.outputs_per_run[temperature == 0])]


def writer(llm, *, fs=None, enabled=True, mode="eager"):
    fs = fs or MemoryFS()
    return MemoryWriteAPI(fs, llm, directory_summary_enabled=False, extraction_mode=mode,
                          source_occurrence_reconciliation_enabled=enabled), fs


def extract(api, source, archive_id="source-archive"):
    return api._pipeline.extract(source, CTX, session_time=TIME, archive_id=archive_id,
                                 session_summary="Saved summary is not source evidence.",
                                 tool_stats_text="Statistics are not source evidence.")


@pytest.mark.parametrize("category", ["event", "case"])
def test_public_source_link_merges_complete_complementary_facts(category):
    llm = SourceLLM(category=category)
    api, fs = writer(llm)
    source = messages() if category == "event" else messages(
        "I, Nora, repaired the receiver once. I replaced its cable.",
        LATE + " For that same receiver repair, I rebooted it and it worked.")
    original = deepcopy(source)
    candidates = api._pipeline.deduplicate(api._pipeline.filter_by_confidence(extract(api, source)))
    assert len(candidates) == 1
    assert candidates[0].provenance_ids == ["prov:1:archive:source-archive:early",
                                           "prov:1:archive:source-archive:late"]
    assert len(candidates[0].source_occurrence_id) == 64
    result = api.write_memory(candidates[0], CTX)
    assert result["action"] == "create" and len(fs.nodes) == 1
    body = next(iter(fs.nodes.values())).content
    assert all(fact in body for fact in (["5 bar", "seal leak"] if category == "event"
                                       else ["cable", "rebooted", "worked"]))
    assert [call[0] for call in llm.json_calls] == [{"spans"}, {"groups"}, {"valid"}]
    partition = json.loads(llm.json_calls[1][1].split("\n\n", 1)[1])
    assert [draft["source_index"] for draft in partition["drafts"]] == [0, 1]
    assert partition["sources"][0]["selected_user_messages"][0]["id"] == "early"
    assert partition["sources"][1]["selected_user_messages"][0]["id"] == "late"
    for item in partition["sources"]:
        assert "Session Time" in item["source_context"] and "assistant" in item["source_context"]
        assert item["session_time"] == TIME.isoformat()
        assert "Saved summary" not in item["source_context"] and "Statistics" not in item["source_context"]
    assert source == original


@pytest.mark.parametrize("same_span", [False, True])
def test_identical_complete_drafts_for_two_real_activities_remain_two_public_nodes(same_span):
    llm = SourceLLM(identical=True, groups=[[0], [1]], same_span=same_span,
                    outputs_per_run=(2, 2) if same_span else (1, 1))
    api, fs = writer(llm)
    source = messages("I inspected Atlas twice today in Lab A: first at 09:00, then again at 09:00.",
                      "I inspected Atlas a second time; this is a different occurrence.")
    candidates = api._pipeline.deduplicate(extract(api, source))
    assert len(candidates) == 2
    assert candidates[0].content == candidates[1].content
    assert len({item.source_occurrence_id for item in candidates}) == 2
    assert len({occurrence_replay_id(item) for item in candidates}) == 2
    plans = [api.write_memory(item, CTX) for item in candidates]
    assert [plan["action"] for plan in plans] == ["create", "create"]
    assert len(fs.nodes) == 2 and len({node.content for node in fs.nodes.values()}) == 1
    assert [api.write_memory(item, CTX)["action"] for item in candidates] == ["skip", "skip"]
    assert [call[0] for call in llm.json_calls] == [{"spans"}, {"groups"}]


@pytest.mark.parametrize("source", [messages(late="I inspected another Atlas in project Elm."),
                                    messages(late="The assistant suggested a later inspection; I did not do it.")])
def test_scope_and_unperformed_suggestion_singletons_keep_original_facts(source):
    api, fs = writer(SourceLLM(groups=[[0], [1]]))
    candidates = extract(api, source)
    assert len(candidates) == 2 and candidates[0].content != candidates[1].content
    assert all(api.write_memory(item, CTX)["action"] == "create" for item in candidates)
    assert len(fs.nodes) == 2


def test_unknown_message_origin_preserves_exact_multiplicity_without_extra_model_calls():
    source = messages()
    for message in source:
        message.pop("id")
    llm = SourceLLM(identical=True)
    api, fs = writer(llm)
    candidates = api._pipeline.deduplicate(extract(api, source))
    assert len(candidates) == 2 and len({item.source_occurrence_id for item in candidates}) == 2
    assert [call[0] for call in llm.json_calls] == [{"spans"}]
    assert all(api.write_memory(item, CTX)["action"] == "create" for item in candidates)
    assert len(fs.nodes) == 2


@pytest.mark.parametrize("enabled,mode", [(False, "eager"), (True, "lazy")])
def test_default_off_and_declared_lazy_even_without_lazy_dependencies_add_no_new_calls(enabled, mode):
    llm = SourceLLM()
    api, _ = writer(llm, enabled=enabled, mode=mode)
    assert len(extract(api, messages())) == 2
    assert not api._pipeline._extractors[0]._use_lazy_mode()
    assert [call[0] for call in llm.json_calls] == [{"spans"}]


def test_real_lazy_react_path_with_explicit_flag_makes_no_new_source_group_calls():
    class LazyLLM(SourceLLM):
        def complete_with_tools_messages(self, *, messages, tools=None, tool_choice="auto"):
            operations = self.complete_with_tools(prompt=messages[-1]["content"], tools=tools,
                                                   tool_choice=tool_choice)
            return [], json.dumps(operations)

    llm = LazyLLM()
    fs = MemoryFS()
    registry = SchemaRegistry()
    api = MemoryWriteAPI(fs, llm, schema_registry=registry, uri_resolver=URIResolver(registry),
                         extraction_mode="lazy", source_occurrence_reconciliation_enabled=True,
                         directory_summary_enabled=False)
    assert api._pipeline._extractors[0]._use_lazy_mode()
    assert len(extract(api, messages())) == 2
    assert len(llm.native_calls) == 4
    assert [call[0] for call in llm.json_calls] == [{"spans"}]


@pytest.mark.parametrize("ranges", [(4, 14), (4, 4, 14), (4, 14, 4), (14, 4, 14)])
def test_identical_extraction_prompts_from_distinct_ranges_do_not_suppress_two_occurrences(ranges):
    class RepeatedPromptLLM(SourceLLM):
        def complete_json(self, prompt, schema):
            if set(schema["properties"]) == {"spans"}:
                self.json_calls.append(({"spans"}, prompt, schema))
                return {"spans": [{"start": i, "end": i, "categories": ["event"]} for i in ranges]}
            return super().complete_json(prompt, schema)

    source = [{"id": f"message-{i}", "role": "user" if i in (4, 14) else "assistant",
               "content": ("I performed another separate inspection of Atlas at 09:00 today in Lab A."
                           if i in (4, 14) else "Please continue.")} for i in range(21)]
    llm = RepeatedPromptLLM(identical=True, groups=[[0], [1]])
    api, fs = writer(llm)
    extractor = api._pipeline._extractors[0]
    prepared = extractor._prepare_unique_spans([_Span(i, i) for i in ranges], source, "en", TIME, "", "", CTX)
    assert len(prepared) == 2 and prepared[0].prompt_sha == prepared[1].prompt_sha
    candidates = api._pipeline.deduplicate(extract(api, source))
    assert len(candidates) == 2 and len(llm.native_calls) == 4
    assert all(api.write_memory(candidate, CTX)["action"] == "create" for candidate in candidates)
    assert len(fs.nodes) == 2


@pytest.mark.parametrize("enabled,mode", [(False, "eager"), (True, "lazy")])
def test_defaultoff_and_declared_lazy_preserve_legacy_prompt_seen_key(enabled, mode):
    source = [{"id": f"message-{i}", "role": "user" if i in (4, 14) else "assistant",
               "content": "I inspected Atlas." if i in (4, 14) else "Please continue."} for i in range(21)]
    api, _ = writer(Mock(), enabled=enabled, mode=mode)
    extractor = api._pipeline._extractors[0]
    spans = [_Span(i, i, message_ids=[f"message-{i}"]) for i in (4, 14, 4)]
    prepared = extractor._prepare_unique_spans(spans, source, "en", TIME, "", "", CTX)
    assert len(prepared) == 1
    assert prepared[0].span.message_ids == ["message-4", "message-14"]


@pytest.mark.parametrize("enabled,expected", [(True, ["first prompt", "second prompt"]),
                                             (False, ["first prompt", "second prompt", "first prompt"])])
def test_actual_prompt_collision_comparison_preserves_distinct_text_and_gate_compatibility(
    enabled, expected, monkeypatch, caplog,
):
    api, _ = writer(Mock(), enabled=enabled)
    extractor = api._pipeline._extractors[0]
    original = extractor._prepare_span
    prompts = iter(["first prompt", "second prompt", "first prompt"])

    def prepare(*args, **kwargs):
        prepared = original(*args, **kwargs)
        return replace(prepared, prompt=next(prompts), prompt_sha="a" * 64)

    monkeypatch.setattr(extractor, "_prepare_span", prepare)
    source = [{"id": "message-one", "role": "user", "content": EARLY}]
    # Exercise the final-prompt collision guard with already focused spans;
    # raw identical source ranges coalesce before prompt preparation.
    prepared = extractor._prepare_unique_spans([_Span(0, 0, focused_messages=source) for _ in range(3)],
                                               source, "en", TIME, "", "", CTX)
    assert [item.prompt for item in prepared] == expected
    assert "prompt_sha collision detected" in caplog.text


def test_same_span_case_complementary_steps_and_full_fallback_origin_are_reconciled():
    class FallbackCaseLLM(SourceLLM):
        def complete_json(self, prompt, schema):
            if set(schema["properties"]) == {"spans"}:
                self.json_calls.append(({"spans"}, prompt, schema))
                return {"spans": []}
            return super().complete_json(prompt, schema)

        def complete_with_tools(self, **kwargs):
            operations = super().complete_with_tools(**kwargs)
            if kwargs.get("temperature") == 0:
                body = "Nora rebooted the receiver and it worked."
                operations[0]["input"].update(abstract=body, overview=body, content=body)
            return operations

    source = [{"id": "repair-one", "role": "user", "content":
               "I, Nora, repaired the receiver once on 2026-10-04 at 09:00 in Lab A for project Cedar. "
               "I replaced the cable, rebooted the receiver, and it worked. The following details all "
               "describe that same single resolved receiver problem; I did not repair it another time."}]
    llm = FallbackCaseLLM(category="case", same_span=True)
    api, fs = writer(llm)
    candidates = extract(api, source)
    assert len(candidates) == 1 and candidates[0].provenance_ids == ["prov:1:archive:source-archive:repair-one"]
    assert all(step in candidates[0].content for step in ["cable", "rebooted", "worked"])
    assert api.write_memory(candidates[0], CTX)["action"] == "create" and len(fs.nodes) == 1
    partition = json.loads(llm.json_calls[1][1].split("\n\n", 1)[1])
    assert partition["sources"][0]["message_ids"] == ["repair-one"]


@pytest.mark.parametrize("mutation", [
    lambda result, _: {"groups": []},
    lambda result, _: {"groups": [dict(result["groups"][0], members=[0])]},
    lambda result, _: {"groups": [dict(result["groups"][0], members=[0, 0, 1])]},
    lambda result, _: {"groups": [dict(result["groups"][0], members=[0, 2])]},
    lambda result, _: {"groups": [dict(result["groups"][0], identity_evidence=[])]},
    lambda result, _: {"groups": [dict(result["groups"][0], identity_evidence=[{
        "member": 0, "source_index": 0, "message_index": 1, "quote": "Please continue."}])]},
    lambda result, _: {"groups": [dict(result["groups"][0], identity_evidence=[{
        "member": 0, "source_index": 1, "message_index": 8, "quote": LATE}])]},
    lambda result, _: {"groups": [dict(result["groups"][0], content="x" * 100001)]},
])
def test_bad_partition_or_unsupported_evidence_fails_before_any_public_write(mutation):
    api, fs = writer(SourceLLM(mutate=mutation))
    with pytest.raises(ExtractionIncompleteError):
        api.commit_session(messages(), CTX, session_time=TIME, archive_id="source-archive")
    assert fs.nodes == {}


def test_independent_verifier_rejection_stops_all_new_writes():
    api, fs = writer(SourceLLM(verify=False))
    with pytest.raises(ExtractionIncompleteError, match="could not be verified"):
        api.commit_session(messages(), CTX, session_time=TIME, archive_id="source-archive")
    assert fs.nodes == {}


def test_complete_escaped_source_budget_overflow_never_calls_partition_or_writes():
    llm = SourceLLM()
    api, fs = writer(llm)
    source = messages(EARLY + '\\"\n' * 80000)
    with pytest.raises(ExtractionIncompleteError, match="complete context budget"):
        api.commit_session(source, CTX, session_time=TIME, archive_id="source-archive")
    assert [call[0] for call in llm.json_calls] == [{"spans"}] and fs.nodes == {}


def test_partition_output_length_failure_is_explicit_and_zero_write():
    llm = SourceLLM()
    original = llm.complete_json

    def complete(prompt, schema):
        if set(schema["properties"]) == {"groups"}:
            raise LLMOutputTruncatedError("finish_reason=length")
        return original(prompt, schema)

    llm.complete_json = complete
    api, fs = writer(llm)
    with pytest.raises(ExtractionIncompleteError, match="finish_reason=length"):
        api.commit_session(messages(), CTX, session_time=TIME, archive_id="source-archive")
    assert fs.nodes == {}


def test_accepted_checkpoint_retry_preserves_two_partition_ids_and_does_not_use_a_model():
    source = messages("I inspected Atlas twice today.", "I inspected Atlas again; a second occurrence.")
    llm = SourceLLM(identical=True, groups=[[0], [1]])
    api, fs = writer(llm)
    manager = SessionManager(get_context_fs=lambda: fs, session_state_bridge_enabled=False)
    for message in source:
        manager.add_message(CTX.session_id, message["role"], message["content"], CTX,
                            created_at=TIME.isoformat(), message_id=message["id"])
    buffer = manager.get_or_create(CTX.session_id, CTX)

    def journal(manager, buffer):
        return ExtractionBatchJournal(manager, buffer, CTX, buffer.messages,
                                      archive_id="source-archive", messages=source,
                                      session_time=TIME, session_summary="", tool_stats_text="")

    checkpoint = journal(manager, buffer).checkpoint("structured")
    first = api.commit_session(source, CTX, session_time=TIME, archive_id="source-archive",
                               candidate_checkpoint=checkpoint)
    accepted, _, _ = checkpoint.load()
    assert first["writes_completed"] == 2
    assert len({candidate.source_occurrence_id for candidate in accepted}) == 2
    assert len({candidate.write_operation_id for candidate in accepted}) == 2
    assert {operation for node in fs.nodes.values() if node.category == "event"
            for operation in node.metadata[RECEIPTS_KEY]} == {candidate.write_operation_id for candidate in accepted}
    stored = deepcopy(fs.nodes)
    no_model = Mock()
    no_model.complete_json.side_effect = AssertionError("Accepted retry must not regroup")
    no_model.complete_with_tools.side_effect = AssertionError("Accepted retry must not extract")
    restarted, _ = writer(no_model, fs=fs)
    restored_manager = SessionManager(get_context_fs=lambda: fs, session_state_bridge_enabled=False)
    restored_buffer = restored_manager.get_or_create(CTX.session_id, CTX)
    restored = journal(restored_manager, restored_buffer).checkpoint("structured")
    second = restarted.commit_session(source, CTX, session_time=TIME, archive_id="source-archive",
                                      candidate_checkpoint=restored)
    assert [(plan["action"], plan["target_uri"]) for plan in second["plans"]] == [
        (plan["action"], plan["target_uri"]) for plan in first["plans"]]
    assert restored.load()[0] == accepted and fs.nodes == stored
    no_model.complete_json.assert_not_called()
    no_model.complete_with_tools.assert_not_called()


def test_bound_accepted_and_partitioned_drafts_never_enter_a_new_partition():
    drafts = [event(target_uri="ctx://other/users/other/memories/events/unsafe", expected_version=1),
              event(write_operation_id="accepted-operation"), event(source_occurrence_id="a" * 64)]
    llm = Mock()
    span = _Span(0, 0)
    assert reconcile_source_occurrences([(span, drafts)], CTX, "source-archive", llm) == drafts
    llm.complete_json.assert_not_called()
    assert CandidatePipeline().deduplicate([drafts[1], replace(drafts[1], write_operation_id="second")]) == [
        drafts[1], replace(drafts[1], write_operation_id="second")]


def test_empty_internal_field_preserves_legacy_replay_and_receipt_fingerprints():
    candidate = event()
    old = asdict(candidate)
    old.pop("source_occurrence_id")
    old.pop("verified_routing_aliases")
    old.pop("verified_object_aliases")
    old.pop("source_contexts")
    old.pop("node_draft_indices")
    for name in ("routing_key", "legacy_routing_key", "target_uri", "expected_version", "confidence",
                 "provenance_ids", "complete_occurrence_context", "write_operation_id"):
        old.pop(name)
    assert occurrence_replay_id(candidate) == hashlib.sha256(
        json.dumps(old, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    old = asdict(candidate)
    old.pop("source_occurrence_id")
    old.pop("verified_routing_aliases")
    old.pop("verified_object_aliases")
    old.pop("source_contexts")
    old.pop("node_draft_indices")
    old.pop("write_operation_id")
    evidence = {"candidate": old, "account_id": CTX.account_id, "user_id": CTX.user_id,
                "agent_id": CTX.agent_id, "session_id": CTX.session_id}
    assert operation_digest(candidate, CTX) == hashlib.sha256(
        json.dumps(evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def test_old_field_checkpoint_and_old_receipt_upgrade_retry_without_model_or_new_node():
    api, fs = writer(Mock(), enabled=False)
    manager = SessionManager(get_context_fs=lambda: fs, session_state_bridge_enabled=False)
    manager.add_message(CTX.session_id, "user", EARLY, CTX,
                        created_at=TIME.isoformat(), message_id="early")
    buffer = manager.get_or_create(CTX.session_id, CTX)
    source = messages()[:1]
    journal = ExtractionBatchJournal(manager, buffer, CTX, buffer.messages,
                                    archive_id="legacy-archive", messages=source,
                                    session_time=TIME, session_summary="", tool_stats_text="")
    accepted = replace(event(attributed_speaker=None, attribution_basis=None),
                       write_operation_id=hashlib.sha256(b"legacy-archive:structured:0").hexdigest())
    legacy_candidate = asdict(accepted)
    legacy_candidate.pop("source_occurrence_id")
    legacy_candidate.pop("verified_routing_aliases")
    legacy_candidate.pop("verified_object_aliases")
    legacy_candidate.pop("source_contexts")
    legacy_candidate.pop("node_draft_indices")
    legacy_payload = {key: value for key, value in legacy_candidate.items() if key != "write_operation_id"}
    legacy_digest = hashlib.sha256(json.dumps(
        {"candidate": legacy_payload, "account_id": CTX.account_id, "user_id": CTX.user_id,
         "agent_id": CTX.agent_id, "session_id": CTX.session_id},
        ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    plan = api.write_memory(accepted, CTX)
    node = fs.nodes[plan["target_uri"]]
    assert node.metadata[RECEIPTS_KEY][accepted.write_operation_id] == legacy_digest
    # Independently construct a persisted prior-format accepted record: no new field,
    # prior canonical digest, and a write acknowledgement lost before checkpointing.
    record = {"candidates": [legacy_candidate], "candidate_digests": [legacy_digest],
              "extracted": 1, "filtered": 0, "operations": {"0": {"status": "writing"}}}
    journal.mutate(lambda batch: batch["slots"].update(structured=record))
    stored = deepcopy(fs.nodes)
    no_model = Mock()
    no_model.complete_json.side_effect = AssertionError("Legacy accepted retry must not regroup")
    no_model.complete_with_tools.side_effect = AssertionError("Legacy accepted retry must not extract")
    restarted, _ = writer(no_model, fs=fs)
    restored_manager = SessionManager(get_context_fs=lambda: fs, session_state_bridge_enabled=False)
    restored_buffer = restored_manager.get_or_create(CTX.session_id, CTX)
    restored_journal = ExtractionBatchJournal(restored_manager, restored_buffer, CTX, restored_buffer.messages,
                                             archive_id="legacy-archive", messages=source,
                                             session_time=TIME, session_summary="", tool_stats_text="")
    checkpoint = restored_journal.checkpoint("structured")
    assert checkpoint.load()[0] == [accepted]
    result = restarted.commit_session(source, CTX, session_time=TIME, archive_id="legacy-archive",
                                      candidate_checkpoint=checkpoint)
    assert result["writes_skipped"] == 1 and result["plans"][0]["target_uri"] == plan["target_uri"]
    assert {uri: node for uri, node in fs.nodes.items() if node.category == "event"} == {
        uri: node for uri, node in stored.items() if node.category == "event"}
    assert set(fs.nodes) == set(stored)
    no_model.complete_json.assert_not_called()
    no_model.complete_with_tools.assert_not_called()


def test_source_partition_identity_is_stable_with_archive_and_frozen_after_acceptance():
    api, _ = writer(SourceLLM(groups=[[0], [1]]))
    first = extract(api, messages())
    second = extract(api, messages())
    assert [item.source_occurrence_id for item in first] == [item.source_occurrence_id for item in second]
    another = extract(api, messages(), archive_id="other-archive")
    assert {item.source_occurrence_id for item in first}.isdisjoint(item.source_occurrence_id for item in another)
    unknown_first = extract(api, messages(), archive_id=None)
    unknown_second = extract(api, messages(), archive_id=None)
    assert {item.source_occurrence_id for item in unknown_first}.isdisjoint(
        item.source_occurrence_id for item in unknown_second)


@pytest.mark.parametrize("category", ["event", "case"])
@pytest.mark.parametrize("change", ["name", "unrelated-draft"])
def test_source_replay_preserves_multiplicity_across_output_changes(category, change):
    class RetryLLM(SourceLLM):
        retry = False

        def complete_with_tools(self, **kwargs):
            calls = super().complete_with_tools(**kwargs)
            if self.retry:
                if change == "name":
                    for call in calls:
                        call["input"]["routing_key"] = "new_generated_name"
                else:
                    extra = deepcopy(calls[0])
                    extra["input"].update(routing_key="other_activity", content="Nora inspected a pump.")
                    calls.insert(0, extra)
            return calls

    llm = RetryLLM(category=category, groups=[[0], [1]], same_span=True, outputs_per_run=(2, 2))
    api, fs = writer(llm)
    source = [{"id": "source-one", "role": "user", "content":
               "Nora performed two separate Atlas inspections, replaced receiver cables twice, "
               "and inspected a pump. These were distinct activities."}]
    first = extract(api, source)
    assert len(first) == 2
    assert len({item.source_occurrence_id for item in first}) == 2
    assert [api.write_memory(item, CTX)["action"] for item in first] == ["create", "create"]

    llm.retry = True
    if change == "unrelated-draft":
        llm.groups = [[0], [1], [2]]
    retried = extract(api, source)
    matching = [item for item in retried if item.content == first[0].content]
    assert len(retried) == (3 if change == "unrelated-draft" else 2)
    assert [item.source_occurrence_id for item in matching] == [item.source_occurrence_id for item in first]
    assert [api.write_memory(item, CTX)["action"] for item in matching] == ["skip", "skip"]
    assert len(fs.nodes) == 2


@pytest.mark.parametrize("category", ["event", "case"])
def test_retry_without_session_time_reuses_source_partition_and_node(monkeypatch, category):
    from extraction import tools

    times = iter([TIME, TIME.replace(day=5)])

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return next(times)

    monkeypatch.setattr(tools, "datetime", Clock)
    api, fs = writer(SourceLLM(category=category))
    source = messages() if category == "event" else messages(
        "I, Nora, repaired the receiver once. I replaced its cable.",
        LATE + " For that same receiver repair, I rebooted it and it worked.")
    first = api._pipeline.extract(source, CTX, archive_id="source-archive")
    second = api._pipeline.extract(source, CTX, archive_id="source-archive")

    assert [item.source_occurrence_id for item in first] == [item.source_occurrence_id for item in second]
    assert [api.write_memory(item, CTX)["action"] for item in first] == ["create"]
    assert [api.write_memory(item, CTX)["action"] for item in second] == ["skip"]
    assert len(fs.nodes) == 1


def test_partition_identifier_cannot_bypass_unsafe_target_or_version_authorization():
    api, fs = writer(Mock())
    uri = "ctx://other/users/other/memories/events/unsafe"
    fs.nodes[uri] = ContextNode(uri, "MEMORY", "event", 0, "user:other", "Existing", "Existing",
                                "Existing", metadata={"version": 1})
    original = deepcopy(fs.nodes)
    candidate = event(source_occurrence_id="a" * 64, write_operation_id="accepted",
                      target_uri=uri, expected_version=1)
    with pytest.raises(WriteNotStartedError, match="authorized category/owner"):
        api.write_memory(candidate, CTX)
    assert fs.nodes == original


def test_complete_verification_budget_overflow_stops_before_verifier_and_writes(monkeypatch):
    from extraction import source_occurrence_reconciliation as reconciliation

    fits = iter([True, False])
    monkeypatch.setattr(reconciliation, "prompt_fits", lambda *_: next(fits))
    llm = SourceLLM()
    api, fs = writer(llm)
    with pytest.raises(ExtractionIncompleteError, match="verification exceeds"):
        api.commit_session(messages(), CTX, session_time=TIME, archive_id="source-archive")
    assert [call[0] for call in llm.json_calls] == [{"spans"}, {"groups"}] and fs.nodes == {}


@pytest.mark.parametrize("changes", [{"owner_scope": "agent"}, {"category": "case"},
                                     {"attributed_speaker": "Ellis"}, {"attribution_basis": "self_named"}])
def test_strict_category_owner_speaker_and_attribution_buckets_never_cross(changes):
    span = _Span(0, 0, occurrence_source={
        "known_origin": True, "source_context": "user: " + EARLY,
        "selected_user_messages": [{"id": "early", "message_index": 0, "role": "user", "content": EARLY}]})
    llm = Mock()
    drafts = [event(), event(**changes)]
    result = reconcile_source_occurrences([(span, drafts)], CTX, "source-archive", llm)
    assert [replace(candidate, source_occurrence_id="") for candidate in result] == drafts
    assert len({candidate.source_occurrence_id for candidate in result}) == 2
    llm.complete_json.assert_not_called()


@pytest.mark.parametrize("value", ["forged", None, 1])
def test_invalid_internal_partition_digest_is_rejected(value):
    with pytest.raises(ValidationError, match="source_occurrence_id"):
        validate_candidate(event(source_occurrence_id=value))


def test_native_tool_cannot_assign_internal_partition_or_operation_identity():
    extractor = Extractor(Mock())
    result = extractor._to_candidate({"tool": "extract_event", "input": {
        "routing_key": "atlas", "abstract": "Atlas inspected", "overview": "Atlas inspected",
        "content": "Atlas inspected", "confidence": .9, "title": "Atlas inspected", "scope_ref": "",
        "source_occurrence_id": "a" * 64, "write_operation_id": "forged"}})
    assert result is not None and not result.source_occurrence_id and not result.write_operation_id


def test_config_resolves_explicit_opt_in_and_defaults_off(monkeypatch):
    from providers import unified_config

    monkeypatch.delenv("OGMEM_SOURCE_OCCURRENCE_RECONCILIATION_ENABLED", raising=False)
    monkeypatch.setattr(unified_config, "_load_yaml", lambda _: {})
    assert OgMemConfig.load("ignored").source_occurrence_reconciliation_enabled is False
    monkeypatch.setattr(unified_config, "_load_yaml", lambda _: {"memory": {"extraction": {
        "source_occurrence_reconciliation_enabled": True}}})
    assert OgMemConfig.load("ignored").source_occurrence_reconciliation_enabled is True
    monkeypatch.setenv("OGMEM_SOURCE_OCCURRENCE_RECONCILIATION_ENABLED", "false")
    monkeypatch.setattr(unified_config, "_load_yaml", lambda _: {})
    assert OgMemConfig.load("ignored").source_occurrence_reconciliation_enabled is False
    monkeypatch.setenv("OGMEM_SOURCE_OCCURRENCE_RECONCILIATION_ENABLED", "true")
    assert OgMemConfig.load("ignored").source_occurrence_reconciliation_enabled is True
    with pytest.raises(ValueError, match="must be a boolean"):
        OgMemConfig(source_occurrence_reconciliation_enabled="false")


def test_api_prompt_manager_fallback_preserves_the_explicit_source_flag(monkeypatch):
    from extraction import prompts

    original = prompts.PromptManager
    calls = []

    def prompt_manager():
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("Template temporarily unavailable")
        return original()

    monkeypatch.setattr(prompts, "PromptManager", prompt_manager)
    api, _ = writer(Mock())
    assert len(calls) == 2
    assert api._pipeline._extractors[0]._reconcile_source_occurrences_enabled()


def test_memory_service_factory_forwards_the_explicit_source_flag(monkeypatch):
    from server.memory_service import MemoryService

    service = MemoryService(config=OgMemConfig(provider="mock", storage_backend="agfs",
                                             source_occurrence_reconciliation_enabled=True))
    monkeypatch.setattr(service, "_get_context_fs", lambda: MemoryFS())
    monkeypatch.setattr(service, "_get_outbox_store", lambda fs: Mock())
    monkeypatch.setattr(service, "get_llm", lambda: Mock())
    monkeypatch.setattr(service, "_get_shared_vector_index", lambda: None)
    monkeypatch.setattr(service, "_get_shared_embedder", lambda: None)
    monkeypatch.setattr(service, "_start_outbox_worker", lambda: None)
    api = service.get_write_api()
    assert api._source_occurrence_reconciliation_enabled
    assert api._pipeline._extractors[0]._reconcile_source_occurrences_enabled()
