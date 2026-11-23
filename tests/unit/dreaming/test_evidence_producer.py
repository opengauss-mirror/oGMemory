from dataclasses import replace
from unittest.mock import Mock

import pytest

from commit.archive_builder import ArchiveBuilder
from commit.dreaming_facts import assess_facts
from commit.identity_resolver import MemoryIdentityResolver
from core.dreaming_evidence import EVIDENCE_INPUT, EVIDENCE_KEY, bind_archive_evidence
from core.models import CandidateMemory, RequestContext, WritePlan
from core.provenance_resolver import ProvenanceResolver
from extraction.event_reconciliation import reconcile_event_drafts
from extraction.schemas.registry import SchemaRegistry
from extraction.tool_builder import build_extraction_tools
from extraction.tools import Extractor
from service.api import MemoryWriteAPI
from session.models import ArchiveEntry
from tests.unit.commit.test_memory_identity import MemoryFS


def _extract(registry, archive, claim, ctx, *, evidence=True):
    quote = archive.messages[0]["content"]
    llm = Mock()
    llm.complete_json.return_value = {"spans": [{"start": 0, "end": 0, "reason": "case", "categories": ["case"]}]}
    payload = {
        "routing_key": "field-trial", "abstract": claim, "overview": claim, "content": claim,
        "who": "Alex", "confidence": .9,
    }
    if evidence:
        payload[EVIDENCE_INPUT] = [{"claim": claim, "subject": "Alex", "quote": quote, "relation": "supports"}]
    llm.complete_with_tools.return_value = [{"tool": "extract_case", "input": payload}]
    candidate, = Extractor(llm, schema_registry=registry).extract(archive.messages, ctx, archive_id=archive.archive_id)
    return candidate, (llm.complete_json.call_count, llm.complete_with_tools.call_count)


def test_scoped_extraction_evidence_is_verified_after_create_and_update(monkeypatch):
    monkeypatch.setenv("OGMEM_DREAMING_ENABLED", "true")
    registry = SchemaRegistry()
    tools = {tool["name"]: tool["input_schema"] for tool in build_extraction_tools(registry)}
    assert tools["extract_case"]["properties"][EVIDENCE_INPUT]["items"]["required"] == [
        "claim", "subject", "quote", "relation",
    ]
    claim = "The field trial is complete."
    archives = {
        key: ArchiveEntry(key, "session", "", "", [{"id": key, "role": "user", "content": claim}])
        for key in ("one", "two")
    }
    ctx = RequestContext("account", "Alex", "agent", "session", "trace")
    with monkeypatch.context() as baseline:
        baseline.setenv("OGMEM_DREAMING_ENABLED", "false")
        _, model_calls = _extract(registry, archives["one"], claim, ctx, evidence=False)
    (first, first_calls), (next_candidate, next_calls) = [
        _extract(registry, archive, claim, ctx) for archive in archives.values()
    ]
    assert first_calls == next_calls == model_calls
    builder = ArchiveBuilder(None)
    uri = "ctx://account/agents/agent/memories/cases/field-trial"
    created = builder.build(first, WritePlan("create", uri), ctx)
    assert assess_facts(created, read_source=archives.get)["confidence"] == .8

    existing = replace(created, metadata={**created.metadata, "version": 1})
    plan = WritePlan("merge", uri, {
        "identity_metadata": {**existing.metadata, "provenance_ids": [
            *created.metadata["provenance_ids"], *next_candidate.provenance_ids,
        ]},
        "expected_version": existing.metadata["version"],
    })
    updated = builder.build(next_candidate, plan, ctx)
    result = assess_facts(updated, read_source=archives.get)
    assert updated.metadata["expected_version"] == 1
    assert [row["message_id"] for row in updated.metadata[EVIDENCE_KEY]] == ["one", "two"]
    assert (result["confidence"], result["recommendation"]) == (64 / 65, "recommended")


@pytest.mark.parametrize("first", [
    "Alex opened the field trial.",
    "Alex completed the field trial\nwith a successful safety check.",
])
def test_merged_event_retains_each_source_backed_fact(first):
    next_claim = "Alex verified the safety checklist."
    claims = (first, next_claim)
    messages = [{"id": name, "role": "user", "content": claim}
                for name, claim in zip(("one", "two"), claims)]
    archive = ArchiveEntry("archive", "session", "", "", messages)
    drafts = []
    for message, claim in zip(messages, claims):
        source = ProvenanceResolver.build_id("archive", archive.archive_id, [message["id"]])
        policy = [{"claim": claim, "subject": "Alex", "quote": claim, "relation": "supports"}]
        drafts.append(CandidateMemory("event", "user", "field_trial", claim, claim, claim, .9, who="Alex",
                                      provenance_ids=[source],
                                      metadata={EVIDENCE_KEY: bind_archive_evidence(policy, source, messages)}))
    llm = Mock()
    llm.complete_json.side_effect = [{"groups": [{
        "members": [0, 1], "abstract": "Field trial", "overview": "Two facts about the field trial",
        "content": "\n".join(claims), "title": "Field trial", "who": "Alex",
        "when": None, "where": None, "scope_ref": "",
    }]}, {"valid": True}]
    candidate, = reconcile_event_drafts(drafts, "\n".join(claims), llm)
    ctx = RequestContext("account", "Alex", "agent", "session", "trace")
    node = ArchiveBuilder(llm).build(candidate, WritePlan("create", "ctx://account/users/Alex/memories/events/field"), ctx)

    review = assess_facts(node, read_source={archive.archive_id: archive}.get)
    assert [fact["content"] for fact in review["facts"].values()] == list(claims)
    assert [fact["confidence"] for fact in review["facts"].values()] == [.8, .8]
    assert review["confidence"] == .8


def test_preference_holder_and_evidence_share_the_resolved_self_identity():
    claim = "Prefers a printed work schedule"
    user_quote, other_quote = "I prefer a printed work schedule.", "Alex prefers a printed work schedule."
    messages = [{"id": name, "role": "user", "content": quote}
                for name, quote in (("self", user_quote), ("other", other_quote))]
    archive = ArchiveEntry("archive", "session", "", "", messages)
    source = ProvenanceResolver.build_id("archive", archive.archive_id, ["self", "other"])
    rows = [{"claim": claim, "subject": subject, "quote": quote, "relation": "supports"}
            for subject, quote in (("user", user_quote), ("Alex", other_quote))]
    candidate = CandidateMemory("preference", "user", "schedule", claim, f"- {claim}",
                                f"# Schedule\n- {claim}", .9, who="user", provenance_ids=[source],
                                metadata={EVIDENCE_KEY: bind_archive_evidence(rows, source, messages)})
    ctx = RequestContext("account", "u123", "agent", "session", "trace")
    resolved = MemoryIdentityResolver._normalize_subject(candidate, ctx)
    uri = "ctx://account/users/u123/memories/preferences/schedule"
    node = ArchiveBuilder(None).build(resolved, WritePlan("create", uri), ctx)

    review = assess_facts(node, read_source={archive.archive_id: archive}.get)
    fact, = review["facts"].values()
    assert node.metadata["who"] == "u123"
    assert [proof["message_id"] for proof in fact["evidence"]] == ["self"]
    assert review["confidence"] == .8


def _write_evidence_fixture():
    claims = ("Lives in Paris.", "Works as an engineer.")
    messages = [{"id": name, "role": "user", "content": claim}
                for name, claim in zip(("one", "two"), claims)]
    archive = ArchiveEntry("archive", "session", "", "", messages)
    sources = [ProvenanceResolver.build_id("archive", archive.archive_id, [message["id"]]) for message in messages]
    rows = [bind_archive_evidence([{"claim": claim, "subject": "Alex", "quote": claim, "relation": "supports"}],
                                  ref, messages) for ref, claim in zip(sources, claims)]
    return claims, archive, sources, rows


def test_merged_memory_heading_is_context_for_scored_facts():
    claims, archive, sources, rows = _write_evidence_fixture()
    text = "# Trial\n" + "\n".join(f"- {claim}" for claim in claims)
    candidate = CandidateMemory("event", "user", "trial", "Trial", "Trial", text, .9, who="Alex",
                                provenance_ids=sources, metadata={EVIDENCE_KEY: rows[0] + rows[1]})
    ctx = RequestContext("account", "Alex", "agent", "session", "trace")
    node = ArchiveBuilder(None).build(candidate, WritePlan("create", "ctx://account/users/Alex/memories/events/trial"), ctx)

    review = assess_facts(node, read_source={archive.archive_id: archive}.get)
    assert [(fact["abstract"], fact["confidence"]) for fact in review["facts"].values()] == [
        (claim, .8) for claim in claims
    ]
    assert all(fact["content"].startswith("# Trial\n") for fact in review["facts"].values())
    assert review["confidence"] == .8


@pytest.mark.parametrize("category", ["profile", "pattern"])
def test_semantic_write_merge_keeps_source_backed_atoms(category):
    claims, archive, sources, rows = _write_evidence_fixture()
    stored, incoming = (f"- {claim}" for claim in claims)
    ctx = RequestContext("account", "Alex", "agent", "session", "trace")
    candidate = CandidateMemory(category, "user", "work", claims[1], incoming, incoming, .9, who="Alex",
                                provenance_ids=[sources[1]], metadata={EVIDENCE_KEY: rows[1]})
    fields = {"existing_content": stored, "existing_overview": stored, "provenance_ids": sources}
    if category == "profile":
        fields.update(existing_abstract=claims[0], existing_metadata={EVIDENCE_KEY: rows[0]})
    else:
        fields.update(content_append=incoming, **{EVIDENCE_KEY: rows[0]})
    llm = Mock()
    llm.complete_json.return_value = {
        "abstract": "Profile", "overview": "- Person", "content": "- An engineer living in Paris.",
        "replaced_claims": [],
    }
    node = ArchiveBuilder(llm).build(candidate, WritePlan("merge", "ctx://account/users/Alex/memories/work", fields), ctx)

    review = assess_facts(node, read_source={archive.archive_id: archive}.get)
    assert [fact["abstract"] for fact in review["facts"].values()] == list(claims)
    assert review["confidence"] == .8
    assert llm.complete_json.call_count == 1


@pytest.mark.parametrize("decision", [{"replaced_claims": [0]}, {}])
def test_profile_write_respects_explicit_undated_replacement(decision):
    old, new = "Lives in Paris.", "Lives in Berlin."
    messages = [{"id": "old", "role": "user", "content": old},
                {"id": "new", "role": "user", "content": "I no longer live in Paris. I live in Berlin."}]
    archive = ArchiveEntry("archive", "session", "", "", messages)
    previous, current = (ProvenanceResolver.build_id("archive", archive.archive_id, [message["id"]])
                         for message in messages)
    rows = [bind_archive_evidence([{"claim": claim, "subject": "Alex", "quote": message["content"], "relation": "supports"}],
                                  reference, messages)
            for claim, message, reference in zip((old, new), messages, (previous, current))]
    candidate = CandidateMemory("profile", "user", "profile", new, f"- {new}", f"- {new}", .9, who="Alex",
                                provenance_ids=[current], metadata={EVIDENCE_KEY: rows[1]})
    fields = {"existing_abstract": old, "existing_overview": f"- {old}", "existing_content": f"- {old}",
              "existing_metadata": {EVIDENCE_KEY: rows[0]}, "provenance_ids": [previous, current]}
    llm = Mock()
    llm.complete_json.return_value = {"abstract": new, "overview": f"- {new}", "content": f"- {new}", **decision}
    ctx = RequestContext("account", "Alex", "agent", "session", "trace")
    node = ArchiveBuilder(llm).build(candidate, WritePlan("merge", "ctx://account/users/Alex/memories/profile", fields), ctx)

    assert node.content == f"- {new}"
    review = assess_facts(node, read_source={archive.archive_id: archive}.get)
    assert review["confidence"] == .8


def test_identity_merge_protects_old_and_incoming_evidence():
    claims, archive, sources, rows = _write_evidence_fixture()
    llm = Mock()
    llm.complete_json.return_value = {
        "content": "# Alex\n- Paris is home.\n- Works as an engineer.",
        "replaced_claims": [],
    }
    fs = MemoryFS()
    api = MemoryWriteAPI(fs, llm, directory_summary_enabled=False)
    ctx = RequestContext("account", "Alex", "agent", "session", "trace")
    def write(index):
        claim = claims[index]
        return api.write_memory(CandidateMemory(
            "preference", "user", "work", claim, f"- {claim}", f"# Alex\n- {claim}", .9, who="Alex",
            provenance_ids=[sources[index]], metadata={EVIDENCE_KEY: rows[index]},
        ), ctx)

    uri = write(0)["target_uri"]
    api._writer._archive_builder._card_format_enabled = True
    assert write(1)["target_uri"] == uri
    review = assess_facts(fs.read_node(uri, ctx), read_source={archive.archive_id: archive}.get)
    assert [(fact["abstract"], fact["confidence"]) for fact in review["facts"].values()] == [
        (claim, .8) for claim in claims
    ]
    assert review["confidence"] == .8
