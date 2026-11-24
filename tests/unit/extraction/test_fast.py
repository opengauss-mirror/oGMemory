from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.dreaming_evidence import EVIDENCE_KEY, verified_assertions
from core.models import ContextNode, RequestContext
from core.uri_resolver import URIResolver
from extraction.fast import FastExtractor
from extraction.prefetch import PrefetchResult
from extraction.schemas.registry import SchemaRegistry
from service.api import MemoryWriteAPI
from tests.unit.commit.test_memory_identity import MemoryFS
from tests.unit.extraction.test_tool_result_stub_extraction import _stub


def test_fast_extracts_card_and_summary_and_applies_read_bound_edit_in_one_call():
    ctx = RequestContext("acme", "alice", "agent", "session", "trace")
    nate_uri = "ctx://acme/users/alice/memories/entities/nate"
    nate = ContextNode(
        nate_uri, "MEMORY", "entity", 0, ctx.user_space_name(), "Nate lives in Denver.",
        "- Lives in Denver.", "# Nate\n\n## Home\n- Lives in Denver.",
        {"version": 3, "routing_key": "nate", "object_name": "Nate", "scope_ref": ""},
    )
    source = "Nate moved to Boston in May. Alice adopted a cat named Fern."
    llm = SimpleNamespace(complete_with_tools=Mock(return_value=[
        {"tool": "edit_memory", "input": {
            "target_uri": nate_uri, "expected_version": 3, "search": "- Lives in Denver.",
            "replacement": "- Moved from Denver to Boston in May.",
            "abstract": "Nate lives in Boston.", "overview": "- Lives in Boston.",
            "evidence_quote": "Nate moved to Boston in May.", "confidence": 0.95,
        }},
        {"tool": "extract_entity", "input": {
            "object_name": "Fern", "scope_ref": "", "routing_key": "fern",
            "abstract": "Fern is Alice's cat.", "overview": "- Alice adopted Fern.",
            "content": "# Fern\n\n## Identity\n- Fern is Alice's cat.", "confidence": 0.95,
        }},
        {"tool": "summarize_session", "input": {
            "events_with_dates": "[May] Nate moved to Boston.",
            "facts_and_details": "Alice adopted a cat named Fern.",
            "preferences": "", "plans_and_intentions": "",
        }},
    ]))
    extractor = FastExtractor(llm, card_format_enabled=True)
    extractor._prefetcher = SimpleNamespace(prefetch_for_span=Mock(return_value=PrefetchResult(
        messages=[f"URI: {nate_uri}\nVersion: 3\n{nate.content}"],
        read_nodes={nate_uri: nate}, read_versions={nate_uri: 3},
    )))

    items = extractor.extract(
        [{"role": "user", "content": source, "id": "m1"}],
        ctx, datetime(2026, 5, 10), archive_id="archive1",
    )

    llm.complete_with_tools.assert_called_once()
    assert nate_uri in llm.complete_with_tools.call_args.kwargs["prompt"]
    assert [item.category for item in items] == ["entity", "entity", "session_summary"]
    assert items[0].target_uri == nate_uri
    assert items[0].expected_version == 3
    assert "Moved from Denver to Boston" in items[0].content
    assert items[1].routing_key == "fern"
    assert "Alice adopted a cat named Fern." in items[2].content
    assert all(item.provenance_ids for item in items)


def _summary(text):
    return {"tool": "summarize_session", "input": {
        "events_with_dates": "", "facts_and_details": text,
        "preferences": "", "plans_and_intentions": "",
    }}


@pytest.mark.parametrize("occupation_confidence, threshold, expected_job, occupation_first, failed_check", [
    (.96, .5, "baker", False, None),
    (.60, .8, "teacher", False, None),
    (.60, .8, "teacher", True, None),
    (.96, .5, "teacher", True, "quote"),
    (.96, .5, "teacher", True, "search"),
])
def test_fast_composes_profile_edits_with_source_evidence_without_vectors(
    monkeypatch, occupation_confidence, threshold, expected_job, occupation_first, failed_check,
):
    monkeypatch.setenv("OGMEM_DREAMING_ENABLED", "true")
    fs = MemoryFS()
    ctx = RequestContext("acme", "nora", "agent", "session", "trace")
    uri = URIResolver(SchemaRegistry()).resolve("profile", {}, ctx)
    original = "My name is Nora. I live in Utrecht. I work as a teacher."
    address = "I moved to Delft from Utrecht."
    occupation = "I now work as a baker."
    correction = f"{address} {occupation}"
    attribution = {"confidence": .96, "attributed_speaker": "user", "attribution_basis": "self_first_person"}
    def edit(search, replacement, quote, *, confidence, occupation):
        return {"tool": "edit_memory", "input": dict(
            attribution, target_uri=uri, expected_version=1, search=search, replacement=replacement,
            abstract=f"Nora lives in Delft and works as a {occupation}.",
            overview=f"Nora lives in Delft and works as a {occupation}.",
            evidence_quote=quote, confidence=confidence,
            evidence_assertions=[{"claim": replacement, "subject": "nora", "quote": quote,
                                  "relation": "supports"}],
        )}

    address_edit = edit("I live in Utrecht.", "I live in Delft.", address, confidence=.96,
                        occupation="baker" if occupation_first else "teacher")
    occupation_edit = edit("I work as a teacher.", "I work as a baker.", occupation,
                           confidence=occupation_confidence, occupation="baker")
    if failed_check == "quote":
        occupation_edit["input"]["evidence_quote"] = "I started working at a bakery."
    elif failed_check == "search":
        occupation_edit["input"]["search"] = "I work as a schoolteacher."
    edits = [occupation_edit, address_edit] if occupation_first else [address_edit, occupation_edit]
    llm = SimpleNamespace(complete_with_tools=Mock(side_effect=[
        [{"tool": "extract_profile", "input": dict(
            attribution, routing_key="identity", abstract="Nora lives in Utrecht.",
            overview=original, content=original, evidence_quote=original,
        )}, _summary(original)],
        [*edits, _summary(correction)],
    ]))
    api = MemoryWriteAPI(fs, llm, extraction_mode="fast", directory_summary_enabled=False)

    messages = [{"role": "user", "content": source, "id": str(idx)}
                for idx, source in enumerate((original, correction), 1)]
    for message in messages:
        result = api.commit_session([message], ctx, archive_id=f"archive-{message['id']}",
                                    confidence_threshold=threshold)
        assert result["writes_completed"] == 2

    if failed_check:
        assert result["degraded"]
        assert [failure["tool"] for failure in result["failed_nodes"]] == ["edit_memory"]
    node = fs.read_node(uri, ctx)
    assert node.metadata["version"] == 2
    assert node.content == f"My name is Nora. I live in Delft. I work as a {expected_job}."
    expected_view = node.content if occupation_first else f"Nora lives in Delft and works as a {expected_job}."
    assert node.abstract == node.overview == expected_view
    quotes = [address, occupation] if expected_job == "baker" else [address]
    assert [row["message_id"] for row in node.metadata[EVIDENCE_KEY]] == ["2"] * len(quotes)
    archive = SimpleNamespace(archive_id="archive-2", session_id=ctx.session_id, messages=messages[1:])
    assert [assertion.quote for assertion, _ in verified_assertions(node, lambda _: archive)] == quotes
    assert llm.complete_with_tools.call_count == 2
    assert f"URI: {uri}\nVersion: 1" in llm.complete_with_tools.call_args.kwargs["prompt"]


@pytest.mark.parametrize("assistant_first", [False, True])
def test_fast_keeps_user_profile_edit_with_externalized_tool_results(assistant_first):
    fs = MemoryFS()
    ctx = RequestContext("acme", "nora", "agent", "session", "trace")
    uri = URIResolver(SchemaRegistry()).resolve("profile", {}, ctx)
    original = "My name is Nora. I live in Utrecht. I work as a teacher."
    accepted = "My name is Nora. I live in Rotterdam. I work as a teacher."
    address = "I live in Rotterdam."
    occupation = "I work as a baker."
    attribution = {"attributed_speaker": "user", "attribution_basis": "self_first_person", "confidence": .96}

    def edit(search, replacement, view):
        return {"tool": "edit_memory", "input": dict(
            attribution, target_uri=uri, expected_version=1, search=search, replacement=replacement,
            abstract=view, overview=view, evidence_quote=replacement,
        )}

    address_view = "Nora lives in Rotterdam and works as a teacher."
    address_edit = edit("I live in Utrecht.", address,
                        "Nora lives in Rotterdam and works as a baker." if assistant_first else address_view)
    occupation_edit = edit("I work as a teacher.", occupation, "Nora works as a baker.")
    edits = [occupation_edit, address_edit] if assistant_first else [address_edit, occupation_edit]
    model = SimpleNamespace(complete_with_tools=Mock(side_effect=[
        [{"tool": "extract_profile", "input": dict(
            attribution, routing_key="identity", abstract=original, overview=original, content=original,
            evidence_quote=original,
        )}, _summary(original)],
        [*edits, _summary(address)],
    ]))
    api = MemoryWriteAPI(fs, model, extraction_mode="fast", directory_summary_enabled=False)
    api.commit_session([{"role": "user", "content": original, "id": "seed"}], ctx, archive_id="seed")
    result = api.commit_session([
        {"role": "user", "content": address, "id": "move"},
        {"role": "toolResult", "content": _stub(), "id": "tool"},
        {"role": "assistant", "content": occupation, "id": "reply"},
    ], ctx, archive_id="change")

    node = fs.read_node(uri, ctx)
    assert result["writes_completed"] == 2
    assert result["degraded"]
    assert result["failed_nodes"] == [{
        "tool": "edit_memory", "error": "Tool-call protocol failed: sensitive memory requires direct user support",
    }]
    expected_view = accepted if assistant_first else address_view
    assert (node.metadata["version"], node.content, node.abstract, node.overview) == (
        2, accepted, expected_view, expected_view,
    )


@pytest.mark.parametrize("assistant_first", [False, True])
def test_fast_creates_user_profile_with_externalized_tool_results(assistant_first):
    fs = MemoryFS()
    ctx = RequestContext("acme", "nora", "agent", "session", "trace")
    address = "I live in Rotterdam."
    occupation = "I work as a baker."

    def profile(text):
        return {"tool": "extract_profile", "input": {
            "routing_key": "identity", "abstract": text, "overview": text, "content": text,
            "evidence_quote": text, "confidence": .96,
            "attributed_speaker": "user", "attribution_basis": "self_first_person",
        }}

    calls = [profile(occupation), profile(address)] if assistant_first else [profile(address), profile(occupation)]
    model = SimpleNamespace(complete_with_tools=Mock(return_value=[*calls, _summary(address)]))
    api = MemoryWriteAPI(fs, model, extraction_mode="fast", directory_summary_enabled=False)
    result = api.commit_session([
        {"role": "user", "content": address, "id": "move"},
        {"role": "toolResult", "content": _stub(), "id": "tool"},
        {"role": "assistant", "content": occupation, "id": "reply"},
    ], ctx, archive_id="profile")

    uri = URIResolver(SchemaRegistry()).resolve("profile", {}, ctx)
    assert result["writes_completed"] == 2
    assert fs.read_node(uri, ctx).content == address


@pytest.mark.parametrize("confidence_order", [(0.35, 0.95), (0.95, 0.35)])
def test_fast_commits_confident_creation_for_same_identity_in_either_order(confidence_order):
    fs = MemoryFS()
    ctx = RequestContext("acme", "alex", "agent", "session", "trace")
    source = "Mira plays the violin."

    def create(confidence):
        instrument = "violin" if confidence == .95 else "flute"
        fact = f"Plays the {instrument}."
        return {"tool": "extract_entity", "input": {
            "routing_key": "mira", "object_name": "Mira", "scope_ref": "",
            "abstract": f"Mira plays the {instrument}.", "overview": f"- {fact}",
            "content": f"# Mira\n## Music\n- {fact}", "confidence": confidence,
        }}

    calls = [create(confidence) for confidence in confidence_order]
    model = SimpleNamespace(complete_with_tools=Mock(return_value=[*calls, _summary(source)]))
    api = MemoryWriteAPI(fs, model, extraction_mode="fast", card_format_enabled=True,
                         directory_summary_enabled=False)
    result = api.commit_session([{"role": "user", "content": source, "id": "m1"}], ctx)

    uri = URIResolver(SchemaRegistry()).resolve("entity", {"routing_key": "mira"}, ctx)
    node = fs.read_node(uri, ctx)
    assert result["writes_completed"] == 2
    assert (node.abstract, node.overview, node.content) == (
        source, "- Plays the violin.", "# Mira\n## Music\n- Plays the violin.",
    )


@pytest.mark.parametrize(("category", "key"), [("skill", "cold_backup"), ("tool", "restic")])
def test_fast_can_store_agent_memory_types_offered_by_the_registry(category, key):
    fs = MemoryFS()
    ctx = RequestContext("acme", "nora", "agent", "session", "trace")
    text = "For a cold backup, stop workers, run restic backup, then verify with restic check."

    def model(*, tools, **kwargs):
        assert f"extract_{category}" in {tool["name"] for tool in tools}
        return [{"tool": f"extract_{category}", "input": {
            "routing_key": key, "confidence": .96, "abstract": "Cold backup workflow.",
            "overview": text, "content": text,
        }}, _summary(text)]

    llm = SimpleNamespace(complete_with_tools=Mock(side_effect=model))
    api = MemoryWriteAPI(fs, llm, extraction_mode="fast", directory_summary_enabled=False)

    result = api.commit_session([{"role": "user", "content": text, "id": "m1"}], ctx)

    assert result["writes_completed"] == 2
    uri = URIResolver(SchemaRegistry()).resolve(category, {"routing_key": key}, ctx)
    assert fs.read_node(uri, ctx).content == text
    llm.complete_with_tools.assert_called_once()
