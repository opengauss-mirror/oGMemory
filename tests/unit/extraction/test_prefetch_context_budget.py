"""Occurrence context stays bounded without authorizing unread update snapshots."""
from __future__ import annotations

import pytest

from core.errors import ToolCallProtocolError
from core.models import CandidateMemory, SeedHit
from extraction.tool_builder import validate_read_target
from extraction.tools import Extractor, _Span
from providers.llm.mock_llm import MockLLM
from tests.unit.extraction.test_prefetch import (
    FakeEmbedder,
    FakeFS,
    FakeVectorIndex,
    _ctx,
    _node,
    _prefetcher,
)


def _occurrences(fs, *, count=10, content="stored fact " * 7000):
    uris = []
    for category, owner, directory in (
        ("event", "user:user", "ctx://acct/users/user/memories/events"),
        ("case", "agent:agent", "ctx://acct/agents/agent/memories/cases"),
    ):
        fs.children[directory] = []
        for index in range(count):
            uri = f"{directory}/inspection_{index:02}"
            node = _node(uri, category)
            node.owner_space = owner
            node.content = content
            node.abstract = f"{category} inspection summary"
            node.metadata = {"version": 3, "scope_ref": "Project A", "who": "Technician"}
            fs.nodes[uri] = node
            fs.children[directory].append(uri)
            uris.append(uri)
    return uris


@pytest.mark.parametrize("categories", [["event", "case"], ["case", "event", "case"]])
def test_occurrence_prefetch_shares_budget_across_categories_and_keeps_hints(categories):
    fs = FakeFS()
    uris = _occurrences(fs)
    result = _prefetcher(fs, FakeVectorIndex(), FakeEmbedder()).prefetch_for_span(
        categories, _ctx(), conversation_text="new equipment inspection",
    )

    assert len("\n".join(result.messages)) <= 32_000
    assert result.listed_uris == set(uris)
    assert len(result.messages) == 20
    assert result.read_uris == set()
    assert result.read_versions == {}
    for uri in uris:
        hint = next(message for message in result.messages if f"URI: {uri}\n" in message)
        assert "inspection summary" in hint and "Project A" in hint
        assert "Full body not supplied" in hint
        assert "without target_uri or expected_version" in hint
        assert "Content:" not in hint


def test_oversized_related_occurrence_keeps_small_fallback_snapshot_complete():
    fs, vector = FakeFS(), FakeVectorIndex()
    directory = "ctx://acct/users/user/memories/events"
    related, recent = f"{directory}/a_related", f"{directory}/z_recent"
    fs.children[directory] = [recent, related]
    for uri in (related, recent):
        node = _node(uri, "event")
        node.metadata = {"version": 7, "scope_ref": "Project A"}
        fs.nodes[uri] = node
    fs.nodes[related].content = "large stored fact " * 5000
    fs.nodes[related].abstract = "related inspection"
    vector.hits = [SeedHit(uri=related, category="event", score=.9, abstract="related inspection", level=2)]
    result = _prefetcher(fs, vector, FakeEmbedder()).prefetch_for_span(
        ["event"], _ctx(), conversation_text="inspection",
    )

    assert result.listed_uris == {related, recent}
    assert f"URI: {related}\n" in result.messages[0]
    assert "Full body not supplied" in result.messages[0]
    assert f"Content: {fs.nodes[recent].content}\n" in result.messages[1]
    assert result.read_uris == {recent}
    assert result.read_versions == {recent: 7}
    unread_update = CandidateMemory("event", "user", "inspection", "Inspection", "Overview",
                                    "Updated fact", .9, target_uri=related, expected_version=7)
    with pytest.raises(ToolCallProtocolError, match="successful read"):
        validate_read_target(unread_update, result.read_versions)


@pytest.mark.parametrize("categories", [["event", "profile"], ["profile", "event"]])
def test_occurrence_budget_accounts_for_other_category_context(categories):
    fs = FakeFS()
    event = _occurrences(fs, count=1, content="event detail " * 1000)[0]
    profile = "ctx://acct/users/user/memories/profile"
    fs.nodes[profile] = _node(profile, "profile")
    fs.nodes[profile].content = "profile fact " * 2000
    result = _prefetcher(fs, FakeVectorIndex(), FakeEmbedder()).prefetch_for_span(
        categories, _ctx(), conversation_text="inspection",
    )

    assert len("\n".join(result.messages)) <= 32_000
    assert result.read_uris == {profile}
    assert event not in result.read_versions
    assert any("Full body not supplied" in message for message in result.messages)
    assert fs.nodes[profile].content in "\n".join(result.messages)


def test_eager_prefetch_reserves_actual_prompt_and_output_space():
    fs = FakeFS()
    _occurrences(fs, count=1, content="stored inspection " * 1000)
    extractor = Extractor(MockLLM())
    extractor._prefetcher = _prefetcher(fs, FakeVectorIndex(), FakeEmbedder())
    span = _Span(start=0, end=0, categories=["event", "case"],
                 focused_messages=[{"role": "user", "content": "inspection"}])
    prompt = "current source and instructions " * 1250
    result_prompt = extractor._with_prefetch_context(span, prompt, _ctx())

    assert result_prompt.endswith(prompt)
    assert len(result_prompt) + 16_000 <= 64_000
    assert span.prefetch_result.read_uris == set()
    assert span.prefetch_result.read_versions == {}
    assert "Full body not supplied" in result_prompt

    # A new call recalculates space from the complete prompt, not the short query.
    large_prompt = "current source " * 4000
    assert extractor._with_prefetch_context(span, large_prompt, _ctx()) == large_prompt
    assert span.prefetch_result.read_versions == {}


def test_lazy_occurrence_budget_preserves_uris_for_on_demand_reads():
    fs = FakeFS()
    uris = _occurrences(fs)
    result = _prefetcher(fs, FakeVectorIndex(), FakeEmbedder()).prefetch_for_span(
        ["event", "case"], _ctx(), conversation_text="inspection", lazy=True,
    )
    assert result.is_lazy
    assert {hint["uri"] for hint in result.uri_hints} == set(uris)
    assert result.messages == []
    assert result.read_uris == set()
    assert result.read_versions == {}


def test_occurrence_identity_and_summary_hints_remain_bounded_for_long_fields():
    fs = FakeFS()
    uris = _occurrences(fs)
    for node in fs.nodes.values():
        node.abstract = "equipment inspection " * 9
        node.overview = "stored inspection detail " * 180
        node.metadata["scope_ref"] = "North facility project " * 1000
    result = _prefetcher(fs, FakeVectorIndex(), FakeEmbedder()).prefetch_for_span(
        ["event", "case"], _ctx(), conversation_text="new inspection",
    )
    assert len("\n".join(result.messages)) <= 32_000
    assert len(result.messages) == len(uris)
    assert all("North facility project" in message for message in result.messages)
    assert all("fields may be abbreviated" in message for message in result.messages)
    assert result.read_uris == set() and result.read_versions == {}


def test_shared_budget_preserves_priority_of_a_fitting_related_body():
    fs, vector = FakeFS(), FakeVectorIndex()
    _occurrences(fs)
    related = "ctx://acct/users/user/memories/events/a_related_inspection"
    node = _node(related, "event")
    node.content = "semantic inspection detail " * 190
    node.metadata = {"version": 7, "scope_ref": "Project A"}
    fs.nodes[related] = node
    vector.hits = [SeedHit(uri=related, category="event", score=.9, abstract="related inspection", level=2)]
    result = _prefetcher(fs, vector, FakeEmbedder()).prefetch_for_span(
        ["event", "case"], _ctx(), conversation_text="inspection",
    )

    assert f"Content: {node.content}\n" in result.messages[0]
    assert result.read_versions == {related: 7}
    assert len(result.messages) == 20
    assert len("\n".join(result.messages)) <= 32_000


@pytest.mark.parametrize("categories", [["event", "case"], ["case", "event"]])
def test_related_bodies_take_priority_over_other_category_recent_fallbacks(categories):
    fs, vector = FakeFS(), FakeVectorIndex()
    event_directory = "ctx://acct/users/user/memories/events"
    recent = [f"{event_directory}/recent_{index:02}" for index in range(10)]
    fs.children[event_directory] = recent
    for uri in recent:
        node = _node(uri, "event")
        node.content = "recent inspection detail " * 120
        fs.nodes[uri] = node
    related = "ctx://acct/agents/agent/memories/cases/related_inspection"
    node = _node(related, "case")
    node.owner_space = "agent:agent"
    node.metadata = {"version": 7, "scope_ref": "Project A"}
    node.content = "related technical case " * 430
    fs.nodes[related] = node
    vector.hits = [SeedHit(uri=related, category="case", score=.9, abstract="related inspection", level=2)]
    result = _prefetcher(fs, vector, FakeEmbedder()).prefetch_for_span(
        categories, _ctx(), conversation_text="technical inspection",
    )

    related_message = next(message for message in result.messages if f"URI: {related}\n" in message)
    assert f"Content: {node.content}\n" in related_message
    assert result.read_versions[related] == 7
    assert result.listed_uris == {*recent, related}
    assert len("\n".join(result.messages)) <= 32_000
    shown_uris = [message.split("URI: ", 1)[1].split("\n", 1)[0] for message in result.messages]
    expected = sorted(recent, reverse=True) + [related] if categories[0] == "event" else [related] + sorted(recent, reverse=True)
    assert shown_uris == expected


def test_fitting_related_full_body_is_kept_when_all_hints_cannot_fit():
    fs, vector = FakeFS(), FakeVectorIndex()
    directory = "ctx://acct/users/user/memories/events"
    related = f"{directory}/a_related_inspection"
    node = _node(related, "event")
    node.metadata = {"version": 7, "scope_ref": "Project A"}
    fs.nodes[related] = node
    vector.hits = [SeedHit(uri=related, category="event", score=.9, abstract="inspection", level=2)]
    prefetcher = _prefetcher(fs, vector, FakeEmbedder())
    full = prefetcher.prefetch_for_span(["event"], _ctx(), conversation_text="inspection").messages[0]
    recent = [f"{directory}/z_recent_{index:02}" for index in range(9)]
    fs.children[directory] = recent
    for uri in recent:
        fallback = _node(uri, "event")
        fallback.content = "large stored detail " * 4000
        fs.nodes[uri] = fallback
    result = prefetcher.prefetch_for_span(
        ["event"], _ctx(), conversation_text="inspection",
        prompt_chars=64_000 - 16_000 - 128 - (len(full) + 1),
    )

    assert result.messages == [full]
    assert result.listed_uris == {*recent, related}
    assert result.read_uris == {related}
    assert result.read_versions == {related: 7}
