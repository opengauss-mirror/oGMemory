"""Compact source output retains facts through final card materialization."""

from datetime import datetime
from dataclasses import replace
from types import SimpleNamespace

import pytest

from extraction.card_format import validate_memory_card
from extraction.summary_generator import SUMMARY_SCHEMA, SessionSummaryGenerator
from extraction.tools import Extractor, _Span
from tests.unit.extraction.test_node_processing import operation, setup


@pytest.mark.parametrize("category", ["entity", "preference"])
def test_eager_source_draft_preserves_identity_evidence_and_facts(category):
    ctx, fs, llm, api = setup(lambda _: [operation([0])])
    facts = "Caroline's necklace is a gift from her grandmother.\nIt has a heart charm."
    if category == "preference":
        facts = "The user prefers wearing the necklace on family occasions."
    inputs = {"routing_key": "necklace", "object_name": "necklace", "scope_ref": "",
              "abstract": "Caroline's necklace" if category == "entity" else "User's necklace preference",
              "content": facts, "confidence": .9,
              "evidence_quote": "I prefer wearing the necklace on family occasions.",
              "who": "user" if category == "preference" else "Caroline; Grandma"}
    source_llm = SimpleNamespace(complete_with_tools=lambda **kwargs: [{"tool": "extract_" + category, "input": inputs}])
    extractor = Extractor(source_llm, node_processing_enabled=True)
    schema = next(tool["input_schema"] for tool in extractor._extraction_tools if tool["name"] == "extract_" + category)
    expected = ["routing_key", "abstract", "content", "confidence", "scope_ref"]
    if category == "entity":
        expected.append("object_name")
    assert schema["required"] == expected
    span = _Span(0, 0, categories=[category], message_ids=["source-message"])
    candidate, = extractor._structure_span_eager(span, facts, ctx, "source-digest", "source-archive")
    assert (candidate.abstract, candidate.overview, candidate.content) == (inputs["abstract"], inputs["abstract"], facts)
    assert candidate.scope_ref == inputs["scope_ref"]
    assert candidate.who == inputs["who"]
    assert candidate.provenance_ids == ["prov:1:archive:source-archive:source-message"]
    if category == "preference":
        assert candidate.evidence_quote == inputs["evidence_quote"]
    op = operation([0])
    op["sections"][0]["facts"] = facts.splitlines()
    llm.decide = lambda _: [op]
    api._pipeline.extract = lambda *args, **kwargs: [candidate]
    result = api.commit_session([{"role": "user", "content": facts}], ctx)
    stored = fs.read_node(result["plans"][0]["target_uri"], ctx)
    assert all(fact in stored.content for fact in facts.splitlines())
    assert stored.metadata["version"] == 1
    validate_memory_card(replace(candidate, abstract=stored.abstract, overview=stored.overview, content=stored.content))


def test_summary_keeps_unique_details_and_reads_legacy_key_details():
    captured = []
    llm = SimpleNamespace(complete_json=lambda prompt, schema: captured.append((prompt, schema)) or {
        "facts_and_details": "Caroline bought The Hobbit by J. R. R. Tolkien for $12.",
        "events_with_dates": "[2026-10-26] Caroline visited Paris.",
    })
    summary = SessionSummaryGenerator(llm).generate("Caroline bought The Hobbit for $12 yesterday.", datetime(2026, 10, 27))
    assert list(SUMMARY_SCHEMA["properties"]) == ["events_with_dates", "facts_and_details", "preferences",
                                                 "plans_and_intentions", "emotional_reactions"]
    assert "Assign each fact to its most specific section exactly once" in captured[0][0]
    text = SessionSummaryGenerator.format_summary(summary)
    assert "The Hobbit by J. R. R. Tolkien for $12" in text
    assert "2026-10-26" in text
    legacy = SessionSummaryGenerator.format_summary({"facts_and_details": "Caroline owns a book.",
                                                     "key_details": "The Hobbit by J. R. R. Tolkien."})
    assert "Caroline owns a book." in legacy
    assert "The Hobbit by J. R. R. Tolkien." in legacy


@pytest.mark.parametrize("source,event", [
    pytest.param(
        "Caroline: I'm putting together an LGBTQ art show next month and I'm gonna show my paintings. Super stoked!",
        "[2023-09] Caroline plans an LGBTQ art show featuring her paintings.",
        id="locomo_next_month",
    ),
    pytest.param(
        "Caroline: Thanks, Melanie! That plate is awesome! Did you make it?\n"
        "Melanie: Yeah, I made it in pottery class yesterday. I love it! Pottery's so relaxing and creative.",
        "[2023-08-24] Melanie made a plate in pottery class.",
        id="yesterday",
    ),
    pytest.param(
        "Caroline: I joined the art group last year.",
        "[2022] Caroline joined the art group.",
        id="last_year",
    ),
    pytest.param(
        "Melanie: I started pottery in March 2023 and painted a sunflower last month.",
        "[2023-03] Melanie started pottery.\n[2023-07] Melanie painted a sunflower.",
        id="explicit_and_relative_months",
    ),
    pytest.param(
        "Caroline: I plan to exhibit my paintings from September 4 through September 6 this year.",
        "[2023-09-04 to 2023-09-06] Caroline plans to exhibit her paintings.",
        id="planned_date_range",
    ),
    pytest.param(
        "Melanie: During the previous calendar week, Monday through Sunday, I practiced pottery. "
        "Last weekend, Saturday and Sunday, I attended an art workshop.",
        "[2023-08-14 to 2023-08-20] Melanie practiced pottery.\n"
        "[2023-08-19 to 2023-08-20] Melanie attended an art workshop.",
        id="calendar_week_and_weekend_ranges",
    ),
    pytest.param(
        "Caroline: I went hiking last week and got into a bad spot with some people. "
        "It really bugged me, so I tried to apologize to them.",
        "[last week; session anchor 2023-08-25] Caroline went hiking, got into a bad spot "
        "with some people, and tried to apologize to them.",
        id="locomo_attempted_action_with_week_anchor",
    ),
    pytest.param(
        "Caroline: I might visit an exhibition next month.",
        "[2023-09] Caroline might visit an exhibition.",
        id="uncertain_month_plan",
    ),
])
def test_summary_time_precision_preserves_source_facts_through_public_write(source, event):
    old_fact = "Caroline's grandmother gave her a necklace with a heart charm."
    conversation = old_fact + "\n" + source
    calls = []

    def summarize(prompt, schema):
        calls.append((prompt, schema))
        assert conversation in prompt
        assert "relative months to calendar months (YYYY-MM)" in prompt
        assert "Use a specific day only when the source supports that day." in prompt
        assert "Weeks and weekends denote time ranges." in prompt
        assert '"tried to", "might" and "plans to" in every section' in prompt
        assert "source-supported year, month, day or range precision" in schema["properties"]["events_with_dates"]["description"]
        return {"events_with_dates": event, "facts_and_details": old_fact,
                "preferences": "", "plans_and_intentions": "", "emotional_reactions": ""}

    ctx, fs, llm, api = setup(lambda _: [])
    llm.complete_json = summarize
    generator = SessionSummaryGenerator(llm)
    session_time = datetime(2023, 8, 25)
    summary = generator.generate(conversation, session_time, "Caroline, Melanie")
    formatted = generator.format_summary(summary)
    candidate = generator.build_candidate(summary, formatted, session_time,
                                          "Caroline, Melanie", ctx.session_id)
    result = api.write_memory(candidate, ctx)
    stored = fs.read_node(result["target_uri"], ctx)
    assert len(calls) == 1
    assert result["action"] == "create"
    assert stored.category == "session_summary"
    assert stored.content == formatted
    assert "### Events\n" + event in stored.content
    assert "### Facts\n" + old_fact in stored.content
    assert stored.metadata["version"] == 1
