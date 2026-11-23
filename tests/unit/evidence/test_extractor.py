"""Deterministic candidate contracts without NLP downloads or generative models."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from extraction.deterministic.extractor import DeterministicExtractor
from extraction.deterministic.nlp import DeterministicConfig, TextParser
from extraction.schemas.registry import SchemaRegistry
from extraction.tool_builder import build_extraction_tools
from tests.unit.evidence.test_sources import CTX, message


class RulesParser:
    degraded_languages = {"en", "zh"}

    def parse(self, text):
        return None


def extractor():
    return DeterministicExtractor(SchemaRegistry(), parser=RulesParser())


def test_quantity_without_entity_and_replay_identity():
    engine = extractor()
    messages = [message("I completed 7 essays this year.")]
    candidates = engine.extract(messages, CTX)
    fact = next(c for c in candidates if c.category == "fact")
    record = fact.metadata["deterministic"]["fact_records"][0]
    assert record["quantity"]["value"] == 7
    assert record["scope"] == ["this year"]
    assert record["subject"] == "user"
    assert engine.extract(messages, CTX) == candidates
    changed = engine.extract([message("I completed 4 essays this year.", id="old")], CTX)
    assert changed[0].routing_key != fact.routing_key
    assert not any(c.category == "entity" for c in candidates)


@pytest.mark.parametrize("text,quotes", [
    ("I visited Paris\nI completed 7 essays.", ["I visited Paris", "I completed 7 essays."]),
    ("我去了巴黎\n我完成了7篇文章。", ["我去了巴黎", "我完成了7篇文章。"]),
    ("I visited Paris\r\n\r\nI completed 7 essays", ["I visited Paris", "I completed 7 essays"]),
])
def test_unpunctuated_lines_preserve_facts_events_and_original_spans(text, quotes):
    candidates = extractor().extract([message(text)], CTX)
    for category in ("fact", "event"):
        records = [c.metadata["deterministic"]["fact_records"][0]
                   for c in candidates if c.category == category]
        assert [record["quote"] for record in records] == quotes
        for record in records:
            start, end = record["source_refs"][0]["span"]
            assert text[start:end] == record["quote"]


def test_decimal_quantities_are_not_split_into_separate_sentences():
    candidates = extractor().extract([message("I ran 3.5 miles. I completed 2.75 essays.")], CTX)
    records = [c.metadata["deterministic"]["fact_records"][0]
               for c in candidates if c.category == "fact"]
    assert [record["quote"] for record in records] == ["I ran 3.5 miles.", "I completed 2.75 essays."]
    assert [record["quantity"]["value"] for record in records] == [3.5, 2.75]
    assert [record["quantity"]["object"] for record in records] == ["miles", "essays"]


@pytest.mark.parametrize("reported", [
    'Alex told me: "I adopted a dog. I moved to Paris."',
    "Alex told me: 'I adopted a dog. I moved to Paris.'",
    'Alex told me: “I adopted a dog. I moved to Paris.”',
    'Alex told me: "I adopted a dog.\nI moved to Paris."',
    'Alex told me: "I adopted a dog. I moved to Paris.',
    'Alex说：“我收养了小狗。我搬到巴黎。”',
    'Alex说：「我收养了小狗。我搬到巴黎。」',
])
def test_quoted_reports_do_not_become_the_speakers_observations(reported):
    candidates = extractor().extract([message("I visited Rome.\n" + reported)], CTX)
    records = [record for candidate in candidates
               for record in candidate.metadata["deterministic"]["fact_records"]]
    assert records
    assert all(record["quote"] == "I visited Rome." for record in records)


def test_apostrophes_in_direct_statements_do_not_create_quoted_speech():
    candidates = extractor().extract([message("I adopted my neighbor's dog.\nI moved to Paris.")], CTX)
    assert len([candidate for candidate in candidates if candidate.category == "fact"]) == 2


@pytest.mark.parametrize("text", ["Hello!", "Thanks.", "Did I complete 7 essays?",
                                  "I might complete 7 essays.", "If I visited the park.",
                                  "我可能参加了比赛。", "我完成了作业吗？"])
def test_no_uncertain_facts_or_greetings(text):
    assert extractor().extract([message(text)], CTX) == []


def test_assistant_recommendation_is_not_user_experience():
    engine = extractor()
    candidates = engine.extract([message("I recommend visiting Paris.", role="assistant")], CTX)
    assert [c.category for c in candidates] == ["fact"]
    record = candidates[0].metadata["deterministic"]["fact_records"][0]
    assert record["role"] == "assistant" and record["subject"] == "assistant"
    assert record["assertion_type"] == "recommendation"
    assert engine.extract([message("I visited Paris.", role="assistant")], CTX) == []


def test_wrapper_subject_is_independent_of_owner_and_message_date():
    candidates = extractor().extract([message("[Caroline]: I visited the park yesterday.\n[Melanie]: I completed 7 essays.",
                                              recorded_at="2026-05-08T10:00:00+08:00")], CTX)
    facts = [c.metadata["deterministic"]["fact_records"][0] for c in candidates if c.category == "fact"]
    assert [record["subject"] for record in facts] == ["Caroline", "Melanie"]
    assert facts[0]["event_time"] == "2026-05-07"
    assert facts[0]["recorded_at"] == "2026-05-08T10:00:00+08:00"
    assert all(c.owner_scope == "user" for c in candidates)


def test_chinese_and_unknown_timezone():
    candidates = extractor().extract([message("我完成了7篇文章。", recorded_at="2026-05-08T10:00:00")], CTX)
    assert any(c.category == "event" for c in candidates)
    record = candidates[0].metadata["deterministic"]["fact_records"][0]
    assert record["subject"] == "user" and record["quantity"]["value"] == 7
    candidates = extractor().extract([message("I visited the park yesterday.", recorded_at="2026-05-08T10:00:00")], CTX)
    assert candidates[0].metadata["deterministic"]["fact_records"][0]["event_time"] is None


def test_type_switches_are_independent_and_llm_tools_do_not_expose_facts():
    engine = extractor()
    engine.registry.register(replace(engine.registry.get("fact"), enabled=False))
    candidates = engine.extract([message("Caroline visited Paris."),
                                 message("Caroline completed 7 essays.", id="two", sequence=4)], CTX)
    assert {c.category for c in candidates} == {"entity", "event"}
    names = {tool["name"] for tool in build_extraction_tools(SchemaRegistry())}
    assert "extract_fact" not in names


def test_missing_models_are_explicit_and_parser_errors_propagate(monkeypatch):
    monkeypatch.setattr("extraction.deterministic.nlp.load_model", Mock(side_effect=OSError("missing")))
    with pytest.raises(RuntimeError, match="installed spaCy"):
        TextParser(DeterministicConfig())
    parser = TextParser(DeterministicConfig(on_model_missing="rules_only"))
    assert parser.degraded_languages == {"en", "zh"}
    assert parser.parse("I completed 7 essays.") is None
    failing = SimpleNamespace(parse=Mock(side_effect=RuntimeError("parser failed")), degraded_languages=set())
    with pytest.raises(RuntimeError, match="parser failed"):
        DeterministicExtractor(SchemaRegistry(), parser=failing).extract([message()], CTX)
