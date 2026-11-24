"""Answer adapters preserve sources, complete dependencies and frozen budgets."""

import json
from copy import deepcopy
from dataclasses import asdict

import pytest

from core.evidence_query import MessageEvidenceResult
from retrieval.evidence.context import fit_evidence_payload, render_evidence_context


def unit(key, text, *, role="user", offset=0):
    return {"evidence_unit_id": key, "text": text, "role": role, "speaker": "",
            "recorded_at": "2026-07-15T23:30:00-04:00", "source_uri": "ctx://source/" + key,
            "source_key": key, "content_hash": key, "session_id": "session", "message_id": key,
            "parent_span": [offset, offset + len(text)]}


def fact(item, quote, *, subject="user"):
    start = item["parent_span"][0] + item["text"].index(quote)
    ref = {name: item[name] for name in ("evidence_unit_id", "source_key", "content_hash", "session_id", "message_id")}
    ref.update(span=[start, start + len(quote)], quote=quote)
    return {"fact_id": "fact-" + item["evidence_unit_id"], "quote": quote, "subject": subject,
            "predicate": "completed", "role": item["role"], "speaker": item["speaker"],
            "recorded_at": item["recorded_at"], "assertion_type": "observation", "source_refs": [ref]}


def result(items, facts=(), groups=None):
    return MessageEvidenceResult("request", "What courses did I complete?", items, list(facts),
                              {} if groups is None else {"evidence_groups": groups})


def test_query_response_budget_keeps_groups_and_facts_aligned():
    first = unit("a", "I completed pottery.")
    dependent = unit("b", "It took two months. " * 50)
    independent = unit("c", "I completed photography.")
    response = result([first, dependent, independent], [fact(first, first["text"])], [["a", "b"], ["c"]])
    response.diagnostics.update(partial=False, truncated=False)
    fitted = fit_evidence_payload(response.to_dict(), remaining_chars=700)
    assert [item["evidence_unit_id"] for item in fitted["evidence"]] == ["c"]
    assert fitted["groups"] == [["c"]] and fitted["facts"] == []
    assert fitted["truncated"]


@pytest.mark.parametrize("has_group", [False, True])
@pytest.mark.parametrize("partial,truncated", [(True, False), (False, True)])
def test_incomplete_query_response_keeps_empty_status_when_it_fits(has_group, partial, truncated):
    item = unit("a", "I completed pottery.")
    response = result([item], [fact(item, item["text"])], [["a"]]) if has_group else result([], groups=[])
    response.diagnostics.update(partial=partial, truncated=truncated)
    expected = {"evidence": [], "facts": [], "groups": [], "partial": partial, "truncated": truncated or has_group}
    budget = len(json.dumps({"message_evidence": expected}, ensure_ascii=False))
    assert fit_evidence_payload(response.to_dict(), remaining_chars=budget) == expected
    assert fit_evidence_payload(response.to_dict(), remaining_chars=budget - 1) is None


def test_independent_quote_uses_exact_unicode_offsets_and_neighbouring_context():
    quote = "我完成了摄影课程。"
    first = unit("a", "背景介绍。" * 20 + "在暑假。" + quote + "我很喜欢。" + "其他话题。" * 20, offset=31)
    response = result([first], [fact(first, quote)], [["a"]])
    response.query = "我完成了什么摄影课程？"
    original = deepcopy(response.to_dict())
    context = render_evidence_context(response)
    assert quote in context.text and "在暑假。" in context.text and "我很喜欢。" in context.text
    assert context.diagnostics["shortened_unit_ids"] == ["a"]
    assert "2026-07-15T23:30:00-04:00" in context.text
    assert '"role": "user"' in context.text and '"source": "ctx://source/a"' in context.text
    for start, end in context.blocks[0]["spans"]:
        assert first["text"][start - 31:end - 31] in context.text
    assert response.to_dict() == original


def test_short_quotes_leave_room_for_other_independent_facts():
    first = unit("a", "Museum notes. " * 100 + "I completed a pottery course. It was useful. " + "Weather notes. " * 100)
    second = unit("b", "I completed a photography course.")
    response = result([first, second], [fact(first, "I completed a pottery course.")], [["a"], ["b"]])
    context = render_evidence_context(response, max_chars=1200)
    assert context.diagnostics["selected_unit_ids"] == ["a", "b"]
    assert len(context.text) <= 1200
    assert "pottery" in context.text and "photography" in context.text


def test_required_question_and_short_answer_are_never_split_or_shortened():
    first = unit("q", "Which course did you complete?", role="assistant")
    second = unit("a", "Pottery.")
    response = result([first, second], groups=[["q", "a"]])
    full = render_evidence_context(response)
    assert full.diagnostics["selected_unit_ids"] == ["q", "a"]
    limited = render_evidence_context(response, max_chars=len(full.text) - 1)
    assert limited.text == ""
    assert limited.diagnostics["omitted_unit_ids"] == ["q", "a"]
    assert limited.diagnostics["skipped_groups"] == [{"unit_ids": ["q", "a"], "reason": "character_budget"}]


def test_later_dependency_requires_atomic_upgrade_of_earlier_excerpt():
    first = unit("a", "Museum notes. " * 100 + "I completed a pottery course. " + "Weather notes. " * 100)
    second = unit("b", "It was the hardest course.")
    response = result([first, second], [fact(first, "I completed a pottery course.")], [["a"], ["a", "b"]])
    context = render_evidence_context(response, max_chars=1000)
    assert context.diagnostics["selected_unit_ids"] == ["a"]
    assert context.diagnostics["shortened_unit_ids"] == ["a"]
    assert context.diagnostics["skipped_groups"][0]["unit_ids"] == ["a", "b"]
    full = render_evidence_context(response)
    assert full.diagnostics["selected_unit_ids"] == ["a", "b"]
    assert full.diagnostics["shortened_unit_ids"] == []
    assert first["text"] in full.text


@pytest.mark.parametrize("change", ["quote", "span", "role", "hash", "missing", "topic"])
def test_invalid_or_unrelated_facts_fall_back_to_raw_evidence(change):
    first = unit("a", "Background. " * 30 + "I completed a pottery course. " + "Weather. " * 30)
    record = fact(first, "I completed a pottery course.")
    if change == "quote":
        record["quote"] = "I completed ten courses."
    elif change == "span":
        record["source_refs"][0]["span"][0] += 1
    elif change == "role":
        record["role"] = "assistant"
    elif change == "hash":
        record["source_refs"][0]["content_hash"] = "outdated"
    elif change == "missing":
        record["source_refs"].append({**record["source_refs"][0], "evidence_unit_id": "absent"})
    else:
        record = fact(first, "Weather.")
    context = render_evidence_context(result([first], [record], [["a"]]))
    assert first["text"] in context.text
    assert context.diagnostics["shortened_unit_ids"] == []


def test_no_fact_semantic_evidence_and_assistant_sources_remain_usable():
    first = unit("a", "Restore the optical instrument with a soft cloth.", role="assistant")
    response = result([first], groups=[["a"]])
    response.query = "What did you recommend for telescope repairs?"
    context = render_evidence_context(response)
    assert first["text"] in context.text and '"role": "assistant"' in context.text
    assert context.text.startswith(response.untrusted_data_notice)


def test_internal_result_and_legacy_dictionary_keep_groups_atomic():
    items = [unit("q", "What course?", role="assistant"), unit("a", "Pottery.")]
    response = result(items)
    context = render_evidence_context(response)
    assert context == render_evidence_context(asdict(response))
    assert context.diagnostics["legacy_grouping"]
    assert render_evidence_context(response, max_chars=len(context.text) - 1).blocks == []


def test_missing_dependency_does_not_become_a_singleton():
    context = render_evidence_context(result([unit("a", "Pottery.")], groups=[["missing", "a"]]))
    assert context.blocks == []
    assert context.diagnostics["skipped_groups"][0]["reason"] == "missing_dependency"


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_budget_is_rejected(limit):
    with pytest.raises(ValueError):
        render_evidence_context(result([]), max_chars=limit)


def test_tiny_budget_and_empty_results_do_not_overrun_or_invent_evidence():
    assert render_evidence_context(result([unit("a", "Pottery.")], groups=[["a"]]), max_chars=1).text == ""
    context = render_evidence_context(result([]))
    assert context.text == "" and context.diagnostics["context_chars"] == 0


def test_other_topical_source_sentences_survive_non_exhaustive_facts():
    first = unit("a", "Museum notes. " * 30 + "I completed a pottery course. "
                 + "Weather notes. " * 30 + "I completed a photography course. " + "Traffic notes. " * 30)
    response = result([first], [fact(first, "I completed a pottery course.")], [["a"]])
    context = render_evidence_context(response)
    assert "I completed a photography course." in context.text
    assert context.diagnostics["shortened_unit_ids"] == ["a"]
