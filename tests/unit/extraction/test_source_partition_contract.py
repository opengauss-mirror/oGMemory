"""Partition source encoding and redundant singleton evidence preserve identity."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from core.errors import ExtractionIncompleteError
from extraction.source_occurrence_reconciliation import reconcile_source_occurrences
from tests.unit.extraction.test_event_draft_boundaries import event
from tests.unit.extraction.test_source_occurrence_reconciliation import (
    CTX,
    EARLY,
    SourceLLM,
    extract,
    messages,
    writer,
)


def redundant_evidence(response, payload):
    for group in response["groups"]:
        for member in group["members"]:
            source_index = payload["drafts"][member]["source_index"]
            message = payload["sources"][source_index]["selected_user_messages"][0]
            group["identity_evidence"].append(
                dict(
                    member=member,
                    source_index=source_index,
                    message_index=message["message_index"],
                    quote=message["content"],
                )
            )
    return response


def test_exact_singleton_evidence_preserves_originals_and_distinct_writes():
    llm = SourceLLM(identical=True, groups=[[0], [1]], mutate=redundant_evidence)
    api, fs = writer(llm)
    candidates = extract(api, messages())
    assert len(candidates) == 2 and candidates[0].content == candidates[1].content
    assert len({c.source_occurrence_id for c in candidates}) == 2
    assert [api.write_memory(c, CTX)["action"] for c in candidates] == ["create", "create"]
    assert len(fs.nodes) == 2
    assert [api.write_memory(c, CTX)["action"] for c in candidates] == ["skip", "skip"]
    assert [call[0] for call in llm.json_calls] == [{"spans"}, {"groups"}]


@pytest.mark.parametrize("bad", ["quote", "member", "source", "message", "extra"])
def test_invalid_singleton_evidence_is_still_rejected(bad):
    def corrupt(response, payload):
        redundant_evidence(response, payload)
        e = response["groups"][0]["identity_evidence"][0]
        if bad == "quote":
            e["quote"] = "Unsupported assertion"
        elif bad == "member":
            e["member"] = 1
        elif bad == "source":
            e["source_index"] = 1
        elif bad == "message":
            e["message_index"] = 8
        else:
            e["unexpected"] = True
        return response

    api, fs = writer(SourceLLM(groups=[[0], [1]], mutate=corrupt))
    with pytest.raises(ExtractionIncompleteError):
        extract(api, messages())
    assert not fs.nodes


def source(text):
    return dict(
        known_origin=True,
        start=0,
        end=0,
        message_ids=["m0"],
        session_time="2026-10-04T00:00:00Z",
        source_context="Session Time 2026-10-04\nUSER: " + text,
        selected_user_messages=[dict(id="m0", message_index=0, role="user", content=text)],
    )


def test_duplicate_complete_sources_fit_without_dropping_text_or_drafts():
    original = source(EARLY + (" Complete source detail." * 290))
    entries = [(SimpleNamespace(occurrence_source=deepcopy(original)), [event()]) for _ in range(5)]
    llm = SourceLLM(groups=[[i] for i in range(5)])
    before = deepcopy(entries)
    output = reconcile_source_occurrences(entries, CTX, "archive", llm)
    assert len(output) == 5 and len({c.source_occurrence_id for c in output}) == 5
    import json

    supplied = json.loads(llm.json_calls[0][1].split("\n\n", 1)[1])
    assert supplied["sources"] == [original]
    assert len(supplied["drafts"]) == 5 and all(d["source_index"] == 0 for d in supplied["drafts"])
    assert [e[0].occurrence_source for e in entries] == [e[0].occurrence_source for e in before]


def test_distinct_sources_are_not_interned_and_oversized_input_still_rejected():
    sources = [source("Long source " * 800) for _ in range(5)]
    for i, item in enumerate(sources):
        item["selected_user_messages"][0]["id"] = str(i)
    entries = [(SimpleNamespace(occurrence_source=s), [event()]) for s in sources]
    llm = SourceLLM(groups=[[i] for i in range(5)])
    with pytest.raises(ExtractionIncompleteError, match="budget"):
        reconcile_source_occurrences(entries, CTX, "archive", llm)
    assert not llm.json_calls
