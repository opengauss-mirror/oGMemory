"""Verified context may help retrieval without becoming a requested subject's fact."""

from copy import deepcopy
from dataclasses import replace

import pytest

from commit.deterministic import deterministic_batch
from service.evidence_service import EvidenceService
from tests.unit.evidence.test_extractor import extractor
from tests.unit.evidence.test_sources import CTX, message
from tests.unit.evidence.test_writer import write_candidates


def seed(setup, monkeypatch, texts, *, opaque=False):
    fs, api, store, _ = setup
    messages = [message(text, id=str(i), sequence=i * 10) for i, text in enumerate(texts)]
    store.register(messages, CTX, batch_id="hints")
    with deterministic_batch():
        for item in messages:
            candidate = next(c for c in extractor().extract([item], CTX) if c.category == "fact")
            if opaque:
                candidate = replace(candidate, content="An opaque card body.")
            api._writer.write_candidate(candidate, CTX)
    # Isolate the memory channel: these sources are absent from the lexical pool.
    monkeypatch.setattr("service.evidence_service.lexical_scores", lambda *args: {})
    return EvidenceService(fs)


def test_narrator_source_can_help_without_becoming_the_actors_structured_fact(setup, monkeypatch):
    service = seed(setup, monkeypatch, ["[Morgan]: Morgan reported that Alex repaired a telescope."])
    result = service.search("Did Alex repair a telescope?", CTX, use_memories=True)
    assert [e["message_id"] for e in result.evidence] == ["0"]
    assert result.evidence[0]["speaker"] == "Morgan"
    assert result.facts == []
    assert not service.search("Did Alex repair a telescope?", CTX).evidence


def test_record_metadata_does_not_require_matching_card_presentation(setup, monkeypatch):
    service = seed(setup, monkeypatch, ["[Alex]: I recommend a telescope."], opaque=True)
    result = service.search("What did Alex recommend?", CTX, use_memories=True)
    assert result.evidence and result.facts
    assert all(r["subject"] == "Alex" and r["role"] == "user" for r in result.facts)


def test_topic_elsewhere_in_the_same_verified_unit_can_supply_context(setup, monkeypatch):
    service = seed(setup, monkeypatch, ["[Morgan]: Morgan greeted Alex. It was a telescope demonstration."])
    result = service.search("Did Alex repair a telescope?", CTX, use_memories=True)
    assert result.evidence and result.facts == []


@pytest.mark.parametrize("narrator_first", [True, False])
@pytest.mark.parametrize("use_events", [False, True])
def test_context_hint_does_not_change_query_topics_with_card_order(setup, monkeypatch,
                                                                 narrator_first, use_events):
    fs, _, _, _ = setup
    messages = [
        message("[Morgan]: Morgan described Alex's visit to Paris. "
                + "The itinerary included museums and galleries. " * 30, id="narrator", sequence=0),
        message("[Alex]: Alex visited Paris.", id="actor", sequence=10),
    ]
    write_candidates(setup, messages, categories={"fact", "event"})
    service = EvidenceService(fs)
    original_uris = service.store.iter_uris

    def ordered_uris(directory, ctx, limit, **kwargs):
        uris = list(original_uris(directory, ctx, limit, **kwargs))
        if directory.endswith("/facts"):
            uris.sort(key=lambda uri: (
                fs.read_node(uri, ctx).metadata["deterministic"]["fact_records"][0]["subject"] != "Morgan"
            ), reverse=not narrator_first)
        return uris

    monkeypatch.setattr(service.store, "iter_uris", ordered_uris)
    result = service.search("What did Alex do in Paris?", CTX,
                                   use_memories=True, use_events=use_events, candidate_limit=1)
    assert [item["message_id"] for item in result.evidence] == ["actor"]
    assert len(result.facts) == 1
    assert result.facts[0]["subject"] == "Alex"
    assert result.facts[0]["quote"] == "Alex visited Paris."
    assert result.diagnostics["context_hints"] == 0
    assert result.diagnostics["events_admitted"] == int(use_events)


@pytest.mark.parametrize("query", ["What did Alex do?", "Did Alex repair a piano?"])
def test_names_alone_do_not_supply_context_hints(setup, monkeypatch, query):
    service = seed(setup, monkeypatch, ["[Morgan]: Morgan greeted Alex at a telescope demonstration."])
    assert not service.search(query, CTX, use_memories=True).evidence


def test_pronouns_and_auxiliaries_are_not_independent_topics(setup, monkeypatch):
    service = seed(setup, monkeypatch, [
        "[Morgan]: Morgan has been working on optical repairs for a long time."])
    assert not service.search("How long has Alex been married?", CTX, use_memories=True).evidence


def test_context_hint_rejects_tampered_record_and_revoked_or_other_user_sources(setup, monkeypatch):
    service = seed(setup, monkeypatch, ["[Morgan]: Morgan reported that Alex repaired a telescope."])
    query = "Did Alex repair a telescope?"
    fs = setup[0]
    original = fs.read_node

    def corrupt(uri, ctx):
        node = original(uri, ctx)
        if node.category == "fact":
            node = deepcopy(node)
            node.metadata["deterministic"]["fact_records"][0]["subject"] = "invented"
        return node

    with monkeypatch.context() as patch:
        patch.setattr(fs, "read_node", corrupt)
        assert not service.search(query, CTX, use_memories=True).evidence
    assert not service.search(query, replace(CTX, user_id="other"), use_memories=True).evidence
    setup[2].revoke_session(CTX)
    assert not service.search(query, CTX, use_memories=True).evidence


_DELETE_FIELD = object()


@pytest.mark.parametrize("field,value", [
    ("quote", None), ("subject", None), ("predicate", None), ("source_refs", 1), ("source_refs", [None]),
    ("nlp_model", {"name": []}), ("nlp_model", {"version": "1"}),
    pytest.param("source_refs.0.span", _DELETE_FIELD, id="missing-span"),
    pytest.param("source_refs.0.span", None, id="null-span"),
    ("source_refs.0.span", [0, 1.5]), ("source_refs.0.span", [False, 1]),
    pytest.param("source_refs.0.quote", _DELETE_FIELD, id="missing-ref-quote"),
    pytest.param("source_refs.0.quote", None, id="null-ref-quote"),
    pytest.param("source_refs.0.evidence_unit_id", _DELETE_FIELD, id="missing-unit-id"),
    pytest.param("source_refs.0.evidence_unit_id", None, id="null-unit-id"),
    ("source_refs.0.evidence_unit_id", []),
    ("source_refs.0.source_key", "invalid"), ("source_refs.0.content_hash", None),
])
@pytest.mark.parametrize("query", ["What did Alex recommend?", "What did Morgan recommend?"])
def test_bad_fact_does_not_hide_verified_facts_or_raw_messages(setup, monkeypatch, field, value, query):
    fs, _, _, _ = setup
    write_candidates(setup, [
        message("[Alex]: I recommend a telescope.", id="alex", sequence=0),
        message("[Alex]: I recommend a microscope.", id="other", sequence=10),
        message("[Morgan]: I recommend a binocular telescope.", id="morgan", sequence=20),
    ], categories={"fact"})
    read = fs.read_node

    def corrupt(uri, ctx):
        node = read(uri, ctx)
        if node.category != "fact":
            return node
        node = deepcopy(node)
        for record in node.metadata["deterministic"]["fact_records"]:
            if record["quote"] != "I recommend a telescope.":
                continue
            if field.startswith("source_refs.0."):
                ref = record["source_refs"][0]
                key = field.rsplit(".", 1)[1]
                if value is _DELETE_FIELD:
                    del ref[key]
                else:
                    ref[key] = value
            else:
                record[field] = value
        return node

    monkeypatch.setattr(fs, "read_node", corrupt)
    result = EvidenceService(fs).search(query, CTX, use_memories=True)
    expected = "I recommend a microscope." if "Alex" in query else "I recommend a binocular telescope."
    assert [fact["quote"] for fact in result.facts] == [expected]
    assert "alex" in {item["message_id"] for item in result.evidence}
    assert result.diagnostics["rejected_facts"] > 0
    assert not result.diagnostics["partial"]


def test_context_hint_verification_is_bounded_by_candidate_budget(setup, monkeypatch):
    import extraction.deterministic.verification as verification

    service = seed(setup, monkeypatch, [
        f"[Morgan]: Morgan reported that Alex repaired telescope number {i}." for i in range(8)])
    original, calls = verification.verify_record, []

    def verify(record, store, ctx, **kwargs):
        calls.append(record["fact_id"])
        return original(record, store, ctx, **kwargs)

    monkeypatch.setattr(verification, "verify_record", verify)
    result = service.search("Did Alex repair a telescope?", CTX,
                                   use_memories=True, candidate_limit=2, top_k=50)
    assert 0 < len(result.evidence) <= 2
    assert len(calls) <= 2
    assert result.facts == []


def test_context_hint_tokenizes_each_unit_once_when_multiple_records_miss(setup, monkeypatch):
    import service.evidence_service as search

    fs, _, _, _ = setup
    text = "Morgan greeted Alex. Morgan visited Rome. Morgan played music."
    messages = [message(text)]
    write_candidates(setup, messages, categories={"fact"})
    terms, seen = search.terms, []

    def track(value):
        if value == text:
            seen.append(value)
        return terms(value)

    monkeypatch.setattr(search, "terms", track)
    monkeypatch.setattr(search, "lexical_scores", lambda *args: {})
    result = EvidenceService(fs).search(
        "Did Alex repair a telescope?", CTX, use_memories=True)
    assert not result.evidence
    assert seen == [text]


@pytest.mark.parametrize("query,text", [
    ("How long have I been repairing bicycles?", "I have been preparing soup."),
    ("Which workshops did I say I've attended?", "I've visited a museum."),
    ("What different tools did I use?", "I enjoyed different exhibitions."),
    ("Which trains did I ride with colleagues?", "I visited a gallery with friends."),
])
def test_verified_records_cannot_rescue_functional_overlap(setup, monkeypatch, query, text):
    service = seed(setup, monkeypatch, [text])
    result = service.search(query, CTX, use_memories=True, use_events=True)
    assert result.evidence == [] and result.facts == []
