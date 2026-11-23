from datetime import UTC, datetime

import pytest

from commit.dreaming_facts import assess_facts, visible_facts
from core.dreaming_evidence import EVIDENCE_KEY, bind_archive_evidence
from core.models import ContextNode
from core.provenance_resolver import ProvenanceResolver
from session.models import ArchiveEntry


def _message(message_id, content, role="user"):
    return {"id": message_id, "role": role, "content": content}


def _archive(archive_id, *messages):
    return ArchiveEntry(archive_id, "session", "", "", list(messages))


def _evidence(claim, subject, quote, **fields):
    return {"claim": claim, "subject": subject, "quote": quote, "relation": "supports", **fields}


def _node(subject, *claims, category="profile"):
    bullets = "\n".join(f"- {claim}" for claim in claims)
    content = f"# {subject}\n\n## Profile\n{bullets}" if category == "entity" else bullets
    return ContextNode("ctx://account/users/user/memories/profiles/main", "MEMORY", category,
                       0, "user:user", subject, bullets, content, {"who": subject, "version": 1})


def _bind(node, archive, rows, message_ids=None):
    reference = ProvenanceResolver.build_id("archive", archive.archive_id,
                                            message_ids or [message["id"] for message in archive.messages])
    node.metadata.setdefault("provenance_ids", []).append(reference)
    node.metadata.setdefault(EVIDENCE_KEY, []).extend(bind_archive_evidence(rows, reference, archive.messages))


@pytest.mark.parametrize("subject,claim,affirmation,denial", [
    ("Alice", "live in Suzhou", "I live in Suzhou", "Alice does not live in Suzhou"),
    ("李四", "苏州人", "李四：我是苏州人", "李四：我不是苏州人"),
])
def test_structured_refutation_disputes_corroborated_claim(subject, claim, affirmation, denial):
    node = _node(subject, claim, category="entity")
    archives = [_archive("initial", _message("first", affirmation)),
                _archive("follow-up", _message("second", affirmation)),
                _archive("corrected", _message("denial", denial))]
    for archive in archives:
        message, = archive.messages
        fields = {"relation": "refutes"} if message["id"] == "denial" else {}
        _bind(node, archive, [_evidence(claim, subject, message["content"], **fields)])

    review = assess_facts(node, read_source={archive.archive_id: archive for archive in archives}.get)
    fact, = review["facts"].values()
    assert (fact["support"], fact["confidence"], review["recommendation"]) == ("disputed", .8, "observation")
    assert [proof["supports"] for proof in fact["evidence"]] == [True, True, False]


def test_subject_and_claim_are_bound_independently():
    claim = "works at Acme"
    node = _node("Alice", claim, "prefers tea")
    bob = "Bob, an engineer, works at Acme"
    alice = "Alice, an engineer, works at Acme"
    archive = _archive("one", _message("one", bob), _message("two", alice))
    _bind(node, archive, [_evidence(claim, "Bob", bob), _evidence(claim, "Alice", alice)])

    review = assess_facts(node, read_source={archive.archive_id: archive}.get)
    work, tea = review["facts"].values()
    assert [proof["quote"] for proof in work["evidence"]] == [alice]
    assert (tea["evidence_state"], review["evidence_state"], review["confidence"]) == ("pending", "pending", None)


def test_revised_source_leaves_claim_pending():
    claim = "The workshop starts at noon"
    node = _node("Alice", claim)
    archive = _archive("original", _message("one", claim))
    _bind(node, archive, [_evidence(claim, "Alice", claim, retain=True)])
    archive.messages[0]["content"] += "?"

    review = assess_facts(node, read_source=lambda _: archive)
    assert (review["evidence_state"], review["confidence"], review["recommendation"]) == ("pending", None, "pending")


def test_same_archived_message_counts_once():
    claim = "Works independently"
    node = _node("Alice", claim)
    archives = [_archive(key, _message("one", claim)) for key in ("first", "copy")]
    for archive in archives:
        _bind(node, archive, [_evidence(claim, "Alice", claim)])

    review = assess_facts(node, read_source={archive.archive_id: archive for archive in archives}.get)
    assert (review["confidence"], review["recommendation"]) == (.8, "observation")


def test_explicit_dated_correction_closes_only_the_matching_slot():
    old, new, other = "Harbor mooring: A7", "Harbor mooring: B6", "Harbor permit: C3"
    node = _node("Alice", old, new, other)
    quotes = ["Use A7 for the berth", "B6 replaces our berth on June 1", "Permit code C3"]
    archive = _archive("one", *(_message(str(index), text) for index, text in enumerate(quotes)))
    _bind(node, archive, [
        _evidence(old, "Alice", quotes[0], slot="berth", value="A7"),
        _evidence(new, "Alice", quotes[1], slot="berth", value="B6", effect="correction", valid_from="2026-06-01"),
        _evidence(other, "Alice", quotes[2], slot="permit", value="C3"),
    ])

    review = assess_facts(node, read_source=lambda _: archive)
    node.metadata["dreaming_review"] = {**review, "version": 1}
    before = visible_facts(node, datetime(2026, 5, 31, tzinfo=UTC))
    after = visible_facts(node, datetime(2026, 6, 1, tzinfo=UTC))
    assert {fact["abstract"] for fact in before.values()} == {old, other}
    assert {fact["abstract"] for fact in after.values()} == {new, other}


def test_quote_resolves_to_the_authorized_user_message():
    claim = "Opt-in confirmed"
    archive = _archive("one", _message("agent", claim, role="assistant"),
                       _message("authorized", claim), _message("outside", claim))
    node = _node("Alice", claim)
    _bind(node, archive, [_evidence(claim, "Alice", claim, retain=True)], ["authorized", "agent"])

    review = assess_facts(node, read_source=lambda _: archive)
    fact, = review["facts"].values()
    assert [proof["message_id"] for proof in fact["evidence"]] == ["authorized"]
    assert (review["confidence"], review["value_reason"]) == (.8, "explicit")


def test_overlapping_support_counts_both_sources_only_in_the_overlap():
    claim = "Access enabled"
    first, second = "Access is enabled for September", "Access also applies Sept 15 through Oct 14"
    archive = _archive("one", _message("first", first), _message("second", second))
    node = _node("Alice", claim)
    _bind(node, archive, [
        _evidence(claim, "Alice", first, valid_from="2026-09-01", valid_to="2026-10-01"),
        _evidence(claim, "Alice", second, valid_from="2026-09-15", valid_to="2026-10-15"),
    ])
    node.metadata["dreaming_review"] = {**assess_facts(node, read_source=lambda _: archive), "version": 1}

    for month, day, expected in [(9, 1, (.8, "observation")), (9, 20, (64 / 65, "corroborated")),
                                 (10, 1, (.8, "observation"))]:
        fact, = visible_facts(node, datetime(2026, month, day, tzinfo=UTC)).values()
        assert (fact["confidence"], fact["support"]) == expected


def test_source_correction_precedes_node_visibility_window():
    old, new = "Home: Suzhou", "Home: Hangzhou"
    archive = _archive("one", _message("first", old), _message("second", new))
    node = _node("Alice", old, new)
    node.metadata["valid_from"] = "2026-10-01"
    _bind(node, archive, [
        _evidence(old, "Alice", old, slot="home", value="Suzhou", valid_from="2026-08-01"),
        _evidence(new, "Alice", new, slot="home", value="Hangzhou", valid_from="2026-09-01", effect="correction"),
    ])

    review = assess_facts(node, read_source=lambda _: archive)
    node.metadata["dreaming_review"] = {**review, "version": 1}
    historical, = (fact for fact in review["facts"].values() if fact["abstract"] == old)
    assert historical["valid_to"] == "2026-09-01T00:00:00+00:00"
    current, = visible_facts(node, datetime(2026, 10, 1, tzinfo=UTC)).values()
    assert (current["abstract"], current["valid_from"]) == (new, "2026-10-01T00:00:00+00:00")
