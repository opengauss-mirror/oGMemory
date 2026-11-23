"""Recommendation intent identifies the actor without changing stored assertions."""

import pytest

from retrieval.evidence.ranking import fact_matches


def record(subject="Alex", *, role="user", predicate="recommend", speaker="Alex"):
    return dict(subject=subject, role=role, speaker=speaker, predicate=predicate,
                assertion_type="recommendation" if role == "assistant" else "observation")


@pytest.mark.parametrize("query", [
    "What did Alex recommend to me?", "What did Alex suggest to you?",
    "What has Alex recommended for telescope repairs?",
])
def test_named_recommender_precedes_first_and_second_person_recipients(query):
    assert fact_matches(query, record())
    assert not fact_matches(query, record("Taylor", speaker="Taylor"))
    assert not fact_matches(query, record("assistant", role="assistant"))
    assert not fact_matches(query, record(predicate="visit"))


@pytest.mark.parametrize("query", [
    "What did you recommend for telescope repairs?",
    "What did the AI assistant suggest?", "你之前推荐过什么？",
])
def test_explicit_assistant_recommendations_keep_their_role(query):
    assert fact_matches(query, record("assistant", role="assistant"))
    assert not fact_matches(query, record())


def test_unspecified_recommender_accepts_both_real_roles_but_not_experiences():
    query = "What was recommended for telescope repairs?"
    assert fact_matches(query, record())
    assert fact_matches(query, record("assistant", role="assistant"))
    assert not fact_matches(query, record(predicate="repair"))


def test_recommendation_never_becomes_adoption_or_another_persons_fact():
    assert not fact_matches("Did I visit Paris?", record("assistant", role="assistant"))
    assert not fact_matches("Did Alex visit Paris?", record("Alex", role="assistant"))
    assert not fact_matches("Did Alex visit Paris?", record("Morgan", speaker="Alex", predicate="visit"))
    assert fact_matches("Did Alex visit Paris?", record("Alex", speaker="Morgan", predicate="visit"))
    assert fact_matches("Paris", record("user", predicate="visit", speaker=""))


def test_self_query_does_not_map_transcript_speaker_to_current_user():
    assert fact_matches("What did I recommend?", record("user", speaker=""))
    assert not fact_matches("What did I recommend?", record())
    assert not fact_matches("What did I recommend?", record("assistant", role="assistant"))


def test_first_person_recipient_does_not_imply_first_person_recommender():
    query = "What was recommended to me for telescope repairs?"
    assert fact_matches(query, record())
    assert fact_matches(query, record("assistant", role="assistant"))
    assert not fact_matches(query, record(predicate="repair"))


def test_recommendations_remain_verified_observations_or_assistant_suggestions(setup):
    from commit.deterministic import deterministic_batch
    from service.evidence_service import EvidenceService
    from tests.unit.evidence.test_extractor import extractor
    from tests.unit.evidence.test_sources import CTX, message

    fs, api, store, _ = setup
    messages = [message("[Alex]: I recommend a telescope.", id="human", sequence=0),
                message("I recommend a telescope.", role="assistant", id="ai", sequence=10),
                message("[Alex]: I repaired a telescope.", id="experience", sequence=20)]
    store.register(messages, CTX, batch_id="recommendations")
    with deterministic_batch():
        for candidate in extractor().extract(messages, CTX):
            api._writer.write_candidate(candidate, CTX)
    service = EvidenceService(fs)
    human = service.search("What did Alex recommend to me?", CTX, use_memories=True)
    assert human.facts and all(r["role"] == "user" and r["subject"] == "Alex"
                               and r["predicate"] == "recommend" for r in human.facts)
    ai = service.search("What did you recommend?", CTX, use_memories=True)
    assert ai.facts and all(r["role"] == "assistant" and r["assertion_type"] == "recommendation"
                            for r in ai.facts)
