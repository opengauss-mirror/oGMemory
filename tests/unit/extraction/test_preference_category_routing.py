"""Default self-only preferences route supported third-party facts without blocking writes."""
from dataclasses import asdict, replace

import pytest

from core.models import CandidateMemory, RequestContext
from extraction.schemas.registry import SchemaRegistry
from extraction.tools import Extractor


def candidate(**changes):
    base = CandidateMemory(category="preference", owner_scope="user", routing_key="tea",
        abstract="Noor prefers unsweetened tea.", overview="Noor's drink choice.",
        content="Noor prefers unsweetened tea.", confidence=0.9, who="Noor",
        evidence_quote="I prefer unsweetened tea.", provenance_ids=["source-one"])
    return replace(base, **changes)


def extract(monkeypatch, item, *, user="Rae", role="user", source=None, registry=None):
    extractor = Extractor(object(), schema_registry=registry or SchemaRegistry())
    monkeypatch.setattr(extractor, "_identify_spans", lambda *a: [object()])
    monkeypatch.setattr(extractor, "_structure_spans", lambda *a: [item])
    ctx = RequestContext(account_id="scope-test", user_id=user, agent_id="agent",
                         session_id="session", trace_id="trace")
    return extractor.extract([{"role": role, "content": source or
        "[Rae]: Which drink?\n[Noor]: I prefer unsweetened tea."}], ctx)


def test_new_third_party_fact_routes_at_real_extractor_exit(monkeypatch):
    before = candidate(scope_ref="work")
    result, = extract(monkeypatch, before)
    assert result.category == "entity"
    assert result.routing_key == "noor" and result.object_name == "Noor"
    assert before.category == "preference"
    for name in ("abstract", "overview", "content", "evidence_quote", "provenance_ids", "scope_ref"):
        assert getattr(result, name) == getattr(before, name)
    assert not result.target_uri and result.expected_version is None


@pytest.mark.parametrize("changes", [
    {"target_uri": "ctx://scope-test/users/Rae/memories/preferences/tea", "expected_version": 2},
    {"expected_version": 1}, {"write_operation_id": "accepted"},
    {"legacy_routing_key": "old"}, {"evidence_quote": ""},
    {"evidence_quote": "She prefers tea."}, {"who": "Someone else"},
    {"owner_scope": "agent"},
])
def test_bound_legacy_or_unsupported_attribution_preserves_original(monkeypatch, changes):
    before = candidate(**changes)
    assert asdict(extract(monkeypatch, before)[0]) == asdict(before)


def test_actual_users_own_preference_stays_preference(monkeypatch):
    before = candidate()
    assert extract(monkeypatch, before, user="Noor")[0] == before


def test_quote_shared_by_two_speakers_is_not_reclassified(monkeypatch):
    before = candidate()
    source = "[Noor]: I prefer unsweetened tea.\n[Rae]: I prefer unsweetened tea."
    assert extract(monkeypatch, before, source=source)[0] == before


def test_custom_schema_is_not_reinterpreted(monkeypatch):
    registry = SchemaRegistry()
    registry.register(replace(registry.get("preference"), description="Preferences of any named holder."))
    before = candidate()
    assert extract(monkeypatch, before, registry=registry)[0] == before


def test_assistant_quote_does_not_authorize_reclassification(monkeypatch):
    assert extract(monkeypatch, candidate(), role="assistant") == []


def test_legal_holder_with_reserved_entity_name_keeps_valid_original(monkeypatch):
    from core.validation import validate_candidate

    before = candidate(who="Entity")
    source = "[Entity]: I prefer unsweetened tea."
    assert validate_candidate(before) == before
    result, = extract(monkeypatch, before, source=source)
    assert result == before
    assert validate_candidate(result) == result
