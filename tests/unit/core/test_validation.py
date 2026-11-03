from unittest.mock import Mock

import pytest

from commit.merge_policies import AppendOnlyPolicy
from core.errors import ValidationError
from core.models import CandidateMemory, RequestContext
from core.uri_resolver import URIResolver
from core.validation import MAX_ABSTRACT_LEN, MAX_OVERVIEW_LEN, validate_candidate
from extraction.schemas.registry import SchemaRegistry


@pytest.mark.parametrize("category", ["entity", "preference", "pattern", "event", "case"])
def test_memory_category_keeps_readable_routing_key(category):
    candidate = CandidateMemory(
        category=category,
        owner_scope="user",
        routing_key="星河项目 / 接收器",
        abstract="星河项目接收器",
        overview="星河项目接收器的资料",
        content="星河项目接收器的详细资料",
        confidence=0.9,
    )

    assert validate_candidate(candidate).routing_key == "星河项目_接收器"


@pytest.mark.parametrize("key", [
    "%2faccounts%2fother%2fusers%2fvictim%2fmemories%2fprofile",
    " ",
])
def test_unsafe_name_cannot_reach_memory_uri(key):
    candidate = CandidateMemory(
        category="entity", owner_scope="user",
        routing_key=key,
        abstract="fact", overview="", content="", confidence=0.9,
    )

    with pytest.raises(ValidationError, match="safe URI path segment"):
        validate_candidate(candidate)


@pytest.mark.parametrize("key", ["a" * 600, "李" * 100])
def test_memory_name_respects_storage_path_limit(key):
    candidate = CandidateMemory(
        category="entity", owner_scope="user", routing_key=key,
        abstract="fact", overview="", content="", confidence=0.9,
    )

    with pytest.raises(ValidationError, match="path segment length limit"):
        validate_candidate(candidate)


@pytest.mark.parametrize("category,count", [("entity", 69), ("event", 61), ("case", 61)])
def test_multibyte_memory_name_fits_agfs_update_lock(category, count):
    candidate = CandidateMemory(
        category=category, owner_scope="user", routing_key="李" * count,
        abstract="fact", overview="", content="", confidence=0.9,
    )
    validated = validate_candidate(candidate)
    ctx = RequestContext(account_id="acct", user_id="alice", agent_id="bot", session_id="s", trace_id="t")
    resolver = URIResolver(SchemaRegistry())
    if category in {"event", "case"}:
        uri = AppendOnlyPolicy(Mock(), resolver).plan(validated, ctx).target_uri
    else:
        uri = resolver.resolve(category, {"routing_key": validated.routing_key}, ctx)

    lock_name = f".{uri.rsplit('/', 1)[-1]}.update-lock-{'a' * 32}"
    assert len(lock_name.encode("utf-8")) <= 255


@pytest.mark.parametrize("category,count", [("entity", 70), ("event", 62), ("case", 62)])
def test_multibyte_memory_name_rejects_oversized_agfs_path(category, count):
    candidate = CandidateMemory(
        category=category, owner_scope="user", routing_key="李" * count,
        abstract="fact", overview="", content="", confidence=0.9,
    )

    with pytest.raises(ValidationError, match="path segment length limit"):
        validate_candidate(candidate)


def test_invalid_utf8_routing_key_reports_validation_error():
    candidate = CandidateMemory(
        category="entity", owner_scope="user", routing_key="1\ud800",
        abstract="fact", overview="", content="", confidence=0.9,
    )

    with pytest.raises(ValidationError, match="Invalid UTF-8 encoding"):
        validate_candidate(candidate)


@pytest.mark.parametrize("count", [69, 70])
def test_memory_name_limit_uses_final_lowercase_bytes(count):
    candidate = CandidateMemory(
        category="entity", owner_scope="user", routing_key="İ" * count,
        abstract="fact", overview="", content="", confidence=0.9,
    )

    if count == 70:
        with pytest.raises(ValidationError, match="path segment length limit"):
            validate_candidate(candidate)
    else:
        validated = validate_candidate(candidate)
        ctx = RequestContext("acct", "alice", "bot", "session", "trace")
        uri = URIResolver(SchemaRegistry()).resolve(
            "entity", {"routing_key": validated.routing_key}, ctx,
        )
        lock_name = f".{uri.rsplit('/', 1)[-1]}.update-lock-{'a' * 32}"
        assert len(lock_name.encode("utf-8")) <= 255


@pytest.mark.parametrize("field,limit", [
    ("abstract", MAX_ABSTRACT_LEN),
    ("overview", MAX_OVERVIEW_LEN),
])
def test_summary_fields_preserve_exact_text_at_limit(field, limit):
    text = "a < b & c" + "x" * (limit - len("a < b & c"))
    candidate = CandidateMemory(
        category="entity", owner_scope="user", routing_key="example",
        abstract="valid", overview="valid", content="valid", confidence=0.9,
    )
    setattr(candidate, field, text)

    validated = validate_candidate(candidate)

    assert getattr(validated, field) == text
    assert getattr(candidate, field) == text


@pytest.mark.parametrize("field,limit", [
    ("abstract", MAX_ABSTRACT_LEN),
    ("overview", MAX_OVERVIEW_LEN),
])
def test_summary_fields_reject_oversize_without_truncation(field, limit):
    candidate = CandidateMemory(
        category="entity", owner_scope="user", routing_key="example",
        abstract="valid", overview="valid", content="valid", confidence=0.9,
    )
    text = "x" * (limit + 1)
    setattr(candidate, field, text)

    with pytest.raises(ValidationError, match=f"Length {limit + 1} exceeds maximum {limit}"):
        validate_candidate(candidate)

    assert getattr(candidate, field) == text
