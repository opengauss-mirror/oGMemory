"""Owner identity remains exact while content fields retain slug formatting."""

from dataclasses import replace

import pytest

from core.errors import ValidationError
from core.models import RequestContext
from core.uri_resolver import URIResolver
from extraction.extraction_react_loop import ExtractionReActLoop
from extraction.schemas.registry import SchemaRegistry
from fs.agfs_adapter.agfs_context_fs import AGFSContextFS, uri_to_path
from fs.sql_adapter.sql_context_fs import SQLContextFS
from tests.unit.commit.test_memory_identity import memory


@pytest.fixture
def registry():
    return SchemaRegistry()


@pytest.mark.parametrize("principal", [
    "MixedCase", "Équipe研发", "No\u0308ra", "Lab.One+East@2", ".hidden", "+", " Space Owner ",
])
def test_all_schema_owner_paths_preserve_exact_identity_and_backend_access(registry, principal):
    ctx = RequestContext(principal, principal, principal, "session", "trace")
    resolver = URIResolver(registry)
    sql, agfs = object.__new__(SQLContextFS), object.__new__(AGFSContextFS)
    for schema in registry.list_compatible_enabled():
        uri = resolver.resolve(schema.memory_type, {"routing_key": "Mixed Topic"}, ctx)
        directory = resolver.get_directory_uri(schema.memory_type, ctx).rstrip("/")
        assert uri == directory if schema.is_single_file else uri.rsplit("/", 1)[0] == directory
        assert uri.startswith(f"ctx://{principal}/")
        owner = "agents" if schema.owner_scope == "agent" else "users"
        assert f"/{owner}/{principal}/" in uri
        assert f"/{owner}/{principal}/" in uri_to_path(uri)
        assert sql._is_accessible(uri, ctx) and agfs._is_accessible(uri, ctx)
        assert resolver.validate_uri(uri, ctx)


@pytest.mark.parametrize("field,category", [
    *(('account_id', category) for category in ('profile', 'event', 'pattern', 'case', 'skill', 'tool')),
    *(('user_id', category) for category in ('profile', 'event', 'pattern')),
    *(('agent_id', category) for category in ('case', 'skill', 'tool')),
])
def test_case_distinct_principals_remain_distinct_and_unauthorized(registry, category, field):
    resolver = URIResolver(registry)
    original = RequestContext("Acme", "Nora", "Agent", "session", "trace")
    variant = replace(original, **{field: getattr(original, field).lower()})
    original_uri = resolver.resolve(category, {"routing_key": "topic"}, original)
    variant_uri = resolver.resolve(category, {"routing_key": "topic"}, variant)
    assert original_uri != variant_uri
    for fs in (object.__new__(SQLContextFS), object.__new__(AGFSContextFS)):
        assert fs._is_accessible(original_uri, original)
        assert not fs._is_accessible(original_uri, variant)
        assert not fs._is_accessible(variant_uri, original)


@pytest.mark.parametrize("category", ["event", "case", "profile"])
def test_reserved_fields_and_internal_tool_stats_cannot_override_context(registry, category):
    resolver = URIResolver(registry)
    ctx = RequestContext("Acme", "Nora", "Agent", "session", "trace")
    reserved = {"account_id": "other", "user_id": "other", "agent_id": "other"}
    expected = resolver.resolve(category, {"routing_key": "topic"}, ctx)
    assert resolver.resolve(category, {"routing_key": "topic", **reserved}, ctx) == expected
    candidate = replace(memory("topic", category=category), tool_stats=reserved)
    fields = ExtractionReActLoop._candidate_to_fields(object.__new__(ExtractionReActLoop), candidate)
    assert resolver.resolve(category, fields, ctx) == expected
    assert candidate.tool_stats == reserved


@pytest.mark.parametrize("value,expected", [
    (" Mixed Topic ", "mixed_topic"), ("A/B\\C..D", "a_bcd"), ("İ", "i\u0307"),
])
def test_lowercase_owner_and_content_slug_behavior_stays_compatible(registry, value, expected):
    ctx = RequestContext("acme", "nora", "agent", "session", "trace")
    resolver = URIResolver(registry)
    assert resolver.resolve("event", {"routing_key": value}, ctx) == f"ctx://acme/users/nora/memories/events/{expected}"
    assert resolver.resolve("profile", {}, ctx) == "ctx://acme/users/nora/memories/profile"


@pytest.mark.parametrize("field,category", [("account_id", "event"), ("user_id", "event"), ("agent_id", "case")])
@pytest.mark.parametrize("value", ["", ".", "..", "a..b", "a/b", "a\\b", "a%2Fb", "a\0b", "a\nb", "a\u0085b", "a\ud800b"])
def test_unsupported_principal_rejected_by_both_uri_builders(registry, field, category, value):
    ctx = replace(RequestContext("acme", "nora", "agent", "session", "trace"), **{field: value})
    resolver = URIResolver(registry)
    for build in (lambda: resolver.resolve(category, {"routing_key": "topic"}, ctx),
                  lambda: resolver.get_directory_uri(category, ctx)):
        with pytest.raises((ValueError, ValidationError), match=field):
            build()


def test_unused_empty_agent_does_not_block_user_schema(registry):
    resolver = URIResolver(registry)
    ctx = RequestContext("Acme", "Nora", "", "session", "trace")
    for schema in registry.list_compatible_enabled():
        if schema.owner_scope == "user":
            assert resolver.resolve(schema.memory_type, {"routing_key": "topic"}, ctx).startswith("ctx://Acme/users/Nora/")
