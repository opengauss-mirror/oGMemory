from __future__ import annotations

from urllib.parse import quote

import pytest

from core.models import RequestContext
from server.mcp_resources import (
    external_to_internal_uri,
    external_to_internal_uri_candidates,
    internal_to_external_uri,
    parse_ogmemory_uri,
)


def _ctx() -> RequestContext:
    return RequestContext(
        account_id="acct-1",
        user_id="alice",
        agent_id="bot",
        session_id="s1",
        trace_id="t1",
    )


def test_parse_status_and_types_resources():
    assert parse_ogmemory_uri("ogmemory://status").kind == "status"
    assert parse_ogmemory_uri("ogmemory://types").kind == "types"


def test_external_profile_uri_maps_to_current_user_context():
    assert (
        external_to_internal_uri("ogmemory://memories/profile/self", _ctx())
        == "ctx://acct-1/users/alice/memories/profile"
    )


def test_external_preference_uri_maps_to_current_user_context():
    assert (
        external_to_internal_uri("ogmemory://memories/preference/coding_style", _ctx())
        == "ctx://acct-1/users/alice/memories/preferences/coding_style"
    )


def test_external_tool_memory_uri_maps_to_current_agent_context():
    assert (
        external_to_internal_uri("ogmemory://memories/tool_memory/grep", _ctx())
        == "ctx://acct-1/agents/bot/memories/tools/grep"
    )


def test_external_agent_pattern_uri_maps_to_legacy_agent_context():
    assert (
        external_to_internal_uri("ogmemory://memories/pattern/agent/review", _ctx())
        == "ctx://acct-1/agents/bot/memories/patterns/review"
    )


def test_internal_uri_maps_to_external_without_leaking_ctx_uri():
    assert (
        internal_to_external_uri("ctx://acct-1/users/alice/memories/preferences/coding")
        == "ogmemory://memories/preference/coding"
    )


def test_public_uri_preserves_encoded_storage_fallback():
    assert external_to_internal_uri_candidates(
        "ogmemory://memories/preference/coffee%20shop", _ctx(),
    ) == [
        "ctx://acct-1/users/alice/memories/preferences/coffee shop",
        "ctx://acct-1/users/alice/memories/preferences/coffee%20shop",
    ]


@pytest.mark.parametrize("directory,public_type,owner", [
    ("entities", "entity", "users/alice"),
    ("patterns", "pattern", "agents/bot"),
])
def test_unicode_memory_uri_roundtrip_matches_stored_uri(directory, public_type, owner):
    internal = f"ctx://acct-1/{owner}/memories/{directory}/李雷"
    public = internal_to_external_uri(internal, category=public_type)

    assert external_to_internal_uri(public, _ctx()) == internal


def test_internal_skill_uri_maps_to_public_skill_type():
    assert (
        internal_to_external_uri(
            "ctx://acct-1/agents/bot/skills/generated/debug_tests"
        )
        == "ogmemory://skills/generated/debug_tests"
    )

    assert (
        internal_to_external_uri("ctx://acct-1/agents/bot/skills/debug_tests")
        == "ogmemory://skills/generated/debug_tests"
    )


def test_skill_uri_source_selects_distinct_internal_subdirectory():
    assert (
        external_to_internal_uri("ogmemory://skills/imported/debug_tests", _ctx())
        == "ctx://acct-1/agents/bot/skills/imported/debug_tests"
    )


def test_legacy_public_skill_uri_defaults_to_generated_subdirectory():
    parsed = parse_ogmemory_uri("ogmemory://skills/debug_tests")

    assert parsed.skill_source == "generated"
    assert (
        external_to_internal_uri("ogmemory://skills/debug_tests", _ctx())
        == "ctx://acct-1/agents/bot/skills/generated/debug_tests"
    )


def test_imported_skill_file_uri_preserves_nested_package_path():
    parsed = parse_ogmemory_uri(
        "ogmemory://skills/imported/review-code/references/api%20guide.md"
    )

    assert parsed.kind == "skill_file"
    assert parsed.memory_id == "review-code"
    assert parsed.skill_source == "imported"
    assert parsed.skill_path == "references/api guide.md"


def test_imported_skill_file_uri_allows_dot_runs_inside_path_segment():
    parsed = parse_ogmemory_uri(
        "ogmemory://skills/imported/review-code/references/v1..v2.md"
    )

    assert parsed.skill_path == "references/v1..v2.md"


@pytest.mark.parametrize(
    "uri",
    [
        "ogmemory://skills/generated/review-code/scripts/run.py",
        "ogmemory://skills/imported/review-code/../secret.txt",
        "ogmemory://skills/imported/review-code/%2e%2e/secret.txt",
    ],
)
def test_skill_file_uri_rejects_non_imported_or_unsafe_paths(uri):
    with pytest.raises(ValueError):
        parse_ogmemory_uri(uri)


def test_legacy_agent_pattern_uri_maps_to_scoped_public_uri():
    assert (
        internal_to_external_uri("ctx://acct-1/agents/bot/memories/patterns/review")
        == "ogmemory://memories/pattern/agent/review"
    )


@pytest.mark.parametrize(
    "uri",
    [
        "ctx://acct-1/users/alice/memories/profile",
        "file:///tmp/memory",
        "ogmemory://memories/preference/../secret",
        "ogmemory://memories/preference/a%2Fb",
        "ogmemory://memories/preference/a\\b",
    ],
)
def test_public_uri_validation_rejects_unsafe_inputs(uri):
    with pytest.raises(ValueError):
        parse_ogmemory_uri(uri)


@pytest.mark.parametrize("uri", [
    "http://example.test/memory",
    "https://example.test/memory",
    "ogmemory://memories/preference/target?view=content",
    "ogmemory://memories/preference/target#content",
    "ogmemory://memories/preference/target?",
    "ogmemory://memories/preference/target#",
    "\nogmemory://memories/preference/target",
    "ogmemory://memories/preference/tar\tget",
    "ogmemory://memories/preference/target%00",
    "ogmemory://memories/preference//target",
    "ogmemory://memories/preference/target/",
    "ogmemory://memories/preference/",
    "ogmemory://status/",
    "ogmemory://memories/preference/coffee shop",
    "ogmemory://memories/preference/bad%GG",
    "ogmemory://memories/preference/bad%FF",
])
def test_public_uri_rejects_text_that_changes_during_parsing(uri):
    with pytest.raises(ValueError):
        parse_ogmemory_uri(uri)


def test_encoded_reserved_characters_remain_part_of_memory_name():
    assert external_to_internal_uri(
        "ogmemory://memories/preference/Symbols%3FHash%23%E6%A0%87%E8%AE%B0", _ctx()
    ) == "ctx://acct-1/users/alice/memories/preferences/Symbols?Hash#标记"


def test_historical_percent_escape_fallback_preserves_ascii_case():
    encoded = quote("Mixed历史", safe="")
    assert external_to_internal_uri_candidates(
        "ogmemory://memories/preference/" + encoded, _ctx()
    ) == [
        "ctx://acct-1/users/alice/memories/preferences/Mixed历史",
        "ctx://acct-1/users/alice/memories/preferences/" + encoded,
        "ctx://acct-1/users/alice/memories/preferences/Mixed" + quote("历史", safe="").lower(),
    ]
