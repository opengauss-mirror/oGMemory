from __future__ import annotations

import logging

import pytest

from extraction.schemas.models import MemoryTypeSchema, SchemaField, FieldType
from extraction.schemas.registry import SchemaRegistry
from extraction.tool_builder import (
    build_extraction_tools,
    build_tool_to_category,
    parse_tool_call,
)


def test_build_extraction_tools_uses_enabled_schema_fields():
    registry = SchemaRegistry()

    tools = {tool["name"]: tool for tool in build_extraction_tools(registry)}

    profile = tools["extract_profile"]
    assert profile["input_schema"]["properties"]["routing_key"]["type"] == "string"
    assert "evidence_quote" in profile["input_schema"]["required"]
    assert profile["input_schema"]["properties"]["attribution_basis"]["enum"] == [
        "self_first_person",
        "self_named",
        "other_named",
    ]
    for tool_name in (
        "extract_preference",
        "extract_entity",
        "extract_event",
        "extract_case",
        "extract_pattern",
        "extract_skill",
        "extract_tool",
    ):
        tool_schema = tools[tool_name]["input_schema"]
        assert "routing_key" in tool_schema["properties"]
        assert "routing_key" in tool_schema["required"]
        assert "topic" not in tool_schema["properties"]
        assert "name" not in tool_schema["properties"]
        assert "event_name" not in tool_schema["properties"]
        assert "case_name" not in tool_schema["properties"]
        assert "skill_name" not in tool_schema["properties"]
        assert "tool_identifier" not in tool_schema["properties"]
        assert "provenance_ids" not in tool_schema["properties"]
    assert "extract_session_archive" not in tools
    assert "extract_session_summary" not in tools


def test_build_tool_to_category_maps_owner_scope():
    registry = SchemaRegistry()

    mapping = build_tool_to_category(registry)

    assert mapping["extract_preference"] == ("preference", "user")
    assert mapping["extract_pattern"] == ("pattern", "user")
    assert mapping["extract_skill"] == ("skill", "agent")
    assert "extract_session_archive" not in mapping
    assert "extract_session_summary" not in mapping


@pytest.mark.parametrize("topic_drafts", [False, True], ids=["complete-cards", "topic-drafts"])
@pytest.mark.parametrize("activation_enabled", [False, True], ids=["standard", "graph"])
def test_source_only_tools_preserve_enabled_fact_and_identity_contracts(topic_drafts, activation_enabled):
    registry = SchemaRegistry()
    native = {tool["name"]: tool for tool in build_extraction_tools(
        registry, topic_drafts=topic_drafts, activation_enabled=activation_enabled)}
    source = {tool["name"]: tool for tool in build_extraction_tools(
        registry, include_updates=False, topic_drafts=topic_drafts, activation_enabled=activation_enabled,
    )}
    assert set(source) == {schema.tool_name for schema in registry.list_compatible_enabled()
                           if schema.memory_type not in {"session_archive", "session_summary"}
                           and (activation_enabled or schema.memory_type != "relation")}
    for name, tool in source.items():
        expected = {key: value for key, value in native[name]["input_schema"]["properties"].items()
                    if key not in {"target_uri", "expected_version"}}
        assert tool["input_schema"]["properties"] == expected
        assert tool["input_schema"]["required"] == native[name]["input_schema"]["required"]
        assert tool["schema_version"] == native[name]["schema_version"]
        assert tool["metadata"] == native[name]["metadata"]
    profile = source["extract_profile"]
    assert "eligible, source-supported profile attributes" in profile["description"]
    assert "containing ALL supported profile attributes" in profile["description"]
    assert "evidence_quote" in profile["input_schema"]["required"]
    assert "attribution_basis" in profile["input_schema"]["required"]


def test_parse_tool_call_rejects_internal_storage_schemas():
    registry = SchemaRegistry()

    assert parse_tool_call(
        "extract_session_archive",
        {
            "routing_key": "chunk_1",
            "abstract": "Raw archive",
            "overview": "Raw archive",
            "content": "Raw archive",
            "confidence": 1.0,
        },
        registry,
    ) is None
    assert parse_tool_call(
        "extract_session_summary",
        {
            "routing_key": "summary_1",
            "abstract": "Summary",
            "events_with_dates": "None",
            "facts_and_details": "None",
            "preferences": "None",
            "plans_and_intentions": "None",
            "content": "Summary",
            "confidence": 1.0,
        },
        registry,
    ) is None


def test_parse_tool_call_auto_fills_common_fields_without_mutating_input():
    registry = SchemaRegistry()
    raw_input = {
        "abstract": "Likes dark roast coffee",
    }

    parsed = parse_tool_call("extract_preference", raw_input, registry)

    assert parsed is not None
    memory_type, owner_scope, candidate = parsed
    assert memory_type == "preference"
    assert owner_scope == "user"
    assert candidate.category == "preference"
    assert candidate.routing_key == "likes_dark_roast"
    assert candidate.content == "Likes dark roast coffee"
    assert candidate.overview == "Likes dark roast coffee"
    assert candidate.confidence == 0.5
    assert "topic" not in raw_input


def test_parse_tool_call_validates_profile_attribution_enum():
    registry = SchemaRegistry()

    parsed = parse_tool_call(
        "extract_profile",
        {
            "routing_key": "occupation",
            "abstract": "User is an engineer",
            "overview": "Engineer",
            "content": "User is an engineer",
            "confidence": 0.9,
            "evidence_quote": "I am an engineer",
            "attributed_speaker": "user",
            "attribution_basis": "unsupported",
        },
        registry,
    )

    assert parsed is None


def test_parse_unknown_tool_returns_none():
    registry = SchemaRegistry()

    assert parse_tool_call("extract_unknown", {"abstract": "x"}, registry) is None


def test_build_extraction_tools_includes_schema_version_metadata():
    registry = SchemaRegistry()

    profile_tool = {
        tool["name"]: tool
        for tool in build_extraction_tools(registry)
    }["extract_profile"]

    assert profile_tool["schema_version"] == "1.0"
    assert profile_tool["metadata"]["schema_version"] == "1.0"


def test_build_extraction_tools_warns_and_skips_incompatible_major_version(caplog):
    registry = SchemaRegistry(schemas_dir="/path/that/does/not/exist")
    registry.register(
        MemoryTypeSchema(
            memory_type="future",
            description="Future schema",
            directory="future",
            filename_template="{{ routing_key }}.md",
            operation_mode="upsert",
            version="99.0",
            fields=[
                SchemaField(
                    name="routing_key",
                    field_type=FieldType.STRING,
                    required=True,
                )
            ],
        )
    )

    with caplog.at_level(logging.WARNING):
        tools = build_extraction_tools(registry)

    assert tools == []
    assert "incompatible schema version" in caplog.text


def test_tool_mapping_skips_incompatible_schema_versions():
    registry = SchemaRegistry(schemas_dir="/path/that/does/not/exist")
    registry.register(
        MemoryTypeSchema(
            memory_type="future",
            description="Future schema",
            directory="future",
            filename_template="{{ routing_key }}.md",
            operation_mode="upsert",
            version="2.0",
            fields=[
                SchemaField(
                    name="routing_key",
                    field_type=FieldType.STRING,
                    required=True,
                )
            ],
        )
    )

    assert build_tool_to_category(registry) == {}


def test_parse_tool_call_rejects_incompatible_schema_versions():
    registry = SchemaRegistry(schemas_dir="/path/that/does/not/exist")
    registry.register(
        MemoryTypeSchema(
            memory_type="future",
            description="Future schema",
            directory="future",
            filename_template="{{ routing_key }}.md",
            operation_mode="upsert",
            version="2.0",
            fields=[
                SchemaField(
                    name="routing_key",
                    field_type=FieldType.STRING,
                    required=True,
                ),
                SchemaField(
                    name="abstract",
                    field_type=FieldType.STRING,
                    required=True,
                ),
                SchemaField(
                    name="overview",
                    field_type=FieldType.STRING,
                    required=True,
                ),
                SchemaField(
                    name="content",
                    field_type=FieldType.STRING,
                    required=True,
                ),
                SchemaField(
                    name="confidence",
                    field_type=FieldType.NUMBER,
                    required=True,
                ),
            ],
        )
    )

    parsed = parse_tool_call(
        "extract_future",
        {
            "routing_key": "x",
            "abstract": "x",
            "overview": "x",
            "content": "x",
            "confidence": 0.9,
        },
        registry,
    )

    assert parsed is None


def test_parse_tool_call_converts_all_supported_field_types():
    registry = SchemaRegistry(schemas_dir="/path/that/does/not/exist")
    registry.register(
        MemoryTypeSchema(
            memory_type="metric",
            description="Metric schema",
            directory="metrics",
            filename_template="{{ routing_key }}.md",
            operation_mode="upsert",
            fields=[
                SchemaField("routing_key", FieldType.STRING, required=True),
                SchemaField("abstract", FieldType.STRING, required=True),
                SchemaField("overview", FieldType.STRING, required=True),
                SchemaField("content", FieldType.STRING, required=True),
                SchemaField("confidence", FieldType.NUMBER, required=True),
                SchemaField("count", FieldType.INTEGER, required=False),
                SchemaField("enabled", FieldType.BOOLEAN, required=False),
                SchemaField("mode", FieldType.STRING, required=False, default="auto", enum=["auto", "manual"]),
            ],
        )
    )

    parsed = parse_tool_call(
        "extract_metric",
        {
            "routing_key": 123,
            "abstract": "Metric",
            "overview": "Metric overview",
            "content": "Metric content",
            "confidence": "0.75",
            "count": "3",
            "enabled": "yes",
        },
        registry,
    )

    assert parsed is not None
    _, _, candidate = parsed
    assert candidate.routing_key == "123"
    assert candidate.confidence == 0.75


def test_parse_tool_call_rejects_invalid_number_integer_boolean_and_enum():
    registry = SchemaRegistry(schemas_dir="/path/that/does/not/exist")
    registry.register(
        MemoryTypeSchema(
            memory_type="metric",
            description="Metric schema",
            directory="metrics",
            filename_template="{{ routing_key }}.md",
            operation_mode="upsert",
            fields=[
                SchemaField("routing_key", FieldType.STRING, required=True),
                SchemaField("abstract", FieldType.STRING, required=True),
                SchemaField("overview", FieldType.STRING, required=True),
                SchemaField("content", FieldType.STRING, required=True),
                SchemaField("confidence", FieldType.NUMBER, required=True),
                SchemaField("count", FieldType.INTEGER, required=False),
                SchemaField("enabled", FieldType.BOOLEAN, required=False),
                SchemaField("mode", FieldType.STRING, required=False, enum=["auto"]),
            ],
        )
    )

    assert parse_tool_call(
        "extract_metric",
        {
            "routing_key": "metric",
            "abstract": "Metric",
            "overview": "Metric",
            "content": "Metric",
            "confidence": "bad",
            "count": "bad",
            "enabled": "maybe",
            "mode": "manual",
        },
        registry,
    ) is None

    assert parse_tool_call(
        "extract_metric",
        {
            "routing_key": "metric",
            "abstract": "Metric",
            "overview": "Metric",
            "content": "Metric",
            "confidence": 0.8,
            "enabled": [],
        },
        registry,
    ) is None


def test_parse_event_keeps_routing_key_without_timestamp_suffix():
    parsed = parse_tool_call(
        "extract_event",
        {
            "routing_key": "launch",
            "abstract": "Product launched",
            "overview": "Launch",
            "content": "Product launched.",
            "confidence": 0.8,
        },
        SchemaRegistry(),
    )

    assert parsed is not None
    _, _, candidate = parsed
    assert candidate.routing_key == "launch"


def test_parse_case_uses_explicit_routing_key_without_timestamp_suffix():
    parsed = parse_tool_call(
        "extract_case",
        {
            "routing_key": "case-1",
            "abstract": "Solved issue",
            "overview": "Issue",
            "content": "Solved issue.",
            "confidence": 0.8,
        },
        SchemaRegistry(),
    )

    assert parsed is not None
    _, _, candidate = parsed
    assert candidate.routing_key == "case-1"


def test_parse_entity_and_tool_specific_fields():
    entity = parse_tool_call(
        "extract_entity",
        {
            "abstract": "OpenAI builds models",
            "overview": "OpenAI",
            "content": "OpenAI builds models.",
            "confidence": 0.9,
        },
        SchemaRegistry(),
    )
    tool = parse_tool_call(
        "extract_tool",
        {
            "routing_key": "grep",
            "abstract": "grep is useful for search",
            "overview": "grep",
            "content": "Use grep for search.",
            "confidence": 0.9,
            "best_for": "text search",
            "optimal_params": "-n",
            "common_failures": "binary files",
            "recommendation": "prefer rg when available",
        },
        SchemaRegistry(),
    )

    assert entity is not None
    assert entity[2].routing_key == "openai_builds_models"
    assert tool is not None
    assert tool[2].tool_stats == {
        "best_for": "text search",
        "optimal_params": "-n",
        "common_failures": "binary files",
        "recommendation": "prefer rg when available",
    }


def test_parse_profile_populates_attribution_fields():
    parsed = parse_tool_call(
        "extract_profile",
        {
            "routing_key": "occupation",
            "abstract": "User is an engineer",
            "overview": "Engineer",
            "content": "User is an engineer.",
            "confidence": 0.9,
            "evidence_quote": "I am an engineer",
            "attributed_speaker": "user",
            "attribution_basis": "self_first_person",
        },
        SchemaRegistry(),
    )

    assert parsed is not None
    _, _, candidate = parsed
    assert candidate.evidence_quote == "I am an engineer"
    assert candidate.attributed_speaker == "user"
    assert candidate.attribution_basis == "self_first_person"


def test_parse_preference_preserves_evidence_quote():
    parsed = parse_tool_call(
        "extract_preference",
        {
            "routing_key": "coffee",
            "abstract": "User likes dark roast coffee",
            "overview": "Coffee preference",
            "content": "User likes dark roast coffee.",
            "confidence": 0.9,
            "evidence_quote": "I like dark roast coffee",
        },
        SchemaRegistry(),
    )

    assert parsed is not None
    _, _, candidate = parsed
    assert candidate.evidence_quote == "I like dark roast coffee"


def test_parse_uses_defaults_for_missing_optional_fields():
    registry = SchemaRegistry(schemas_dir="/path/that/does/not/exist")
    registry.register(
        MemoryTypeSchema(
            memory_type="note",
            description="Note schema",
            directory="notes",
            filename_template="{{ routing_key }}.md",
            operation_mode="upsert",
            fields=[
                SchemaField("routing_key", FieldType.STRING, required=True),
                SchemaField("abstract", FieldType.STRING, required=True),
                SchemaField("overview", FieldType.STRING, required=True),
                SchemaField("content", FieldType.STRING, required=True),
                SchemaField("confidence", FieldType.NUMBER, required=True),
                SchemaField("where", FieldType.STRING, required=False, default="unknown"),
            ],
        )
    )

    parsed = parse_tool_call(
        "extract_note",
        {
            "routing_key": "n",
            "abstract": "Note",
            "overview": "Note",
            "content": "Note",
            "confidence": 0.8,
        },
        registry,
    )

    assert parsed is not None
    assert parsed[2].where == "unknown"


def test_parse_auto_fills_entity_pattern_event_and_boolean_false_branches():
    registry = SchemaRegistry()

    entity = parse_tool_call(
        "extract_entity",
        {
            "abstract": "OpenAI organization",
            "overview": "OpenAI",
            "content": "OpenAI organization.",
            "confidence": 0.9,
        },
        registry,
    )
    pattern = parse_tool_call(
        "extract_pattern",
        {
            "abstract": "Works late",
            "overview": "Late work",
            "content": "Works late.",
            "confidence": 0.9,
        },
        registry,
    )
    event = parse_tool_call(
        "extract_event",
        {
            "abstract": "Launch happened",
            "overview": "Launch",
            "content": "Launch happened.",
            "confidence": 0.9,
        },
        registry,
    )

    assert entity is not None
    assert entity[2].routing_key == "openai_organization"
    assert pattern is not None
    assert pattern[2].routing_key == "works_late"
    assert event is not None
    assert event[2].routing_key == "launch_happened"

    custom = SchemaRegistry(schemas_dir="/path/that/does/not/exist")
    custom.register(
        MemoryTypeSchema(
            memory_type="flag",
            description="Flag schema",
            directory="flags",
            filename_template="{{ routing_key }}.md",
            operation_mode="upsert",
            fields=[
                SchemaField("routing_key", FieldType.STRING, required=True),
                SchemaField("abstract", FieldType.STRING, required=True),
                SchemaField("overview", FieldType.STRING, required=True),
                SchemaField("content", FieldType.STRING, required=True),
                SchemaField("confidence", FieldType.NUMBER, required=True),
                SchemaField("enabled", FieldType.BOOLEAN, required=False),
            ],
        )
    )

    assert parse_tool_call(
        "extract_flag",
        {
            "routing_key": "flag",
            "abstract": "Flag",
            "overview": "Flag",
            "content": "Flag",
            "confidence": 0.8,
            "enabled": "false",
        },
        custom,
    ) is not None


def test_parse_fallback_routing_key_for_schema_without_identifier_field():
    registry = SchemaRegistry(schemas_dir="/path/that/does/not/exist")
    registry.register(
        MemoryTypeSchema(
            memory_type="memo",
            description="Memo schema",
            directory="memos",
            filename_template="content.md",
            operation_mode="upsert",
            fields=[
                SchemaField("abstract", FieldType.STRING, required=True),
                SchemaField("overview", FieldType.STRING, required=True),
                SchemaField("content", FieldType.STRING, required=True),
                SchemaField("confidence", FieldType.NUMBER, required=True),
            ],
        )
    )

    parsed = parse_tool_call(
        "extract_memo",
        {
            "abstract": "Fallback key value",
            "overview": "Memo",
            "content": "Memo",
            "confidence": 0.8,
        },
        registry,
    )

    assert parsed is not None
    assert parsed[2].routing_key == "fallback_key_value"


def test_parse_tool_call_rejects_null_required_fields_before_storage(caplog):
    raw_input = {
        "routing_key": "null_regression",
        "abstract": None,
        "overview": None,
        "content": None,
        "confidence": 0.8,
    }

    with caplog.at_level(logging.WARNING):
        parsed = parse_tool_call("extract_pattern", raw_input, SchemaRegistry())

    assert parsed is None
    for name in ("abstract", "overview", "content"):
        assert f"{name}: required field must not be null" in caplog.text
    assert raw_input["content"] is None


def test_parse_tool_call_preserves_valid_content_and_optional_null():
    parsed = parse_tool_call(
        "extract_pattern",
        {
            "routing_key": "cycling",
            "abstract": "Alice cycles daily.",
            "overview": "Daily cycling routine.",
            "content": "Alice cycles daily for exercise.",
            "confidence": 0.8,
            "when": None,
        },
        SchemaRegistry(),
    )

    assert parsed is not None
    assert parsed[2].content == "Alice cycles daily for exercise."
    assert parsed[2].when is None


def test_parse_tool_call_fills_null_generated_fields_before_required_validation():
    raw_input = {
        "abstract": "Alice cycles daily.",
        "overview": None,
        "content": None,
        "confidence": None,
    }

    parsed = parse_tool_call("extract_pattern", raw_input, SchemaRegistry())

    assert parsed is not None
    assert parsed[2].content == raw_input["abstract"]
    assert parsed[2].overview == raw_input["abstract"]
    assert parsed[2].confidence == 0.5
    assert raw_input["content"] is None
    assert raw_input["confidence"] is None
