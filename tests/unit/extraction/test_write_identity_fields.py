"""Keep source identity through model tool parsing and memory writes."""

from dataclasses import replace

import pytest

from commit.candidate_pipeline import CandidatePipeline
from core.models import RequestContext
from core.validation import MAX_ABSTRACT_LEN, MAX_CONTENT_LEN, MAX_OVERVIEW_LEN
from extraction.schemas.models import FieldType, SchemaField
from extraction.schemas.registry import SchemaRegistry
from extraction.tool_builder import build_extraction_tools, parse_tool_call
from extraction.tools import Extractor
from service.api import MemoryWriteAPI
from tests.unit.commit.test_memory_identity import MemoryFS, MergeLLM


def operation(**fields):
    return dict(routing_key="atlas", abstract="Atlas project", overview="Atlas project",
                content="Atlas project", confidence=0.9, **fields)


def test_tools_expose_source_identity_and_text_limits():
    tools = {t["name"]: t["input_schema"] for t in build_extraction_tools(SchemaRegistry())}
    for category in ("entity", "preference", "pattern", "event", "case"):
        assert tools[f"extract_{category}"]["properties"]["scope_ref"]["type"] == "string"
        assert "scope_ref" in tools[f"extract_{category}"]["required"]
    assert "object_name" in tools["extract_entity"]["properties"]
    assert "title" in tools["extract_event"]["properties"]
    for field, limit in (("abstract", MAX_ABSTRACT_LEN), ("overview", MAX_OVERVIEW_LEN),
                         ("content", MAX_CONTENT_LEN)):
        assert tools["extract_entity"]["properties"][field]["maxLength"] == limit
    assert "scope_ref" not in tools["extract_profile"]["properties"]


def test_custom_identity_field_constraints_are_preserved():
    registry = SchemaRegistry()
    schema = registry.get("entity")
    registry.register(replace(schema, fields=[*schema.fields,
        SchemaField("scope_ref", FieldType.STRING, required=True,
                    description="Authorized projects only", enum=["Project A", "Project B"])]))
    tool = next(t for t in build_extraction_tools(registry) if t["name"] == "extract_entity")
    assert tool["input_schema"]["properties"]["scope_ref"] == {
        "type": "string", "description": "Authorized projects only", "enum": ["Project A", "Project B"]}
    assert parse_tool_call("extract_entity", operation(scope_ref="Project C"), registry) is None


def test_custom_identity_default_is_retained_when_field_is_omitted():
    registry = SchemaRegistry()
    schema = registry.get("entity")
    registry.register(replace(schema, fields=[*schema.fields,
        SchemaField("scope_ref", FieldType.STRING, default="Project A")]))
    candidate = parse_tool_call("extract_entity", operation(), registry)[2]
    assert candidate.scope_ref == "Project A"


def test_read_target_merges_aliases_without_creating_competing_updates():
    registry = SchemaRegistry()
    candidates = [parse_tool_call("extract_entity", operation(
        object_name=name, scope_ref="Project A", target_uri="ctx://test/entities/atlas", expected_version=1), registry)[2]
        for name in ("Atlas", "Atlas device")]
    assert len(Extractor._merge_dual_run(candidates)) == 1


def test_parse_preserves_original_subject_scope_and_input():
    raw = operation(object_name="阿特拉斯", scope_ref="项目甲")
    candidate = parse_tool_call("extract_entity", raw, SchemaRegistry())[2]
    assert candidate.object_name == "阿特拉斯"
    assert candidate.scope_ref == "项目甲"
    assert raw == operation(object_name="阿特拉斯", scope_ref="项目甲")


@pytest.mark.parametrize("field,value", [("scope_ref", {}), ("scope_ref", 1), ("object_name", []),
                                        ("object_name", False)])
def test_identity_fields_reject_non_text_values(field, value):
    assert parse_tool_call("extract_entity", operation(**{field: value}), SchemaRegistry()) is None


def test_named_entities_with_same_key_in_different_scopes_do_not_merge():
    registry = SchemaRegistry()
    fs = MemoryFS()
    api = MemoryWriteAPI(fs, MergeLLM(), schema_registry=registry, directory_summary_enabled=False)
    ctx = RequestContext("identity-fields", "reader", "agent", "session", "trace")
    candidates = [parse_tool_call("extract_entity", operation(object_name="Atlas", scope_ref=scope), registry)[2]
                  for scope in ("project-a", "project-b")]
    first, second = [api.write_memory(c, ctx) for c in candidates]
    assert first["target_uri"] != second["target_uri"]
    assert len(fs.nodes) == 2
    assert {n.metadata["scope_ref"] for n in fs.nodes.values()} == {"project-a", "project-b"}


def test_event_title_is_stored_with_source_scope():
    registry = SchemaRegistry()
    candidate = parse_tool_call("extract_event", operation(title="Atlas review", scope_ref="project-a"), registry)[2]
    fs = MemoryFS()
    api = MemoryWriteAPI(fs, MergeLLM(), schema_registry=registry, directory_summary_enabled=False)
    ctx = RequestContext("identity-fields", "reader", "agent", "session", "trace")
    result = api.write_memory(candidate, ctx)
    stored = fs.read_node(result["target_uri"], ctx)
    assert stored.metadata["title"] == "Atlas review"
    assert stored.metadata["scope_ref"] == "project-a"


def test_scoped_candidates_survive_dual_run_merge_and_pipeline_deduplication():
    registry = SchemaRegistry()
    candidates = [parse_tool_call("extract_entity", operation(object_name="Atlas", scope_ref=scope), registry)[2]
                  for scope in ("project-a", "project-b")]
    merged = Extractor._merge_dual_run(candidates * 2)
    assert len(merged) == 2
    assert len(CandidatePipeline().deduplicate(merged)) == 2


def test_preferences_with_distinct_holders_survive_deduplication():
    registry = SchemaRegistry()
    candidates = [parse_tool_call("extract_preference", operation(who=holder), registry)[2]
                  for holder in ("Alice", "Bob")]
    assert len(Extractor._merge_dual_run(candidates)) == 2
    assert len(CandidatePipeline().deduplicate(candidates)) == 2
