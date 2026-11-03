"""Allocated scope/holder prefixes do not consume the extraction-name budget."""

from dataclasses import replace

import pytest

from commit.candidate_pipeline import CandidatePipeline
from core.errors import ToolCallProtocolError, ValidationError
from core.models import RequestContext
from core.validation import MAX_MEMORY_KEY_BYTES, validate_candidate
from extraction.schemas.registry import SchemaRegistry
from extraction.tool_builder import parse_tool_call, validate_read_target
from service.api import MemoryWriteAPI
from tests.unit.commit.test_memory_identity import DecisionLLM, MemoryFS, api, memory


@pytest.fixture
def ctx():
    return RequestContext("qualified-test", "reader", "agent", "session", "trace")


def update_input(operation, uri, node):
    inputs = dict(target_uri=uri, expected_version=node.metadata["version"],
                  abstract="Updated fact and retained fact.",
                  overview="Updated fact and retained fact.", confidence=.9)
    if operation == "append_memory":
        inputs["text"] = "Additional fact."
    elif operation == "edit_memory":
        inputs.update(search="Initial", replacement="Updated")
    else:
        inputs.update(content="Updated fact. Retained fact.",
                      routing_key=node.metadata["routing_key"],
                      scope_ref=node.metadata["scope_ref"],
                      object_name=node.metadata["object_name"],
                      who=node.metadata.get("who"))
    return inputs


@pytest.mark.parametrize("category", ["entity", "preference", "pattern"])
@pytest.mark.parametrize("operation", ["append_memory", "edit_memory", "snapshot"])
@pytest.mark.parametrize("scope", ["project_" + "a" * 112, "a" * 100 + "中" * 33])
def test_long_allocated_key_supports_read_update_and_revalidation(ctx, category, operation, scope):
    fs = MemoryFS()
    writer = api(fs, DecisionLLM())
    original = validate_candidate(memory("subject", category=category, scope_ref=scope,
                                         content="Initial fact. Retained fact."))
    uri = writer.write_memory(original, ctx)["target_uri"]
    assert writer.write_memory(original, ctx)["target_uri"] == uri
    node = fs.read_node(uri, ctx)
    assert len(node.metadata["routing_key"]) > 100
    assert len(uri.rsplit("/", 1)[-1].encode()) <= MAX_MEMORY_KEY_BYTES
    name = f"extract_{category}" if operation == "snapshot" else operation
    parsed = parse_tool_call(name, update_input(operation, uri, node), SchemaRegistry(),
                             read_nodes={uri: node}, ctx=ctx)
    candidate = replace(parsed[2], provenance_ids=["source:update"])
    validate_read_target(candidate, {uri: 1})
    candidate = CandidatePipeline._validate_candidates([candidate], ctx)[0]
    assert candidate.routing_key == node.metadata["routing_key"]
    assert writer.write_memory(candidate, ctx)["target_uri"] == uri
    stored = fs.read_node(uri, ctx)
    assert stored.content == (node.content + "\nAdditional fact." if operation == "append_memory"
                              else "Updated fact. Retained fact.")
    assert stored.metadata["scope_ref"] == scope
    assert stored.metadata["routing_key"] == node.metadata["routing_key"]
    assert stored.metadata["version"] == 2
    assert set(stored.metadata["provenance_ids"]) == {*original.provenance_ids, "source:update"}
    assert len(fs.nodes) == 1
    with pytest.raises(ValueError, match="changed since extraction"):
        writer.write_memory(replace(candidate, content="A stale overwrite."), ctx)


@pytest.mark.parametrize("operation", ["append_memory", "edit_memory", "snapshot"])
@pytest.mark.parametrize("scope,holder", [
    ("project", "review_panel_" + "a" * 110),
    ("review_panel", "review_panel_" + "a" * 110),
    ("review_panel_" + "a" * 110, "review_panel"),
])
def test_long_preference_holder_key_supports_updates(ctx, operation, scope, holder):
    fs = MemoryFS()
    writer = api(fs, DecisionLLM())
    original = memory("briefs", category="preference", who=holder,
                      scope_ref=scope, content="Initial fact. Retained fact.")
    uri = writer.write_memory(original, ctx)["target_uri"]
    node = fs.read_node(uri, ctx)
    name = "extract_preference" if operation == "snapshot" else operation
    candidate = parse_tool_call(name, update_input(operation, uri, node), SchemaRegistry(),
                                read_nodes={uri: node}, ctx=ctx)[2]
    candidate = validate_candidate(candidate)
    assert writer.write_memory(candidate, ctx)["target_uri"] == uri
    assert fs.read_node(uri, ctx).metadata["who"] == holder


@pytest.mark.parametrize("bad", ["no_read", "owner", "category", "version", "path", "body"])
def test_long_key_incremental_update_still_requires_complete_authorized_read(ctx, bad):
    fs = MemoryFS()
    uri = api(fs, DecisionLLM()).write_memory(
        memory("subject", scope_ref="project_" + "a" * 112, content="Initial fact."), ctx,
    )["target_uri"]
    node = fs.read_node(uri, ctx)
    inputs = update_input("append_memory", uri, node)
    reads = {uri: node}
    if bad == "no_read":
        reads.clear()
    elif bad == "owner":
        node.owner_space = "user:other"
    elif bad == "category":
        node.category = "preference"
    elif bad == "version":
        inputs["expected_version"] = 2
    elif bad == "path":
        node.uri += "/child"
    else:
        node.content = ""
    with pytest.raises(ToolCallProtocolError):
        parse_tool_call("append_memory", inputs, SchemaRegistry(), read_nodes=reads, ctx=ctx)
    assert fs.read_node(uri, ctx).metadata["version"] == 1


@pytest.mark.parametrize("bad", ["scope", "missing_target", "owner", "category"])
def test_long_key_snapshot_cannot_change_target_namespace_or_create(ctx, bad):
    fs = MemoryFS()
    writer = api(fs, DecisionLLM())
    original = memory("subject", scope_ref="project_" + "a" * 112, content="Initial fact.")
    uri = writer.write_memory(original, ctx)["target_uri"]
    node = fs.read_node(uri, ctx)
    candidate = validate_candidate(replace(original, routing_key=node.metadata["routing_key"],
                                           target_uri=uri, expected_version=1, content="Updated fact."))
    if bad == "scope":
        candidate = replace(candidate, scope_ref="another project")
    elif bad == "missing_target":
        candidate = replace(candidate, target_uri=uri + "_missing")
    elif bad == "owner":
        candidate = replace(candidate, target_uri=uri.replace("/reader/", "/other/"))
    else:
        candidate = replace(candidate, category="preference")
    with pytest.raises(ValueError):
        writer.write_memory(candidate, ctx)
    assert len(fs.nodes) == 1
    assert fs.read_node(uri, ctx).metadata["version"] == 1


@pytest.mark.parametrize("category", ["entity", "preference", "pattern", "event", "case"])
@pytest.mark.parametrize("target", ["", "ctx://qualified-test/users/reader/memories/entities/" + "a" * 101])
def test_unqualified_long_model_key_is_rejected_even_with_target(category, target):
    with pytest.raises(ValidationError, match="path segment length limit"):
        validate_candidate(memory("a" * 101, category=category, target_uri=target,
                                  expected_version=1 if target else None))


@pytest.mark.parametrize("category", ["event", "case"])
def test_occurrence_name_limit_does_not_discount_scope_or_holder(category):
    with pytest.raises(ValidationError, match="path segment length limit"):
        validate_candidate(memory("a" * 101 + "_subject", category=category,
                                  scope_ref="a" * 101, who="a" * 101))


@pytest.mark.parametrize("key", ["project_" + "a" * 101, "project_" + "中" * 68,
                                 "project_" + "a" * 101 + "/child"])
def test_qualified_key_retains_stem_byte_and_path_limits(key):
    with pytest.raises(ValidationError):
        validate_candidate(memory(key, scope_ref="project"))


@pytest.mark.parametrize("operation", ["append_memory", "edit_memory", "snapshot"])
def test_custom_filename_retains_long_routing_key_and_rendered_byte_budget(ctx, operation):
    registry = SchemaRegistry()
    registry.register(replace(registry.get("entity"), filename_template="item_{{ routing_key }}.md"))
    fs = MemoryFS()
    writer = MemoryWriteAPI(fs, DecisionLLM(), schema_registry=registry, directory_summary_enabled=False)
    # The name plus the template prefix exactly fills the path-byte budget.
    scope = "a" * (MAX_MEMORY_KEY_BYTES - len("item__subject"))
    original = memory("subject", scope_ref=scope, content="Initial fact. Retained fact.")
    uri = writer.write_memory(original, ctx)["target_uri"]
    assert len(uri.rsplit("/", 1)[-1].encode()) == MAX_MEMORY_KEY_BYTES
    node = fs.read_node(uri, ctx)
    name = "extract_entity" if operation == "snapshot" else operation
    parsed = parse_tool_call(name, update_input(operation, uri, node), registry,
                             read_nodes={uri: node}, ctx=ctx)
    candidate = validate_candidate(parsed[2])
    assert writer.write_memory(candidate, ctx)["target_uri"] == uri
    assert fs.read_node(uri, ctx).metadata["routing_key"] == scope + "_subject"
    with pytest.raises(ValidationError, match="path segment length limit"):
        writer.write_memory(replace(original, scope_ref=scope + "a"), ctx)
    assert len(fs.nodes) == 1
