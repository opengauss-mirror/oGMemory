from dataclasses import replace

from core.models import RequestContext
from memorybot.templates import TemplateRegistry


def identity():
    return RequestContext(
        "account",
        "alice",
        "assistant",
        "session",
        "trace",
        visible_owner_spaces=("user:alice", "agent:assistant"),
    )


def published(template_id, requirements, **extra):
    value = TemplateRegistry().get("weekly_report")
    value.update(id=template_id, permission_requirements=requirements, **extra)
    return value


def test_builtin_templates_expose_public_contract_for_current_user():
    rows = TemplateRegistry().visible(identity(), {"memory_scope": "user", "resource_ids": []})
    assert {row["id"] for row in rows} == {
        "weekly_report",
        "handover",
        "handbook",
        "release_check",
        "csv_summary",
    }
    assert all(
        set(row) == {"id", "name", "input", "output", "permission_requirements"} for row in rows
    )


def test_published_requirements_use_current_identity_selected_resource_and_registered_tool():
    ctx = identity()
    source = "ctx://account/agents/assistant/skills/imported/review"
    resource = "ctx://account/users/alice/memories/documents/release"
    template = published(
        "review",
        [
            {"action": "memory.read", "scope": "user:alice"},
            {"action": "resource.read", "scope": resource},
            {"action": "file.write", "scope": "user"},
            {"action": "tool.execute", "scope": "read_memory"},
        ],
        source_uri=source,
        tools=["read_memory"],
    )
    registry = TemplateRegistry([template])
    reads = []

    def read(uri):
        reads.append(uri)
        return {"uri": uri, "owner_space": "agent:assistant" if uri == source else "user:alice"}

    scope = {"memory_scope": "user", "resource_ids": [resource]}
    assert "review" in {row["id"] for row in registry.visible(ctx, scope, read)}
    frozen = registry.validate(
        "review", "整理发布", [], "markdown", ctx=ctx, scope=scope, resource_reader=read
    )
    assert frozen["tools"] == ["read_memory"] and len(frozen["revision"]) == 64
    first_reads = len(reads)
    assert registry.authorize(frozen, replace(ctx, trace_id="resumed"), scope, read) == frozen
    assert reads[first_reads:] == [source, resource]


def test_authorized_agent_requirements_follow_native_space_grants():
    ctx = replace(identity(), visible_owner_spaces=("user_alice", "agent_assistant"))
    resource = "ctx://account/agents/assistant/memories/documents/checklist"
    registry = TemplateRegistry(
        [
            published(
                "agent_review",
                [
                    {"action": "memory.read", "scope": "agent:assistant"},
                    {"action": "resource.read", "scope": "authorized"},
                    {"action": "file.write", "scope": "user:alice"},
                    {"action": "tool.execute", "scope": "search_memory"},
                ],
                tools=["search_memory"],
            )
        ]
    )
    scope = {"memory_scope": "authorized", "resource_ids": [resource]}

    def read(uri):
        return {"uri": uri, "owner_space": "agent:assistant"}

    frozen = registry.validate(
        "agent_review", "整理案例", [], "markdown", ctx=ctx, scope=scope, resource_reader=read
    )
    assert frozen["id"] == "agent_review"
    assert "agent_review" in {row["id"] for row in registry.visible(ctx, scope, read)}


def test_resource_owner_requirement_and_file_input_are_validated_together():
    ctx = identity()
    resource = "ctx://account/users/alice/memories/documents/notes"
    registry = TemplateRegistry(
        [
            published(
                "source_review",
                [
                    {"action": "memory.read", "scope": "authorized"},
                    {"action": "resource.read", "scope": "user"},
                    {"action": "file.write", "scope": "user"},
                ],
            )
        ]
    )
    scope = {"memory_scope": "user", "resource_ids": [resource]}
    frozen = registry.validate(
        "source_review",
        "按资料整理",
        [{"mime_type": "text/markdown"}],
        "markdown",
        ctx=ctx,
        scope=scope,
        resource_reader=lambda uri: {"uri": uri, "owner_space": "user:alice"},
    )
    assert frozen["id"] == "source_review"
