from types import SimpleNamespace

import pytest

from core.errors import NodeNotFoundError
from core.models import RequestContext
from memorybot.gateway import MemoryGateway
from memorybot.models import Scope
from memorybot.templates import TemplateRegistry


def test_template_materials_follow_memory_and_resource_scopes():
    ctx = SimpleNamespace(user_space_name=lambda: "user:reader")
    scope = {
        "memory_scope": "authorized",
        "memory_owners": ["user:reader"],
        "resource_owners": [],
        "resource_uris": ["ctx://a/agents/writer/resources/checklist"],
    }
    values = [
        {
            "uri": "ctx://a/users/reader/memories/entities/release",
            "owner_space": "user:reader",
            "context_type": "MEMORY",
        },
        {
            "uri": "ctx://a/agents/writer/resources/checklist/content.md",
            "owner_space": "agent:writer",
            "context_type": "RESOURCE",
        },
        {
            "uri": "ctx://a/agents/writer/memories/skills/release",
            "owner_space": "agent:writer",
            "context_type": "MEMORY",
        },
    ]
    selected = [row["uri"] for row in values if MemoryGateway.in_scope(row, ctx, scope)]
    assert selected == [values[0]["uri"], values[1]["uri"]]


def test_template_resource_owner_scope_accepts_selected_user_document():
    ctx = SimpleNamespace(user_space_name=lambda: "user:reader")
    scope = {"memory_scope": "user", "resource_owners": ["user:reader"], "resource_uris": []}
    value = {
        "uri": "ctx://a/users/reader/memories/documents/guide",
        "owner_space": "user:reader",
        "context_type": "RESOURCE",
    }
    assert MemoryGateway.in_scope(value, ctx, scope) is True


def test_public_scope_combines_user_memory_with_selected_shared_resource():
    ctx = SimpleNamespace(user_space_name=lambda: "user:reader")
    selected = "ctx://a/agents/writer/resources/checklist"
    scope = Scope(resource_ids=[selected]).model_dump()
    rows = [
        {
            "uri": selected + "/content.md",
            "owner_space": "agent:writer",
            "context_type": "RESOURCE",
        },
        {
            "uri": "ctx://a/users/reader/resources/other",
            "owner_space": "user:reader",
            "context_type": "RESOURCE",
        },
        {
            "uri": "ctx://a/users/reader/memories/profile",
            "owner_space": "user:reader",
            "context_type": "MEMORY",
        },
    ]
    included = [row["uri"] for row in rows if MemoryGateway.in_scope(row, ctx, scope)]
    assert included == [rows[0]["uri"], rows[2]["uri"]]


def test_gateway_templates_lists_builtin_templates():
    gateway = object.__new__(MemoryGateway)
    gateway.service = SimpleNamespace(list_published_skills=lambda *args, **kwargs: {"skills": []})
    registry = gateway.templates(SimpleNamespace())
    assert registry.get("weekly_report")["output"]["formats"] == ["markdown"]


def test_template_authorization_allows_selected_shared_resource_with_user_memory():
    uri = "ctx://a/agents/writer/resources/checklist"
    ctx = SimpleNamespace(
        account_id="a",
        user_space_name=lambda: "user:reader",
        visible_owner_spaces=("user:reader", "agent:writer"),
    )
    scope = Scope(resource_ids=[uri]).model_dump()
    template = {
        "permission_requirements": [
            {"action": "memory.read", "scope": "user"},
            {"action": "resource.read", "scope": uri},
        ],
    }
    assert (
        TemplateRegistry.authorize(template, ctx, scope, lambda _: {"owner_space": "agent:writer"})
        == template
    )


@pytest.mark.parametrize(
    ("action", "expected_type"), [("memory.read", "MEMORY"), ("resource.read", "RESOURCE")]
)
def test_template_lookup_returns_only_its_declared_source_type(action, expected_type):
    ctx = RequestContext("a", "reader", "writer", "session", "trace")
    memory = "ctx://a/users/reader/memories/profile"
    resource = "ctx://a/users/reader/memories/documents/guide"
    scope = Scope(resource_ids=[resource]).model_dump()
    template = {"permission_requirements": [{"action": action, "scope": "user"}]}
    effective = TemplateRegistry.execution_scope(template, ctx, scope)
    sources = [
        {"uri": memory, "owner_space": "user:reader", "context_type": "MEMORY"},
        {"uri": resource, "owner_space": "user:reader", "context_type": "RESOURCE"},
    ]
    assert [
        row["context_type"] for row in sources if MemoryGateway.in_scope(row, ctx, effective)
    ] == [expected_type]


@pytest.mark.parametrize("fields", [("",), ("/name", "/location"), ("", "/name")])
def test_empty_query_lists_existing_profile_nodes(fields):
    ctx = RequestContext("a", "reader", "writer", "session", "trace")
    profile = "ctx://a/users/reader/memories/profile"
    stored = [profile + field for field in fields]
    fs = SimpleNamespace(
        count_memory_children=lambda _: {profile: len(stored)},
        list_children=lambda parent, _: [uri for uri in stored if uri.startswith(parent + "/")],
    )
    gateway = object.__new__(MemoryGateway)
    gateway.service = SimpleNamespace(_get_context_fs=lambda: fs)

    def read(_, uri):
        if uri not in stored:
            raise NodeNotFoundError(uri)
        return {"uri": uri, "owner_space": "user:reader", "context_type": "MEMORY"}

    gateway.read_memory = read
    assert [row["uri"] for row in gateway.list_memories(ctx)] == stored
