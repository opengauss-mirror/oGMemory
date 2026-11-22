from core.models import RequestContext
from memorybot.templates import TemplateRegistry


def context():
    return RequestContext(
        "account",
        "alice",
        "writer",
        "session",
        "trace",
        visible_owner_spaces=("user_alice", "agent_writer", "agent_reviewer"),
    )


def declaration(requirements):
    return {
        "tools": ["search_memory", "read_memory"],
        "permission_requirements": [
            {"action": action, "scope": scope} for action, scope in requirements
        ],
    }


def test_builtin_user_ranges_remain_precise_in_authorized_session():
    template = TemplateRegistry().get("weekly_report")
    selected = {"memory_scope": "authorized", "resource_ids": []}
    effective = TemplateRegistry.execution_scope(template, context(), selected)
    assert effective["memory_owners"] == ["user:alice"]
    assert effective["resource_owners"] == ["user:alice"]
    assert effective["resource_uris"] == []
    assert effective["tool_names"] == ["read_memory", "search_memory"]
    assert selected == {"memory_scope": "authorized", "resource_ids": []}


def test_multiple_declared_owners_form_union_within_native_session_range():
    template = declaration(
        [
            ("memory.read", "user"),
            ("memory.read", "agent:writer"),
            ("resource.read", "agent:reviewer"),
            ("resource.read", "user"),
        ]
    )
    broad = {"memory_scope": "authorized", "resource_ids": []}
    scoped = TemplateRegistry.execution_scope(template, context(), broad)
    assert scoped["memory_owners"] == ["agent:writer", "user:alice"]
    assert scoped["resource_owners"] == ["agent:reviewer", "user:alice"]
    assert TemplateRegistry.authorize(template, context(), broad) == template
    personal = TemplateRegistry.execution_scope(template, context(), {"memory_scope": "user"})
    assert personal["memory_owners"] == ["user:alice"]
    assert personal["resource_owners"] == scoped["resource_owners"]
    assert TemplateRegistry.authorize(template, context(), {"memory_scope": "user"}) == template


def test_resource_uri_and_owner_ranges_combine_with_selected_resource():
    uri = "ctx://account/resources/accepted-notes"
    template = declaration(
        [
            ("memory.read", "authorized"),
            ("resource.read", "user"),
            ("resource.read", uri),
            ("tool.execute", "read_memory"),
        ]
    )
    scope = {"memory_scope": "authorized", "resource_ids": [uri]}
    effective = TemplateRegistry.execution_scope(template, context(), scope)
    assert effective["memory_owners"] == ["agent:reviewer", "agent:writer", "user:alice"]
    assert effective["resource_owners"] == ["user:alice"]
    assert effective["resource_uris"] == [uri]
    assert effective["tool_names"] == ["read_memory"]
    assert (
        TemplateRegistry.authorize(
            template, context(), scope, lambda value: {"uri": value, "owner_space": "agent:writer"}
        )
        == template
    )


def test_memory_only_template_keeps_session_resource_selection():
    template = declaration([("memory.read", "user")])
    uri = "ctx://account/resources/upload"
    scope = {"memory_scope": "authorized", "resource_ids": [uri]}
    effective = TemplateRegistry.execution_scope(template, context(), scope)
    assert effective["memory_owners"] == ["user:alice"]
    assert effective["resource_owners"] == []
    assert effective["resource_uris"] == []
    assert effective["resource_ids"] == [uri]
