from core.errors import NodeNotFoundError
from core.models import ContextNode
from fs.agfs_adapter.node_scanner import AGFSCanonicalNodeScanner


def test_agfs_scanner_streams_current_owner_memories_skills_and_resources():
    account = "acme"
    user_root = f"ctx://{account}/users/alice/memories"
    agents = f"ctx://{account}/agents/helper"
    skills_root = f"{agents}/skills"
    resource_root = f"ctx://{account}/resources"
    entities = f"{user_root}/entities"
    generated = f"{skills_root}/generated"
    children = {
        user_root: [entities],
        entities: [f"{entities}/capybara"],
        skills_root: [generated],
        generated: [f"{generated}/setup"],
        resource_root: [f"{resource_root}/manual"],
    }
    nodes = {
        user_root: ContextNode(user_root, "MEMORY", "memories", 0, "user:alice", "Summary", "Overview", "",
                               {"node_type": "directory"}),
        f"{entities}/capybara": ContextNode(f"{entities}/capybara", "MEMORY", "entity", 2, "user:alice", "", "", "Habitat"),
        f"{generated}/setup": ContextNode(f"{generated}/setup", "SKILL", "skill", 2, "agent:helper", "", "", "Steps"),
        f"{resource_root}/manual": ContextNode(f"{resource_root}/manual", "RESOURCE", "resource", 2, "resource:acme", "", "", "Manual"),
    }

    class FS:
        def list_children_including_inactive(self, uri, ctx):
            assert uri.startswith(f"ctx://{ctx.account_id}/")
            return children.get(uri, [])

        def read_node(self, uri, ctx):
            if uri not in nodes:
                raise NodeNotFoundError(uri)
            if uri.startswith(user_root):
                assert ctx.user_id == "alice"
            if uri.startswith(agents):
                assert ctx.agent_id == "helper"
            return nodes[uri]

    result = AGFSCanonicalNodeScanner(FS()).scan_active_nodes(
        account_id=account, owner_spaces=["user:alice", "agent:helper", "resource:acme"], page_size=1,
    )
    assert list(result) == list(nodes.values())
