from __future__ import annotations

import base64
import io
import zipfile
from types import MethodType

import pytest

from core.errors import ConcurrentModificationError
from core.models import RequestContext, Role
from providers.unified_config import OgMemConfig
from server.auth import ResolvedIdentity
from server.mcp_server import create_mcp_server
from server.mcp_tools import delete_skill, install_skill, update_skill
from server.memory_service import MemoryService


class _FakeAuth:
    def __init__(self, *, active=False):
        self.active = active

    def role_control_active(self):
        return self.active

    def require_role(self, ctx, *roles):
        assert ctx.role in roles


class _FakeService:
    def __init__(self):
        self._cfg = OgMemConfig()
        self.calls = []

    def get_auth_service(self):
        return _FakeAuth()

    def build_context(self, params, identity=None):
        return RequestContext(
            account_id="acct-1",
            user_id="root",
            agent_id=params.get("agentId") or "bot",
            session_id="skill-management",
            trace_id="skill-management",
            role=Role.ROOT,
        )

    def add_managed_skill(self, ctx, data=None, **kwargs):
        self.calls.append(("install", ctx.agent_id, data, kwargs))
        return {"status": "success", "action": "create"}

    def update_managed_skill(self, ctx, name, data=None, **kwargs):
        self.calls.append(("update", ctx.agent_id, name, data, kwargs))
        return {"status": "success", "action": "update"}

    def delete_published_skill(self, ctx, name, source):
        self.calls.append(("delete", ctx.agent_id, name, source))
        return {"action": "delete", "name": name, "source": source}


SKILL_MD = """---
name: review-code
description: Review code changes.
---

# Review code
"""


def test_install_update_and_delete_skill_use_selected_agent_scope():
    service = _FakeService()

    created = install_skill(
        service,
        archive_base64="UEsDBAoAAAAAAA==",
        agent_id="bot",
    )
    updated = update_skill(
        service,
        name="review-code",
        path="/srv/skills/review-code",
        agent_id="bot",
    )
    deleted = delete_skill(
        service,
        name="review-code",
        source="imported",
        agent_id="bot",
    )

    assert created["action"] == "create"
    assert updated["action"] == "update"
    assert deleted == {
        "action": "delete",
        "name": "review-code",
        "source": "imported",
    }
    assert service.calls == [
        ("install", "bot", None, {"archive_base64": "UEsDBAoAAAAAAA==", "path": None}),
        (
            "update",
            "bot",
            "review-code",
            None,
            {"archive_base64": None, "path": "/srv/skills/review-code"},
        ),
        ("delete", "bot", "review-code", "imported"),
    ]


class _MemoryFS:
    def __init__(self):
        self.nodes = {}

    def write_node(self, node, ctx):
        expected = node.metadata.get("_expected_version")
        if expected == 0 and node.uri in self.nodes:
            raise ConcurrentModificationError(node.uri, 0, 1)
        if isinstance(expected, int) and expected > 0:
            node.metadata["version"] = expected + 1
        else:
            node.metadata["version"] = 1
        self.nodes[node.uri] = node

    def read_node(self, uri, ctx):
        if uri not in self.nodes:
            raise FileNotFoundError(uri)
        return self.nodes[uri]

    def delete_node(self, uri, ctx):
        del self.nodes[uri]


def _real_service(fs, *, auth_active=False):
    cfg = OgMemConfig(mcp_default_toolset="all", agent_id="bot")
    service = MemoryService.__new__(MemoryService)
    service._cfg = cfg
    service._get_context_fs = MethodType(lambda self: fs, service)
    service.get_auth_service = MethodType(lambda self: _FakeAuth(active=auth_active), service)
    service.build_context = MethodType(
        lambda self, params, identity=None: RequestContext(
            account_id=identity.account_id if identity else "acct-1",
            user_id=identity.user_id if identity else "root",
            agent_id=params.get("agentId") or "bot",
            session_id="mcp-skill-management",
            trace_id="mcp-skill-management",
            role=identity.role if identity else Role.ROOT,
        ),
        service,
    )
    return service, cfg


@pytest.mark.parametrize("input_form", ["data", "archive_base64"])
def test_http_admin_installs_and_updates_uploaded_package(input_form):
    fs = _MemoryFS()
    service, _ = _real_service(fs, auth_active=True)
    identity = ResolvedIdentity(role=Role.ADMIN, account_id="tenant-a", user_id="admin-a")
    auth_kwargs = {
        "agent_id": "bot",
        "allow_stdio_env_auth": False,
        "resolved_identity": identity,
    }

    def package_input(body):
        files = {"SKILL.md": SKILL_MD, "references/checklist.txt": body}
        if input_form == "data":
            return {"data": {"files": files}}
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as package:
            for path, content in files.items():
                package.writestr(path, content)
        return {"archive_base64": base64.b64encode(archive.getvalue()).decode("ascii")}

    created = install_skill(service, **package_input("Before"), **auth_kwargs)
    updated = update_skill(
        service, name="review-code", **package_input("After"), **auth_kwargs
    )
    ctx = service.build_context({"agentId": "bot"}, identity=identity)
    attachment = service.read_managed_skill_file(ctx, "review-code", "references/checklist.txt")

    assert created["action"] == "create"
    assert updated["action"] == "update"
    assert updated["version"] == 2
    assert attachment["content"] == "After"
    assert all(node.uri.startswith("ctx://tenant-a/") for node in fs.nodes.values())


@pytest.mark.asyncio
async def test_mcp_client_installs_zip_updates_path_and_deletes_skill(tmp_path):
    from mcp import Client

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr(
            "review-code/SKILL.md",
            "---\nname: review-code\ndescription: Before.\n---\n\n# Before\n",
        )
        package.writestr("review-code/scripts/check.py", "print('before')\n")

    skill_dir = tmp_path / "review-code"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_bytes(
        b"---\nname: review-code\ndescription: After.\n---\n\n# After\n"
    )
    scripts = skill_dir / "scripts"
    scripts.mkdir()
    (scripts / "check.py").write_bytes(b"print('after')\n")

    fs = _MemoryFS()
    service, cfg = _real_service(fs)
    mcp = create_mcp_server(service=service, config=cfg)

    async with Client(mcp) as client:
        installed = await client.call_tool(
            "ogm_install_skill",
            {
                "archive_base64": base64.b64encode(archive.getvalue()).decode("ascii"),
                "agent_id": "bot",
            },
        )
        updated = await client.call_tool(
            "ogm_update_skill",
            {"name": "review-code", "path": str(skill_dir), "agent_id": "bot"},
        )
        deleted = await client.call_tool(
            "ogm_delete_skill",
            {"name": "review-code", "source": "imported", "agent_id": "bot"},
        )

    assert installed.structured_content["action"] == "create"
    assert updated.structured_content["action"] == "update"
    assert deleted.structured_content["action"] == "delete"
    assert fs.nodes == {}
