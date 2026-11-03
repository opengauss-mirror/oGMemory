"""Production MCP transport/handlers with deterministic in-memory service doubles.

For client interoperability checks only. Does NOT validate extraction, vectors,
SQL/AGFS persistence, model behavior, or production authorization policy.
Run: python -m tests.integration.agent_plugin_fixture --port 18991
"""
from __future__ import annotations

import argparse
from uuid import uuid4

import uvicorn

from core.models import ContextNode, RequestContext, Role, RetrievedBlock, SearchMemoryResult
from providers.unified_config import OgMemConfig
from server.auth import AuthenticationError, ResolvedIdentity
from server.mcp_server import create_mcp_http_app, create_mcp_server
from tests.unit.server.test_mcp_tools_readonly import _FakeService
from tests.unit.server.test_mcp_tools_write import _FakeService as WriteService


class FixtureAuth:
    def role_control_active(self):
        return True

    def resolve_identity(self, headers):
        headers = {k.lower(): v for k, v in headers.items()}
        if headers.get("x-api-key") != "agent-plugin-test-key":
            raise AuthenticationError("Invalid fixture key")
        return ResolvedIdentity(
            role=Role.MEMBER, account_id=headers["x-account-id"],
            user_id=headers["x-user-id"],
        )


class FixtureService(_FakeService, WriteService):
    def __init__(self):
        cfg = OgMemConfig(mcp_default_toolset="all")
        _FakeService.__init__(self, cfg)
        WriteService.__init__(self, cfg)
        self.values = {}

    def get_auth_service(self):
        return FixtureAuth()

    def build_context(self, params, identity=None):
        return RequestContext(
            account_id=identity.account_id, user_id=identity.user_id,
            agent_id=params.get("agentId", ""), session_id=params.get("sessionId") or "unknown",
            trace_id=str(uuid4()), role=identity.role,
        )

    def after_turn(self, params):
        ctx = params["_ctx"]
        self.values[(ctx.account_id, ctx.user_id)] = "\n".join(m["content"] for m in params["messages"])
        return {"ok": True, "status": "completed", "writes_completed": 1}

    def get_read_api(self):
        return self

    def search_memory(self, query, ctx, **kwargs):
        value = self.values.get((ctx.account_id, ctx.user_id))
        hits = [] if value is None else [RetrievedBlock(
            uri=f"ctx://{ctx.account_id}/users/{ctx.user_id}/memories/preferences/test",
            category="preference", abstract=value, score=1.0, has_content=True,
        )]
        return SearchMemoryResult(request_id=str(uuid4()), query=query, hits=hits)

    def _get_context_fs(self):
        return self

    def read_node(self, uri, ctx):
        expected = f"ctx://{ctx.account_id}/users/{ctx.user_id}/memories/preferences/test"
        if uri != expected or (ctx.account_id, ctx.user_id) not in self.values:
            raise FileNotFoundError(uri)
        value = self.values[(ctx.account_id, ctx.user_id)]
        return ContextNode(uri=uri, context_type="MEMORY", category="preference", level=0,
                           owner_space=f"user:{ctx.user_id}", abstract=value, overview=value,
                           content=value, metadata={})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18991)
    args = parser.parse_args()
    service = FixtureService()
    mcp = create_mcp_server(service=service, config=service._cfg, host="127.0.0.1", port=args.port)
    uvicorn.run(create_mcp_http_app(mcp, service), host="127.0.0.1", port=args.port, log_level="warning")
