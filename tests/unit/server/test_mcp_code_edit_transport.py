"""Real stdio transport and exact-file edits, with an isolated workspace registry."""
from contextlib import asynccontextmanager
from pathlib import Path
import os
import sys
import json

import anyio
import pytest


def _serve(root):
    from types import SimpleNamespace
    from core.models import RequestContext
    from core.resource_identity import compute_root_fingerprint
    from providers.unified_config import OgMemConfig
    from server.mcp_server import create_mcp_server
    from service.code_file_edit import CodeFileEditService
    from service.code_workspace_sync import ResolvedCodeWorkspace, WorkspaceResolutionError
    from retrieval.code_retrieval import ActiveCodeSearchRequest, CodeRetrievalService
    from tests.code_search_helpers import local_code_tokenizer

    workspace = ResolvedCodeWorkspace(
        root_id="test", root_slug="repo", root_fingerprint=compute_root_fingerprint(root),
        physical_root=root, max_files=10, language_hints={},
    )
    def resolve(**kwargs):
        if kwargs.get("root_slugs") not in ((), ("repo",)):
            raise WorkspaceResolutionError("workspace_not_registered")
        return workspace

    editor = CodeFileEditService(
        workspace_resolver=SimpleNamespace(resolve=resolve),
        code_retrieval_mode="candidate_local",
    )
    retrieval = CodeRetrievalService(
        workspace_resolver=SimpleNamespace(resolve=resolve),
        config=SimpleNamespace(code_retrieval_enabled=True, code_retrieval_mode="candidate_local",
                               code_retrieval_route_enabled=lambda route: route == "symbolic"),
    )
    class Service:
        _cfg = OgMemConfig(mcp_default_toolset="code", mcp_enable_code_edit_tools=True)
        def get_auth_service(self):
            return SimpleNamespace(role_control_active=lambda: False)
        def build_context(self, params, identity=None):
            return RequestContext("acct", "user", "agent", "session", "trace")
        def edit_code_file(self, params):
            return editor.edit(**{k: v for k, v in params.items() if k != "_ctx"}, ctx=params["_ctx"])
        def search_code_resources(self, params):
            request = ActiveCodeSearchRequest(params["query"], scope=params["scope"], budget=params["budget"])
            return retrieval.search_active(request, params["_ctx"]).to_dict()
    service = Service()
    with local_code_tokenizer():
        create_mcp_server(service=service, config=service._cfg).run(transport="stdio")


@asynccontextmanager
async def _stdio_session(root):
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    params = StdioServerParameters(
        command=sys.executable, args=[str(Path(__file__).resolve()), str(root)],
        env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[3])),
    )
    with (root / "server.log").open("w") as log:
        async with stdio_client(params, errlog=log) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                yield client


def test_stdio_edit_errors_and_corrected_retry(tmp_path):
    (tmp_path / "src").mkdir()
    target = tmp_path / "src/sample.py"
    source = "first = 1\nsecond = 1\n"
    target.write_text(source)
    async def exercise():
        async with _stdio_session(tmp_path) as client:
            attempts = [
                ({"locator": "repo/src/sample.py:SomeClass", "old_string": "first = 1"}, "target_file_not_found"),
                ({"locator": "missing/src/sample.py:1-2", "old_string": "first = 1"}, "workspace_not_registered"),
                ({"locator": "ogmemory://code/repo/src/sample.py", "old_string": "first = 1"}, "invalid_locator"),
                ({"locator": "repo/src/sample.py:1-2", "old_string": "missing"}, "old_string_not_found"),
                ({"locator": "repo/src/sample.py:1-2", "old_string": "= 1"}, "ambiguous_old_string"),
            ]
            for arguments, code in attempts:
                result = await client.call_tool("ogm_edit_file", {**arguments, "new_string": "first = 2"})
                assert result.is_error
                assert code in "\n".join(block.text for block in result.content if hasattr(block, "text"))
                assert target.read_text() == source
            result = await client.call_tool("ogm_edit_file", {"locator": "repo/src/sample.py:1-2", "old_string": "first = 1", "new_string": "first = 2"})
            assert not result.is_error
    anyio.run(exercise)
    assert target.read_text() == "first = 2\nsecond = 1\n"


@pytest.mark.parametrize("filename", ["module.py", "my module.py", "插件.py"])
def test_stdio_search_target_copies_directly_into_edit(tmp_path, filename):
    (tmp_path / "src").mkdir()
    target = tmp_path / "src" / filename
    source = "def edit_marker():\n    return 1\n"
    target.write_text(source, encoding="utf-8")
    decoy = tmp_path / "decoy.py"
    decoy.write_text(source, encoding="utf-8")
    async def exercise():
        async with _stdio_session(tmp_path) as client:
            result = await client.call_tool("ogm_search_code", {
                "query": "edit_marker", "scope": {"paths": ["src/" + filename]}})
            assert not result.is_error
            text = "\n".join(b.text for b in result.content if hasattr(b, "text"))
            line = next(line for line in text.splitlines() if line.startswith("edit_target: "))
            edit_target = json.loads(line.removeprefix("edit_target: "))
            assert edit_target == {"locator": "repo/src/" + filename}
            assert "# lines " in text and "symbol=edit_marker" in text
            result = await client.call_tool("ogm_edit_file", {
                **edit_target, "old_string": "return 1", "new_string": "return 2"})
            assert not result.is_error
    anyio.run(exercise)
    assert target.read_text(encoding="utf-8") == source.replace("return 1", "return 2")
    assert decoy.read_text(encoding="utf-8") == source


def test_edit_feedback_never_exposes_unknown_backend_details():
    from server.mcp_server import _code_edit_exception_message
    from service.code_workspace_sync import WorkspaceResolutionError

    for exc in [FileNotFoundError("/private/backend/secret"),
                RuntimeError("database password=secret"),
                ValueError("/private/backend/secret"),
                WorkspaceResolutionError("workspace_unavailable")]:
        assert _code_edit_exception_message(exc) is None


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    _serve(Path(sys.argv[1]).resolve())
