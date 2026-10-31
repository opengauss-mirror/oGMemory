from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import anyio
import pytest
from pydantic import ValidationError as PydanticValidationError
from starlette.testclient import TestClient

from core.models import ContextNode, RequestContext, Role
from providers.unified_config import OgMemConfig
from server.auth import ResolvedIdentity
from server.mcp_security import CODE_EVIDENCE_UNTRUSTED_NOTICE
from server.mcp_server import (
    DELETE_SKILL_TOOL_NAME,
    EDIT_FILE_TOOL_NAME,
    INSTALL_SKILL_TOOL_NAME,
    LIST_MEMORY_TOOL_NAME,
    LIST_SKILL_CANDIDATES_TOOL_NAME,
    LIST_SKILLS_TOOL_NAME,
    READ_SKILL_TOOL_NAME,
    SEARCH_CODE_TOOL_NAME,
    SEARCH_SKILLS_TOOL_NAME,
    UPDATE_SKILL_TOOL_NAME,
    _auth_kwargs_from_context,
    _MCPHttpTransportGuard,
    _MCPRuntimeSettings,
    create_mcp_http_app,
    create_mcp_server,
    create_mcp_sse_app,
    run_mcp_server,
)

_SKILL_TOOLS = {
    SEARCH_SKILLS_TOOL_NAME,
    LIST_SKILLS_TOOL_NAME,
    READ_SKILL_TOOL_NAME,
    LIST_SKILL_CANDIDATES_TOOL_NAME,
}
_SKILL_WRITE_TOOLS = {INSTALL_SKILL_TOOL_NAME, UPDATE_SKILL_TOOL_NAME}
_SKILL_DELETE_TOOLS = {DELETE_SKILL_TOOL_NAME}
_READ_TOOLS = {
    "ogm_query_memory",
    LIST_MEMORY_TOOL_NAME,
    SEARCH_CODE_TOOL_NAME,
    "ogm_read_memory",
} | _SKILL_TOOLS


def test_read_memory_tool_forwards_selected_unit(monkeypatch):
    calls = []

    def read_selected_unit(service, **kwargs):
        calls.append(kwargs)
        return {"uri": kwargs["uri"], "unit_key": kwargs["unit_key"], "content": "Use batches of 20."}

    monkeypatch.setattr("server.mcp_server.read_memory_tool", read_selected_unit)
    mcp = create_mcp_server(service=_FakeService(), config=OgMemConfig())
    anyio.run(mcp.call_tool, "ogm_read_memory", {
        "uri": "ogmemory://memories/tool_memory/search", "unit_key": "batch", "level": "content",
    })
    assert [(call["uri"], call["unit_key"], call["level"]) for call in calls] == [
        ("ogmemory://memories/tool_memory/search", "batch", "content")]


class _FakeRequest:
    def __init__(self, headers):
        self.headers = headers


class _FakeRequestContext:
    def __init__(self, request):
        self.request = request


class _FakeMCPContext:
    def __init__(self, request):
        self.request_context = _FakeRequestContext(request)


class _FakeService:
    _cfg = OgMemConfig()

    def get_auth_service(self):
        return _FakeAuth()


class _RecordingCodeSearchService(_FakeService):
    def __init__(self):
        self.auth = _FakeAuth(role=Role.MEMBER)
        self.build_context_params = None
        self.build_context_identity = None
        self.code_search_params = None

    def get_auth_service(self):
        return self.auth

    def build_context(self, params, identity=None):
        self.build_context_params = dict(params)
        self.build_context_identity = identity
        return RequestContext(
            account_id=identity.account_id,
            user_id=identity.user_id,
            agent_id=str(params.get("agentId") or ""),
            session_id=str(params.get("sessionId") or "unknown"),
            trace_id="trace-code-search",
        )

    def search_code_resources(self, params):
        self.code_search_params = dict(params)
        return {"status": "zero_result", "evidence": []}


class _FakeNoAuth:
    def role_control_active(self):
        return False


class _FakePatternFS:
    def read_node(self, uri, ctx):
        if uri != "ctx://acct-1/agents/bot/memories/patterns/review":
            raise FileNotFoundError(uri)
        return ContextNode(
            uri=uri,
            context_type="MEMORY",
            category="pattern",
            level=0,
            owner_space="agent:bot",
            abstract="Legacy agent review pattern.",
            overview="Review pattern overview.",
            content="Use the legacy agent review pattern.",
        )


class _FakePatternService:
    _cfg = OgMemConfig()

    def __init__(self):
        self.fs = _FakePatternFS()

    def get_auth_service(self):
        return _FakeNoAuth()

    def build_context(self, params, identity=None):
        return RequestContext(
            account_id="acct-1",
            user_id="alice",
            agent_id="bot",
            session_id=params.get("sessionId") or "s1",
            trace_id="t1",
        )

    def _get_context_fs(self):
        return self.fs


class _FakeAuth:
    def __init__(self, role=Role.MEMBER):
        self.headers = None
        self.role = role

    def role_control_active(self):
        return True

    def resolve_identity(self, headers):
        self.headers = headers
        return ResolvedIdentity(role=self.role, account_id="acct-auth", user_id="user-auth")


class _KeyedFakeAuth(_FakeAuth):
    def resolve_identity(self, headers):
        normalized = {str(key).lower(): value for key, value in headers.items()}
        api_key = normalized.get("x-api-key")
        if not api_key:
            raise RuntimeError("missing test key")
        return ResolvedIdentity(role=Role.MEMBER, account_id="acct-auth", user_id=api_key)


class _FakeDisposeService:
    _cfg = OgMemConfig()

    def __init__(self):
        self.auth = _FakeAuth()
        self.build_context_params = None
        self.build_context_identity = None
        self.dispose_params = None
        self.idle_params = None

    def get_auth_service(self):
        return self.auth

    def build_context(self, params, identity=None):
        self.build_context_params = dict(params)
        self.build_context_identity = identity
        return "ctx-from-same-service"

    def dispose(self, params):
        self.dispose_params = dict(params)
        return {"disposed": True, "ctx": params["_ctx"]}

    def wait_until_idle(self, params):
        self.idle_params = dict(params)
        return {"idle": True, "ctx": params["_ctx"]}


class _FakeThreadCheckingDisposeService(_FakeDisposeService):
    def wait_until_idle(self, params):
        self.idle_params = dict(params)
        try:
            asyncio.get_running_loop()
            running_loop = True
        except RuntimeError:
            running_loop = False
        return {"idle": True, "running_loop": running_loop, "ctx": params["_ctx"]}


def test_create_mcp_server_excludes_context_from_tool_schema():
    mcp = create_mcp_server(service=_FakeService(), config=OgMemConfig())

    tools = anyio.run(mcp.list_tools)
    query_tool = next(tool for tool in tools if tool.name == "ogm_query_memory")
    search_tool = next(tool for tool in tools if tool.name == SEARCH_CODE_TOOL_NAME)
    read_tool = next(tool for tool in tools if tool.name == "ogm_read_memory")
    list_tool = next(tool for tool in tools if tool.name == LIST_MEMORY_TOOL_NAME)

    assert "ctx" not in query_tool.input_schema.get("properties", {})
    assert "ctx" not in read_tool.input_schema.get("properties", {})
    assert "ctx" not in list_tool.input_schema.get("properties", {})
    assert {"code_search", "code_scope", "code_budget"}.isdisjoint(
        query_tool.input_schema.get("properties", {})
    )
    search_schema = search_tool.input_schema
    search_properties = search_schema.get("properties", {})
    assert set(search_properties) == {"query", "scope", "budget"}
    assert search_schema.get("required") == ["query"]
    assert search_properties["query"] == {
        "description": "Natural language or keyword query.",
        "title": "Query",
        "type": "string",
    }
    assert search_properties["scope"]["description"] == "Optional search scope."
    assert search_properties["budget"]["description"] == "Optional search budget."

    scope_schema = search_schema["$defs"]["CodeSearchScopeInput"]
    assert scope_schema["additionalProperties"] is False
    assert set(scope_schema["properties"]) == {"paths", "glob_patterns", "grep_terms"}
    path_description = scope_schema["properties"]["paths"]["description"]
    assert "repository-relative POSIX" in path_description
    assert "without the edit locator's root slug" in path_description
    assert "paths=['src/module.py']" in path_description
    assert scope_schema["properties"]["glob_patterns"]["description"] == (
        "Optional glob patterns to narrow candidate files before ranking."
    )
    grep_description = scope_schema["properties"]["grep_terms"]["description"]
    assert "case-insensitive literal substrings (AND)" in grep_description
    assert "same file's complete source content" in grep_description
    assert "Paths, metadata and generated summaries do not count" in grep_description
    for field in ("paths", "glob_patterns", "grep_terms"):
        array_schema = scope_schema["properties"][field]["anyOf"][0]
        assert array_schema == {"items": {"type": "string"}, "type": "array"}

    budget_schema = search_schema["$defs"]["CodeSearchBudgetInput"]
    assert budget_schema["additionalProperties"] is False
    assert set(budget_schema["properties"]) == {"result_limit"}
    assert budget_schema["properties"]["result_limit"]["description"] == (
        "Optional max hits; omit to use the server default."
    )
    assert budget_schema["properties"]["result_limit"]["anyOf"][0] == {
        "exclusiveMinimum": 0,
        "type": "integer",
    }
    for guidance in (
        "compact editable snippets",
        "symbols in query",
        "scope.paths for known files",
        "edit_target.locator",
        "complete implementations",
        "read_lines",
    ):
        assert guidance in search_tool.description
    assert "recover_original" in read_tool.input_schema.get("properties", {})


def test_search_code_tool_converts_narrow_inputs_to_existing_adapter_mappings():
    service = _RecordingCodeSearchService()
    mcp = create_mcp_server(service=service, config=OgMemConfig())
    tool = mcp._tool_manager.get_tool(SEARCH_CODE_TOOL_NAME)

    arguments = tool.fn_metadata.validate_arguments(
        {
            "query": "find service",
            "scope": {
                "paths": ["src"],
                "glob_patterns": ["**/*.py"],
                "grep_terms": ["Service"],
            },
            "budget": {"result_limit": 3},
        },
    )
    result = tool.fn(**arguments)

    assert result == (
        "# code_search status: zero_result\n"
        "# query: find service\n"
        "# hit_count: 0\n"
        f"# {CODE_EVIDENCE_UNTRUSTED_NOTICE}\n"
    )
    assert service.build_context_params == {}
    assert service.code_search_params["scope"] == {
        "paths": ["src"],
        "glob_patterns": ["**/*.py"],
        "grep_terms": ["Service"],
    }
    assert service.code_search_params["budget"]["result_limit"] == 3
    assert service.code_search_params["_ctx"].account_id == "acct-auth"
    assert service.code_search_params["_ctx"].agent_id == ""


@pytest.mark.parametrize(
    "arguments",
    [
        {"query": "find service", "scope": {"root_slugs": ["repo"]}},
        {"query": "find service", "budget": {"candidate_limit": 3}},
        {"query": "find service", "scope": {"paths": "src"}},
        {"query": "find service", "budget": {"result_limit": True}},
        {"query": "find service", "budget": {"result_limit": 0}},
    ],
)
def test_search_code_tool_rejects_invalid_nested_input_before_delegation(arguments):
    service = _RecordingCodeSearchService()
    mcp = create_mcp_server(service=service, config=OgMemConfig())
    tool = mcp._tool_manager.get_tool(SEARCH_CODE_TOOL_NAME)

    with pytest.raises(PydanticValidationError):
        tool.fn_metadata.validate_arguments(arguments)

    assert service.build_context_params is None
    assert service.code_search_params is None


def test_read_memory_tool_rejects_boolean_recovery_budget_before_delegation():
    service = _FakeService()
    service.read_tool_output_original = Mock()
    mcp = create_mcp_server(service=service, config=OgMemConfig())
    tool = mcp._tool_manager.get_tool("ogm_read_memory")

    with pytest.raises(PydanticValidationError):
        tool.fn_metadata.validate_arguments(
            {
                "uri": (
                    "ctx://acct/agents/bot/memories/tool_outputs/"
                    "tool_output_read_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                ),
                "recover_original": True,
                "max_chars": True,
            }
        )

    service.read_tool_output_original.assert_not_called()


def test_create_mcp_server_exposes_recovery_parameter_and_instruction():
    mcp = create_mcp_server(service=_FakeService(), config=OgMemConfig())
    tools = {tool.name: tool for tool in anyio.run(mcp.list_tools)}
    read_tool = tools["ogm_read_memory"]

    assert "recover_original" in read_tool.input_schema.get("properties", {})
    assert "original tool output" in read_tool.description.lower()
    assert "recover_original=true" in mcp.instructions


def test_create_mcp_server_defaults_to_read_only_tools():
    mcp = create_mcp_server(service=_FakeService(), config=OgMemConfig())

    tools = anyio.run(mcp.list_tools)

    tool_names = {tool.name for tool in tools}

    assert tool_names == _READ_TOOLS
    assert all("." not in name for name in tool_names)


def test_create_mcp_server_read_write_toolset_registers_write_only():
    mcp = create_mcp_server(
        service=_FakeService(),
        config=OgMemConfig(mcp_default_toolset="read-write"),
    )

    tools = anyio.run(mcp.list_tools)

    assert {tool.name for tool in tools} == _READ_TOOLS | {
        "ogm_write_memory",
    } | _SKILL_WRITE_TOOLS


def test_create_mcp_server_all_toolset_registers_write_and_delete():
    mcp = create_mcp_server(
        service=_FakeService(),
        config=OgMemConfig(mcp_default_toolset="all"),
    )

    tools = anyio.run(mcp.list_tools)

    assert {tool.name for tool in tools} == _READ_TOOLS | {
        "ogm_write_memory",
        "ogm_delete_memory",
    } | _SKILL_WRITE_TOOLS | _SKILL_DELETE_TOOLS


def test_create_mcp_server_instructions_describe_generic_memory_timing():
    mcp = create_mcp_server(
        service=_FakeService(),
        config=OgMemConfig(mcp_default_toolset="all"),
    )

    tools = {tool.name: tool for tool in anyio.run(mcp.list_tools)}
    instructions = mcp.instructions.lower()

    assert "persistent memory tools" in instructions
    assert "before answering" in instructions
    assert "not present in the current prompt" in instructions
    assert "information that may be needed later" in instructions
    assert "durable context" in instructions
    assert "interaction text or messages" in instructions
    assert "decide what to persist" in instructions
    assert "ogm_query_memory" in instructions
    assert "ogm_read_memory" in instructions
    assert "ogm_list_memories" in instructions
    assert SEARCH_CODE_TOOL_NAME in instructions
    assert "recover_original=true" in instructions
    assert "ogm_list_skills" in instructions
    assert "ogm_search_skills" in instructions
    assert "ogm_read_skill" in instructions
    assert INSTALL_SKILL_TOOL_NAME in instructions
    assert UPDATE_SKILL_TOOL_NAME in instructions
    assert DELETE_SKILL_TOOL_NAME in instructions
    assert "ogm_write_memory" in instructions
    assert "ogm_delete_memory" in instructions
    assert "ogm." not in instructions
    assert "people" not in instructions
    assert "preferences" not in instructions
    assert "relationships" not in instructions
    assert "locomo" not in instructions
    assert "benchmark" not in instructions
    assert "chat transcript" not in instructions
    query_tool = tools["ogm_query_memory"]
    list_tool = tools["ogm_list_memories"]
    read_tool = tools["ogm_read_memory"]
    write_tool = tools["ogm_write_memory"]
    delete_tool = tools["ogm_delete_memory"]

    assert "semantically recall" in query_tool.description.lower()
    assert "generic memory" in query_tool.description.lower()
    assert "relevance score" in query_tool.description.lower()
    assert "browse or inventory" in list_tool.description.lower()
    assert "native summary metadata" in list_tool.description.lower()
    assert "pagination" in list_tool.description.lower()
    assert "exact ogmemory:// uri" in read_tool.description.lower()
    assert "original tool output" in read_tool.description.lower()
    assert "deduplication" in write_tool.description.lower()
    assert "correction" in write_tool.description.lower()
    assert "explicit user forget request" in delete_tool.description.lower()
    assert "compact editable snippets" in tools[SEARCH_CODE_TOOL_NAME].description.lower()

    query_schema = query_tool.input_schema["properties"]
    assert "self-contained" in query_schema["query"]["description"]
    assert "valid range is 1-20" in query_schema["top_k"]["description"]
    assert "memory-type filter" in query_schema["types"]["description"]

    list_schema = list_tool.input_schema["properties"]
    assert "valid range is 1-500" in list_schema["limit"]["description"]
    assert "next_cursor" in list_schema["cursor"]["description"]

    read_schema = read_tool.input_schema["properties"]
    assert "returned by another memory call" in read_schema["uri"]["description"]
    assert "narrowest level" in read_schema["level"]["description"]

    write_schema = write_tool.input_schema["properties"]
    assert "mutually exclusive" in write_schema["text"]["description"]
    assert "verify the returned status" in write_schema["wait"]["description"]

    delete_schema = delete_tool.input_schema["properties"]
    assert "confirmed for deletion" in delete_schema["uri"]["description"]
    forbidden = {"search_code", "get_original_tool_output", "index_codebase", "read_filtered"}
    assert not forbidden & set(tools)


def test_create_mcp_server_read_only_instructions_do_not_advertise_disabled_tools():
    mcp = create_mcp_server(service=_FakeService(), config=OgMemConfig())

    instructions = mcp.instructions.lower()

    assert "ogm_query_memory" in instructions
    assert "ogm_read_memory" in instructions
    assert "ogm_list_memories" in instructions
    assert "ogm_write_memory" not in instructions
    assert "ogm_delete_memory" not in instructions
    assert EDIT_FILE_TOOL_NAME not in instructions


def test_run_mcp_server_rejects_disabled_config(monkeypatch):
    monkeypatch.setattr(
        "server.mcp_server.get_config",
        lambda config_path=None: OgMemConfig(mcp_enabled=False),
    )

    with pytest.raises(RuntimeError, match="MCP server is disabled"):
        run_mcp_server()


def test_run_mcp_server_resets_cached_config_for_explicit_path(monkeypatch):
    calls = []

    def fake_reset_config():
        calls.append("reset")

    def fake_get_config(config_path=None):
        calls.append(("get_config", config_path))
        return OgMemConfig(mcp_enabled=False)

    monkeypatch.setattr("server.mcp_server.reset_config", fake_reset_config)
    monkeypatch.setattr("server.mcp_server.get_config", fake_get_config)

    with pytest.raises(RuntimeError, match="MCP server is disabled"):
        run_mcp_server(config_path="custom-ogmem.yaml")

    assert calls == ["reset", ("get_config", "custom-ogmem.yaml")]


def test_run_mcp_server_accepts_sse_transport(monkeypatch):
    calls = []
    service = _FakeService()
    service.start_extraction_recovery = lambda: calls.append("recovery_started")

    class _FakeMcp:
        pass

    async def _fake_run_sse_app(mcp, service):
        calls.append(("run", "sse"))

    monkeypatch.setattr(
        "server.mcp_server.get_config",
        lambda config_path=None: OgMemConfig(mcp_enabled=True),
    )
    monkeypatch.setattr("server.mcp_server.MemoryService", lambda config: service)
    monkeypatch.setattr("server.mcp_server.create_mcp_server", lambda **kwargs: _FakeMcp())
    monkeypatch.setattr("server.mcp_server._run_sse_app", _fake_run_sse_app)

    run_mcp_server(transport="sse")

    assert calls == ["recovery_started", ("run", "sse")]


def test_run_mcp_server_rejects_sse_when_http_auth_is_disabled(monkeypatch):
    class _DisabledAuthService(_FakeService):
        def get_auth_service(self):
            return _FakeNoAuth()

    monkeypatch.setattr(
        "server.mcp_server.get_config",
        lambda config_path=None: OgMemConfig(mcp_enabled=True),
    )
    monkeypatch.setattr("server.mcp_server.MemoryService", lambda config: _DisabledAuthService())
    monkeypatch.setattr(
        "server.mcp_server.create_mcp_server",
        lambda **kwargs: pytest.fail("SSE server must fail before MCPServer is created"),
    )

    with pytest.raises(RuntimeError, match="SSE transport requires HTTP authentication"):
        run_mcp_server(transport="sse")


def test_run_mcp_server_rejects_unknown_transport():
    with pytest.raises(ValueError, match="transport must be stdio, sse, or streamable-http"):
        run_mcp_server(transport="websocket")


def test_create_mcp_server_static_resources_are_readable():
    mcp = create_mcp_server(service=_FakeService(), config=OgMemConfig())

    resources = anyio.run(mcp.list_resources)
    payloads = anyio.run(mcp.read_resource, "ogmemory://status")

    resource_uris = {str(resource.uri) for resource in resources}
    assert {"ogmemory://status", "ogmemory://types", "ogmemory://skills"} <= resource_uris
    assert '"status": "ok"' in payloads[0].content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "protocol_version",
    ["2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"],
)
async def test_create_mcp_server_serves_each_legacy_protocol(monkeypatch, protocol_version):
    from mcp import Client
    from mcp.client import session as client_session

    monkeypatch.setattr(client_session, "LATEST_HANDSHAKE_VERSION", protocol_version)
    monkeypatch.setattr(
        "server.mcp_server.query_memory_tool",
        lambda service, **kwargs: {"hits": []},
    )
    mcp = create_mcp_server(service=_FakeService(), config=OgMemConfig())

    async with Client(mcp, mode="legacy") as client:
        tools = await client.list_tools()
        result = await client.call_tool("ogm_query_memory", {"query": "compatibility"})

        assert client.protocol_version == protocol_version

    assert {tool.name for tool in tools.tools} == _READ_TOOLS
    assert result.is_error is False


@pytest.mark.asyncio
async def test_create_mcp_server_serves_current_protocol(monkeypatch):
    from mcp import Client

    monkeypatch.setattr(
        "server.mcp_server.query_memory_tool",
        lambda service, **kwargs: {"hits": []},
    )
    mcp = create_mcp_server(service=_FakeService(), config=OgMemConfig())

    async with Client(mcp) as client:
        tools = await client.list_tools()
        result = await client.call_tool("ogm_query_memory", {"query": "compatibility"})
        status = await client.read_resource("ogmemory://status")

        assert client.protocol_version == "2026-07-28"

    assert {tool.name for tool in tools.tools} == _READ_TOOLS
    assert result.is_error is False
    assert '"status": "ok"' in status.contents[0].text
def test_create_mcp_server_reads_scoped_agent_pattern_resource():
    mcp = create_mcp_server(service=_FakePatternService(), config=OgMemConfig())

    templates = anyio.run(mcp.list_resource_templates)
    payloads = anyio.run(
        mcp.read_resource,
        "ogmemory://memories/pattern/agent/review",
    )

    assert "ogmemory://memories/pattern/agent/{memory_id}" in {
        str(template.uri_template) for template in templates
    }
    assert '"uri": "ogmemory://memories/pattern/agent/review"' in payloads[0].content
    assert "Use the legacy agent review pattern." in payloads[0].content


def test_create_mcp_server_keeps_legacy_generated_skill_resource(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "server.mcp_server.read_skill_tool",
        lambda service, **kwargs: calls.append(kwargs) or {"source": kwargs["source"]},
    )
    mcp = create_mcp_server(service=_FakeService(), config=OgMemConfig())

    templates = anyio.run(mcp.list_resource_templates)
    template_uris = {str(template.uri_template) for template in templates}
    payloads = anyio.run(mcp.read_resource, "ogmemory://skills/review-code")

    assert "ogmemory://skills/{source}/{name}" in template_uris
    assert "ogmemory://skills/{name}" in template_uris
    assert calls[0]["name"] == "review-code"
    assert calls[0]["source"] == "generated"
    assert '"source": "generated"' in payloads[0].content


def test_create_mcp_server_exposes_imported_skill_files(monkeypatch):
    file_calls = []
    monkeypatch.setattr(
        "server.mcp_server.build_mcp_context",
        lambda service, params, **kwargs: "skill-file-context",
    )
    monkeypatch.setattr(
        "server.mcp_server.render_resource",
        lambda service, uri, ctx: file_calls.append((uri, ctx)) or {"uri": uri},
    )
    mcp = create_mcp_server(service=_FakeService(), config=OgMemConfig())

    tools = anyio.run(mcp.list_tools)
    read_skill_tool = next(tool for tool in tools if tool.name == READ_SKILL_TOOL_NAME)
    templates = anyio.run(mcp.list_resource_templates)
    payloads = anyio.run(
        mcp.read_resource,
        "ogmemory://skills/imported/review-code/scripts/run.py",
    )

    assert "include_files" in read_skill_tool.input_schema.get("properties", {})
    assert (
        read_skill_tool.input_schema["properties"]["include_files"]["default"]
        is False
    )
    assert "ogmemory://skills/imported/{name}/{+path}" in {
        str(template.uri_template) for template in templates
    }
    assert file_calls[0] == (
        "ogmemory://skills/imported/review-code/scripts/run.py",
        "skill-file-context",
    )
    assert "ogm_read_skill_file" not in {tool.name for tool in tools}
    assert "scripts/run.py" in payloads[0].content


def test_create_mcp_server_registers_write_tool_when_enabled():
    mcp = create_mcp_server(
        service=_FakeService(),
        config=OgMemConfig(mcp_enable_write_tools=True),
    )

    tools = anyio.run(mcp.list_tools)
    write_tool = next(tool for tool in tools if tool.name == "ogm_write_memory")

    assert {tool.name for tool in tools} == _READ_TOOLS | {
        "ogm_write_memory",
    } | _SKILL_WRITE_TOOLS
    assert "ctx" not in write_tool.input_schema.get("properties", {})
    assert "session_id" in write_tool.input_schema.get("properties", {})
    force_extract_schema = write_tool.input_schema["properties"]["force_extract"]
    assert "explicitly asks to remember" in force_extract_schema["description"]
    assert "记住" in force_extract_schema["description"]


def test_create_mcp_server_registers_delete_tool_when_enabled():
    mcp = create_mcp_server(
        service=_FakeService(),
        config=OgMemConfig(mcp_enable_delete_tools=True),
    )

    tools = anyio.run(mcp.list_tools)
    delete_tool = next(tool for tool in tools if tool.name == "ogm_delete_memory")

    assert {tool.name for tool in tools} == _READ_TOOLS | {
        "ogm_delete_memory",
    } | _SKILL_DELETE_TOOLS
    assert "ctx" not in delete_tool.input_schema.get("properties", {})
    assert "uri" in delete_tool.input_schema.get("properties", {})


def test_create_mcp_server_registers_read_write_and_delete_tools_when_enabled():
    mcp = create_mcp_server(
        service=_FakeService(),
        config=OgMemConfig(mcp_enable_write_tools=True, mcp_enable_delete_tools=True),
    )

    tools = anyio.run(mcp.list_tools)

    assert {tool.name for tool in tools} == _READ_TOOLS | {
        "ogm_write_memory",
        "ogm_delete_memory",
    } | _SKILL_WRITE_TOOLS | _SKILL_DELETE_TOOLS


def test_create_mcp_server_exposes_skill_management_schemas():
    mcp = create_mcp_server(
        service=_FakeService(),
        config=OgMemConfig(mcp_default_toolset="all"),
    )

    tools = {tool.name: tool for tool in anyio.run(mcp.list_tools)}

    assert set(tools[INSTALL_SKILL_TOOL_NAME].input_schema["properties"]) == {
        "data", "archive_base64", "path", "agent_id",
    }
    assert set(tools[UPDATE_SKILL_TOOL_NAME].input_schema["properties"]) == {
        "name", "data", "archive_base64", "path", "agent_id",
    }
    assert set(tools[DELETE_SKILL_TOOL_NAME].input_schema["properties"]) == {
        "name", "source", "agent_id",
    }
    assert "imported" in tools[INSTALL_SKILL_TOOL_NAME].description
    assert "explicit user request" in tools[DELETE_SKILL_TOOL_NAME].description


def test_create_mcp_server_registers_code_edit_tool_only_under_independent_capability():
    mcp = create_mcp_server(
        service=_FakeService(),
        config=OgMemConfig(
            mcp_enable_code_edit_tools=True,
            mcp_enable_write_tools=False,
            mcp_enable_delete_tools=False,
        ),
    )

    tools = anyio.run(mcp.list_tools)
    edit_tool = next(tool for tool in tools if tool.name == EDIT_FILE_TOOL_NAME)
    properties = edit_tool.input_schema.get("properties", {})

    assert {tool.name for tool in tools} == _READ_TOOLS | {EDIT_FILE_TOOL_NAME}
    assert "ctx" not in properties
    assert set(properties) == {"locator", "old_string", "new_string", "replace_all"}
    assert "edit_target.locator" in properties["locator"]["description"]
    assert "not the JSON object" in properties["locator"]["description"]
    old_string_description = properties["old_string"]["description"].lower()
    assert "smallest unique exact text" in old_string_description
    assert "whole file, class, or complete search excerpt" in old_string_description
    assert "replacement for old_string only" in properties["new_string"]["description"].lower()
    assert "smallest unique old_string" in edit_tool.description.lower()
    for guidance in (
        "Reuse the locator",
        "preserve indentation and line breaks",
        "combined diff",
        "error feedback",
    ):
        assert guidance in edit_tool.description
    assert SEARCH_CODE_TOOL_NAME in mcp.instructions
    assert EDIT_FILE_TOOL_NAME in mcp.instructions
    assert "without searching again" in mcp.instructions


def test_code_edit_tool_is_not_implied_by_generic_write_or_code_only_profile(monkeypatch):
    monkeypatch.setenv("OGMEM_CODE_MEMORY_ONLY", "true")
    mcp = create_mcp_server(
        service=_FakeService(),
        config=OgMemConfig(mcp_enable_write_tools=True, mcp_enable_code_edit_tools=False),
    )

    assert {tool.name for tool in anyio.run(mcp.list_tools)} == _READ_TOOLS | {
        "ogm_write_memory"
    } | _SKILL_WRITE_TOOLS


@pytest.mark.parametrize("field,code", [
    ("scope.source_snapshot", "source_snapshot_required"),
    ("scope.source_snapshot_conflict", "source_snapshot_conflict"),
])
def test_mcp_search_requires_source_snapshot_with_actionable_private_feedback(monkeypatch, field, code):
    from core.errors import ValidationError
    from mcp.server.mcpserver.exceptions import ToolError

    def search(service, **kwargs):
        raise ValidationError(field, "/private/source.py")
    monkeypatch.setattr("server.mcp_server.search_code_tool", search)
    mcp = create_mcp_server(service=_FakeService(), config=OgMemConfig())
    with pytest.raises(ToolError, match=code) as error:
        anyio.run(mcp.call_tool, SEARCH_CODE_TOOL_NAME, {"query": "sample.py"})
    assert "refresh the affected code files individually" in str(error.value)
    assert "bootstrap is for initial import" in str(error.value)
    assert "/private/source.py" not in str(error.value)


@pytest.mark.parametrize("tool,adapter,arguments", [
    (SEARCH_CODE_TOOL_NAME, "search_code_tool", {"query": "sample.py"}),
    (EDIT_FILE_TOOL_NAME, "edit_file_tool", {"locator": "repo/file.py", "old_string": "old", "new_string": "new"}),
])
@pytest.mark.parametrize("failure", [ValueError, RuntimeError, FileNotFoundError])
def test_mcp_code_unexpected_errors_remain_private(monkeypatch, caplog, tool, adapter, arguments, failure):
    from mcp.server.mcpserver.exceptions import ToolError

    def fail(service, **kwargs):
        raise failure("private backend path /srv/private-secret")
    monkeypatch.setattr(f"server.mcp_server.{adapter}", fail)
    mcp = create_mcp_server(service=_FakeService(), config=OgMemConfig(mcp_enable_code_edit_tools=True))
    with pytest.raises(ToolError, match="internal_error") as error:
        anyio.run(mcp.call_tool, tool, arguments)
    assert "private-secret" not in str(error.value)
    assert any(r.name == "ogmem.mcp" and r.exc_info and "private-secret" in str(r.exc_info[1]) for r in caplog.records)


@pytest.mark.parametrize("tool,adapter,arguments", [
    (SEARCH_CODE_TOOL_NAME, "search_code_tool", {"query": "sample.py"}),
    (EDIT_FILE_TOOL_NAME, "edit_file_tool", {"locator": "repo/file.py", "old_string": "old", "new_string": "new"}),
])
def test_mcp_code_preserves_protocol_errors(monkeypatch, tool, adapter, arguments):
    from mcp.shared.exceptions import MCPError

    failure = MCPError(-32003, "temporarily_unavailable", data={"retryable": True})
    def fail(service, **kwargs):
        raise failure
    monkeypatch.setattr(f"server.mcp_server.{adapter}", fail)
    mcp = create_mcp_server(service=_FakeService(), config=OgMemConfig(mcp_enable_code_edit_tools=True))
    with pytest.raises(MCPError) as error:
        anyio.run(mcp.call_tool, tool, arguments)
    assert error.value is failure


def test_create_mcp_server_code_toolset_isolates_search_read_and_enabled_edit():
    mcp = create_mcp_server(
        service=_FakeService(),
        config=OgMemConfig(
            mcp_default_toolset="code",
            mcp_enable_write_tools=True,
            mcp_enable_delete_tools=True,
            mcp_enable_code_edit_tools=True,
        ),
    )

    tool_names = {tool.name for tool in anyio.run(mcp.list_tools)}

    assert tool_names == {SEARCH_CODE_TOOL_NAME, "ogm_read_memory", EDIT_FILE_TOOL_NAME}
    assert "ogm_query_memory" not in tool_names
    assert tool_names.isdisjoint(_SKILL_TOOLS)
    assert "ogm_query_memory" not in mcp.instructions
    assert "skill" not in mcp.instructions.lower()
    assert "ogm_write_memory" not in mcp.instructions
    assert "ogm_delete_memory" not in mcp.instructions


def test_create_mcp_server_code_toolset_registers_only_status_resource():
    mcp = create_mcp_server(
        service=_FakeService(),
        config=OgMemConfig(mcp_default_toolset="code"),
    )

    resources = anyio.run(mcp.list_resources)
    templates = anyio.run(mcp.list_resource_templates)

    assert {str(resource.uri) for resource in resources} == {"ogmemory://status"}
    assert templates == []


def test_code_toolset_read_memory_only_delegates_filtered_output_recovery(monkeypatch):
    calls = []

    def fake_read_memory(service, **kwargs):
        calls.append(kwargs)
        return {"ok": True, "memory_uri": kwargs["uri"], "original": "full"}

    monkeypatch.setattr("server.mcp_server.read_memory_tool", fake_read_memory)
    mcp = create_mcp_server(
        service=_FakeService(),
        config=OgMemConfig(mcp_default_toolset="code"),
    )
    tool = mcp._tool_manager.get_tool("ogm_read_memory")

    with pytest.raises(
        ValueError,
        match="only supports ogm_read_memory with recover_original=true",
    ):
        tool.fn(uri="ogmemory://memories/preference/editor", recover_original=False)
    with pytest.raises(
        ValueError,
        match="only supports filtered-output memory URI recovery",
    ):
        tool.fn(
            uri="ogmemory://memories/preference/editor",
            recover_original=True,
        )

    recovery_uri = (
        "ctx://acct/agents/bot/memories/tool_outputs/"
        "tool_output_read_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    )
    schema = next(t for t in anyio.run(mcp.list_tools) if t.name == "ogm_read_memory").input_schema
    assert set(schema["properties"]) == {"uri", "max_chars", "recover_original"}
    assert schema["properties"]["recover_original"]["const"] is True
    assert schema["properties"]["max_chars"]["minimum"] == 1
    assert schema["properties"]["max_chars"]["maximum"] == 50000
    result = tool.fn(uri=recovery_uri)

    assert result == {"ok": True, "memory_uri": recovery_uri, "original": "full"}
    assert calls == [
        {
            "uri": recovery_uri,
            "level": "all",
            "max_chars": 12000,
            "recover_original": True,
            "session_id": None,
            "agent_id": None,
            "transport_headers": {},
            "allow_stdio_env_auth": True,
            "resolved_identity": None,
        }
    ]


def test_default_toolset_read_memory_still_delegates_generic_memory(monkeypatch):
    calls = []

    def fake_read_memory(service, **kwargs):
        calls.append(kwargs)
        return {"uri": kwargs["uri"], "content": "dark"}

    monkeypatch.setattr("server.mcp_server.read_memory_tool", fake_read_memory)
    mcp = create_mcp_server(service=_FakeService(), config=OgMemConfig())
    tool = mcp._tool_manager.get_tool("ogm_read_memory")

    uri = "ogmemory://memories/preference/editor"
    result = tool.fn(uri=uri)

    assert result == {"uri": uri, "content": "dark"}
    assert calls[0]["recover_original"] is False


def test_create_mcp_server_applies_streamable_http_settings():
    cfg = OgMemConfig(mcp_host="127.0.0.1", mcp_port=8092, mcp_path="/custom-mcp")

    mcp = create_mcp_server(service=_FakeService(), config=cfg)

    runtime = mcp._ogmemory_runtime_settings
    assert runtime.host == "127.0.0.1"
    assert runtime.port == 8092
    assert runtime.streamable_http_path == "/custom-mcp"
    assert runtime.sse_path == "/custom-mcp"


@pytest.mark.parametrize("sse_path", ["/messages", "/messages/custom"])
def test_create_mcp_sse_app_rejects_message_namespace_path(sse_path):
    class _FakeMcp:
        _ogmemory_runtime_settings = _MCPRuntimeSettings(
            host="127.0.0.1",
            port=8091,
            streamable_http_path="/mcp",
            sse_path=sse_path,
            message_path="/messages/",
            transport_security=None,
            max_request_body_size=1048576,
        )

        def sse_app(self):
            pytest.fail("conflicting SSE path must be rejected before building the SDK app")

    with pytest.raises(RuntimeError, match="must not equal or be nested under"):
        create_mcp_sse_app(_FakeMcp(), _FakeService())


def test_create_mcp_sse_app_keeps_legacy_transport_paths():
    service = _FakeService()
    mcp = create_mcp_server(
        service=service,
        config=OgMemConfig(mcp_path="/legacy-sse"),
        sse_path="/legacy-sse",
    )

    app = create_mcp_sse_app(mcp, service)

    assert app._sse_path == "/legacy-sse"
    assert app._message_path == "/messages"


@pytest.mark.asyncio
async def test_mcp_http_guard_does_not_replay_disconnected_partial_body():
    app_called = False

    async def app(scope, receive, send):
        nonlocal app_called
        app_called = True

    service = _FakeDisposeService()
    guard = _MCPHttpTransportGuard(
        app,
        service,
        sse_path="/sse",
        message_path="/messages/",
    )
    session_id = "1" * 32
    identity = service.auth.resolve_identity({})
    guard._session_owners[session_id] = guard._identity_fingerprint(identity)
    incoming = iter(
        [
            {"type": "http.request", "body": b"{}", "more_body": True},
            {"type": "http.disconnect"},
        ]
    )

    async def receive():
        return next(incoming)

    async def send(message):
        pytest.fail(f"disconnected request must not send a response: {message}")

    await guard(
        {
            "type": "http",
            "method": "POST",
            "path": "/messages/",
            "query_string": f"session_id={session_id}".encode(),
            "headers": [],
        },
        receive,
        send,
    )

    assert app_called is False


def test_mcp_http_app_exposes_dispose_on_same_service():
    service = _FakeDisposeService()
    mcp = create_mcp_server(service=service, config=OgMemConfig())
    app = create_mcp_http_app(mcp, service)

    response = TestClient(app).post(
        "/api/v1/dispose",
        json={"sessionId": "s1", "agentId": "eval-agent"},
        headers={
            "Authorization": "Bearer root-key",
            "X-Account-ID": "acct-1",
            "X-User-ID": "eval-user",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"disposed": True, "ctx": "ctx-from-same-service"}
    assert service.auth.headers["authorization"] == "Bearer root-key"
    assert service.build_context_params == {"sessionId": "s1", "agentId": "eval-agent"}
    assert service.build_context_identity.role == Role.MEMBER
    assert service.dispose_params["_ctx"] == "ctx-from-same-service"
def test_mcp_http_static_resource_preserves_request_headers(monkeypatch):
    captured = {}

    def fake_list_skills(service, **kwargs):
        captured.update(kwargs)
        return {"skills": []}

    monkeypatch.setattr("server.mcp_server.list_skills_tool", fake_list_skills)
    service = _FakeService()
    mcp = create_mcp_server(
        service=service,
        config=OgMemConfig(mcp_allowed_hosts=["testserver"]),
    )
    app = create_mcp_http_app(mcp, service)
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "resources/read",
        "params": {
            "uri": "ogmemory://skills",
            "_meta": {
                "io.modelcontextprotocol/protocolVersion": "2026-07-28",
                "io.modelcontextprotocol/clientInfo": {"name": "test", "version": "1"},
                "io.modelcontextprotocol/clientCapabilities": {},
            },
        },
    }

    with TestClient(app) as client:
        response = client.post(
            "/mcp",
            json=request,
            headers={
                "Accept": "application/json, text/event-stream",
                "MCP-Protocol-Version": "2026-07-28",
                "Mcp-Method": "resources/read",
                "Mcp-Name": "ogmemory://skills",
                "X-API-Key": "header-key",
            },
        )

    assert response.status_code == 200, response.text
    assert captured["transport_headers"]["x-api-key"] == "header-key"
    assert captured["allow_stdio_env_auth"] is False


def test_mcp_http_concurrent_resources_keep_identity_and_agent_headers_isolated(monkeypatch):
    barrier = threading.Barrier(2)
    captured = {}

    def fake_list_skills(service, **kwargs):
        headers = kwargs["transport_headers"]
        api_key = headers["x-api-key"]
        barrier.wait(timeout=5)
        captured[api_key] = (
            headers["x-agent-id"],
            kwargs["resolved_identity"].user_id,
        )
        return {"skills": []}

    monkeypatch.setattr("server.mcp_server.list_skills_tool", fake_list_skills)
    service = _FakeDisposeService()
    service.auth = _KeyedFakeAuth()
    mcp = create_mcp_server(
        service=service,
        config=OgMemConfig(mcp_allowed_hosts=["testserver"]),
    )
    app = create_mcp_http_app(mcp, service)
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "resources/read",
        "params": {
            "uri": "ogmemory://skills",
            "_meta": {
                "io.modelcontextprotocol/protocolVersion": "2026-07-28",
                "io.modelcontextprotocol/clientInfo": {"name": "test", "version": "1"},
                "io.modelcontextprotocol/clientCapabilities": {},
            },
        },
    }

    def read_resource(client, api_key, agent_id):
        return client.post(
            "/mcp",
            json=request,
            headers={
                "Accept": "application/json, text/event-stream",
                "MCP-Protocol-Version": "2026-07-28",
                "Mcp-Method": "resources/read",
                "Mcp-Name": "ogmemory://skills",
                "X-API-Key": api_key,
                "X-Agent-ID": agent_id,
            },
        )

    with TestClient(app) as client, ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(read_resource, client, "user-a", "agent-a"),
            executor.submit(read_resource, client, "user-b", "agent-b"),
        ]
        responses = [future.result(timeout=10) for future in futures]

    assert [response.status_code for response in responses] == [200, 200]
    assert captured == {
        "user-a": ("agent-a", "user-a"),
        "user-b": ("agent-b", "user-b"),
    }


@pytest.mark.parametrize("protocol_version", ["2025-03-26", "2025-06-18", "2025-11-25"])
def test_mcp_http_app_accepts_legacy_initialize(protocol_version):
    service = _FakeService()
    mcp = create_mcp_server(
        service=service,
        config=OgMemConfig(mcp_allowed_hosts=["testserver"]),
    )
    app = create_mcp_http_app(mcp, service)

    with TestClient(app) as client:
        response = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": protocol_version,
                    "capabilities": {},
                    "clientInfo": {"name": "legacy-test", "version": "1"},
                },
            },
            headers={"Accept": "application/json, text/event-stream"},
        )

    assert response.status_code == 200, response.text
    assert response.json()["result"]["protocolVersion"] == protocol_version
    assert response.headers["mcp-session-id"]


def test_mcp_http_session_cannot_be_reused_by_another_identity():
    service = _FakeDisposeService()
    service.auth = _KeyedFakeAuth()
    mcp = create_mcp_server(
        service=service,
        config=OgMemConfig(mcp_allowed_hosts=["testserver"]),
    )
    app = create_mcp_http_app(mcp, service)

    with TestClient(app) as client:
        initialize = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "identity-test", "version": "1"},
                },
            },
            headers={
                "Accept": "application/json, text/event-stream",
                "X-API-Key": "user-a",
            },
        )
        assert initialize.status_code == 200, initialize.text
        session_id = initialize.headers["mcp-session-id"]
        hijack = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            headers={
                "Accept": "application/json, text/event-stream",
                "MCP-Protocol-Version": "2025-11-25",
                "MCP-Session-ID": session_id,
                "X-API-Key": "user-b",
            },
        )

    assert hijack.status_code == 404, hijack.text


def test_mcp_http_app_exposes_idle_wait_on_same_service():
    service = _FakeDisposeService()
    mcp = create_mcp_server(service=service, config=OgMemConfig())
    app = create_mcp_http_app(mcp, service)

    response = TestClient(app).post(
        "/api/v1/idle",
        json={"sessionId": "s1", "agentId": "eval-agent"},
        headers={
            "Authorization": "Bearer root-key",
            "X-Account-ID": "acct-1",
            "X-User-ID": "eval-user",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"idle": True, "ctx": "ctx-from-same-service"}
    assert service.idle_params["_ctx"] == "ctx-from-same-service"


def test_mcp_http_app_runs_idle_wait_outside_event_loop():
    service = _FakeThreadCheckingDisposeService()
    mcp = create_mcp_server(service=service, config=OgMemConfig())
    app = create_mcp_http_app(mcp, service)

    response = TestClient(app).post(
        "/api/v1/idle",
        json={"sessionId": "s1", "agentId": "eval-agent"},
        headers={
            "Authorization": "Bearer root-key",
            "X-Account-ID": "acct-1",
            "X-User-ID": "eval-user",
        },
    )

    assert response.status_code == 200
    assert response.json()["running_loop"] is False


def test_mcp_http_app_rejects_dispose_without_role_control():
    service = _FakeDisposeService()
    service.auth = _FakeNoAuth()
    mcp = create_mcp_server(service=service, config=OgMemConfig())
    app = create_mcp_http_app(mcp, service)

    response = TestClient(app).post(
        "/api/v1/dispose",
        json={"sessionId": "s1", "agentId": "eval-agent"},
    )

    assert response.status_code == 401
    assert response.json() == {"error": "authentication_required"}
    assert service.dispose_params is None


def test_mcp_http_app_rejects_oversized_harness_body():
    service = _FakeDisposeService()
    cfg = OgMemConfig(mcp_max_request_bytes=128)
    mcp = create_mcp_server(service=service, config=cfg)
    app = create_mcp_http_app(mcp, service)

    response = TestClient(app).post(
        "/api/v1/idle",
        json={"sessionId": "s1", "padding": "x" * 256},
        headers={"X-API-Key": "root-key"},
    )

    assert response.status_code == 413
    assert service.idle_params is None


def test_mcp_http_app_rejects_identity_fields_on_harness_routes():
    service = _FakeDisposeService()
    mcp = create_mcp_server(service=service, config=OgMemConfig())
    app = create_mcp_http_app(mcp, service)

    response = TestClient(app).post(
        "/api/v1/idle",
        json={"sessionId": "s1", "userId": "body-user"},
        headers={
            "Authorization": "Bearer root-key",
            "X-Account-ID": "acct-1",
            "X-User-ID": "eval-user",
        },
    )

    assert response.status_code == 400
    assert response.json() == {"error": "invalid_request"}
    assert service.idle_params is None


def test_create_mcp_server_applies_transport_security_allowlists():
    cfg = OgMemConfig(
        mcp_host="127.0.0.1",
        mcp_allowed_hosts=["memory.local"],
        mcp_allowed_origins=["https://agent.example.com"],
    )

    mcp = create_mcp_server(service=_FakeService(), config=cfg)

    security = mcp._ogmemory_runtime_settings.transport_security
    assert security.enable_dns_rebinding_protection is True
    assert security.allowed_hosts == ["memory.local", "memory.local:*"]
    assert security.allowed_origins == ["https://agent.example.com"]


def test_create_mcp_server_defaults_transport_security_origins_from_hosts():
    cfg = OgMemConfig(mcp_allowed_hosts=["127.0.0.1", "localhost"])

    mcp = create_mcp_server(service=_FakeService(), config=cfg)

    security = mcp._ogmemory_runtime_settings.transport_security
    assert "127.0.0.1:*" in security.allowed_hosts
    assert "localhost:*" in security.allowed_hosts
    assert "http://127.0.0.1:*" in security.allowed_origins
    assert "http://localhost:*" in security.allowed_origins


def test_http_context_disables_stdio_env_auth_fallback():
    ctx = _FakeMCPContext(_FakeRequest({"x-api-key": "header-key"}))

    kwargs = _auth_kwargs_from_context(ctx)

    assert kwargs["transport_headers"]["x-api-key"] == "header-key"
    assert kwargs["allow_stdio_env_auth"] is False


def test_missing_context_allows_stdio_env_auth_fallback():
    kwargs = _auth_kwargs_from_context(None)

    assert kwargs["transport_headers"] == {}
    assert kwargs["allow_stdio_env_auth"] is True
