"""Exercise raw write arguments through the MCP SDK dispatcher and production tools."""
import json
from dataclasses import fields

import pytest
from mcp import Client

from core.models import CandidateMemory, ContextNode
from providers.unified_config import OgMemConfig
from server.mcp_server import create_mcp_server
from tests.unit.server.test_mcp_tools_write import _FakeService


@pytest.mark.parametrize("field", [
    "uri", "category", "type", "memory_type", "memory_id", "abstract", "overview",
    "content", "context_type", "owner_space", "level", "metadata", "account_id", "user_id",
    "unknown_field",
])
@pytest.mark.asyncio
async def test_mcp_write_rejects_unknown_top_level_fields_before_writing(field):
    service = _FakeService(OgMemConfig(mcp_enable_write_tools=True))
    async with Client(create_mcp_server(service=service, config=service._cfg)) as client:
        result = await client.call_tool("ogm_write_memory", {
            "session_id": "write-boundary", "text": "Remember the release date.",
            field: "caller-supplied-atomic-or-identity-value",
        })

    assert result.is_error is True
    assert service.after_turn_params is None
    assert service.trace_events == {}


@pytest.mark.parametrize("field", sorted(
    {field.name for model in (CandidateMemory, ContextNode) for field in fields(model)} - {"content"}
))
@pytest.mark.asyncio
async def test_mcp_write_rejects_atomic_message_fields_before_writing(field):
    service = _FakeService(OgMemConfig(mcp_enable_write_tools=True))
    async with Client(create_mcp_server(service=service, config=service._cfg)) as client:
        result = await client.call_tool("ogm_write_memory", {
            "session_id": "write-boundary",
            "messages": [{"role": "user", "content": "Remember the release date.", field: "atomic"}],
        })

    assert result.is_error is True
    assert service.after_turn_params is None


@pytest.mark.parametrize("source", ["text", "messages"])
@pytest.mark.asyncio
async def test_mcp_write_accepts_json_facts_inside_raw_user_text(source):
    service = _FakeService(OgMemConfig(mcp_enable_write_tools=True))
    fact = json.dumps({"uri": "catalog://book/42", "category": "book", "abstract": "A book", "content": "facts"})
    args = {"session_id": "write-json-fact"}
    args.update({"text": fact} if source == "text" else {"messages": [{"role": "user", "content": fact}]})
    async with Client(create_mcp_server(service=service, config=service._cfg)) as client:
        result = await client.call_tool("ogm_write_memory", args)

    assert result.is_error is False
    assert service.after_turn_params["messages"] == [{"role": "user", "content": fact}]


@pytest.mark.parametrize("fact", ["null", "true", '[{"uri":"catalog://book/42"}]'])
@pytest.mark.asyncio
async def test_mcp_write_preserves_json_literal_text(fact):
    service = _FakeService(OgMemConfig(mcp_enable_write_tools=True))
    async with Client(create_mcp_server(service=service, config=service._cfg)) as client:
        result = await client.call_tool("ogm_write_memory", {
            "session_id": "write-json-literal", "text": fact,
        })

    assert result.is_error is False
    assert service.after_turn_params["messages"] == [{"role": "user", "content": fact}]


@pytest.mark.parametrize("value", [{"content": "atomic"}, [{"content": "atomic"}], True])
@pytest.mark.asyncio
async def test_mcp_write_requires_text_string(value):
    service = _FakeService(OgMemConfig(mcp_enable_write_tools=True))
    async with Client(create_mcp_server(service=service, config=service._cfg)) as client:
        result = await client.call_tool("ogm_write_memory", {
            "session_id": "write-text-type", "text": value,
        })

    assert result.is_error is True
    assert service.after_turn_params is None


@pytest.mark.asyncio
async def test_mcp_write_text_schema_and_explicit_null_with_messages():
    service = _FakeService(OgMemConfig(mcp_enable_write_tools=True))
    async with Client(create_mcp_server(service=service, config=service._cfg)) as client:
        tools = await client.list_tools()
        tool = next(tool for tool in tools.tools if tool.name == "ogm_write_memory")
        assert tool.input_schema["properties"]["text"]["anyOf"] == [
            {"type": "string"}, {"type": "null"},
        ]
        result = await client.call_tool("ogm_write_memory", {
            "session_id": "write-null-text", "text": None,
            "messages": [{"role": "user", "content": "Remember this fact."}],
        })

    assert result.is_error is False
    assert service.after_turn_params["messages"] == [
        {"role": "user", "content": "Remember this fact."},
    ]


@pytest.mark.parametrize("tool,required,fields", [
    ("query_memory", {"query": "fact"}, ["session_id", "agent_id"]),
    ("list_memories", {}, ["cursor", "session_id", "agent_id"]),
    ("read_memory", {"uri": "ogmemory://memories/entity/example"}, ["unit_key", "session_id", "agent_id"]),
    ("write_memory", {"session_id": "required", "messages": [{"role": "user", "content": "fact"}]}, ["text", "agent_id", "client_request_id"]),
    ("delete_memory", {"uri": "ogmemory://memories/entity/example"}, ["reason", "session_id", "agent_id", "client_request_id"]),
])
@pytest.mark.parametrize("value", ["123", "null", "true", '{"fact":"value"}', "[]"])
@pytest.mark.asyncio
async def test_mcp_memory_optional_strings_preserve_raw_values(monkeypatch, tool, required, fields, value):
    captured = {}

    def handler(service, **kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr("server.mcp_server." + tool + "_tool", handler)
    cfg = OgMemConfig(mcp_enable_write_tools=True, mcp_enable_delete_tools=True)
    service = _FakeService(cfg)
    args = {**required, **dict.fromkeys(fields, value)}
    async with Client(create_mcp_server(service=service, config=cfg)) as client:
        result = await client.call_tool("ogm_" + tool, args)

    assert result.is_error is False
    assert {key: captured[key] for key in fields} == dict.fromkeys(fields, value)
