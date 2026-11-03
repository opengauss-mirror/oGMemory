"""Exercise the real Deep Agents graph with a fake model and HTTP transport."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
import pytest

pytest.importorskip("deepagents")

from deepagents import create_deep_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from integrations.deepagents import OGMemoryMiddleware
from integrations.deepagents.middleware import _serialize


class RecordingModel(FakeMessagesListChatModel):
    requests: list[list[Any]] = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.requests.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


class MemoryServer:
    def __init__(self, status=200):
        self.calls = []
        self.status = status
        self.transport = httpx.MockTransport(self.handle)

    def handle(self, request):
        body = json.loads(request.content)
        self.calls.append((request.url.path, body, request.headers))
        if request.url.path.endswith("compose"):
            return httpx.Response(
                self.status,
                json={
                    "messages": [
                        {"role": "user", "content": "User prefers uv", "_ogmem": True},
                        {"role": "user", "content": "Do not inject this echoed input"},
                    ]
                },
            )
        return httpx.Response(self.status, json={"ok": True})

    def middleware(self, **kwargs):
        return OGMemoryMiddleware(
            "https://memory.example/base",
            account_id="account",
            user_id="alice",
            agent_id="deep-agent",
            api_key="private-key",
            client=httpx.Client(transport=self.transport),
            async_client=httpx.AsyncClient(transport=self.transport),
            **kwargs,
        )


def config(thread_id="conversation-1"):
    return {"configurable": {"thread_id": thread_id}}


def graph(memory, responses=None):
    model = RecordingModel(responses=responses or [AIMessage(content="Use uv")])
    return create_deep_agent(model=model, middleware=[memory], checkpointer=InMemorySaver()), model


def test_sync_recall_capture_checkpoint_and_session_isolation():
    server = MemoryServer()
    agent, model = graph(
        server.middleware(),
        [AIMessage(content="one"), AIMessage(content="two"), AIMessage(content="three")],
    )
    for text, session in [("first", "a"), ("second", "a"), ("third", "b")]:
        result = agent.invoke({"messages": [HumanMessage(content=text)]}, config(session))
        assert not any("ogmemory-context" in str(m.content) for m in result["messages"])
        assert "ogmemory_context" not in result
    assert [c[0].split("/")[-1] for c in server.calls] == ["compose", "after_turn"] * 3
    writes = [body for path, body, _ in server.calls if path.endswith("after_turn")]
    assert [[m["content"] for m in body["messages"]] for body in writes] == [
        ["first", "one"],
        ["second", "two"],
        ["third", "three"],
    ]
    assert [b["sessionId"] for b in writes] == ["a", "a", "b"]
    for _, body, headers in server.calls:
        assert (body["accountId"], body["userId"], body["agentId"]) == (
            "account",
            "alice",
            "deep-agent",
        )
        assert headers["X-API-Key"] == "private-key"
        assert headers["X-Account-ID"] == "account"
        assert headers["X-User-ID"] == "alice"
    assert all(
        any("User prefers uv" in str(m.content) for m in request) for request in model.requests
    )
    assert not any("echoed input" in str(m.content) for request in model.requests for m in request)


def test_tools_are_written_once_at_end_of_run():
    def lookup() -> str:
        """Look up a package manager."""
        return "uv found"

    server = MemoryServer()
    model = RecordingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "lookup", "args": {}, "id": "call-1", "type": "tool_call"}],
            ),
            AIMessage(content="Use uv"),
        ]
    )
    agent = create_deep_agent(model=model, tools=[lookup], middleware=[server.middleware()])
    result = agent.invoke({"messages": [HumanMessage(content="find it")]}, config())
    assert len(server.calls) == 2
    assert server.calls[1][1]["messages"] == [
        {"role": "user", "content": "find it", "id": result["messages"][0].id},
        {"role": "tool", "content": "uv found", "tool_call_id": "call-1", "id": result["messages"][2].id},
        {"role": "assistant", "content": "Use uv", "id": result["messages"][3].id},
    ]
    assert len(model.requests) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_identical_turns_keep_distinct_stable_message_ids(asynchronous):
    server = MemoryServer()
    agent, _ = graph(server.middleware(), [AIMessage(content="same", id="a1"), AIMessage(content="same", id="a2")])
    for message_id in ["u1", "u2"]:
        inputs = {"messages": [HumanMessage(content="continue", id=message_id)]}
        if asynchronous:
            await agent.ainvoke(inputs, config())
        else:
            agent.invoke(inputs, config())
    writes = [body["messages"] for path, body, _ in server.calls if path.endswith("after_turn")]
    assert [[m["id"] for m in messages] for messages in writes] == [["u1", "a1"], ["u2", "a2"]]
    assert [[m["content"] for m in messages] for messages in writes] == [["continue", "same"]] * 2


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_return_direct_tool_captures_completed_turn(asynchronous):
    from langchain_core.tools import tool

    @tool(return_direct=True)
    def lookup() -> str:
        """Look up a package manager."""
        return "uv found"

    server = MemoryServer()
    model = RecordingModel(responses=[AIMessage(content="", tool_calls=[
        {"name": "lookup", "args": {}, "id": "call-direct", "type": "tool_call"}
    ])])
    agent = create_deep_agent(model=model, tools=[lookup], middleware=[server.middleware()])
    inputs = {"messages": [HumanMessage(content="find it", id="user-direct")]}
    result = await agent.ainvoke(inputs, config()) if asynchronous else agent.invoke(inputs, config())
    assert result["messages"][-1].type == "tool"
    assert len(model.requests) == 1
    writes = [body["messages"] for path, body, _ in server.calls if path.endswith("after_turn")]
    assert writes == [[
        {"role": "user", "content": "find it", "id": "user-direct"},
        {"role": "tool", "content": "uv found", "tool_call_id": "call-direct", "id": result["messages"][-1].id},
    ]]


def test_pending_tool_calls_are_not_captured():
    memory = MemoryServer().middleware()
    state = {
        "ogmemory_input": [{"role": "user", "content": "find it", "id": "u"}],
        "ogmemory_seen_ids": ["u"],
        "messages": [
            AIMessage(content="", id="a", tool_calls=[
                {"name": "lookup", "args": {}, "id": "call-1", "type": "tool_call"},
                {"name": "lookup", "args": {}, "id": "call-2", "type": "tool_call"},
            ]),
            ToolMessage(content="first result", tool_call_id="call-1", id="t"),
        ],
    }
    assert memory._capture(state) is None


@pytest.mark.asyncio
async def test_async_recall_capture_and_dispose():
    server = MemoryServer()
    memory = server.middleware(force_extract=True)
    agent, model = graph(memory)
    result = await agent.ainvoke({"messages": [HumanMessage(content="hello")]}, config("async"))
    assert result["messages"][-1].content == "Use uv"
    assert "User prefers uv" in str(model.requests[0])
    await memory.adispose("async")
    assert [c[0].split("/")[-1] for c in server.calls] == ["compose", "after_turn", "dispose"]
    assert server.calls[1][1]["forceExtract"] is True
    assert all(c[1]["sessionId"] == "async" for c in server.calls)
    await memory.async_client.aclose()
    memory.client.close()


@pytest.mark.parametrize(
    "recall,capture,paths",
    [
        (True, False, ["compose"]),
        (False, True, ["after_turn"]),
        (False, False, []),
    ],
)
def test_independent_auto_switches(recall, capture, paths):
    server = MemoryServer()
    agent, _ = graph(server.middleware(auto_recall=recall, auto_capture=capture))
    agent.invoke({"messages": [HumanMessage(content="hello")]}, config())
    assert [c[0].split("/")[-1] for c in server.calls] == paths


@pytest.mark.parametrize("status", [401, 403, 500])
def test_http_failure_is_fail_open_without_exposing_secrets(status, caplog):
    server = MemoryServer(status)
    agent, model = graph(server.middleware())
    result = agent.invoke({"messages": [HumanMessage(content="secret input")]}, config())
    assert result["messages"][-1].content == "Use uv"
    assert "ogmemory-context" not in str(model.requests)
    assert "continuing agent" in caplog.text
    assert "private-key" not in caplog.text
    assert "secret input" not in caplog.text


def test_strict_failure_and_missing_thread_are_reported():
    agent, _ = graph(MemoryServer(500).middleware(fail_open=False))
    with pytest.raises(httpx.HTTPStatusError):
        agent.invoke({"messages": [HumanMessage(content="hello")]}, config())
    server = MemoryServer()
    model = RecordingModel(responses=[AIMessage(content="ok")])
    agent = create_deep_agent(model=model, middleware=[server.middleware()])
    with pytest.raises(ValueError, match="thread_id"):
        agent.invoke({"messages": [HumanMessage(content="hello")]})
    assert not server.calls


def test_concurrent_threads_do_not_share_turn_state():
    server = MemoryServer()
    agent, _ = graph(server.middleware())
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(
            pool.map(
                lambda session: agent.invoke(
                    {"messages": [HumanMessage(content=session)]}, config(session)
                ),
                ["a", "b"],
            )
        )
    writes = [body for path, body, _ in server.calls if path.endswith("after_turn")]
    assert len(writes) == 2
    assert all(body["messages"][0]["content"] == body["sessionId"] for body in writes)


def test_multimodal_and_system_messages_are_filtered():
    assert _serialize(
        [
            SystemMessage(content="system secret"),
            HumanMessage(
                content=[
                    {"type": "text", "text": "remember"},
                    {"type": "image_url", "image_url": "secret"},
                ]
            ),
            AIMessage(
                content=[
                    {"type": "reasoning", "reasoning": "private"},
                    {"type": "text", "text": "answer"},
                ]
            ),
            ToolMessage(content="result", tool_call_id="call"),
        ]
    ) == [
        {"role": "user", "content": "remember"},
        {"role": "assistant", "content": "answer"},
        {"role": "tool", "content": "result", "tool_call_id": "call"},
    ]


def test_environment_and_export(monkeypatch):
    from cli.commands.plugin_config import Connection, build_artifacts

    files = build_artifacts(
        "deepagents", Connection("https://memory.example", "key", "a", "u", "g")
    )
    assert "OGMEM_API_KEY" in files["ogmemory.env.ps1"]
    assert "middleware=[memory]" in files["ogmemory_deepagents.py"]
    assert "mcp.config.json" not in files
    for name, value in {
        "ACCOUNT_ID": "a",
        "USER_ID": "u",
        "AGENT_ID": "g",
        "AUTO_CAPTURE": "false",
    }.items():
        monkeypatch.setenv("OGMEM_" + name, value)
    memory = OGMemoryMiddleware.from_env()
    assert memory.identity == {"accountId": "a", "userId": "u", "agentId": "g"}
    assert memory.auto_recall is True
    assert memory.auto_capture is False


@pytest.mark.parametrize("data", [[], {"error": "secret"}, {"ok": False}, "invalid"])
def test_invalid_success_response_fails_open(data):
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=data))
    memory = OGMemoryMiddleware(
        account_id="a", user_id="u", agent_id="g", client=httpx.Client(transport=transport)
    )
    assert memory.dispose("session") == {}


def test_dispose_does_not_dispose_between_turns():
    server = MemoryServer()
    memory = server.middleware()
    agent, _ = graph(memory)
    agent.invoke({"messages": [HumanMessage(content="hello")]}, config())
    assert not any(path.endswith("dispose") for path, _, _ in server.calls)
    assert memory.dispose("conversation-1") == {"ok": True}
    assert server.calls[-1][0].endswith("dispose")


def test_interrupt_resume_captures_only_completed_turn():
    from langgraph.types import Command

    def lookup() -> str:
        """Look up a package manager."""
        return "uv found"

    server = MemoryServer()
    model = RecordingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "lookup", "args": {}, "id": "call-1", "type": "tool_call"}],
            ),
            AIMessage(content="Use uv"),
        ]
    )
    agent = create_deep_agent(
        model=model,
        tools=[lookup],
        middleware=[server.middleware()],
        checkpointer=InMemorySaver(),
        interrupt_on={"lookup": True},
    )
    paused = agent.invoke({"messages": [HumanMessage(content="find it")]}, config())
    assert paused["__interrupt__"]
    assert [c[0].split("/")[-1] for c in server.calls] == ["compose"]
    result = agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config())
    assert [c[0].split("/")[-1] for c in server.calls] == ["compose", "after_turn"]
    assert server.calls[1][1]["messages"][0] == {
        "role": "user", "content": "find it", "id": result["messages"][0].id
    }


def test_stream_finishes_capture_and_private_state_is_not_exposed():
    server = MemoryServer()
    agent, _ = graph(server.middleware())
    chunks = list(
        agent.stream({"messages": [HumanMessage(content="hello")]}, config(), stream_mode="values")
    )
    assert chunks[-1]["messages"][-1].content == "Use uv"
    assert server.calls[-1][0].endswith("after_turn")
    input_properties = agent.get_input_jsonschema().get("properties", {})
    output_properties = agent.get_output_jsonschema().get("properties", {})
    assert not any(k.startswith("ogmemory_") for k in input_properties)
    assert not any(k.startswith("ogmemory_") for k in output_properties)


@pytest.mark.asyncio
async def test_async_stream_and_transport_failure(caplog):
    def fail(request):
        raise httpx.ReadTimeout("private-key and content should not be logged", request=request)

    memory = OGMemoryMiddleware(
        account_id="a",
        user_id="u",
        agent_id="g",
        async_client=httpx.AsyncClient(transport=httpx.MockTransport(fail)),
    )
    agent, _ = graph(memory)
    chunks = [
        chunk
        async for chunk in agent.astream(
            {"messages": [HumanMessage(content="hello")]}, config(), stream_mode="values"
        )
    ]
    assert chunks[-1]["messages"][-1].content == "Use uv"
    assert caplog.text.count("continuing agent") == 2
    assert "private-key" not in caplog.text
    await memory.async_client.aclose()


@pytest.mark.parametrize(
    "endpoint",
    [
        "file:///tmp",
        "https://user:key@example.com",
        "https://example.com/api/v1",
        "http://example.com:0",
        "http://example.com:99999",
    ],
)
def test_bad_connection_configuration_is_rejected(endpoint):
    with pytest.raises(ValueError):
        OGMemoryMiddleware(endpoint, account_id="a", user_id="u", agent_id="g")


def test_deepagent_alias_exports_native_sample(tmp_path):
    from cli.commands.plugin_config import run
    from cli.main import build_parser

    args = build_parser().parse_args(
        [
            "onboard",
            "--mode",
            "plugin",
            "--agent",
            "deepagent",
            "--non-interactive",
            "--ogmem-url",
            "https://memory.example",
            "--output-dir",
            str(tmp_path / "export"),
        ]
    )
    assert run(args) == 0
    script = (tmp_path / "export/ogmemory_deepagents.py").read_text(encoding="utf-8")
    compile(script, "ogmemory_deepagents.py", "exec")
