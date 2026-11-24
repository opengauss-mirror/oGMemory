"""Exercise the real Deep Agents graph with a fake model and HTTP transport."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
import pytest

pytest.importorskip("deepagents")

from deepagents import create_deep_agent
from langchain.tools import ToolRuntime
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
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
        data = {"ok": True}
        if request.url.path.endswith("after_turn"):
            data.update(trace_captured=True, trace_status="CAPTURED", status="accumulating")
        return httpx.Response(self.status, json=data)

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
    assert [b["traceMessages"] for b in writes] == [b["messages"] for b in writes]
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
    assert memory._capture(state, {}) is None


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
    if not capture:
        assert agent.get_state(config()).values.get("ogmemory_input") == []


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


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_interrupt_resume_captures_only_completed_turn(asynchronous):
    def lookup() -> str:
        """Look up a package manager."""
        return "uv found"

    server = MemoryServer()
    saver = InMemorySaver()

    def build(model):
        return create_deep_agent(model=model, tools=[lookup], middleware=[server.middleware()],
                                 checkpointer=saver, interrupt_on={"lookup": True})

    agent = build(RecordingModel(responses=[AIMessage(content="", tool_calls=[
        {"name": "lookup", "args": {}, "id": "call-1", "type": "tool_call"}])]))
    inputs = {"messages": [HumanMessage(content="find it")]}
    paused = await agent.ainvoke(inputs, config()) if asynchronous else agent.invoke(inputs, config())
    assert paused["__interrupt__"]
    assert [c[0].split("/")[-1] for c in server.calls] == ["compose"]
    saved = await agent.aget_state(config()) if asynchronous else agent.get_state(config())
    request_id = saved.values["ogmemory_request_id"]
    # Rebuild middleware and graph: state must come from the checkpoint.
    agent = build(RecordingModel(responses=[AIMessage(content="Use uv")]))
    resume = Command(resume={"decisions": [{"type": "approve"}]})
    result = await agent.ainvoke(resume, config()) if asynchronous else agent.invoke(resume, config())
    assert [c[0].split("/")[-1] for c in server.calls] == ["compose", "after_turn"]
    body = server.calls[1][1]
    assert body["messages"][0] == {
        "role": "user", "content": "find it", "id": result["messages"][0].id
    }
    assert body["clientRequestId"] == request_id
    assert [m["role"] for m in body["traceMessages"]] == ["user", "assistant", "tool", "assistant"]


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


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_trace_upload_preserves_parallel_calls(asynchronous):
    from langchain_core.tools import tool

    @tool
    def lookup(query: str) -> str:
        """Find a value."""
        return query + " found"

    server = MemoryServer()
    model = RecordingModel(responses=[AIMessage(content="", tool_calls=[
        {"name": "lookup", "args": {"query": "uv"}, "id": "c1"},
        {"name": "lookup", "args": {"query": "pip"}, "id": "c2"},
    ]), AIMessage(content="done")])
    agent = create_deep_agent(model=model, tools=[lookup], middleware=[server.middleware()])
    inputs = {"messages": [HumanMessage(content="find", id="u")]}
    if asynchronous:
        await agent.ainvoke(inputs, config())
    else:
        agent.invoke(inputs, config())
    body = server.calls[-1][1]
    trace = body["traceMessages"]
    assert [m["role"] for m in trace] == ["user", "assistant", "tool", "tool", "assistant"]
    assert trace[1]["content"] == ""
    assert trace[1]["tool_calls"] == [
        {"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": '{"query":"uv"}'}},
        {"id": "c2", "type": "function", "function": {"name": "lookup", "arguments": '{"query":"pip"}'}},
    ]
    assert {(m["tool_call_id"], m["content"]) for m in trace if m["role"] == "tool"} == {
        ("c1", "uv found"), ("c2", "pip found")}
    assert body["clientRequestId"]


def test_trace_serializer_excludes_private_blocks_and_preserves_empty_errors():
    from integrations.deepagents.middleware import _serialize_trace

    trace = _serialize_trace([
        SystemMessage(content="private system"),
        HumanMessage(content=[{"type": "text", "text": "question"},
                              {"type": "image_url", "image_url": "data:image/png;base64,secret"}]),
        AIMessage(content=[{"type": "reasoning", "text": "private thought"}],
                  tool_calls=[{"id": "c", "name": "lookup", "args": {"q": "uv"}}]),
        ToolMessage(content="", tool_call_id="c", name="lookup", status="error", artifact={"secret": True}),
    ])
    assert trace[0] == {"role": "user", "content": "question"}
    assert trace[1]["tool_calls"][0]["function"]["name"] == "lookup"
    assert trace[2] == {"role": "tool", "content": "", "tool_call_id": "c", "name": "lookup", "is_error": True}
    assert "secret" not in json.dumps(trace) and "private" not in json.dumps(trace)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("fail_open", [False, True])
async def test_pending_trace_is_not_reported_as_captured(asynchronous, fail_open, caplog):
    server = MemoryServer()
    original = server.handle

    def handle(request):
        response = original(request)
        if request.url.path.endswith("after_turn"):
            return httpx.Response(202, json={"ok": True, "status": "accumulating",
                "trace_captured": False, "trace_status": "PENDING",
                "trace_error": {"code": "TRACE_CAPTURE_PENDING", "retry_action": "retry_trace"}})
        return response

    server.transport = httpx.MockTransport(handle)
    agent, _ = graph(server.middleware(fail_open=fail_open))
    async def run():
        if asynchronous:
            return await agent.ainvoke({"messages": [HumanMessage(content="secret")]}, config())
        return agent.invoke({"messages": [HumanMessage(content="secret")]}, config())
    if fail_open:
        result = await run()
        assert result["messages"][-1].content == "Use uv"
        assert "trace" in caplog.text.lower()
        assert "secret" not in caplog.text
    else:
        with pytest.raises(ValueError, match="trace"):
            await run()
    assert len([c for c in server.calls if c[0].endswith("after_turn")]) == 1


def test_unresolved_call_followed_by_final_text_is_not_complete():
    memory = MemoryServer().middleware()
    state = {"ogmemory_input": [{"role": "user", "content": "q", "id": "u"}],
             "ogmemory_seen_ids": ["u"], "messages": [
                 AIMessage(content="", id="a", tool_calls=[{"id": "c", "name": "lookup", "args": {}}]),
                 AIMessage(content="premature final", id="b")]}
    assert memory._capture(state, {}) is None


def test_trace_checkpoint_payload_is_stable_and_turns_are_distinct(monkeypatch):
    import copy

    import integrations.deepagents.middleware as module

    monkeypatch.setattr(module, "get_config", lambda: config())
    memory = MemoryServer().middleware(auto_recall=False)
    def turn_state(user_id, assistant_id):
        state = {"messages": [HumanMessage(content="q", id=user_id)]}
        update = memory._before(state)[0]
        update["ogmemory_trace"] = update["ogmemory_trace"].value
        update["ogmemory_trace_error"] = update["ogmemory_trace_error"].value
        state.update(update)
        state["messages"].append(AIMessage(content="a", id=assistant_id))
        return state

    state = turn_state("u", "a")
    identity = memory._payload()
    first = memory._capture(state, identity)
    replay = memory._capture(copy.deepcopy(state), identity)
    assert first == replay
    assert first["clientRequestId"]
    assert memory._capture(turn_state("u2", "a2"), identity)["clientRequestId"] != first["clientRequestId"]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_compaction_keeps_original_tool_trace_and_excludes_summary(asynchronous):
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.messages import RemoveMessage
    from langchain_core.tools import tool
    from langgraph.graph.message import REMOVE_ALL_MESSAGES

    class Compact(AgentMiddleware):
        def before_model(self, state, runtime):
            if state["messages"][-1].type == "tool":
                return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES),
                    HumanMessage(content="synthetic summary", id="summary")]}
        async def abefore_model(self, state, runtime):
            return self.before_model(state, runtime)

    @tool
    def lookup() -> str:
        """Return the original result."""
        return "original result"

    server = MemoryServer()
    model = RecordingModel(responses=[AIMessage(content="", id="call-msg", tool_calls=[
        {"id": "c", "name": "lookup", "args": {}}]), AIMessage(content="answer", id="final")])
    agent = create_deep_agent(model=model, tools=[lookup], middleware=[server.middleware(), Compact()],
                              checkpointer=InMemorySaver())
    inputs = {"messages": [HumanMessage(content="find", id="u")]}
    if asynchronous:
        await agent.ainvoke(inputs, config())
    else:
        agent.invoke(inputs, config())
    trace = server.calls[-1][1]["traceMessages"]
    assert [m["role"] for m in trace] == ["user", "assistant", "tool", "assistant"]
    assert trace[2]["content"] == "original result"
    assert "synthetic summary" not in json.dumps(trace)
    assert "User prefers uv" not in json.dumps(trace)


def test_trace_observations_do_not_replace_original_with_offloaded_result():
    from integrations.deepagents.middleware import _merge_trace

    assert _merge_trace([{"id": "t", "role": "tool", "content": "original"}],
                        [{"id": "t", "role": "tool", "content": "offloaded pointer"}]) == [
                            {"id": "t", "role": "tool", "content": "original"}]


def test_trace_tool_args_filter_raw_media_without_dropping_visible_parameters():
    from integrations.deepagents.middleware import _serialize_trace

    trace = _serialize_trace([AIMessage(content="", tool_calls=[{"id": "c", "name": "analyze",
        "args": {"query": "visible", "image": "data:image/png;base64,secret",
                 "blocks": [{"type": "image_url", "image_url": "secret"},
                            {"type": "text", "text": "visible"}]}}])])
    args = json.loads(trace[0]["tool_calls"][0]["function"]["arguments"])
    assert args["query"] == "visible"
    # Tool arguments are business data, not framework content blocks.
    assert args["blocks"] == [{"type": "image_url", "image_url": "secret"},
                              {"type": "text", "text": "visible"}]
    assert "data:image" not in json.dumps(trace)


def test_trace_captures_all_trailing_user_inputs_without_prior_turns():
    server = MemoryServer()
    agent, _ = graph(server.middleware())
    agent.invoke({"messages": [HumanMessage(content="old", id="old-u"),
        AIMessage(content="old answer", id="old-a"),
        HumanMessage(content="first input", id="u1"), HumanMessage(content="second input", id="u2")]}, config())
    body = server.calls[-1][1]
    assert [m["content"] for m in body["traceMessages"]] == ["first input", "second input", "Use uv"]
    assert [m["content"] for m in body["messages"]] == ["second input", "Use uv"]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_command_tool_dict_preserves_error_trace_when_result_is_replaced(asynchronous):
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.tools import tool
    from langgraph.types import Command

    class Offload(AgentMiddleware):
        def before_model(self, state, runtime):
            if state["messages"][-1].type == "tool":
                message = state["messages"][-1].model_copy(update={"content": "offloaded"})
                return {"messages": [message]}
        async def abefore_model(self, state, runtime):
            return self.before_model(state, runtime)

    @tool
    def lookup(runtime: ToolRuntime) -> Command:
        """Return a command with an explicit error status."""
        return Command(update={"messages": [{"type": "tool", "content": "original error",
            "tool_call_id": runtime.tool_call_id, "name": "lookup", "status": "error"}]})

    server = MemoryServer()
    model = RecordingModel(responses=[AIMessage(content="", tool_calls=[
        {"id": "c", "name": "lookup", "args": {}}]), AIMessage(content="failed")])
    agent = create_deep_agent(model=model, tools=[lookup], middleware=[server.middleware(), Offload()])
    inputs = {"messages": [HumanMessage(content="find", id="u")]}
    if asynchronous:
        await agent.ainvoke(inputs, config())
    else:
        agent.invoke(inputs, config())
    result = next(m for m in server.calls[-1][1]["traceMessages"] if m["role"] == "tool")
    assert result["content"] == "original error"
    assert result["is_error"] is True


@pytest.mark.parametrize("data", [
    {"ok": True}, {"ok": True, "trace_captured": True, "trace_status": "PENDING"},
    {"ok": True, "trace_captured": False, "trace_status": "CAPTURED"},
])
def test_unconfirmed_trace_responses_fail_strict_without_retry(data):
    calls = []
    def handle(request):
        calls.append(request)
        return httpx.Response(202, json=data)
    memory = OGMemoryMiddleware(account_id="a", user_id="u", agent_id="g", fail_open=False,
        client=httpx.Client(transport=httpx.MockTransport(handle)))
    with pytest.raises(ValueError, match="trace"):
        memory._post("after_turn", {"traceMessages": [{"role": "user", "content": "q"}]})
    assert len(calls) == 1


def test_raw_data_uri_text_is_not_uploaded_in_either_message_format():
    from integrations.deepagents.middleware import _serialize_trace

    messages = [HumanMessage(content="data:audio/wav;base64,secret"),
                ToolMessage(content="data:image/png;base64,secret", tool_call_id="c")]
    assert _serialize(messages) == []
    assert _serialize_trace(messages) == [
        {"role": "tool", "content": "", "tool_call_id": "c", "is_error": False}]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("command_result", [False, True])
async def test_tool_result_lists_remain_compatible(asynchronous, command_result):
    from langchain.agents import create_agent
    from langchain_core.tools import tool

    @tool
    def lookup(runtime: ToolRuntime):
        """Return a supported list of tool outputs."""
        message = ToolMessage(content="result", tool_call_id=runtime.tool_call_id)
        return [Command(update={"messages": [message]})] if command_result else [message]

    server = MemoryServer()
    model = RecordingModel(responses=[AIMessage(content="", tool_calls=[
        {"id": "c", "name": "lookup", "args": {}}]), AIMessage(content="done")])
    # Deep Agents 0.7.23 FilesystemMiddleware rejects lists independently.
    # Exercise the LangChain ToolNode contract directly at this boundary.
    agent = create_agent(model=model, tools=[lookup], middleware=[server.middleware()])
    inputs = {"messages": [HumanMessage(content="find", id="u")]}
    if asynchronous:
        result = await agent.ainvoke(inputs, config())
    else:
        result = agent.invoke(inputs, config())
    assert result["messages"][-1].content == "done"
    trace = server.calls[-1][1]["traceMessages"]
    assert next(m for m in trace if m["role"] == "tool")["content"] == "result"


def test_tool_dictionary_result_does_not_upload_embedded_media():
    def lookup() -> dict:
        """Return text and a raw image."""
        return {"caption": "visible", "image": "data:image/png;base64,c2VjcmV0",
                "audio": "data:audio/wav;base64,secret",
                "note": "visible with data:audio/wav;base64,c2VjcmV0 attached"}

    server = MemoryServer()
    model = RecordingModel(responses=[AIMessage(content="", tool_calls=[
        {"id": "c", "name": "lookup", "args": {}}]), AIMessage(content="done")])
    agent = create_deep_agent(model=model, tools=[lookup], middleware=[server.middleware()])
    agent.invoke({"messages": [HumanMessage(content="find")]}, config())
    body = server.calls[-1][1]
    for key in ["messages", "traceMessages"]:
        serialized = json.dumps(body[key])
        assert "c2VjcmV0" not in serialized and "secret" not in serialized
        assert "visible" in serialized


def test_visible_json_configuration_is_not_mistaken_for_media_payload():
    from integrations.deepagents.middleware import _serialize_trace

    content = '{"type": "file", "path": "/workspace/report.txt"}'
    messages = [HumanMessage(content=content)]
    assert _serialize(messages) == [{"role": "user", "content": content}]
    assert _serialize_trace(messages) == [{"role": "user", "content": content}]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("entry", ["user", "args", "result"])
@pytest.mark.parametrize("kind", [["string", "null"], {"category": "schema"}, "thinking", "image", "file"])
async def test_business_json_is_preserved_at_every_entry(asynchronous, entry, kind):
    value = {"type": kind, "url": "business-url", "path": "/business",
             "nested": {"type": ["object", "null"]}, "text": '{"type":"thinking"}',
             "ordinary": "c2VjcmV0"}
    def lookup(payload: dict) -> dict:
        """Return business data."""
        assert payload == (value if entry == "args" else {})
        return value if entry == "result" else {"ok": True}
    server = MemoryServer()
    model = RecordingModel(responses=[AIMessage(content="", tool_calls=[
        {"id": "c", "name": "lookup", "args": {"payload": value if entry == "args" else {}}}]),
        AIMessage(content="done")])
    agent = create_deep_agent(model=model, tools=[lookup], middleware=[server.middleware()])
    text = json.dumps(value) if entry == "user" else "find"
    inputs = {"messages": [HumanMessage(content=text)]}
    result = await agent.ainvoke(inputs, config()) if asynchronous else agent.invoke(inputs, config())
    assert result["messages"][-1].content == "done"
    body = server.calls[-1][1]
    trace = body["traceMessages"]
    if entry == "args":
        assert json.loads(trace[1]["tool_calls"][0]["function"]["arguments"])["payload"] == value
    else:
        role = "user" if entry == "user" else "tool"
        for key in ["messages", "traceMessages"]:
            assert json.loads(next(m["content"] for m in body[key] if m["role"] == role)) == value


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("fail_open", [False, True])
@pytest.mark.parametrize("entry", ["user", "args", "result", "capture"])
async def test_conversion_failure_obeys_fail_open_without_partial_capture(
    asynchronous, fail_open, entry, monkeypatch, caplog
):
    import integrations.deepagents.middleware as module
    original = module._serialize_trace
    def serialize(messages):
        if ((entry == "user" and messages[0].type == "human")
            or (entry == "args" and messages[0].type == "ai")
            or (entry == "result" and messages[0].type == "tool")
            or (entry == "capture" and len(messages) > 1)):
            raise TypeError("private-key secret input")
        return original(messages)
    monkeypatch.setattr(module, "_serialize_trace", serialize)
    calls = []
    def lookup() -> str:
        """Return original tool result."""
        calls.append(True)
        return "original"
    server = MemoryServer()
    memory = server.middleware(fail_open=fail_open)
    model = RecordingModel(responses=[AIMessage(content="", tool_calls=[
        {"id": "c", "name": "lookup", "args": {}}]), AIMessage(content="done")])
    agent = create_deep_agent(model=model, tools=[lookup], middleware=[memory], checkpointer=InMemorySaver())
    async def run():
        inputs = {"messages": [HumanMessage(content="secret input")]}
        return await agent.ainvoke(inputs, config()) if asynchronous else agent.invoke(inputs, config())
    if fail_open:
        result = await run()
        assert result["messages"][-1].content == "done"
        assert next(m for m in result["messages"] if m.type == "tool").content == "original"
        assert len(model.requests) == 2 and calls == [True]
        writes = [body for path, body, _ in server.calls if path.endswith("after_turn")]
        if entry == "user":
            assert writes == []
        else:
            assert len(writes) == 1 and "traceMessages" not in writes[0]
            assert [m["content"] for m in writes[0]["messages"]] == ["secret input", "original", "done"]
        assert "continuing agent" in caplog.text
        assert "private-key" not in caplog.text and "secret input" not in caplog.text
    else:
        from integrations.deepagents.middleware import TraceCaptureError
        if entry == "user":
            with pytest.raises(TypeError, match="private-key"):
                await run()
        else:
            with pytest.raises(TraceCaptureError, match="TypeError"):
                await run()
            writes = [body for path, body, _ in server.calls if path.endswith("after_turn")]
            assert len(writes) == 1 and "traceMessages" not in writes[0]
            assert writes[0]["messages"][-1]["content"] == "done"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_failed_parallel_observation_preserves_messages_and_next_trace_recovers(asynchronous, monkeypatch):
    import integrations.deepagents.middleware as module
    original = module._serialize_trace
    def serialize(messages):
        if any(m.type == "tool" and m.content == "bad" for m in messages):
            raise ValueError("private payload")
        return original(messages)
    monkeypatch.setattr(module, "_serialize_trace", serialize)
    def lookup(value: str) -> str:
        """Return a value."""
        return value
    server = MemoryServer()
    model = RecordingModel(responses=[AIMessage(content="", tool_calls=[
        {"id": "bad", "name": "lookup", "args": {"value": "bad"}},
        {"id": "good", "name": "lookup", "args": {"value": "good"}}]),
        AIMessage(content="first done"), AIMessage(content="second done")])
    agent = create_deep_agent(model=model, tools=[lookup], middleware=[server.middleware()],
                              checkpointer=InMemorySaver())
    for text in ["first", "second"]:
        inputs = {"messages": [HumanMessage(content=text)]}
        result = await agent.ainvoke(inputs, config()) if asynchronous else agent.invoke(inputs, config())
        assert result["messages"][-1].content == text + " done"
    writes = [body for path, body, _ in server.calls if path.endswith("after_turn")]
    assert len(writes) == 2
    assert "traceMessages" not in writes[0]
    assert writes[0]["messages"][-1]["content"] == "first done"
    assert [m["content"] for m in writes[1]["traceMessages"]] == ["second", "second done"]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_tool_exception_is_not_swallowed_by_memory_fail_open(asynchronous):
    def lookup() -> str:
        """Raise an original tool failure."""
        raise RuntimeError("original tool failure")
    model = RecordingModel(responses=[AIMessage(content="", tool_calls=[
        {"id": "c", "name": "lookup", "args": {}}])])
    agent = create_deep_agent(model=model, tools=[lookup], middleware=[MemoryServer().middleware()])
    with pytest.raises(RuntimeError, match="original tool failure"):
        inputs = {"messages": [HumanMessage(content="find")]}
        if asynchronous:
            await agent.ainvoke(inputs, config())
        else:
            agent.invoke(inputs, config())


def test_non_string_content_block_types_are_ignored_without_reinterpreting_text():
    from integrations.deepagents.middleware import _serialize_trace
    message = HumanMessage(content=[{"type": ["text"], "text": "hidden"},
        {"type": {"category": "text"}, "text": "hidden"},
        {"type": "thinking", "text": "private"},
        {"type": "file", "path": "/private"},
        {"type": "text", "text": '{"type":"thinking","value":"visible"}'}])
    assert _serialize_trace([message]) == [{"role": "user",
        "content": '{"type":"thinking","value":"visible"}'}]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("fail_open", [False, True])
async def test_command_normalization_failure_obeys_fail_open(
    asynchronous, fail_open, monkeypatch
):
    import integrations.deepagents.middleware as module
    def fail(messages):
        raise TypeError("normalization failed")
    monkeypatch.setattr(module, "convert_to_messages", fail)
    def lookup(runtime: ToolRuntime) -> Command:
        """Return a framework-valid command."""
        return Command(update={"messages": [ToolMessage(content="original",
            tool_call_id=runtime.tool_call_id)]})
    server = MemoryServer()
    model = RecordingModel(responses=[AIMessage(content="", tool_calls=[
        {"id": "c", "name": "lookup", "args": {}}]), AIMessage(content="done")])
    agent = create_deep_agent(model=model, tools=[lookup], middleware=[server.middleware(fail_open=fail_open)])
    async def run():
        inputs = {"messages": [HumanMessage(content="find")]}
        return await agent.ainvoke(inputs, config()) if asynchronous else agent.invoke(inputs, config())
    if fail_open:
        result = await run()
        assert result["messages"][-1].content == "done"
        assert next(m for m in result["messages"] if m.type == "tool").content == "original"
        writes = [body for path, body, _ in server.calls if path.endswith("after_turn")]
        assert len(writes) == 1 and "traceMessages" not in writes[0]
        assert [m["content"] for m in writes[0]["messages"]] == ["find", "original", "done"]
    else:
        from integrations.deepagents.middleware import TraceCaptureError
        with pytest.raises(TraceCaptureError, match="TypeError"):
            await run()
        writes = [body for path, body, _ in server.calls if path.endswith("after_turn")]
        assert len(writes) == 1 and "traceMessages" not in writes[0]
        assert writes[0]["messages"][-1]["content"] == "done"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_model_exception_is_not_swallowed_by_memory_fail_open(asynchronous):
    class FailingModel(RecordingModel):
        def _generate(self, *args, **kwargs):
            raise RuntimeError("original model failure")
    agent = create_deep_agent(model=FailingModel(responses=[AIMessage(content="unused")]),
                              middleware=[MemoryServer().middleware()])
    with pytest.raises(RuntimeError, match="original model failure"):
        inputs = {"messages": [HumanMessage(content="find")]}
        if asynchronous:
            await agent.ainvoke(inputs, config())
        else:
            agent.invoke(inputs, config())


def test_trace_merge_deduplicates_ids_and_unidentified_content():
    from integrations.deepagents.middleware import _merge_trace
    plain = {"role": "assistant", "content": "visible"}
    new = {"role": "assistant", "content": "new"}
    assert _merge_trace([plain], [plain, new, new,
        {"id": "next", "role": "assistant", "content": "next"},
        {"id": "next", "role": "assistant", "content": "changed"},
    ]) == [plain, new, {"id": "next", "role": "assistant", "content": "next"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_command_replayed_history_is_state_only_not_current_trace(asynchronous):
    def replay_history(runtime: ToolRuntime) -> Command:
        """Return historical state and the current result."""
        return Command(update={"messages": [*runtime.state["messages"],
            ToolMessage(content="new tool result", id="new-tool", tool_call_id=runtime.tool_call_id)]})
    server = MemoryServer()
    model = RecordingModel(responses=[AIMessage(content="old answer", id="old-a"),
        AIMessage(content="", id="new-call", tool_calls=[
            {"id": "call-new", "name": "replay_history", "args": {}}]),
        AIMessage(content="new answer", id="new-a")])
    agent = create_deep_agent(model=model, tools=[replay_history], middleware=[server.middleware()],
                              checkpointer=InMemorySaver())
    old = [HumanMessage(content="old question", id="old-u"),
        AIMessage(content="", id="old-call", tool_calls=[
            {"id": "call-old", "name": "replay_history", "args": {}}]),
        ToolMessage(content="old tool result", id="old-tool", tool_call_id="call-old"),
        HumanMessage(content="old continuation", id="old-u2")]
    async def run(messages):
        return await agent.ainvoke({"messages": messages}, config()) if asynchronous else agent.invoke(
            {"messages": messages}, config())
    await run(old)
    result = await run([HumanMessage(content="new question", id="new-u")])
    body = server.calls[-1][1]
    assert [m["id"] for m in body["traceMessages"]] == ["new-u", "new-call", "new-tool", "new-a"]
    assert [m["content"] for m in body["messages"]] == ["new question", "new tool result", "new answer"]
    assert "old" not in json.dumps(body["traceMessages"])
    assert {m.id for m in result["messages"]} >= {m.id for m in old} | {"old-a", "new-tool"}
    assert body["traceMessages"][2]["tool_call_id"] == "call-new"


@pytest.mark.parametrize("text", [
    "data: temperature=20", 'data: {"status":"ok"}', 'data: {"status":"ok","count":2}',
    "data:orders", "data:orders,active", "metadata:image/png;base64,Zm9v",
])
def test_business_data_labels_are_not_data_uris(text):
    from integrations.deepagents.middleware import _sanitize_trace_args, _visible_text
    assert _visible_text(text) == text
    assert _sanitize_trace_args({"label": text}) == {"label": text}


@pytest.mark.parametrize("uri", [
    "data:image/png;base64,Zm9v", "data:image/svg+xml,%3Csvg%3Eprivate%3C/svg%3E",
    "data:,private%20payload", "data:;charset=UTF-8,private%20payload",
    "DATA:text/plain;charset=utf-8;name=a%22b;base64,cHJpdmF0ZQ==",
])
def test_data_uri_boundaries_preserve_surrounding_business_text(uri):
    from integrations.deepagents.middleware import _sanitize_trace_args, _visible_text
    assert _visible_text(uri) == ""
    assert _sanitize_trace_args({"raw": uri}) == {}
    for prefix, suffix in [("", " visible suffix"), ("prefix ", " suffix"),
                           ('{"raw":"', '","note":"visible"}'),
                           ('{"quoted":"\\"', '\\" visible"}')]:
        text = prefix + uri + suffix
        expected = prefix + "[media omitted]" + suffix
        assert _visible_text(text) == expected
        assert _sanitize_trace_args({"raw": text}) == {"raw": expected}


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("text", ["data: temperature=20", 'data: {"status":"ok","count":2}'])
async def test_business_data_text_recall_and_tool_capture(asynchronous, text):
    def lookup(label: str) -> str:
        """Return an SSE business result."""
        assert label == "data:orders"
        return text
    server = MemoryServer()
    model = RecordingModel(responses=[AIMessage(content="", tool_calls=[
        {"id": "c", "name": "lookup", "args": {"label": "data:orders"}}]), AIMessage(content="ack")])
    agent = create_deep_agent(model=model, tools=[lookup], middleware=[server.middleware()])
    inputs = {"messages": [HumanMessage(content=text)]}
    result = await agent.ainvoke(inputs, config()) if asynchronous else agent.invoke(inputs, config())
    assert result["messages"][-1].content == "ack"
    assert [path.split("/")[-1] for path, _, _ in server.calls] == ["compose", "after_turn"]
    assert server.calls[0][1]["prompt"] == text
    body = server.calls[-1][1]
    for key in ["messages", "traceMessages"]:
        assert [m["content"] for m in body[key] if m["role"] in {"user", "tool"}] == [text, text]
    assert json.loads(body["traceMessages"][1]["tool_calls"][0]["function"]["arguments"]) == {
        "label": "data:orders"}


def test_command_observation_filter_preserves_updates_and_control_flow():
    from types import SimpleNamespace
    old = HumanMessage(content="old", id="old")
    current = ToolMessage(content="current", id="new", tool_call_id="c")
    command = Command(update={"messages": [old, current], "custom": {"value": 1}},
                      goto="next-node", resume="resume-value")
    request = SimpleNamespace(state={"ogmemory_input": [{"role": "user", "content": "q"}],
                                     "ogmemory_seen_ids": ["old"]})
    result = MemoryServer().middleware()._tool_result(request, command)
    assert result.goto == command.goto and result.resume == command.resume and result.graph == command.graph
    assert result.update["custom"] == command.update["custom"]
    assert result.update["messages"] == [old, current]
    assert [m["id"] for m in result.update["ogmemory_trace"]] == ["new"]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_mixed_data_uri_text_is_removed_without_losing_business_context(asynchronous):
    uri = "data:image/svg+xml;charset=UTF-8,%3Csvg%3Eprivate%3C/svg%3E"
    text = uri + " visible suffix"
    expected = "[media omitted] visible suffix"
    def lookup(label: str) -> str:
        """Return a data URI surrounded by business text."""
        assert label == text
        return json.dumps({"raw": uri, "note": text, "label": "data:orders"})
    server = MemoryServer()
    model = RecordingModel(responses=[AIMessage(content="", tool_calls=[
        {"id": "c", "name": "lookup", "args": {"label": text}}]), AIMessage(content="ack")])
    agent = create_deep_agent(model=model, tools=[lookup], middleware=[server.middleware()])
    inputs = {"messages": [HumanMessage(content=text)]}
    if asynchronous:
        await agent.ainvoke(inputs, config())
    else:
        agent.invoke(inputs, config())
    assert server.calls[0][1]["prompt"] == expected
    body = server.calls[-1][1]
    for key in ["messages", "traceMessages"]:
        assert next(m["content"] for m in body[key] if m["role"] == "user") == expected
        tool = json.loads(next(m["content"] for m in body[key] if m["role"] == "tool"))
        assert tool == {"raw": "[media omitted]", "note": expected, "label": "data:orders"}
        assert "private" not in json.dumps(body[key])
    args = json.loads(body["traceMessages"][1]["tool_calls"][0]["function"]["arguments"])
    assert args == {"label": expected}


@pytest.mark.parametrize("recall,capture,expected", [
    (True, True, 1), (True, False, 1), (False, True, 1), (False, False, 0),
])
def test_turn_preparation_resolves_identity_once_when_enabled(recall, capture, expected, monkeypatch):
    import integrations.deepagents.middleware as module
    monkeypatch.setattr(module, "get_config", lambda: config())
    memory = MemoryServer().middleware(auto_recall=recall, auto_capture=capture)
    original = memory._payload
    calls = []
    def payload(**values):
        calls.append(True)
        return original(**values)
    monkeypatch.setattr(memory, "_payload", payload)
    update, prepared = memory._before({"messages": [HumanMessage(content="q", id="u")]})
    assert len(calls) == expected
    assert bool(update["ogmemory_input"]) == (recall or capture)
    if recall:
        assert prepared["sessionId"] == "conversation-1"
    else:
        assert prepared is None


@pytest.mark.parametrize("recall,capture", [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize("fail_open", [False, True])
def test_turn_preparation_keeps_session_validation_outside_fail_open(recall, capture, fail_open, monkeypatch):
    import integrations.deepagents.middleware as module
    monkeypatch.setattr(module, "get_config", lambda: {})
    memory = MemoryServer().middleware(auto_recall=recall, auto_capture=capture, fail_open=fail_open)
    with pytest.raises(ValueError, match="thread_id"):
        memory._before({"messages": [HumanMessage(content="q", id="u")]})


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_completed_turn_resolves_identity_once_per_lifecycle_hook(asynchronous, monkeypatch):
    server = MemoryServer()
    memory = server.middleware(auto_recall=False)
    original = memory._payload
    resolutions = []

    def payload(**values):
        resolutions.append(values)
        return original(**values)

    monkeypatch.setattr(memory, "_payload", payload)
    agent, _ = graph(memory)
    inputs = {"messages": [HumanMessage(content="hello")]}
    if asynchronous:
        await agent.ainvoke(inputs, config("identity-reuse"))
    else:
        agent.invoke(inputs, config("identity-reuse"))
    assert server.calls[0][1]["sessionId"] == "identity-reuse"
    assert len(resolutions) == 2  # One before_agent and one after_agent resolution.


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_media_argument_keys_are_omitted_only_from_uploaded_copy(asynchronous):
    from server.mcp_trace import message_events
    from skills.sanitizer import sanitize_trace_text
    raw = "data:image/png;base64,aW1hZ2U="
    images = {raw: "caption", "prefix " + raw + " suffix": "embedded caption",
              "data:orders": "ordinary", "aW1hZ2U=": "ordinary base64",
              "schema": {"type": ["string", "null"], raw: "nested"}}
    received = []
    def inspect_images(images: dict, label: str) -> str:
        """Inspect the original business mapping."""
        received.append((images, label))
        return "inspected"
    server = MemoryServer()
    model = RecordingModel(responses=[AIMessage(content="", tool_calls=[
        {"id": "c", "name": "inspect_images", "args": {"images": images, "label": "ordinary business"}}]),
        AIMessage(content="done")])
    agent = create_deep_agent(model=model, tools=[inspect_images], middleware=[server.middleware()])
    inputs = {"messages": [HumanMessage(content="inspect images")]}
    if asynchronous:
        await agent.ainvoke(inputs, config())
    else:
        agent.invoke(inputs, config())
    assert received == [(images, "ordinary business")]
    body = server.calls[-1][1]
    assert raw not in json.dumps(body)
    args = json.loads(body["traceMessages"][1]["tool_calls"][0]["function"]["arguments"])
    assert args == {"images": {"data:orders": "ordinary", "aW1hZ2U=": "ordinary base64",
                               "schema": {"type": ["string", "null"]}}, "label": "ordinary business"}
    events = message_events(body["traceMessages"])
    assert raw not in sanitize_trace_text(json.dumps(events))
    assert json.loads(next(e["tool_input"] for e in events if e["event_type"] == "tool_start")) == args


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("fail_open", [False, True])
async def test_mixed_valid_and_invalid_calls_write_completed_messages(asynchronous, fail_open, caplog):
    executed = []
    def lookup(value: str) -> str:
        """Return a valid tool result."""
        executed.append(value)
        return "result " + value
    server = MemoryServer()
    model = RecordingModel(responses=[AIMessage(content="", tool_calls=[
        {"id": "good", "name": "lookup", "args": {"value": "valid"}}], invalid_tool_calls=[
        {"id": "bad", "name": "lookup", "args": '{"value":', "error": "invalid JSON", "type": "invalid_tool_call"}]),
        AIMessage(content="Final answer after handling tool failure.")])
    agent = create_deep_agent(model=model, tools=[lookup],
        middleware=[server.middleware(auto_recall=False, fail_open=fail_open)])
    inputs = {"messages": [HumanMessage(content="perform lookup")]}
    result = await agent.ainvoke(inputs, config()) if asynchronous else agent.invoke(inputs, config())
    assert executed == ["valid"] and len(model.requests) == 2
    tools = {m.tool_call_id: m for m in model.requests[-1] if m.type == "tool"}
    assert tools["bad"].status == "error" and "malformed" in tools["bad"].content
    assert tools["good"].status == "success" and tools["good"].content == "result valid"
    assert [(m.tool_call_id, m.status) for m in model.requests[-1] if m.type == "tool"] == [
        ("bad", "error"), ("good", "success")]
    assert result["messages"][-1].content == "Final answer after handling tool failure."
    writes = [body for path, body, _ in server.calls if path.endswith("after_turn")]
    assert len(writes) == 1
    body = writes[0]
    assert [m["content"] for m in body["messages"]] == [
        "perform lookup", tools["bad"].content, "result valid", "Final answer after handling tool failure."]
    assert [(m["role"], m.get("tool_call_id")) for m in body["traceMessages"]] == [
        ("user", None), ("assistant", None), ("tool", "good"), ("tool", "bad"), ("assistant", None)]
    from server.mcp_trace import message_events
    events = message_events(body["traceMessages"])
    starts = {event["tool_use_id"]: event for event in events if event["event_type"] == "tool_start"}
    assert starts["bad"]["tool_input"] == '{"value":'
    assert json.loads(starts["good"]["tool_input"]) == {"value": "valid"}
    outcomes = {event["tool_use_id"]: event for event in events
                if event["event_type"] in {"tool_failure", "tool_success"}}
    assert outcomes["bad"]["event_type"] == "tool_failure"
    assert outcomes["bad"]["tool_response"] == tools["bad"].content
    assert outcomes["good"]["event_type"] == "tool_success"
    assert "trace unavailable" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("fail_open", [False, True])
async def test_return_direct_with_unobserved_parse_error_writes_messages_only(asynchronous, fail_open, caplog):
    from langchain_core.tools import tool
    @tool(return_direct=True)
    def lookup(value: str) -> str:
        """Return immediately without a second model request."""
        return "result " + value
    server = MemoryServer()
    model = RecordingModel(responses=[AIMessage(content="", tool_calls=[
        {"id": "good", "name": "lookup", "args": {"value": "valid"}}], invalid_tool_calls=[
        {"id": "bad", "name": "lookup", "args": '{"value":', "error": "invalid JSON"}])])
    agent = create_deep_agent(model=model, tools=[lookup],
        middleware=[server.middleware(auto_recall=False, fail_open=fail_open)])
    inputs = {"messages": [HumanMessage(content="perform lookup")]}
    result = await agent.ainvoke(inputs, config()) if asynchronous else agent.invoke(inputs, config())
    assert len(model.requests) == 1
    assert result["messages"][-1].content == "result valid"
    writes = [body for path, body, _ in server.calls if path.endswith("after_turn")]
    assert len(writes) == 1 and "traceMessages" not in writes[0]
    assert [m["content"] for m in writes[0]["messages"]] == ["perform lookup", "result valid"]
    assert "trace unavailable" in caplog.text
