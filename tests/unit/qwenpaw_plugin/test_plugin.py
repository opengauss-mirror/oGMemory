"""QwenPaw lifecycle checks with its native AgentScope SDK."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

pytest.importorskip("agentscope")
from agentscope.agent import Agent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import TextBlock, ToolCallBlock, UserMsg
from agentscope.model import ChatModelBase, ChatResponse
from agentscope.permission import PermissionMode
from agentscope.state import AgentState
from agentscope.tool import FunctionTool, Toolkit

ROOT = Path(__file__).resolve().parents[3]
PLUGIN_DIR = ROOT / "plugins" / "qwenpaw"
spec = importlib.util.spec_from_file_location(
    "qwenpaw_ogmemory_test",
    PLUGIN_DIR / "plugin.py",
    submodule_search_locations=[str(PLUGIN_DIR)],
)
plugin = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plugin
spec.loader.exec_module(plugin)


def context(session="session-1", user="alice"):
    return SimpleNamespace(
        session_id=session,
        root_session_id=session,
        agent_id="qwenpaw",
        request=SimpleNamespace(user_id=user),
    )


def make_client():
    requests = []

    def handle(request):
        body = json.loads(request.content)
        requests.append((request.url.path, body, dict(request.headers)))
        data = {"ok": True}
        if request.url.path == "/api/v1/compose":
            data["messages"] = [{"role": "user", "content": "Alice prefers tea.", "_ogmem": True}]
        return httpx.Response(200, json=data)

    client = plugin.MemoryClient(
        {"account_id": "test-account", "api_key": "test-key"},
        transport=httpx.MockTransport(handle),
    )
    return client, requests


class Model(ChatModelBase):
    def __init__(self, tool=False):
        super().__init__(
            credential=None, model="fixture", parameters=self.Parameters(), stream=False
        )
        self.inputs = []
        self.formatter = OpenAIChatFormatter()
        self.tool = tool

    async def _call_api(self, model_name, messages, **kwargs):
        self.inputs.append(messages)
        if self.tool and len(self.inputs) == 1:
            return ChatResponse(
                content=[ToolCallBlock(id="tool-1", name="lookup", input='{"query":"tea"}')],
                is_last=True,
            )
        return ChatResponse(content=[TextBlock(text="Tea selected.")], is_last=True)


@pytest.mark.asyncio
async def test_real_agent_turn_recall_write_and_canonical_tool_trace():
    client, requests = make_client()
    model = Model(tool=True)

    async def lookup(query: str) -> str:
        """Look up the requested drink."""
        return "Tea is available."

    state = AgentState(session_id="session-1")
    state.permission_context.mode = PermissionMode.BYPASS
    agent = Agent(
        name="qwenpaw",
        system_prompt="Help the user.",
        model=model,
        state=state,
        toolkit=Toolkit(tools=[FunctionTool(lookup)]),
        middlewares=[plugin.OGMemoryMiddleware(client, context())],
    )
    outputs = [
        item
        async for item in agent.reply_stream(
            inputs=UserMsg(
                name="alice",
                content="Choose my preferred drink.",
                metadata={"qwenpaw_tag": "external_user_query"},
            )
        )
    ]
    await client.close()

    assert outputs
    assert any("Alice prefers tea." in m.get_text_content() for m in model.inputs[0])
    assert len(model.inputs) == 2
    writes = [body for path, body, _ in requests if path == "/api/v1/after_turn"]
    assert writes[0]["messages"] == [
        {"role": "user", "content": "Choose my preferred drink."},
        {"role": "assistant", "content": "Tea selected."},
    ]
    traces = [body for path, body, _ in requests if path == "/api/v1/trace/events"]
    kinds = [body["event"]["event_type"] for body in traces]
    for kind in (
        "session_start",
        "user_message",
        "assistant_message",
        "tool_start",
        "tool_success",
        "turn_end",
        "session_end",
    ):
        assert kind in kinds
    assert kinds.index("tool_start") < kinds.index("tool_success") < kinds.index("turn_end")
    tool = next(body["event"] for body in traces if body["event"]["event_type"] == "tool_success")
    assert tool["tool_use_id"] == "tool-1"
    assert tool["tool_name"] == "lookup"
    assert tool["tool_response"] == "Tea is available."
    assert all(body["session_id"] == "session-1" for body in traces)
    assert all(headers["x-api-key"] == "test-key" for _, _, headers in requests)
    sanitizer_spec = importlib.util.spec_from_file_location(
        "qwenpaw_trace_sanitizer", ROOT / "skills" / "sanitizer.py"
    )
    sanitizer = importlib.util.module_from_spec(sanitizer_spec)
    sys.modules[sanitizer_spec.name] = sanitizer
    sanitizer_spec.loader.exec_module(sanitizer)
    result = sanitizer.sanitize_trace([body["event"] for body in traces])
    assert "lookup" in result.tools


@pytest.mark.asyncio
async def test_concurrent_requests_keep_identity_and_turns_separate():
    import asyncio

    client, requests = make_client()

    async def run(session, user):
        agent = Agent(
            name="qwenpaw",
            system_prompt="Help.",
            model=Model(),
            state=AgentState(session_id=session),
            middlewares=[plugin.OGMemoryMiddleware(client, context(session, user))],
        )
        return [
            item
            async for item in agent.reply_stream(
                inputs=UserMsg(name=user, content=user + " prompt")
            )
        ]

    await asyncio.gather(run("s-alice", "alice"), run("s-bob", "bob"))
    await client.close()
    writes = [body for path, body, _ in requests if path == "/api/v1/after_turn"]
    assert {(body["sessionId"], body["userId"]) for body in writes} == {
        ("s-alice", "alice"),
        ("s-bob", "bob"),
    }
    for body in writes:
        assert body["messages"][0]["content"] == body["userId"] + " prompt"


@pytest.mark.asyncio
async def test_session_reuse_sends_each_completed_turn():
    client, requests = make_client()
    state = AgentState(session_id="session-1")
    for prompt in ("First turn", "Second turn"):
        agent = Agent(
            name="qwenpaw",
            system_prompt="Help.",
            model=Model(),
            state=state,
            middlewares=[plugin.OGMemoryMiddleware(client, context())],
        )
        _ = [
            item async for item in agent.reply_stream(inputs=UserMsg(name="alice", content=prompt))
        ]
    await client.close()
    writes = [body for path, body, _ in requests if path == "/api/v1/after_turn"]
    assert [body["messages"][0]["content"] for body in writes] == ["First turn", "Second turn"]
    assert sum(path == "/api/v1/bootstrap" for path, _, _ in requests) == 1


@pytest.mark.asyncio
async def test_trace_redaction_and_canonical_size():
    client, requests = make_client()
    identity = client.identity(context())
    client.emit(
        identity,
        "tool_start",
        "redaction-test",
        tool_name="lookup",
        tool_use_id="redaction-test",
        tool_input={
            "api_key": "private-value",
            "query": "password=private-value",
            "nested": ["x" * 30000] * 64,
        },
    )
    await client.close()
    trace = next(body for path, body, _ in requests if path == "/api/v1/trace/events")
    text = json.dumps(trace)
    assert "[TRUNCATED]" in text
    assert len(text.encode()) < 65536
    clean = plugin.MemoryClient.__module__
    client_module = sys.modules[clean]
    assert client_module.redact({"apiKey": "private-value"}) == {"apiKey": "[REDACTED]"}


@pytest.mark.asyncio
async def test_plugin_registers_native_interfaces():
    client, requests = make_client()
    registered = {}
    api = SimpleNamespace(
        config={},
        register_runtime_hook=lambda hook: registered.update(runtime_hook=hook),
        register_middleware=lambda factory: registered.update(factory=factory),
        register_shutdown_hook=lambda name, callback: registered.update(shutdown=callback),
        register_uninstall_hook=lambda name, callback: registered.update(uninstall=callback),
    )
    integration = plugin.OGMemoryPlugin()
    integration.register(api)
    await integration.client.close()
    integration.client = client
    middleware = registered["factory"](context(), None)
    assert middleware.identity == client.identity(context())
    assert registered["runtime_hook"].name == "ogmemory_compact"
    await registered["uninstall"]()
    await client.close()



@pytest.mark.asyncio
async def test_compaction_preserves_turn_and_emits_boundaries():
    from agentscope.message import AssistantMsg

    client, requests = make_client()
    middleware = plugin.OGMemoryMiddleware(client, context())
    user = UserMsg(
        name="alice", content="First task", metadata={"qwenpaw_tag": "external_user_query"}
    )
    answer = AssistantMsg(name="qwenpaw", content="First task completed.")
    state = AgentState(session_id="session-1", context=[user, answer])
    agent = SimpleNamespace(state=state)
    middleware.input_ids = {user.id}
    middleware.turn_id = user.id

    async def compress(**kwargs):
        state.context = [UserMsg(name="summary", content="Compressed history")]

    await middleware.on_compress_context(agent, {}, compress)
    await client.close()
    assert list(middleware.messages.values()) == [
        {"role": "user", "content": "First task"},
        {"role": "assistant", "content": "First task completed."},
    ]
    kinds = [
        body["event"]["event_type"] for path, body, _ in requests if path == "/api/v1/trace/events"
    ]
    assert kinds[-2:] == ["pre_compact", "post_compact"]
    assert [path for path, _, _ in requests if path != "/api/v1/trace/events"] == [
        "/api/v1/after_turn", "/api/v1/compact",
    ]


@pytest.mark.asyncio
async def test_json_message_trace_preserves_text_contract():
    client, requests = make_client()
    identity = client.identity(context())
    client.emit(
        identity,
        "user_message",
        "json-message",
        prompt='{"request":"find notes","api_key":"private"}',
    )
    await client.close()
    trace = next(body for path, body, _ in requests if path == "/api/v1/trace/events")
    assert json.loads(trace["event"]["prompt"]) == {
        "request": "find notes",
        "api_key": "[REDACTED]",
    }


@pytest.mark.asyncio
async def test_child_session_trace_retains_parent_relationship():
    client, requests = make_client()
    child = context("child-session")
    child.root_session_id = "parent-session"
    middleware = plugin.OGMemoryMiddleware(client, child)
    await client.ensure_session(middleware.identity, middleware.parent)
    await client.close()
    trace = next(body for path, body, _ in requests if path == "/api/v1/trace/events")
    assert trace["session_id"] == "child-session"
    assert trace["event"]["parent_session_id"] == "parent-session"


@pytest.mark.asyncio
async def test_reply_stream_can_advance_from_successive_heartbeat_tasks():
    import asyncio

    client, requests = make_client()
    agent = Agent(
        name="qwenpaw",
        system_prompt="Help.",
        model=Model(),
        state=AgentState(session_id="heartbeat-session"),
        middlewares=[plugin.OGMemoryMiddleware(client, context("heartbeat-session"))],
    )
    stream = agent.reply_stream(inputs=UserMsg(name="alice", content="Remember the release date."))
    events = []
    while True:
        try:
            events.append(await asyncio.create_task(anext(stream)))
        except StopAsyncIteration:
            break
    await client.close()
    assert events
    writes = [body for path, body, _ in requests if path == "/api/v1/after_turn"]
    assert writes[0]["messages"][-1]["content"] == "Tea selected."


@pytest.mark.asyncio
async def test_configured_timeouts_reach_http_requests(monkeypatch):
    for key in ("OGMEMORY_TIMEOUT_SECONDS", "OGMEMORY_TRACE_TIMEOUT_SECONDS"):
        monkeypatch.delenv(key, raising=False)
    timeouts = {}

    def handle(request):
        timeouts[request.url.path] = request.extensions["timeout"]["read"]
        return httpx.Response(200, json={"ok": True})

    client = plugin.MemoryClient(
        {"write_timeout_seconds": 43, "trace_timeout_seconds": 7},
        transport=httpx.MockTransport(handle),
    )
    identity = client.identity(context())
    await client.post("/api/v1/after_turn", identity, {"messages": []})
    client.emit(identity, "session_start", "timeout-check")
    await client.close()
    assert timeouts == {"/api/v1/after_turn": 43, "/api/v1/trace/events": 7}


@pytest.mark.asyncio
async def test_large_json_tool_result_is_redacted_before_trace_limit():
    client, requests = make_client()

    async def lookup(query: str) -> str:
        """Return structured lookup data."""
        return json.dumps({"api_key": "private-value", "result": "x" * 20000})

    state = AgentState(session_id="session-1")
    state.permission_context.mode = PermissionMode.BYPASS
    agent = Agent(
        name="qwenpaw", system_prompt="Help.", model=Model(tool=True), state=state,
        toolkit=Toolkit(tools=[FunctionTool(lookup)]),
        middlewares=[plugin.OGMemoryMiddleware(client, context())],
    )
    _ = [item async for item in agent.reply_stream(inputs=UserMsg(name="alice", content="Lookup"))]
    await client.close()
    result = next(
        body["event"]["tool_response"] for path, body, _ in requests
        if path == "/api/v1/trace/events" and body["event"]["event_type"] == "tool_success"
    )
    assert result.startswith('{"api_key": "[REDACTED]"')
    assert len(result) <= 12000


@pytest.mark.asyncio
async def test_reused_agent_writes_only_the_current_turn():
    client, requests = make_client()
    agent = Agent(
        name="qwenpaw", system_prompt="Help.", model=Model(),
        state=AgentState(session_id="session-1"),
        middlewares=[plugin.OGMemoryMiddleware(client, context())],
    )
    for prompt in ("First turn", "Second turn"):
        _ = [item async for item in agent.reply_stream(inputs=UserMsg(name="alice", content=prompt))]
    await client.close()
    writes = [body["messages"] for path, body, _ in requests if path == "/api/v1/after_turn"]
    assert writes == [
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": "Tea selected."}]
        for prompt in ("First turn", "Second turn")
    ]
    turns = [body["event"]["assistant_message"] for path, body, _ in requests
             if path == "/api/v1/trace/events" and body["event"]["event_type"] == "turn_end"]
    assert turns == ["Tea selected.", "Tea selected."]


@pytest.mark.asyncio
async def test_sessions_bootstrap_concurrently():
    import asyncio

    first_started, second_started, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def handle(request):
        if request.url.path == "/api/v1/bootstrap":
            if json.loads(request.content)["sessionId"] == "first":
                first_started.set()
                await release.wait()
            else:
                second_started.set()
        return httpx.Response(200, json={"ok": True})

    client = plugin.MemoryClient(transport=httpx.MockTransport(handle))
    first = asyncio.create_task(client.ensure_session(client.identity(context("first"))))
    await first_started.wait()
    second = asyncio.create_task(client.ensure_session(client.identity(context("second"))))
    try:
        await asyncio.wait_for(second_started.wait(), timeout=1)
    finally:
        release.set()
        await asyncio.gather(first, second)
        await client.close()


@pytest.mark.asyncio
async def test_shared_bootstrap_finishes_once_for_concurrent_waiters():
    import asyncio

    requests = []

    async def handle(request):
        requests.append((request.url.path, request.extensions["timeout"]["read"]))
        await asyncio.sleep(0.01)
        return httpx.Response(200, json={"ok": True})

    client = plugin.MemoryClient(
        {"bootstrap_timeout_seconds": 2}, transport=httpx.MockTransport(handle)
    )
    identity = client.identity(context())
    await asyncio.gather(client.ensure_session(identity), client.ensure_session(identity))
    await client.close()
    assert [timeout for path, timeout in requests if path == "/api/v1/bootstrap"] == [2]


@pytest.mark.asyncio
async def test_tool_trace_caps_oversized_results():
    from agentscope.event import ToolResultEndEvent, ToolResultStartEvent, ToolResultTextDeltaEvent
    from agentscope.message import ToolResultState

    client, requests = make_client()
    middleware = plugin.OGMemoryMiddleware(client, context())
    agent = SimpleNamespace(state=AgentState(session_id="session-1"))
    event = {"reply_id": "reply", "tool_call_id": "large"}
    middleware.capture_event(agent, ToolResultStartEvent(**event, tool_call_name="lookup"))
    for delta in ('{"api_key": "private-value", "data": "', "x" * 70000, '"}'):
        middleware.capture_event(agent, ToolResultTextDeltaEvent(**event, delta=delta))
    middleware.capture_event(agent, ToolResultEndEvent(**event, state=ToolResultState.SUCCESS))
    await client.close()
    result = next(body["event"] for path, body, _ in requests
                  if path == "/api/v1/trace/events" and body["event"]["event_type"] == "tool_success")
    assert result["tool_response"] == "[TRUNCATED]"
    assert result["truncated"] is True


def test_quoted_secret_keys_in_text_are_redacted():
    client_module = sys.modules[plugin.MemoryClient.__module__]
    assert client_module.redact('payload {"api_key":"private-value", unfinished') == (
        'payload {"[REDACTED], unfinished'
    )


@pytest.mark.asyncio
async def test_reused_agent_recovers_pending_write_with_original_request_id():
    writes = []

    def handle(request):
        body = json.loads(request.content)
        if request.url.path == "/api/v1/after_turn":
            writes.append(body)
            if len(writes) == 1:
                raise httpx.ConnectError("Temporary connection failure", request=request)
        return httpx.Response(200, json={"ok": True})

    client = plugin.MemoryClient(transport=httpx.MockTransport(handle))
    state = AgentState(session_id="session-1")
    agent = Agent(
        name="qwenpaw", system_prompt="Help.", model=Model(), state=state,
        middlewares=[plugin.OGMemoryMiddleware(client, context())],
    )
    for prompt in ("First turn", "Second turn"):
        _ = [item async for item in agent.reply_stream(inputs=UserMsg(name="alice", content=prompt))]
    await client.close()
    assert writes[0] == writes[1]
    assert [row["messages"][0]["content"] for row in writes[1:]] == ["First turn", "Second turn"]
    assert state.middle_context["ogmemory"]["pending"] == []


@pytest.mark.asyncio
async def test_bootstrap_timeout_keeps_conversation_trace_complete():
    requests = []

    def handle(request):
        if request.url.path == "/api/v1/bootstrap":
            raise httpx.ReadTimeout("Bootstrap took too long", request=request)
        requests.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"ok": True})

    client = plugin.MemoryClient(transport=httpx.MockTransport(handle))
    agent = Agent(
        name="qwenpaw", system_prompt="Help.", model=Model(),
        state=AgentState(session_id="session-1"),
        middlewares=[plugin.OGMemoryMiddleware(client, context())],
    )
    _ = [item async for item in agent.reply_stream(inputs=UserMsg(name="alice", content="Continue"))]
    await client.close()
    kinds = [body["event"]["event_type"] for path, body in requests
             if path == "/api/v1/trace/events"]
    assert kinds == ["session_start", "user_message", "assistant_message", "turn_end", "session_end"]


@pytest.mark.asyncio
@pytest.mark.parametrize("text", [
    "Authorization: Basic dXNlcjpwYXNz",
    "Cookie: session=private-cookie; token=private-token",
    "OG_AUTH_API_KEY=private-value",
])
async def test_plain_text_credentials_are_redacted(text):
    client, requests = make_client()
    client.emit(client.identity(context()), "tool_success", "credentials", tool_response=text)
    await client.close()
    trace = next(body for path, body, _ in requests if path == "/api/v1/trace/events")
    assert trace["event"]["tool_response"] == "[REDACTED]"


@pytest.mark.asyncio
async def test_tool_loop_preserves_assistant_messages_with_shared_reply_id():
    class SpeakingModel(Model):
        async def _call_api(self, model_name, messages, **kwargs):
            response = await super()._call_api(model_name, messages, **kwargs)
            if len(self.inputs) == 1:
                response.content.insert(0, TextBlock(text="Before lookup."))
            return response

    async def lookup(query: str) -> str:
        """Look up a drink."""
        await agent.observe(UserMsg(name="observer", content="Lookup completed."))
        return "Tea"

    client, requests = make_client()
    state = AgentState(session_id="session-1")
    state.permission_context.mode = PermissionMode.BYPASS
    agent = Agent(name="qwenpaw", system_prompt="Help.", model=SpeakingModel(tool=True),
                  state=state, toolkit=Toolkit(tools=[FunctionTool(lookup)]),
                  middlewares=[plugin.OGMemoryMiddleware(client, context())])
    _ = [item async for item in agent.reply_stream(inputs=UserMsg(name="alice", content="Choose tea"))]
    await client.close()
    write = next(body for path, body, _ in requests if path == "/api/v1/after_turn")
    text = "\n".join(msg["content"] for msg in write["messages"] if msg["role"] == "assistant")
    assert text == "Before lookup.\nTea selected."
    turn = next(body["event"] for path, body, _ in requests
                if path == "/api/v1/trace/events" and body["event"]["event_type"] == "turn_end")
    assert turn["assistant_message"] == text


@pytest.mark.asyncio
async def test_deep_json_trace_allows_agent_reply_to_complete():
    client, requests = make_client()
    agent = Agent(name="qwenpaw", system_prompt="Help.", model=Model(),
                  state=AgentState(session_id="session-1"),
                  middlewares=[plugin.OGMemoryMiddleware(client, context())])
    nested = "[" * 10000 + "0" + "]" * 10000
    _ = [item async for item in agent.reply_stream(inputs=UserMsg(name="alice", content=nested))]
    await client.close()
    prompt = next(body["event"]["prompt"] for path, body, _ in requests
                  if path == "/api/v1/trace/events" and body["event"]["event_type"] == "user_message")
    assert prompt == "[DEPTH_LIMIT]"
    write = next(body for path, body, _ in requests if path == "/api/v1/after_turn")
    assert write["messages"][-1]["content"] == "Tea selected."


@pytest.mark.asyncio
async def test_deep_json_tool_trace_allows_agent_reply_to_complete():
    async def lookup(query: str) -> str:
        """Return a deeply nested tool result."""
        return "["*10000 + "0" + "]"*10000

    client, requests = make_client()
    state = AgentState(session_id="session-1")
    state.permission_context.mode = PermissionMode.BYPASS
    agent = Agent(name="qwenpaw", system_prompt="Help.", model=Model(tool=True), state=state,
                  toolkit=Toolkit(tools=[FunctionTool(lookup)]),
                  middlewares=[plugin.OGMemoryMiddleware(client, context())])
    _ = [item async for item in agent.reply_stream(inputs=UserMsg(name="alice", content="Lookup"))]
    await client.close()
    result = next(body["event"]["tool_response"] for path, body, _ in requests
                  if path == "/api/v1/trace/events" and body["event"]["event_type"] == "tool_success")
    assert result == "[DEPTH_LIMIT]"
    write = next(body for path, body, _ in requests if path == "/api/v1/after_turn")
    assert write["messages"][-1]["content"] == "Tea selected."


@pytest.mark.asyncio
async def test_shutdown_delivers_end_events_beyond_queue_capacity():
    client, requests = make_client()
    for index in range(300):
        await client.ensure_session(client.identity(context(f"session-{index}")))
    await client._traces.join()
    await client.close()
    ended = [body["session_id"] for path, body, _ in requests
             if path == "/api/v1/trace/events" and body["event"]["event_type"] == "session_end"]
    assert ended == [f"session-{index}" for index in range(300)]


@pytest.mark.asyncio
async def test_message_identity_survives_copy_and_preserves_repeated_text():
    from agentscope.message import AssistantMsg

    client, requests = make_client()
    middleware = plugin.OGMemoryMiddleware(client, context())
    messages = [AssistantMsg(id="shared", name="qwenpaw", content="Same text.", created_at=created)
                for created in ("2026-01-01T00:00:00.000001", "2026-01-01T00:00:00.000002")]
    middleware.capture(messages)
    middleware.capture([msg.model_copy(deep=True) for msg in messages])
    await client.close()
    assert list(middleware.messages.values()) == [{"role": "assistant", "content": "Same text."}]*2
    traces = [body for path, body, _ in requests if path == "/api/v1/trace/events"]
    assert len(traces) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("text, expected", [
    ('headers: {"Authorization":"Basic dXNlcjpwYXNz"}', 'headers: {"[REDACTED]'),
    ('```json\n{"Cookie":"session=private-cookie"}\n```', '```json\n{"[REDACTED]\n```'),
    ("headers: {'Proxy-Authorization':'Basic dXNlcjpwYXNz'}", "headers: {'[REDACTED]"),
    ('headers: {"Set-Cookie":"session=private-cookie"}', 'headers: {"[REDACTED]'),
])
async def test_quoted_headers_in_tool_output_are_redacted(text, expected):
    client, requests = make_client()
    client.emit(client.identity(context()), "tool_success", "headers", tool_response=text)
    await client.close()
    trace = next(body for path, body, _ in requests if path == "/api/v1/trace/events")
    assert trace["event"]["tool_response"] == expected


@pytest.mark.asyncio
async def test_observed_user_input_is_written_once_per_turn():
    client, requests = make_client()
    state = AgentState(session_id="session-1")
    for prompt in ("First observed question", "Second observed question"):
        agent = Agent(name="qwenpaw", system_prompt="Help.", model=Model(), state=state,
                      middlewares=[plugin.OGMemoryMiddleware(client, context())])
        await agent.observe(UserMsg(name="alice", content=prompt,
                                    metadata={"qwenpaw_tag": "external_user_query"}))
        _ = [item async for item in agent.reply_stream()]
        state = AgentState.model_validate_json(state.model_dump_json())
    await client.close()
    writes = [body["messages"] for path, body, _ in requests if path == "/api/v1/after_turn"]
    assert writes == [
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": "Tea selected."}]
        for prompt in ("First observed question", "Second observed question")
    ]
    prompts = [body["event"]["prompt"] for path, body, _ in requests
               if path == "/api/v1/trace/events" and body["event"]["event_type"] == "user_message"]
    assert prompts == ["First observed question", "Second observed question"]


@pytest.mark.asyncio
@pytest.mark.parametrize("confirmed", [True, False])
async def test_confirmation_resume_finishes_one_logical_turn(confirmed):
    from agentscope.event import RequireUserConfirmEvent, UserConfirmResultEvent

    from tests.unit.fs.test_agfs_trace_store import _store

    async def lookup(query: str) -> str:
        """Look up the requested drink."""
        return "Tea is available."

    client, requests = make_client()
    model = Model(tool=True)
    state = AgentState(session_id="session-1")
    agent = Agent(name="qwenpaw", system_prompt="Help.", model=model, state=state,
                  toolkit=Toolkit(tools=[FunctionTool(lookup)]),
                  middlewares=[plugin.OGMemoryMiddleware(client, context())])
    user = UserMsg(name="alice", content="Choose a drink.",
                   metadata={"qwenpaw_tag": "external_user_query"})
    events = [item async for item in agent.reply_stream(inputs=user)]
    confirmation = next(item for item in events if isinstance(item, RequireUserConfirmEvent))
    _ = [item async for item in agent.reply_stream(inputs=UserConfirmResultEvent(
        reply_id=confirmation.reply_id, confirm_results=[],
    ))]
    state = AgentState.model_validate_json(state.model_dump_json())
    resumed = Agent(name="qwenpaw", system_prompt="Help.", model=model, state=state,
                    toolkit=Toolkit(tools=[FunctionTool(lookup)]),
                    middlewares=[plugin.OGMemoryMiddleware(client, context())])
    _ = [item async for item in resumed.reply_stream(inputs=UserConfirmResultEvent(
        reply_id=confirmation.reply_id,
        confirm_results=[{"confirmed": confirmed, "tool_call": call} for call in confirmation.tool_calls],
    ))]
    await client.close()
    traces = [body for path, body, _ in requests if path == "/api/v1/trace/events"]
    store = _store()
    common = {"account_id": "test-account", "owner_space": "user:alice",
              "agent_id": "qwenpaw", "session_id": "session-1"}
    for body in traces:
        event = body["event"]
        saved = store.insert(**common, event_id=body["request_id"],
                             hook_type=event["event_type"], content=event)
        if event["event_type"] == "turn_end":
            finalized = store.finalize_session(**common, stop_event_index=saved["event_index"])
    turn = store._read_state("test-account", collections=("turns",),
                             session_id="session-1")["turns"][finalized["uri"]]
    assert turn["content"][-1]["assistant_message"] == "Tea selected."
    terminal_tool = "tool_success" if confirmed else "tool_failure"
    assert terminal_tool in [event["event_type"] for event in turn["content"]]
    ends = [body for body in traces if body["event"]["event_type"] == "turn_end"]
    assert len(ends) == 1
    assert ends[0]["event"]["assistant_message"] == "Tea selected."
    kinds = [body["event"]["event_type"] for body in traces]
    assert kinds.index("permission_request") < kinds.index(terminal_tool) < kinds.index("turn_end")
    assert [body["event"]["prompt"] for body in traces
            if body["event"]["event_type"] == "user_message"] == [user.get_text_content()]


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt", ["请记住：Atlas 每周四发布。", "Remember: Atlas releases on Thursday."])
async def test_explicit_memory_request_extracts_below_threshold(prompt):
    from dataclasses import replace

    from core.models import RequestContext
    from retrieval.context_reader import ContextReader
    from tests.unit.commit.test_memory_identity import MemoryFS
    from tests.unit.server.test_extraction_batch_recovery import candidate, setup_service

    ctx = RequestContext(account_id="test-account", user_id="alice", agent_id="qwenpaw",
                         session_id="session-1", trace_id="short-request")
    fs = MemoryFS()
    service, writer, _ = setup_service(fs, ctx, entries=[candidate(body=prompt)])
    service._cfg.after_turn_threshold = 400
    results = []

    def handle(request):
        body = json.loads(request.content)
        result = {"ok": True}
        if request.url.path == "/api/v1/after_turn":
            result = service.after_turn({**body, "_ctx": ctx})
            results.append(result)
        return httpx.Response(200, json=result)

    client = plugin.MemoryClient({"account_id": ctx.account_id}, transport=httpx.MockTransport(handle))
    agent = Agent(name="qwenpaw", system_prompt="Help.", model=Model(),
                  state=AgentState(session_id="session-1"),
                  middlewares=[plugin.OGMemoryMiddleware(client, context())])
    _ = [item async for item in agent.reply_stream(inputs=UserMsg(name="alice", content=prompt))]
    await client.close()
    assert results[0]["status"] == "completed"
    writer._pipeline.extract.assert_called_once()
    new_ctx = replace(ctx, session_id="session-2")
    saved = [node for node in fs.nodes.values() if node.category == "event"]
    assert len(saved) == 1
    assert ContextReader(fs).read(saved[0].uri, new_ctx).content_excerpt == prompt

@pytest.mark.asyncio
@pytest.mark.parametrize("during_reply", [False, True])
async def test_compaction_extracts_accumulated_short_turns(during_reply):
    from core.models import RequestContext
    from tests.unit.commit.test_memory_identity import MemoryFS
    from tests.unit.server.test_extraction_batch_recovery import candidate, setup_service

    ctx = RequestContext(account_id="test-account", user_id="alice", agent_id="qwenpaw",
                         session_id="session-1", trace_id="compact")
    fs = MemoryFS()
    service, writer, manager = setup_service(fs, ctx, entries=[candidate(body="Atlas ships Thursday.")])
    service._cfg.after_turn_threshold = 10000
    calls = []

    def handle(request):
        body = json.loads(request.content)
        result = {"ok": True}
        if request.url.path == "/api/v1/after_turn":
            result = service.after_turn({**body, "_ctx": ctx})
        elif request.url.path == "/api/v1/compact":
            result = service.compact({**body, "_ctx": ctx})
        calls.append((request.url.path, body, result))
        return httpx.Response(200, json=result)

    client = plugin.MemoryClient({"account_id": ctx.account_id}, transport=httpx.MockTransport(handle))
    state = AgentState(session_id="session-1")
    middleware = plugin.OGMemoryMiddleware(client, context())
    agent = Agent(name="qwenpaw", system_prompt="Help.", model=Model(), state=state,
                  middlewares=[middleware])
    _ = [item async for item in agent.reply_stream(inputs=UserMsg(
        name="alice", content="Atlas ships Thursday.", metadata={"qwenpaw_tag": "external_user_query"},
    ))]
    assert next(result for path, _, result in calls if path.endswith("after_turn"))["status"] == "accumulating"

    async def compress(**kwargs):
        state.context = [UserMsg(name="summary", content="Atlas schedule discussed.")]

    if during_reply:
        async def reply(**kwargs):
            await middleware.on_compress_context(agent, {}, compress)
            from agentscope.event import ReplyEndEvent
            yield ReplyEndEvent(session_id="session-1", reply_id="reply", finished_reason="completed")
        _ = [item async for item in middleware.on_reply(agent, {}, reply)]
    else:
        state = AgentState.model_validate_json(state.model_dump_json())
        agent.state = state
        middleware = plugin.OGMemoryMiddleware(client, context())
        await middleware.on_compress_context(agent, {}, compress)
    await client.close()
    writer._pipeline.extract.assert_called_once()
    assert [node.content for node in fs.nodes.values() if node.category == "event"] == ["Atlas ships Thursday."]
    assert sum(path.endswith("after_turn") for path, _, _ in calls) == 1
    assert sum(path.endswith("compact") for path, _, _ in calls) == 1

@pytest.mark.asyncio
async def test_compaction_retries_after_service_recovers():
    from agentscope.message import AssistantMsg

    calls = []
    available = False

    def handle(request):
        calls.append(request.url.path)
        if request.url.path.endswith("compact") and not available:
            raise httpx.ConnectError("Service reconnecting", request=request)
        return httpx.Response(200, json={"ok": True})

    client = plugin.MemoryClient(transport=httpx.MockTransport(handle))
    state = AgentState(session_id="session-1", context=[
        UserMsg(name="alice", content="Atlas ships Thursday.", metadata={"qwenpaw_tag": "external_user_query"}),
        AssistantMsg(name="qwenpaw", content="Understood."),
    ])
    middleware = plugin.OGMemoryMiddleware(client, context())
    agent = Agent(name="qwenpaw", system_prompt="Help.", model=Model(), state=state,
                  middlewares=[middleware])

    async def compress(**kwargs):
        state.context = [UserMsg(name="summary", content="Atlas schedule discussed.")]

    await middleware.on_compress_context(agent, {}, compress)
    assert state.middle_context["ogmemory"]["compact_retry"] is True
    available = True
    _ = [item async for item in agent.reply_stream(inputs=UserMsg(name="alice", content="Continue."))]
    await client.close()
    assert state.middle_context["ogmemory"].get("compact_pending", False) is False
    assert state.middle_context["ogmemory"].get("compact_retry", False) is False
    assert calls.count("/api/v1/compact") == 2

@pytest.mark.asyncio
async def test_manual_compact_dispatch_extracts_before_command(tmp_path):
    from qwenpaw.app.chats.session import SafeJSONSession
    from qwenpaw.runtime.hooks import HookRegistry
    from qwenpaw.runtime.phases import Phase
    from qwenpaw.runtime.slash_command_registry import CommandSpec, SlashCommandRegistry

    calls = []

    def handle(request):
        calls.append(request.url.path)
        return httpx.Response(200, json={"ok": True})

    integration = plugin.OGMemoryPlugin()
    hooks = HookRegistry()
    api = SimpleNamespace(config={}, register_runtime_hook=hooks.register,
        register_middleware=lambda factory: None,
        register_shutdown_hook=lambda *args: None, register_uninstall_hook=lambda *args: None)
    integration.register(api)
    await integration.client.http.aclose()
    integration.client.http = httpx.AsyncClient(base_url="http://memory", transport=httpx.MockTransport(handle))
    registry = SlashCommandRegistry()

    async def compact(ctx, args):
        calls.append("native_compact")
        return UserMsg(name="system", content="Compacted.")

    registry.register(CommandSpec(name="compact", handler=compact, category="conversation"))
    ctx = context()
    ctx.input_msgs = [UserMsg(name="alice", content="/compact keep decisions")]
    session = SafeJSONSession(str(tmp_path))
    await session.update_session_state(session_id=ctx.session_id, user_id="alice",
        key="agent.state", value=AgentState().model_dump(mode="json"))
    ctx.workspace = SimpleNamespace(session=session,
        plugins=SimpleNamespace(slash_command_registry=registry))
    await hooks.run(Phase.PRE_DISPATCH, ctx)
    await registry.dispatch(ctx.input_msgs[0].get_text_content(), ctx)
    await integration.client.close()
    assert calls.index("/api/v1/compact") < calls.index("native_compact")


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_path", ["bootstrap", "after_turn", "compact"])
async def test_manual_compact_preserves_retry_across_restart(tmp_path, failure_path):
    from agentscope.agent import ContextConfig
    from agentscope.message import AssistantMsg
    from qwenpaw.app.chats.session import SafeJSONSession
    from qwenpaw.runtime.hooks import HookRegistry
    from qwenpaw.runtime.phases import Phase
    from qwenpaw.runtime.slash_command_registry import CommandSpec, SlashCommandRegistry

    calls = []
    available = False

    def handle(request):
        body = json.loads(request.content)
        calls.append((request.url.path, body))
        if not available and request.url.path in {f"/api/v1/{failure_path}", "/api/v1/compact"}:
            return httpx.Response(500, json={"ok": False, "error": "Extraction unavailable"})
        return httpx.Response(200, json={"ok": True})

    async def install():
        integration = plugin.OGMemoryPlugin()
        hooks = HookRegistry()
        api = SimpleNamespace(config={}, register_runtime_hook=hooks.register,
            register_middleware=lambda factory: None, register_shutdown_hook=lambda *args: None,
            register_uninstall_hook=lambda *args: None)
        integration.register(api)
        await integration.client.http.aclose()
        integration.client.http = httpx.AsyncClient(base_url="http://memory",
            transport=httpx.MockTransport(handle))
        return integration, hooks

    session = SafeJSONSession(str(tmp_path))
    ctx = context()
    ctx.request.channel = "console"
    ctx.input_msgs = [UserMsg(name="alice", content="/compact")]
    session_args = {"session_id": ctx.session_id, "user_id": "alice", "channel": "console"}
    original = "Atlas releases Thursday."
    batch = {"clientRequestId": "original-turn", "messages": [{"role": "user", "content": original}],
             "forceExtract": False}
    state = AgentState(session_id=ctx.session_id, context=[
        UserMsg(name="alice", content=original, metadata={"qwenpaw_tag": "external_user_query"}),
        AssistantMsg(name="qwenpaw", content="Understood."),
    ], middle_context={"ogmemory": {"pending": [batch]}})
    await session.update_session_state(**session_args, key="agent", value={
        "state": state.model_dump(mode="json"), "mode_state": {"coding": {"active": True}},
        "scroll": {"checkpoint": "saved"},
    })

    class SummaryModel(Model):
        async def generate_structured_output(self, messages, structured_model, **kwargs):
            return SimpleNamespace(finished_reason="stop",
                content={key: original for key in structured_model["properties"]})

    async def compact(ctx, args):
        saved = await session.get_session_state_dict(**session_args)
        restored = AgentState.model_validate(saved["agent"]["state"])
        agent = Agent(name="qwenpaw", system_prompt="Help.", model=SummaryModel(), state=restored)
        cfg = ContextConfig().model_copy(update={"trigger_ratio": 0, "reserve_ratio": 0})
        await agent.compress_context(cfg)
        await session.update_session_state(**session_args, key="agent.state",
            value=restored.model_dump(mode="json"))
        return AssistantMsg(name="qwenpaw", content="Compact complete.")

    registry = SlashCommandRegistry()
    registry.register(CommandSpec(name="compact", handler=compact, category="conversation"))
    ctx.workspace = SimpleNamespace(session=session,
        plugins=SimpleNamespace(slash_command_registry=registry))
    integration, hooks = await install()
    await hooks.run(Phase.PRE_DISPATCH, ctx)
    result = await registry.dispatch("/compact", ctx)
    assert result.get_text_content() == "Compact complete."
    saved = await session.get_session_state_dict(**session_args)
    assert original in saved["agent"]["state"]["summary"]
    retry_state = saved["agent"]["state"]["middle_context"]["ogmemory"]
    assert retry_state.get("compact_pending") or retry_state.get("compact_retry")
    assert saved["agent"]["mode_state"] == {"coding": {"active": True}}
    assert saved["agent"]["scroll"] == {"checkpoint": "saved"}
    if failure_path == "after_turn":
        assert saved["agent"]["state"]["middle_context"]["ogmemory"]["pending"] == [batch]
    await integration.client.close()

    available = True
    integration, _ = await install()
    session = SafeJSONSession(str(tmp_path))
    saved = await session.get_session_state_dict(**session_args)
    restored = AgentState.model_validate(saved["agent"]["state"])
    agent = Agent(name="qwenpaw", system_prompt="Help.", model=Model(), state=restored,
                  middlewares=[plugin.OGMemoryMiddleware(integration.client, ctx)])
    _ = [item async for item in agent.reply_stream(inputs=UserMsg(name="alice", content="Continue."))]
    await session.update_session_state(**session_args, key="agent.state",
        value=restored.model_dump(mode="json"))
    saved = await session.get_session_state_dict(**session_args)
    retry_state = saved["agent"]["state"]["middle_context"]["ogmemory"]
    assert retry_state.get("compact_pending", False) is False
    assert retry_state.get("compact_retry", False) is False
    assert retry_state["pending"] == []
    assert restored.context[-1].get_text_content() == "Tea selected."
    recovered_paths = [path for path, _ in calls if path in {"/api/v1/after_turn", "/api/v1/compact"}]
    assert recovered_paths[-1] == ("/api/v1/compact" if failure_path == "after_turn" else "/api/v1/after_turn")
    replayed = [{key: body[key] for key in batch} for path, body in calls if path.endswith("after_turn")
                and body["clientRequestId"] == "original-turn"]
    assert replayed == [batch] * (2 if failure_path == "after_turn" else 1)
    await integration.client.close()

@pytest.mark.asyncio
async def test_compact_recovery_finishes_service_batch_before_new_turn():
    from dataclasses import replace

    from core.errors import ExtractionIncompleteError
    from core.models import RequestContext
    from service.lifecycle_checkpoint import lifecycle_kind
    from tests.unit.commit.test_memory_identity import MemoryFS
    from tests.unit.server.test_extraction_batch_recovery import candidate, setup_service

    ctx = RequestContext(account_id="test-account", user_id="alice", agent_id="qwenpaw",
                         session_id="session-1", trace_id="compact-recovery")
    fs = MemoryFS()
    available = False

    def extract(**kwargs):
        if not available:
            raise ExtractionIncompleteError("Extraction service recovering")
        return [replace(candidate(body="Atlas ships Thursday."),
                        provenance_ids=[f"prov:1:archive:{kwargs['archive_id']}:msg-one"])]

    service, _, manager = setup_service(fs, ctx, extract=extract)
    service._cfg.after_turn_threshold = 10000
    service.after_turn({"sessionId": ctx.session_id, "clientRequestId": "original",
        "messages": [{"role": "user", "content": "Atlas ships Thursday."}], "_ctx": ctx})
    calls = []

    def handle(request):
        body = json.loads(request.content)
        try:
            result = (service.compact({**body, "_ctx": ctx}) if request.url.path.endswith("compact")
                      else service.after_turn({**body, "_ctx": ctx}))
            response = httpx.Response(200, json=result)
        except ExtractionIncompleteError as exc:
            response = httpx.Response(500, json={"ok": False, "error": str(exc)})
        calls.append((request.url.path, response.json()))
        return response

    client = plugin.MemoryClient(transport=httpx.MockTransport(handle))
    identity = client.identity(context())
    state = {"pending": [], "compact_pending": True}
    await plugin.flush_pending(client, identity, state)
    assert state["compact_retry"] is True
    assert lifecycle_kind(manager.get_or_create(ctx.session_id, ctx=ctx)) == "compaction"

    state = json.loads(json.dumps(state))
    state["pending"].append({"clientRequestId": "next-turn",
                            "messages": [{"role": "user", "content": "Continue."}]})
    available = True
    await plugin.flush_pending(client, identity, state)
    assert [path for path, _ in calls] == ["/api/v1/compact", "/api/v1/compact", "/api/v1/after_turn"]
    assert calls[1][1]["compacted"] is True
    assert calls[2][1]["status"] == "accumulating"
    assert state == {"pending": []}
    assert [node.content for node in fs.nodes.values() if node.category == "event"] == ["Atlas ships Thursday."]
    await client.close()


@pytest.mark.asyncio
async def test_native_agent_compression_flushes_original_text():
    from agentscope.agent import ContextConfig
    from agentscope.message import AssistantMsg

    class SummaryModel(Model):
        async def generate_structured_output(self, messages, structured_model, **kwargs):
            return SimpleNamespace(finished_reason="stop",
                content={key: "Atlas schedule" for key in structured_model["properties"]})

    client, requests = make_client()
    state = AgentState(session_id="session-1", context=[
        UserMsg(name="alice", content="Atlas ships Thursday.", metadata={"qwenpaw_tag": "external_user_query"}),
        AssistantMsg(name="qwenpaw", content="Understood."),
    ])
    agent = Agent(name="qwenpaw", system_prompt="Help.", model=SummaryModel(), state=state,
                  middlewares=[plugin.OGMemoryMiddleware(client, context())])
    cfg = ContextConfig().model_copy(update={"trigger_ratio": 0, "reserve_ratio": 0})
    await agent.compress_context(cfg)
    await client.close()
    assert state.summary
    writes = [body for path, body, _ in requests if path.endswith("after_turn")]
    assert writes[0]["messages"][0]["content"] == "Atlas ships Thursday."
    assert sum(path.endswith("compact") for path, _, _ in requests) == 1
