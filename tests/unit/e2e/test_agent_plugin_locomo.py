from __future__ import annotations

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tests" / "e2e"))

from agent_plugin_locomo import ReadOnlyQaProxy  # noqa: E402
from eval import build_parser  # noqa: E402
from run_claude_code_plugin_locomo import ClaudeCodeHost  # noqa: E402
from run_opencode_plugin_locomo import OpenCodeHost  # noqa: E402
from run_qwenpaw_plugin_locomo import QwenPawHost  # noqa: E402


@pytest.mark.parametrize("reply_fields", [{"reply_id": "current"}, {"reply_context": {"reply_id": "current"}}])
def test_qwenpaw_usage_in_independent_runner_environment(tmp_path, monkeypatch, reply_fields):
    import builtins

    args = build_parser("qwenpaw").parse_args(["qa", "locomo.json", "--output-dir", str(tmp_path)])
    args.llm_api_key = args.ogmemory_api_key = ""
    args.user_id = "alice:team"
    host = QwenPawHost(args)
    host.state = tmp_path / "data"
    path = host.state / "workspaces/default/sessions/console/alice--team_qp-usage.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"agent": {"state": {**reply_fields, "context": [
        {"id": "previous", "role": "assistant", "usage": {"input_tokens": 999, "output_tokens": 99}},
        {"id": "current", "role": "assistant", "usage": {"input_tokens": 100, "output_tokens": 10}},
        {"id": "current", "role": "assistant", "usage": {"input_tokens": 200, "output_tokens": 20}},
    ]}}}))
    original_import = builtins.__import__

    def independent_import(name, *args, **kwargs):
        if name.split(".")[0] in {"qwenpaw", "agentscope"}:
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", independent_import)
    assert host._turn_usage("qp-usage") == {"input_tokens": 300, "output_tokens": 30, "total_tokens": 330}


def test_qwenpaw_host_uses_native_task_session_and_answer(tmp_path, monkeypatch):
    pytest.importorskip("qwenpaw")
    from qwenpaw.app.chats.session import session_filename

    args = build_parser("qwenpaw").parse_args(
        ["qa", "locomo.json", "--output-dir", str(tmp_path)]
    )
    args.llm_api_key = args.ogmemory_api_key = ""
    host = QwenPawHost(args)
    host.http = requests
    host.state = tmp_path / "data"
    session_path = host.state / "workspaces/default/sessions/console" / session_filename(
        "qp-session", args.user_id
    )
    session_path.parent.mkdir(parents=True)
    session_path.write_text(json.dumps({"agent": {"state": {
        "reply_id": "reply-1", "context": [{"role": "assistant", "id": "reply-1",
        "name": "qwenpaw", "content": [{"type": "text", "text": "Answer"}],
        "usage": {"input_tokens": 30, "output_tokens": 5}}],
    }}}))
    host.base = "http://127.0.0.1:18891/api/agents/default/console"
    calls = []

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self.payload

    def post(url, **kwargs):
        calls.append(kwargs["json"])
        return Response({"task_id": "task-1"})

    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(requests, "get", lambda *a, **k: Response({
        "status": "finished", "result": {"status": "completed", "session_id": "qp-session",
        "output": [{"role": "assistant", "content": [{"type": "text", "text": "May 7, 2023."}]}],
        "usage": {"input_tokens": 30, "output_tokens": 5}},
    }))
    answer, usage = host.invoke("When?", "question-1")
    assert answer == "May 7, 2023."
    assert usage == {"input_tokens": 30, "output_tokens": 5, "total_tokens": 35}
    assert host.sessions["question-1"] == "qp-session"
    host.invoke("Why?", "question-1")
    assert calls[-1]["session_id"] == "qp-session"


def test_qwenpaw_is_available_in_shared_eval_parser():
    args = build_parser().parse_args(["ingest", "locomo.json", "--agent", "qwenpaw"])
    assert args.qwenpaw_command == "qwenpaw"
    assert args.qwenpaw_port == 18891


@pytest.mark.asyncio
async def test_qwenpaw_usage_includes_all_tool_loop_model_calls(tmp_path):
    pytest.importorskip("qwenpaw")
    from agentscope.agent import Agent
    from agentscope.formatter import OpenAIChatFormatter
    from agentscope.message import TextBlock, ToolCallBlock, UserMsg
    from agentscope.model import ChatModelBase, ChatResponse, ChatUsage
    from agentscope.permission import PermissionMode
    from agentscope.state import AgentState
    from agentscope.tool import FunctionTool, Toolkit
    from qwenpaw.app.chats.session import session_filename
    from qwenpaw.runtime.envelope import Envelope

    class UsageModel(ChatModelBase):
        def __init__(self):
            super().__init__(credential=None, model="fixture", parameters=self.Parameters(), stream=False)
            self.calls = 0
            self.formatter = OpenAIChatFormatter()

        async def _call_api(self, model_name, messages, **kwargs):
            self.calls += 1
            content = ([ToolCallBlock(id="tool-1", name="lookup", input='{"query":"tea"}')]
                       if self.calls == 1 else [TextBlock(text="Tea selected.")])
            return ChatResponse(content=content, is_last=True,
                                usage=ChatUsage(input_tokens=100*self.calls,
                                                output_tokens=10*self.calls, time=0.01))

    async def lookup(query: str) -> str:
        """Look up a drink."""
        return "Tea"

    state = AgentState(session_id="qp-usage")
    state.permission_context.mode = PermissionMode.BYPASS
    agent = Agent(name="qwenpaw", system_prompt="Help.", model=UsageModel(), state=state,
                  toolkit=Toolkit(tools=[FunctionTool(lookup)]))
    envelope = Envelope(session_id="qp-usage")
    async for event in agent.reply_stream(inputs=UserMsg(name="alice", content="Choose tea")):
        _ = [item async for item in envelope.translate_event(event)]
    output = [item async for item in envelope.finalize()]
    result = output[-1].model_dump(mode="json")
    assert result["usage"] == {"input_tokens": 200, "output_tokens": 20}

    args = build_parser("qwenpaw").parse_args(["qa", "locomo.json", "--output-dir", str(tmp_path)])
    args.llm_api_key = args.ogmemory_api_key = ""
    args.user_id = "alice"
    host = QwenPawHost(args)
    host.state = tmp_path / "data"
    session_path = host.state / "workspaces/default/sessions/console" / session_filename("qp-usage", "alice")
    session_path.parent.mkdir(parents=True)
    session_path.write_text(json.dumps({"agent": {"state": state.model_dump(mode="json")}}))

    class Handler(BaseHTTPRequestHandler):
        def reply(self, data):
            payload = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.reply({"task_id": "task-usage"})

        def do_GET(self):
            self.reply({"status": "finished", "result": result})

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host.http = requests.Session()
    host.http.trust_env = False
    host.base = f"http://127.0.0.1:{server.server_port}"
    try:
        text, usage = host.invoke("Choose tea", "usage")
        assert text == "Tea selected."
        assert usage == {"input_tokens": 300, "output_tokens": 30, "total_tokens": 330}
    finally:
        host.close()
        server.shutdown()
        server.server_close()
        thread.join()


def test_opencode_preserves_tool_steps_and_final_answer_usage_and_session(tmp_path, monkeypatch) -> None:
    args = build_parser("opencode").parse_args(
        ["ingest", "locomo.json", "--output-dir", str(tmp_path)]
    )
    args.llm_api_key = args.ogmemory_api_key = ""
    host = OpenCodeHost(args)
    session_id = "ses_locomo_tool_step"
    events = [
        {"type": "step_start", "sessionID": session_id, "part": {"type": "step-start"}},
        {"type": "tool_use", "sessionID": session_id,
         "part": {"type": "tool", "tool": "read", "state": {"status": "completed"}}},
        {"type": "step_finish", "sessionID": session_id,
         "part": {"type": "step-finish", "reason": "tool-calls",
                  "tokens": {"input": 17, "output": 3}}},
        {"type": "step_start", "sessionID": session_id, "part": {"type": "step-start"}},
        {"type": "text", "sessionID": session_id,
         "part": {"type": "text", "text": "The answer is Paris."}},
        {"type": "step_finish", "sessionID": session_id,
         "part": {"type": "step-finish", "reason": "stop",
                  "tokens": {"input": 11, "output": 5}}},
    ]
    monkeypatch.setattr(
        host, "_run", lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, "\n".join(json.dumps(event) for event in events), ""
        )
    )

    reply, usage = host.invoke("Conversation transcript", "ingest-conv-26")

    assert reply == "The answer is Paris."
    assert usage == {
        "input_tokens": 28, "output_tokens": 8, "total_tokens": 36,
        "cacheRead": 0, "cacheWrite": 0,
    }
    assert host.sessions["ingest-conv-26"] == session_id
    assert json.loads(host.session_file.read_text()) == {"ingest-conv-26": session_id}


@pytest.mark.parametrize("answer,tokens,expected", [
    pytest.param(
        "May 7, 2023.",
        {"total": 12612, "input": 12602, "output": 10, "reasoning": 0,
         "cache": {"write": 0, "read": 0}},
        {"input_tokens": 12602, "output_tokens": 10, "total_tokens": 12612,
         "cacheRead": 0, "cacheWrite": 0},
        id="sample0-q1-uncached",
    ),
    pytest.param(
        "In 2022 (she called it “last year” in the May 8, 2023 chat).",
        {"total": 9353, "input": 6454, "output": 27, "reasoning": 0,
         "cache": {"write": 0, "read": 2872}},
        {"input_tokens": 9326, "output_tokens": 27, "total_tokens": 9353,
         "cacheRead": 2872, "cacheWrite": 0},
        id="sample0-q2-cache-hit",
    ),
])
def test_opencode_normalizes_successful_native_usage(tmp_path, monkeypatch, answer, tokens, expected):
    args = build_parser("opencode").parse_args(
        ["qa", "locomo.json", "--output-dir", str(tmp_path)]
    )
    args.llm_api_key = args.ogmemory_api_key = ""
    host = OpenCodeHost(args)
    events = [
        {"type": "text", "sessionID": "session-a", "part": {"type": "text", "text": answer}},
        {"type": "step_finish", "sessionID": "session-a",
         "part": {"type": "step-finish", "reason": "stop", "tokens": tokens}},
    ]
    monkeypatch.setattr(host, "_run", lambda *_args, **_kwargs: subprocess.CompletedProcess(
        [], 0, "\n".join(json.dumps(event) for event in events), "",
    ))

    assert host.invoke("Answer the question directly", "qa-a-q1") == (answer, expected)
    assert host.sessions["qa-a-q1"] == "session-a"


def test_opencode_delivers_verbatim_prompt_through_stdin(tmp_path, monkeypatch) -> None:
    args = build_parser("opencode").parse_args(
        ["ingest", "locomo.json", "--output-dir", str(tmp_path)]
    )
    args.llm_api_key = args.ogmemory_api_key = "test"
    host = OpenCodeHost(args)
    prompt = 'Conversation transcript:\nCaroline: "I moved to Paris."\nPath: C:\\notes\\travel\n'
    calls = []

    def execute(command, **options):
        calls.append((command, options["input"]))
        return subprocess.CompletedProcess(command, 0, json.dumps({
            "type": "text", "sessionID": "session-a", "part": {"text": "saved"},
        }), "")

    monkeypatch.setattr("agent_plugin_locomo.subprocess.run", execute)
    assert host.invoke(prompt, "ingest-a")[0] == "saved"
    assert calls == [([
        args.opencode_command, "run", "--format", "json", "--model",
        f"locomo/{args.llm_model}", "--dir", str(host.runtime / "workspace"),
    ], prompt)]
    assert host.sessions["ingest-a"] == "session-a"


def test_claude_supports_separate_anthropic_endpoint(tmp_path: Path, monkeypatch) -> None:
    args = build_parser("claude-code").parse_args(
        [
            "ingest",
            "locomo.json",
            "--output-dir",
            str(tmp_path),
            "--llm-base-url",
            "https://provider.example/openai/v1",
            "--claude-base-url",
            "https://provider.example/anthropic",
            "--claude-auth-token",
        ]
    )
    args.llm_api_key = "test-key"
    args.ogmemory_api_key = ""
    host = ClaudeCodeHost(args)
    monkeypatch.setattr(host, "_require", lambda _executable: None)
    monkeypatch.setattr(
        host,
        "_run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, "claude 1.0\n", ""),
    )

    host.prepare()

    assert host.env["ANTHROPIC_BASE_URL"] == "https://provider.example/anthropic"
    assert host.env["ANTHROPIC_AUTH_TOKEN"] == "test-key"
    assert "ANTHROPIC_API_KEY" not in host.env


def test_shared_evaluation_defaults() -> None:
    args = build_parser("hermes").parse_args(["qa", "locomo.json", "--output-dir", "results"])

    assert args.mode == "qa"
    assert args.sample is None
    assert args.user_id == "eval-1"
    assert args.qa_parallel == 2
    assert args.agent_attempts == 3
    assert args.rate_limit_retry_delay == 30


def test_count_must_be_positive() -> None:
    parser = build_parser("dsh")

    with pytest.raises(SystemExit):
        parser.parse_args(["qa", "locomo.json", "--output-dir", "results", "--count", "0"])


def test_read_only_qa_proxy_forwards_compose_and_suppresses_writeback() -> None:
    calls: list[str] = []

    class TargetHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            calls.append(self.path)
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            payload = b'{"retrievedEvidence":"memory"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, _format: str, *_args) -> None:
            return

    target = ThreadingHTTPServer(("127.0.0.1", 0), TargetHandler)
    target_thread = threading.Thread(target=target.serve_forever, daemon=True)
    target_thread.start()
    proxy = ReadOnlyQaProxy(f"http://127.0.0.1:{target.server_port}")
    proxy.start()
    proxy.read_only = True
    try:
        compose = requests.post(f"{proxy.url}/api/v1/compose", json={"prompt": "q"}, timeout=5)
        writeback = requests.post(
            f"{proxy.url}/api/v1/after_turn", json={"messages": []}, timeout=5
        )
    finally:
        proxy.close()
        target.shutdown()
        target.server_close()
        target_thread.join(timeout=5)

    assert compose.json()["retrievedEvidence"] == "memory"
    assert writeback.json() == {"ok": True, "readOnly": True}
    assert calls == ["/api/v1/compose"]
    assert proxy.intercepted == ["/api/v1/after_turn"]
