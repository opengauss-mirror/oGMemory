"""Opt-in real OpenCode/oGMemory smoke test using a dedicated test identity.

Required: OGMEM_URL/API_KEY/ACCOUNT_ID/USER_ID/AGENT_ID and
LLM_BASE_URL/MODEL/API_KEY. Optional: OPENCODE_BIN, OGMEM_OPENCODE_PLUGIN_DIR,
OGMEM_OPENCODE_RUNTIME. Writes one unique fact and invokes a harmless printf tool.
The default opencode-verification/ directory is gitignored because it contains
service credentials. Custom runtime paths should be outside Git or gitignored.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from uuid import uuid4

import httpx


def _write_private_config(path: Path, config: dict) -> None:
    """Publish a complete config from a temporary file created with mode 0600."""
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".ogmemory-config-")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as config_file:
            config_file.write(json.dumps(config, ensure_ascii=False, indent=2))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    runtime = Path(os.getenv("OGMEM_OPENCODE_RUNTIME", "opencode-verification")).resolve()
    runtime.mkdir(mode=0o700, parents=True, exist_ok=True)
    runtime.chmod(0o700)
    plugin_dir = Path(
        os.getenv(
            "OGMEM_OPENCODE_PLUGIN_DIR",
            str(Path(__file__).resolve().parents[1] / "plugins/opencode"),
        )
    ).resolve()
    binary = os.getenv("OPENCODE_BIN", "opencode")
    run_id = uuid4().hex[:12]
    project = "OC验证" + run_id
    expected = f"每周五 {9 + int(run_id[:2], 16) % 12:02d}:{int(run_id[2:4], 16) % 60:02d}"
    audit_path = runtime / f"plugin-audit-{run_id}.jsonl"
    config_path = runtime / "ogmemory-config.json"
    config = {
        "enabled": True,
        "autoCapture": True,
        "autoRecall": True,
        "endpoint": os.environ["OGMEM_URL"],
        "apiKey": os.environ["OGMEM_API_KEY"],
        "accountId": os.environ["OGMEM_ACCOUNT_ID"],
        "userId": os.environ["OGMEM_USER_ID"],
        "agentId": os.environ["OGMEM_AGENT_ID"],
        "composeTimeoutMs": 60000,
        "writeTimeoutMs": 120000,
        "traceTimeoutMs": 10000,
    }
    _write_private_config(config_path, config)
    # Observe real hooks/HTTP without changing their payloads or responses.
    wrapper = runtime / f"observed-plugin-{run_id}.mjs"
    wrapper.write_text(
        'import { appendFile } from "node:fs/promises";\n'
        f"import {{ OgMemoryPlugin }} from {json.dumps((plugin_dir / 'index.mjs').as_uri())};\n"
        f"const audit = {json.dumps(str(audit_path))};\n"
        'const record = (value) => appendFile(audit, JSON.stringify(value) + "\\n", {mode: 0o600});\n'
        "export default async function(input) {\n"
        "  const hooks = await OgMemoryPlugin(input, {\n"
        f"    configOptions: {{configPath: {json.dumps(str(config_path))}}},\n"
        "    fetch: async (url, options) => {\n"
        "      const body = JSON.parse(options.body);\n"
        "      const response = await fetch(url, options);\n"
        "      const data = await response.clone().json().catch(() => ({}));\n"
        '      await record({kind:"http", path:new URL(url).pathname, sessionId:body.sessionId || body.session_id,\n'
        "        status:response.status, ok:data.ok, resultStatus:data.status, messages:body.messages,\n"
        "        traceType:body.event?.event_type,\n"
        "        memories:Array.isArray(data.messages) ? data.messages.filter(m => m._ogmem === true) : []});\n"
        "      return response;\n"
        "    }\n"
        "  });\n"
        '  const transform = hooks["experimental.chat.system.transform"];\n'
        '  hooks["experimental.chat.system.transform"] = async (input, output) => {\n'
        "    await transform?.(input, output);\n"
        '    await record({kind:"injection", sessionId:input.sessionID || input.sessionId,\n'
        '      memory:output.system.filter(text => text.includes("[oGMemory retrieved context]"))});\n'
        "  };\n"
        "  return hooks;\n"
        "}\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    for key in ["OPENCODE_CONFIG", "OPENCODE_CONFIG_DIR", "OPENCODE_CONFIG_CONTENT"]:
        environment.pop(key, None)
    environment.update(
        {
            "OG_MEMORY_URL": config["endpoint"],
            "OG_AUTH_API_KEY": config["apiKey"],
            "OG_AUTH_ACCOUNT_ID": config["accountId"],
            "OG_AUTH_USER_ID": config["userId"],
            "OG_AUTH_AGENT_ID": config["agentId"],
        }
    )
    for key, name in [
        ("HOME", "home"),
        ("XDG_CONFIG_HOME", "config"),
        ("XDG_DATA_HOME", "data"),
        ("XDG_CACHE_HOME", "cache"),
        ("TMPDIR", "tmp"),
    ]:
        directory = runtime / name
        directory.mkdir(exist_ok=True)
        environment[key] = str(directory)
    environment["OPENCODE_DISABLE_AUTOUPDATE"] = "true"
    environment["OPENCODE_DISABLE_DEFAULT_PLUGINS"] = "true"
    environment["OPENCODE_DISABLE_MODELS_FETCH"] = "true"
    environment["npm_config_registry"] = "https://repo.huaweicloud.com/repository/npm/"
    environment["OGMEM_AGENT_LLM_API_KEY"] = os.environ["LLM_API_KEY"]
    environment["OPENCODE_CONFIG_CONTENT"] = json.dumps(
        {
            "plugin": [wrapper.as_uri()],
            "enabled_providers": ["ogmemoryverify"],
            "provider": {
                "ogmemoryverify": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "oGMemory verification",
                    "options": {
                        "baseURL": os.environ["LLM_BASE_URL"],
                        "apiKey": "{env:OGMEM_AGENT_LLM_API_KEY}",
                    },
                    "models": {
                        os.environ["LLM_MODEL"]: {
                            "name": os.environ["LLM_MODEL"],
                            "limit": {"context": 64000, "output": 4096},
                        }
                    },
                }
            },
            "permission": {"*": "deny", "bash": {"*": "deny", "printf *": "allow"}},
        }
    )
    workspace = runtime / "workspace"
    workspace.mkdir(exist_ok=True)

    def invoke(prompt, label):
        stdout_path = runtime / f"{label}-{run_id}.stdout.log"
        stderr_path = runtime / f"{label}-{run_id}.stderr.log"
        with (
            stdout_path.open("w", encoding="utf-8") as stdout,
            stderr_path.open("w", encoding="utf-8") as stderr,
        ):
            result = subprocess.run(
                [
                    binary,
                    "run",
                    "--format",
                    "json",
                    "--model",
                    "ogmemoryverify/" + os.environ["LLM_MODEL"],
                    "--dir",
                    str(workspace),
                    prompt,
                ],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                timeout=240,
            )
        output = stdout_path.read_text(encoding="utf-8")
        if result.returncode:
            raise RuntimeError(
                f"OpenCode {label} failed with exit {result.returncode}; inspect its private log"
            )
        answer, sessions = [], set()
        for line in output.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("sessionID"):
                sessions.add(event["sessionID"])
            part = event.get("part", {})
            if event.get("type") == "text" and part.get("text"):
                answer.append(part["text"])
        if not answer:
            raise RuntimeError(f"OpenCode {label} produced no answer; inspect its log")
        return "".join(answer).strip(), sessions

    report = {
        "run_id": run_id,
        "project": project,
        "expected": expected,
        "model": os.environ["LLM_MODEL"],
        "endpoint": config["endpoint"],
        "identity": {k: config[k] for k in ["accountId", "userId", "agentId"]},
    }
    report["opencode_version"] = subprocess.check_output(
        [binary, "--version"], env=environment, text=True
    ).strip()
    print("VERIFY", json.dumps(report, ensure_ascii=False), flush=True)
    # Exceed the deployed server's normal 200-token extraction threshold,
    # without changing global settings or manually forcing an extraction.
    padding = (
        "以下仅为接口验证说明，不是需要记忆的业务事实：测试结束后将以新会话检查固定发布安排。" * 30
    )
    report["write_answer"], write_sessions = invoke(
        f"请记住：项目{project}的发布窗口是{expected}。这是一条固定的项目安排，请不要创建或修改文件。\n{padding}\n只回复‘已记住’。",
        "write",
    )
    print("WRITE_ANSWER", report["write_answer"], flush=True)

    def audit():
        return (
            [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
            if audit_path.exists()
            else []
        )

    writes = [
        item
        for item in audit()
        if item.get("path", "").endswith("after_turn") and item.get("ok") is True
    ]
    assert writes, "OpenCode plugin did not automatically write the completed turn"
    write_sessions.update(item["sessionId"] for item in writes)
    assert any(
        any(m["role"] == "user" for m in item["messages"])
        and any(m["role"] == "assistant" for m in item["messages"])
        for item in writes
    ), "Writeback did not contain user and assistant messages"
    query = f"项目{project}的发布窗口是什么？只回复日期和时间，不要使用工具或读取文件。"
    headers = {
        "X-API-Key": config["apiKey"],
        "X-Account-ID": config["accountId"],
        "X-User-ID": config["userId"],
    }
    evidence = ""
    deadline = time.monotonic() + 300
    with httpx.Client(timeout=120) as client:
        while time.monotonic() < deadline:
            response = client.post(
                config["endpoint"] + "/api/v1/compose",
                headers=headers,
                json={
                    **report["identity"],
                    "sessionId": "oc-poll-" + run_id,
                    "prompt": query,
                    "messages": [{"role": "user", "content": query}],
                },
            )
            response.raise_for_status()
            evidence = "\n".join(
                m.get("content", "")
                for m in response.json().get("messages", [])
                if m.get("_ogmem") is True
            )
            if run_id in evidence and expected.split()[-1] in evidence:
                print("MEMORY_INDEX_READY", flush=True)
                break
            print("WAITING_FOR_EXTRACTION_AND_INDEX", flush=True)
            time.sleep(10)
        else:
            raise RuntimeError("Automatic writeback was not searchable before timeout")
    report["read_answer"], read_sessions = invoke(query, "read")
    assert expected.split()[-1] in report["read_answer"] and (
        "周五" in report["read_answer"] or "星期五" in report["read_answer"]
    ), "New OpenCode session failed to recall the release window"
    assert write_sessions.isdisjoint(read_sessions), "Read reused the write session"
    assert any(
        item.get("kind") == "injection"
        and item.get("sessionId") in read_sessions
        and run_id in str(item.get("memory", []))
        for item in audit()
    ), "Model system prompt did not receive the test memory"
    print("NEW_SESSION_RECALL_ANSWER", report["read_answer"], flush=True)
    report["tool_answer"], tool_sessions = invoke(
        "必须实际调用 bash 工具执行 printf 'OGMEM_OPENCODE_TOOL_OK'，然后只回复该输出。不要执行其他命令。",
        "tool",
    )
    assert report["tool_answer"] == "OGMEM_OPENCODE_TOOL_OK", "Tool output was not returned"
    records = audit()
    trace_types = sorted(
        {
            item["traceType"]
            for item in records
            if item.get("traceType") and item.get("status") == 200 and item.get("ok") is True
        }
    )
    for required in ["user_message", "assistant_message", "turn_end", "tool_start", "tool_success"]:
        assert required in trace_types, f"Missing real OpenCode Trace: {required}"
    report.update(
        {
            "status": "passed",
            "write_sessions": sorted(write_sessions),
            "read_sessions": sorted(read_sessions),
            "tool_sessions": sorted(tool_sessions),
            "trace_types": trace_types,
            "evidence": evidence[:3000],
            "audit_file": str(audit_path),
            "http_calls": [
                {k: v for k, v in record.items() if k not in ["messages", "memories"]}
                for record in records
                if record.get("kind") == "http"
            ],
        }
    )
    report_path = runtime / "opencode-verification.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("PASS: real OpenCode + automatic writeback + new-session recall + tool Trace", flush=True)
    print("REPORT:", report_path, flush=True)


if __name__ == "__main__":
    main()
