from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "plugins/ogmemory/scripts/ogmemory_codex_hook.py"
RUNNER = """
import json, runpy, sys
module = runpy.run_path(sys.argv[1])
main = module['main']
def post(path, body, timeout):
    print(json.dumps({'path': path, 'body': body}, ensure_ascii=False), file=sys.stderr)
    return {'ok': True, 'identityContext': '中文偏好：简洁回答。'}
main.__globals__['_post_json'] = post
main.__globals__['_get_json'] = lambda *args, **kwargs: None
raise SystemExit(main())
""".encode("ascii", "backslashreplace").decode("ascii")


@pytest.mark.parametrize("encoding", ["gbk:surrogateescape", "utf-8"])
def test_utf8_prompt_and_stop_round_trip(tmp_path, encoding):
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    env = dict(os.environ, PYTHONIOENCODING=encoding, PLUGIN_DATA=str(tmp_path),
               OG_MEMORY_CODEX_CONFIG=str(config), OGMEM_AUTO_CAPTURE="true",
               OGMEM_AUTO_RECALL="true")
    prompt = "重启了，查看一下呢？😀"
    answer = "已修复，中文记忆正常。✅"
    for event, field, text in [("UserPromptSubmit", "prompt", prompt),
                               ("Stop", "last_assistant_message", answer)]:
        payload = {"hook_event_name": event, "session_id": "utf8-session",
                   "turn_id": "utf8-turn", field: text}
        result = subprocess.run(
            [sys.executable, "-c", RUNNER, str(SCRIPT)],
            input=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            capture_output=True, env=env, check=True, timeout=15,
        )
        output = json.loads(result.stdout.decode("utf-8"))
        calls = [json.loads(line) for line in result.stderr.decode("utf-8").splitlines()]
        trace = next(call["body"]["event"] for call in calls
                     if call["path"] == "/api/v1/trace/events")
        if event == "UserPromptSubmit":
            assert trace["prompt"] == prompt
            recall = next(call["body"] for call in calls if call["path"] == "/api/v1/compose")
            assert recall["prompt"] == prompt
            assert "中文偏好" in output["hookSpecificOutput"]["additionalContext"]
        else:
            assert trace["assistant_message"] == answer
            write = next(call["body"] for call in calls if call["path"] == "/api/v1/after_turn")
            assert write["messages"] == [{"role": "user", "content": prompt},
                                         {"role": "assistant", "content": answer}]
            assert write["client_request_id"] == "codex:utf8-session:utf8-turn"
            assert output == {"continue": True}
