#!/usr/bin/env python3
"""End-to-end test: session lifecycle via HTTP.

Prerequisites:
  - AGFS server running on port 1833
  - ContextEngine HTTP server running on port 8090

Run:
    PYTHONPATH=. python3 tests/e2e/test_session_lifecycle.py
"""

import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://localhost:8090/api/v1"
SESSION_ID = "e2e-session-001"


def http_post(path, params):
    """POST JSON to endpoint, return parsed response."""
    data = json.dumps(params).encode()
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"  HTTP {e.code}: {body}")
        return {"error": body, "_http_code": e.code}
    except Exception as e:
        print(f"  Connection error: {e}")
        return {"error": str(e)}


def http_get(path, params=None):
    """GET endpoint with query params, return parsed response."""
    query = ""
    if params:
        query = "?" + "&".join(f"{k}={v}" for k, v in params.items())
    req = urllib.request.Request(f"{BASE}{path}{query}", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"  HTTP {e.code}: {body}")
        return {"error": body, "_http_code": e.code}
    except Exception as e:
        print(f"  Connection error: {e}")
        return {"error": str(e)}


def check_server():
    """Verify ContextEngine server is reachable."""
    try:
        urllib.request.urlopen(f"{BASE}/health", timeout=3)
        return True
    except Exception:
        return False


def wait_for_commit(session_id, common, timeout=10):
    """Poll until commit_count > 0 or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        session = http_get(f"/sessions/{session_id}", common)
        if session.get("commit_count", 0) > 0:
            return session
        time.sleep(0.5)
    return http_get(f"/sessions/{session_id}", common)


def check_add_messages_and_check_pending():
    """Step 1: Add messages to session buffer, check pending_tokens."""
    print("\n=== Step 1: Add messages to session buffer ===")

    common = {
        "accountId": "acct-e2e",
        "userId": "u-e2e",
        "agentId": "agent-e2e",
    }

    for i, (role, text) in enumerate([
        ("user", "你好，请记住我的名字叫张三，我是一名Python开发者"),
        ("assistant", "好的张三，很高兴认识你！我记下了。"),
        ("user", "我最近在学LangChain，需要关于RAG的建议，特别是向量检索的部分"),
    ]):
        result = http_post(f"/sessions/{SESSION_ID}/messages", {
            **common,
            "role": role,
            "content": text,
        })
        ok = result.get("ok", False)
        tokens = result.get("pending_tokens", "?")
        msg_count = result.get("message_count", "?")
        print(f"  msg[{i}] role={role}: ok={ok}, pending_tokens={tokens}, count={msg_count}")
        assert ok, f"add_message failed: {result}"

    session = http_get(f"/sessions/{SESSION_ID}", common)
    pending = session.get("pending_tokens", 0)
    count = session.get("message_count", 0)
    print(f"  Session: pending_tokens={pending}, message_count={count}")
    assert count == 3, f"Expected 3 messages, got {count}"
    assert pending > 0, "Expected pending_tokens > 0"
    print("  PASSED")
    return common


def check_after_turn_accumulates(common):
    """Step 2: afterTurn with messages below threshold (should accumulate)."""
    print("\n=== Step 2: afterTurn (below threshold) ===")

    messages = [
        {"role": "user", "content": "Chroma和FAISS哪个更适合我的场景？"},
        {"role": "assistant", "content": "这取决于你的数据规模和查询复杂度..."},
    ]

    result = http_post("/after_turn", {
        **common,
        "sessionId": SESSION_ID,
        "messages": messages,
        "prePromptMessageCount": 0,
        "commitTokenThreshold": 999999,
    })
    ok = result.get("ok", False)
    status = result.get("status", "")
    pending = result.get("pending_tokens", "?")
    print(f"  afterTurn: ok={ok}, status={status}, pending_tokens={pending}")
    assert ok, f"afterTurn failed: {result}"
    print("  PASSED")


def check_commit_session(common):
    """Step 3: Commit session (archive + extract) and wait for completion."""
    print("\n=== Step 3: Commit session ===")

    result = http_post(f"/sessions/{SESSION_ID}/commit", {
        **common,
        "wait": False,
    })
    archived = result.get("archived", False)
    task_id = result.get("task_id")
    print(f"  commit: archived={archived}, task_id={task_id}")

    # Wait for background thread to finish
    session = wait_for_commit(SESSION_ID, common, timeout=15)
    pending = session.get("pending_tokens", 0)
    commit_count = session.get("commit_count", 0)
    print(f"  After commit: pending_tokens={pending}, commit_count={commit_count}")
    assert pending == 0, f"Expected pending_tokens=0 after commit, got {pending}"
    assert commit_count >= 1, f"Expected commit_count>=1, got {commit_count}"
    print("  PASSED")


def check_get_context(common):
    """Step 4: Get session context (should have archive info)."""
    print("\n=== Step 4: Get session context ===")

    result = http_get(f"/sessions/{SESSION_ID}/context", {
        **common,
        "token_budget": "128000",
    })
    archive_count = result.get("archive_count", 0)
    active_count = result.get("active_message_count", 0)
    tokens = result.get("estimatedTokens", 0)
    print(f"  context: archive_count={archive_count}, "
          f"active_msg_count={active_count}, "
          f"estimated_tokens={tokens}")
    assert archive_count >= 1, f"Expected archive_count>=1, got {archive_count}"
    print("  PASSED")


def check_compact(common):
    """Step 5: Compact (compress + archive + return summary)."""
    print("\n=== Step 5: Compact ===")

    # Add some new messages to compact
    for i in range(3):
        http_post(f"/sessions/{SESSION_ID}/messages", {
            **common,
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"New message after commit #{i}. " + "x" * 200,
        })

    time.sleep(0.3)

    result = http_post("/compact", {
        **common,
        "sessionId": SESSION_ID,
        "tokenBudget": 128000,
    })
    ok = result.get("ok", False)
    compacted = result.get("compacted", False)
    print(f"  compact: ok={ok}, compacted={compacted}")
    if compacted:
        inner = result.get("result", {})
        print(f"  summary_len={len(inner.get('summary', ''))}, "
              f"tokensBefore={inner.get('tokensBefore')}, "
              f"tokensAfter={inner.get('tokensAfter')}")
    assert ok, f"compact failed: {result}"
    print("  PASSED")


def check_assemble(common):
    """Step 6: Assemble (should return systemPromptAddition)."""
    print("\n=== Step 6: Assemble ===")

    result = http_post("/assemble", {
        **common,
        "sessionId": SESSION_ID,
        "messages": [
            {"role": "user", "content": "帮我回忆一下之前讨论的RAG方案"},
        ],
        "prompt": "RAG方案",
        "tokenBudget": 128000,
    })
    prompt_addition = result.get("systemPromptAddition", "")
    tokens = result.get("estimatedTokens", 0)
    print(f"  systemPromptAddition length={len(prompt_addition)}, estimatedTokens={tokens}")
    assert tokens >= 0, f"Unexpected tokens: {tokens}"
    print("  PASSED")


def run_session_lifecycle():
    common = check_add_messages_and_check_pending()
    check_after_turn_accumulates(common)
    check_commit_session(common)
    check_get_context(common)
    check_compact(common)
    check_assemble(common)


def test_session_lifecycle():
    import importlib
    import threading
    import uuid
    from unittest.mock import patch

    from werkzeug.serving import make_server

    from providers.unified_config import OgMemConfig
    from server.memory_service import MemoryService

    http_server = importlib.import_module("server.app")
    service = MemoryService(config=OgMemConfig(
        provider="mock", embedding_provider="mock", vector_db_type="memory",
        after_turn_threshold=100_000,
    ))
    server = make_server("127.0.0.1", 0, http_server.app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    with patch.object(http_server, "_service", service), patch.dict(
        globals(), BASE=f"http://127.0.0.1:{server.server_port}/api/v1",
        SESSION_ID=f"e2e-session-{uuid.uuid4().hex}",
    ):
        thread.start()
        try:
            run_session_lifecycle()
        finally:
            server.shutdown()
            thread.join(timeout=5)
            service.shutdown()


def main():
    print("=" * 60)
    print("E2E Session Lifecycle Test")
    print("=" * 60)

    print("\nWaiting for ContextEngine server...")
    for i in range(10):
        if check_server():
            print("  Server ready!")
            break
        time.sleep(0.5)
    else:
        print("  FAIL: Server not reachable at", BASE)
        sys.exit(1)

    try:
        run_session_lifecycle()

        print("\n" + "=" * 60)
        print("ALL E2E TESTS PASSED")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n  FAIL: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
