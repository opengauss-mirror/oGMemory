import json
import sqlite3
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[3] / "plugins/zcode/scripts"
sys.path.insert(0, str(SCRIPTS))
from capture import capture
from state import Queue, deliver


def test_continued_stop_keeps_trace_and_only_writes_user_once(tmp_path):
    queue = Queue(tmp_path / "state.sqlite", "partition")
    identity = {"accountId": "a", "userId": "u", "agentId": "g", "sessionId": "s"}
    base = {"sessionId": "s", "turnId": "t", "traceId": "source"}
    for i, (name, fields) in enumerate([
        ("UserPromptSubmit", {"prompt": "remember x"}),
        ("PreToolUse", {"toolCallId": "c", "toolName": "Read"}),
        ("PostToolUseFailure", {"toolCallId": "c", "error": "missing"}),
        ("Stop", {"responseText": "first"}),
        ("Stop", {"responseText": "second", "stopHookActive": True}),
    ]):
        event = dict(base, hookEventName=name, timestamp=str(i), **fields)
        for _ in range(2):
            capture(queue, event, identity, True, True, True, False)
    with sqlite3.connect(queue.path) as db:
        rows = db.execute("SELECT channel, payload FROM delivery ORDER BY seq").fetchall()
    traces = [json.loads(body)["event"] for channel, body in rows if channel == "trace"]
    writes = [json.loads(body) for channel, body in rows if channel == "write"]
    assert [e["event_type"] for e in traces] == [
        "user_message", "tool_start", "tool_failure", "turn_end", "turn_end"
    ]
    assert traces[1]["object_id"] == traces[2]["object_id"] == "c"
    assert [[m["role"] for m in w["messages"]] for w in writes] == [
        ["user", "assistant"], ["assistant"]
    ]
    assert len({w["client_request_id"] for w in writes}) == 2
    assert writes[0]["forceExtract"]
    assert traces[0]["trace_id"] == "source"


def test_lost_write_response_checks_receipt_and_advances_queue(tmp_path):
    queue = Queue(tmp_path / "state.sqlite", "partition")
    with queue.transaction() as db:
        for request in ("one", "two"):
            queue.enqueue(db, "s", "write", {"sessionId": "s", "client_request_id": request})

    class Client:
        calls = []

        def request(self, endpoint, payload=None, timeout=3):
            self.calls.append(endpoint)
            if len(self.calls) == 1:
                raise TimeoutError()
            if endpoint.startswith("sessions"):
                return 200, {"request_status": "ACCEPTED", "accepted": True, "status": "accumulating"}, None
            return 200, {"status": "accumulating"}, None

    client = Client()
    deliver(queue, client, queue.claim(), lambda _: None)
    assert queue.claim() is None
    with queue.transaction() as db:
        db.execute("UPDATE delivery SET due=0")
    deliver(queue, client, queue.claim(), lambda _: None)
    deliver(queue, client, queue.claim(), lambda _: None)
    assert queue.claim() is None
    assert client.calls == ["after_turn", "sessions/s/write-requests/one", "after_turn"]


def test_blocked_trace_preserves_order_without_blocking_write(tmp_path):
    queue = Queue(tmp_path / "state.sqlite", "partition")
    with queue.transaction() as db:
        queue.enqueue(db, "s", "trace", {"event": "tool_start"})
        queue.enqueue(db, "s", "trace", {"event": "turn_end"})
        queue.enqueue(db, "s", "write", {"sessionId": "s", "client_request_id": "one"})
    first = queue.claim()
    queue.finish(first[0], first[-1], "blocked")
    assert queue.claim()[2] == "write"
    assert queue.claim() is None


def test_projecting_receipt_releases_submission_and_observes_success(tmp_path):
    queue = Queue(tmp_path / "state.sqlite", "partition")
    with queue.transaction() as db:
        queue.enqueue(db, "s", "write", {"sessionId": "s", "client_request_id": "one"})

    class Client:
        def request(self, endpoint, payload=None, timeout=3):
            if endpoint == "after_turn":
                return 202, {"accepted": True, "request_status": "PROJECTING", "retry_action": "poll"}, None
            return 200, {"accepted": True, "request_status": "SUCCEEDED", "terminal": True}, None

    client = Client()
    deliver(queue, client, queue.claim(), lambda _: None)
    observation = queue.claim()
    assert observation[2] == "observe"
    deliver(queue, client, observation, lambda _: None)
    assert queue.claim() is None
    with sqlite3.connect(queue.path) as db:
        assert json.loads(db.execute("SELECT summary FROM receipt").fetchone()[0])["request_status"] == "SUCCEEDED"


def test_trace_redacts_cli_tokens_before_persistence(tmp_path):
    queue = Queue(tmp_path / "state.sqlite", "partition")
    event = {"hookEventName": "PreToolUse", "toolInput": {"command": "curl --token very-private"},
             "turnId": "t", "sessionId": "s", "timestamp": "clock"}
    identity = {"accountId": "a", "userId": "u", "agentId": "g", "sessionId": "s"}
    capture(queue, event, identity, True, True, True, False)
    with sqlite3.connect(queue.path) as db:
        payload = db.execute("SELECT payload FROM delivery").fetchone()[0]
    assert "very-private" not in payload and "[REDACTED]" in payload
