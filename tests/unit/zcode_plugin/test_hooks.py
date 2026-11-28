import json
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "plugins/zcode/scripts"
ROOT = SCRIPTS.parents[2]
sys.path.insert(0, str(SCRIPTS))
from capture import capture
import ogmemory_zcode_hook as hook
from state import Queue, deliver


def test_hook_wakeups_share_one_worker_during_retry_after(tmp_path, monkeypatch):
    queue = Queue(tmp_path / "state.sqlite", "partition")

    class RateLimitedClient:
        def request(self, *args, **kwargs):
            return 429, {}, "60"

    with queue.transaction() as db:
        queue.enqueue(db, "s", "trace", {"event": {"event_type": "tool_start"}})
    deliver(queue, RateLimitedClient(), queue.claim(), lambda _: None)
    launches = []
    monkeypatch.setattr(hook.subprocess, "Popen", lambda args, **kwargs: launches.append(args))
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda _: hook.start_worker(queue), range(20)))
    assert len(launches) == 1

    other = Queue(queue.path, "other")
    with other.transaction() as db:
        other.enqueue(db, "s", "trace", {})
    hook.start_worker(other)
    assert len(launches) == 2


def test_failed_launch_and_expired_worker_allow_recovery(tmp_path, monkeypatch):
    queue = Queue(tmp_path / "state.sqlite", "partition")
    with queue.transaction() as db:
        queue.enqueue(db, "s", "trace", {})

    def fail(*args, **kwargs):
        raise OSError("spawn failed")

    monkeypatch.setattr(hook.subprocess, "Popen", fail)
    with pytest.raises(OSError):
        hook.start_worker(queue)
    first = queue.reserve_worker(65)
    assert first
    now = hook.time.time()
    monkeypatch.setattr(hook.time, "time", lambda: now + 600)
    replacement = queue.reserve_worker(65)
    assert replacement and replacement != first
    assert not queue.renew_worker(first)
    queue.release_worker(first)
    assert queue.reserve_worker(65) is None


def test_worker_exit_rechecks_work_enqueued_before_release(tmp_path, monkeypatch):
    queue = Queue(tmp_path / "state.sqlite", "partition")
    with queue.transaction() as db:
        queue.enqueue(db, "s", "trace", {"event": {"event_type": "turn_end"}})
    launches = []
    monkeypatch.setattr(hook.subprocess, "Popen", lambda args, **kwargs: launches.append(args))
    current = queue.reserve_worker(65)

    class Client:
        def request(self, *args, **kwargs):
            return 200, {"ok": True}, None

    release = queue.release_worker

    def enqueue_and_release(token):
        with queue.transaction() as db:
            queue.enqueue(db, "s", "write", {"sessionId": "s", "client_request_id": "one"})
        hook.start_worker(queue)
        assert not launches
        release(token)

    monkeypatch.setattr(queue, "release_worker", enqueue_and_release)
    hook.run_worker(queue, Client(), current)
    assert len(launches) == 1


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
    queue.finish(first, "blocked")
    assert queue.claim().channel == "write"
    assert queue.claim() is None


def test_schedule_and_claim_share_channel_order_and_observation_exception(tmp_path, monkeypatch):
    queue = Queue(tmp_path / "state.sqlite", "partition")
    monkeypatch.setattr(hook.time, "time", lambda: 1000)
    with queue.transaction() as db:
        queue.enqueue(db, "s", "trace", {})
        queue.enqueue(db, "s", "trace", {})
        queue.enqueue(db, "s", "observe", {})
        queue.enqueue(db, "s", "observe", {})
        db.execute("UPDATE delivery SET state='blocked' WHERE seq IN (1,3)")
        db.execute("UPDATE delivery SET due=1060 WHERE seq=4")
    assert queue.next_due() == 1060
    assert queue.claim() is None
    monkeypatch.setattr(hook.time, "time", lambda: 1060)
    observation = queue.claim()
    assert observation.channel == "observe"
    queue.finish(observation, "done")
    assert queue.next_due() is None
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
    write = queue.claim()
    deliver(queue, client, write, lambda _: None)
    queue.observe_write(write, {})
    with queue.transaction() as db:
        queue.prune(db)
        assert db.execute("SELECT state FROM delivery WHERE channel='write'").fetchone() == ("done",)
    observation = queue.claim()
    assert observation.channel == "observe"
    deliver(queue, client, observation, lambda _: None)
    assert queue.claim() is None
    with sqlite3.connect(queue.path) as db:
        assert json.loads(db.execute("SELECT summary FROM receipt").fetchone()[0])["request_status"] == "SUCCEEDED"
    with queue.transaction() as db:
        db.execute("UPDATE delivery SET due=1 WHERE channel='write'")
        queue.prune(db)
        assert db.execute("SELECT 1 FROM delivery WHERE channel='write'").fetchone() is None


def test_trace_redacts_cli_tokens_before_persistence(tmp_path):
    queue = Queue(tmp_path / "state.sqlite", "partition")
    event = {"hookEventName": "PreToolUse", "toolInput": {"command": "curl --token very-private"},
             "turnId": "t", "sessionId": "s", "timestamp": "clock"}
    identity = {"accountId": "a", "userId": "u", "agentId": "g", "sessionId": "s"}
    capture(queue, event, identity, True, True, True, False)
    with sqlite3.connect(queue.path) as db:
        payload = db.execute("SELECT payload FROM delivery").fetchone()[0]
    assert "very-private" not in payload and "[REDACTED]" in payload


def test_plugin_bundles_shared_redaction_rules():
    expected = ROOT / "skills/trace_redaction_rules.json"
    bundled = SCRIPTS / expected.name
    assert json.loads(bundled.read_text(encoding="utf-8")) == json.loads(expected.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", json.loads((ROOT / "tests/fixtures/trace/redaction_cases.json").read_text(encoding="utf-8")) + [
    {"text": "-----BEGIN PRIVATE KEY-----\nfixture-pem-body\n-----END PRIVATE KEY-----",
     "plaintext": "fixture-pem-body"},
])
def test_queue_redacts_shared_trace_fixtures(tmp_path, case):
    queue = Queue(tmp_path / "state.sqlite", "partition")
    identity = dict(accountId="a", userId="u", agentId="g", sessionId="s")
    common = dict(turnId="t", sessionId="s")
    for event in [
        dict(common, hookEventName="PostToolUse", toolResponse={"result": case["text"]}),
        dict(common, hookEventName="UserPromptSubmit", prompt=json.dumps(case["text"], ensure_ascii=False)
             if isinstance(case["text"], dict) else case["text"]),
        dict(common, hookEventName="Stop", responseText="done"),
    ]:
        capture(queue, event, identity, True, True, True, False)
    with queue.transaction() as db:
        pending = db.execute("SELECT channel,payload FROM delivery").fetchall()
    assert {"trace", "write"} == {channel for channel, _ in pending}
    secrets = case["plaintext"] if isinstance(case["plaintext"], list) else [case["plaintext"]]
    for _, payload in pending:
        for secret in secrets:
            assert secret not in payload


@pytest.mark.parametrize("trace_fields", [{"trace_id": "trace-1"}, {"traceId": "trace-1"}, {}])
@pytest.mark.parametrize("reply_fields", [
    {"responseText": "reply", "last_assistant_message": "older", "responsePreview": "preview"},
    {"last_assistant_message": "reply", "responsePreview": "preview"},
    {"responsePreview": "reply"},
])
def test_stop_uses_one_reply_and_trace_id_across_outputs(tmp_path, trace_fields, reply_fields):
    queue = Queue(tmp_path / "state.sqlite", "partition")
    identity = dict(accountId="a", userId="u", agentId="g", sessionId="s")
    base = dict(turn_id="t", session_id="s")
    capture(queue, dict(base, hook_event_name="UserPromptSubmit", prompt="question"), identity, True, True, True, False)
    capture(queue, dict(base, hook_event_name="Stop", **trace_fields, **reply_fields), identity, True, True, True, False)
    with queue.transaction() as db:
        deliveries = [json.loads(body) for body, in db.execute("SELECT payload FROM delivery ORDER BY seq")]
    stop, write = deliveries[-2:]
    assert stop["trace_id"] == stop["event"]["trace_id"] == (next(iter(trace_fields.values()), stop["request_id"]))
    assert stop["event"]["assistant_message"] == write["messages"][-1]["content"] == "reply"
