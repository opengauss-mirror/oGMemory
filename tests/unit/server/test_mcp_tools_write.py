from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from core.models import RequestContext, Role
from fs.agfs_adapter.agfs_trace_store import AGFSTraceEventStore
from providers.unified_config import OgMemConfig
from server.mcp_server import create_mcp_server
from server.mcp_tools import write_memory
from server.memory_service import MemoryService
from tests.unit.fs.test_agfs_trace_store import _MemoryAGFS

_PII_CASES = json.loads((Path(__file__).parents[3] / "plugins/deepseek-harness/tests/fixtures/pii_cases.json").read_text())


class _FakeAuth:
    def role_control_active(self):
        return False


class _FakeAudit:
    def __init__(self):
        self.records = []

    def record(self, account_id, **kwargs):
        self.records.append({"account_id": account_id, **kwargs})


class _FakeService:
    def __init__(self, cfg=None):
        self._cfg = cfg or OgMemConfig()
        self.after_turn_params = None
        self.audit = _FakeAudit()
        self.after_turn_result = {"ok": True, "status": "accumulating", "pending_tokens": 12}
        self.trace_events = {}
        self.finalized_traces = []

    def add_session_trace(self, params):
        event_id = params["eventId"]
        self.trace_events.setdefault(event_id, params)
        return {"ok": True, "event_index": list(self.trace_events).index(event_id) + 1}

    def finalize_session_trace(self, params):
        self.finalized_traces.append(params)
        return {"ok": True, "turn_id": f"turn_{params['stopEventIndex']}"}

    def get_auth_service(self):
        return _FakeAuth()

    def build_context(self, params, identity=None):
        return RequestContext(
            account_id="acct-1",
            user_id="alice",
            agent_id=params.get("agentId") or "bot",
            session_id=params.get("sessionId") or "unknown",
            trace_id="t1",
            role=Role.MEMBER,
        )

    def after_turn(self, params):
        self.after_turn_params = params
        return self.after_turn_result

    def get_audit_service(self):
        return self.audit


class _StoredService(_FakeService):
    add_session_trace = MemoryService.add_session_trace
    finalize_session_trace = MemoryService.finalize_session_trace

    def __init__(self):
        super().__init__(OgMemConfig(mcp_enable_write_tools=True))
        self.store = AGFSTraceEventStore(client=_MemoryAGFS(), mount_prefix="/local")

    def _get_trace_event_store(self):
        return self.store


def test_write_memory_submits_text_to_after_turn_flow():
    service = _FakeService()

    payload = write_memory(
        service,
        text="Remember that MCP writes should use the server-side flow.",
        session_id="s1",
        agent_id="codex",
        client_request_id="req-1",
    )

    assert payload["ok"] is True
    assert payload["status"] == "accumulating"
    assert payload["retryable"] is False
    assert payload["submitted_messages"] == 1
    assert payload["accepted_messages"] == 1
    assert service.after_turn_params["sessionId"] == "s1"
    assert service.after_turn_params["agentId"] == "codex"
    assert service.after_turn_params["clientRequestId"] == "req-1"
    assert service.after_turn_params["messages"] == [
        {"role": "user", "content": "Remember that MCP writes should use the server-side flow."}
    ]
    assert service.after_turn_params["_ctx"].user_id == "alice"
    assert service.audit.records[0]["action"] == "mcp.write_memory"
    events = [item["event"] for item in service.trace_events.values()]
    assert [item["event_type"] for item in events] == ["user_message", "turn_end"]
    assert events[0]["prompt"] == service.after_turn_params["messages"][0]["content"]
    assert events[1]["boundary"] == "mcp_write"
    assert service.finalized_traces[0]["stopEventIndex"] == 2


def test_write_memory_forwards_force_extract_to_after_turn():
    service = _FakeService()

    write_memory(
        service,
        text="Extract this short memory now.",
        session_id="s1",
        force_extract=True,
        wait=True,
    )

    assert service.after_turn_params["forceExtract"] is True
    assert service.after_turn_params["wait"] is True


def test_write_memory_rejects_request_larger_than_configured_limit():
    service = _FakeService(OgMemConfig(mcp_max_request_bytes=32))

    with pytest.raises(ValueError, match="mcp_max_request_bytes"):
        write_memory(service, text="x" * 128, session_id="s1")


def test_write_memory_submits_messages_to_after_turn_flow():
    service = _FakeService()

    write_memory(
        service,
        messages=[
            {"role": "user", "content": "Use resources for memory type browsing.", "id": "m1"},
            {"role": "assistant", "content": "Acknowledged.", "created_at": "2026-06-23T00:00:00Z"},
        ],
        session_id="s1",
    )

    assert service.after_turn_params["messages"] == [
        {"role": "user", "content": "Use resources for memory type browsing.", "id": "m1"},
        {"role": "assistant", "content": "Acknowledged.", "created_at": "2026-06-23T00:00:00Z"},
    ]


def test_write_memory_does_not_return_internal_write_result_uris():
    service = _FakeService()
    service.after_turn_result = {
        "ok": True,
        "status": "completed",
        "plans": [{"target_uri": "ctx://acct-1/users/alice/memories/preferences/style"}],
        "write_result": {"target_uri": "ctx://acct-1/users/alice/memories/preferences/style"},
        "pending_tokens": 0,
    }

    payload = write_memory(service, text="Remember this safely.", session_id="s1")

    assert payload["status"] == "completed"
    assert payload["stats"]["pending_tokens"] == 0
    assert "ctx://" not in str(payload)
def test_write_memory_requires_session_id():
    with pytest.raises(ValueError, match="session_id is required"):
        write_memory(_FakeService(), text="hello")


def test_write_memory_requires_exactly_one_material_source():
    with pytest.raises(ValueError, match="either text or messages"):
        write_memory(
            _FakeService(),
            text="hello",
            messages=[{"role": "user", "content": "hello"}],
            session_id="s1",
        )


@pytest.mark.parametrize("field", [
    "type", "memory_type", "memory_id", "category", "context_type", "owner_space",
    "level", "metadata", "abstract", "overview", "uri",
])
def test_write_memory_rejects_atomic_memory_fields(field):
    service = _FakeService()
    with pytest.raises(ValueError, match="cannot set atomic memory fields"):
        write_memory(
            service,
            messages=[{"role": "user", "content": "hello", field: "forbidden"}],
            session_id="s1",
        )
    assert service.after_turn_params is None


@pytest.mark.parametrize("source", ["text", "messages"])
def test_write_memory_preserves_json_facts_as_user_content(source):
    service = _FakeService()
    text = json.dumps({"uri": "catalog://item/42", "category": "book", "abstract": "A book", "content": "facts"})
    material = {"text": text} if source == "text" else {"messages": [{"role": "user", "content": text}]}

    write_memory(service, session_id="s1", **material)

    assert service.after_turn_params["messages"] == [{"role": "user", "content": text}]


def test_write_memory_rejects_both_text_and_empty_messages():
    service = _FakeService()
    with pytest.raises(ValueError, match="either text or messages"):
        write_memory(service, text="Remember this", messages=[], session_id="s1")
    assert service.after_turn_params is None


def test_write_trace_preserves_order_identity_and_retry_event_ids():
    service = _FakeService()
    messages = [
        {"role": "user", "content": "Use Python for this task."},
        {"role": "tool", "content": "Validation completed."},
        {"role": "assistant", "content": "Python selected and validated."},
    ]
    for _ in range(2):
        write_memory(service, messages=messages, session_id="s1", client_request_id="write-1")
    stored = list(service.trace_events.values())
    assert [item["event"]["event_type"] for item in stored] == [
        "user_message", "tool_message", "assistant_message", "turn_end",
    ]
    assert stored[1]["event"]["content"] == "Validation completed."
    assert stored[-1]["event"]["assistant_message"] == messages[-1]["content"]
    assert all(item["_ctx"].user_id == "alice" for item in stored)
    assert all(item["sessionId"] == "s1" for item in stored)
    assert [item["stopEventIndex"] for item in service.finalized_traces] == [4, 4]


def test_write_trace_sanitizes_and_bounds_content():
    service = _FakeService()
    write_memory(service, text="password=example-secret " + "a" * 12000, session_id="s1")
    event = next(iter(service.trace_events.values()))["event"]
    assert event["prompt"].startswith("[REDACTED_SECRET]")
    assert len(event["prompt"]) == 10000
    assert event["truncated"] is True


def test_write_trace_assigns_new_events_to_independent_writes():
    service = _FakeService()
    for _ in range(2):
        write_memory(service, text="Use Python.", session_id="s1")
    assert len(service.trace_events) == 4
    assert [item["stopEventIndex"] for item in service.finalized_traces] == [2, 4]


def test_registered_mcp_write_tool_persists_and_finalizes_trace():
    service = _StoredService()
    store = service.store
    mcp = create_mcp_server(service=service, config=service._cfg)
    tool = mcp._tool_manager.get_tool("ogm_write_memory")
    for _ in range(2):
        result = tool.fn(
            session_id="s1",
            client_request_id="persisted-write",
            messages=[
                {"role": "user", "content": "Use Python."},
                {"role": "assistant", "content": "Python selected."},
            ],
        )
        assert result["ok"] is True
    turns = store._read_state("acct-1", collections=("turns",), session_id="s1")["turns"]
    assert len(turns) == 1
    turn = next(iter(turns.values()))
    assert turn["status"] == "finalized"
    assert turn["event_count"] == 3
    assert turn["owner_space"] == "user:alice"
    page = store.read_skill_trace_event_page(
        account_id="acct-1", evidence_uri=turn["uri"], after_ordinal=0, limit=10,
    )
    assert [item["event"]["event_type"] for item in page["events"]] == [
        "user_message", "assistant_message", "turn_end",
    ]


@pytest.mark.parametrize("result", ["Batch limit: 100.", [{"type": "text", "text": "Batch limit: 100."}], []])
@pytest.mark.parametrize("is_error", [False, True])
@pytest.mark.parametrize("role_padding", ["", " \t\n"])
def test_registered_mcp_tool_result_completes_skill_trace(result, is_error, role_padding):
    from skills.sanitizer import sanitize_trace

    service = _StoredService()
    tool = create_mcp_server(service=service, config=service._cfg)._tool_manager.get_tool("ogm_write_memory")
    messages = [
        {"role": "user", "content": "Read the API documentation."},
        {"role": "assistant", "content": "Reading the documentation.", "reasoning": "Check the documented limit.", "tool_calls": [
            {"id": "c1", "function": {"name": "read_file", "arguments": '{"path":"api.md"}'}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "name": "transport_alias", "content": result, "is_error": is_error},
        {"role": "assistant", "content": "Checked the response."},
    ]
    messages = [{**message, "role": f"{role_padding}{message['role']}{role_padding}"} for message in messages]
    for _ in range(2):
        assert tool.fn(session_id="s1", client_request_id="paired-write", messages=messages)["ok"]
    turns = service.store._read_state("acct-1", collections=("turns",), session_id="s1")["turns"]
    assert len(turns) == 1
    turn = next(iter(turns.values()))
    page = service.store.read_skill_trace_event_page(
        account_id="acct-1", evidence_uri=turn["uri"], after_ordinal=0, limit=10,
    )
    events = [row["event"] for row in page["events"]]
    assert events[0]["prompt"] == "Read the API documentation."
    assert events[1]["reasoning"] == "Check the documented limit."
    outcome = events[3]
    assert outcome["event_type"] == ("tool_failure" if is_error else "tool_success")
    assert (outcome["tool_name"], outcome["tool_use_id"]) == ("read_file", "c1")
    assert outcome["tool_response"] == (result if isinstance(result, str) else json.dumps(result))
    trace = sanitize_trace(events, allow_failed=is_error)
    assert (trace.failed_tools if is_error else trace.tools) == {"read_file"}
    assert trace.events[-1]["assistant_message"] == "Checked the response."


def test_write_trace_redacts_nested_json_before_persistence():
    service = _StoredService()
    write_memory(
        service,
        text=json.dumps({
            "password": "hunter2",
            "nested": [{"api_key": "example-key", "label": "db"}],
            "credentials": {"value": "example-credential"},
        }),
        session_id="s1",
    )
    turns = service.store._read_state("acct-1", collections=("turns",), session_id="s1")["turns"]
    prompt = next(iter(turns.values()))["content"][0]["prompt"]
    assert json.loads(prompt) == {
        "password": "[REDACTED_SECRET]",
        "nested": [{"api_key": "[REDACTED_SECRET]", "label": "db"}],
        "credentials": "[REDACTED_SECRET]",
    }


@pytest.mark.parametrize("wrapper", ["Config: {}", "```json\n{}\n```"])
def test_write_trace_redacts_json_embedded_in_messages(wrapper):
    service = _FakeService()
    text = wrapper.format('{"password":"hunter2"}')
    write_memory(service, messages=[{"role": "assistant", "content": text}], session_id="s1")
    event = next(iter(service.trace_events.values()))["event"]
    assert event["assistant_message"] == wrapper.format('{"password": "[REDACTED_SECRET]"}')


def test_concurrent_write_batches_keep_complete_turns_with_retries():
    class SlowService(_StoredService):
        def add_session_trace(self, params):
            result = super().add_session_trace(params)
            if params["hookType"] == "user_message":
                time.sleep(0.02)
            return result

    service = SlowService()
    ready = Barrier(4)

    def write(index):
        ready.wait(timeout=5)
        messages = [
            {"role": "user", "content": f"request {index}"},
            {"role": "assistant", "content": f"answer {index}"},
        ]
        for _ in range(2):
            write_memory(service, messages=messages, session_id="s1", client_request_id=str(index))

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(write, range(4)))
    turns = service.store._read_state("acct-1", collections=("turns",), session_id="s1")["turns"]
    contents = [turn["content"] for turn in turns.values()]
    assert len(contents) == 4
    assert all([event["event_type"] for event in turn] == [
        "user_message", "assistant_message", "turn_end",
    ] for turn in contents)
    assert sorted((turn[0]["prompt"], turn[1]["assistant_message"]) for turn in contents) == [
        (f"request {index}", f"answer {index}") for index in range(4)
    ]


@pytest.mark.parametrize("encoding_depth", [1, 2, 4])
def test_write_trace_redacts_json_encoded_string_values(encoding_depth):
    secret = {"password": "hunter2", "label": "database"}
    expected = {"password": "[REDACTED_SECRET]", "label": "database"}
    for _ in range(encoding_depth):
        secret = json.dumps(secret)
        expected = json.dumps(expected)
    service = _StoredService()
    write_memory(service, text=json.dumps({"payload": [secret]}), session_id="s1")
    turns = service.store._read_state("acct-1", collections=("turns",), session_id="s1")["turns"]
    prompt = next(iter(turns.values()))["content"][0]["prompt"]
    assert json.loads(prompt) == {"payload": [expected]}


def test_write_trace_recovers_complete_batches_after_partial_persistence():
    service = _StoredService()
    append = service.add_session_trace
    interrupted = False

    def append_with_interruption(params):
        nonlocal interrupted
        if not interrupted and params["event"].get("assistant_message") == "answer A":
            interrupted = True
            return {"ok": False, "retryable": True}
        return append(params)

    service.add_session_trace = append_with_interruption
    for label in ["A", "B", "A"]:
        result = write_memory(
            service, session_id="s1", client_request_id=label,
            messages=[
                {"role": "user", "content": f"request {label}"},
                {"role": "assistant", "content": f"answer {label}"},
            ],
        )
        assert result["ok"] is True
    turns = service.store._read_state("acct-1", collections=("turns",), session_id="s1")["turns"]
    assert len(turns) == 2
    for turn in turns.values():
        events = turn["content"]
        assert [event["event_type"] for event in events] == [
            "user_message", "assistant_message", "turn_end",
        ]
        label = events[0]["prompt"].split()[-1]
        assert events[1]["assistant_message"] == events[2]["assistant_message"] == f"answer {label}"
        assert len({event["write_batch_id"] for event in events}) == 1


@pytest.mark.parametrize("wrapper", ["Config: {}", "```json\n{}\n```"])
@pytest.mark.parametrize("encoding_depth", [1, 3])
def test_write_trace_redacts_encoded_json_inside_wrapped_text(wrapper, encoding_depth):
    secret = {"password": "hunter2"}
    expected = {"password": "[REDACTED_SECRET]"}
    for _ in range(encoding_depth):
        secret = json.dumps(secret)
        expected = json.dumps(expected)
    service = _StoredService()
    write_memory(service, text=wrapper.format(json.dumps({"payload": secret})), session_id="s1")
    turns = service.store._read_state("acct-1", collections=("turns",), session_id="s1")["turns"]
    prompt = next(iter(turns.values()))["content"][0]["prompt"]
    assert prompt == wrapper.format(json.dumps({"payload": expected}))


@pytest.mark.parametrize("wrapper", ["Config: {}", "```json\n{}\n```"])
@pytest.mark.parametrize("credential", [{"value": "opaque-example"}, ["opaque-example"], 12345, True, None])
def test_write_trace_redacts_embedded_sensitive_values(wrapper, credential):
    service = _StoredService()
    write_memory(
        service, session_id="s1",
        text=wrapper.format(json.dumps({"credentials": credential, "label": "db", "count": 3})),
    )
    turns = service.store._read_state("acct-1", collections=("turns",), session_id="s1")["turns"]
    prompt = next(iter(turns.values()))["content"][0]["prompt"]
    assert prompt == wrapper.format(json.dumps({"credentials": "[REDACTED_SECRET]", "label": "db", "count": 3}))


@pytest.mark.parametrize("wrapper", ["Config: {}", "```json\n{}\n```"])
@pytest.mark.parametrize("encoding_depth", [1, 3])
def test_write_trace_redacts_wrapped_top_level_json_strings(wrapper, encoding_depth):
    value = {"password": "hunter2", "label": "db"}
    expected = {"password": "[REDACTED_SECRET]", "label": "db"}
    for _ in range(encoding_depth):
        value = json.dumps(value)
        expected = json.dumps(expected)
    service = _StoredService()
    write_memory(service, text=wrapper.format(json.dumps(value)), session_id="s1")
    turns = service.store._read_state("acct-1", collections=("turns",), session_id="s1")["turns"]
    prompt = next(iter(turns.values()))["content"][0]["prompt"]
    assert prompt == wrapper.format(json.dumps(expected))


@pytest.mark.parametrize("text", ['Config: ' + '\\\"' * 20000, 'x' * 65537])
def test_write_trace_persists_placeholder_for_content_exceeding_redaction_budget(text):
    service = _StoredService()
    result = write_memory(service, text=text, session_id="s1")
    assert result["ok"] is True
    turns = service.store._read_state("acct-1", collections=("turns",), session_id="s1")["turns"]
    assert next(iter(turns.values()))["content"][0]["prompt"] == "[REDACTED_SECRET]"


@pytest.mark.parametrize("text", [
    '{"password":"hunter2',
    'Config: {"credentials":{"value":"opaque-example"',
    '```json\n{"password":"hunter2\n```',
    '"password":"hunter2',
    r'{"pass\u0077ord":"hunter2',
])
def test_write_trace_persists_redacted_incomplete_json(text):
    service = _StoredService()
    result = write_memory(service, text=text, session_id="s1")
    assert result["ok"] is True
    turns = service.store._read_state("acct-1", collections=("turns",), session_id="s1")["turns"]
    assert next(iter(turns.values()))["content"][0]["prompt"] == "[REDACTED_SECRET]"


@pytest.mark.parametrize("credential", [{"value": "opaque-example"}, ["opaque-example"], "opaque-example", 123, True, None])
@pytest.mark.parametrize("wrapper", ["{}", "Config: {}", "```json\n{}\n```"])
def test_write_trace_redacts_standalone_sensitive_fields(credential, wrapper):
    service = _StoredService()
    text = wrapper.format('"credentials":' + json.dumps(credential))
    result = write_memory(service, text=text, session_id="s1")
    assert result["ok"] is True
    turns = service.store._read_state("acct-1", collections=("turns",), session_id="s1")["turns"]
    assert next(iter(turns.values()))["content"][0]["prompt"] == "[REDACTED_SECRET]"


@pytest.mark.parametrize("text", [
    "{'password': 'hunter2'}",
    "Config: {'credentials': {'value': 'opaque-example'}}",
    "```python\n{'api_key': ['opaque-example']}\n```",
    "'privateKey': 'opaque-example'",
    "{'password': 123}",
    "{'credentials': None}",
])
def test_write_trace_redacts_single_quoted_sensitive_fields(text):
    service = _StoredService()
    result = write_memory(service, text=text, session_id="s1")
    assert result["ok"] is True
    turns = service.store._read_state("acct-1", collections=("turns",), session_id="s1")["turns"]
    assert next(iter(turns.values()))["content"][0]["prompt"] == "[REDACTED_SECRET]"


@pytest.mark.parametrize("text, expected", [
    ('{"pwd":"hunter2"}', '{"pwd": "[REDACTED_SECRET]"}'),
    ('{"PwD":"hunter2"}', '{"PwD": "[REDACTED_SECRET]"}'),
    ('PWD=hunter2', '[REDACTED_SECRET]'),
    ('pwd=hunter2', '[REDACTED_SECRET]'),
    ('Server=db;UID=user;PWD=hunter2;', 'Server=db;UID=user;[REDACTED_SECRET];'),
    ('--pwd hunter2', '[REDACTED_SECRET]'),
])
def test_write_trace_redacts_pwd_credentials_before_persistence(text, expected):
    service = _StoredService()
    result = write_memory(service, text=text, session_id="s1")
    assert result["ok"] is True
    turns = service.store._read_state("acct-1", collections=("turns",), session_id="s1")["turns"]
    assert next(iter(turns.values()))["content"][0]["prompt"] == expected


@pytest.mark.parametrize("key_type", ["PRIVATE KEY", "RSA PRIVATE KEY", "EC PRIVATE KEY"])
@pytest.mark.parametrize("as_json", [False, True])
def test_write_trace_redacts_unclosed_private_keys_before_persistence(key_type, as_json):
    value = f"-----BEGIN {key_type}-----\nopaque-example"
    text = json.dumps({"output": value}) if as_json else value
    service = _StoredService()
    result = write_memory(service, text=text, session_id="s1")
    assert result["ok"] is True
    turns = service.store._read_state("acct-1", collections=("turns",), session_id="s1")["turns"]
    prompt = next(iter(turns.values()))["content"][0]["prompt"]
    expected = json.dumps({"output": "[REDACTED_SECRET]"}) if as_json else "[REDACTED_SECRET]"
    assert prompt == expected


@pytest.mark.parametrize("text, expected", [
    ("--passwd hunter2", "[REDACTED_SECRET]"),
    ("--private-key opaque-example", "[REDACTED_SECRET]"),
    ("--authorization opaque-example", "[REDACTED_SECRET]"),
    ("Cookie: theme=light; session=opaque-example", "[REDACTED_SECRET]"),
    ("Set-Cookie: session=opaque-example; Path=/; HttpOnly", "[REDACTED_SECRET]"),
    ("cookie: theme=light; session=opaque-example\nAccept: text/plain", "[REDACTED_SECRET]\nAccept: text/plain"),
])
def test_write_trace_redacts_cli_credentials_and_complete_cookie_headers(text, expected):
    service = _StoredService()
    result = write_memory(service, text=text, session_id="s1")
    assert result["ok"] is True
    turns = service.store._read_state("acct-1", collections=("turns",), session_id="s1")["turns"]
    assert next(iter(turns.values()))["content"][0]["prompt"] == expected


@pytest.mark.parametrize("case", _PII_CASES, ids=lambda case: case["name"])
@pytest.mark.parametrize("wrapper", ["{}", "Config: {}", "```json\n{}\n```"])
def test_write_trace_persists_shared_pii_redaction_cases(case, wrapper):
    text = wrapper.format(case["input"] if isinstance(case["input"], str) else json.dumps(case["input"]))
    service = _StoredService()
    result = write_memory(service, messages=[
        {"role": "user", "content": text},
        {"role": "assistant", "content": text},
    ], session_id="s1")
    assert result["ok"] is True
    turns = service.store._read_state("acct-1", collections=("turns",), session_id="s1")["turns"]
    events = next(iter(turns.values()))["content"]
    expected = wrapper.format(case["expected"] if isinstance(case["expected"], str) else json.dumps(case["expected"]))
    assert events[0]["prompt"] == expected
    assert events[1]["assistant_message"] == expected
    assert events[2]["assistant_message"] == expected


def test_trace_rule_bundles_match():
    root = Path(__file__).parents[3]
    assert json.loads((root / "skills/trace_redaction_rules.json").read_text()) == json.loads(
        (root / "plugins/deepseek-harness/lib/trace_redaction_rules.json").read_text()
    )

@pytest.mark.parametrize("text, expected", [
    ("accessToken=opaque-example", "[REDACTED_SECRET]"),
    ("dbPassword=hunter2", "[REDACTED_SECRET]"),
    ("Config: accessToken=opaque-example", "Config: [REDACTED_SECRET]"),
    ("accessToken: Bearer opaque-example", "[REDACTED_SECRET]"),
    ("--accessToken opaque-example", "[REDACTED_SECRET]"),
    ('--mode safe --dbPassword "hunter 2" --count 3', '--mode safe [REDACTED_SECRET] --count 3'),
])
def test_write_trace_redacts_complete_assignment_keys(text, expected):
    service = _StoredService()
    result = write_memory(service, text=text, session_id="s1")
    assert result["ok"] is True
    turns = service.store._read_state("acct-1", collections=("turns",), session_id="s1")["turns"]
    assert next(iter(turns.values()))["content"][0]["prompt"] == expected

@pytest.mark.parametrize("text", [
    "DB_PASSWORD=\"abc'def\"",
    r'DB_PASSWORD="abc\"def"',
    "DB_PASSWORD='abc\"def'",
    r'--password "abc\"def"',
    "--password \"abc'def\"",
    "dbPassword: 'abc''def'",
    "dbPassword: 'abc''''def'",
    "password: 'abc''def'",
    "--dbPassword 'abc''def'",
    'DB_PASSWORD="prefix actual-secret',
    "DB_PASSWORD='prefix actual-secret",
    "DB_PASSWORD=\"prefix \"'actual-secret'",
    'DB_PASSWORD=prefix"actual secret"tail',
    'DB_PASSWORD="prefix actual-secret\\',
    r'DB_PASSWORD=prefix\ actual-secret',
])
@pytest.mark.parametrize("as_json", [False, True])
def test_write_trace_redacts_complete_quoted_credentials(text, as_json):
    service = _StoredService()
    write_memory(service, text=json.dumps({"output": text}) if as_json else text, session_id="s1")
    turns = service.store._read_state("acct-1", collections=("turns",), session_id="s1")["turns"]
    expected = json.dumps({"output": "[REDACTED_SECRET]"}) if as_json else "[REDACTED_SECRET]"
    assert next(iter(turns.values()))["content"][0]["prompt"] == expected


@pytest.mark.parametrize("text", json.loads(
    (Path(__file__).parents[3] / "plugins/deepseek-harness/tests/fixtures/yaml_secret_cases.json").read_text()
))
@pytest.mark.parametrize("as_json", [False, True])
def test_write_trace_redacts_yaml_multiline_secrets(text, as_json):
    service = _StoredService()
    write_memory(service, text=json.dumps({"output": text}) if as_json else text, session_id="s1")
    turns = service.store._read_state("acct-1", collections=("turns",), session_id="s1")["turns"]
    expected = json.dumps({"output": "[REDACTED_SECRET]"}) if as_json else "[REDACTED_SECRET]"
    assert next(iter(turns.values()))["content"][0]["prompt"] == expected
