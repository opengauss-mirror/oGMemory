from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from pyagfs.exceptions import AGFSClientError

from fs.agfs_adapter.agfs_trace_store import AGFSTraceEventStore, _iso


class _MemoryAGFS:
    def __init__(self):
        self.directories = {"/", "/local"}
        self.files = {}

    def mkdir(self, path):
        path = path.rstrip("/") or "/"
        if path in self.directories:
            raise AGFSClientError("already exists")
        parent = path.rpartition("/")[0] or "/"
        if parent not in self.directories:
            raise AGFSClientError("parent directory does not exist")
        self.directories.add(path)

    def write(self, path, data):
        self.files[path] = bytes(data)

    def mv(self, old_path, new_path):
        if old_path in self.files:
            self.files[new_path] = self.files.pop(old_path)
            return
        raise AGFSClientError("not found")

    def rm(self, path, recursive=False):
        if path in self.files:
            del self.files[path]
            return
        raise AGFSClientError("not found")

    def read(self, path, offset=0, size=-1, stream=False):
        if path not in self.files:
            raise AGFSClientError("not found")
        data = self.files[path]
        return data[offset:] if size < 0 else data[offset : offset + size]

    def ls(self, path):
        parent = path.rstrip("/")
        prefix = parent + "/"
        names = set()
        for directory in self.directories:
            if directory.startswith(prefix):
                suffix = directory[len(prefix) :]
                if suffix and "/" not in suffix:
                    names.add(suffix)
        entries = [{"name": name, "isDir": True} for name in sorted(names)]
        file_names = {
            file_path[len(prefix) :]
            for file_path in self.files
            if file_path.startswith(prefix) and "/" not in file_path[len(prefix) :]
        }
        entries.extend({"name": name, "isDir": False} for name in sorted(file_names))
        return entries


class _StrictMemoryAGFS(_MemoryAGFS):
    def ls(self, path):
        if path.rstrip("/") not in self.directories:
            raise AGFSClientError(f"no such directory: {path}")
        return super().ls(path)


def _store():
    return AGFSTraceEventStore(client=_MemoryAGFS(), mount_prefix="/local")


def test_compaction_rebuilds_legacy_sample_checkpoint_from_full_trace():
    from skills.sanitizer import StreamingTraceSanitizer
    from skills.sedimentation import SkillSedimentationService
    from skills.trace_compaction import TraceEvidenceCompactor
    from tests.unit.skills.test_sedimentation import FakeFS, FakeLLM, _settings

    store = AGFSTraceEventStore(client=_MemoryAGFS(), max_turn_events=4)
    common = {"account_id": "acct", "owner_space": "user:alice", "agent_id": "bot", "session_id": "upgrade"}
    events = [
        {"event_type": "user_message", "prompt": "Read and verify."},
        {"event_type": "tool_start", "tool_name": "read_file", "tool_use_id": "a"},
        {"event_type": "tool_success", "tool_name": "read_file", "tool_use_id": "a", "tool_response": "Read."},
        {"event_type": "tool_start", "tool_name": "verify_file", "tool_use_id": "b"},
        {"event_type": "tool_success", "tool_name": "verify_file", "tool_use_id": "b", "tool_response": "Verified."},
        {"event_type": "turn_end", "assistant_message": "File verified."},
    ]
    for index, event in enumerate(events):
        saved = store.insert(**common, event_id=str(index), hook_type=event["event_type"], content=event)
    finalized = store.finalize_session(**common, stop_event_index=saved["event_index"])
    store.enqueue_skill_candidate_from_trace(account_id="acct", agent_id="bot", session_id="upgrade",
                                            turn_id=finalized["turn_id"], trace_uri=finalized["uri"])
    turn = store.claim_skill_candidates(account_id="acct", worker_id="worker", limit=1, lease_seconds=60)[0]
    retained = store._read_state("acct", collections=("turns",), session_id="upgrade")["turns"][turn["uri"]]
    sanitizer = StreamingTraceSanitizer(allow_failed=True)
    compactor = TraceEvidenceCompactor(max_chars=_settings().max_trace_chars)
    for ordinal, event in enumerate(retained["content"][:2], start=1):
        compactor.feed(ordinal=ordinal, raw_event=event, sanitized_event=sanitizer.feed(event))
    checkpoint = compactor.checkpoint(sanitizer=sanitizer.checkpoint(), done=False)
    checkpoint["policy_version"] = 3
    assert store.save_skill_compaction_checkpoint(account_id="acct", uri=turn["uri"], worker_id="worker",
                                                  checkpoint=checkpoint, lease_seconds=60)
    turn["assessment"] = {"compaction": checkpoint}
    service = SkillSedimentationService(trace_store=store, context_fs=FakeFS(), assessor_llm=FakeLLM([]),
                                       reviewer_llm=FakeLLM([]), settings=_settings())

    trace, metrics = service._load_trace_evidence(turn, "worker", lambda: False)

    assert trace.tools == {"read_file", "verify_file"}
    assert metrics["compaction_resume_count"] == 0
    assert trace.events[-1]["assistant_message"] == "File verified."


def test_finalize_session_treats_missing_agfs_directory_as_empty():
    store = AGFSTraceEventStore(client=_StrictMemoryAGFS(), mount_prefix="/local")
    common = {
        "account_id": "acct",
        "owner_space": "user:alice",
        "agent_id": "bot",
        "session_id": "session",
    }
    store.insert(
        **common,
        event_id="event-1",
        hook_type="user_message",
        content={"event_type": "user_message", "prompt": "Review release."},
    )
    stop = store.insert(
        **common,
        event_id="event-2",
        hook_type="turn_end",
        content={"event_type": "turn_end", "assistant_message": "Reviewed."},
    )

    turn = store.finalize_session(**common, stop_event_index=stop["event_index"])

    assert turn["status"] == "finalized"


def test_agfs_trace_store_creates_nested_mount_prefix():
    client = _MemoryAGFS()

    AGFSTraceEventStore(client=client, mount_prefix="/local/plugin")

    assert "/local/plugin/.ogmemory/trace-store/accounts" in client.directories


def test_agfs_trace_store_does_not_create_the_mount_point():
    client = _MemoryAGFS()
    original_mkdir = client.mkdir

    def reject_mount_mkdir(path):
        if path == "/local":
            raise AGFSClientError("permission denied (use mount instead)")
        original_mkdir(path)

    client.mkdir = reject_mount_mkdir

    AGFSTraceEventStore(client=client, mount_prefix="/local")

    assert "/local/.ogmemory/trace-store/accounts" in client.directories


def test_trace_hooks_write_bounded_event_shards_and_small_metadata():
    client = _MemoryAGFS()
    write_sizes = []
    original_write = client.write

    def record_write(path, data):
        write_sizes.append((path, len(data)))
        original_write(path, data)

    client.write = record_write
    store = AGFSTraceEventStore(client=client, mount_prefix="/local")

    for index in range(100):
        store.insert(
            event_id=f"event-{index}",
            account_id="acct",
            owner_space="user:alice",
            agent_id="bot",
            session_id="session",
            hook_type="PreToolUse",
            content={"event_type": "tool_start", "payload": "x" * 512},
        )

    metadata = json.loads(client.files[store._state_path("acct")])
    event_files = [
        data for path, data in client.files.items() if "/events/" in path and path.endswith(".json")
    ]
    assert "events" not in metadata
    assert metadata["next_event_index"] == 101
    assert len(event_files) == 100
    assert len(client.files[store._state_path("acct")]) < 256
    assert max(size for _, size in write_sizes) < 2048


def test_state_backup_is_read_after_interrupted_replacement():
    client = _MemoryAGFS()
    store = AGFSTraceEventStore(client=client, mount_prefix="/local")
    first = store.insert(
        event_id="event-1",
        account_id="acct",
        owner_space="user:alice",
        agent_id="bot",
        session_id="session",
        hook_type="PreToolUse",
        content={"event_type": "tool_start"},
    )
    state_path = store._state_path("acct")
    client.mv(state_path, state_path + ".bak")

    restarted = AGFSTraceEventStore(client=client, mount_prefix="/local")
    second = restarted.insert(
        event_id="event-2",
        account_id="acct",
        owner_space="user:alice",
        agent_id="bot",
        session_id="session",
        hook_type="PreToolUse",
        content={"event_type": "tool_start"},
    )

    assert (first["event_index"], second["event_index"]) == (1, 2)
    assert state_path in client.files


def test_failed_event_write_is_repaired_from_the_idempotency_index():
    client = _MemoryAGFS()
    original_write = client.write
    fail_once = [True]

    def fail_event(path, data):
        if fail_once[0] and "/events/" in path:
            fail_once[0] = False
            raise AGFSClientError("injected event write failure")
        original_write(path, data)

    client.write = fail_event
    store = AGFSTraceEventStore(client=client, mount_prefix="/local")
    common = {
        "account_id": "acct",
        "owner_space": "user:alice",
        "agent_id": "bot",
        "session_id": "session",
        "hook_type": "PreToolUse",
        "content": {"event_type": "tool_start"},
    }

    with pytest.raises(AGFSClientError, match="injected"):
        store.insert(event_id="event-1", **common)
    retried = store.insert(event_id="event-1", **common)
    following = store.insert(event_id="event-2", **common)

    assert retried["inserted"] is False
    assert (retried["event_index"], following["event_index"]) == (1, 2)


def test_event_idempotency_is_scoped_to_account_across_sessions():
    store = _store()
    common = {
        "event_id": "event-1",
        "account_id": "acct",
        "owner_space": "user:alice",
        "agent_id": "bot",
        "hook_type": "PreToolUse",
        "content": {"event_type": "tool_start"},
    }

    first = store.insert(**common, session_id="session-a")
    duplicate = store.insert(**common, session_id="session-b")

    assert duplicate == first | {"inserted": False}


def test_legacy_monolithic_state_is_migrated_to_record_shards():
    client = _MemoryAGFS()
    store = AGFSTraceEventStore(client=client, mount_prefix="/local")
    account = "acct"
    store._mkdir_p(store._account_dir(account))
    legacy = store._empty_state(account)
    legacy["schema_version"] = 1
    legacy["events"]["event-1"] = {
        "event_id": "event-1",
        "uri": "ctx://acct/traces/s/events/event-1",
        "session_id": "s",
        "event_index": 1,
    }
    client.write(store._state_path(account), json.dumps(legacy).encode())

    migrated = store._read_state(account, collections=("events",))

    metadata = json.loads(client.files[store._state_path(account)])
    assert migrated["events"]["event-1"]["event_index"] == 1
    assert metadata["schema_version"] == 2
    assert "events" not in metadata
    assert store._read_record(account, "events", "event-1", "s") is not None
    assert store._read_record(account, "event_ids", "event-1") is not None


def test_flat_session_shards_are_migrated_once():
    client = _MemoryAGFS()
    store = AGFSTraceEventStore(client=client, mount_prefix="/local")
    account = "acct"
    event = {
        "event_id": "event-1",
        "uri": "ctx://acct/traces/s/events/event-1",
        "session_id": "s",
        "event_index": 1,
    }
    state = store._empty_state(account)
    state.pop("session_partitioned")
    state.pop("event_ids_indexed")
    store._mkdir_p(store._collection_dir(account, "events"))
    client.write(
        store._record_path(account, "events", "event-1"),
        json.dumps(event).encode(),
    )
    store._mkdir_p(store._account_dir(account))
    client.write(store._state_path(account), json.dumps(state).encode())

    migrated = store._read_state(account, collections=("events",), session_id="s")

    metadata = json.loads(client.files[store._state_path(account)])
    assert migrated["events"]["event-1"] == event
    assert metadata["session_partitioned"] is True
    assert store._record_path(account, "events", "event-1") not in client.files
    assert store._read_record(account, "events", "event-1", "s") == event
    assert store._read_record(account, "event_ids", "event-1")["session_id"] == "s"


def test_finalize_session_requests_do_not_grow_with_unrelated_history():
    def finalize_read_count(history_sessions):
        client = _MemoryAGFS()
        store = AGFSTraceEventStore(client=client, mount_prefix="/local")
        for index in range(history_sessions):
            common = {
                "account_id": "acct",
                "owner_space": "user:alice",
                "agent_id": "bot",
                "session_id": f"old-{index}",
            }
            stop = store.insert(
                **common,
                event_id=f"old-{index}",
                hook_type="Stop",
                content={},
            )
            store.finalize_session(**common, stop_event_index=stop["event_index"])
        current = {
            "account_id": "acct",
            "owner_space": "user:alice",
            "agent_id": "bot",
            "session_id": "current",
        }
        stop = store.insert(
            **current,
            event_id="current",
            hook_type="Stop",
            content={},
        )
        reads = 0
        original_read = client.read

        def count_read(*args, **kwargs):
            nonlocal reads
            reads += 1
            return original_read(*args, **kwargs)

        client.read = count_read
        store.finalize_session(**current, stop_event_index=stop["event_index"])
        return reads

    assert finalize_read_count(0) == finalize_read_count(25)


def test_candidate_claim_requests_do_not_grow_with_terminal_history():
    def claim_request_count(history):
        client = _MemoryAGFS()
        store = AGFSTraceEventStore(client=client, mount_prefix="/local")

        def add_candidate(session_id):
            common = {
                "account_id": "acct",
                "owner_space": "user:alice",
                "agent_id": "bot",
                "session_id": session_id,
            }
            stop = store.insert(
                **common,
                event_id=f"event-{session_id}",
                hook_type="Stop",
                content={},
            )
            turn = store.finalize_session(**common, stop_event_index=stop["event_index"])
            store.enqueue_skill_candidate_from_trace(
                account_id="acct",
                agent_id="bot",
                session_id=session_id,
                turn_id=turn["turn_id"],
                trace_uri=turn["uri"],
            )
            return turn["uri"]

        for index in range(history):
            uri = add_candidate(f"old-{index}")
            store.claim_skill_candidates(
                account_id="acct", worker_id="worker", limit=1, lease_seconds=60
            )
            store.reject_skill_candidate(
                account_id="acct",
                uri=uri,
                worker_id="worker",
                state="skipped",
                error="done",
            )
        target = add_candidate("target")
        calls = {"read": 0, "ls": 0}
        original_read, original_ls = client.read, client.ls

        def count_read(*args, **kwargs):
            calls["read"] += 1
            return original_read(*args, **kwargs)

        def count_ls(*args, **kwargs):
            calls["ls"] += 1
            return original_ls(*args, **kwargs)

        client.read, client.ls = count_read, count_ls
        claimed = store.claim_skill_candidates(
            account_id="acct", worker_id="worker", limit=1, lease_seconds=60
        )
        assert claimed[0]["uri"] == target
        return calls["read"], calls["ls"]

    assert claim_request_count(0) == claim_request_count(25)


def test_evidence_claim_requests_do_not_grow_with_terminal_history():
    def claim_request_count(history):
        client = _MemoryAGFS()
        store = AGFSTraceEventStore(client=client, mount_prefix="/local")
        store._write_state(store._empty_state("acct"))
        for index in range(history + 1):
            uri = f"ctx://acct/traces/session-{index}/turns/turn-{index}"
            candidate = {
                "candidate_id": f"ctx://acct/skills/candidates/turn-{index}",
                "account_id": "acct",
                "agent_id": "bot",
                "session_id": f"session-{index}",
                "evidence_uri": uri,
                "claim_turn": {"ended_at": _iso(), "status": "finalized"},
                "state": "consumed" if index < history else "assessed",
                "capability_key": "old" if index < history else "target",
                "assessment": {"source_events": []},
                "attempts": 1,
                "compile_attempts": 0,
                "next_attempt_at": _iso(),
                "lease_owner": None,
                "lease_until": None,
                "skill_uri": None,
                "error": None,
                "created_at": _iso(),
                "updated_at": _iso(),
            }
            store._write_record("acct", "candidates", candidate)
            if index == history:
                store._write_record(
                    "acct",
                    "candidate_work",
                    {"candidate_id": candidate["candidate_id"]},
                )
        counts = {"read": 0, "ls": 0}
        original_read = client.read
        original_ls = client.ls

        def read(*args, **kwargs):
            counts["read"] += 1
            return original_read(*args, **kwargs)

        def ls(*args, **kwargs):
            counts["ls"] += 1
            return original_ls(*args, **kwargs)

        client.read = read
        client.ls = ls
        evidence = store.claim_skill_evidence(
            account_id="acct",
            agent_id="bot",
            capability_key="target",
            worker_id="worker",
            min_evidence_turns=1,
            min_distinct_sessions=1,
            max_evidence_turns=1,
            lease_seconds=60,
        )
        assert len(evidence) == 1
        return counts

    assert claim_request_count(0) == claim_request_count(25)


def test_checkpoint_requests_do_not_grow_with_terminal_history():
    def checkpoint_request_count(history):
        client = _MemoryAGFS()
        store = AGFSTraceEventStore(client=client, mount_prefix="/local")
        store._write_state(store._empty_state("acct"))
        for index in range(history + 1):
            uri = f"ctx://acct/traces/session-{index}/turns/turn-{index}"
            candidate = {
                "candidate_id": f"ctx://acct/skills/candidates/turn-{index}",
                "account_id": "acct",
                "agent_id": "bot",
                "session_id": f"session-{index}",
                "evidence_uri": uri,
                "state": "consumed" if index < history else "processing",
                "capability_key": "old" if index < history else None,
                "assessment": {},
                "lease_owner": None if index < history else "worker",
                "lease_until": (
                    None if index < history else _iso(datetime.now(UTC) + timedelta(minutes=5))
                ),
            }
            store._write_record("acct", "candidates", candidate)
        store._recovered_accounts.add("acct")
        counts = {"read": 0, "ls": 0}
        original_read = client.read
        original_ls = client.ls

        def read(*args, **kwargs):
            counts["read"] += 1
            return original_read(*args, **kwargs)

        def ls(*args, **kwargs):
            counts["ls"] += 1
            return original_ls(*args, **kwargs)

        client.read = read
        client.ls = ls
        assert store.save_skill_compaction_checkpoint(
            account_id="acct",
            uri=f"ctx://acct/traces/session-{history}/turns/turn-{history}",
            worker_id="worker",
            checkpoint={"cursor": 1},
            lease_seconds=60,
        )
        return counts

    assert checkpoint_request_count(0) == checkpoint_request_count(25)


def test_candidate_listing_requests_do_not_grow_with_unrelated_terminal_history():
    def list_requests(history):
        client = _MemoryAGFS()
        store = AGFSTraceEventStore(client=client, mount_prefix="/local")
        store._write_state(store._empty_state("acct"))
        for index in range(history):
            store._write_candidate_state(
                "acct",
                {
                    "candidate_id": f"ctx://acct/skills/candidates/old-{index}",
                    "account_id": "acct",
                    "agent_id": "other",
                    "session_id": f"old-{index}",
                    "evidence_uri": f"ctx://acct/traces/old/turns/old-{index}",
                    "state": "held",
                    "capability_key": "old",
                    "assessment": {},
                },
            )
        candidate = {
            "candidate_id": "ctx://acct/skills/candidates/current",
            "account_id": "acct",
            "agent_id": "bot",
            "session_id": "current",
            "evidence_uri": "ctx://acct/traces/current/turns/current",
            "state": "assessed",
            "capability_key": "target",
            "assessment": {"summary": "ready"},
            "updated_at": _iso(),
        }
        store._write_candidate_state("acct", candidate)
        counts = {"read": 0, "ls": 0}
        original_read = client.read
        original_ls = client.ls

        def read(*args, **kwargs):
            counts["read"] += 1
            return original_read(*args, **kwargs)

        def ls(*args, **kwargs):
            counts["ls"] += 1
            return original_ls(*args, **kwargs)

        client.read = read
        client.ls = ls
        assert store.list_skill_candidates(account_id="acct", agent_id="bot") == [
            {
                "capability_key": "target",
                "state": "accumulating",
                "evidence_count": 1,
                "distinct_session_count": 1,
                "summary": "ready",
                "reason": "",
            }
        ]
        return counts

    assert list_requests(0) == list_requests(25)


def test_session_count_requests_do_not_grow_with_unrelated_terminal_history():
    def count_requests(history):
        client = _MemoryAGFS()
        store = AGFSTraceEventStore(client=client, mount_prefix="/local")
        store._write_state(store._empty_state("acct"))
        for index in range(history):
            store._write_record(
                "acct",
                "candidates",
                {
                    "candidate_id": f"ctx://acct/skills/candidates/old-{index}",
                    "evidence_uri": f"ctx://acct/traces/old/turns/old-{index}",
                    "session_id": f"old-{index}",
                    "state": "consumed",
                    "skill_uri": "other-skill",
                },
            )
        uri = "ctx://acct/traces/current/turns/current"
        store._write_record(
            "acct",
            "candidates",
            {
                "candidate_id": "ctx://acct/skills/candidates/current",
                "evidence_uri": uri,
                "session_id": "current",
                "state": "processing",
                "lease_owner": "worker",
                "lease_until": _iso(datetime.now(UTC) + timedelta(minutes=5)),
            },
        )
        counts = {"read": 0, "ls": 0}
        original_read = client.read
        original_ls = client.ls

        def read(*args, **kwargs):
            counts["read"] += 1
            return original_read(*args, **kwargs)

        def ls(*args, **kwargs):
            counts["ls"] += 1
            return original_ls(*args, **kwargs)

        client.read = read
        client.ls = ls
        assert (
            store.count_skill_distinct_sessions(
                account_id="acct",
                skill_uri="target-skill",
                evidence_uris=[uri],
                worker_id="worker",
            )
            == 1
        )
        return counts

    assert count_requests(0) == count_requests(25)


def test_session_ledgers_are_migrated_from_previous_sharded_state():
    client = _MemoryAGFS()
    store = AGFSTraceEventStore(client=client, mount_prefix="/local")
    state = store._empty_state("acct")
    state["schema_version"] = 4
    state.pop("session_ledgers")
    state["example_evidence"] = [
        {
            "skill_uri": "skill",
            "example_id": "primary",
            "example_hash": "a" * 64,
            "session_id": "session",
        }
    ]
    state["_loaded_collections"] = {"example_evidence"}
    state["_persisted_hashes"] = {"example_evidence": {}}
    store._write_state(state)
    store._write_record(
        "acct",
        "candidates",
        {
            "candidate_id": "ctx://acct/skills/candidates/turn",
            "evidence_uri": "ctx://acct/traces/session/turns/turn",
            "session_id": "session",
            "state": "consumed",
            "skill_uri": "skill",
        },
    )

    assert (
        store.count_skill_distinct_sessions(
            account_id="acct", skill_uri="skill", evidence_uris=[], worker_id="worker"
        )
        == 1
    )
    assert store.count_skill_example_distinct_sessions(
        account_id="acct",
        skill_uri="skill",
        examples=[
            {
                "example_id": "primary",
                "example_hash": "a" * 64,
                "evidence_uris": [],
            }
        ],
        worker_id="worker",
    ) == {f"primary:{'a' * 64}": 1}


def test_legacy_session_ledgers_are_migrated_to_bounded_entries():
    client = _MemoryAGFS()
    store = AGFSTraceEventStore(client=client, mount_prefix="/local")
    state = store._empty_state("acct")
    state["schema_version"] = 4
    store._write_state(state)
    skill_uri = "ctx://acct/skill"
    example_hash = "a" * 64
    ledger_key = store._example_ledger_key(skill_uri, "primary", example_hash)
    store._write_record(
        "acct",
        "skill_sessions",
        {"skill_uri": skill_uri, "sessions": ["one", "two"]},
    )
    store._write_record(
        "acct",
        "example_sessions",
        {
            "ledger_key": ledger_key,
            "skill_uri": skill_uri,
            "example_id": "primary",
            "example_hash": example_hash,
            "sessions": ["one", "two"],
        },
    )

    assert (
        store.count_skill_distinct_sessions(
            account_id="acct", skill_uri=skill_uri, evidence_uris=[], worker_id="worker"
        )
        == 2
    )
    assert store.count_skill_example_distinct_sessions(
        account_id="acct",
        skill_uri=skill_uri,
        examples=[
            {
                "example_id": "primary",
                "example_hash": example_hash,
                "evidence_uris": [],
            }
        ],
        worker_id="worker",
    ) == {f"primary:{example_hash}": 2}
    assert json.loads(client.files[store._state_path("acct")])["schema_version"] == 5


def test_session_ledger_writes_remain_bounded_as_history_grows():
    client = _MemoryAGFS()
    write_sizes = []
    original_write = client.write

    def record_write(path, data):
        if "/skill_session_entries/" in path:
            write_sizes.append(len(data))
        original_write(path, data)

    client.write = record_write
    store = AGFSTraceEventStore(client=client, mount_prefix="/local")
    skill_uri = "ctx://acct/skill"

    for index in range(100):
        store._add_session_to_ledger(
            "acct",
            "skill_session_entries",
            skill_uri,
            f"session-{index}",
            skill_uri=skill_uri,
        )

    assert len(write_sizes) == 100
    assert max(write_sizes) < 160


def test_active_candidate_index_is_migrated_from_previous_sharded_state():
    client = _MemoryAGFS()
    store = AGFSTraceEventStore(client=client, mount_prefix="/local")
    state = store._empty_state("acct")
    state.pop("candidate_work_indexed")
    store._write_state(state)
    candidate = {
        "candidate_id": "ctx://acct/skills/candidates/turn",
        "account_id": "acct",
        "agent_id": "bot",
        "session_id": "session",
        "evidence_uri": "ctx://acct/traces/session/turns/turn",
        "claim_turn": {"ended_at": _iso(), "status": "finalized"},
        "state": "assessed",
        "capability_key": "target",
        "assessment": {"source_events": []},
        "next_attempt_at": _iso(),
    }
    store._write_record("acct", "candidates", candidate)

    evidence = store.claim_skill_evidence(
        account_id="acct",
        agent_id="bot",
        capability_key="target",
        worker_id="worker",
        min_evidence_turns=1,
        min_distinct_sessions=1,
        max_evidence_turns=1,
        lease_seconds=60,
    )

    assert len(evidence) == 1
    metadata = json.loads(client.files[store._state_path("acct")])
    assert metadata["candidate_work_indexed"] is True


def test_candidate_listing_index_is_migrated_from_previous_sharded_state():
    client = _MemoryAGFS()
    store = AGFSTraceEventStore(client=client, mount_prefix="/local")
    state = store._empty_state("acct")
    state["schema_version"] = 3
    store._write_state(state)
    store._write_record(
        "acct",
        "candidates",
        {
            "candidate_id": "ctx://acct/skills/candidates/turn",
            "account_id": "acct",
            "agent_id": "bot",
            "session_id": "session",
            "evidence_uri": "ctx://acct/traces/session/turns/turn",
            "state": "held",
            "capability_key": "target",
            "assessment": {},
            "updated_at": _iso(),
        },
    )

    rows = store.list_skill_candidates(account_id="acct", agent_id="bot")

    assert rows[0]["state"] == "held"
    metadata = json.loads(client.files[store._state_path("acct")])
    assert metadata["schema_version"] == 4


def test_duplicate_enqueue_repairs_missing_candidate_claim_index():
    store = _store()
    common = {
        "account_id": "acct",
        "owner_space": "user:alice",
        "agent_id": "bot",
        "session_id": "session",
    }
    stop = store.insert(**common, event_id="event", hook_type="Stop", content={})
    turn = store.finalize_session(**common, stop_event_index=stop["event_index"])
    enqueue = {
        "account_id": "acct",
        "agent_id": "bot",
        "session_id": "session",
        "turn_id": turn["turn_id"],
        "trace_uri": turn["uri"],
    }
    candidate_id = store.enqueue_skill_candidate_from_trace(**enqueue)["candidate_id"]
    store._remove_if_present(store._record_path("acct", "candidate_claims", candidate_id))

    store.enqueue_skill_candidate_from_trace(**enqueue)

    claimed = store.claim_skill_candidates(
        account_id="acct", worker_id="worker", limit=1, lease_seconds=60
    )
    assert [item["uri"] for item in claimed] == [turn["uri"]]


@pytest.mark.parametrize(
    "publication",
    [
        {"capability_key": ""},
        {"evidence_uris": "ctx://acct/trace"},
        {"evidence_uris": [], "example_evidence_keys": [{"example_id": "bad"}]},
    ],
)
def test_agfs_trace_store_rejects_invalid_publication(publication):
    with pytest.raises(ValueError):
        _store().stage_skill_publication(
            account_id="acct",
            uris=["ctx://acct/trace"],
            worker_id="worker",
            publication=publication,
        )


def test_agfs_trace_store_runs_skill_candidate_lifecycle():
    store = _store()
    common = {
        "account_id": "acct",
        "owner_space": "user:alice",
        "agent_id": "bot",
        "session_id": "session-1",
    }
    first = store.insert(
        **common,
        event_id="event-1",
        hook_type="UserPromptSubmit",
        content={"event_type": "user_message", "prompt": "Review release."},
    )
    duplicate = store.insert(
        **common,
        event_id="event-1",
        hook_type="UserPromptSubmit",
        content={"event_type": "user_message", "prompt": "Review release."},
    )
    stop = store.insert(
        **common,
        event_id="event-2",
        hook_type="turn_end",
        content={"event_type": "turn_end", "assistant_message": "Reviewed."},
    )

    assert first["inserted"] is True
    assert duplicate == first | {"inserted": False}

    turn = store.finalize_session(**common, stop_event_index=stop["event_index"])
    queued = store.enqueue_skill_candidate_from_trace(
        account_id="acct",
        agent_id="bot",
        session_id="session-1",
        turn_id=turn["turn_id"],
        trace_uri=turn["uri"],
    )
    assert queued["ok"] is True
    assert store.list_pending_skill_accounts() == ["acct"]

    claimed = store.claim_skill_candidates(
        account_id="acct",
        worker_id="worker",
        limit=1,
        lease_seconds=300,
    )
    assert [item["uri"] for item in claimed] == [turn["uri"]]
    page = store.read_skill_trace_event_page(
        account_id="acct",
        evidence_uri=turn["uri"],
        after_ordinal=0,
        limit=10,
    )
    assert page["done"] is True
    assert len(page["events"]) == 2

    assessment = {
        "summary": "Review a release",
        "source_events": [item["event"] for item in page["events"]],
    }
    assert (
        store.mark_skill_assessed(
            account_id="acct",
            uri=turn["uri"],
            worker_id="worker",
            capability_key="review-release",
            assessment=assessment,
        )
        is True
    )
    ready = store.list_ready_skill_capabilities(
        account_id="acct",
        min_evidence_turns=1,
        min_distinct_sessions=1,
        limit=10,
    )
    assert ready[0]["capability_key"] == "review-release"

    evidence = store.claim_skill_evidence(
        account_id="acct",
        agent_id="bot",
        capability_key="review-release",
        worker_id="worker",
        min_evidence_turns=1,
        min_distinct_sessions=1,
        max_evidence_turns=2,
        lease_seconds=300,
    )
    assert evidence[0]["content"] == assessment["source_events"]
    publication = {
        "capability_key": "review-release",
        "evidence_uris": [turn["uri"]],
        "example_evidence_keys": [
            {
                "example_id": "primary",
                "example_hash": "a" * 64,
                "evidence_uris": [turn["uri"]],
            }
        ],
    }
    assert (
        store.stage_skill_publication(
            account_id="acct",
            uris=[turn["uri"]],
            worker_id="worker",
            publication=publication,
        )
        == 1
    )
    assert (
        store.mark_skill_terminal(
            account_id="acct",
            uris=[turn["uri"]],
            worker_id="worker",
            state="consumed",
            skill_uri="ctx://acct/agents/bot/skills/review-release",
        )
        == 1
    )
    skill_uri = "ctx://acct/agents/bot/skills/review-release"
    assert (
        store.count_skill_distinct_sessions(
            account_id="acct", skill_uri=skill_uri, evidence_uris=[], worker_id="worker"
        )
        == 1
    )
    assert store.count_skill_example_distinct_sessions(
        account_id="acct",
        skill_uri=skill_uri,
        examples=[publication["example_evidence_keys"][0]],
        worker_id="worker",
    ) == {f"primary:{'a' * 64}": 1}
    assert store.list_pending_skill_accounts() == []


def test_trace_event_page_supports_session_ids_with_slashes():
    store = _store()
    common = {
        "account_id": "acct",
        "owner_space": "user:alice",
        "agent_id": "bot",
        "session_id": "group/session",
    }
    stop = store.insert(
        **common,
        event_id="stop",
        hook_type="Stop",
        content={"event_type": "turn_end"},
    )
    turn = store.finalize_session(**common, stop_event_index=stop["event_index"])

    page = store.read_skill_trace_event_page(
        account_id="acct",
        evidence_uri=turn["uri"],
        after_ordinal=0,
        limit=10,
    )

    assert [row["event"] for row in page["events"]] == [{"event_type": "turn_end"}]
    assert page["done"] is True


def test_agfs_trace_store_recovers_processing_candidate_after_restart():
    client = _MemoryAGFS()
    store = AGFSTraceEventStore(client=client, mount_prefix="/local")
    common = {
        "account_id": "acct",
        "owner_space": "user:alice",
        "agent_id": "bot",
        "session_id": "session-1",
    }
    stop = store.insert(
        **common,
        event_id="stop",
        hook_type="turn_end",
        content={"event_type": "turn_end", "assistant_message": "Done."},
    )
    turn = store.finalize_session(**common, stop_event_index=stop["event_index"])
    store.enqueue_skill_candidate_from_trace(
        account_id="acct",
        agent_id="bot",
        session_id="session-1",
        turn_id=turn["turn_id"],
        trace_uri=turn["uri"],
    )
    assert store.claim_skill_candidates(
        account_id="acct",
        worker_id="old",
        limit=1,
        lease_seconds=300,
    )

    restarted = AGFSTraceEventStore(client=client, mount_prefix="/local")
    claimed = restarted.claim_skill_candidates(
        account_id="acct",
        worker_id="new",
        limit=1,
        lease_seconds=300,
    )

    assert len(claimed) == 1


def test_agfs_trace_store_persists_recovery_after_pending_account_scan():
    client = _MemoryAGFS()
    store = AGFSTraceEventStore(client=client, mount_prefix="/local")
    common = {
        "account_id": "acct",
        "owner_space": "user:alice",
        "agent_id": "bot",
        "session_id": "session-1",
    }
    stop = store.insert(
        **common,
        event_id="stop",
        hook_type="turn_end",
        content={"event_type": "turn_end", "assistant_message": "Done."},
    )
    turn = store.finalize_session(**common, stop_event_index=stop["event_index"])
    store.enqueue_skill_candidate_from_trace(
        account_id="acct",
        agent_id="bot",
        session_id="session-1",
        turn_id=turn["turn_id"],
        trace_uri=turn["uri"],
    )
    assert store.claim_skill_candidates(
        account_id="acct",
        worker_id="old",
        limit=1,
        lease_seconds=300,
    )

    restarted = AGFSTraceEventStore(client=client, mount_prefix="/local")
    assert restarted.list_pending_skill_accounts() == ["acct"]

    claimed = restarted.claim_skill_candidates(
        account_id="acct",
        worker_id="new",
        limit=1,
        lease_seconds=300,
    )

    assert len(claimed) == 1


def test_pending_account_limit_is_applied_after_filtering_actionable_work():
    client = _MemoryAGFS()
    store = AGFSTraceEventStore(client=client, mount_prefix="/local")
    dormant, pending = sorted(["acct-a", "acct-b"], key=store._account_key)

    def common(account_id):
        return {
            "account_id": account_id,
            "owner_space": "user:alice",
            "agent_id": "bot",
            "session_id": "session-1",
        }

    store.insert(
        **common(dormant),
        event_id="dormant",
        hook_type="PreToolUse",
        content={"event_type": "tool_start"},
    )
    stop = store.insert(
        **common(pending),
        event_id="stop",
        hook_type="turn_end",
        content={"event_type": "turn_end", "assistant_message": "Done."},
    )
    turn = store.finalize_session(
        **common(pending),
        stop_event_index=stop["event_index"],
    )
    store.enqueue_skill_candidate_from_trace(
        account_id=pending,
        agent_id="bot",
        session_id="session-1",
        turn_id=turn["turn_id"],
        trace_uri=turn["uri"],
    )

    assert store.list_pending_skill_accounts(limit=1) == [pending]


def test_pending_account_requests_do_not_grow_with_dormant_accounts():
    def request_count(dormant_count):
        client = _MemoryAGFS()
        store = AGFSTraceEventStore(client=client, mount_prefix="/local")
        for index in range(dormant_count):
            store._write_state(store._empty_state(f"dormant-{index}"))
        store._write_state(store._empty_state("target"))
        candidate = {
            "candidate_id": "ctx://target/skills/candidates/turn",
            "account_id": "target",
            "agent_id": "bot",
            "session_id": "session",
            "evidence_uri": "ctx://target/traces/session/turns/turn",
            "state": "pending",
            "capability_key": None,
            "assessment": {},
            "next_attempt_at": _iso(),
        }
        store._write_record("target", "candidates", candidate)
        store._write_record(
            "target",
            "candidate_work",
            {"candidate_id": candidate["candidate_id"]},
        )
        assert store.list_pending_skill_accounts(limit=1) == ["target"]
        counts = {"read": 0, "ls": 0}
        original_read = client.read
        original_ls = client.ls

        def read(*args, **kwargs):
            counts["read"] += 1
            return original_read(*args, **kwargs)

        def ls(*args, **kwargs):
            counts["ls"] += 1
            return original_ls(*args, **kwargs)

        client.read = read
        client.ls = ls
        assert store.list_pending_skill_accounts(limit=1) == ["target"]
        return counts

    assert request_count(0) == request_count(25)


def test_new_work_is_discovered_after_account_index_initialization():
    store = _store()
    assert store.list_pending_skill_accounts() == []
    common = {
        "account_id": "acct",
        "owner_space": "user:alice",
        "agent_id": "bot",
        "session_id": "session",
    }
    stop = store.insert(
        **common,
        event_id="stop",
        hook_type="turn_end",
        content={"event_type": "turn_end", "assistant_message": "Done."},
    )
    turn = store.finalize_session(**common, stop_event_index=stop["event_index"])
    store.enqueue_skill_candidate_from_trace(
        account_id="acct",
        agent_id="bot",
        session_id="session",
        turn_id=turn["turn_id"],
        trace_uri=turn["uri"],
    )

    assert store.list_pending_skill_accounts() == ["acct"]


@pytest.mark.parametrize("failed_collection", ("candidate_claims", "candidate_work"))
def test_interrupted_enqueue_repairs_candidate_indexes_after_restart(failed_collection):
    client = _MemoryAGFS()
    store = AGFSTraceEventStore(client=client, mount_prefix="/local")
    assert store.list_pending_skill_accounts() == []
    common = {
        "account_id": "acct",
        "owner_space": "user:alice",
        "agent_id": "bot",
        "session_id": "session",
    }
    stop = store.insert(**common, event_id="stop", hook_type="Stop", content={})
    turn = store.finalize_session(**common, stop_event_index=stop["event_index"])
    original_write = client.write

    def fail_candidate_index(path, data):
        if f"/{failed_collection}/" in path:
            raise AGFSClientError("injected candidate index write failure")
        original_write(path, data)

    client.write = fail_candidate_index
    with pytest.raises(AGFSClientError, match="injected"):
        store.enqueue_skill_candidate_from_trace(
            account_id="acct",
            agent_id="bot",
            session_id="session",
            turn_id=turn["turn_id"],
            trace_uri=turn["uri"],
        )
    client.write = original_write

    restarted = AGFSTraceEventStore(client=client, mount_prefix="/local")

    assert restarted.list_pending_skill_accounts() == ["acct"]
    claimed = restarted.claim_skill_candidates(
        account_id="acct", worker_id="worker", limit=1, lease_seconds=60
    )
    assert [item["uri"] for item in claimed] == [turn["uri"]]


def test_pending_account_discovery_round_robins_actionable_accounts():
    store = _store()

    for account_id in ("acct-a", "acct-b"):
        common = {
            "account_id": account_id,
            "owner_space": "user:alice",
            "agent_id": "bot",
            "session_id": "session-1",
        }
        stop = store.insert(
            **common,
            event_id=f"stop-{account_id}",
            hook_type="turn_end",
            content={"event_type": "turn_end", "assistant_message": "Done."},
        )
        turn = store.finalize_session(
            **common,
            stop_event_index=stop["event_index"],
        )
        store.enqueue_skill_candidate_from_trace(
            account_id=account_id,
            agent_id="bot",
            session_id="session-1",
            turn_id=turn["turn_id"],
            trace_uri=turn["uri"],
        )

    first = store.list_pending_skill_accounts(limit=1)
    second = store.list_pending_skill_accounts(limit=1)

    assert set(first + second) == {"acct-a", "acct-b"}


def test_skill_evidence_claim_preserves_session_diversity_when_limited():
    store = _store()
    start = datetime(2026, 1, 1, tzinfo=UTC)

    for index, session_id in enumerate(("session-a", "session-a", "session-b")):
        common = {
            "account_id": "acct",
            "owner_space": "user:alice",
            "agent_id": "bot",
            "session_id": session_id,
        }
        stop = store.insert(
            **common,
            event_id=f"stop-{index}",
            hook_type="turn_end",
            content={"event_type": "turn_end", "assistant_message": "Done."},
            event_time=start + timedelta(minutes=index),
        )
        turn = store.finalize_session(
            **common,
            stop_event_index=stop["event_index"],
        )
        store.enqueue_skill_candidate_from_trace(
            account_id="acct",
            agent_id="bot",
            session_id=session_id,
            turn_id=turn["turn_id"],
            trace_uri=turn["uri"],
        )
        claimed = store.claim_skill_candidates(
            account_id="acct",
            worker_id="assessor",
            limit=1,
            lease_seconds=300,
        )
        assert store.mark_skill_assessed(
            account_id="acct",
            uri=claimed[0]["uri"],
            worker_id="assessor",
            capability_key="review-release",
            assessment={"summary": "Review"},
        )

    evidence = store.claim_skill_evidence(
        account_id="acct",
        agent_id="bot",
        capability_key="review-release",
        worker_id="compiler",
        min_evidence_turns=2,
        min_distinct_sessions=2,
        max_evidence_turns=2,
        lease_seconds=300,
    )

    assert {item["session_id"] for item in evidence} == {"session-a", "session-b"}


def test_skill_evidence_leases_are_scoped_to_agent():
    store = _store()

    for agent_id in ("agent-a", "agent-b"):
        common = {
            "account_id": "acct",
            "owner_space": "user:alice",
            "agent_id": agent_id,
            "session_id": f"session-{agent_id}",
        }
        stop = store.insert(
            **common,
            event_id=f"stop-{agent_id}",
            hook_type="turn_end",
            content={"event_type": "turn_end", "assistant_message": "Done."},
        )
        turn = store.finalize_session(
            **common,
            stop_event_index=stop["event_index"],
        )
        store.enqueue_skill_candidate_from_trace(
            account_id="acct",
            agent_id=agent_id,
            session_id=common["session_id"],
            turn_id=turn["turn_id"],
            trace_uri=turn["uri"],
        )
        candidate = store.claim_skill_candidates(
            account_id="acct",
            worker_id="assessor",
            limit=1,
            lease_seconds=300,
        )[0]
        assert store.mark_skill_assessed(
            account_id="acct",
            uri=candidate["uri"],
            worker_id="assessor",
            capability_key=f"capability-{agent_id}",
            assessment={},
        )

    for agent_id in ("agent-a", "agent-b"):
        evidence = store.claim_skill_evidence(
            account_id="acct",
            agent_id=agent_id,
            capability_key=f"capability-{agent_id}",
            worker_id=agent_id,
            min_evidence_turns=1,
            min_distinct_sessions=1,
            max_evidence_turns=1,
            lease_seconds=300,
        )
        assert len(evidence) == 1


def test_ready_capability_limit_prioritizes_staged_recovery():
    client = _MemoryAGFS()
    store = AGFSTraceEventStore(client=client, mount_prefix="/local")

    for index, capability in enumerate(("ordinary", "recover-me")):
        common = {
            "account_id": "acct",
            "owner_space": "user:alice",
            "agent_id": "bot",
            "session_id": f"session-{index}",
        }
        stop = store.insert(
            **common,
            event_id=f"stop-{index}",
            hook_type="turn_end",
            content={"event_type": "turn_end", "assistant_message": "Done."},
        )
        turn = store.finalize_session(
            **common,
            stop_event_index=stop["event_index"],
        )
        store.enqueue_skill_candidate_from_trace(
            account_id="acct",
            agent_id="bot",
            session_id=common["session_id"],
            turn_id=turn["turn_id"],
            trace_uri=turn["uri"],
        )
        claimed = store.claim_skill_candidates(
            account_id="acct",
            worker_id="worker",
            limit=1,
            lease_seconds=300,
        )
        assert store.mark_skill_assessed(
            account_id="acct",
            uri=claimed[0]["uri"],
            worker_id="worker",
            capability_key=capability,
            assessment={"summary": capability},
        )

    evidence = store.claim_skill_evidence(
        account_id="acct",
        agent_id="bot",
        capability_key="recover-me",
        worker_id="worker",
        min_evidence_turns=1,
        min_distinct_sessions=1,
        max_evidence_turns=1,
        lease_seconds=300,
    )
    assert (
        store.stage_skill_publication(
            account_id="acct",
            uris=[evidence[0]["uri"]],
            worker_id="worker",
            publication={"capability_key": "recover-me"},
        )
        == 1
    )

    restarted = AGFSTraceEventStore(client=client, mount_prefix="/local")
    ready = restarted.list_ready_skill_capabilities(
        account_id="acct",
        min_evidence_turns=1,
        min_distinct_sessions=1,
        limit=1,
    )

    assert [item["capability_key"] for item in ready] == ["recover-me"]
