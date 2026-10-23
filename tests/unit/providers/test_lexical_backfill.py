import threading
import time
from types import SimpleNamespace

from core.models import ContextNode
from providers.lexical_backfill import LexicalBackfill


class _Scanner:
    def __init__(self):
        self.calls = 0
        self.quiescent = True

    def projection_is_quiescent(self, **kwargs):
        assert kwargs == {"account_id": "acct"}
        return self.quiescent

    def scan_active_nodes(self, **kwargs):
        self.calls += 1
        assert kwargs == {
            "account_id": "acct",
            "owner_spaces": ("user_u1",),
            "page_size": 500,
        }
        yield ContextNode(
            uri="ctx://acct/users/u1/memories/profile",
            context_type="MEMORY",
            category="profile",
            level=2,
            owner_space="user_u1",
            abstract="profile abstract",
            overview="",
            content="profile content",
            metadata={"when": "now"},
        )


class _Lexical:
    def __init__(self):
        self.ready = frozenset()
        self.records = []
        self.rebuilds = 0
        self.completed = threading.Event()
        self.ready_checks = 0

    def ready_scopes(self, account_id, owner_spaces):
        self.ready_checks += 1
        return self.ready.intersection(owner_spaces)

    def begin_rebuild(self):
        self.rebuilds += 1
        return f"rebuild-{self.rebuilds}"

    def replace_scopes(self, *, rebuild_id, account_id, owner_spaces, records):
        self.records = list(records)
        self.ready = frozenset(owner_spaces)
        self.completed.set()
        return True

    def abort_rebuild(self, rebuild_id):
        pass


def test_backfill_excludes_graph_descriptors(monkeypatch):
    from dataclasses import replace
    import providers.lexical_backfill as module
    original=module.build_index_records
    def with_descriptor(node):
        records=original(node)
        return records+[replace(records[0],id='activation',level=3,
            metadata={'record_kind':'update_activation'})]
    monkeypatch.setattr(module,'build_index_records',with_descriptor)
    lexical=_Lexical();backfill=LexicalBackfill(_Scanner(),lexical)
    try:
        assert backfill._run_once(('acct','user_u1'))
        assert {record.level for record in lexical.records}=={0,2}
    finally:backfill.close(timeout=1.)


def test_missing_scope_schedules_background_backfill_and_then_becomes_ready():
    scanner = _Scanner()
    lexical = _Lexical()
    backfill = LexicalBackfill(scanner, lexical, page_size=500)
    ctx = SimpleNamespace(account_id="acct")

    assert backfill.ensure_ready(ctx, ["user_u1"]) is False
    assert lexical.completed.wait(1.0)
    assert backfill.ensure_ready(ctx, ["user_u1"]) is True
    assert {record.level for record in lexical.records} == {0, 2}
    assert all(record.account_id == "acct" for record in lexical.records)
    assert scanner.calls == 1
    assert lexical.rebuilds == 1
    deadline = time.monotonic() + 1.0
    while (
        backfill.health()["queued"] or backfill.health()["running"]
    ) and time.monotonic() < deadline:
        time.sleep(0.001)
    assert backfill.health()["queued"] == 0
    assert backfill.health()["running"] is False
    assert backfill.close(timeout=1.0) is True


def test_pending_canonical_projection_fails_closed_without_scheduling_backfill():
    scanner = _Scanner()
    scanner.quiescent = False
    lexical = _Lexical()
    backfill = LexicalBackfill(scanner, lexical)

    assert backfill.ensure_ready(SimpleNamespace(account_id="acct"), ["user_u1"]) is False
    assert lexical.ready_checks == 0
    assert lexical.rebuilds == 0
    assert backfill.health()["queued"] == 0
    assert backfill.close() is True


def test_canonical_projection_check_error_propagates_and_fails_closed():
    scanner = _Scanner()
    scanner.projection_is_quiescent = lambda **_kwargs: (_ for _ in ()).throw(
        RuntimeError("canonical unavailable")
    )
    lexical = _Lexical()
    backfill = LexicalBackfill(scanner, lexical)

    try:
        backfill.ensure_ready(SimpleNamespace(account_id="acct"), ["user_u1"])
    except RuntimeError as exc:
        assert str(exc) == "canonical unavailable"
    else:
        raise AssertionError("canonical projection check should fail closed")

    assert lexical.ready_checks == 0
    assert backfill.close() is True


def test_failed_backfill_retries_without_another_user_query():
    scanner = _Scanner()
    lexical = _Lexical()
    original = lexical.replace_scopes
    attempts = 0

    def fail_once(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary")
        return original(**kwargs)

    lexical.replace_scopes = fail_once
    backfill = LexicalBackfill(
        scanner, lexical, page_size=500, retry_base_seconds=0.1
    )

    assert backfill.ensure_ready(SimpleNamespace(account_id="acct"), ["user_u1"]) is False
    assert lexical.completed.wait(1.0)
    assert attempts == 2
    assert backfill.close(timeout=1.0) is True


def test_persistent_failure_stops_after_bounded_attempts():
    scanner = _Scanner()
    lexical = _Lexical()
    lexical.replace_scopes = lambda **_kwargs: (_ for _ in ()).throw(
        RuntimeError("persistent")
    )
    backfill = LexicalBackfill(
        scanner,
        lexical,
        page_size=500,
        retry_base_seconds=0.1,
        max_attempts=2,
        circuit_cooldown_seconds=0.1,
    )

    assert backfill.ensure_ready(SimpleNamespace(account_id="acct"), ["user_u1"]) is False
    deadline = time.monotonic() + 1.0
    while not backfill.health()["terminal_failures"] and time.monotonic() < deadline:
        time.sleep(0.005)

    assert lexical.rebuilds == 2
    assert backfill.health()["terminal_failures"] == 1
    assert backfill.health()["cooldown_failures"] == 1

    lexical.replace_scopes = _Lexical.replace_scopes.__get__(lexical, _Lexical)
    time.sleep(0.11)
    assert backfill.ensure_ready(SimpleNamespace(account_id="acct"), ["user_u1"]) is False
    assert lexical.completed.wait(1.0)
    assert backfill.ensure_ready(SimpleNamespace(account_id="acct"), ["user_u1"]) is True
    assert lexical.rebuilds == 3
    assert backfill.health()["cooldown_failures"] == 0
    assert backfill.close(timeout=1.0) is True
