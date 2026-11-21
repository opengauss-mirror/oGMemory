import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts.run_index_service import CoordinatedIndexRunner
from service.index_service import IndexService
from service.migration.maintenance import MaintenanceError, MaintenanceStore
from service.migration.participants import wait_participants


def config(tmp_path):
    return SimpleNamespace(
        migration_workspace=str(tmp_path),
        account_id="only-account",
        index_interval=3600,
        index_workers=3,
    )


def factory(events):
    def build(cfg, admission):
        assert admission._operation.get().write
        service = IndexService(
            Mock(),
            None,
            None,
            lambda: [cfg.account_id],
            interval_seconds=cfg.index_interval,
            worker_count=cfg.index_workers,
            admission=admission,
        )
        events.append(service)
        return service, lambda: events.append("closed")

    return build


def eventually(predicate):
    until = time.monotonic() + 3
    while not predicate():
        if time.monotonic() > until:
            pytest.fail("lifecycle did not finish")
        time.sleep(0.01)


@pytest.mark.parametrize("mode", ["export", "import"])
def test_real_scheduler_worker_is_blocked_and_clients_reopen(tmp_path, mode):
    events = []
    runner = CoordinatedIndexRunner(config(tmp_path), build_service=factory(events))
    old = runner.service
    eventually(lambda: len(old.schedulers) == 3)
    assert old._get_account_ids() == ["only-account"]
    runner.store.enter("task", mode=mode, resources={})
    try:
        with pytest.raises(MaintenanceError):
            old._worker.run_once(Mock(), ["only-account"])
        if mode == "import":
            wait_participants(runner.store, "task", "drain", timeout=3)
            assert runner.service is None and events.count("closed") == 1
            runner.store.request_refresh("task")
            wait_participants(runner.store, "task", "refresh", timeout=3)
        else:
            assert runner.service is old
        runner.store.reopen("task")
        if mode == "import":
            eventually(lambda: runner.service is not None)
            assert runner.service is not old
            eventually(lambda: len(runner.service.schedulers) == 3)
        assert runner.service._get_account_ids() == ["only-account"]
    finally:
        runner.close()


def test_process_start_during_maintenance_has_no_storage_side_effects(tmp_path):
    cfg = config(tmp_path)
    gate = MaintenanceStore(tmp_path / "maintenance.sqlite", str(tmp_path.resolve()))
    gate.enter("task", mode="import", resources={})
    events = []
    runner = CoordinatedIndexRunner(cfg, build_service=factory(events))
    try:
        assert not events and runner.service is None
        wait_participants(gate, "task", "drain", timeout=3)
        with pytest.raises(MaintenanceError):
            runner.close()
        assert runner.participant._thread.is_alive()
        gate.request_refresh("task")
        wait_participants(gate, "task", "refresh", timeout=3)
        gate.reopen("task")
        eventually(lambda: runner.service is not None)
    finally:
        runner.close()


def test_inflight_work_blocks_pause_before_resource_close(tmp_path):
    events = []
    runner = CoordinatedIndexRunner(config(tmp_path), build_service=factory(events))
    entered, finish = threading.Event(), threading.Event()

    def work():
        with runner.admission.scope(write=True):
            entered.set()
            finish.wait(3)

    thread = threading.Thread(target=work)
    thread.start()
    assert entered.wait(2)
    runner.store.enter("task", mode="import", resources={})
    try:
        with pytest.raises(TimeoutError):
            runner.store.wait_drained("task", 0.05)
        assert "closed" not in events
        finish.set()
        thread.join(2)
        wait_participants(runner.store, "task", "drain", timeout=3)
        assert events.count("closed") == 1
        runner.store.request_refresh("task")
        wait_participants(runner.store, "task", "refresh", timeout=3)
        runner.store.reopen("task")
    finally:
        finish.set()
        thread.join(2)
        runner.close()


@pytest.mark.parametrize("mode", ["export", "import"])
def test_official_factory_guards_concrete_storage_without_vector_provider(
    tmp_path, monkeypatch, mode
):
    from core.models import ContextNode, RequestContext, Role
    from fs.agfs_adapter.client import NativeAGFSHTTPError
    from providers.config import ProviderConfig
    from providers.unified_config import OgMemConfig
    from tests.unit.service.migration.test_tree_backup import LocalClient

    class Client(LocalClient):
        def mkdir(self, path):
            try:
                super().mkdir(path)
            except FileExistsError as exc:
                raise NativeAGFSHTTPError("exists", status_code=409) from exc

    root = tmp_path / "files"
    (root / "local").mkdir(parents=True)
    monkeypatch.setattr("fs.agfs_adapter.client.create_agfs_client", lambda *a, **kw: Client(root))
    monkeypatch.setattr(ProviderConfig, "create_llm", lambda self: Mock())
    monkeypatch.setattr(ProviderConfig, "create_embedder", lambda self: pytest.fail("no embedder"))
    monkeypatch.setattr(
        ProviderConfig, "create_vector_index", lambda self: pytest.fail("no vectors")
    )
    cfg = OgMemConfig(
        storage_backend="agfs",
        retrieval_mode="agfs",
        skill_sedimentation_enabled=False,
        index_interval=3600,
        index_workers=2,
        migration_workspace=str(tmp_path / "control"),
        agfs_mount_prefix="/local/review",
    )
    runner = CoordinatedIndexRunner(cfg)
    fs = runner.service._fs
    ctx = RequestContext("acme", "alice", "", "s", "review", Role.ROOT)
    node = ContextNode(
        uri="ctx://acme/users/alice/memories/preferences/blocked",
        context_type="MEMORY",
        category="preference",
        level=0,
        owner_space="user:alice",
        content="must not be written",
        abstract="",
        overview="",
    )
    runner.store.enter("task", mode=mode, resources={})
    try:
        with pytest.raises(MaintenanceError):
            fs.write_node(node, ctx)
        assert not list(root.rglob("content.md"))
        if mode == "import":
            wait_participants(runner.store, "task", "drain", timeout=3)
            runner.store.request_refresh("task")
            wait_participants(runner.store, "task", "refresh", timeout=3)
        runner.store.reopen("task")
    finally:
        runner.close()


def test_unknown_worker_write_is_not_retired_by_maintenance(tmp_path):
    events = []
    runner = CoordinatedIndexRunner(config(tmp_path), build_service=factory(events))
    with pytest.raises(TimeoutError):
        with runner.admission.backend(write=True):
            raise TimeoutError("remote completion unknown")
    runner.store.enter("task", mode="import", resources={})
    try:
        with pytest.raises(TimeoutError):
            runner.store.wait_drained("task", 0.03)
        assert "closed" not in events
        with runner.store._transaction() as db:
            tokens = [row[0] for row in db.execute("SELECT token FROM operations")]
        assert len(tokens) == 1
        # Only this synthetic test reconciles its own injected operation.
        runner.store.finish_operation(tokens[0])
        wait_participants(runner.store, "task", "drain", timeout=3)
        runner.store.request_refresh("task")
        wait_participants(runner.store, "task", "refresh", timeout=3)
        runner.store.reopen("task")
    finally:
        runner.close()


def test_coordinated_factory_preserves_lexical_projection_and_close_retry(tmp_path, monkeypatch):
    from providers.config import ProviderConfig
    from providers.unified_config import OgMemConfig
    from scripts import run_index_service as runner_module
    from service.migration.admission import Admission

    class Lexical:
        def mark_scopes_dirty(self, scopes):
            return None
    class Projection:
        lexical_index = Lexical()
        finished = False
        def close(self):
            return self.finished
    class Storage:
        _client = SimpleNamespace()
    projection = Projection()
    cfg = OgMemConfig(storage_backend='agfs', retrieval_mode='vector_free',
                      retrieval_lexical_provider='opengauss_native')
    monkeypatch.setattr('providers.composite_index.create_projection_writer', lambda cfg, vector: projection)
    monkeypatch.setattr(ProviderConfig, 'create_embedder', lambda self: pytest.fail('no model'))
    monkeypatch.setattr(ProviderConfig, 'create_vector_index', lambda self: pytest.fail('no vector'))
    monkeypatch.setattr(ProviderConfig, 'create_llm', lambda self: SimpleNamespace())
    seen = []
    def storage(cfg, *, invalidate_lexical_scopes=None):
        seen.append(invalidate_lexical_scopes)
        return Storage(), SimpleNamespace(), None
    monkeypatch.setattr(runner_module, '_create_storage', storage)
    gate = MaintenanceStore(tmp_path / 'gate.db', 'factory')
    admission = Admission(gate)
    with admission.scope(write=True):
        service, cleanup = runner_module._build_service(cfg, admission)
    assert service._vector_index is projection
    assert callable(seen[0])
    with pytest.raises(RuntimeError, match='finished closing'):
        cleanup()
    projection.finished = True
    cleanup()
