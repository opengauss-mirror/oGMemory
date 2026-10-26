from threading import Event

from scene.config import SceneConfig
from scene.decision_contracts import SceneDecisionResult
from scene.jobs import InMemorySceneJobStore, SceneJobStatus
from scene.manager import SceneMemoryManager
from scene.models import SceneSnapshot


def test_background_lifecycle_processes_enqueued_tail():
    entered, release = Event(), Event()

    class Engine:
        config = SceneConfig()

        def refresh(self, store, account_id, owner_space, *, node_uris):
            entered.set()
            release.wait(3)
            return SceneDecisionResult(SceneSnapshot(account_id, owner_space, 1), "CREATE")

    jobs = InMemorySceneJobStore()
    manager = SceneMemoryManager(jobs, object(), Engine())
    manager.after_memory_write("acct", "user:a", ["n1"], token_count=3, notice_id="write-1")
    tail = manager.flush("acct", "user:a")
    manager.start(poll_seconds=.01)
    try:
        assert entered.wait(1)
        assert manager.running
        manager.after_memory_write("acct", "user:a", ["n2"], token_count=3, notice_id="write-2")
        assert jobs.summary("acct", "user:a")["pending_nodes"] == 1
    finally:
        release.set()
        assert manager.stop(timeout=2)
    assert jobs.get(tail.job_id).status == SceneJobStatus.SUCCEEDED


def test_background_rate_limit_is_visible_and_preserves_failure():
    class ProviderError(RuntimeError):
        status_code = 429

    class Engine:
        config = SceneConfig()

        def refresh(self, store, account_id, owner_space, *, node_uris):
            raise ProviderError("rate limited")

    jobs = InMemorySceneJobStore()
    manager = SceneMemoryManager(jobs, object(), Engine())
    job = manager.after_memory_write("acct", "user:a", ["n1"], token_count=3, force=True)
    manager.process_next("worker")
    assert manager.halted
    assert jobs.get(job.job_id).status == SceneJobStatus.FAILED
    assert manager.stop()

    class RecoveredEngine:
        config = SceneConfig()

        def refresh(self, store, account_id, owner_space, *, node_uris):
            return SceneDecisionResult(SceneSnapshot(account_id, owner_space, 1), "CREATE")

    recovered = SceneMemoryManager(jobs, object(), RecoveredEngine())
    recovered.after_memory_write("acct", "user:a", ["n2"], token_count=3, force=True)
    assert recovered.process_next("recovered-worker").job.status == SceneJobStatus.SUCCEEDED
