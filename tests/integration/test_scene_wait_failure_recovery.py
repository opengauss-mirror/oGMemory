"""Scene idle waits follow each owner's current queue through recovery."""

from threading import Event
from types import SimpleNamespace

import httpx
import pytest
from openai import RateLimitError

from core.models import RequestContext
from providers.unified_config import OgMemConfig
from scene.decision import SceneDecisionEngine
from scene.decision_contracts import SceneDecisionError
from scene.jobs import InMemorySceneJobStore, SceneWorker
from scene.manager import SceneMemoryManager
from server.memory_service import MemoryService
from tests.integration.test_scene_sql_persistence import scene_sql as scene_sql
from tests.unit.scene.test_decision_preflight import JSONModel
from tests.unit.scene.test_engine_refresh import source_case


@pytest.fixture(params=[False, True], ids=["memory", "sql"])
def wait_runtime(request, monkeypatch):
    jobs = request.getfixturevalue("scene_sql")[3] if request.param else InMemorySceneJobStore()
    source, nodes, _, proposal = source_case()
    entered, release = Event(), Event()

    class RecoveringModel(JSONModel):
        def complete_json(self, prompt, schema):
            entered.set()
            assert release.wait(3)
            return super().complete_json(prompt, schema)

    model = RecoveringModel({"status": "READY"}, proposal.model_dump(), {"decision": "ACCEPT"})
    manager = SceneMemoryManager(jobs, source, SceneDecisionEngine(model))
    service = MemoryService(config=OgMemConfig(provider="mock", scene_enabled=True))
    service._scene_manager = manager
    monkeypatch.setattr(service, "_ensure_scene_runtime", lambda: None)
    monkeypatch.setattr(service, "_wait_until_session_idle", lambda params: {"idle": True, "reason": "idle"})
    ctx = RequestContext("acct", "alice", "helper", "session", "trace")
    yield SimpleNamespace(manager=manager, service=service, source=source, uris=[node.uri for node in nodes],
                          entered=entered, release=release, params={
                              "_ctx": ctx, "sessionId": "session", "waitScenes": True, "timeoutSeconds": 2,
                          })
    release.set()
    assert manager.stop(timeout=3)


def _failed_job(runtime, owner="user:alice", *, halted=False):
    failure = (RateLimitError("provider rate limit", response=httpx.Response(
        429, request=httpx.Request("POST", "http://provider.test/v1")), body=None)
        if halted else SceneDecisionError("terminal decision failure"))

    class FailedEngine:
        config = runtime.manager.engine.config

        def refresh(self, *args, **kwargs):
            raise failure

    worker = runtime.manager.worker
    engine = worker.engine
    worker.engine = FailedEngine()
    try:
        runtime.manager.after_memory_write("acct", owner, runtime.uris, force=True)
        result = runtime.manager.process_next("failed-worker")
    finally:
        worker.engine = engine
    assert result.job.status == "FAILED"
    return result.job


@pytest.mark.parametrize("recovery_state", ["PENDING", "RUNNING", "backlog"])
def test_wait_scenes_follows_successor_until_publication(wait_runtime, monkeypatch, recovery_state):
    runtime = wait_runtime
    manager = runtime.manager
    failed = _failed_job(runtime)
    if recovery_state == "backlog":
        flush = manager.flush

        def enqueue_after_flush(account, owner):
            result = flush(account, owner)
            if owner == "user:alice":
                manager.after_memory_write(account, owner, runtime.uris, token_count=1)
            return result

        monkeypatch.setattr(manager, "flush", enqueue_after_flush)
    else:
        manager.after_memory_write("acct", "user:alice", runtime.uris, force=True)
        if recovery_state == "RUNNING":
            manager.start(poll_seconds=.01)
            assert runtime.entered.wait(1)

    summary = manager.jobs.summary
    observed = []

    def resume_after_snapshot(account, owner):
        state = summary(account, owner)
        if owner == "user:alice" and not observed:
            observed.append(state)
            runtime.release.set()
            if recovery_state == "backlog":
                manager.jobs.flush(account, owner)
            manager.start(poll_seconds=.01)
        return state

    monkeypatch.setattr(manager.jobs, "summary", resume_after_snapshot)
    result = runtime.service.wait_until_idle(runtime.params)
    initial = observed[0]
    assert initial["last_job"]["job_id"] == failed.job_id
    if recovery_state == "backlog":
        assert initial["pending_nodes"] == len(runtime.uris)
    else:
        assert initial["active_job"]["status"] == recovery_state
    assert result["idle"] is True
    assert result["reason"] == "idle"
    assert result["scenes"]["user:alice"]["last_job"]["status"] == "SUCCEEDED"
    assert runtime.source.repository.snapshot.revision == 1
    assert runtime.source.repository.snapshot.scenes[0].cluster.members


@pytest.mark.parametrize("failure_scope", ["current", "other-owner", "halted"])
def test_wait_scenes_reports_current_failure_then_recovers(wait_runtime, failure_scope):
    runtime = wait_runtime
    manager = runtime.manager
    owner = "agent:helper" if failure_scope == "other-owner" else "user:alice"
    _failed_job(runtime, owner, halted=failure_scope == "halted")
    if failure_scope in {"other-owner", "halted"}:
        manager.after_memory_write("acct", "user:alice", runtime.uris, force=True)
    result = runtime.service.wait_until_idle(runtime.params)
    assert result["reason"] == "scene_failed"
    assert result["scenes"][owner]["last_job"]["status"] == "FAILED"
    if failure_scope == "halted":
        assert manager.halted
        manager.worker = SceneWorker(manager.jobs, manager.engine, manager.scene_store)
    elif failure_scope == "current":
        manager.after_memory_write("acct", "user:alice", runtime.uris, force=True)
    runtime.release.set()
    completed = manager.process_next("recovered-worker")
    assert completed.job.status == "SUCCEEDED"
    assert runtime.source.repository.snapshot.revision == 1
    if failure_scope == "other-owner":
        manager.after_memory_write("acct", owner, ["ctx://acct/agents/helper/memories/events/recovered"], force=True)
        agent_job = manager.jobs.claim("agent-worker")
        assert agent_job.owner_space == owner
        manager.jobs.mark_succeeded(agent_job, reason="CREATE")
    recovered = runtime.service.wait_until_idle(runtime.params)
    assert recovered["idle"] is True
    assert recovered["reason"] == "idle"
    assert recovered["scenes"][owner]["last_job"]["status"] == "SUCCEEDED"


def test_wait_scenes_times_out_then_recovers(wait_runtime):
    runtime = wait_runtime
    manager = runtime.manager
    _failed_job(runtime)
    manager.after_memory_write("acct", "user:alice", runtime.uris, force=True)
    timed_out = runtime.service.wait_until_idle({**runtime.params, "timeoutSeconds": 0})
    assert timed_out["reason"] == "scene_in_progress_timeout"
    assert timed_out["scenes"]["user:alice"]["active_job"]["status"] == "PENDING"
    runtime.release.set()
    assert manager.process_next("recovered-worker").job.status == "SUCCEEDED"
    recovered = runtime.service.wait_until_idle(runtime.params)
    assert recovered["idle"] is True
    assert recovered["scenes"]["user:alice"]["last_job"]["status"] == "SUCCEEDED"
    assert runtime.source.repository.snapshot.revision == 1
