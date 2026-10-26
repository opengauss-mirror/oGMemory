"""Leased scene-analysis queue with bounded evidence windows."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable
from contextvars import ContextVar
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from threading import Event, RLock, Thread
from typing import Protocol
from uuid import uuid4

from core.models import RequestContext
from providers.token_tracker import request_usage_scope
from retrieval.owner_scope import canonical_owner_space
from scene.config import SceneConfigCache, SceneConfigProvider
from scene.decision import SceneDecisionEngine
from scene.decision_contracts import (
    SceneDecisionError,
    SceneDecisionResult,
    SceneRetryableDecisionError,
)
from scene.store import SceneDecisionStore


class SceneJobStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    WAITING = "WAITING"
    FAILED = "FAILED"


@dataclass(frozen=True)
class SceneJob:
    job_id: str
    account_id: str
    owner_space: str
    node_uris: tuple[str, ...]
    token_count: int = 0
    status: SceneJobStatus = SceneJobStatus.PENDING
    attempt: int = 0
    worker_id: str = ""
    reason: str = ""
    next_run_at: datetime | None = None
    created_at: datetime = datetime.min.replace(tzinfo=UTC)
    lease_token: str = ""
    lease_until: datetime | None = None
    failure_count: int = 0


class SceneLeaseLostError(RuntimeError):
    """The job belongs to a newer claim or its lease has expired."""


current_scene_job: ContextVar[SceneJob | None] = ContextVar("current_scene_job", default=None)


class SceneJobStore(Protocol):
    def enqueue(self, account_id: str, owner_space: str, node_uris: Iterable[str], *,
                token_count: int = 0, trigger_tokens: int = 1, max_pending_nodes: int = 32,
                max_wait_seconds: int = 1800, force: bool = False,
                notice_id: str | None = None) -> SceneJob | None: ...

    def claim(self, worker_id: str, *, lease_seconds: float = 300) -> SceneJob | None: ...

    def renew(self, job: SceneJob, *, lease_seconds: float = 300) -> SceneJob: ...

    def mark_succeeded(self, job: SceneJob, *, reason: str = "",
                       remaining_uris: Iterable[str] = (),
                       min_evidence_nodes: int = 3) -> SceneJob: ...

    def mark_waiting(self, job: SceneJob, *, reason: str = "",
                     remaining_uris: Iterable[str] | None = None) -> SceneJob: ...

    def mark_retry(self, job: SceneJob, *, reason: str, delay_seconds: float) -> SceneJob: ...

    def mark_failed(self, job: SceneJob, *, reason: str) -> SceneJob: ...

    def summary(self, account_id: str, owner_space: str) -> dict: ...

    def flush(self, account_id: str, owner_space: str) -> SceneJob | None: ...

def _utcnow() -> datetime:
    return datetime.now(UTC)


def job_to_dict(job: SceneJob) -> dict:
    result = asdict(job)
    for field in ("created_at", "next_run_at", "lease_until"):
        value = result[field]
        result[field] = value.isoformat() if value else None
    result["status"] = str(job.status)
    return result


def job_from_dict(raw: dict) -> SceneJob:
    values = dict(raw)
    for field in ("created_at", "next_run_at", "lease_until"):
        value = values.get(field)
        values[field] = datetime.fromisoformat(value) if value else None
    values["created_at"] = values["created_at"] or datetime.min.replace(tzinfo=UTC)
    values["status"] = SceneJobStatus(values["status"])
    values["node_uris"] = tuple(values["node_uris"])
    return SceneJob(**values)


class InMemorySceneJobStore:
    """Queue state machine shared by tests and the SQL persistence adapter.

    Claimed windows are immutable. Fresh writes accumulate in an independent
    backlog. WAIT consumes the trigger while retaining evidence for a future
    window. The configured node bound applies to each analysis batch.
    """

    def __init__(self, *, clock: Callable[[], datetime] = _utcnow) -> None:
        self._jobs: dict[str, SceneJob] = {}
        self._scopes: OrderedDict[tuple[str, str], dict] = OrderedDict()
        self._clock = clock
        self._lock = RLock()
        self._notices: set[tuple[str, str, str]] = set()

    def _scope(self, account_id: str, owner_space: str) -> dict:
        return self._scopes.setdefault((account_id, owner_space), {
            "active": None, "last_job": None, "pending": {}, "retained": [], "force": False,
            "trigger_tokens": 1, "max_pending_nodes": 32, "max_wait_seconds": 1800,
        })

    def enqueue(self, account_id: str, owner_space: str, node_uris: Iterable[str], *,
                token_count: int = 0, trigger_tokens: int = 1, max_pending_nodes: int = 32,
                max_wait_seconds: int = 1800, force: bool = False,
                notice_id: str | None = None) -> SceneJob | None:
        if not account_id or not owner_space:
            raise ValueError("account_id and owner_space are required")
        for name, value, minimum in (("token_count", token_count, 0),
                                     ("trigger_tokens", trigger_tokens, 1),
                                     ("max_pending_nodes", max_pending_nodes, 1),
                                     ("max_wait_seconds", max_wait_seconds, 1)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        uris = list(dict.fromkeys(str(uri).strip() for uri in node_uris if str(uri).strip()))
        if not uris:
            return None
        with self._lock:
            state = self._scope(account_id, owner_space)
            if notice_id is not None:
                if not isinstance(notice_id, str) or not 1 <= len(notice_id) <= 512:
                    raise ValueError("notice_id must contain 1..512 characters")
                notice = (account_id, owner_space, notice_id)
                if notice in self._notices:
                    return self._jobs.get(state["active"])
                self._notices.add(notice)
            state.update(trigger_tokens=trigger_tokens, max_pending_nodes=max_pending_nodes,
                         max_wait_seconds=max_wait_seconds)
            state["force"] |= force
            per_uri, extra = divmod(token_count, len(uris))
            now = self._clock().isoformat()
            for index, uri in enumerate(uris):
                previous = state["pending"].get(uri)
                state["pending"][uri] = {
                    "tokens": max(previous["tokens"] if previous else 0, per_uri + (index < extra)),
                    "at": previous["at"] if previous else now,
                }
            active = self._jobs.get(state["active"])
            if active is not None and active.status in {SceneJobStatus.RUNNING, SceneJobStatus.PENDING}:
                return active
            return self._promote(account_id, owner_space, state)

    def _promote(self, account_id: str, owner_space: str, state: dict) -> SceneJob | None:
        pending = state["pending"]
        if not pending:
            return self._jobs.get(state["active"])
        tokens = sum(item["tokens"] for item in pending.values())
        oldest = min(datetime.fromisoformat(item["at"]) for item in pending.values())
        due = oldest + timedelta(seconds=state["max_wait_seconds"]) <= self._clock()
        if (not state["force"] and tokens < state["trigger_tokens"]
                and len(pending) < state["max_pending_nodes"] and not due):
            return self._jobs.get(state["active"])
        fresh = sorted(pending, key=lambda uri: (pending[uri]["at"], uri))
        retained = [uri for uri in state["retained"] if uri not in pending]
        selected = []
        # Alternate fresh and retained evidence; FIFO rotates WAIT evidence fairly.
        sources = (fresh, retained)
        if fresh and retained and state["max_pending_nodes"] == 1:
            if state.get("prefer_retained", False):
                sources = (retained, fresh)
            state["prefer_retained"] = not state.get("prefer_retained", False)
        for index in range(max(len(fresh), len(retained))):
            for source in sources:
                if index < len(source) and len(selected) < state["max_pending_nodes"]:
                    selected.append(source[index])
            if len(selected) == state["max_pending_nodes"]:
                break
        selected_tokens = sum(pending.pop(uri)["tokens"] for uri in selected if uri in pending)
        state["retained"] = [uri for uri in retained if uri not in selected]
        job = SceneJob(uuid4().hex, account_id, owner_space, tuple(selected),
                       token_count=selected_tokens, created_at=self._clock())
        self._jobs[job.job_id] = job
        state["active"] = job.job_id
        state["force"] = bool(pending)
        return job

    def claim(self, worker_id: str, *, lease_seconds: float = 300) -> SceneJob | None:
        if not worker_id or lease_seconds <= 0:
            raise ValueError("worker_id and positive lease_seconds are required")
        with self._lock:
            now = self._clock()
            for (account_id, owner_space), state in self._scopes.items():
                job = self._jobs.get(state["active"])
                if job is None or job.status == SceneJobStatus.WAITING:
                    job = self._promote(account_id, owner_space, state)
                ready = job is not None and (
                    job.status == SceneJobStatus.PENDING
                    and (job.next_run_at is None or job.next_run_at <= now)
                    or job.status == SceneJobStatus.RUNNING
                    and job.lease_until is not None and job.lease_until <= now)
                if ready:
                    claimed = replace(job, status=SceneJobStatus.RUNNING, attempt=job.attempt + 1,
                                      worker_id=worker_id, lease_token=uuid4().hex,
                                      lease_until=now + timedelta(seconds=lease_seconds), next_run_at=None)
                    self._jobs[job.job_id] = claimed
                    self._scopes.move_to_end((account_id, owner_space))
                    return claimed
        return None

    def _owned(self, reference: SceneJob) -> SceneJob:
        job = self._jobs[reference.job_id]
        if job.status != SceneJobStatus.RUNNING or job.lease_until is None or job.lease_until <= self._clock():
            raise SceneLeaseLostError("scene job lease expired")
        if reference.lease_token != job.lease_token:
            raise SceneLeaseLostError("scene job was reclaimed")
        return job

    def renew(self, job: SceneJob, *, lease_seconds: float = 300) -> SceneJob:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        with self._lock:
            current = self._owned(job)
            renewed = replace(current, lease_until=self._clock() + timedelta(seconds=lease_seconds))
            self._jobs[job.job_id] = renewed
            return renewed

    def _finish(self, reference, status, reason, *, remaining_uris=(), delay_seconds=0,
                min_evidence_nodes=3) -> SceneJob:
        with self._lock:
            job = self._owned(reference)
            state = self._scope(job.account_id, job.owner_space)
            result = replace(job, status=status, reason=reason, lease_until=None,
                             failure_count=job.failure_count + (status in {SceneJobStatus.PENDING, SceneJobStatus.FAILED}),
                             next_run_at=self._clock() + timedelta(seconds=delay_seconds)
                             if status == SceneJobStatus.PENDING else None)
            self._jobs[job.job_id] = result
            state["last_job"] = job.job_id
            if status == SceneJobStatus.WAITING:
                state["retained"] = list(dict.fromkeys([*state["retained"], *remaining_uris]))
            elif status == SceneJobStatus.SUCCEEDED:
                remaining = list(dict.fromkeys(remaining_uris))
                consumed = set(job.node_uris) - set(remaining)
                if consumed and len(remaining) >= min_evidence_nodes:
                    for uri in remaining:
                        state["pending"].setdefault(uri, {"tokens": 0, "at": job.created_at.isoformat()})
                    state["retained"] = [uri for uri in state["retained"] if uri not in remaining]
                    state["force"] = True
                else:
                    state["retained"] = list(dict.fromkeys([*state["retained"], *remaining]))
                state["active"] = None
            elif status == SceneJobStatus.FAILED:
                state["active"] = None
            return result

    def mark_succeeded(self, job, *, reason="", remaining_uris=(), min_evidence_nodes=3) -> SceneJob:
        if isinstance(min_evidence_nodes, bool) or not isinstance(min_evidence_nodes, int) or min_evidence_nodes < 1:
            raise ValueError("min_evidence_nodes must be a positive integer")
        return self._finish(job, SceneJobStatus.SUCCEEDED, reason, remaining_uris=remaining_uris,
                            min_evidence_nodes=min_evidence_nodes)

    def mark_waiting(self, job, *, reason="", remaining_uris=None) -> SceneJob:
        return self._finish(job, SceneJobStatus.WAITING, reason,
                            remaining_uris=job.node_uris if remaining_uris is None else remaining_uris)

    def mark_retry(self, job, *, reason, delay_seconds) -> SceneJob:
        if delay_seconds < 0:
            raise ValueError("delay_seconds must be nonnegative")
        return self._finish(job, SceneJobStatus.PENDING, reason, delay_seconds=delay_seconds)

    def mark_failed(self, job, *, reason) -> SceneJob:
        return self._finish(job, SceneJobStatus.FAILED, reason)

    def get(self, job_id: str) -> SceneJob:
        with self._lock:
            return self._jobs[job_id]

    def flush(self, account_id: str, owner_space: str) -> SceneJob | None:
        with self._lock:
            state = self._scope(account_id, owner_space)
            active = self._jobs.get(state["active"])
            state["force"] = bool(state["pending"])
            if active and active.status in {SceneJobStatus.PENDING, SceneJobStatus.RUNNING}:
                return active
            return self._promote(account_id, owner_space, state)

    def summary(self, account_id: str, owner_space: str) -> dict:
        with self._lock:
            state = self._scope(account_id, owner_space)
            active = self._jobs.get(state["active"])
            last = self._jobs.get(state.get("last_job"))
            return {"pending_nodes": len(state["pending"]),
                    "pending_tokens": sum(item["tokens"] for item in state["pending"].values()),
                    "retained_nodes": len(state["retained"]),
                    "active_job": job_to_dict(active) if active else None,
                    "last_job": job_to_dict(last) if last else None}

    def due_at(self, account_id: str, owner_space: str) -> datetime | None:
        with self._lock:
            state = self._scope(account_id, owner_space)
            active = self._jobs.get(state["active"])
            if active and active.status == SceneJobStatus.RUNNING:
                return active.lease_until
            if active and active.status == SceneJobStatus.PENDING:
                return active.next_run_at or self._clock()
            pending = state["pending"]
            if not pending:
                return None
            if (state["force"] or len(pending) >= state["max_pending_nodes"]
                    or sum(item["tokens"] for item in pending.values()) >= state["trigger_tokens"]):
                return self._clock()
            return min(datetime.fromisoformat(item["at"]) for item in pending.values()) + timedelta(
                seconds=state["max_wait_seconds"])


@dataclass(frozen=True)
class SceneWorkerOutcome:
    job: SceneJob
    decision: SceneDecisionResult | None
    halted: bool = False


def _rate_limited(exc: BaseException) -> bool:
    current = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        response = getattr(current, "response", None)
        status = getattr(current, "status_code", getattr(response, "status_code", None))
        if status == 429 or "ratelimit" in type(current).__name__.lower():
            return True
        current = current.__cause__ or current.__context__
    return False


def _usage_context(job: SceneJob) -> RequestContext:
    owner_space = canonical_owner_space(job.owner_space)
    kind, separator, identifier = owner_space.partition(":")
    if not job.account_id or not separator or not identifier or kind not in {"user", "agent"}:
        raise SceneDecisionError("scene scope requires an account and user or agent owner")
    return RequestContext(
        account_id=job.account_id,
        user_id=identifier if kind == "user" else "",
        agent_id=identifier if kind == "agent" else "",
        session_id="",
        trace_id=job.job_id,
        visible_owner_spaces=(owner_space,),
    )


class SceneWorker:
    """Lease one scope window, execute outside storage locks, then acknowledge."""

    def __init__(self, jobs: SceneJobStore, engine: SceneDecisionEngine, scene_store: SceneDecisionStore,
                 *, retry_delay_seconds: float = 5.0,
                 config_provider: SceneConfigProvider | SceneConfigCache | None = None,
                 lease_seconds: float = 300) -> None:
        if retry_delay_seconds < 0 or lease_seconds <= 0:
            raise ValueError("retry delay and lease duration must be valid")
        self.jobs, self.engine, self.store = jobs, engine, scene_store
        self.retry_delay_seconds, self.config_provider = retry_delay_seconds, config_provider
        self.lease_seconds = lease_seconds
        self.halted = False

    def process_next(self, worker_id: str) -> SceneWorkerOutcome | None:
        if self.halted:
            return None
        job = self.jobs.claim(worker_id, lease_seconds=self.lease_seconds)
        return None if job is None else self.process(job)

    def process(self, job: SceneJob) -> SceneWorkerOutcome:
        engine = self.engine
        stop = Event()
        lease_lost = Event()

        def renew():
            while not stop.wait(self.lease_seconds / 3):
                try:
                    self.jobs.renew(job, lease_seconds=self.lease_seconds)
                except Exception:
                    lease_lost.set()
                    return

        renewer = Thread(target=renew, daemon=True, name="scene-job-lease")
        renewer.start()
        fence = current_scene_job.set(job)
        try:
            if self.config_provider is not None:
                provider = self.config_provider
                snapshot = (provider.get(job.account_id, job.owner_space)
                            if isinstance(provider, SceneConfigCache)
                            else provider.get_effective(job.account_id, job.owner_space))
                engine = self.engine.for_config(snapshot.config)
            with request_usage_scope(_usage_context(job)):
                result = engine.refresh(self.store, job.account_id, job.owner_space, node_uris=job.node_uris)
            if lease_lost.is_set():
                raise SceneLeaseLostError("scene analysis lost its lease")
            if result.action == "WAIT":
                done = self.jobs.mark_waiting(job, reason=result.reason,
                                              remaining_uris=result.pending_member_uris)
            else:
                done = self.jobs.mark_succeeded(job, reason=result.action,
                                                remaining_uris=result.pending_member_uris,
                                                min_evidence_nodes=engine.config.min_evidence_nodes)
            return SceneWorkerOutcome(done, result)
        except SceneLeaseLostError:
            return SceneWorkerOutcome(job, None)
        except Exception as exc:
            halted = not isinstance(exc, SceneRetryableDecisionError) and _rate_limited(exc)
            self.halted |= halted
            reason = ("provider rate limit (429)" if halted else
                      str(exc) if isinstance(exc, SceneRetryableDecisionError) else type(exc).__name__)
            permanent = isinstance(exc, SceneDecisionError) and not isinstance(exc, SceneRetryableDecisionError)
            try:
                if halted or permanent or job.failure_count + 1 >= engine.config.retry_limit:
                    failed = self.jobs.mark_failed(job, reason=reason)
                else:
                    failed = self.jobs.mark_retry(job, reason=reason,
                        delay_seconds=self.retry_delay_seconds * 2 ** job.failure_count)
            except SceneLeaseLostError:
                failed = job
            return SceneWorkerOutcome(failed, None, halted)
        finally:
            current_scene_job.reset(fence)
            stop.set()
            renewer.join(timeout=1)
