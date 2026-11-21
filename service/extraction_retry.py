"""Durable source retries with bounded model calls and source-only completion.

Storage discovery and authenticated session visits wake saved tasks. SQL session
fences serialize claims, preserving accepted candidates and write receipts.
"""

import logging
import threading
import time
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps
from queue import Empty, Queue
from uuid import uuid4

from commit.outbox_scope import capture_outbox_events
from core.errors import LLMTransientError, MemoryCardFormatError
from core.operation_deadline import operation_deadline
from core.write_contract import error_fields
from fs.sql_adapter.session_mutation import SessionMutationBusyError
from providers.token_tracker import request_usage_scope
from providers.unified_config import DEFAULT_AGENT_TURN_WORKERS
from service.agent_turn_scheduler import AgentTurnQueue
from service.extraction_checkpoint import (
    EXTENSION_KEY,
    ExtractionBatchJournal,
    RecoveryNeedsConfirmationError,
    recovery_owner,
)
from service.extraction_recovery_discovery import AGENT_TURN_TASK

logger = logging.getLogger(__name__)
_running = ContextVar("extraction_retry_running", default=None)
_format_feedback = ContextVar("memory_card_format_feedback", default="")
_source_only_reason = ContextVar("extraction_source_only_reason", default="")
MAX_ATTEMPTS = 2
MAX_TRANSIENT_ATTEMPTS = 3
ATTEMPT_SECONDS = 1200
AGENT_LEASE_SECONDS = 30


def is_frozen_source_retry(ctx, batch):
    return _running.get() == (recovery_owner(ctx), batch.get("archive_id"))


def background_retry_running():
    return _running.get() is not None


def card_format_feedback():
    return _format_feedback.get()


@contextmanager
def source_only_recovery(ctx, archive_id, reason):
    token = _running.set((recovery_owner(ctx), archive_id))
    reason_token = _source_only_reason.set(reason)
    try:
        yield
    finally:
        _source_only_reason.reset(reason_token)
        _running.reset(token)


def source_only_reason():
    return _source_only_reason.get()


def retry_failed_source(function):
    @wraps(function)
    def wrapped(self, params):
        requested = params.get("retryInBackground", False)
        if type(requested) is not bool:
            raise ValueError("retryInBackground must be a boolean")
        client_request_id = str(params.get("clientRequestId") or params.get("client_request_id") or "").strip()
        if requested and not client_request_id:
            raise ValueError("retryInBackground requires a stable clientRequestId")
        result = function(self, params)
        if (not _running.get() and client_request_id and result.get("extraction_archive_id")
                and result.get("ok") is False
                and result.get("status") == "failed" and result.get("retryable") is True
                and result.get("client_request_id") == client_request_id):
            ctx = params.get("_ctx") or self.build_context(params)
            try:
                task = self._extraction_retry.register(ctx, result)
                if task:
                    state = "NEEDS_CONFIRMATION" if task.get("status") == "needs_confirmation" else "RECOVERING"
                    return {**result, "recovery_task": task, "request_status": state,
                            "source_status": "CHECKPOINTED", "semantic_status": "PENDING", "accepted": True}
            except Exception as exc:
                logger.exception("Cannot persist extraction retry")
                return {**result, "retryable": False, "status": "needs_confirmation",
                        "error": "retry checkpoint not acknowledged: " + str(exc)}
        return result
    return wrapped


class ExtractionRetryWorker:
    def __init__(self, service):
        self.service = service
        config = getattr(service, "_cfg", None)
        self._agent_workers = getattr(config, "agent_turn_workers", DEFAULT_AGENT_TURN_WORKERS)
        self._agent_owner = getattr(config, "agent_turn_scheduler_owner", True)
        self._queue = Queue()
        self._condition = threading.Condition()
        self._queued = set()
        self._agent_queue = AgentTurnQueue()
        self._thread = None
        self._legacy_thread = None
        self._agent_threads = set()
        self._stop = threading.Event()
        self._discover = None
        self._discovery_interval = 5
        self._next_discovery = 0

    def start_discovery(self, discover, interval=5):
        with self._condition:
            self._discover = discover
            self._discovery_interval = interval
            self._next_discovery = 0
            self._start_thread()
            self._condition.notify()

    def _start_thread(self):
        if not self._stop.is_set() and (self._thread is None or not self._thread.is_alive()):
            self._thread = threading.Thread(target=self._run, name="extraction-retry", daemon=True)
            self._thread.start()

    def _journal(self, ctx, archive_id=None):
        service = self.service
        service._load_extraction_recovery_before_accumulation(ctx.session_id, ctx)
        manager = service.get_session_manager()
        buf = manager.get_or_create(ctx.session_id, ctx=ctx)
        with buf.session_state_save_lock:
            batch = buf.runtime_extensions.get(EXTENSION_KEY, {}).get("batch")
            if batch is None:
                return None
            if batch.get("owner") != recovery_owner(ctx) or "lifecycle" in batch:
                raise RecoveryNeedsConfirmationError("retry source belongs to another owner or lifecycle")
            if archive_id is not None and batch.get("archive_id") != archive_id:
                return None
            return ExtractionBatchJournal.resume(manager, buf, ctx)

    def register(self, ctx, failure, *, wake=True):
        archive_id = failure.get("extraction_archive_id")
        client_request_id = failure.get("client_request_id")
        if not archive_id or not client_request_id:
            return None
        with self.service._session_mutation_scope(ctx):
            journal = self._journal(ctx, archive_id)
            if journal is None:
                return None
            with journal.buffer.session_state_save_lock:
                if journal.batch.get("client_request_id") != client_request_id:
                    return None
                task = journal.batch.get("retry_task")
                if task is None:
                    # A new task never replaces accepted candidates or their receipts.
                    task = {"id": journal.batch["archive_id"], "status": "queued", "attempts": 0,
                            "max_attempts": MAX_ATTEMPTS, "next_attempt_at": time.time() + 2,
                            "last_error": str(failure.get("error", ""))[:1000]}
                else:
                    task = deepcopy(task)
                kind = failure.get("failure_kind", "")
                if failure.get("error_detail"):
                    task["error_detail"] = deepcopy(failure["error_detail"])
                if (task["status"] == "queued" and failure.get("error")
                        and (kind != "busy" or not task.get("last_error"))):
                    task.update(last_error=str(failure["error"])[:1000],
                                failure_kind=kind if kind != "busy" else "")
                    if kind == "memory_card_format":
                        task["format_error"] = task["last_error"]
                journal.mutate(lambda batch: batch.update(retry_task=task))
                task = deepcopy(task)
        if wake:
            self.wake(ctx, task)
        return task

    def register_pending_source(self, ctx, journal, *, wake=False):
        return self.register_background_failure(ctx, journal, None, wake=wake)

    def register_background_failure(self, ctx, journal, error, *, wake=True):
        """Give a frozen asynchronous source a durable, bounded recovery task."""
        if journal is None:
            return None
        if not journal.batch.get("client_request_id"):
            journal.mutate(lambda batch: batch.update(
                client_request_id="extraction-retry-" + batch["archive_id"]))
        return self.register(ctx, {
            "extraction_archive_id": journal.batch["archive_id"],
            "client_request_id": journal.batch["client_request_id"], "error": str(error) if error else "",
            "failure_kind": ("memory_card_format" if isinstance(error, MemoryCardFormatError) else
                             "llm_transient" if isinstance(error, LLMTransientError) else
                             "busy" if isinstance(error, SessionMutationBusyError) else ""),
            **(error_fields(error, journal.batch["client_request_id"]) if error else {}),
        }, wake=wake)

    def wake(self, ctx, task):
        if task.get("status") not in {"queued", "running", "failed"}:
            return
        archive_id = AGENT_TURN_TASK if task.get("agent_async") else task["id"]
        with self._condition:
            if self._stop.is_set():
                return
            if archive_id == AGENT_TURN_TASK:
                if not self._agent_owner:
                    return
                wall_time = time.time()
                now = time.monotonic()
                due_at = now + max(0, (task.get("next_attempt_at") or 0) - wall_time)
                ready_at_ms = task.get("ready_at_ms")
                self._agent_queue.wake(
                    ctx, ready_at_ms if ready_at_ms is not None else int(wall_time * 1000), due_at, now,
                    pending=ready_at_ms is not None and not task.get("agent_async"),
                )
            else:
                identity = (*recovery_owner(ctx).values(), archive_id)
                if identity in self._queued:
                    return
                self._queued.add(identity)
                self._queue.put((identity, ctx, archive_id))
            self._start_thread()
            self._condition.notify()

    def _run(self):
        while not self._stop.is_set():
            if self._discover is not None and time.monotonic() >= self._next_discovery:
                self._next_discovery = time.monotonic() + self._discovery_interval
                try:
                    for ctx, task in self._discover():
                        self.wake(ctx, task)
                except Exception:
                    logger.exception("Extraction retry discovery failed")
            with self._condition:
                if self._stop.is_set():
                    break
                if self._legacy_thread is None:
                    try:
                        legacy = self._queue.get_nowait()
                    except Empty:
                        pass
                    else:
                        self._legacy_thread = threading.Thread(
                            target=self._run_legacy_item, args=legacy, daemon=True,
                            name="extraction-retry-attempt",
                        )
                        self._legacy_thread.start()
                deadlines = [self._next_discovery] if self._discover is not None else []
                if len(self._agent_threads) < self._agent_workers:
                    turn = self._agent_queue.pop(time.monotonic())
                    if turn is not None:
                        thread = threading.Thread(target=self._run_agent_item, args=(turn,), daemon=True,
                                                  name="agent-extraction")
                        self._agent_threads.add(thread)
                        thread.start()
                        continue
                    due_at = self._agent_queue.next_deadline()
                    if due_at is not None:
                        deadlines.append(due_at)
                timeout = max(0, min(deadlines) - time.monotonic()) if deadlines else None
                self._condition.wait(timeout=timeout)

    def _run_agent_item(self, turn):
        delay = None
        try:
            if not self._stop.is_set():
                delay = self.run_once(turn.ctx, AGENT_TURN_TASK)
        except Exception:
            logger.exception("Extraction retry worker stopped an Agent turn")
        finally:
            with self._condition:
                self._agent_queue.finish(turn, delay, time.monotonic(), int(time.time() * 1000))
                self._agent_threads.discard(threading.current_thread())
                self._condition.notify()

    def _run_legacy_item(self, identity, ctx, archive_id):
        requeued = False
        try:
            delay = self.run_once(ctx, archive_id)
            if delay is not None and not self._stop.is_set():
                self._stop.wait(min(delay, .2))
                if not self._stop.is_set():
                    self._queue.put((identity, ctx, archive_id))
                    requeued = True
        except Exception:
            logger.exception("Extraction retry worker stopped a source task")
        finally:
            with self._condition:
                if not requeued:
                    self._queued.discard(identity)
                self._queue.task_done()
                self._legacy_thread = None
                self._condition.notify()

    def _run_agent_turn(self, ctx):
        service = self.service
        token = uuid4().hex
        with service._session_mutation_scope(ctx):
            journal, delay = service._agent_turn_journal(ctx)
            if journal is None:
                return delay
            task = deepcopy(journal.batch.get("retry_task", {}))
            if task.get("status") == "needs_confirmation":
                return None
            now = time.time()
            delay = max(task.get("lease_until", 0), task.get("next_attempt_at", 0)) - now
            if delay > 0:
                return delay
            initial = "attempts" not in task
            source_only = (task.get("source_only") or task.get("attempts", 0) >= MAX_ATTEMPTS
                           or task.get("transient_attempts", 0) >= MAX_TRANSIENT_ATTEMPTS)
            task.update(id=journal.batch["archive_id"], status="running", lease_token=token, agent_async=True,
                        lease_until=now + AGENT_LEASE_SECONDS,
                        attempts=task.get("attempts", 0) + int(not initial and not source_only))
            journal.mutate(lambda batch: batch.update(retry_task=task))
            archive_id = task["id"]
            request_id = journal.batch["client_request_id"]
            ready_at_ms = journal.batch.get("ready_at_ms")
            pinned = journal.manager.begin_managed_operation(ctx.session_id, ctx, create=False)

        started_ms = int(time.time() * 1000)
        started = time.monotonic()
        ready_wait_ms = started_ms - ready_at_ms if ready_at_ms is not None and started_ms >= ready_at_ms else None
        logger.info(
            "Agent turn started session_id=%s request_id=%s archive_id=%s ready_at_ms=%s "
            "started_at_ms=%d ready_wait_ms=%s attempt=%d",
            ctx.session_id, request_id, archive_id, ready_at_ms, started_ms, ready_wait_ms, task["attempts"],
        )
        @contextmanager
        def publication_scope():
            with service._session_mutation_scope(ctx):
                current = journal.batch.get("retry_task", {})
                if current.get("lease_token") != token or current.get("lease_until", 0) <= time.time():
                    raise SessionMutationBusyError("agent extraction lease is no longer owned")
                yield

        done = threading.Event()

        def renew():
            while not done.wait(AGENT_LEASE_SECONDS / 3):
                try:
                    with publication_scope():
                        def extend(batch):
                            batch["retry_task"]["lease_until"] = time.time() + AGENT_LEASE_SECONDS
                        journal.mutate(extend, capture_events=False)
                except Exception:
                    logger.debug("Unable to renew agent extraction lease", exc_info=True)

        heartbeat = threading.Thread(target=renew, name="agent-extraction-lease", daemon=True)
        heartbeat.start()
        journal.publication_scope = publication_scope
        feedback = _format_feedback.set(task.get("format_error", ""))
        try:
            recovery = (source_only_recovery(ctx, archive_id, task.get("last_error") or "extraction retries exhausted")
                        if source_only else nullcontext())
            with request_usage_scope(ctx), operation_deadline(ATTEMPT_SECONDS), recovery:
                next_ready_at_ms = service._process_agent_turn(ctx, journal, publication_scope)
        except Exception as exc:
            journal.publication_scope = nullcontext
            with service._session_mutation_scope(ctx):
                current = journal.batch.get("retry_task", {})
                if current.get("lease_token") != token or current.get("lease_until", 0) <= time.time():
                    return 1
                transient = isinstance(exc, LLMTransientError)
                session_busy = isinstance(exc, SessionMutationBusyError)
                if transient or session_busy:
                    task["attempts"] -= int(not initial and not source_only)
                if transient:
                    task["transient_attempts"] = task.get("transient_attempts", 0) + 1
                if isinstance(exc, MemoryCardFormatError):
                    task["format_error"] = str(exc)[:1000]
                blocked = isinstance(exc, RecoveryNeedsConfirmationError) or (source_only and not session_busy)
                delay = (1 if session_busy else
                         min(10 * 2 ** max(0, task["transient_attempts"] - 1), 60) if transient else 2)
                task.update(status="needs_confirmation" if blocked else "queued",
                            source_only=bool(source_only), last_error=str(exc)[:1000],
                            error_detail=error_fields(exc, request_id)["error_detail"],
                            next_attempt_at=time.time() + delay)
                task.pop("lease_token", None)
                task.pop("lease_until", None)
                journal.mutate(lambda batch: batch.update(retry_task=task))
            logger.warning(
                "Agent turn attempt ended session_id=%s request_id=%s archive_id=%s status=%s "
                "duration_ms=%.3f: %s",
                ctx.session_id, request_id, archive_id, task["status"], (time.monotonic() - started) * 1000, exc,
            )
            return None if blocked else delay
        finally:
            _format_feedback.reset(feedback)
            done.set()
            heartbeat.join()
            if pinned is not None:
                pinned.end_operation()

        logger.info(
            "Agent turn published session_id=%s request_id=%s archive_id=%s finished_at_ms=%d duration_ms=%.3f",
            ctx.session_id, request_id, archive_id, int(time.time() * 1000), (time.monotonic() - started) * 1000,
        )
        service._wake_agent_projection(ctx.account_id)
        if next_ready_at_ms is not None:
            self.wake(ctx, {"id": AGENT_TURN_TASK, "status": "queued", "ready_at_ms": next_ready_at_ms})
        return 0

    @capture_outbox_events
    def run_once(self, ctx, archive_id):
        """One fenced attempt; return backoff seconds or None for a terminal task."""
        try:
            if archive_id == AGENT_TURN_TASK:
                return self._run_agent_turn(ctx)
            delay, attempted = self._run_fenced(ctx, archive_id)
        except SessionMutationBusyError:
            return 1
        # Nested after_turn correctly defers indexing until the outer SQL fence
        # commits; launch it only after that transaction has actually finished.
        if attempted:
            self.service._async_drain(account_id=ctx.account_id)
        return delay

    def _run_fenced(self, ctx, archive_id):
        with self.service._session_mutation_scope(ctx):
            journal = self._journal(ctx, archive_id)
            if journal is None or journal.batch.get("agent_async"):
                return None, False
            if not journal.batch.get("retry_task"):
                self.register_pending_source(ctx, journal, wake=False)
            with journal.buffer.session_state_save_lock:
                task = deepcopy(journal.batch.get("retry_task", {}))
                if task.get("status") not in {"queued", "running", "failed"}:
                    return None, False
                delay = task.get("next_attempt_at", 0) - time.time()
                if delay > 0:
                    return min(delay, 20), False
                source_only = (task.get("source_only") or task.get("attempts", 0) >= MAX_ATTEMPTS
                               or task.get("transient_attempts", 0) >= MAX_TRANSIENT_ATTEMPTS)
                task.update(status="running", attempts=task.get("attempts", 0) + int(not source_only))
                if source_only:
                    task["source_only"] = True
                journal.mutate(lambda batch: batch.update(retry_task=task))
                batch = journal.batch
                if not batch.get("client_request_id"):
                    raise RecoveryNeedsConfirmationError("background retry source lacks a stable request identity")
                params = {"_ctx": ctx, "sessionId": ctx.session_id, "wait": True,
                          "forceExtract": True, "timeoutSeconds": ATTEMPT_SECONDS,
                          "clientRequestId": batch.get("client_request_id", ""),
                          "messages": deepcopy(batch["request_messages"])}
            token = _running.set((recovery_owner(ctx), archive_id))
            feedback_token = _format_feedback.set(
                task.get("format_error", task.get("last_error", "")
                         if task.get("failure_kind") == "memory_card_format" else ""))
            try:
                if source_only:
                    result = self.service._complete_extraction_source(
                        ctx, archive_id, task.get("last_error") or "extraction retries exhausted")
                else:
                    result = self.service.after_turn(params)
            finally:
                _format_feedback.reset(feedback_token)
                _running.reset(token)
            journal = self._journal(ctx, archive_id)
            if journal is None:
                return None, True  # finish() acknowledged this exact source.
            with journal.buffer.session_state_save_lock:
                current = journal.batch.get("retry_task", {})
                if current.get("status") != "running" or current.get("attempts") != task["attempts"]:
                    return None, True
                processing = result.get("status") == "processing"
                retryable = result.get("retryable") is True and result.get("status") != "needs_confirmation"
                if processing:
                    task.update(status="queued", attempts=task["attempts"] - int(not source_only))
                elif retryable:
                    task["status"] = "queued"
                    if not source_only and result.get("failure_kind") == "llm_transient":
                        task.update(attempts=task["attempts"] - 1,
                                    transient_attempts=task.get("transient_attempts", 0) + 1)
                else:
                    task["status"] = "needs_confirmation"
                if not processing:
                    task.update(last_error=str(result.get("error", result.get("status", "pending")))[:1000],
                                failure_kind=result.get("failure_kind", ""))
                    if result.get("error_detail"):
                        task["error_detail"] = deepcopy(result["error_detail"])
                    if task["failure_kind"] == "memory_card_format":
                        task["format_error"] = task["last_error"]
                exhausted = (task["attempts"] >= MAX_ATTEMPTS
                             or task.get("transient_attempts", 0) >= MAX_TRANSIENT_ATTEMPTS)
                delay = 0 if retryable and exhausted and not source_only else 10
                if result.get("failure_kind") == "llm_transient" and not exhausted:
                    delay = min(10 * 2 ** max(0, task.get("transient_attempts", 1) - 1), 60)
                task["next_attempt_at"] = time.time() + delay
                journal.mutate(lambda batch: batch.update(retry_task=task))
                return (delay if task["status"] == "queued" else None), not processing

    def close(self, timeout=1):
        self._stop.set()
        deadline = time.monotonic() + timeout
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout)
        with self._condition:
            threads = list(self._agent_threads)
            if self._legacy_thread is not None:
                threads.append(self._legacy_thread)
        for thread in threads:
            if thread is not threading.current_thread():
                thread.join(max(0, deadline - time.monotonic()))
        return (self._thread is None or not self._thread.is_alive()) and not any(
            thread.is_alive() for thread in threads)
