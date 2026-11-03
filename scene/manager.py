"""Non-blocking scene-memory orchestration."""

from __future__ import annotations

from collections.abc import Iterable
from threading import Event, Lock, Thread, current_thread
from uuid import uuid4

from core.logging_config import get_logger
from scene.config import SceneConfig, SceneConfigCache, SceneConfigProvider
from scene.decision import SceneDecisionEngine
from scene.jobs import SceneJob, SceneJobStore, SceneWorker, SceneWorkerOutcome
from scene.store import SceneDecisionStore

logger = get_logger(__name__)


class SceneMemoryManager:
    """Bridge canonical memory writes to the asynchronous scene worker.

    ``after_memory_write`` validates owner configuration and durably enqueues
    evidence; the background worker publishes accepted scene projections.
    """

    def __init__(
        self,
        jobs: SceneJobStore,
        scene_store: SceneDecisionStore,
        engine: SceneDecisionEngine,
        *,
        config: SceneConfig | None = None,
        config_provider: SceneConfigProvider | SceneConfigCache | None = None,
        retry_delay_seconds: float = 5.0,
        projection_refresh=None,
    ) -> None:
        self.jobs = jobs
        self.projection_refresh = projection_refresh
        self.scene_store = scene_store
        self.engine = engine
        self._config = config or engine.config
        self._config_provider = config_provider
        self.worker = SceneWorker(
            jobs,
            engine,
            scene_store,
            retry_delay_seconds=retry_delay_seconds,
            config_provider=config_provider,
        )
        self._stop = Event()
        self._wakeup = Event()
        self._lifecycle_lock = Lock()
        self._thread: Thread | None = None

    @property
    def halted(self) -> bool:
        return self.worker.halted

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, *, poll_seconds: float = 1.0) -> None:
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        with self._lifecycle_lock:
            if self.running or self.halted:
                return
            self._stop.clear()
            worker_id = "scene-" + uuid4().hex

            def run():
                while not self._stop.is_set() and not self.halted:
                    try:
                        outcome = self.process_next(worker_id)
                    except Exception:
                        logger.exception("Scene worker storage operation failed")
                        outcome = None
                    if outcome is None:
                        self._wakeup.wait(poll_seconds)
                        self._wakeup.clear()

            self._thread = Thread(target=run, daemon=True, name="scene-worker")
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> bool:
        self._stop.set()
        self._wakeup.set()
        thread = self._thread
        if thread is not None and thread is not current_thread():
            thread.join(timeout=max(0, timeout))
        return not self.running

    def after_memory_write(
        self,
        account_id: str,
        owner_space: str,
        node_uris: Iterable[str],
        *,
        token_count: int = 0,
        force: bool = False,
        notice_id: str | None = None,
    ) -> SceneJob | None:
        if not account_id or not owner_space:
            raise ValueError("account_id and owner_space are required")
        config = self._effective_config(account_id, owner_space)
        job = self.jobs.enqueue(
            account_id,
            owner_space,
            node_uris,
            token_count=token_count,
            trigger_tokens=config.analysis_trigger_tokens,
            max_pending_nodes=config.max_pending_nodes,
            max_wait_seconds=config.max_wait_seconds,
            force=force,
            notice_id=notice_id,
        )
        self._wakeup.set()
        return job

    def process_next(self, worker_id: str) -> SceneWorkerOutcome | None:
        if self.projection_refresh is not None:
            try:
                self.projection_refresh()
            except Exception:
                logger.exception("Scene lexical projection refresh failed")
        return self.worker.process_next(worker_id)

    def flush(self, account_id: str, owner_space: str) -> SceneJob | None:
        job = self.jobs.flush(account_id, owner_space)
        self._wakeup.set()
        return job

    def _effective_config(self, account_id: str, owner_space: str) -> SceneConfig:
        provider = self._config_provider
        if provider is None:
            return self._config
        if isinstance(provider, SceneConfigCache):
            return provider.get(account_id, owner_space).config
        return provider.get_effective(account_id, owner_space).config
