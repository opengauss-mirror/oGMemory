"""Register server processes before initializing business storage or workers."""
from pathlib import Path

from service.migration.maintenance import MaintenanceError, MaintenanceStore
from service.migration.participants import Participant


def create_service(config, *, service_factory=None):
    from server.memory_service import MemoryService
    from server.migration_lifecycle import ServiceLifecycle

    workspace = Path(config.migration_workspace).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True, mode=0o700)
    gate = MaintenanceStore(workspace / 'maintenance.sqlite', str(workspace))
    lifecycle = None
    participant = Participant(
        gate, pause=lambda: lifecycle.pause(), refresh=lambda: lifecycle.refresh(),
        resume=lambda: lifecycle.resume(),
    )
    try:
        service = (service_factory or MemoryService)(
            config=config, admission=participant.admission, defer_start=True,
        )
        lifecycle = ServiceLifecycle(service)
        service._migration_participant = participant
        service._migration_workspace = workspace
        # Construction is inert. Startup acquires admission atomically against
        # entering maintenance, including the first request in a new process.
        try:
            with participant.admission.scope(write=True):
                lifecycle.resume()
            participant._paused = False
        except MaintenanceError:
            if gate.status() is None:
                raise
            participant._paused = True
        participant.start()
        return service
    except BaseException:
        # Keep any uncertain operation registered; a failed constructor must
        # not make it disappear from a later migration drain.
        participant.close()
        raise


def close_service(service, *, timeout=30):
    """Stop workers before retiring the process registration and OS lease."""
    import time

    from server.migration_lifecycle import ServiceLifecycle

    participant = getattr(service, '_migration_participant', None)
    if participant is None:
        shutdown = getattr(service, 'shutdown', None)
        if callable(shutdown):
            shutdown()
        return
    admission = participant.admission
    # The shutdown scope remains valid for session checkpoint writes after
    # closing new admission. Existing operations must finish before checkpoint.
    try:
        with admission.scope(write=True, _shutdown=True):
            # A maintenance rejection must leave its lifecycle watcher running so
            # that this process can still acknowledge the refresh barrier.
            participant.stop.set()
            if participant._thread is not None:
                participant._thread.join(timeout)
                if participant._thread.is_alive():
                    participant.stop.clear()
                    raise MaintenanceError('maintenance callback is still running')
            with admission._lock:
                admission._closing = True
            own = admission._operation.get().token
            deadline = time.monotonic() + timeout
            while True:
                with participant.store._transaction() as db:
                    pending = db.execute('SELECT 1 FROM operations WHERE participant=? AND token<>?',
                                         (participant.id, own)).fetchone()
                if pending is None:
                    break
                if time.monotonic() >= deadline:
                    raise MaintenanceError('business operations have not stopped')
                time.sleep(.05)
            if ServiceLifecycle(service, timeout=max(0, deadline - time.monotonic())).pause() is not True:
                raise MaintenanceError('business workers have not stopped')
            service.shutdown(join_timeout=max(0, deadline - time.monotonic()))
    except BaseException:
        # Closing still rejects new business requests, but a failed shutdown
        # must remain available for maintenance callbacks and a later retry.
        participant.stop.clear()
        if participant._thread is not None and not participant._thread.is_alive():
            participant._thread = None
            participant.start()
        raise
    participant.close()
    service._migration_participant = None
