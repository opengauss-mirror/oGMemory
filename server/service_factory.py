"""Register server processes before initializing business storage or workers."""
from service.migration.maintenance import MaintenanceError, MaintenanceStore
from service.migration.participants import Participant


def create_service(config, *, service_factory=None):
    from server.memory_service import MemoryService
    from server.migration_lifecycle import ServiceLifecycle

    gate = MaintenanceStore.for_workspace(config.migration_workspace)
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
        service._migration_workspace = gate.path.parent
        # Construction is inert. Startup acquires admission atomically against
        # entering maintenance, including the first request in a new process.
        participant.start_service()
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
    # The shutdown scope remains valid for session checkpoint writes after
    # closing new admission. Existing operations must finish before checkpoint.
    with participant.shutdown_scope(timeout):
        deadline = time.monotonic() + timeout
        participant.wait_for_operations(deadline)
        if ServiceLifecycle(service, timeout=max(0, deadline - time.monotonic())).pause() is not True:
            raise MaintenanceError('business workers have not stopped')
        service.shutdown(join_timeout=max(0, deadline - time.monotonic()))
    participant.close()
    service._migration_participant = None
