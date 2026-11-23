"""Bridge participant transitions to MemoryService's resource lifecycle."""
from service.migration.maintenance import MaintenanceError


class ServiceLifecycle:
    def __init__(self, service, *, timeout=30):
        self.service = service
        self.timeout = timeout

    def pause(self):
        return self.service._pause_for_maintenance(self.timeout)

    def refresh(self):
        active = self.service._admission.store.status()
        if active is None:
            raise MaintenanceError("cache refresh requires active maintenance")
        self.service._reset_business_caches(modified=active["modified"])

    def resume(self):
        self.service._resume_after_maintenance()
