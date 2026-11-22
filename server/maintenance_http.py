"""HTTP response contract for maintenance admission failures."""
from http import HTTPStatus

MAINTENANCE_STATUS = HTTPStatus.SERVICE_UNAVAILABLE
MAINTENANCE_HEADERS = {"Retry-After": "5"}


def maintenance_payload(message="Memory migration is in progress"):
    return {"error": "migration_maintenance", "message": message, "retryable": True}
