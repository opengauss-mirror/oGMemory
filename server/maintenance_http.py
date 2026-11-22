"""HTTP response contract for maintenance admission failures."""
import json
from contextlib import ExitStack
from contextvars import copy_context
from http import HTTPStatus

from core.errors import MaintenanceError

MAINTENANCE_STATUS = HTTPStatus.SERVICE_UNAVAILABLE
MAINTENANCE_HEADERS = {"Retry-After": "5"}


def maintenance_payload(message="Memory migration is in progress"):
    return {"error": "migration_maintenance", "message": message, "retryable": True}


def business_wsgi(application, get_service):
    """Keep HTTP operations admitted through the last streamed response chunk."""

    def serve(environ, start_response):
        path = environ.get('PATH_INFO', '')
        admin_read = (environ.get('REQUEST_METHOD') == 'GET'
                      and path.startswith('/api/v1/admin/'))
        control = (admin_read or path == '/api/v1/health' or path.startswith((
            '/memory-viewer/', '/api/v1/auth/', '/api/v1/migration/')))
        admission = None if control else getattr(get_service(), '_admission', None)
        with ExitStack() as lifetime:
            try:
                if admission is not None:
                    lifetime.enter_context(admission.request_scope())
            except MaintenanceError:
                payload = json.dumps(maintenance_payload()).encode()
                start_response(f'{MAINTENANCE_STATUS.value} {MAINTENANCE_STATUS.phrase}', [
                    ('Content-Type', 'application/json'), ('Content-Length', str(len(payload))),
                    *MAINTENANCE_HEADERS.items()])
                yield payload
                return
            response = application(environ, start_response)
            try:
                yield from response
            finally:
                close = getattr(response, 'close', None)
                if callable(close):
                    close()
    class Response:
        def __init__(self, environ, start_response):
            self.context = copy_context()
            self.iterator = serve(environ, start_response)

        def __iter__(self):
            return self

        def __next__(self):
            return self.context.run(next, self.iterator)

        def close(self):
            self.context.run(self.iterator.close)

        def __del__(self):
            # Finalize in the Context that owns the request's admission token.
            try:
                self.close()
            except Exception:
                pass  # Uncertain backend operations stay durably registered.

    return Response
