"""Instance operation scopes and guards on actual backend calls."""
import asyncio
import inspect
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import partial, wraps

from service.migration.maintenance import MaintenanceError


@dataclass
class Operation:
    token: str
    owner: tuple
    write: bool = False
    active: bool = True
    uncertain: bool = False


class Admission:
    def __init__(self, store, participant=None):
        self.store = store
        self.participant = participant
        self._operation = ContextVar(f"migration-admission-{id(self)}", default=None)
        self._lock = threading.RLock()
        self._closing = False

    @staticmethod
    def _owner():
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        return threading.get_ident(), task

    @contextmanager
    def scope(self, *, write=False, _shutdown=False):
        inherited = self._operation.get()
        if inherited is not None and inherited.owner == self._owner():
            if not inherited.active:
                raise MaintenanceError("operation has ended")
            if write:
                self.require_write()
            yield
            return
        with self._lock:
            if self._closing and not _shutdown:
                raise MaintenanceError('business process is stopping')
            operation = Operation(self.store.admit(write=write, participant=self.participant), self._owner(), write)
        marker = self._operation.set(operation)
        try:
            yield
        finally:
            with self._lock:
                operation.active = False
                try:
                    if not operation.uncertain:
                        self.store.finish_operation(operation.token)
                finally:
                    self._operation.reset(marker)

    @contextmanager
    def maintenance_scope(self, task):
        """Private lifecycle callback scope; never selected by request parameters."""
        if self._operation.get() is not None:
            raise MaintenanceError('maintenance cannot run inside a business operation')
        token = self.store.owned_operation(task, self.participant)
        operation = Operation(token, self._owner(), True)
        marker = self._operation.set(operation)
        try:
            yield
            if operation.uncertain:
                raise MaintenanceError('maintenance backend completion is unknown')
        finally:
            operation.active = False
            try:
                if not operation.uncertain:
                    self.store.finish_operation(token)
            finally:
                self._operation.reset(marker)

    def require_write(self):
        with self._lock:
            operation = self._operation.get()
            if operation is None or not operation.active:
                raise MaintenanceError("storage mutation requires an active operation")
            if not operation.write:
                self.store.upgrade(operation.token)
                operation.write = True

    @contextmanager
    def backend(self, *, write, http_rejections=False):
        with self.scope(write=write):
            try:
                yield
            except BaseException as exc:
                # Only backend mutations can be uncertain, not rejected parameters
                # or business result dictionaries. Never release on a guessed timeout.
                from pyagfs.exceptions import AGFSClientError
                from requests.exceptions import HTTPError

                from core.errors import (
                    AccessDeniedError,
                    ConcurrentModificationError,
                    NodeNotFoundError,
                )
                from fs.agfs_adapter.client import NativeAGFSHTTPError

                # pyagfs wraps HTTP rejections but preserves their exception context.
                # Only definite filesystem rejections are complete; timeouts and
                # gateway/server errors can still leave remote writes in flight.
                cause = exc.__context__
                rejected = (isinstance(exc, NativeAGFSHTTPError) or
                            (isinstance(exc, AccessDeniedError)
                             and getattr(exc, 'sql_write_not_started', False) is True) or
                            (isinstance(exc, (ConcurrentModificationError, NodeNotFoundError))
                             and getattr(exc, 'sql_rollback_confirmed', False) is True) or
                            (isinstance(exc, ConcurrentModificationError)
                             and getattr(exc, 'agfs_rejection_settled', False) is True) or
                            (http_rejections and isinstance(exc, AGFSClientError) and isinstance(cause, HTTPError)
                             and cause.response is not None
                             and cause.response.status_code in {403, 404, 409}))
                if write and not rejected:
                    self._operation.get().uncertain = True
                raise

    def launch(self, target, *, args=(), kwargs=None, name=None, daemon=True):
        """Reserve the child's write permit before it can be delayed by scheduling."""
        with self._lock:
            if self._closing:
                raise MaintenanceError('business process is stopping')
            token = self.store.admit(write=True, participant=self.participant)
        operation = Operation(token, (), True)

        def run():
            operation.owner = self._owner()
            marker = self._operation.set(operation)
            try:
                target(*args, **(kwargs or {}))
            finally:
                operation.active = False
                try:
                    if not operation.uncertain:
                        self.store.finish_operation(token)
                finally:
                    self._operation.reset(marker)

        thread = None
        try:
            thread = threading.Thread(target=run, name=name, daemon=daemon)
            thread.start()
        except BaseException:
            if thread is None or thread.ident is None:
                self.store.finish_operation(token)
            raise
        return thread

    def wrap(self, function, *, write=False, backend=False, http_rejections=False):
        scope = partial(self.backend, http_rejections=http_rejections) if backend else self.scope
        if inspect.isasyncgenfunction(function):
            @wraps(function)
            async def async_iterate(*args, **kwargs):
                with scope(write=write):
                    async for value in function(*args, **kwargs):
                        yield value
            return async_iterate
        if inspect.isgeneratorfunction(function):
            @wraps(function)
            def iterate(*args, **kwargs):
                with scope(write=write):
                    yield from function(*args, **kwargs)
            return iterate
        if inspect.isgeneratorfunction(getattr(function, "__wrapped__", None)):
            @wraps(function)
            @contextmanager
            def context(*args, **kwargs):
                with scope(write=write), function(*args, **kwargs) as value:
                    yield value
            return context
        if inspect.iscoroutinefunction(function):
            @wraps(function)
            async def asynchronous(*args, **kwargs):
                with scope(write=write):
                    return await function(*args, **kwargs)
            return asynchronous
        @wraps(function)
        def call(*args, **kwargs):
            with scope(write=write):
                return function(*args, **kwargs)
        return call

    def guard(self, storage):
        """Keep concrete adapter types while guarding public backend methods.

        Unknown operations are mutations. Private transport members must not be
        handed to business callers; migration uses independently opened adapters.
        """
        if storage is None or getattr(storage, "_migration_guard", None) is self:
            return storage
        if getattr(storage, "_migration_guard", None) is not None:
            raise MaintenanceError("storage is already bound to another instance")
        skip = {"close", "close_all", "shutdown", "stop"}
        reads = ("read", "get", "list", "exists", "search", "count", "scan", "iter", "has", "is_", "health", "probe", "check", "validate", "ls", "stat", "cat")
        for name in dir(type(storage)):
            if name.startswith("_") or name in skip:
                continue
            function = getattr(storage, name, None)
            if not inspect.ismethod(function):
                continue
            write = not name.startswith(reads) or name.startswith(("get_or_create", "get_or_init"))
            # mkdir and rm issue one HTTP request. In particular write()
            # retries internally: a final rejection cannot settle earlier timeouts.
            from pyagfs.client import AGFSClient
            single_attempt = isinstance(storage, AGFSClient) and name in {'mkdir', 'rm'}
            if isinstance(storage, AGFSClient) and name == 'write':
                original_write = function
                @wraps(original_write)
                def write_once(path, data, max_retries=0, _write=original_write):
                    return _write(path, data, max_retries=0)
                function = write_once
            setattr(storage, name, self.wrap(function, write=write, backend=True,
                                             http_rejections=single_attempt))
        # AGFS reads may finish an acknowledged deletion. That branch mutates
        # storage even though its public entry point is a read. Trace reads also
        # repair their indexes through the guarded file mutation primitives.
        for name in ("_recover_acknowledging_delete", "_replace_file", "_remove_if_present", "_mkdir_p"):
            mutation = getattr(storage, name, None)
            if callable(mutation):
                setattr(storage, name, self.wrap(mutation, write=True, backend=True))
        storage._migration_guard = self
        return storage


def control_client(client, admission, prefix):
    """Allow only control-namespace reads; every mutation retains admission."""
    import posixpath

    readers = {name: getattr(client, name) for name in ('read', 'cat', 'ls') if callable(getattr(client, name, None))}
    admission.guard(client)
    root = posixpath.normpath('/' + prefix.strip('/'))
    for name, original in readers.items():
        @wraps(original)
        def read(path, *args, _original=original, _name=name, **kwargs):
            normalized = posixpath.normpath(path)
            relative = normalized[len(root.rstrip('/')) + 1:] if normalized.startswith(root.rstrip('/') + '/') else ''
            parts = relative.split('/')
            account_listing = _name == 'ls' and parts == ['accounts'] and not args and not kwargs
            if not (account_listing or parts[0] == '_system' or
                    (len(parts) >= 3 and parts[0] == 'accounts' and parts[2] == '_system')):
                raise MaintenanceError('control client cannot read business storage')
            return _original(path, *args, **kwargs)
        setattr(client, name, read)
    return client


def install(service, admission):
    """Bind one service and its concrete storage clients to admission control."""
    service._admission = admission
    factories = {
        "_get_context_fs", "_get_outbox_store", "_get_relation_store",
        "_get_trace_event_store", "_get_archive_store", "_get_shared_vector_index",
        "_get_shared_projection_index", "_create_tool_result_store", "_get_agfs_client",
    }
    for name in factories:
        original = getattr(service, name)
        cached_attribute = {
            '_get_shared_vector_index': '_vector_index',
            '_get_shared_projection_index': '_projection_index',
            '_get_trace_event_store': '_trace_event_store',
            '_get_archive_store': '_archive_store',
            '_get_context_fs': '_migration_context_fs',
            '_get_agfs_client': '_native_agfs_client',
        }.get(name)
        def factory(*args, _original=original, _cached=cached_attribute, **kwargs):
            # Admission precedes adapter construction, which may initialize a
            # schema or a directory before its returned methods can be guarded.
            constructing = _cached is None or getattr(service, _cached, None) is None
            with admission.backend(write=constructing):
                if _cached == '_migration_context_fs':
                    with service._init_lock:
                        value = getattr(service, _cached, None)
                        if value is None:
                            value = admission.guard(_original(*args, **kwargs))
                            setattr(service, _cached, value)
                        return value
                return admission.guard(_original(*args, **kwargs))
        setattr(service, name, wraps(original)(factory))
    control = {
        "shutdown", "health", "build_context", "get_auth_service", "get_key_manager",
        "get_tenant_admin_service", "get_control_plane_store", "get_audit_service", "get_llm",
        "start_extraction_recovery", "stop_skill_sedimentation",
    }
    for name, method in inspect.getmembers(type(service), inspect.isfunction):
        if name.startswith("_") or name in control or isinstance(vars(type(service)).get(name), staticmethod):
            continue
        setattr(service, name, admission.wrap(getattr(service, name)))
    for name in ("_run_dispose_background", "_process_agent_memory", "_refresh_write_outcomes"):
        setattr(service, name, admission.wrap(getattr(service, name)))


def business_wsgi(application, get_service):
    """Keep HTTP operations admitted through the last streamed response chunk."""
    from contextlib import ExitStack
    from contextvars import copy_context

    def serve(environ, start_response):
        path = environ.get('PATH_INFO', '')
        admin_read = (environ.get('REQUEST_METHOD') == 'GET'
                      and path.startswith('/api/v1/admin/'))
        control = (admin_read or path == '/api/v1/health' or path.startswith((
            '/memory-viewer/', '/api/v1/auth/', '/api/v1/migration/')))
        admission = None if control else getattr(get_service(), '_admission', None)
        if admission is not None:
            admission._operation.set(None)
        with ExitStack() as lifetime:
            try:
                if admission is not None:
                    lifetime.enter_context(admission.scope())
            except MaintenanceError:
                payload = b'{"error":"migration_maintenance","message":"Memory migration is in progress"}'
                start_response('503 Service Unavailable', [
                    ('Content-Type', 'application/json'), ('Content-Length', str(len(payload))),
                    ('Retry-After', '5')])
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
