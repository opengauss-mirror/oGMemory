"""Instance operation scopes and guards on actual backend calls."""
import asyncio
import inspect
import posixpath
import sys
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import partial, wraps

from core.persistence_outcome import WriteCompletion, record_write_completion, write_completion
from service.migration.maintenance import MaintenanceError


def _content_path(path):
    """Conservative classification of AGFS payload mutations, excluding runtime trees."""
    from pathlib import PurePosixPath
    parts = PurePosixPath(posixpath.normpath(str(path))).parts
    # Coordination files have an exact reserved layout, never a basename rule.
    import re
    if any(re.fullmatch(r'\..+\.resource-mutation-lock(?:-[0-9a-f]{32})?', p) for p in parts):
        return parts[-1] in {'content.md', '.meta.json', '.abstract.md', '.overview.md', '.relations.json'}
    if 'accounts' in parts:
        i = parts.index('accounts')
        tail = parts[i + 2:]
        if tail and tail[0] in {'_system', '.outbox', '.outbox-done', '.lexical-outbox', 'traces', 'tool-results'}:
            return False
        from service.migration.records import runtime_path
        return not runtime_path(tail)
    return True


def _content_arguments(*args, **kwargs):
    values = list(args) + [kwargs[k] for k in ('node', 'uri', 'path', 'from_uri', 'to_uri') if k in kwargs]
    uris = [getattr(value, 'uri', value) for value in values]
    uris = [uri for uri in uris if isinstance(uri, str) and uri.startswith('ctx://')]
    if uris:
        from service.migration.records import runtime_node
        return any(not runtime_node({'uri': uri}) for uri in uris)
    value = args[0] if args else kwargs.get('path')
    return _content_path(value) if isinstance(value, str) else True


def _transport_paths(signature, args, kwargs):
    return [value for name, value in signature.bind(*args, **kwargs).arguments.items()
            if name in {'path', 'src', 'dst', 'source', 'destination'}]


@dataclass
class Operation:
    token: str
    owner: tuple
    write: bool = False
    active: bool = True
    uncertain: bool = False
    content: bool = False


class Admission:
    def __init__(self, store, participant=None):
        self.store = store
        self.participant = participant
        self._operation = ContextVar(f"migration-admission-{id(self)}", default=None)
        self._runtime_backend = ContextVar(f'migration-runtime-{id(self)}', default=False)
        self._lock = threading.RLock()
        self._closing = False

    @contextmanager
    def request_scope(self):
        marker = self._operation.set(None)
        try:
            with self.scope():
                yield
        finally:
            self._operation.reset(marker)

    def current_token(self):
        operation = self._operation.get()
        if operation is None or not operation.active or operation.owner != self._owner():
            raise MaintenanceError('active operation required')
        return operation.token

    def stop_accepting(self):
        with self._lock:
            self._closing = True

    @staticmethod
    def _owner():
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        return threading.get_ident(), task

    @contextmanager
    def scope(self, *, write=False, _shutdown=False, _content=False):
        inherited = self._operation.get()
        if inherited is not None and inherited.owner == self._owner():
            if not inherited.active:
                raise MaintenanceError("operation has ended")
            if write:
                self.require_write(content=_content)
            yield
            return
        with self._lock:
            if self._closing and not _shutdown:
                raise MaintenanceError('business process is stopping')
            operation = Operation(self.store.admit(write=write, participant=self.participant, content=_content),
                                  self._owner(), write, content=_content)
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
        operation = Operation(token, self._owner(), True, content=True)
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

    def require_write(self, *, content=False):
        with self._lock:
            operation = self._operation.get()
            if operation is None or not operation.active:
                raise MaintenanceError("storage mutation requires an active operation")
            if not operation.write or (content and not operation.content):
                self.store.upgrade(operation.token, content=content)
                operation.content = operation.content or content
                operation.write = True

    @contextmanager
    def content_scope(self):
        """Own a complete persistence operation independently of request lifetime."""
        current = self._operation.get()
        if current is not None and current.owner == self._owner() and current.content:
            with self.scope(write=True):
                yield
            return
        marker = self._operation.set(None)
        try:
            with self.scope(write=True, _content=True):
                yield
        finally:
            self._operation.reset(marker)

    @contextmanager
    def backend(self, *, write, content=True):
        lifetime = self.content_scope() if write and content else self.scope(write=write)
        with lifetime:
            runtime = self._runtime_backend.set(not content)
            try:
                yield
            except BaseException as exc:
                if write and write_completion(exc) is not WriteCompletion.SETTLED:
                    self._operation.get().uncertain = True
                raise
            finally:
                self._runtime_backend.reset(runtime)

    def bind_child(self, target):
        """Bind a joined lease worker to its already admitted parent operation."""
        parent = self._operation.get()
        if parent is None or parent.owner != self._owner() or not parent.active:
            raise MaintenanceError('lease worker requires an active parent operation')
        runtime = self._runtime_backend.get()

        @wraps(target)
        def run(*args, **kwargs):
            if not parent.active:
                raise MaintenanceError('parent operation has ended')
            child = replace(parent, owner=self._owner())
            marker = self._operation.set(child)
            mode = self._runtime_backend.set(runtime)
            try:
                return target(*args, **kwargs)
            finally:
                with self._lock:
                    parent.uncertain = parent.uncertain or child.uncertain
                child.active = False
                self._runtime_backend.reset(mode)
                self._operation.reset(marker)
        return run

    def launch(self, target, *, args=(), kwargs=None, name=None, daemon=True):
        """Reserve the child's write permit before it can be delayed by scheduling."""
        with self._lock:
            if self._closing:
                raise MaintenanceError('business process is stopping')
            token = self.store.admit(write=True, participant=self.participant, content=False)
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

    def wrap(self, function, *, write=False, backend=False, content=True):
        def scope_for(args, kwargs):
            payload = content(*args, **kwargs) if callable(content) else content
            return (partial(self.backend, content=payload)
                    if backend else self.scope)
        if inspect.isasyncgenfunction(function):
            @wraps(function)
            async def async_iterate(*args, **kwargs):
                with scope_for(args, kwargs)(write=write):
                    async for value in function(*args, **kwargs):
                        yield value
            return async_iterate
        if inspect.isgeneratorfunction(function):
            @wraps(function)
            def iterate(*args, **kwargs):
                with scope_for(args, kwargs)(write=write):
                    yield from function(*args, **kwargs)
            return iterate
        if inspect.isgeneratorfunction(getattr(function, "__wrapped__", None)):
            @wraps(function)
            @contextmanager
            def context(*args, **kwargs):
                # Only acquisition/release are backend calls. The caller's
                # exception is not evidence of an unfinished remote mutation.
                payload = content(*args, **kwargs) if callable(content) else content
                lifetime = self.content_scope() if write and payload else self.scope(write=write)
                with lifetime:
                    manager = function(*args, **kwargs)
                    with scope_for(args, kwargs)(write=write):
                        value = manager.__enter__()
                    try:
                        yield value
                    except BaseException:
                        error = sys.exc_info()
                        with scope_for(args, kwargs)(write=write):
                            suppressed = manager.__exit__(*error)
                        if not suppressed:
                            raise
                    else:
                        with scope_for(args, kwargs)(write=write):
                            manager.__exit__(None, None, None)
            return context
        if inspect.iscoroutinefunction(function):
            @wraps(function)
            async def asynchronous(*args, **kwargs):
                with scope_for(args, kwargs)(write=write):
                    return await function(*args, **kwargs)
            return asynchronous
        @wraps(function)
        def call(*args, **kwargs):
            with scope_for(args, kwargs)(write=write):
                return function(*args, **kwargs)
        return call

    def guard(self, storage):
        """Keep concrete adapter types while guarding public backend methods.

        Adapters declare reads, runtime writes and internal mutation entry points.
        Unlisted methods are writes, with content protection unless the adapter
        declares itself runtime-only. Migration opens its own adapters.
        """
        if storage is None or getattr(storage, "_migration_guard", None) is self:
            return storage
        if getattr(storage, "_migration_guard", None) is not None:
            raise MaintenanceError("storage is already bound to another instance")
        from pyagfs.client import AGFSClient
        from fs.agfs_adapter.client import (
            HTTP_AGFS_READS, HTTP_AGFS_SINGLE_MUTATIONS, NativeAGFSClient, confirmed_rejection,
        )

        # Unknown operations require a content-write permit by default.
        adapter = type(storage)
        sdk = isinstance(storage, AGFSClient)
        reads = HTTP_AGFS_READS if sdk else getattr(adapter, "_migration_reads", ())
        runtime_writes = getattr(adapter, "_migration_runtime_writes", ())
        internal_writes = getattr(adapter, "_migration_internal_writes", ())
        runtime_only = getattr(adapter, "_migration_runtime_only", False)
        skip = {"close", "close_all", "release_for_migration", "shutdown", "stop"}
        for name in dir(adapter):
            if name in skip or (name.startswith("_") and name not in internal_writes):
                continue
            function = getattr(storage, name, None)
            if not inspect.ismethod(function):
                continue
            write = name not in reads
            # The HTTP SDK retries writes by default. A final rejection cannot
            # confirm whether the preceding attempt wrote anything.
            if sdk and name == "write":
                original_write = function
                @wraps(original_write)
                def write_once(path, data, max_retries=0, _write=original_write):
                    return _write(path, data, max_retries=0)
                function = write_once
            content = False if runtime_only or name in runtime_writes else _content_arguments
            if content and name in internal_writes:
                content = lambda *a, **kw: not self._runtime_backend.get() and _content_arguments(*a, **kw)
            single_attempt = sdk and name in HTTP_AGFS_SINGLE_MUTATIONS
            if write and isinstance(storage, (AGFSClient, NativeAGFSClient)):
                signature = inspect.signature(function)
                @wraps(function)
                def transport(*args, _function=function, _signature=signature,
                              _rejections=single_attempt, **kwargs):
                    paths = _transport_paths(_signature, args, kwargs)
                    payload = any(_content_path(p) for p in paths) if paths else True
                    if self._runtime_backend.get():
                        payload = False  # Actual content adapters still acquire their own permit.
                    with self.backend(write=True, content=payload):
                        try:
                            return _function(*args, **kwargs)
                        except Exception as exc:
                            if _rejections and confirmed_rejection(exc):
                                record_write_completion(exc, WriteCompletion.SETTLED)
                            raise
                setattr(storage, name, transport)
            else:
                setattr(storage, name, self.wrap(function, write=write, backend=True, content=content))
        storage._migration_guard = self
        return storage


def control_client(client, admission, prefix):
    """Allow only control-namespace reads; every mutation retains admission."""

    readers = {name: getattr(client, name) for name in ('read', 'cat', 'ls') if callable(getattr(client, name, None))}
    admission.guard(client)
    root = posixpath.normpath('/' + prefix.strip('/'))
    def control_path(path, *, parents=False, listing=False):
        normalized = posixpath.normpath(path)
        relative = normalized[len(root.rstrip('/')) + 1:] if normalized.startswith(root.rstrip('/') + '/') else ''
        parts = relative.split('/')
        return ((listing and parts == ['accounts']) or parts[0] == '_system' or
                (len(parts) >= 3 and parts[0] == 'accounts' and parts[2] == '_system') or
                (parents and (normalized in {'/', root} or
                              (parts[0] == 'accounts' and len(parts) <= 2))))
    for method in ('write', 'write_stream', 'mkdir', 'rm', 'mv', 'create'):
        mutation = getattr(client, method, None)
        if not callable(mutation):
            continue
        signature = inspect.signature(mutation)
        @wraps(mutation)
        def mutate(*args, _method=method, _mutation=mutation, _signature=signature, **kwargs):
            paths = _transport_paths(_signature, args, kwargs)
            if not paths or not all(control_path(path, parents=_method == 'mkdir') for path in paths):
                raise MaintenanceError('control client cannot mutate business storage')
            with admission.backend(write=True, content=False):
                return _mutation(*args, **kwargs)
        setattr(client, method, mutate)
    for name, original in readers.items():
        @wraps(original)
        def read(path, *args, _original=original, _name=name, **kwargs):
            if not control_path(path, listing=_name == 'ls' and not args and not kwargs):
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
            with admission.backend(write=constructing, content=False):
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
