"""HTTP-compatible client for the process-local AGFS C ABI."""

import ctypes as c
import json
import os
import posixpath
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import yaml
from pyagfs import AGFSClient
from pyagfs.exceptions import AGFSClientError, AGFSHTTPError

# The C library shares one mount table among all clients in a process.
_mount_lock = threading.Lock()
_mounts = {}
_registration = ContextVar("agfs_registration", default=None)
_REGISTRATION_HEADER = "X-AGFS-Registration"


def enable_registration_fence(client):
    """Tag SDK requests in the registering context, including lifecycle binding writes."""
    if not isinstance(client, AGFSClient):
        return
    session = client.session
    prepare = session.prepare_request
    if getattr(prepare, "_registration_fenced", False):
        return

    def prepare_request(request):
        prepared = prepare(request)
        active = _registration.get()
        if active is not None:
            base, token = active
            if not prepared.url.startswith(base + "/"):
                raise RuntimeError("registration request uses a different AGFS endpoint")
            prepared.url = base + "/registration-session" + prepared.url[len(base):]
            prepared.headers[_REGISTRATION_HEADER] = token
        return prepared

    prepare_request._registration_fenced = True
    session.prepare_request = prepare_request


@contextmanager
def registration_session(client, path):
    if isinstance(client, NativeAGFSClient):
        if not hasattr(client.lib, "AGFS_LockRegistration"):
            raise RuntimeError("AGFS does not support workspace registration sessions")
        try:
            result = client._json("LockRegistration", path.encode())
        except AGFSClientError as exc:
            if str(exc) == "workspace registration is busy":
                raise RuntimeError("workspace registration is busy") from None
            raise RuntimeError("cannot acquire workspace registration session") from None
        if not result["supported"]:
            raise RuntimeError("mount does not support workspace registration sessions")
        reset = _registration.set((client.api_base, result["id"]))
        try:
            yield
        finally:
            _registration.reset(reset)
            client.lib.AGFS_UnlockRegistration(result["id"])
    elif isinstance(client, AGFSClient):
        with client.session.post(
            f"{client.api_base}/registration-session", params={"path": path},
            stream=True, timeout=client.timeout, allow_redirects=False,
        ) as response:
            if response.status_code in {404, 501}:
                raise RuntimeError("mount does not support workspace registration sessions")
            if response.status_code == 409:
                raise RuntimeError("workspace registration is busy")
            if response.status_code != 200:
                raise RuntimeError("cannot acquire workspace registration session")
            token = response.headers.get(_REGISTRATION_HEADER)
            if not token:
                raise RuntimeError("AGFS registration session unavailable")
            reset = _registration.set((client.api_base, token))
            try:
                yield
            finally:
                _registration.reset(reset)
    else:
        raise RuntimeError("AGFS does not support workspace registration sessions")


def list_workspace_registry(client, path):
    """Both directory levels of a claim scan must bypass instance-local caches."""
    if isinstance(client, NativeAGFSClient):
        return client._file_op("LsFresh", path)["files"]
    if not isinstance(client, AGFSClient):
        return client.ls(path)
    response = client.session.get(
        f"{client.api_base}/directories", params={"path": path, "fresh": "true"}, timeout=client.timeout,
    )
    if response.headers.get("X-AGFS-Fresh") != "true":
        raise RuntimeError("AGFS does not support fresh directory listings")
    if response.status_code == 404:
        raise FileNotFoundError
    response.raise_for_status()
    entries = response.json()["files"]
    if entries is not None and not isinstance(entries, list):
        raise ValueError("invalid directory listing")
    return entries or []


def publish_workspace_file(client, staged, target, payload):
    """Publish a staged registry file while its record lock is held."""
    try:
        client.mv(staged, target)
    except AGFSClientError as exc:
        if str(exc).partition(":")[0].lower() != "file already exists":
            raise
        info = client.stat(target)
        if info.get("isDir") is not False or info.get("meta", {}).get("Name") != "sqlfs":
            raise
        # SQLFS writes all bytes with one SQL UPDATE; the old row stays visible until commit.
        client.write(target, payload)
        try:
            client.rm(staged)
        except AGFSClientError:
            pass  # Publication succeeded; the unlisted staging file is safe to leave.


class NativeAGFSClient:

    def __init__(self, library_path, config_path):
        library_path = str(Path(library_path).resolve())
        self.api_base = f"agfs-native://{library_path}"
        self.lib = c.CDLL(library_path)
        signatures = {
            "NewClient": ([], c.c_int64),
            "FreeClient": ([c.c_int64], None),
            "FreeString": ([c.c_void_p], None),
            "GetLastError": ([c.c_int64], c.c_void_p),
            "Mount": ([c.c_int64, c.c_char_p, c.c_char_p, c.c_char_p], c.c_void_p),
            "Read": (
                [
                    c.c_int64,
                    c.c_char_p,
                    c.c_int64,
                    c.c_int64,
                    c.POINTER(c.c_void_p),
                    c.POINTER(c.c_int64),
                ],
                c.c_int64,
            ),
            "Write": ([c.c_int64, c.c_char_p, c.c_void_p, c.c_int64], c.c_void_p),
            "Create": ([c.c_int64, c.c_char_p], c.c_void_p),
            "Mkdir": ([c.c_int64, c.c_char_p, c.c_uint], c.c_void_p),
            "Ls": ([c.c_int64, c.c_char_p], c.c_void_p),
            "Stat": ([c.c_int64, c.c_char_p], c.c_void_p),
            "Mv": ([c.c_int64, c.c_char_p, c.c_char_p], c.c_void_p),
            "Rm": ([c.c_int64, c.c_char_p, c.c_int], c.c_void_p),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(self.lib, "AGFS_" + name)
            fn.argtypes = args
            fn.restype = result
        if hasattr(self.lib, "AGFS_LsFresh"):
            self.lib.AGFS_LsFresh.argtypes = signatures["Ls"][0]
            self.lib.AGFS_LsFresh.restype = c.c_void_p
        if hasattr(self.lib, "AGFS_LockRegistration"):
            self.lib.AGFS_LockRegistration.argtypes = signatures["Ls"][0]
            self.lib.AGFS_LockRegistration.restype = c.c_void_p
            self.lib.AGFS_UnlockRegistration.argtypes = [c.c_int64]
            self.lib.AGFS_UnlockRegistration.restype = None
        self._id = self.lib.AGFS_NewClient()
        settings = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        volumes = {}
        with _mount_lock:
            mounted = _mounts.setdefault((os.getpid(), library_path), {})
            for kind, entries in settings["plugins"].items():
                for entry in entries if isinstance(entries, list) else [entries]:
                    if not entry.get("enabled", False):
                        continue
                    path = posixpath.normpath("/" + entry["path"].lstrip("/"))
                    options = entry.get("config") or {}
                    volumes[path] = os.path.abspath(options["local_dir"]) if kind == "localfs" else None
                    config = json.dumps(options, sort_keys=True)
                    identity = (kind, config)
                    if path in mounted:
                        if mounted[path] != identity:
                            raise AGFSClientError(f"Conflicting native mount: {path}")
                        continue
                    self._json("Mount", kind.encode(), path.encode(), config.encode())
                    mounted[path] = identity
        self._volumes = sorted(volumes.items(), key=lambda item: len(item[0]), reverse=True)

    @property
    def id(self):
        active = _registration.get()
        if active is None:
            return self._id
        if active[0] != self.api_base:
            raise RuntimeError("registration request uses a different native AGFS library")
        return active[1]

    @contextmanager
    def _volume_lock(self, path, *, shared):
        # Native workers share this directory inode; a read must span stat and open
        # while a full write must span truncation and publication of the new bytes.
        path = posixpath.normpath("/" + path.lstrip("/"))
        directory = next((directory for mount, directory in self._volumes
                          if path == mount or path.startswith(mount.rstrip("/") + "/")), None)
        if directory is None:
            yield
            return
        import fcntl

        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            fcntl.flock(fd, fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    def _file_op(self, method, path, *args):
        with self._volume_lock(path, shared=method in {"Ls", "LsFresh", "Stat"}):
            return self._json(method, path.encode(), *args)

    def _string(self, pointer):
        if not pointer:
            raise AGFSClientError("AGFS returned an empty response pointer")
        try:
            return c.string_at(pointer).decode()
        finally:
            self.lib.AGFS_FreeString(pointer)

    def _error(self, error_id):
        message = self._string(self.lib.AGFS_GetLastError(error_id))
        lower = message.lower()
        if "not found" in lower or "no such file" in lower:
            raise AGFSHTTPError(message, status_code=404)
        if "already exists" in lower or "file exists" in lower:
            raise AGFSHTTPError(message, status_code=409)
        if "permission denied" in lower or "access denied" in lower:
            raise AGFSHTTPError(message, status_code=403)
        raise AGFSClientError(message)

    def _json(self, method, *args):
        value = json.loads(
            self._string(getattr(self.lib, "AGFS_" + method)(self.id, *args))
        )
        if isinstance(value, dict) and "error_id" in value:
            self._error(value["error_id"])
        return value

    def read(self, path, offset=0, size=-1):
        pointer = c.c_void_p()
        length = c.c_int64()
        with self._volume_lock(path, shared=True):
            error = self.lib.AGFS_Read(
                self.id, path.encode(), offset, size, c.byref(pointer), c.byref(length)
            )
        if error:
            self._error(error)
        try:
            return c.string_at(pointer, length.value) if length.value else b""
        finally:
            if pointer:
                self.lib.AGFS_FreeString(pointer)

    def write(self, path, data):
        return self._file_op("Write", path, data, len(data))["message"]

    def create(self, path):
        return self._json("Create", path.encode())

    def mkdir(self, path, mode="755"):
        return self._json("Mkdir", path.encode(), int(mode, 8))

    def ls(self, path="/"):
        return self._file_op("Ls", path)["files"]

    def stat(self, path):
        return self._file_op("Stat", path)

    def mv(self, source, destination):
        return self._file_op("Mv", source, destination.encode())

    def rm(self, path, recursive=False):
        return self._json("Rm", path.encode(), int(recursive))

    def close(self):
        self.lib.AGFS_FreeClient(self.id)


def create_agfs_client(config, *, http_client_class=AGFSClient):
    if config.agfs_transport == "native":
        return NativeAGFSClient(config.agfs_native_library, config.agfs_native_config)
    if config.agfs_transport == "http":
        return http_client_class(api_base_url=config.agfs_base_url)
    raise ValueError("agfs.transport must be http or native")
