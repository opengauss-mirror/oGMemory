"""HTTP-compatible client for the process-local AGFS C ABI."""

import ctypes as c
import json
import os
import posixpath
import threading
from contextlib import contextmanager
from pathlib import Path

import yaml
from pyagfs import AGFSClient
from pyagfs.exceptions import AGFSClientError, AGFSHTTPError

# The C library shares one mount table among all clients in a process.
_mount_lock = threading.Lock()
_mounts = {}


class NativeAGFSHTTPError(AGFSHTTPError):
    """The synchronous C call has returned; no remote write remains in flight."""


def confirmed_rejection(exc):
    """A single request returned a filesystem rejection, not a transport failure."""
    from requests.exceptions import HTTPError

    cause = exc.__context__
    return (isinstance(exc, NativeAGFSHTTPError)
            or (isinstance(exc, AGFSClientError) and isinstance(cause, HTTPError)
                and cause.response is not None and cause.response.status_code in {403, 404, 409}))


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
        self.id = self.lib.AGFS_NewClient()
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
        with self._volume_lock(path, shared=method in {"Ls", "Stat"}):
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
            raise NativeAGFSHTTPError(message, status_code=404)
        if "already exists" in lower or "file exists" in lower:
            raise NativeAGFSHTTPError(message, status_code=409)
        if "permission denied" in lower or "access denied" in lower:
            raise NativeAGFSHTTPError(message, status_code=403)
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

    def write_stream(self, path, stream):
        """Restore a file through the C ABI with bounded transfer buffers."""
        write = getattr(self.lib, "AGFS_WriteChunk", None)
        if write is None:
            raise AGFSClientError("native AGFS library requires WriteChunk for migration recovery")
        write.argtypes = [c.c_int64, c.c_char_p, c.c_void_p, c.c_int64, c.c_int64, c.c_int]
        write.restype = c.c_void_p
        with self._volume_lock(path, shared=False):
            offset = 0
            while True:
                chunk = stream.read(1024 * 1024)
                value = json.loads(self._string(write(self.id, path.encode(), chunk, len(chunk), offset, int(offset == 0))))
                if 'error_id' in value:
                    self._error(value['error_id'])
                if value.get('bytes_written') != len(chunk):
                    raise AGFSClientError('incomplete AGFS recovery write')
                offset += len(chunk)
                if not chunk:
                    break

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
