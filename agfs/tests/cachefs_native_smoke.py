"""Run against a built libagfsbinding and a disposable Redis endpoint.

python agfs/tests/cachefs_native_smoke.py /absolute/libagfsbinding.so redis://127.0.0.1:6379
"""
import ctypes as c
import json
import sys
import tempfile

lib = c.CDLL(sys.argv[1])
for name, args in {
    "AGFS_ConfigureCache": [c.c_char_p],
    "AGFS_Mount": [c.c_int64, c.c_char_p, c.c_char_p, c.c_char_p],
    "AGFS_Write": [c.c_int64, c.c_char_p, c.c_void_p, c.c_int64],
    "AGFS_CacheStats": [],
    "AGFS_Shutdown": [],
}.items():
    fn = getattr(lib, name)
    fn.argtypes = args
    fn.restype = c.c_void_p
lib.AGFS_FreeString.argtypes = [c.c_void_p]
lib.AGFS_FreeString.restype = None
lib.AGFS_Read.argtypes = [c.c_int64, c.c_char_p, c.c_int64, c.c_int64,
                          c.POINTER(c.c_void_p), c.POINTER(c.c_int64)]
lib.AGFS_Read.restype = c.c_int64


def returned(ptr):
    if not ptr:
        return None
    try:
        return c.string_at(ptr).decode()
    finally:
        lib.AGFS_FreeString(ptr)


def read(path):
    data, size = c.c_void_p(), c.c_int64()
    error = lib.AGFS_Read(1, path, 0, -1, c.byref(data), c.byref(size))
    assert error == 0, error
    try:
        return c.string_at(data, size.value) if size.value else b""
    finally:
        lib.AGFS_FreeString(data)


def write(path, value):
    result = json.loads(returned(lib.AGFS_Write(1, path, value, len(value))))
    assert "error_id" not in result, result


with tempfile.TemporaryDirectory() as directory:
    config = {
        "cache": {"provider": "redis", "command_timeout_ms": 500,
                  "params": {"endpoints": [sys.argv[2]]}},
        "cachefs": {"enabled": True, "namespace": "native-smoke", "mounts": ["/local"]},
    }
    assert returned(lib.AGFS_ConfigureCache(json.dumps(config).encode())) is None
    try:
        result = json.loads(returned(lib.AGFS_Mount(
            1, b"localfs", b"/local", json.dumps({"local_dir": directory}).encode())))
        assert "error_id" not in result, result
        for payload in [b"old", b"new\0binary", b""]:
            write(b"/local/.meta.json", payload)
            assert read(b"/local/.meta.json") == payload
            assert read(b"/local/.meta.json") == payload
        stats = json.loads(returned(lib.AGFS_CacheStats()))["/local"]
        assert stats["Hits"] == 3, stats
        write(b"/local/file.lock", b"owner")
        assert read(b"/local/file.lock") == b"owner"
        assert read(b"/local/file.lock") == b"owner"
        stats = json.loads(returned(lib.AGFS_CacheStats()))["/local"]
        assert stats["Hits"] == 3 and stats["Bypasses"] == 2, stats
        print(json.dumps(stats))
    finally:
        assert returned(lib.AGFS_Shutdown()) is None
    assert returned(lib.AGFS_Shutdown()) is None
print("native CacheFS smoke passed")
