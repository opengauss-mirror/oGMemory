"""Real C ABI integration: AGFS_NATIVE_LIBRARY selects the compiled library."""

import multiprocessing
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest
import yaml
from pyagfs.exceptions import AGFSHTTPError

from fs.agfs_adapter.client import NativeAGFSClient, create_agfs_client
from providers.unified_config import OgMemConfig


@pytest.fixture
def native(tmp_path):
    library = os.environ.get("AGFS_NATIVE_LIBRARY")
    if not library:
        pytest.skip("AGFS_NATIVE_LIBRARY supplies the integration library")
    mount = "/native-" + uuid4().hex
    config = tmp_path / "agfs.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "plugins": {
                    "localfs": {
                        "enabled": True,
                        "path": mount,
                        "config": {"local_dir": str(tmp_path)},
                    }
                }
            }
        )
    )
    return NativeAGFSClient(library, config), mount, config


def test_native_crud_binary_unicode_and_concurrency(native):
    client, mount, _ = native
    client.mkdir(mount + "/目录")
    path = mount + "/目录/数据"
    payload = "你好".encode() + bytes(range(256))
    client.create(path)
    assert client.stat(path)["size"] == 0
    client.write(path, payload)
    assert client.read(path) == payload
    assert client.read(path, 2, 9) == payload[2:11]
    assert client.stat(path)["size"] == len(payload)
    client.mv(path, path + "-renamed")
    assert client.read(path + "-renamed") == payload
    assert client.ls(mount + "/目录")[0]["name"] == "数据-renamed"

    def roundtrip(i):
        target = mount + "/thread-" + str(i)
        client.write(target, str(i).encode())
        assert client.read(target) == str(i).encode()

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(roundtrip, range(32)))
    client.rm(path + "-renamed")
    client.mkdir(mount + "/empty")
    assert client.stat(mount + "/empty")["isDir"]


def _claim_files(config, mount, owner, ready, results):
    client = NativeAGFSClient(os.environ["AGFS_NATIVE_LIBRARY"], config)
    claimed = []
    for i in range(12):
        ready.wait(timeout=30)
        path = f"{mount}/claim-{i}"
        try:
            client.create(path)
        except AGFSHTTPError as exc:
            if exc.status_code != 409:
                raise
        else:
            client.write(path, owner.encode())
            claimed.append((path, owner))
    results.put(claimed)


def test_processes_share_one_owner_per_claim(native):
    client, mount, config = native
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Barrier(4)
    results = ctx.Queue()
    workers = [
        ctx.Process(target=_claim_files, args=(config, mount, str(i), ready, results))
        for i in range(4)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(45)
        assert worker.exitcode == 0
    claims = [claim for _ in workers for claim in results.get(timeout=5)]
    assert len(claims) == len(dict(claims)) == 12
    assert all(client.read(path) == owner.encode() for path, owner in claims)


def _publish_revisions(config, path, ready, size):
    client = NativeAGFSClient(os.environ["AGFS_NATIVE_LIBRARY"], config)
    ready.wait(timeout=30)
    for revision in range(100):
        client.write(path, bytes([revision]) * size)


def test_processes_read_complete_revisions(native):
    client, mount, config = native
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Barrier(2)
    path = mount + "/shared-state"
    size = 1024 * 1024
    client.write(path, b"A" * size)
    writer = ctx.Process(target=_publish_revisions, args=(config, path, ready, size))
    writer.start()
    ready.wait(timeout=30)
    while writer.is_alive():
        data = client.read(path)
        assert len(data) == size
        assert data == data[:1] * size
    writer.join(30)
    assert writer.exitcode == 0
    assert client.read(path) == bytes([99]) * size


def test_native_session_archive_and_process_reopen(native):
    from core.models import RequestContext, Role
    from fs.agfs_adapter.agfs_context_fs import AGFSContextFS
    from session.archive_store import SessionArchiveStore

    client, mount, config = native
    store = SessionArchiveStore(AGFSContextFS(client, mount_prefix=mount))
    ctx = RequestContext("acme", "alice", "agent", "session", "trace", Role.ADMIN)
    result = store.write_archive(
        "session",
        "overview",
        "abstract",
        [{"role": "user", "content": "你好"}],
        ctx,
        archive_id="archive",
    )
    assert result.success
    assert store.read_archive("session", "archive", ctx).overview == "overview"
    code = """from fs.agfs_adapter.client import NativeAGFSClient
from fs.agfs_adapter.agfs_context_fs import AGFSContextFS
from session.archive_store import SessionArchiveStore
from core.models import RequestContext, Role
import sys
client=NativeAGFSClient(sys.argv[1],sys.argv[2])
store=SessionArchiveStore(AGFSContextFS(client,mount_prefix=sys.argv[3]))
ctx=RequestContext('acme','alice','agent','session','trace',Role.ADMIN)
assert store.read_archive('session','archive',ctx).overview == 'overview'
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            os.environ["AGFS_NATIVE_LIBRARY"],
            str(config),
            mount,
        ],
        check=True,
    )
    second = NativeAGFSClient(os.environ["AGFS_NATIVE_LIBRARY"], config)
    assert second.ls(mount)


def test_transport_factories(native):
    client, mount, config = native
    cfg = OgMemConfig(
        agfs_transport="native",
        agfs_native_library=os.environ["AGFS_NATIVE_LIBRARY"],
        agfs_native_config=str(config),
    )
    assert create_agfs_client(cfg).ls(mount)
    cfg.agfs_transport = "http"
    http = create_agfs_client(cfg)
    assert http.api_base == cfg.agfs_base_url + "/api/v1"


def test_native_service_health_reports_library(native):
    from server.memory_service import MemoryService

    client, mount, config = native
    cfg = OgMemConfig(
        provider="mock",
        vector_db_type="memory",
        storage_backend="agfs",
        agfs_transport="native",
        agfs_native_library=os.environ["AGFS_NATIVE_LIBRARY"],
        agfs_native_config=str(config),
    )
    health = MemoryService(config=cfg).health()
    assert health["status"] == "ok"
    assert health["agfs"] is True
    assert health["agfs_transport"] == "native"
    assert health["agfs_url"] == client.api_base


def test_bundled_transport_and_explicit_http_endpoint(tmp_path, monkeypatch):
    config = tmp_path / "ogmemory.yaml"
    config.write_text("{}")
    monkeypatch.delenv("AGFS_TRANSPORT", raising=False)
    monkeypatch.delenv("AGFS_BASE_URL", raising=False)
    monkeypatch.setenv("AGFS_DEFAULT_TRANSPORT", "native")
    assert OgMemConfig.load(str(config)).agfs_transport == "native"

    url = "http://storage:1833"
    monkeypatch.setenv("AGFS_BASE_URL", url)
    cfg = OgMemConfig.load(str(config))
    assert cfg.agfs_transport == "http"
    assert create_agfs_client(cfg).api_base == url + "/api/v1"

    config.write_text("agfs:\n  transport: native\n")
    assert OgMemConfig.load(str(config)).agfs_transport == "native"
