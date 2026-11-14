"""Independent AGFS readers observe complete Atlas preference revisions."""

import json
import multiprocessing
import os
import socket
import subprocess
import time
import traceback
from pathlib import Path

import pytest
import requests
import yaml

from core.models import ContextNode, RequestContext
from fs.agfs_adapter.agfs_context_fs import AGFSContextFS
from pyagfs import AGFSClient


FINAL_VERSION = 17


def _runtime(address, account):
    client = AGFSClient(address)
    client.session.trust_env = False
    client.session.proxies.clear()
    return client, AGFSContextFS(client, mount_prefix="/local"), RequestContext(
        account, "ben", "main", "atlas-release", "snapshot-check")


def _preference(ctx, version):
    facts = "Atlas 每周五北京时间19点发布，包管理工具统一使用 npm。"
    if version >= 2:
        facts += "发布前必须运行 npm test。"
    if version >= 3:
        facts += f"已完成第{version - 2}次发布准备确认。"
    return ContextNode(
        uri=f"ctx://{ctx.account_id}/users/ben/memories/preferences/atlas_release_schedule",
        context_type="MEMORY", category="preference", level=2,
        owner_space=ctx.user_space_name(), abstract=facts,
        overview="## Atlas 发布约定\n" + facts,
        content="# Atlas 发布习惯\n\n" + facts,
        metadata={"_expected_version": version - 1, "routing_key": "atlas_release_schedule",
                  "object_name": "release_schedule", "scope_ref": "Atlas", "who": "ben"},
    )


def _assert_revision(node, ctx):
    version = node.metadata["version"]
    assert 1 <= version <= FINAL_VERSION
    expected = _preference(ctx, version)
    assert (node.uri, node.abstract, node.overview, node.content) == (
        expected.uri, expected.abstract, expected.overview, expected.content)
    assert {key: node.metadata[key] for key in ("routing_key", "object_name", "scope_ref", "who")} == {
        key: expected.metadata[key] for key in ("routing_key", "object_name", "scope_ref", "who")}
    return version


def _exercise(mode, address, account, ready, finished, output):
    result = {"mode": mode, "pid": os.getpid(), "operations": []}
    client = None
    try:
        client, fs, ctx = _runtime(address, account)
        uri = _preference(ctx, 1).uri
        ready.wait(timeout=30)
        if mode == "writer":
            for version in range(2, FINAL_VERSION + 1):
                started = time.monotonic_ns()
                fs.write_node(_preference(ctx, version), ctx)
                result["operations"].append({"start_ns": started, "end_ns": time.monotonic_ns(),
                                             "version": version})
        else:
            while not finished.is_set() or len(result["operations"]) < 24:
                started = time.monotonic_ns()
                node = (fs.read_node(uri, ctx) if mode == "read_node"
                        else fs.read_node_fields(uri, ctx, "all"))
                ended = time.monotonic_ns()
                version = _assert_revision(node, ctx)
                result["operations"].append({"start_ns": started, "end_ns": ended, "version": version})
        result["success"] = True
    except BaseException:
        result["error"] = traceback.format_exc()
        raise
    finally:
        if mode == "writer":
            finished.set()
        if client is not None:
            client.session.close()
        Path(output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


@pytest.fixture
def native_agfs(tmp_path, request):
    binary = os.environ.get("OGMEM_TEST_AGFS_BINARY")
    if not binary:
        pytest.skip("set OGMEM_TEST_AGFS_BINARY to compiled AGFS")
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / "config/agfs/config.reference.yaml").read_text(encoding="utf-8"))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    config["server"]["address"] = f"127.0.0.1:{port}"
    data = tmp_path / "agfs"
    data.mkdir()
    config["plugins"]["localfs"]["config"]["local_dir"] = str(data)
    backend = getattr(request, "param", None)
    if backend in {"sqlfs", "tidb"}:
        options = {"backend": "sqlite", "db_path": str(tmp_path / "registry.db")}
        if backend == "tidb":
            # SQLFS's existing utf8mb4 path primary key requires max-index-length = 12288.
            dsn = os.environ.get("OGMEM_TEST_TIDB_DSN")
            if not dsn:
                pytest.skip("set OGMEM_TEST_TIDB_DSN to an empty, disposable TiDB test database")
            options = {"backend": "tidb", "dsn": dsn}
        config["plugins"]["sqlfs"] = {
            "enabled": True, "path": "/sqlfs",
            "config": {**options, "cache_ttl_seconds": 600},
        }
    cfg = tmp_path / "agfs.yaml"
    cfg.write_text(yaml.safe_dump(config), encoding="utf-8")
    address = f"http://127.0.0.1:{port}"
    log_path = tmp_path / "server.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen([binary, "-c", str(cfg)], stdout=log, stderr=log)
        try:
            with requests.Session() as client:
                client.trust_env = False
                deadline = time.monotonic() + 15
                while True:
                    assert proc.poll() is None, log_path.read_text(encoding="utf-8")
                    try:
                        mount = "/sqlfs" if backend in {"sqlfs", "tidb"} else "/local"
                        if client.get(address + "/api/v1/stat", params={"path": mount}, timeout=.25).ok:
                            break
                    except requests.RequestException:
                        pass
                    assert time.monotonic() < deadline, log_path.read_text(encoding="utf-8")
            yield address, tmp_path, proc.pid
        finally:
            if proc.poll() is None:
                proc.terminate()
            exit_code = proc.wait(timeout=10)
            evidence = {"pid": proc.pid, "exit_code": exit_code, "port": port, "directory": str(tmp_path)}
            (tmp_path / "server-exit.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
            print("AGFS_EXIT " + json.dumps(evidence))


@pytest.mark.parametrize("native_agfs", ["sqlfs"], indirect=True)
def test_sqlfs_registration_overwrites_binding_and_workspace_state(native_agfs):
    import sqlite3
    from pyagfs import AGFSClientError
    from server.control_plane_store import ControlPlaneStore
    from service.code_resource_lifecycle import CodeResourceLifecycleService
    from tests.unit.service.test_code_workspace_sync import _config, _ctx, _register, _service

    address, directory, _ = native_agfs
    client = AGFSClient(address)
    with client.session:
        client.session.trust_env = False
        store = ControlPlaneStore("/sqlfs", client=client)
        config = _config(code_workspace_allowed_roots=[str(directory)])
        lifecycle = CodeResourceLifecycleService(
            context_fs=AGFSContextFS(client, mount_prefix="/sqlfs"), outbox_store=object(), config=config,
        )
        service = _service(store, lifecycle, config=config)
        repo = directory / "repo"
        repo.mkdir()
        root_id = _register(service, repo)
        assert _register(service, repo) == root_id
        other = directory / "second"
        other.mkdir()
        _register(service, other, name="second")
        store.mutate_code_workspace_sync_state("acct", root_id, lambda state: {**state, "dirty": False})
        service.mark_dirty(root_id=root_id, ctx=_ctx())
        state = store.read_code_workspace_sync_state("acct", root_id)
        assert state["dirty"] and state["dirty_generation"] == 3
        # Reject an UPDATE of the published row, but allow staging and lock operations.
        with sqlite3.connect(directory / "registry.db") as db:
            db.execute(f"""CREATE TRIGGER reject_update BEFORE UPDATE OF data ON files
                WHEN OLD.path = '/accounts/acct/_system/code_workspace_sync/{root_id}.json'
                BEGIN SELECT RAISE(ABORT, 'injected update failure'); END""")
        with pytest.raises(AGFSClientError, match="injected update failure"):
            store.write_code_workspace_sync_state("acct", root_id, {**state, "dirty": False})
        assert store.read_code_workspace_sync_state("acct", root_id) == state


def _registration_store(transport, address, directory, mount="/sqlfs"):
    from fs.agfs_adapter.client import NativeAGFSClient
    from server.control_plane_store import ControlPlaneStore

    if transport == "local":
        return ControlPlaneStore("", local_root=str(directory / "agfs"))
    if transport == "native":
        client = NativeAGFSClient(os.environ["AGFS_NATIVE_LIBRARY"], directory / "agfs.yaml")
    else:
        client = AGFSClient(address)
        client.session.trust_env = False
    return ControlPlaneStore(mount, client=client)


def _native_claims(directory, action):
    from tests.unit.service.test_code_workspace_sync import FakeLifecycle, _config, _ctx, _service

    store = _registration_store("native", None, directory)
    service = _service(store, FakeLifecycle(), config=_config(code_workspace_allowed_roots=[str(directory)]))
    if action == "seed":
        service.register(path=str(directory / "seed"), ctx=_ctx("old"))
    elif action == "warm":
        assert {entry["name"] for entry in store._client.ls("/sqlfs/accounts")} == {"old"}
        assert len(store._client.ls("/sqlfs/accounts/old/_system/code_workspace_sync")) == 2
    elif action == "publish":
        for account in ("old", "new"):
            service.register(path=str(directory / account / "repo"), ctx=_ctx(account))
    else:
        # The ordinary listing is still stale in this very worker.
        assert {entry["name"] for entry in store._client.ls("/sqlfs/accounts")} == {"old"}
        assert len(store._client.ls("/sqlfs/accounts/old/_system/code_workspace_sync")) == 2
        for account in ("old", "new"):
            root = directory / account / "repo"
            for candidate in (root, root.parent, root / "child"):
                with pytest.raises(FileExistsError, match="conflicts with an existing registration"):
                    service.register(path=str(candidate), ctx=_ctx("contender"))
        assert store.list_code_workspace_sync_states("contender") == []
        assert service.register(path=str(directory / "sibling"), ctx=_ctx("contender"))["accepted"]


@pytest.mark.parametrize("native_agfs", ["sqlfs", "tidb"], indirect=True)
def test_native_sqlfs_claims_bypass_both_warm_directory_caches(native_agfs):
    from concurrent.futures import ProcessPoolExecutor

    if not os.environ.get("AGFS_NATIVE_LIBRARY"):
        pytest.skip("set AGFS_NATIVE_LIBRARY to compiled native AGFS")
    _, directory, _ = native_agfs
    for path in ("seed", "old/repo/child", "new/repo/child", "sibling"):
        (directory / path).mkdir(parents=True)
    spawn = multiprocessing.get_context("spawn")
    with (ProcessPoolExecutor(1, mp_context=spawn) as writer,
          ProcessPoolExecutor(1, mp_context=spawn) as reader):
        for worker, action in ((writer, "seed"), (reader, "warm"), (writer, "publish"), (reader, "claim")):
            worker.submit(_native_claims, directory, action).result(timeout=30)


def _hold_registration(transport, address, directory, mount, pipe):
    store = _registration_store(transport, address, directory, mount)
    with store.code_workspace_registration_lock():
        pipe.send(True)
        pipe.recv()


@pytest.mark.parametrize(("native_agfs", "mount", "holder"), [
    ("sqlfs", "/sqlfs", "http"), ("sqlfs", "/sqlfs", "native"),
    ("localfs", "/local", "local"), ("localfs", "/local", "native"), ("localfs", "/local", "http"),
], indirect=["native_agfs"])
def test_registration_recovers_after_killed_owner(native_agfs, mount, holder):
    from tests.unit.service.test_code_workspace_sync import FakeLifecycle, _config, _ctx, _service

    if holder == "native" and not os.environ.get("AGFS_NATIVE_LIBRARY"):
        pytest.skip("set AGFS_NATIVE_LIBRARY to compiled native AGFS")
    address, directory, _ = native_agfs
    spawn = multiprocessing.get_context("spawn")
    parent, child = spawn.Pipe()
    owner = spawn.Process(target=_hold_registration, args=(holder, address, directory, mount, child))
    store = _registration_store("http", address, directory, mount)
    owner.start()
    child.close()
    try:
        assert parent.poll(30)
        assert parent.recv()
        with pytest.raises(RuntimeError, match="registration is busy"):
            with store.code_workspace_registration_lock():
                pytest.fail("live owner lost the session")
    finally:
        owner.kill()
        owner.join(10)
        parent.close()
    assert owner.exitcode is not None and owner.exitcode != 0
    root = directory / "recovered"
    root.mkdir()
    service = _service(store, FakeLifecycle(), config=_config(code_workspace_allowed_roots=[str(directory)]))
    assert service.register(path=str(root), ctx=_ctx("other"))["accepted"]


@pytest.mark.parametrize("native_agfs", ["sqlfs"], indirect=True)
def test_disconnected_http_worker_cannot_overwrite_successor(native_agfs, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from pyagfs import AGFSClientError

    address, directory, _ = native_agfs
    owner = _registration_store("http", address, directory)
    successor = _registration_store("http", address, directory)
    binding_client = AGFSClient(address)
    binding_client.session.trust_env = False
    AGFSContextFS(binding_client, mount_prefix="/sqlfs")
    original_post = owner._client.session.post
    streams = []

    def post(*args, **kwargs):
        response = original_post(*args, **kwargs)
        if kwargs.get("stream"):
            streams.append(response)
        return response

    def publish():
        with successor.code_workspace_registration_lock():
            successor.write_code_workspace_sync_state("acct", "root", {"revision": 1})

    monkeypatch.setattr(owner._client.session, "post", post)
    with owner.code_workspace_registration_lock(), ThreadPoolExecutor(1) as executor:
        streams[0].close()
        executor.submit(publish).result(timeout=10)
        with pytest.raises(AGFSClientError, match="registration session closed"):
            owner.write_code_workspace_sync_state("acct", "root", {"revision": 2})
        with pytest.raises(AGFSClientError, match="registration session closed"):
            binding_client.write(owner.code_workspace_sync_path("acct", "root"), b"stale")
    assert successor.read_code_workspace_sync_state("acct", "root") == {"revision": 1}
    binding_client.session.close()


def test_parallel_atlas_preference_reads_keep_one_committed_revision(native_agfs):
    address, directory, server_pid = native_agfs
    account = "atlas-snapshot"
    client, fs, ctx = _runtime(address, account)
    fs.write_node(_preference(ctx, 1), ctx)
    assert _assert_revision(fs.read_node(_preference(ctx, 1).uri, ctx), ctx) == 1
    spawn = multiprocessing.get_context("spawn")
    ready, finished = spawn.Barrier(3), spawn.Event()
    modes = ("writer", "read_node", "read_node_fields")
    processes = [spawn.Process(target=_exercise, args=(
        mode, address, account, ready, finished, str(directory / f"{mode}.json"))) for mode in modes]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=90)
        results = [json.loads((directory / f"{mode}.json").read_text(encoding="utf-8")) for mode in modes]
        assert all(process.exitcode == 0 for process in processes), results
        assert all(result.get("success") for result in results), results
        writer, *readers = results
        summary = {"server_pid": server_pid, "writer_pid": writer["pid"],
                   "writes": len(writer["operations"]), "readers": []}
        for reader in readers:
            operations = reader["operations"]
            overlap = sum(any(max(read["start_ns"], write["start_ns"]) < min(read["end_ns"], write["end_ns"])
                              for write in writer["operations"]) for read in operations)
            summary["readers"].append({"mode": reader["mode"], "pid": reader["pid"], "reads": len(operations),
                                       "versions": sorted({read["version"] for read in operations}),
                                       "overlap_count": overlap})
        final = fs.read_node(_preference(ctx, 1).uri, ctx)
        summary["final_version"] = _assert_revision(final, ctx)
        summary["worker_exit_codes"] = {mode: process.exitcode for mode, process in zip(modes, processes)}
        evidence_path = directory / "snapshot-read-evidence.json"
        evidence_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print("AGFS_SNAPSHOT " + json.dumps({**summary, "evidence": str(evidence_path)}, ensure_ascii=False))
        assert summary["writes"] == FINAL_VERSION - 1
        assert summary["final_version"] == FINAL_VERSION
        assert all(reader["overlap_count"] > 0 for reader in summary["readers"])
        assert all(reader["reads"] >= 24 for reader in summary["readers"])
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            if process.pid is not None:
                process.join(timeout=10)
        client.session.close()
