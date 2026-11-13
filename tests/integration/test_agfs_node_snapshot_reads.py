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
def native_agfs(tmp_path):
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
                        if client.get(address + "/api/v1/files?path=/serverinfo/version", timeout=.25).ok:
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
