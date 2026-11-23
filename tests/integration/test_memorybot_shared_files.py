from __future__ import annotations

import hashlib
import json
import os

import pytest

from memorybot.files import AGFSFileStore
from memorybot.models import new_id
from tests.integration.test_memorybot_runtime import execute, source, task
from tests.integration.test_memorybot_runtime import runtime as runtime

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("purpose", ["source", "parsed", "artifact"])
def test_two_agfs_instances_share_source_parsed_and_artifact_bytes(purpose, tmp_path, request):
    urls = [
        os.environ.get("MEMORYBOT_TEST_AGFS_URL"),
        os.environ.get("MEMORYBOT_TEST_AGFS_PEER_URL"),
    ]
    if not all(urls):
        pytest.skip("Two AGFS endpoints select a shared integration mount")
    stores = [AGFSFileStore(url, "/memorybot") for url in urls]
    active_runtime = request.getfixturevalue("runtime") if purpose == "artifact" else None
    account = active_runtime.ctx.account_id if active_runtime else new_id("shared")
    document = source(active_runtime) if active_runtime else None
    size = 64 * 1024 if active_runtime else 1024 * 1024
    evidence, saved = [], []
    try:
        for direction, (writer, reader) in enumerate((stores, stores[::-1]), 1):
            data = bytes([64 + direction]) * size
            if active_runtime:
                active_runtime.worker.files = writer
                active_runtime.llm.stream_hook = lambda *_: ([], data.decode())
                item = task(
                    active_runtime, "chat",
                    {"text": "整理为成果", "file_ids": [document["file_id"]]},
                )
                completed = execute(active_runtime, item)
                file_id, = completed["result"]["file_ids"]
                publication = active_runtime.store.get(account, "bot_file", {"file_id": file_id})
                assert publication["purpose"] == "artifact" and publication["state"] == "ready"
                assert publication["task_id"] == item["task_id"]
            else:
                publication = writer.publish(account, new_id("f"), data, name=purpose + ".md")
            saved.append(publication["storage_key"])
            key = publication["storage_key"]
            expected = data
            if purpose == "parsed":
                parsed = {"text": data.decode()[:size - 12]}
                key = writer.write_parsed(account, key, parsed)
                expected = json.dumps(parsed, ensure_ascii=False).encode()
            received = reader.read(account, key)
            assert len(received) == len(expected) == size
            assert hashlib.sha256(received).hexdigest() == hashlib.sha256(expected).hexdigest()
            assert reader.read(account, key, offset=1024, size=4096) == expected[1024:5120]
            evidence.append(
                {
                    "purpose": purpose,
                    "direction": direction,
                    "bytes": len(received),
                    "sha256": hashlib.sha256(received).hexdigest(),
                }
            )
    finally:
        for key in saved:
            stores[0].delete(account, key)
    directory = os.environ.get("MEMORYBOT_TEST_EVIDENCE_DIR")
    if directory:
        from pathlib import Path

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
    else:
        target = tmp_path
    (target / f"agfs-shared-{purpose}.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
