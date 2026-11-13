"""Two AGFS processes project the LoCoMo sample0 session14 sunflower event."""

from dataclasses import asdict
import hashlib
import importlib
import json
import math
import multiprocessing
import os
from pathlib import Path
from queue import Empty
import sys
import time
import traceback
from uuid import uuid4

import pytest


def _runtime(address, mount):
    source = os.environ.get("OGMEM_OUTBOX_TEST_SOURCE")
    if source:
        sys.path.insert(0, str(Path(source).resolve()))
    importlib.import_module("core.models")
    from pyagfs import AGFSClient
    from commit.outbox_store import OutboxStore
    from fs.agfs_adapter.agfs_context_fs import AGFSContextFS

    client = AGFSClient(address)
    client.session.trust_env = False
    client.session.proxies.clear()
    fs = AGFSContextFS(client, mount_prefix=mount)
    return client, fs, OutboxStore(client, fs, mount_prefix=mount)


def _sunflower():
    path = os.environ.get("OGMEM_OUTBOX_TEST_CASE")
    if path:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    content = "\n".join((
        "Caroline asked Melanie what type of art she loves.",
        "Melanie said painting landscapes and still life is her favorite.",
        "Melanie said nature is amazing.",
        "Melanie shared https://i.redd.it/k9wcp85ledi91.jpg, described as a photo of a painting of a sunflower on a canvas.",
        "Melanie said she did the sunflower painting recently.",
        "Caroline asked whether Melanie had more paintings coming.",
        "Melanie said she felt inspired by autumn.",
        "Melanie said she was planning a few paintings.",
    ))
    return {"source": "LoCoMo sample0 session14", "node": {
        "metadata": {"routing_key": "melanie_shares_sunflower_painting_plans_autumn_works_august_2023",
                     "when": "2023-08-25 (painting done recently; autumn works planned)",
                     "who": "Melanie; Caroline", "where": "Group chat",
                     "title": "Melanie Shares a Sunflower Painting and Plans Autumn Paintings",
                     "provenance_ids": ["prov:1:archive:session14:message-one"]},
        "abstract": "Melanie shared a recent sunflower painting and planned autumn paintings.",
        "overview": "Melanie loves landscapes and still life. She shared a sunflower painting and planned autumn works.",
        "content": content}}


class _SharedIndex:
    def __init__(self, rows, operations, lock):
        self.rows, self.operations, self.lock = rows, operations, lock

    def upsert(self, records):

        with self.lock:
            for record in records:
                value = asdict(record)
                value["metadata"] = dict(value["metadata"])
                self.rows[record.id] = value
            self.operations.append({"pid": os.getpid(), "operation": "upsert",
                                    "ids": [record.id for record in records], "at_ns": time.monotonic_ns()})

    def delete(self, ids):
        with self.lock:
            for record_id in ids:
                self.rows.pop(record_id, None)
            self.operations.append({"pid": os.getpid(), "operation": "delete",
                                    "ids": list(ids), "at_ns": time.monotonic_ns()})


def _consume(mode, address, mount, account, reference, ready, read_gates, gates, queue, rows, operations, lock):
    def emit(phase, **values):
        queue.put({"phase": phase, "mode": mode, "pid": os.getpid(),
                   "at_ns": time.monotonic_ns(), **values})

    try:
        client, fs, store = _runtime(address, mount)
        from index.outbox_worker import OutboxWorker
        from providers.embedder.mock_embedder import MockEmbedder

        phase = {"claim": False, "read": False}
        read = client.read
        processing_path = fs._uri_to_agfs_path(reference[0]) + ".outbox/" + reference[1] + ".processing"

        def coordinated_read(path, *args, **kwargs):
            coordinate = phase["claim"] and not phase["read"] and path == processing_path
            observed = "present"
            try:
                return read(path, *args, **kwargs)
            except Exception as exc:
                observed = "absent" if store._is_not_found_error(exc) else type(exc).__name__
                raise
            finally:
                if coordinate:
                    phase["read"] = True
                    emit("processing_read", outcome=observed)
                    assert read_gates[mode].wait(30)

        client.read = coordinated_read
        claim, acknowledge = store.try_acquire, store.mark_done

        def observed_claim(*args, **kwargs):
            acquired = claim(*args, **kwargs)
            emit("claimed", acquired=acquired)
            return acquired

        def observed_acknowledgement(*args, **kwargs):
            result = acknowledge(*args, **kwargs)
            emit("acknowledged", event_id=args[0].event_id)
            return result

        store.try_acquire, store.mark_done = observed_claim, observed_acknowledgement
        load = store.list_pending if mode == "scan" else store.list_pending_events

        def coordinated_load(*args):
            events = load(*args)
            assert [(uri, event.event_id) for uri, event in events] == [reference]
            emit("pending_loaded", event_id=reference[1])
            ready.wait(timeout=30)
            phase["claim"] = True
            return events

        class GatedEmbedder(MockEmbedder):
            def embed_texts(self, texts):
                emit("embedding")
                assert gates[mode].wait(30)
                return super().embed_texts(texts)

        worker = OutboxWorker(_SharedIndex(rows, operations, lock), GatedEmbedder(dimension=8),
                             fs=fs, directory_summary_enabled=False)
        if mode == "scan":
            store.list_pending = coordinated_load
            stats = worker.run_once(store, [account], worker_id="background-scan")
        else:
            stats = worker.run_once(store, [account], worker_id="request-targeted",
                                    pending_events=coordinated_load([reference]))
        modules = (sys.modules["commit.outbox_store"], sys.modules["index.outbox_worker"])
        emit("done", stats=stats,
             products={module.__name__: {"path": module.__file__,
                       "sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}
                       for module in modules})
    except BaseException:
        emit("error", traceback=traceback.format_exc())


@pytest.mark.integration
def test_scanning_and_targeted_processes_complete_one_persisted_sunflower_event():
    address = os.environ.get("OGMEM_AGFS_TEST_URL")
    if not address:
        pytest.skip("set OGMEM_AGFS_TEST_URL to a disposable AGFS server")
    mount = os.environ.get("OGMEM_AGFS_TEST_MOUNT", "/local")
    _, fs, store = _runtime(address, mount)
    from commit.outbox_scope import capture_outbox_events, current_outbox_events
    from commit.write_receipts import RECEIPTS_KEY, operation_digest, receipt_plan
    from core.models import CandidateMemory, RequestContext
    from core.uri_resolver import URIResolver
    from extraction.schemas.registry import SchemaRegistry
    from index.index_record_builder import build_index_records
    from service.api import MemoryWriteAPI

    ctx = RequestContext("outbox-race-" + uuid4().hex, "reader", "agent", "session14", "trace")
    fixture = _sunflower()
    source = fixture["node"]
    meta = source["metadata"]
    candidate = CandidateMemory(
        category="event", owner_scope="user", routing_key=meta["routing_key"], confidence=.9,
        abstract=source["abstract"], overview=source["overview"], content=source["content"],
        when=meta["when"], who=meta["who"], where=meta["where"], title=meta["title"],
        provenance_ids=meta["provenance_ids"],
        write_operation_id="receipt-v2:" + hashlib.sha256((ctx.account_id + source["content"]).encode()).hexdigest(),
    )

    @capture_outbox_events
    def publish():
        result = MemoryWriteAPI(fs, object(), store, directory_summary_enabled=False).write_memory(candidate, ctx)
        return result, current_outbox_events(ctx.account_id)

    result, references = publish()
    assert result["action"] == "create"
    assert len(references) == 1
    reference = references[0]
    committed = fs.read_node(result["target_uri"], ctx)
    expected = build_index_records(committed)
    assert len(expected) == 2
    mp = multiprocessing.get_context("spawn")
    with mp.Manager() as manager:
        rows, operations, lock = manager.dict(), manager.list(), manager.Lock()
        queue, ready = mp.Queue(), mp.Barrier(2)
        read_gates = {mode: mp.Event() for mode in ("scan", "request")}
        gates = {mode: mp.Event() for mode in read_gates}
        processes = [mp.Process(target=_consume, args=(mode, address, mount, ctx.account_id, reference,
                     ready, read_gates, gates, queue, rows, operations, lock)) for mode in read_gates]
        messages, first, release_at, reads_release_at = [], None, None, None
        deadline = time.monotonic() + 75
        try:
            for process in processes:
                process.start()
            while sum(item["phase"] in {"done", "error"} for item in messages) < 2:
                assert time.monotonic() < deadline, messages
                try:
                    item = queue.get(timeout=.1)
                except Empty:
                    item = None
                if item is not None:
                    messages.append(item)
                    if item["phase"] == "processing_read" and reads_release_at is None:
                        reads_release_at = time.monotonic() + 1
                    if item["phase"] == "embedding" and first is None:
                        first, release_at = item["mode"], time.monotonic() + 1
                    if item["phase"] == "error" or (item["phase"] == "done" and item["mode"] == first):
                        for gate in gates.values():
                            gate.set()
                if release_at is not None and time.monotonic() >= release_at:
                    gates[first].set()
                if reads_release_at is not None and time.monotonic() >= reads_release_at:
                    for gate in read_gates.values():
                        gate.set()
        finally:
            for gate in (*gates.values(), *read_gates.values()):
                gate.set()
            for process in processes:
                process.join(timeout=5)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
        records, vector_operations = dict(rows), list(operations)

    _, fresh_fs, fresh_store = _runtime(address, mount)
    persisted = fresh_fs.read_node(committed.uri, ctx)
    receipt = receipt_plan(fresh_fs, URIResolver(SchemaRegistry()), candidate, ctx)
    unfinished = fresh_store.count_unfinished(ctx.account_id)
    receipts_match = persisted.metadata[RECEIPTS_KEY] == {
        candidate.write_operation_id: operation_digest(candidate, ctx)}
    expected_ids = {record.id for record in expected}
    checks = {"two_processes_completed": len({item["pid"] for item in messages if item["phase"] == "done"}) == 2,
              "two_projected_records": len(records) == 2 and set(records) == expected_ids,
              "source_receipt_preserved": receipts_match and receipt.target_uri == committed.uri,
              "source_preserved": persisted == committed,
              "outbox_complete": unfinished == {"pending": 0, "processing": 0, "total": 0},
              "one_completion": sum(item["stats"]["succeeded"] for item in messages if item["phase"] == "done") == 1}
    evidence = {"case": "LoCoMo sample0 session14 sunflower", "fixture_source_audit_sha256": fixture.get("source_audit_sha256"),
                "account_id": ctx.account_id, "uri": committed.uri, "event_id": reference[1],
                "vector_provider": "process-shared test index", "messages": messages, "checks": checks,
                "vector_operations": vector_operations, "records": records,
                "source_receipts": persisted.metadata[RECEIPTS_KEY], "unfinished": unfinished}
    output = os.environ.get("OGMEM_OUTBOX_TEST_EVIDENCE")
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, ensure_ascii=False))
    assert all(checks.values()), checks
    for expected_record in expected:
        actual = records[expected_record.id]
        assert (actual["uri"], actual["level"], actual["text"], actual["filters"]) == (
            expected_record.uri, expected_record.level, expected_record.text, expected_record.filters)
        vector = actual["metadata"]["_embedding"]
        assert len(vector) == 8 and all(math.isfinite(value) for value in vector) and any(vector)
