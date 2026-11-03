"""Runtime recovery of canonical scene evidence and member expansion."""

import json
from dataclasses import replace
from threading import Event, Thread

import pytest

from core.provenance_resolver import ProvenanceResolver
from fs.agfs_adapter.agfs_context_fs import AGFSContextFS
from index.index_record_builder import build_index_records
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from scene.adapters import CanonicalSceneAdapter, _scope_context
from scene.candidate_index import LexicalSceneIndex
from scene.config import InMemorySceneConfigProvider
from session.archive_store import SessionArchiveStore
from tests.unit.fs.test_agfs_context_fs import _MemoryAGFSClient
from tests.unit.scene.test_adapters import (
    Embedder,
    IndexedVectors,
    Repository,
    adapter,
    node,
    scene,
)
from tests.unit.scene.test_candidate_index import MemoryIndex

CTX = _scope_context("acme", "user:alice")


def test_indexed_member_fills_available_slot():
    first, second = node("a"), node("b")
    source = adapter([first, second], [scene("release", [first, second])])
    source.vector_index.vectors.pop(first.uri)
    hits = source.search("release", CTX, limit=1, query_vector=[1, 0])
    assert [hit.uri for hit in source.resolve_members(hits, CTX, limit=1)] == [second.uri]
    source.vector_index.vectors[first.uri] = [1, 0]
    hits = source.search("release", CTX, limit=1, query_vector=[1, 0])
    assert [hit.uri for hit in source.resolve_members(hits, CTX, limit=2)] == [first.uri, second.uri]


def test_request_uses_pinned_route_configuration():
    item = node("a")
    provider = InMemorySceneConfigProvider()
    source = adapter([item], [scene("release", [item])], config_provider=provider)
    source.candidate_index = LexicalSceneIndex(MemoryIndex())
    source.candidate_index.publish(source.repository.snapshot)
    bound = source.for_request(CTX, [])
    provider.publish("acme", "user:alice", 0, {"lexical_enabled": True})
    assert bound.search("release", CTX, limit=1, query_vector=[1, 0])[0].routes == ("semantic",)
    assert source.for_request(CTX, []).search("release", CTX, limit=1, query_vector=[1, 0])[0].routes == (
        "semantic", "lexical",
    )
    provider.publish("acme", "user:alice", 1, {"lexical_enabled": False})
    assert source.for_request(CTX, []).search("release", CTX, limit=1, query_vector=[1, 0])[0].routes == (
        "semantic",
    )


@pytest.mark.parametrize("explicit_store", [True, False])
def test_bare_archive_provenance_reads_actual_store(explicit_store):
    fs = AGFSContextFS(_MemoryAGFSClient())
    archives = SessionArchiveStore(fs)
    for ctx, content in ((CTX, "Release Friday"), (replace(CTX, account_id="other"), "Other tenant"),
                         (replace(CTX, user_id="bob"), "Other owner")):
        saved = archives.write_archive(
            ctx.user_id, "Plan", "Plan", [{"id": "m1", "role": "user", "content": content}],
            ctx, archive_id="release",
        )
        assert saved.success
    source = CanonicalSceneAdapter(
        Repository(), fs, IndexedVectors([]), Embedder(),
        archive_store=archives if explicit_store else None,
    )
    ref = ProvenanceResolver.build_id("archive", "release", ["m1"])
    card = source.read_source_context("acme", "user:alice", [ref], 1)[0]
    assert card["message_ids"] == ["m1"]
    assert card["text"] == "[m1]  user: Release Friday\n"


def test_healthy_member_survives_broken_source():
    first, second = node("a"), node("b")
    client = _MemoryAGFSClient()
    fs = AGFSContextFS(client)
    for item in (first, second):
        fs.write_node(item, CTX)
    path = fs._uri_to_agfs_path(first.uri) + fs.FILE_META
    meta = json.loads(client.read(path))
    meta["status"] = "BROKEN"
    client.write(path, json.dumps(meta).encode())
    source = CanonicalSceneAdapter(
        Repository([scene("release", [first, second])]), fs, IndexedVectors([first, second]), Embedder(),
    )
    hits = source.search("release", CTX, limit=1, query_vector=[1, 0])
    assert [hit.uri for hit in source.resolve_members(hits, CTX, limit=1)] == [second.uri]
    batch = source.load_input("acme", "user:alice", node_uris=[second.uri])
    assert [item.uri for item in batch.nodes] == [second.uri]
    assert batch.snapshot.scenes[0].status == "ARCHIVED"


def test_pending_index_recovers_with_matching_text():
    item, current = node("a"), node("b")
    source = adapter([item, current], [scene("release", [item, current])])
    index = InMemoryVectorIndex(dimension=2)
    for original in (item, current):
        record = build_index_records(original)[0]
        record.metadata["_embedding"] = [1.0, 0.0]
        index.upsert([record])
    source.vector_index = index
    source.fs.nodes[item.uri].abstract = "New abstract"
    hits = source.search("release", CTX, limit=1, query_vector=[1, 0])
    assert [hit.uri for hit in source.resolve_members(hits, CTX, limit=1)] == [current.uri]
    record = build_index_records(source.fs.nodes[item.uri])[0]
    record.metadata["_embedding"] = [0.0, 1.0]
    index.upsert([record])
    batch = source.load_input("acme", "user:alice", node_uris=[item.uri])
    assert batch.nodes[0].abstract == record.text
    assert batch.embeddings[item.uri] == [0.0, 1.0]


def test_publication_retries_with_concurrent_source_revision():
    item = node("a")
    source = adapter([item])
    begin, done = Event(), Event()

    def writer():
        if begin.wait(5):
            source.fs.write_node(replace(item, content="Updated", metadata={"version": 2}), CTX)
            done.set()

    worker = Thread(target=writer)
    worker.start()
    checked = source.repository.compare_and_swap_checked

    def concurrent_publish(expected, replacement, check):
        begin.set()
        assert done.wait(5)
        return checked(expected, replacement, check)

    source.repository.compare_and_swap_checked = concurrent_publish
    try:
        for _ in range(2):
            batch = source.load_input("acme", "user:alice", node_uris=[item.uri])
            record = scene("release", batch.nodes)
            member = replace(record.cluster.members[0], node_version=batch.nodes[0].metadata["version"])
            record = replace(record, cluster=replace(record.cluster, members=(member,)))
            if source.compare_and_swap(batch.snapshot, replace(batch.snapshot, revision=1, scenes=(record,))):
                break
    finally:
        begin.set()
        worker.join(5)
    assert source.repository.snapshot.scenes[0].cluster.members[0].node_version == 2


def test_publication_protects_evidence_until_commit():
    item = node("a")
    source = adapter([item])
    batch = source.load_input("acme", "user:alice", node_uris=[item.uri])
    checked, attempted = Event(), Event()

    def writer():
        if checked.wait(5):
            attempted.set()
            source.fs.write_node(replace(item, content="Updated", metadata={"version": 2}), CTX)

    worker = Thread(target=writer)
    worker.start()
    commit = source.repository.compare_and_swap

    def commit_after_writer_attempt(expected, replacement):
        checked.set()
        assert attempted.wait(5)
        assert source.fs.read_node(item.uri, CTX).metadata["version"] == 1
        return commit(expected, replacement)

    source.repository.compare_and_swap = commit_after_writer_attempt
    try:
        assert source.compare_and_swap(batch.snapshot, replace(batch.snapshot, revision=1))
    finally:
        checked.set()
        worker.join(5)
    assert source.repository.snapshot.revision == 1
    assert source.fs.read_node(item.uri, CTX).metadata["version"] == 2
