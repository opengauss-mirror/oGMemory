from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import logging
import math
import threading

import pytest

from index.edge_projection import (
    AcceptedEdgeProjectionInput,
    EdgeProjectionWorker,
    InMemoryEdgeProjectionStore,
    build_searchable_text,
)
from index.embedding_components import EmbeddingComponentCache, EmbeddingComponentKey
from providers.relation_store.projection_identity import FACTORIZED_REPRESENTATION_VERSION
from providers.token_tracker import (
    UsageTracker,
    _embedding_inputs,
    distributed_usage_scope,
    embedding_scope_weights,
)


PROFILE = dict(profile_id="components-v1", model="test", dimension=2,
               representation_version=FACTORIZED_REPRESENTATION_VERSION)


def edge(name, *, owner="alice", text="Abstract\nOverview\nComplete target facts", relation="supports"):
    return AcceptedEdgeProjectionInput(name, f"ctx://acct/users/{owner}/memories/{name}",
        f"ctx://acct/users/{owner}/memories/target", relation, "canonical evidence",
        text, 1, 2, claim_token=f"claim-{name}")


class RecordingEmbedder:
    max_input_chars = 30000

    def __init__(self):
        self.calls = []
        self.scopes = []
        self.provider_usage = []
        self.tracker = UsageTracker()

    def embed_texts(self, texts):
        self.calls.append(list(texts))
        self.scopes.append(_embedding_inputs.get())
        tokens = len(texts) * 7
        self.provider_usage.append(tokens)
        self.tracker.record_embed(tokens)
        return [[0.0, 4.0] if text.startswith("Relationship:") else [3.0, 0.0] for text in texts]


def worker(embedder, cache, profile=PROFILE):
    return EdgeProjectionWorker(embedder, InMemoryEdgeProjectionStore(), profile,
                                component_cache=cache)


def key(text, **changes):
    values = dict(owner="ctx://acct/users/alice", profile_id="components-v1", model="test",
        dimension=2, representation=FACTORIZED_REPRESENTATION_VERSION, kind="target", text=text)
    values.update(changes)
    return EmbeddingComponentKey(**values)


def test_shared_full_target_uses_fixed_normalized_fusion_and_each_edge_metadata(caplog):
    embedder = RecordingEmbedder()
    first = edge("a")
    second = replace(edge("b", relation="explains"), source_revision=4, target_revision=8)
    cache = EmbeddingComponentCache()
    with caplog.at_level(logging.INFO, logger="index.edge_projection"):
        projected = worker(embedder, cache).project_accepted_sync([first, second])
    assert embedder.calls == [[first.target_text, "Relationship: supports", "Relationship: explains"]]
    expected = (.75 / math.sqrt(.75 ** 2 + .25 ** 2), .25 / math.sqrt(.75 ** 2 + .25 ** 2))
    assert projected[0].vector == pytest.approx(expected)
    assert projected[1].vector == pytest.approx(expected)
    assert [item.text for item in projected] == [build_searchable_text(first), build_searchable_text(second)]
    assert [(item.edge_id, item.source_revision, item.target_revision, item.claim_token) for item in projected] == [
        ("a", 1, 2, "claim-a"), ("b", 4, 8, "claim-b")]
    assert {item.representation_version for item in projected} == {FACTORIZED_REPRESENTATION_VERSION}
    assert cache.snapshot() == dict(hits=0, misses=3, waits=0, entries=3, inflight=0, capacity=1024)
    message = next(record.getMessage() for record in caplog.records
                   if record.getMessage().startswith("graph_component_cache_totals "))
    assert json.loads(message.removeprefix("graph_component_cache_totals ")) == cache.snapshot()


def test_empty_target_preserves_a_relationship_projection_and_current_metadata():
    embedder = RecordingEmbedder()
    item = edge("a", text="  \n ")
    projected = worker(embedder, EmbeddingComponentCache()).project_accepted_sync([item])[0]
    assert embedder.calls == [["Relationship: supports"]]
    assert projected.vector == (0.0, 1.0)
    assert projected.text == "Relationship: supports"
    assert (projected.edge_id, projected.source_revision, projected.target_revision, projected.claim_token) == (
        "a", 1, 2, "claim-a")


def test_service_cache_reuses_components_across_workers_and_updated_revisions():
    embedder = RecordingEmbedder()
    cache = EmbeddingComponentCache()
    first = edge("a")
    worker(embedder, cache).project_accepted_sync([first])
    revised = replace(first, target_revision=3, source_revision=5, claim_token="claim-next")
    second = worker(embedder, cache).project_accepted_sync([revised])[0]
    updated = replace(revised, target_text=first.target_text + "\nUpdated evidence", target_revision=4)
    third = worker(embedder, cache).project_accepted_sync([updated])[0]
    assert embedder.calls == [[first.target_text, "Relationship: supports"], [updated.target_text]]
    assert (second.source_revision, second.target_revision, second.claim_token) == (5, 3, "claim-next")
    assert third.target_revision == 4
    assert third.text == build_searchable_text(updated)


def test_each_owner_keeps_independent_components_for_identical_text():
    embedder = RecordingEmbedder()
    cache = EmbeddingComponentCache()
    alice, bob = edge("a"), edge("b", owner="bob")
    worker(embedder, cache).project_accepted_sync([alice])
    worker(embedder, cache).project_accepted_sync([bob])
    assert embedder.calls == [[alice.target_text, "Relationship: supports"],
                              [bob.target_text, "Relationship: supports"]]
    assert len(cache._vectors) == 4


def test_concurrent_workers_share_target_and_progress_with_distinct_relations():
    target_started = threading.Event()
    distinct_relation_done = threading.Event()
    release_target = threading.Event()
    call_lock = threading.Lock()
    calls = []

    class ConcurrentEmbedder:
        def embed_texts(self, texts):
            with call_lock:
                calls.extend(texts)
            if edge("a").target_text in texts:
                target_started.set()
                assert release_target.wait(5)
            else:
                distinct_relation_done.set()
            return [[0.0, 1.0] if text.startswith("Relationship:") else [1.0, 0.0] for text in texts]

    embedder = ConcurrentEmbedder()
    cache = EmbeddingComponentCache()
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(worker(embedder, cache).project_accepted_sync, [edge("a")])
        assert target_started.wait(5)
        second = executor.submit(worker(embedder, cache).project_accepted_sync, [edge("b", relation="explains")])
        try:
            assert distinct_relation_done.wait(5)
        finally:
            release_target.set()
        assert first.result(timeout=5)[0].edge_id == "a"
        assert second.result(timeout=5)[0].edge_id == "b"
    assert sorted(calls) == sorted([edge("a").target_text, "Relationship: supports", "Relationship: explains"])
    assert len(cache._vectors) == 3
    assert cache.snapshot() == dict(hits=0, misses=3, waits=1, entries=3, inflight=0, capacity=1024)


def test_lru_capacity_supports_continued_computation_and_recent_reuse():
    cache = EmbeddingComponentCache(max_entries=2)
    computed = []

    def compute(keys):
        computed.extend(item.text for item in keys)
        return [[1.0, 0.0] for _ in keys]

    for text in ("a", "b", "a", "c", "a", "b"):
        assert cache.resolve([key(text)], compute)[key(text)] == (1.0, 0.0)
    assert computed == ["a", "b", "c", "b"]
    assert len(cache._vectors) == 2
    assert cache.snapshot() == dict(hits=2, misses=4, waits=0, entries=2, inflight=0, capacity=2)


@pytest.mark.parametrize("change", [
    {"profile_id": "components-v2"}, {"model": "next-model"}, {"dimension": 3},
    {"representation": "next-representation"}, {"kind": "relation"},
    {"preprocessing": (("max_input_chars", "8000"),)},
])
def test_profile_and_component_identity_select_their_own_cached_vector(change):
    cache = EmbeddingComponentCache()
    original, updated = key("same text"), key("same text", **change)
    computed = []

    def compute(keys):
        computed.extend(keys)
        return [[1.0, 0.0] for _ in keys]

    cache.resolve([original], compute)
    cache.resolve([updated], compute)
    assert computed == [original, updated]
    assert len(cache._vectors) == 2


def test_complete_components_use_bounded_utf8_batches_and_preserve_large_input():
    embedder = RecordingEmbedder()
    inputs = [edge("a", text="甲" * 2000), edge("b", text="乙" * 2000), edge("c", text="丙" * 3000)]
    projected = worker(embedder, EmbeddingComponentCache()).project_accepted_sync(inputs)
    assert embedder.calls == [[inputs[0].target_text, "Relationship: supports"],
                              [inputs[1].target_text], [inputs[2].target_text]]
    assert [item.text for item in projected] == [build_searchable_text(item) for item in inputs]
    assert all(sum(len(text.encode("utf-8")) for text in batch) <= 8192 or len(batch) == 1
               for batch in embedder.calls)


def test_component_usage_maps_original_edges_with_component_byte_weights():
    embedder = RecordingEmbedder()
    cache = EmbeddingComponentCache()
    inputs = [edge("a", text="完整事实"), edge("b", text="完整事实", relation="explains"),
              edge("c", owner="bob", text="完整事实")]
    scopes = [(('acct', 'user:alice'), 100), (('acct', 'user:alice'), 100),
              (('acct', 'user:bob'), 100)]
    with distributed_usage_scope(embedding_scope_weights(scopes), inputs=scopes):
        projected = worker(embedder, cache).project_accepted_sync(inputs)
        worker(embedder, cache).project_accepted_sync(inputs)
    expected = [(scopes[index][0], len(text.encode("utf-8"))) for index, text in [
        (0, "完整事实"), (0, "Relationship: supports"), (1, "Relationship: explains"),
        (2, "完整事实"), (2, "Relationship: supports")]]
    assert embedder.scopes == [expected]
    assert embedder.provider_usage == [35]
    assert embedder.tracker.snapshot().embed_tokens == 35
    assert len(projected) == 3
