"""Moving a memory while its latest projection is embedding."""

from dataclasses import replace
from unittest.mock import Mock

from index.index_record_builder import build_index_records
from index.outbox_worker import OutboxWorker, create_upsert_event
from providers.embedder.mock_embedder import MockEmbedder
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from tests.unit.server.test_scene_runtime import CanonicalStore, Outbox, _node


def test_move_during_embedding_cleans_every_generated_record():
    fs, vector = CanonicalStore(), InMemoryVectorIndex(dimension=8)
    current = _node()
    fs.nodes[current.uri] = current
    event = create_upsert_event(current.uri, build_index_records(replace(current, overview="", content="")))
    outbox = Outbox(event)
    outbox.mark_done = lambda event, node_uri: outbox.acknowledged.append(event.event_id)
    embedder = MockEmbedder(dimension=8)
    embed = embedder.embed_texts

    def move_and_embed(texts):
        outbox.events.pop(event.event_id)
        fs.nodes.pop(current.uri)
        vector.delete([r.id for r in build_index_records(current)])
        return embed(texts)

    embedder.embed_texts = move_and_embed
    worker = OutboxWorker(vector, embedder, fs=fs, memory_projection_handler=Mock())
    stats = {"succeeded": 0, "failed": 0, "skipped": 0}
    worker._process_claimed_event(current.uri, event, outbox, stats)
    hits = vector.search_by_vector([1.0] * 8, {"account_id": "acme"}, top_k=20)
    assert hits == []
