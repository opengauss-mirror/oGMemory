"""PR314: version protection and graph-independent occurrence replay."""
import hashlib
import json
from dataclasses import replace

import pytest

from core.candidate_dedup import occurrence_replay_fields
from index.activation_projection import ACTIVATION_VERSION
from tests.unit.commit.test_memory_identity import MemoryFS, DecisionLLM, api, memory, ctx


def descriptor(text):
    return dict(update_activation=text, update_activation_version=ACTIVATION_VERSION)


def test_stale_descriptor_update_cannot_overwrite_newer_node(ctx):
    fs = MemoryFS(); writer = api(fs, DecisionLLM())
    first = memory("topic", metadata=descriptor("original descriptor"))
    uri = writer.write_memory(first, ctx)["target_uri"]
    writer.write_memory(replace(first, target_uri=uri, expected_version=1,
        metadata=descriptor("current descriptor")), ctx)
    before = fs.read_node(uri, ctx)
    with pytest.raises(ValueError, match="changed since extraction"):
        writer.write_memory(replace(first, target_uri=uri, expected_version=1,
            metadata=descriptor("stale descriptor")), ctx)
    assert fs.read_node(uri, ctx) == before


@pytest.mark.parametrize("category,owner", [("event", "user"), ("case", "agent")])
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("initial_graph", [False, True])
def test_descriptor_refresh_keeps_occurrence_identity(ctx, category, owner, legacy, initial_graph):
    fs = MemoryFS(); writer = api(fs, DecisionLLM())
    first = replace(memory("repair", category=category, content="Nora repaired the receiver.",
        metadata=descriptor("original descriptor") if initial_graph else {}), owner_scope=owner)
    uri = writer.write_memory(first, ctx)["target_uri"]
    if legacy:
        fields = occurrence_replay_fields(first, include_graph_annotations=True)
        old_hash = hashlib.sha256(json.dumps(fields, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        fs.nodes[uri].metadata["occurrence_replay_sources"] = {old_hash: first.provenance_ids}
    refreshed = replace(first, metadata=descriptor("new descriptor"))
    result = writer.write_memory(refreshed, ctx)
    assert len(fs.nodes) == 1 and result["target_uri"] == uri
    stored = fs.read_node(uri, ctx)
    assert stored.content == first.content
    assert stored.metadata["update_activation"] == "new descriptor"
    assert stored.metadata["version"] == 2
    assert writer.write_memory(refreshed, ctx)["action"] == "skip"
    # Different source evidence or facts must still be separate occurrences.
    result = writer.write_memory(replace(refreshed, provenance_ids=["another-source"]), ctx)
    assert result["target_uri"] != uri and len(fs.nodes) == 2
