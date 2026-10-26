"""Stable source and vector projection across process restarts."""

import json
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest

from scene.decision_sources import node_source_refs
from tests.unit.scene.test_decision_evidence import node


@pytest.mark.parametrize("field", ["provenance_ids", "source_refs", "source_ref"])
def test_sparse_vector_and_set_source_projection_replays_across_hash_seeds(field):
    program = dedent("""
        import json
        import sys
        from dataclasses import asdict
        from scene.decision_application import SceneDecisionApplier
        from scene.decision_evidence import SceneEvidenceBuilder
        from scene.models import SceneCluster, SceneMember, SceneRecord, SceneSnapshot
        from tests.unit.scene.test_adapters import adapter
        from tests.unit.scene.test_decision_application import application_case
        from tests.unit.scene.test_decision_evidence import node

        snapshot, _, _, decision, _ = application_case()
        nodes = [node(f"{index:02d}") for index in range(18)]
        members = [nodes[index].uri for index in (1, 9, 17)]
        decision = decision.model_copy(update={
            "draft": decision.draft.model_copy(update={"member_uris": members}),
            "evidence_uris": members,
            "unassigned_member_uris": [item.uri for item in nodes if item.uri not in members],
        })
        nodes[1].metadata = {sys.argv[1]: {f"ref-{index}" for index in range(6)}}
        source = adapter(nodes)
        source.vector_index.vectors = {
            item.uri: [1.0, index * 0.013, index * 0.027] for index, item in enumerate(nodes)
        }
        saved = json.load(sys.stdin)
        if saved:
            records = []
            for record in saved.pop("scenes"):
                cluster = record.pop("cluster")
                cluster["members"] = tuple(
                    SceneMember(**{**member, "source_refs": tuple(member["source_refs"])})
                    for member in cluster["members"]
                )
                for key in ("centroid", "source_archive_ids"):
                    cluster[key] = tuple(cluster[key])
                cluster["boundary"] = tuple(tuple(item) for item in cluster["boundary"])
                records.append(SceneRecord(**record, cluster=SceneCluster(**cluster)))
            snapshot = SceneSnapshot(**saved, scenes=tuple(records))
        source.repository.snapshot = snapshot
        batch = source.load_input("acct", "user:alice", node_uris=[item.uri for item in nodes])
        result = SceneDecisionApplier().apply(batch.snapshot, list(batch.nodes), batch.embeddings, decision)
        if saved:
            assert result is batch.snapshot
        else:
            assert source.compare_and_swap(batch.snapshot, result)
        print(json.dumps({
            "snapshot": asdict(source.repository.load_snapshot("acct", "user:alice")),
            "card": SceneEvidenceBuilder().node_card(nodes[1])["source_refs"],
        }))
    """)
    checkpoint = None
    outputs = []
    for seed in (1, 2, 42):
        result = subprocess.run(
            [sys.executable, "-c", program, field], input=json.dumps(checkpoint),
            capture_output=True, text=True, timeout=30,
            cwd=Path(__file__).resolve().parents[3], env={**os.environ, "PYTHONHASHSEED": str(seed)},
        )
        assert result.returncode == 0, result.stderr
        outputs.append(json.loads(result.stdout))
        checkpoint = outputs[0]["snapshot"]
    assert outputs[0] == outputs[1] == outputs[2]
    expected = [f"ref-{index}" for index in range(6)]
    assert checkpoint["scenes"][0]["cluster"]["members"][0]["source_refs"] == expected
    assert outputs[0]["card"] == expected[:4]


@pytest.mark.parametrize("ordered", [list, tuple])
def test_ordered_sources_keep_priority_while_sets_are_normalized(ordered):
    memory = node(0)
    memory.metadata = {
        "provenance_ids": ordered(["priority-z", "priority-a", "priority-z"]),
        "source_refs": {"ref-z", "ref-a", "priority-a", "", "  ", 2, "2"},
        "source_ref": ordered(["tail-z", "tail-a", "ref-a"]),
    }
    assert node_source_refs(memory) == (
        "priority-z", "priority-a", "2", "ref-a", "ref-z", "tail-z", "tail-a",
    )
