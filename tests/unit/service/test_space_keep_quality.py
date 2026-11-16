"""Current quality policy applied to retained team memories."""
import json
from copy import deepcopy

from server.space_rules import validate_rule
from service.space_lifecycle import generate_plans


class Model:
    def __init__(self, operation):
        self.operation = operation

    def complete_json(self, prompt, schema):
        return {"operations": [self.operation]}


def run_case(count, score, action="KEEP"):
    sources = [{"uri": f"source:{i}", "content": "Inspect connection pool waits.",
                "grant_rev": 1, "content_rev": 1, "sha256": "hash", "epoch": "epoch"}
               for i in range(2)]
    contributions = [{"source_uri": s["uri"], "grant_rev": 1,
                      "source_content_rev": 1, "source_hash": "hash", "source_epoch": "epoch"}
                     for s in sources[:count]]
    spec = validate_rule({"description": "Reusable troubleshooting", "memory_types": ["case"],
                          "quality_policy": {"min_sources": 2, "min_score": 0.8}})
    snapshot = {"rule": {"spec": spec}, "sources": sources,
                "items": [{"item_id": "memory-1", "quality_score": score, "sources": contributions}]}
    original = deepcopy(snapshot)
    operation = {"action": action, "item_id": "memory-1"}
    if action == "UPDATE":
        operation.update(source_uris=[s["uri"] for s in sources], quality_score=0.9,
                         topic="pool", content="Inspect waits and restore capacity.", abstract="Pool recovery")
    result = generate_plans(Model(operation), snapshot)
    assert snapshot == original
    print(json.dumps({"request": snapshot, "model_response": operation, "response": result}))
    return result


def test_current_source_threshold_archives_old_item():
    assert run_case(1, 0.9) == [{"action": "ARCHIVE", "item_id": "memory-1"}]


def test_current_score_threshold_archives_old_item():
    assert run_case(2, 0.7) == [{"action": "ARCHIVE", "item_id": "memory-1"}]


def test_current_quality_boundary_retains_item():
    assert run_case(2, 0.8) == [{"action": "KEEP", "item_id": "memory-1"}]


def test_recomputed_item_uses_current_evidence_and_score():
    assert run_case(1, 0.7, "UPDATE")[0]["action"] == "UPDATE"
