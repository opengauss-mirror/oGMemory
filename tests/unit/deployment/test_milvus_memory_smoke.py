"""The Milvus acceptance must prove extracted entity recall, not archive echoes."""

import pytest

from scripts.milvus_memory_smoke import wait_for_atomic_recall


def test_archive_or_summary_only_recall_is_rejected():
    def post(route, payload):
        assert route == "memories/query"
        return {
            "hits": [
                {"uri": "ogmemory://memories/session_archive/raw_turn"},
                {"uri": "ogmemory://memories/session_summary/summary"},
            ]
        }

    with pytest.raises(RuntimeError, match="persisted entity"):
        wait_for_atomic_recall(post, "marker", timeout=0.01, interval=0)


def test_entity_requires_expected_persisted_project_facts():
    calls = []

    def post(route, payload):
        calls.append((route, payload))
        if route == "memories/query":
            assert payload["types"] == ["entity"]
            return {"hits": [{"uri": "ogmemory://memories/entity/project_marker"}]}
        return {"content": "project_marker 使用 Python"}

    assert wait_for_atomic_recall(post, "PROJECT_MARKER", timeout=0.1, interval=0).endswith(
        "project_marker"
    )
    assert calls[0][1]["sessionId"] == calls[1][1]["sessionId"]


def test_entity_missing_language_fact_is_rejected():
    def post(route, payload):
        if route == "memories/query":
            return {"hits": [{"uri": "ogmemory://memories/entity/project_marker"}]}
        return {"content": "project_marker 使用 Java"}

    with pytest.raises(RuntimeError):
        wait_for_atomic_recall(post, "project_marker", timeout=0.01, interval=0)
