"""Moving a workspace must change resource identity, not search selection."""
from types import SimpleNamespace

import pytest

from core.models import RequestContext
from core.resource_identity import compute_root_fingerprint
from retrieval.code_retrieval import (
    ActiveCodeSearchRequest, CodeRetrievalService, CodeSearchBudget,
)
from service.code_workspace_sync import ResolvedCodeWorkspace
from tests.code_search_helpers import deterministic_code_tokenizer  # noqa: F401


CTX = RequestContext("acct", "user", "agent", "session", "trace")


@pytest.mark.parametrize("fusion_mode", ["weighted_rrf", "weighted_sum"])
@pytest.mark.parametrize("literal_scope", [False, True])
def test_candidate_search_relocation_preserves_evidence_and_embedding_inputs(tmp_path, fusion_mode, literal_scope):
    results, inputs = [], []
    for location in ("first", "second", "third"):
        root = tmp_path / location
        root.mkdir()
        # More equally relevant snippets than either candidate cap.
        (root / "source.py").write_text("\n".join(
            f"def needle_{i}(needle):\n    # {'unique_marker' if i == 19 else 'padding'}\n    return needle_{(i + 1) % 20}()\n" for i in range(20)
        ))
        calls = []

        def embed(texts, *, timeout_seconds):
            calls.extend(texts)
            return [[1.0, 0.0] for _ in texts]

        workspace = ResolvedCodeWorkspace(location, "repo", compute_root_fingerprint(root), root, 100, {})
        service = CodeRetrievalService(
            config=SimpleNamespace(code_retrieval_enabled=True, code_retrieval_mode="candidate_local",
                                   code_retrieval_route_enabled=lambda route: True,
                                   code_retrieval_fusion_mode=fusion_mode,
                                   code_retrieval_candidate_snippet_limit=7,
                                   code_retrieval_candidate_semantic_limit=3),
            workspace_resolver=SimpleNamespace(resolve=lambda **kw: workspace),
            candidate_embedder=SimpleNamespace(embed_texts_with_timeout=embed),
        )
        result = service.search_active(ActiveCodeSearchRequest(
            "needle", scope={"grep_terms": ["unique_marker"]} if literal_scope else {},
            budget=CodeSearchBudget(candidate_limit=4, result_limit=3, graph_neighbor_limit=2),
        ), CTX)
        assert result.status == "ok" and result.evidence
        assert calls
        if literal_scope:
            assert any("unique_marker" in text for text in calls)
        inputs.append(calls)
        results.append(result)
    assert inputs[0] == inputs[1] == inputs[2]
    def evidence(result):
        return [(e.relative_path, e.start_line, e.end_line, e.excerpt, e.fusion_score, e.route_ranks, e.route_scores, e.graph_context) for e in result.evidence]
    assert evidence(results[0]) == evidence(results[1]) == evidence(results[2])
    uris = [{e.uri for e in result.evidence} for result in results]
    assert all(uris[i].isdisjoint(uris[j]) for i in range(3) for j in range(i))
