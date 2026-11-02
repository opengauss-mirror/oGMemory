"""Unit coverage for bounded request-local candidate discovery."""

from __future__ import annotations

import re
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from nltk.stem import PorterStemmer

from core.models import RequestContext
from retrieval import code_retrieval
from retrieval.code_retrieval import (
    ActiveCodeSearchRequest,
    CodeEvidence,
    CodeRetrievalService,
    CodeSearchBudget,
    CodeSearchScope,
)
from service.code_workspace_sync import ResolvedCodeWorkspace, WorkspaceResolutionError

# Keep candidate-local unit tests independent of optional NLTK corpora.  The
# shared BM25 index behavior remains covered by test_bm25.py; these tests
# exercise request-local discovery and ranking with deterministic tokenization.
_MOCK_STOPS = frozenset(
    "a an the and or but is are was were be been being have has had do does did "
    "will would could should may might shall can to of in for on with at by from "
    "as into through during before after above below between out off over under "
    "again further then once here there when where why how all both each few more "
    "most other some such no nor not only own same so than too very it its he she "
    "they them we you i me my his her our your this that these those what which who"
    .split()
)

_stemmer = PorterStemmer()


def _mock_tokenize(text: str) -> list[str]:
    """Simple regex tokenization + Porter stemming + stop filtering."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [_stemmer.stem(t) for t in tokens if t not in _MOCK_STOPS]


@pytest.fixture(autouse=True)
def _patch_nltk():
    """Avoid network-dependent NLTK data downloads in candidate tests."""
    with patch("retrieval.bm25_index._ensure_nltk"), \
         patch("retrieval.bm25_index._tokenize", _mock_tokenize):
        yield


def _workspace(root) -> ResolvedCodeWorkspace:
    return ResolvedCodeWorkspace(
        root_id="root-1",
        root_slug="repo",
        root_fingerprint="f" * 64,
        physical_root=root,
        max_files=300,
        language_hints={},
    )


def test_symbolic_hits_survive_a_full_bm25_semantic_shortlist():
    records = [{"uri": f"ctx://acct/resources/{i}", "content": "code"} for i in range(21)]
    corpus = SimpleNamespace(records=records, file_ranks={})
    service = CodeRetrievalService(config=SimpleNamespace(code_retrieval_candidate_semantic_limit=4))
    selected = service._candidate_semantic_records(corpus, {"bm25": records[:20], "symbolic": [records[-1]]})
    assert records[-1]["uri"] in {r["uri"] for r in selected}
    assert len(selected) == 4
    assert len({r["uri"] for r in selected}) == 4


def test_assignment_definitions_outrank_incidental_short_identifier_mentions(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "test_tree.py").write_text("a1 = Node()\na2 = Node()\na21 = Node(a1)\n")
    for i in range(30):
        (root / f"test_other_{i}.py").write_text(f"def test_other_{i}():\n    assert a1 != a2\n")
    service = CodeRetrievalService(
        workspace_resolver=SimpleNamespace(resolve=lambda **kwargs: _workspace(root)),
        config=SimpleNamespace(code_retrieval_enabled=True, code_retrieval_mode="candidate_local",
                               code_retrieval_route_enabled=lambda route: route == "symbolic"),
    )
    result = service.search_active(
        ActiveCodeSearchRequest("Definition of a1 a2 a21 in test_tree.py"),
        RequestContext("acct", "user", "agent", "session", "trace"),
    )
    assert result.evidence[0].relative_path == "test_tree.py"
    assert "a21 = Node(a1)" in result.evidence[0].excerpt


def test_candidate_active_search_is_request_local_and_never_touches_full_dependencies(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "target.py").write_text(
        "def parse_target(value):\n    return value\n",
        encoding="utf-8",
    )

    class Resolver:
        def resolve(self, *, ctx, root_slugs):
            assert ctx.account_id == "acct"
            assert root_slugs == ()
            return _workspace(root)

    class ExplodingDependency:
        def __getattr__(self, name):
            raise AssertionError(f"candidate search touched full dependency: {name}")

    class RecordingEmbedder:
        def __init__(self):
            self.calls = []

        def embed_texts(self, texts):
            self.calls.append(list(texts))
            return [[float(index + 1) for index in range(8)] for _ in texts]

        def embed_texts_with_timeout(self, texts, *, timeout_seconds):
            assert timeout_seconds > 0
            return self.embed_texts(texts)

    embedder = RecordingEmbedder()
    service = CodeRetrievalService(
        context_fs=ExplodingDependency(),
        read_api=ExplodingDependency(),
        bm25_index=ExplodingDependency(),
        relation_store=ExplodingDependency(),
        workspace_resolver=Resolver(),
        candidate_embedder=embedder,
        config=SimpleNamespace(
            code_retrieval_enabled=True,
            code_retrieval_mode="candidate_local",
            code_retrieval_route_enabled=lambda route: True,
        ),
    )
    ctx = RequestContext("acct", "user", "agent", "session", "trace")

    outcome = service.search_active(
        ActiveCodeSearchRequest(
            "parse_target",
            budget=CodeSearchBudget(candidate_limit=8, result_limit=8),
        ),
        ctx,
    )

    assert outcome.status == "ok"
    assert outcome.evidence
    assert outcome.evidence[0].origin == "candidate_local"
    assert outcome.evidence[0].persisted is False
    assert outcome.evidence[0].focused_read_available is False
    assert outcome.diagnostics.mode == "candidate_local"
    assert outcome.diagnostics.embedding["logical_inputs"] <= 17
    diagnostics = outcome.diagnostics.to_dict()
    assert diagnostics["evidence_chars"] == sum(len(item.excerpt) for item in outcome.evidence)
    assert diagnostics["line_span_count"] == sum(
        1
        for item in outcome.evidence
        if isinstance(item.start_line, int)
        and not isinstance(item.start_line, bool)
        and item.start_line > 0
        and isinstance(item.end_line, int)
        and not isinstance(item.end_line, bool)
        and item.end_line >= item.start_line
    )
    assert len(embedder.calls) == 2
    second_outcome = service.search_active(
        ActiveCodeSearchRequest(
            "parse_target",
            budget=CodeSearchBudget(candidate_limit=8, result_limit=8),
        ),
        ctx,
    )
    assert second_outcome.evidence
    assert len(embedder.calls) == 4
    with pytest.raises(FileNotFoundError):
        service.read_focused_snippet(uri=outcome.evidence[0].uri, ctx=ctx)


def test_candidate_long_class_excerpt_centers_grep_match_and_updates_locator(tmp_path):
    root = tmp_path / "repo"
    source = root / "pylint" / "checkers" / "variables.py"
    source.parent.mkdir(parents=True)
    filler = [f"    filler_{index:03d} = {index}" for index in range(140)]
    source_text = "\n".join(
        [
            "class VariablesChecker:",
            *filler,
            "    def visit_dictcomp(self, node):",
            "        self._check_dictcomp_target(node)",
        ]
    ) + "\n"
    source.write_text(source_text, encoding="utf-8")

    class Resolver:
        def resolve(self, *, ctx, root_slugs):
            assert ctx.account_id == "acct"
            assert root_slugs == ()
            return _workspace(root)

    class RecordingEmbedder:
        def embed_texts_with_timeout(self, texts, *, timeout_seconds):
            assert timeout_seconds > 0
            return [[float(index + 1) for index in range(8)] for _ in texts]

    service = CodeRetrievalService(
        workspace_resolver=Resolver(),
        candidate_embedder=RecordingEmbedder(),
        config=SimpleNamespace(
            code_retrieval_enabled=True,
            code_retrieval_mode="candidate_local",
            code_retrieval_route_enabled=lambda route: True,
        ),
    )
    outcome = service.search_active(
        ActiveCodeSearchRequest(
            "VariablesChecker behavior",
            scope=CodeSearchScope(grep_terms=("visit_dictcomp",)),
            budget=CodeSearchBudget(
                candidate_limit=8,
                result_limit=1,
                snippet_chars=220,
                batch_chars=220,
            ),
        ),
        RequestContext("acct", "user", "agent", "session", "trace"),
    )

    assert outcome.status == "ok"
    assert len(outcome.evidence) == 1
    evidence = outcome.evidence[0]
    target_start = source_text.splitlines().index("    def visit_dictcomp(self, node):") + 1
    target_end = target_start + 1
    assert "def visit_dictcomp(self, node):" in evidence.excerpt
    assert "self._check_dictcomp_target(node)" in evidence.excerpt
    assert "class VariablesChecker:" not in evidence.excerpt
    assert len(evidence.excerpt) <= 220
    assert evidence.start_line is not None and evidence.start_line <= target_start
    assert evidence.end_line is not None and evidence.end_line >= target_end
    assert evidence.locator == (
        f"repo/pylint/checkers/variables.py:{evidence.start_line}-{evidence.end_line}"
    )


def test_candidate_evidence_diagnostic_counts_reject_invalid_line_spans():
    service = CodeRetrievalService(config=SimpleNamespace())
    evidence = (
        CodeEvidence(
            uri="ctx://acct/resources/first",
            locator="repo:first.py",
            excerpt="abc",
            start_line=1,
            end_line=3,
        ),
        CodeEvidence(
            uri="ctx://acct/resources/zero",
            locator="repo:zero.py",
            excerpt="de",
            start_line=0,
            end_line=1,
        ),
        CodeEvidence(
            uri="ctx://acct/resources/reversed",
            locator="repo:reversed.py",
            excerpt="f",
            start_line=5,
            end_line=4,
        ),
        CodeEvidence(
            uri="ctx://acct/resources/bool",
            locator="repo:bool.py",
            excerpt="gh",
            start_line=True,
            end_line=1,
        ),
    )

    assert service._candidate_evidence_diagnostic_counts(evidence) == {
        "evidence_chars": 8,
        "line_span_count": 1,
    }


def test_candidate_active_executes_all_local_routes_and_marks_route_fill(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "target.py").write_text(
        "def target_helper(value):\n    return linked_helper(value)\n",
        encoding="utf-8",
    )
    (root / "helper.py").write_text(
        "def linked_helper(value):\n    return value + 1\n",
        encoding="utf-8",
    )

    class Resolver:
        def resolve(self, *, ctx, root_slugs):
            assert ctx.account_id == "acct"
            assert root_slugs == ()
            return _workspace(root)

    class ExplodingDependency:
        def __getattr__(self, name):
            raise AssertionError(f"candidate search touched full dependency: {name}")

    class RecordingEmbedder:
        def __init__(self):
            self.calls = []

        def embed_texts(self, texts):
            self.calls.append(list(texts))
            return [[1.0, 2.0, 3.0] for _ in texts]

        def embed_texts_with_timeout(self, texts, *, timeout_seconds):
            assert timeout_seconds > 0
            return self.embed_texts(texts)

    embedder = RecordingEmbedder()
    service = CodeRetrievalService(
        context_fs=ExplodingDependency(),
        read_api=ExplodingDependency(),
        bm25_index=ExplodingDependency(),
        relation_store=ExplodingDependency(),
        workspace_resolver=Resolver(),
        candidate_embedder=embedder,
        config=SimpleNamespace(
            code_retrieval_enabled=True,
            code_retrieval_mode="candidate_local",
            code_retrieval_candidate_limit=4,
            code_retrieval_result_limit=4,
            code_retrieval_route_enabled=lambda route: True,
        ),
    )
    original_fuse = code_retrieval.fuse_route_hits

    def underfilled_fuse(route_hits, **kwargs):
        return original_fuse(route_hits, **kwargs)[:1]

    monkeypatch.setattr(code_retrieval, "fuse_route_hits", underfilled_fuse)
    outcome = service.search_active(
        ActiveCodeSearchRequest(
            "target_helper linked_helper",
            budget=CodeSearchBudget(candidate_limit=4, result_limit=4),
        ),
        RequestContext("acct", "user", "agent", "session", "trace"),
    )

    assert outcome.status == "ok"
    assert outcome.evidence
    assert len(embedder.calls) == 2
    for route in ("bm25", "symbolic", "graph", "semantic"):
        assert outcome.diagnostics.routes[route]["state"] == "ok"
        assert outcome.diagnostics.routes[route]["count"] > 0
    assert outcome.diagnostics.fusion["route_fill_count"] > 0
    assert any(item.score_kind == "candidate_route_fill" for item in outcome.evidence)
    assert all(item.retrieval_source == "candidate_local" for item in outcome.evidence)


def test_candidate_literal_fallback_remains_transient_and_line_grounded(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "needle.py").write_text(
        "def only_literal():\n    return 'needle-token'\n",
        encoding="utf-8",
    )

    class Resolver:
        def resolve(self, *, ctx, root_slugs):
            assert ctx.account_id == "acct"
            assert root_slugs == ()
            return _workspace(root)

    class ExplodingDependency:
        def __getattr__(self, name):
            raise AssertionError(f"candidate search touched full dependency: {name}")

    service = CodeRetrievalService(
        context_fs=ExplodingDependency(),
        read_api=ExplodingDependency(),
        bm25_index=ExplodingDependency(),
        relation_store=ExplodingDependency(),
        workspace_resolver=Resolver(),
        config=SimpleNamespace(
            code_retrieval_enabled=True,
            code_retrieval_mode="candidate_local",
            code_retrieval_route_enabled=lambda route: False,
        ),
    )
    ctx = RequestContext("acct", "user", "agent", "session", "trace")
    outcome = service.search_active(
        ActiveCodeSearchRequest(
            "find literal evidence",
            scope=CodeSearchScope(grep_terms=("needle-token",)),
            budget=CodeSearchBudget(candidate_limit=4, result_limit=4),
        ),
        ctx,
    )

    assert outcome.status == "ok"
    assert outcome.evidence
    assert outcome.diagnostics.fusion["literal_fallback_count"] == len(outcome.evidence)
    evidence = outcome.evidence[0]
    assert evidence.score_kind == "candidate_literal_fallback"
    assert evidence.persisted is False
    assert "needle-token" in evidence.excerpt
    with pytest.raises(FileNotFoundError):
        service.read_focused_snippet(uri=evidence.uri, ctx=ctx)


def test_candidate_semantic_drops_an_entire_invalid_document_batch():
    class PartialBatchEmbedder:
        def __init__(self):
            self.calls = []

        def embed_texts(self, texts):
            self.calls.append(list(texts))
            if len(self.calls) == 1:
                return [[1.0, 0.0]]
            return [[1.0, 0.0], [float("nan"), 0.0]]

        def embed_texts_with_timeout(self, texts, *, timeout_seconds):
            assert timeout_seconds > 0
            return self.embed_texts(texts)

    embedder = PartialBatchEmbedder()
    service = CodeRetrievalService(
        config=SimpleNamespace(
            code_retrieval_candidate_semantic_limit=2,
            code_retrieval_candidate_embedding_batch_texts=2,
            code_retrieval_candidate_embedding_batch_tokens=8000,
            code_retrieval_candidate_embedding_chars=4096,
            code_retrieval_candidate_embedding_deadline_ms=330000,
        ),
        candidate_embedder=embedder,
    )
    records = (
        {
            "uri": "ctx://acct/resources/one",
            "content": "first document",
            "metadata": {"resource_type": "code", "parent_uri": None},
        },
        {
            "uri": "ctx://acct/resources/two",
            "content": "second document",
            "metadata": {"resource_type": "code", "parent_uri": None},
        },
    )

    hits, diagnostics = service._search_candidate_semantic(
        query="query",
        ctx=RequestContext("acct", "user", "agent", "session", "trace"),
        records=records,
        embedding_deadline=time.monotonic() + 5,
        request_deadline=time.monotonic() + 5,
    )

    assert hits == []
    assert diagnostics["state"] == "partial_failure"
    assert diagnostics["error_kind"] == "document_embedding_error"
    assert diagnostics["logical_inputs"] == 3
    assert len(embedder.calls) == 2


def test_candidate_semantic_deadline_reports_no_unsent_provider_call():
    class ExplodingEmbedder:
        def __init__(self):
            self.calls = 0

        def embed_texts(self, texts):
            del texts
            self.calls += 1
            raise AssertionError("expired semantic route must not call the provider")

        def embed_texts_with_timeout(self, texts, *, timeout_seconds):
            assert timeout_seconds > 0
            return self.embed_texts(texts)

    embedder = ExplodingEmbedder()
    service = CodeRetrievalService(
        config=SimpleNamespace(
            code_retrieval_candidate_semantic_limit=1,
            code_retrieval_candidate_embedding_batch_texts=8,
            code_retrieval_candidate_embedding_batch_tokens=8000,
            code_retrieval_candidate_embedding_chars=4096,
            code_retrieval_candidate_embedding_deadline_ms=330000,
        ),
        candidate_embedder=embedder,
    )
    records = (
        {
            "uri": "ctx://acct/resources/one",
            "content": "first document",
            "metadata": {"resource_type": "code", "parent_uri": None},
        },
    )

    hits, diagnostics = service._search_candidate_semantic(
        query="query",
        ctx=RequestContext("acct", "user", "agent", "session", "trace"),
        records=records,
        embedding_deadline=time.monotonic() - 1,
        request_deadline=time.monotonic() + 5,
    )

    assert hits == []
    assert diagnostics["state"] == "timeout"
    assert diagnostics["provider_calls"] == 0
    assert diagnostics["logical_inputs"] == 0
    assert embedder.calls == 0


def test_candidate_semantic_bounds_each_provider_call_by_remaining_deadline(monkeypatch):
    class Clock:
        now = 0.0

        def monotonic(self):
            return self.now

    class DeadlineAwareEmbedder:
        def __init__(self, clock):
            self.clock = clock
            self.timeouts = []

        def embed_texts_with_timeout(self, texts, *, timeout_seconds):
            self.timeouts.append(timeout_seconds)
            self.clock.now += 4.0
            return [[1.0, 0.0] for _ in texts]

    clock = Clock()
    embedder = DeadlineAwareEmbedder(clock)
    monkeypatch.setattr(code_retrieval.time, "monotonic", clock.monotonic)
    service = CodeRetrievalService(
        config=SimpleNamespace(
            code_retrieval_candidate_semantic_limit=2,
            code_retrieval_candidate_embedding_batch_texts=1,
            code_retrieval_candidate_embedding_batch_tokens=8000,
            code_retrieval_candidate_embedding_chars=4096,
            code_retrieval_candidate_embedding_deadline_ms=330000,
        ),
        candidate_embedder=embedder,
    )
    records = (
        {
            "uri": "ctx://acct/resources/one",
            "content": "first document",
            "metadata": {"resource_type": "code", "parent_uri": None},
        },
        {
            "uri": "ctx://acct/resources/two",
            "content": "second document",
            "metadata": {"resource_type": "code", "parent_uri": None},
        },
    )

    hits, diagnostics = service._search_candidate_semantic(
        query="query",
        ctx=RequestContext("acct", "user", "agent", "session", "trace"),
        records=records,
        embedding_deadline=10.0,
        request_deadline=12.0,
    )

    assert len(hits) == 2
    assert embedder.timeouts == [10.0, 6.0, 2.0]
    assert diagnostics["state"] == "partial_failure"
    assert diagnostics["error_kind"] == "embedding_timeout"
    assert diagnostics["provider_calls"] == 3
    assert diagnostics["batches"] == 2


def test_candidate_semantic_rejects_provider_without_deadline_capability_before_call():
    class UnboundedEmbedder:
        def __init__(self):
            self.calls = 0

        def embed_texts(self, texts):
            del texts
            self.calls += 1
            raise AssertionError("candidate must not call an unbounded provider")

    embedder = UnboundedEmbedder()
    service = CodeRetrievalService(
        config=SimpleNamespace(
            code_retrieval_candidate_semantic_limit=1,
            code_retrieval_candidate_embedding_batch_texts=8,
            code_retrieval_candidate_embedding_batch_tokens=8000,
            code_retrieval_candidate_embedding_chars=4096,
            code_retrieval_candidate_embedding_deadline_ms=330000,
        ),
        candidate_embedder=embedder,
    )
    records = (
        {
            "uri": "ctx://acct/resources/one",
            "content": "first document",
            "metadata": {"resource_type": "code", "parent_uri": None},
        },
    )

    hits, diagnostics = service._search_candidate_semantic(
        query="query",
        ctx=RequestContext("acct", "user", "agent", "session", "trace"),
        records=records,
        embedding_deadline=time.monotonic() + 5,
        request_deadline=time.monotonic() + 5,
    )

    assert hits == []
    assert diagnostics["state"] == "error"
    assert diagnostics["error_kind"] == "embedder_deadline_unsupported"
    assert diagnostics["logical_inputs"] == 0
    assert diagnostics["provider_calls"] == 0
    assert embedder.calls == 0


def test_candidate_workspace_error_projects_only_safe_root_cardinality():
    class Resolver:
        def resolve(self, *, ctx, root_slugs):
            del ctx, root_slugs
            raise WorkspaceResolutionError(
                "workspace_scope_ambiguous",
                available_root_count=2,
            )

    service = CodeRetrievalService(
        config=SimpleNamespace(
            code_retrieval_enabled=True,
            code_retrieval_mode="candidate_local",
            code_retrieval_route_enabled=lambda route: True,
        ),
        workspace_resolver=Resolver(),
    )

    outcome = service.search_active(
        ActiveCodeSearchRequest("find symbol"),
        RequestContext("acct", "user", "agent", "session", "trace"),
    )

    assert outcome.status == "error"
    assert outcome.error == "workspace_scope_ambiguous"
    assert outcome.diagnostics.workspace == {
        "state": "error",
        "selection": "sole_root",
        "available_root_count": 2,
    }


def test_candidate_discovery_deadline_starts_after_root_resolution(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    observed_deadlines: list[float] = []
    clock_values = iter((100.0, 112.0, 112.0, 112.0))

    def monotonic_clock() -> float:
        return next(clock_values, 112.0)

    class Resolver:
        def resolve(self, *, ctx, root_slugs):
            del ctx, root_slugs
            return _workspace(root)

    def timeout_discovery(**kwargs):
        observed_deadlines.append(kwargs["budget"].deadline)
        raise TimeoutError

    monkeypatch.setattr(code_retrieval.time, "monotonic", monotonic_clock)
    monkeypatch.setattr(code_retrieval, "discover_candidate_files", timeout_discovery)
    service = CodeRetrievalService(
        config=SimpleNamespace(
            code_retrieval_enabled=True,
            code_retrieval_mode="candidate_local",
            code_retrieval_candidate_discovery_deadline_ms=15_000,
            code_retrieval_candidate_request_deadline_ms=360_000,
            code_retrieval_route_enabled=lambda route: True,
        ),
        workspace_resolver=Resolver(),
    )

    outcome = service.search_active(
        ActiveCodeSearchRequest("find symbol"),
        RequestContext("acct", "user", "agent", "session", "trace"),
    )

    assert outcome.status == "partial_failure"
    assert observed_deadlines == [127.0]


def test_candidate_embedding_deadline_starts_when_semantic_stage_begins():
    service = CodeRetrievalService(
        config=SimpleNamespace(
            code_retrieval_candidate_embedding_deadline_ms=330_000,
        ),
    )

    # Root resolution/discovery may have consumed twelve seconds since the
    # request began at t=100. The direct embedding stage still receives its
    # configured 330-second allowance, subject to the 360-second request cap.
    assert service._candidate_embedding_deadline(
        stage_started=112.0,
        request_deadline=460.0,
    ) == 442.0
