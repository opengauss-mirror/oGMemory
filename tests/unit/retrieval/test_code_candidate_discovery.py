"""Unit coverage for bounded request-local candidate discovery primitives."""

from __future__ import annotations

import re
import time
from types import SimpleNamespace

import pytest

from retrieval import code_candidate_local
from retrieval.code_candidate_local import (
    CandidateDiscoveryBudget,
    CandidateDiscoveryOutcome,
    CandidateFile,
    CandidateReadMeter,
    build_candidate_corpus,
    discover_candidate_files,
)
from service.code_workspace_sync import ResolvedCodeWorkspace


def _workspace(root) -> ResolvedCodeWorkspace:
    return ResolvedCodeWorkspace(
        root_id="root-1",
        root_slug="repo",
        root_fingerprint="f" * 64,
        physical_root=root,
        max_files=300,
        language_hints={},
    )


def test_file_and_predicate_is_ranked_before_selected_file_cap(tmp_path, monkeypatch):
    for i in range(3):
        (tmp_path / f"target_{i}.py").write_text("first = 1\n" * 10)
    (tmp_path / "z.py").write_text("first = second = 1\n")
    monkeypatch.setattr(code_candidate_local, "_run_rg_argv", lambda *a, **k: pytest.fail("AND scope needs only one full-source scan"))
    outcome = discover_candidate_files(
        workspace=_workspace(tmp_path),
        scope=SimpleNamespace(paths=(), glob_patterns=(), grep_terms=("first", "second")),
        query_terms=("target",),
        budget=CandidateDiscoveryBudget(4, 1, 1024 * 1024, time.monotonic() + 5),
    )
    assert [f.relative_path for f in outcome.files] == ["z.py"]
    assert outcome.diagnostics["files_matching_all_grep_terms"] == 1
    assert outcome.diagnostics["file_limit_reached"] is True
    assert outcome.diagnostics["read_bytes"] <= 1024 * 1024


@pytest.mark.parametrize("read_passes,expected_files", [(1, 0), (2, 4)])
def test_file_scope_leaves_canonical_read_budget(tmp_path, monkeypatch, read_passes, expected_files):
    from retrieval.code_retrieval import CodeSearchScope, CodeRetrievalService

    size = 8192
    head = 'def target():\n    return "needle"\n#'
    for i in range(4):
        (tmp_path / f"module_{i}.py").write_text(head + "x" * (size - len(head)), encoding="utf-8")
    monkeypatch.setattr(code_candidate_local, "_run_rg_argv", lambda *a, **k: pytest.fail("Redundant ripgrep scan"))
    scope = CodeSearchScope(grep_terms=("target", "needle"))
    budget = CandidateDiscoveryBudget(4, 4, 4 * size * read_passes, time.monotonic() + 10)
    discovery = discover_candidate_files(workspace=_workspace(tmp_path), scope=scope,
        query_terms=("target",), budget=budget)
    assert discovery.read_meter.used == 4 * size
    assert discovery.diagnostics["files_matching_all_grep_terms"] == 4
    assert discovery.diagnostics["backend"] == "literal_scan"
    corpus = build_candidate_corpus(workspace=_workspace(tmp_path), discovery=discovery,
        account_id="acct", scope=scope, query_terms=("target",), snippet_limit=10,
        deadline=budget.deadline, record_matches_scope=CodeRetrievalService._record_matches_scope)
    assert corpus.diagnostics["files_read"] == expected_files
    assert len(corpus.records) == expected_files
    assert discovery.read_meter.used == budget.read_bytes
    assert corpus.diagnostics["read_limit_reached"] is (read_passes == 1)


@pytest.mark.parametrize("paths,globs,expected", [
    (("z/target.py",), (), {"z/target.py"}),
    (("z",), (), {"z/target.py", "z/other.py"}),
    ((), ("z/*.py",), {"z/target.py", "z/other.py"}),
    (("z",), ("*/target.py",), {"z/target.py"}),
    (("z/target.py", "a.py"), ("z/*.py",), {"z/target.py"}),
])
def test_scoped_files_keep_read_budget_with_outside_files(tmp_path, paths, globs, expected):
    from retrieval.code_retrieval import CodeSearchScope, CodeRetrievalService

    source = 'def target():\n    return "needle"\n'
    for name in ("a.py", "a/target.py", "z/other.py", "z/target.py"):
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(source, encoding="utf-8")
    scope = CodeSearchScope(paths=paths, glob_patterns=globs, grep_terms=("needle",))
    budget = CandidateDiscoveryBudget(10, 10, 2 * len(source) * len(expected), time.monotonic() + 10)
    discovery = discover_candidate_files(workspace=_workspace(tmp_path), scope=scope,
        query_terms=("target",), budget=budget)
    assert {f.relative_path for f in discovery.files} == expected
    assert discovery.read_meter.used == len(source) * len(expected)
    corpus = build_candidate_corpus(workspace=_workspace(tmp_path), discovery=discovery,
        account_id="acct", scope=scope, query_terms=("target",), snippet_limit=10,
        deadline=budget.deadline, record_matches_scope=CodeRetrievalService._record_matches_scope)
    assert len(corpus.records) == len(expected)
    assert discovery.read_meter.used == budget.read_bytes
    assert not corpus.partial_failure


@pytest.mark.parametrize("scope_override,expected", [
    ({}, True),
    ({"grep_terms": ("def atoms", "Atom")}, True),
    ({"grep_terms": ("def atoms", "absent_source_term")}, False),
    ({"root_slugs": ("other",)}, False),
    ({"paths": ("other",)}, False),
    ({"glob_patterns": ("other/*.py",)}, False),
])
def test_file_grep_scope_survives_method_chunking(tmp_path, scope_override, expected):
    from retrieval.code_retrieval import CodeSearchScope, CodeRetrievalService

    source = tmp_path / "basic.py"
    source.write_text(
        "class Basic:\n"
        "    def atoms(self):\n        return {Atom()}\n" +
        "".join(f"    def method_{i}(self):\n        return self.args\n" for i in range(180)),
        encoding="utf-8",
    )
    # Terms in a different file must never satisfy this file's AND predicate.
    (tmp_path / "other.py").write_text("absent_source_term = 1\n")
    scope = CodeSearchScope(**{
        "root_slugs": ("repo",), "paths": ("basic.py",),
        "grep_terms": ("def atoms", "Atom", "args"), **scope_override,
    })
    discovery = CandidateDiscoveryOutcome(
        files=(CandidateFile("basic.py", 0, 0, 0, ("walker",)),),
        paths_by_relative={"basic.py": source}, literal_matches={},
        read_meter=CandidateReadMeter(cap=1024 * 1024), diagnostics={},
    )
    corpus = build_candidate_corpus(
        workspace=_workspace(tmp_path), discovery=discovery, account_id="acct",
        scope=scope, query_terms=("atoms",), snippet_limit=500,
        deadline=time.monotonic() + 10,
        record_matches_scope=CodeRetrievalService._record_matches_scope,
    )
    targets = [r for r in corpus.records if r["metadata"]["symbol"] == "Basic.atoms"]
    assert bool(targets) is expected
    if expected:
        assert "args" not in targets[0]["content"]
        assert CodeRetrievalService._record_matches_scope(targets[0], scope)
        assert all("_file_grep_terms" not in node.metadata for node in corpus.nodes_by_uri.values())
        if scope.grep_terms == ("def atoms", "Atom"):
            assert not any(r["metadata"]["symbol"].startswith("Basic.method_") for r in corpus.records)


@pytest.fixture(autouse=True)
def _patch_bm25_tokenizer(monkeypatch):
    """Keep candidate unit tests independent of unavailable NLTK corpora."""

    monkeypatch.setattr(
        "retrieval.bm25_index._tokenize",
        lambda text: re.findall(r"[a-z0-9]+", text.lower()),
    )


def test_glob_scope_is_seeded_before_generic_pool_cap(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "aaa.py").write_text("value = 1\n", encoding="utf-8")
    target_dir = root / "z-target"
    target_dir.mkdir()
    (target_dir / "match.py").write_text("value = 2\n", encoding="utf-8")

    outcome = discover_candidate_files(
        workspace=_workspace(root),
        scope=SimpleNamespace(paths=(), glob_patterns=("z-target/*.py",), grep_terms=()),
        query_terms=(),
        budget=CandidateDiscoveryBudget(
            pool_limit=1,
            file_limit=1,
            read_bytes=1024 * 1024,
            deadline=time.monotonic() + 5,
        ),
    )

    assert [candidate.relative_path for candidate in outcome.files] == ["z-target/match.py"]
    assert outcome.files[0].glob_anchor_match is True
    assert "scope_glob" in outcome.files[0].discovery_sources


def test_literal_content_search_is_bounded_by_the_merged_candidate_pool(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "aaa.py").write_text("value = 1\n", encoding="utf-8")
    (root / "z.py").write_text("needle = True\n", encoding="utf-8")
    observed_argv = []

    def run_rg(argv, **kwargs):
        del kwargs
        observed_argv.extend(argv)
        return "rg_ok", (("z.py", 1),)

    monkeypatch.setattr(code_candidate_local, "_run_rg_argv", run_rg)

    outcome = discover_candidate_files(
        workspace=_workspace(root),
        scope=SimpleNamespace(paths=(), glob_patterns=(), grep_terms=()),
        query_terms=("needle",),
        budget=CandidateDiscoveryBudget(
            pool_limit=1,
            file_limit=1,
            read_bytes=1024 * 1024,
            deadline=time.monotonic() + 5,
        ),
    )

    assert [candidate.relative_path for candidate in outcome.files] == ["aaa.py"]
    assert outcome.diagnostics["content_universe_count"] == 1
    assert "aaa.py" in observed_argv
    assert "z.py" not in observed_argv


def test_rg_and_canonical_corpus_each_charge_the_shared_read_budget(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "target.py"
    source.write_text("needle = True\n", encoding="utf-8")
    monkeypatch.setattr(
        code_candidate_local,
        "_run_rg_argv",
        lambda argv, **kwargs: ("rg_ok", (("target.py", 1),)),
    )
    one_read_budget = CandidateDiscoveryBudget(
        pool_limit=1,
        file_limit=1,
        read_bytes=source.stat().st_size,
        deadline=time.monotonic() + 5,
    )
    discovery = discover_candidate_files(
        workspace=_workspace(root),
        scope=SimpleNamespace(paths=(), glob_patterns=(), grep_terms=()),
        query_terms=("needle",),
        budget=one_read_budget,
    )

    corpus = build_candidate_corpus(
        workspace=_workspace(root),
        discovery=discovery,
        account_id="acct",
        scope=SimpleNamespace(grep_terms=()),
        query_terms=("needle",),
        snippet_limit=1,
        deadline=time.monotonic() + 5,
        record_matches_scope=lambda record, scope: True,
    )

    # Ripgrep has already consumed the whole one-file budget.  The canonical
    # builder would read the same file a second time, so it must not proceed
    # under a budget that covers only the discovery scan.
    assert corpus.records == ()
    assert corpus.partial_failure is True
    assert corpus.diagnostics["read_bytes"] == source.stat().st_size

    two_read_budget = CandidateDiscoveryBudget(
        pool_limit=1,
        file_limit=1,
        read_bytes=source.stat().st_size * 2,
        deadline=time.monotonic() + 5,
    )
    discovery = discover_candidate_files(
        workspace=_workspace(root),
        scope=SimpleNamespace(paths=(), glob_patterns=(), grep_terms=()),
        query_terms=("needle",),
        budget=two_read_budget,
    )
    corpus = build_candidate_corpus(
        workspace=_workspace(root),
        discovery=discovery,
        account_id="acct",
        scope=SimpleNamespace(grep_terms=()),
        query_terms=("needle",),
        snippet_limit=1,
        deadline=time.monotonic() + 5,
        record_matches_scope=lambda record, scope: True,
    )

    assert corpus.records
    assert corpus.diagnostics["read_bytes"] == source.stat().st_size * 2


def test_corpus_revalidates_a_replaced_candidate_before_canonical_build(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "target.py"
    source.write_text("value = 1\n", encoding="utf-8")
    discovery = CandidateDiscoveryOutcome(
        files=(CandidateFile("target.py", 0, 0, 0, ("walker",)),),
        paths_by_relative={"target.py": source},
        literal_matches={},
        read_meter=CandidateReadMeter(cap=1024 * 1024),
        diagnostics={},
    )
    outside = tmp_path / "outside.py"
    outside.write_text("secret = True\n", encoding="utf-8")
    source.unlink()
    source.symlink_to(outside)

    corpus = build_candidate_corpus(
        workspace=_workspace(root),
        discovery=discovery,
        account_id="acct",
        scope=SimpleNamespace(grep_terms=()),
        query_terms=(),
        snippet_limit=1,
        deadline=time.monotonic() + 5,
        record_matches_scope=lambda record, scope: True,
    )

    assert corpus.records == ()
    assert corpus.partial_failure is True
    assert corpus.diagnostics["files_failed"] == 1


def test_literal_fallback_revalidates_before_opening_a_replaced_candidate(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "target.py"
    source.write_text("needle = True\n", encoding="utf-8")
    outside = tmp_path / "outside.py"
    outside.write_text("needle = 'outside'\n", encoding="utf-8")

    def unavailable_rg(**kwargs):
        del kwargs
        source.unlink()
        source.symlink_to(outside)
        return "rg_unavailable", {}

    monkeypatch.setattr(code_candidate_local, "_ripgrep_literal_candidates", unavailable_rg)
    monkeypatch.setattr(
        code_candidate_local,
        "_read_literal_source",
        lambda *args, **kwargs: pytest.fail("fallback must not open the replaced symlink"),
    )

    outcome = discover_candidate_files(
        workspace=_workspace(root),
        scope=SimpleNamespace(paths=(), glob_patterns=(), grep_terms=()),
        query_terms=("needle",),
        budget=CandidateDiscoveryBudget(
            pool_limit=1,
            file_limit=1,
            read_bytes=1024 * 1024,
            deadline=time.monotonic() + 5,
        ),
    )

    assert outcome.literal_matches == {}
    assert outcome.diagnostics["fallback"] is True
    assert outcome.diagnostics["validation_failures"] >= 1


def test_corpus_clip_keeps_line_grounded_literal_chunk_when_query_differs(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    ordinary = root / "a.py"
    ordinary.write_text("def target_symbol():\n    return 1\n", encoding="utf-8")
    literal = root / "z.py"
    literal.write_text("needle = True\n", encoding="utf-8")
    discovery = CandidateDiscoveryOutcome(
        files=(
            CandidateFile("a.py", 0, 0, 0, ("walker",)),
            CandidateFile("z.py", 0, 0, 1, ("literal_scan",)),
        ),
        paths_by_relative={"a.py": ordinary, "z.py": literal},
        literal_matches={"z.py": (1,)},
        read_meter=CandidateReadMeter(cap=1024 * 1024),
        diagnostics={},
    )

    corpus = build_candidate_corpus(
        workspace=_workspace(root),
        discovery=discovery,
        account_id="acct",
        # The real scope matcher independently enforces the grep predicate.
        # This permissive double isolates the corpus clipping tie-break.
        scope=SimpleNamespace(grep_terms=("needle",)),
        query_terms=("target_symbol",),
        snippet_limit=1,
        deadline=time.monotonic() + 5,
        record_matches_scope=lambda record, scope: True,
    )

    assert [record["metadata"]["relative_path"] for record in corpus.records] == ["z.py"]
    assert len(corpus.literal_uris) == 1
