"""Regressions for full-index file scopes and query-focused source excerpts."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from tests.code_search_helpers import deterministic_code_tokenizer  # noqa: F401

from core.models import RequestContext
from retrieval.code_retrieval import (
    ActiveCodeSearchRequest, CodeRetrievalService, CodeSearchScope,
    _code_excerpt,
)
from service.code_resource_lifecycle import build_code_resource_nodes
from service.code_workspace_sync import ResolvedCodeWorkspace


def _source_service(path, *, mode="full", retain_file_source=True):
    _, _, nodes = build_code_resource_nodes(
        account_id="acct", workspace_root=path.parent, file_path=path, resource_root_slug="repo",
        retain_file_source=retain_file_source,
    )
    by_uri = {n.uri.rstrip("/"): n for n in nodes}
    workspace = ResolvedCodeWorkspace("root", "repo", "f" * 64, path.parent, 300, {})
    service = CodeRetrievalService(
        context_fs=SimpleNamespace(read_node=lambda uri, ctx: by_uri[uri.rstrip("/")],
                                   list_children=lambda uri, ctx: list(by_uri)),
        workspace_resolver=SimpleNamespace(resolve=lambda **kwargs: workspace),
        config=SimpleNamespace(code_retrieval_enabled=True, code_retrieval_mode=mode,
                               code_retrieval_route_enabled=lambda route: route == "symbolic"),
    )
    return service, nodes


def _search_source(path, query, scope, *, mode="full"):
    """Use the real snapshot loader, preserving persisted nodes and cached records."""
    service, nodes = _source_service(path, mode=mode)
    ctx = RequestContext("acct", "user", "agent", "session", "trace")
    records = service._get_snapshot(ctx).records
    original = deepcopy((nodes, records))
    outcome = service.search_active(
        ActiveCodeSearchRequest(query, scope=scope), ctx,
    )
    assert (nodes, records) == original
    return outcome


@pytest.mark.parametrize("terms,expected", [
    (("NEEDLE", "actual"), True),
    (("needle", "missing"), False),
    (("module_1",), False),
    (("relative_path",), False),
    ((".py",), False),
    (("python",), False),
    (("needle|actual",), False),
    (("needle.*actual",), False),
])
def test_full_and_candidate_local_share_source_literal_file_scope(tmp_path, terms, expected):
    path = tmp_path / "relative_path.py"
    path.write_text("needle = 1\ndef actual():\n    return 2\n", encoding="utf-8")
    # A different file cannot satisfy a missing term for the scoped file.
    (tmp_path / "other.py").write_text("missing = 1\n", encoding="utf-8")
    scope = CodeSearchScope(paths=(path.name,), grep_terms=terms)
    outcomes = []
    for mode in ("full", "candidate_local"):
        outcome = _search_source(path, "needle actual", scope, mode=mode)
        assert outcome.status == ("ok" if expected else "zero_result"), (mode, terms, outcome)
        outcomes.append({(e.relative_path, e.symbol) for e in outcome.evidence})
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize("source,terms,expected", [
    ("# needle\ndef actual():\n    return 2\n", ("NEEDLE", "actual"), True),
    ("def other(): pass\n# needle\ndef actual(): pass\n", ("needle", "actual"), True),
    ("def actual(): pass\n# needle\n", ("needle", "actual"), True),
    ("def first(): pass\n\ndef actual(): pass\n", ("pass\n\ndef actual", "actual"), True),
    ("# needle\ndef actual(): pass\n", ("needle",), False),
    ("# needle\ndef actual(): pass\n", ("absent", "actual"), False),
], ids=["leading-comment", "between-functions", "trailing-comment", "cross-chunk-literal",
        "no-chunk-anchor", "missing-term"])
def test_full_and_local_filter_complete_source(tmp_path, source, terms, expected):
    path = tmp_path / "sample.py"
    path.write_text(source, encoding="utf-8")
    outcomes = [_search_source(path, "sample.py", CodeSearchScope(grep_terms=terms), mode=mode)
                for mode in ("full", "candidate_local")]
    assert all(o.status == ("ok" if expected else "zero_result") for o in outcomes)
    assert {(e.symbol, e.excerpt) for e in outcomes[0].evidence} == {
        (e.symbol, e.excerpt) for e in outcomes[1].evidence}
    assert all(any(term.lower() in e.excerpt.lower() for term in terms)
               for o in outcomes for e in o.evidence)


def test_full_source_snapshot_is_single_private_and_independent_of_live_file(tmp_path):
    from index.index_record_builder import build_index_records

    path = tmp_path / "sample.py"
    source = "# needle\ndef actual(): pass\ndef other(): pass\n"
    path.write_text(source)
    service, nodes = _source_service(path)
    assert [n.metadata["file_source"] for n in nodes if "file_source" in n.metadata] == [source]
    assert all("file_source" not in r.metadata and "needle" not in r.text
               for n in nodes for r in build_index_records(n))
    path.unlink()  # Full mode must read the persisted snapshot, never the live workspace.
    ctx = RequestContext("acct", "user", "agent", "session", "trace")
    result = service.search_active(ActiveCodeSearchRequest("sample.py", scope=CodeSearchScope(
        grep_terms=("needle", "actual"))), ctx)
    assert result.status == "ok"
    assert all("file_source" not in r["metadata"] for r in service._get_snapshot(ctx).records)
    items, _ = service.search("sample.py", ctx)
    assert items and all("file_source" not in item["metadata"] for item in items)
    path.write_text(source)
    _, local_nodes = _source_service(path, retain_file_source=False)
    assert all("file_source" not in n.metadata for n in local_nodes)


def test_legacy_full_index_requires_source_refresh_only_for_grep(tmp_path):
    from core.errors import ValidationError

    path = tmp_path / "sample.py"
    path.write_text("# needle\ndef actual(): pass\n")
    service, _ = _source_service(path, retain_file_source=False)
    ctx = RequestContext("acct", "user", "agent", "session", "trace")
    assert service.search_active(ActiveCodeSearchRequest("sample.py"), ctx).status == "ok"
    with pytest.raises(ValidationError) as error:
        service.search_active(ActiveCodeSearchRequest("sample.py", scope=CodeSearchScope(
            grep_terms=("needle", "actual"))), ctx)
    assert error.value.field == "scope.source_snapshot"
    from server.mcp_server import _code_search_exception_message
    assert "source_snapshot_required" in _code_search_exception_message(error.value)
    assert "refresh the affected code files individually" in _code_search_exception_message(error.value)


@pytest.mark.parametrize("second_source,conflict", [
    ("actual = 1", False), ("ACTUAL = 1", True), ("actual = 2", True),
])
def test_full_scope_checks_snapshot_consistency_before_matching(second_source, conflict):
    from core.errors import ValidationError

    records = [dict(uri=f"ctx://acct/resources/{i}", content="actual = 1", _file_source=source,
                    metadata=dict(resource_type="code", resource_root_slug="repo", relative_path="a.py"))
               for i, source in enumerate(("actual = 1", second_source))]
    original = deepcopy(records)
    scope = CodeSearchScope(grep_terms=("actual",))
    for ordered in (records, list(reversed(records))):
        if conflict:
            with pytest.raises(ValidationError) as error:
                CodeRetrievalService._scope_full_records(ordered, scope)
            assert error.value.field == "scope.source_snapshot_conflict"
        else:
            assert len(CodeRetrievalService._scope_full_records(ordered, scope)) == 2
    assert records == original


@pytest.mark.parametrize("difference", ["relative_path", "resource_root_slug", "root_fingerprint", "file_key", "missing_identity"])
def test_full_file_scope_never_combines_unrelated_records(difference):
    metadata = dict(resource_type="code", resource_root_slug="repo", relative_path="a.py",
                    root_fingerprint="root-a", file_key="file-a")
    first = dict(uri="ctx://acct/resources/first", metadata=dict(metadata), content="def alpha(): pass")
    second = dict(uri="ctx://acct/resources/second", metadata=dict(metadata), content="def beta(): pass")
    for record in (first, second):
        record["_file_source"] = record["content"]
    if difference == "missing_identity":
        for record in [first, second]:
            record["metadata"].pop("resource_root_slug")
            record["metadata"].pop("relative_path")
    else:
        second["metadata"][difference] = "other"
    assert not CodeRetrievalService._scope_full_records(
        [first, second], CodeSearchScope(grep_terms=("def alpha", "def beta")),
    )


@pytest.mark.parametrize("mode", ["full", "candidate_local"])
def test_long_documentation_returns_complete_short_implementation(tmp_path, mode):
    source = ('def compute(value):\n    """compute value examples.\n' +
              '    compute value additional explanation.\n' * 42 +
              '    """\n    result = value + 1\n    return result\n')
    path = tmp_path / "sample.py"
    path.write_text(source)
    result = _search_source(path, "compute", CodeSearchScope(paths=("sample.py",)), mode=mode)
    hit = next(h.to_dict() for h in result.evidence if "return result" in h.excerpt)
    last = len(source.splitlines())
    assert hit["excerpt"] == "    result = value + 1\n    return result"
    assert (hit["start_line"], hit["end_line"]) == (last - 1, last)
    assert hit["coverage"] == dict(truncated=True, source_lines=[1, last],
        implementation_lines=[last - 1, last], implementation_complete=True, read_lines=[])


@pytest.mark.parametrize("primary,fallback,expected", [
    ((), ("compute",), "return value"),
    ((), ("compute", "implementation", "after", "docstring"), "return value"),
    ((), ("compute", "documentation"), "DOCUMENTED_OPTION"),
    (("DOCUMENTED_OPTION",), ("compute",), "DOCUMENTED_OPTION"),
])
def test_implementation_preference_preserves_explicit_documentation(primary, fallback, expected):
    source = ('def compute(value):\n    """DOCUMENTED_OPTION compute documentation.\n' +
              '    Extra explanation.\n' * 40 + '    """\n    return value')
    last = len(source.splitlines())
    metadata = dict(start_line=1, symbol_start_line=1, symbol_end_line=last,
                    implementation_start_line=last, symbol="compute")
    excerpt, start, end, coverage = _code_excerpt(source, metadata, 180, primary, fallback)
    assert expected in excerpt
    assert excerpt == "\n".join(source.splitlines()[start - 1:end]).rstrip()
    assert coverage["implementation_complete"] == (expected == "return value")
    assert bool(coverage["read_lines"]) == (expected != "return value")


@pytest.mark.parametrize("partial_line", [False, True])
def test_excerpt_missing_ranges_cover_split_function_and_partial_lines(partial_line):
    source = "    first = 1\n    second = first + 1\n    return second"
    metadata = dict(start_line=30, symbol_start_line=10, symbol_end_line=50,
                    implementation_start_line=20, partial_source_line=partial_line)
    excerpt, start, end, coverage = _code_excerpt(source, metadata, 200)
    assert not coverage["implementation_complete"]
    assert coverage["truncated"]
    assert coverage["read_lines"] == ([[20, 50]] if partial_line else [[20, 29], [33, 50]])
    assert excerpt == source and (start, end) == (30, 32)


def test_long_implementation_clips_to_matching_lines_and_reports_complement():
    source = "def compute(value):\n" + "".join(f"    value += {i}\n" for i in range(60)) + "    return value"
    metadata = dict(start_line=1, symbol_start_line=1, symbol_end_line=62,
                    implementation_start_line=2, symbol="compute")
    excerpt, start, end, coverage = _code_excerpt(source, metadata, 110, ("value += 30",), ("compute",))
    assert "value += 30" in excerpt and len(excerpt) <= 110
    assert excerpt == "\n".join(source.splitlines()[start - 1:end]).rstrip()
    assert coverage["read_lines"] == [[1, start - 1], [end + 1, 62]]
    assert coverage["implementation_complete"] is False
    # A cut inside a source line must request that line again.
    excerpt, start, end, coverage = _code_excerpt("    return " + "x" * 200,
        dict(start_line=3, symbol_start_line=1, symbol_end_line=3, implementation_start_line=3), 30)
    assert len(excerpt) <= 30 and coverage["read_lines"] == [[3, 3]]
    assert coverage["implementation_complete"] is False


@pytest.mark.parametrize("mode", ["full", "candidate_local"])
def test_complete_implementation_includes_first_statement_decorators(tmp_path, mode):
    body = "    @register\n    def inner():\n        return 1\n    return inner"
    source = 'def outer():\n    """outer documentation.\n' + '    outer explanation.\n' * 65 + '    """\n' + body
    path = tmp_path / "sample.py"
    path.write_text(source)
    outcome = _search_source(path, "outer implementation", CodeSearchScope(paths=("sample.py",)), mode=mode)
    hit = next(h for h in outcome.evidence if h.symbol == "outer")
    assert hit.excerpt == body
    assert hit.coverage["implementation_complete"] is True
    assert hit.coverage["read_lines"] == []


@pytest.mark.parametrize("mode", ["full", "candidate_local"])
def test_module_long_line_tail_remains_partial_through_mcp(tmp_path, mode):
    from server.mcp_tools import _render_code_search_result

    source = 'CONFIG = "' + 'x' * 5100 + 'TAIL_MARKER"\n'
    path = tmp_path / "sample.py"
    path.write_text(source)
    outcome = _search_source(path, "CONFIG", CodeSearchScope(
        paths=("sample.py",), grep_terms=("TAIL_MARKER",)), mode=mode)
    assert outcome.status == "ok" and len(outcome.evidence) == 1
    hit = outcome.evidence[0]
    assert len(hit.excerpt) < 1200 and hit.excerpt != source.rstrip()
    assert hit.coverage == dict(truncated=True, source_lines=[1, 1], read_lines=[[1, 1]])
    rendered = _render_code_search_result(outcome.to_dict(), query="CONFIG", max_chars=5000)
    assert hit.excerpt in rendered
    assert '"truncated":true' in rendered and '"read_lines":[[1,1]]' in rendered
