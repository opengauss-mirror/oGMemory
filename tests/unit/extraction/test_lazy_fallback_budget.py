"""Optional prefetch cannot prevent an otherwise bounded source-only recovery."""
import pytest

from core.errors import ExtractionIncompleteError, ToolCallProtocolError
from core.uri_resolver import URIResolver
from extraction.extraction_react_loop import ExtractionReActLoop
from extraction.tools import Extractor, _Span
from tests.unit.commit.test_profile_merge import profile
from tests.unit.extraction.test_memory_update_flow import operation, setup_flow

SOURCE = "I work as an engineer."


class NativeProvider:
    def __init__(self, uri=None):
        self.uri = uri
        self.calls = []

    def complete_with_tools(self, **kwargs):
        self.calls.append(kwargs)
        op = operation("profile", "identity", SOURCE, self.uri or "")
        if self.uri is None:
            op["input"].pop("target_uri")
            op["input"].pop("expected_version")
        return [{"tool": op["name"], "input": op["input"]}]


def setup(monkeypatch, *, large=True, bound=False):
    old = "OLD_FACT " * (7000 if large else 4)
    ctx, fs, registry, uri, embedder, index, _ = setup_flow("profile", profile("identity", old))
    llm = NativeProvider(uri if bound else None)
    extractor = Extractor(llm, mode="lazy", fs=fs, schema_registry=registry,
                          uri_resolver=URIResolver(registry), embedder=embedder, vector_index=index)
    def failed_loop(*args):
        raise ToolCallProtocolError("native extraction contains an oversized abstract")
    monkeypatch.setattr(ExtractionReActLoop, "run", failed_loop)
    span = _Span(0, 0, categories=["profile"], safe_user_text=SOURCE,
                 focused_messages=[{"role": "user", "content": SOURCE}])
    return extractor, llm, span, ctx, uri


def test_oversized_prefetch_does_not_block_complete_source_recovery(monkeypatch):
    extractor, llm, span, ctx, uri = setup(monkeypatch)
    candidates = extractor._structure_span_lazy(span, SOURCE, ctx)
    assert len(candidates) == len(llm.calls) == 1
    assert SOURCE in llm.calls[0]["prompt"] and "OLD_FACT" not in llm.calls[0]["prompt"]
    assert candidates[0].target_uri == "" and span.successful_read_versions == [{}]


def test_discarded_prefetch_cannot_authorize_bound_write(monkeypatch):
    extractor, llm, span, ctx, uri = setup(monkeypatch, bound=True)
    with pytest.raises(ToolCallProtocolError, match="successful read"):
        extractor._structure_span_lazy(span, SOURCE, ctx)
    assert len(llm.calls) == 1


def test_small_prefetch_is_kept_and_authorizes_its_actual_read(monkeypatch):
    extractor, llm, span, ctx, uri = setup(monkeypatch, large=False, bound=True)
    candidate, = extractor._structure_span_lazy(span, SOURCE, ctx)
    assert "OLD_FACT" in llm.calls[0]["prompt"]
    assert candidate.target_uri == uri and span.successful_read_versions == [{uri: 1}]


def test_oversized_source_still_fails_without_model_call(monkeypatch):
    extractor, llm, span, ctx, uri = setup(monkeypatch)
    with pytest.raises(ExtractionIncompleteError, match="budget"):
        extractor._structure_span_lazy(span, SOURCE * 5000, ctx)
    assert llm.calls == []
