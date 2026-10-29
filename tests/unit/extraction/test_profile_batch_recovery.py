"""A mixed profile delta and read snapshot must produce one CAS-protected write."""
import json
from dataclasses import replace

import pytest

from core.errors import ConcurrentModificationError, ToolCallProtocolError
from core.uri_resolver import URIResolver
from extraction.schemas.registry import SchemaRegistry
from extraction.tools import Extractor, _Span
from service.api import MemoryWriteAPI
from tests.unit.commit.test_memory_identity import api
from tests.unit.commit.test_profile_merge import profile
from tests.unit.extraction.test_memory_update_flow import operation, setup_flow
from tests.unit.extraction.test_prefetch import FakeEmbedder, FakeVectorIndex

OLD = "Nora lives in Utrecht."
NEW = "Nora lives in Utrecht, works as an engineer and plays chess."


class Reconciler:
    def __init__(self, response=None):
        self.payloads = []
        self.response = dict.fromkeys(("abstract", "overview", "content"), NEW) if response is None else response

    def complete_json(self, prompt, schema):
        self.payloads.append(json.loads(prompt.split("\n\n", 1)[1]))
        return self.response


def fixture():
    ctx, fs, registry, uri, _, _, _ = setup_flow("profile", profile("identity", OLD))
    delta = profile("occupation", "Nora works as an engineer.")
    delta.provenance_ids = ["source:delta"]
    snapshot = replace(profile("identity", OLD + " Nora plays chess."),
                       target_uri=uri, expected_version=1, provenance_ids=["source:snapshot"])
    span = _Span(0, 0, safe_user_text="I work as an engineer and play chess.")
    return ctx, fs, registry, uri, delta, snapshot, span


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("source_on", [False, True])
def test_mixed_profile_batch_writes_once_and_preserves_both_sources(reverse, source_on):
    ctx, fs, registry, uri, delta, snapshot, span = fixture()
    llm = Reconciler()
    extractor = Extractor(llm, fs=fs, schema_registry=registry, uri_resolver=URIResolver(registry),
                          source_occurrence_reconciliation_enabled=source_on)
    drafts = [snapshot, delta] if reverse else [delta, snapshot]
    candidates = extractor._merge_span_results([(span, drafts)], ctx)
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.target_uri == uri and candidate.expected_version == 1
    assert set(candidate.provenance_ids) == {"source:delta", "source:snapshot"}
    assert {item["operation"] for item in llm.payloads[0]["updates"]} == {"source_facts", "snapshot"}
    api(fs, llm).write_memory(candidate, ctx)
    stored = fs.read_node(uri, ctx)
    assert stored.metadata["version"] == 2
    assert stored.content == NEW
    assert delta.target_uri == "" and snapshot.expected_version == 1


def test_mixed_profile_still_rejects_concurrent_change_after_reconciliation():
    ctx, fs, registry, uri, delta, snapshot, span = fixture()
    llm = Reconciler()
    extractor = Extractor(llm, fs=fs, schema_registry=registry, uri_resolver=URIResolver(registry))
    candidates = extractor._merge_span_results([(span, [delta, snapshot])], ctx)
    assert len(candidates) == 1
    newer = fs.read_node(uri, ctx)
    newer.content = "Concurrent edit survives."
    fs.write_node(newer, ctx)
    with pytest.raises(ConcurrentModificationError):
        api(fs, llm).write_memory(candidates[0], ctx)
    assert fs.read_node(uri, ctx).content == newer.content


@pytest.mark.parametrize("response", [{}, {"abstract": "ok", "overview": "ok", "content": ""}])
def test_bad_profile_reconciliation_does_not_partially_write(response):
    ctx, fs, registry, uri, delta, snapshot, span = fixture()
    extractor = Extractor(Reconciler(response), fs=fs, schema_registry=registry,
                          uri_resolver=URIResolver(registry))
    with pytest.raises(ToolCallProtocolError):
        extractor._merge_span_results([(span, [delta, snapshot])], ctx)
    assert fs.read_node(uri, ctx).content == OLD
    assert fs.read_node(uri, ctx).metadata["version"] == 1


class DualProfileProvider(Reconciler):
    def __init__(self, uri):
        super().__init__()
        self.uri = uri
        self.version = 1

    def complete_with_tools(self, **kwargs):
        bound = kwargs.get("temperature") == 0
        op = operation("profile", "identity", OLD + " Nora plays chess." if bound
                       else "Nora works as an engineer.", self.uri if bound else "", version=self.version)
        if not bound:
            op["input"].pop("target_uri")
            op["input"].pop("expected_version")
        return [{"tool": op["name"], "input": op["input"]}]

    def complete_json(self, prompt, schema):
        if "spans" in schema.get("properties", {}):
            return {"spans": [{"start": 0, "end": 0, "categories": ["profile"]}]}
        return super().complete_json(prompt, schema)


def public_commit(fs, ctx):
    uri = api(fs, Reconciler()).write_memory(profile("identity", OLD), ctx)["target_uri"]
    provider = DualProfileProvider(uri)
    provider.version = fs.read_node(uri, ctx).metadata.get("version", 0)
    writer = MemoryWriteAPI(fs, provider, directory_summary_enabled=False, extraction_mode="eager",
                            vector_index=FakeVectorIndex(), embedder=FakeEmbedder(),
                            uri_resolver=URIResolver(SchemaRegistry()))
    result = writer.commit_session([{"id": "source-1", "role": "user",
                                     "content": "I work as an engineer and play chess."}], ctx)
    stored = fs.read_node(uri, ctx)
    return result, stored, provider


def test_public_extractor_commit_mixed_profile_no_self_conflict():
    from tests.unit.commit.test_memory_identity import MemoryFS
    from tests.unit.commit.test_profile_merge import profile_context

    result, stored, provider = public_commit(MemoryFS(), profile_context())
    assert result["candidates_extracted"] == result["writes_completed"] == 1
    assert result["writes_failed"] == 0
    assert stored.content == NEW and stored.metadata["version"] == 2
    assert len(provider.payloads) == 1


def test_other_person_delta_keeps_its_separate_entity_attribution():
    from commit.candidate_pipeline import CandidatePipeline

    ctx, fs, registry, uri, delta, snapshot, span = fixture()
    delta = replace(delta, attributed_speaker="Mina", attribution_basis="other_named")
    llm = Reconciler()
    extractor = Extractor(llm, fs=fs, schema_registry=registry, uri_resolver=URIResolver(registry))
    candidates = extractor._merge_span_results([(span, [delta, snapshot])], ctx)
    validated = CandidatePipeline._validate_candidates(candidates, ctx)
    assert {candidate.category for candidate in validated} == {"profile", "entity"}
    assert len(validated) == 2 and llm.payloads == []


def test_complementary_context_fields_survive_mixed_profile():
    ctx, fs, registry, uri, delta, snapshot, span = fixture()
    delta = replace(delta, when="2026-10-01", who="Nora", where="Utrecht")
    extractor = Extractor(Reconciler(), fs=fs, schema_registry=registry, uri_resolver=URIResolver(registry))
    candidate, = extractor._merge_span_results([(span, [delta, snapshot])], ctx)
    assert (candidate.when, candidate.who, candidate.where) == (delta.when, delta.who, delta.where)


def test_complete_profile_prompt_budget_checked_before_call(monkeypatch):
    from core.errors import ExtractionIncompleteError

    ctx, fs, registry, uri, delta, snapshot, span = fixture()
    llm = Reconciler()
    extractor = Extractor(llm, fs=fs, schema_registry=registry, uri_resolver=URIResolver(registry))
    monkeypatch.setattr("extraction.tools.prompt_fits", lambda *args: False)
    with pytest.raises(ExtractionIncompleteError, match="budget"):
        extractor._merge_span_results([(span, [delta, snapshot])], ctx)
    assert llm.payloads == [] and fs.read_node(uri, ctx).content == OLD


def test_agent_only_skill_has_no_user_profile_requirement():
    from tests.unit.commit.test_memory_identity import MemoryFS, memory
    from tests.unit.commit.test_profile_merge import profile_context

    ctx = replace(profile_context(), user_id="")
    candidate = replace(memory("review_steps", category="skill", content="Check inputs."), owner_scope="agent")
    llm, registry = Reconciler(), SchemaRegistry()
    extractor = Extractor(llm, fs=MemoryFS(), schema_registry=registry, uri_resolver=URIResolver(registry))
    assert extractor._merge_span_results([(_Span(0, 0), [candidate])], ctx) == [candidate]
    assert llm.payloads == []
