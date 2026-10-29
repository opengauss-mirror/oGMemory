"""Complete extraction updates retain stable identities through the public write API."""

import json
from dataclasses import replace

import pytest

from core.errors import ToolCallProtocolError
from core.models import RequestContext, SeedHit
from core.uri_resolver import URIResolver
from extraction.extraction_react_loop import ExtractionReActLoop
from extraction.prefetch import MemoryPrefetcher
from extraction.prompts import PromptManager
from extraction.schemas.registry import SchemaRegistry
from extraction.tools import Extractor, _Span
from tests.unit.commit.test_memory_identity import MemoryFS, api, memory
from tests.unit.commit.test_profile_merge import ProfileLLM, profile
from tests.unit.extraction.test_prefetch import FakeEmbedder, FakeVectorIndex


class ExtractionProvider(ProfileLLM):
    def __init__(self, operation):
        super().__init__()
        self.operation = operation
        self.extractions = 0

    def complete_with_tools(self, **kwargs):
        self.extractions += 1
        return [{"tool": self.operation["name"], "input": self.operation["input"]}]

    def complete_with_tools_messages(self, messages, tools, tool_choice):
        self.extractions += 1
        return [self.operation], ""


def setup_flow(category, original):
    ctx = RequestContext(account_id="update-flow", user_id="reader", agent_id="agent",
                         session_id="session", trace_id="trace")
    fs, registry = MemoryFS(), SchemaRegistry()
    uri = api(fs, ProfileLLM()).write_memory(original, ctx)["target_uri"]
    embedder, index = FakeEmbedder(), FakeVectorIndex()
    node = fs.read_node(uri, ctx)
    index.hits = [SeedHit(uri=uri, category=category, abstract=node.abstract, score=.95, level=node.level)]
    prefetcher = MemoryPrefetcher(fs, index, embedder, registry, URIResolver(registry))
    return ctx, fs, registry, uri, embedder, index, prefetcher


def operation(category, key, text, uri, *, version=1):
    fields = dict(routing_key=key, abstract=text, overview=text, content=text,
                  confidence=.9, target_uri=uri, expected_version=version,
                  evidence_quote="I now work as an engineer.", who="user")
    if category == "profile":
        fields.update(attributed_speaker="user", attribution_basis="self_first_person")
    return {"name": "extract_" + category, "input": fields}


def test_eager_profile_snapshot_preserves_facts_with_one_extraction():
    old = "Name: Nora. Residence: Utrecht. Occupation: technician."
    updated = "Name: Nora. Residence: Utrecht. Occupation: engineer."
    ctx, fs, registry, uri, embedder, index, _ = setup_flow("profile", profile("identity", old))
    llm = ExtractionProvider(operation("profile", "identity", updated, uri))
    extractor = Extractor(llm, prompt_manager=PromptManager(), mode="eager", schema_registry=registry,
                          fs=fs, vector_index=index, embedder=embedder, uri_resolver=URIResolver(registry))
    prepared = extractor._prepare_span(
        _Span(0, 0, categories=["profile"]), [{"role": "user", "content": "I now work as an engineer."}],
        "English", None, "", "", ctx,
    )
    candidates = extractor._structure_prepared_span(prepared, ctx)
    writer = api(fs, llm)
    written = writer.write_memory(candidates[0], ctx)
    stored = fs.read_node(uri, ctx)
    assert written["target_uri"] == uri
    assert (stored.abstract, stored.overview, stored.content) == (updated, updated, updated)
    assert stored.metadata["version"] == 2
    assert (llm.extractions, len(llm.calls)) == (1, 0)
    replay = replace(candidates[0], expected_version=2)
    assert writer.write_memory(replay, ctx)["action"] == "skip"
    assert fs.read_node(uri, ctx).metadata["version"] == 2


def test_eager_preference_update_reuses_read_uri_across_names():
    initial = memory("beverage", category="preference", content="Prefers tea.", who="user")
    ctx, fs, registry, uri, embedder, index, _ = setup_flow("preference", initial)
    llm = ExtractionProvider(operation("preference", "drink_choice", "Prefers green tea.", uri))
    extractor = Extractor(llm, mode="eager", schema_registry=registry, fs=fs,
                          vector_index=index, embedder=embedder, uri_resolver=URIResolver(registry))
    messages = [{"role": "user", "content": "I prefer green tea."}]
    prepared = extractor._prepare_span(_Span(0, 0, categories=["preference"]), messages,
                                       "English", None, "", "", ctx)
    candidate = extractor._structure_prepared_span(prepared, ctx)[0]
    written = api(fs, llm).write_memory(candidate, ctx)
    stored = fs.read_node(uri, ctx)
    assert embedder.queries == ["I prefer green tea."]
    assert written["target_uri"] == uri
    assert len(fs.nodes) == 1
    assert stored.content == "Prefers green tea."
    assert stored.metadata["routing_key"] == "beverage"
    assert (llm.extractions, len(llm.calls)) == (1, 0)


def test_lazy_read_binds_complete_update_to_persisted_version():
    old = "Name: Nora. Occupation: technician."
    updated = "Name: Nora. Occupation: engineer."
    ctx, fs, registry, uri, _, _, _ = setup_flow("profile", profile("identity", old))
    llm = ExtractionProvider(operation("profile", "identity", updated, uri))
    def read_then_update(messages, tools, tool_choice):
        llm.extractions += 1
        if messages[-1]["role"] != "tool":
            return [{"name": "read", "input": {"uri": uri}}], ""
        return [], json.dumps([llm.operation])

    llm.complete_with_tools_messages = read_then_update
    loop = ExtractionReActLoop(llm, fs, registry, URIResolver(registry), PromptManager())
    result = loop.run("I now work as an engineer.", ctx)
    candidate = result.candidates[0]
    assert (candidate.target_uri, candidate.expected_version) == (uri, 1)
    api(fs, llm).write_memory(candidate, ctx)
    assert fs.read_node(uri, ctx).content == updated
    assert (llm.extractions, len(llm.calls)) == (2, 0)


def test_profile_spans_share_complete_input_and_one_prepared_prompt():
    old = "Name: Nora."
    ctx, fs, registry, uri, embedder, index, _ = setup_flow("profile", profile("identity", old))
    updated = "Name: Nora. Residence: Utrecht. Occupation: engineer."
    llm = ExtractionProvider(operation("profile", "identity", updated, uri))
    extractor = Extractor(llm, mode="eager", schema_registry=registry, fs=fs,
                          vector_index=index, embedder=embedder, uri_resolver=URIResolver(registry))
    messages = [{"role": "user", "content": "I live in Utrecht."}]
    messages.extend({"role": "assistant", "content": "Understood."} for _ in range(10))
    messages.append({"role": "user", "content": "I now work as an engineer."})
    prepared = extractor._prepare_unique_spans(
        [_Span(0, 0, categories=["profile"]), _Span(11, 11, categories=["event"])],
        messages, "English", None, "", "", ctx,
    )
    assert len(prepared) == 1
    assert prepared[0].span.categories == ["profile", "event"]
    assert "I live in Utrecht." in prepared[0].prompt
    assert "I now work as an engineer." in prepared[0].prompt
    candidate = extractor._structure_prepared_span(prepared[0], ctx)[0]
    api(fs, llm).write_memory(candidate, ctx)
    assert fs.read_node(uri, ctx).content == updated
    assert (llm.extractions, len(llm.calls)) == (1, 0)


def test_dual_run_complete_updates_keep_layers_from_one_snapshot():
    ctx, fs, registry, uri, _, _, _ = setup_flow("profile", profile("identity", "Name: Nora."))
    previous = replace(profile("identity", "Name: Nora. Residence: Utrecht."),
                       target_uri=uri, expected_version=1, confidence=.8)
    preferred = replace(previous, confidence=.95, abstract="Nora in Utrecht.",
                        overview="Nora's residence is Utrecht.", content="Nora lives in Utrecht.")
    candidates = Extractor._merge_dual_run([previous, preferred])
    assert candidates == [preferred]
    assert (candidates[0].abstract, candidates[0].overview, candidates[0].content) == (
        "Nora in Utrecht.", "Nora's residence is Utrecht.", "Nora lives in Utrecht.",
    )


def test_custom_append_only_profile_retains_independent_source_scopes():
    registry = SchemaRegistry()
    registry.register(replace(registry.get("profile"), operation_mode="add_only"))
    ctx = RequestContext(account_id="scopes", user_id="reader", agent_id="agent",
                         session_id="session", trace_id="trace")
    extractor = Extractor(ProfileLLM(), schema_registry=registry)
    messages = [{"role": "user", "content": "I live in Utrecht."}]
    messages.extend({"role": "assistant", "content": "Understood."} for _ in range(10))
    messages.append({"role": "user", "content": "I work as an engineer."})
    prepared = extractor._prepare_unique_spans(
        [_Span(0, 0, categories=["profile"]), _Span(11, 11, categories=["profile"])],
        messages, "English", None, "", "", ctx,
    )
    assert [(item.span.start, item.span.end) for item in prepared] == [(0, 0), (11, 11)]


def test_dual_run_uri_identity_selects_one_current_snapshot_across_names():
    target = "ctx://scopes/users/reader/memories/preferences/beverage"
    first = replace(memory("drink_choice", category="preference", content="Prefers tea."),
                    target_uri=target, expected_version=1)
    current = replace(first, routing_key="beverage", expected_version=2,
                      abstract="Prefers green tea.", overview="Prefers green tea.", content="Prefers green tea.")
    assert Extractor._merge_dual_run([first, current]) == [current]


class CrossSpanProvider(ExtractionProvider):
    def __init__(self, uri, first, last, combined):
        super().__init__(None)
        self.uri, self.first, self.last, self.combined = uri, first, last, combined
        self.reconciliations = []

    def complete_with_tools(self, **kwargs):
        self.extractions += 1
        text = self.first if "I prefer tea without sugar." in kwargs["prompt"] else self.last
        op = operation("preference", "drink_choice", text, self.uri)
        return [{"tool": op["name"], "input": op["input"]}]

    def complete_json(self, prompt, schema):
        self.reconciliations.append(json.loads(prompt.split("\n\n", 1)[1]))
        return dict(abstract=self.combined, overview=self.combined, content=self.combined)


def test_independent_spans_reconcile_one_target_and_preserve_all_sources():
    initial = memory("beverage", category="preference", content="Prefers tea.", who="user")
    initial.provenance_ids = ["source:original"]
    ctx, fs, registry, uri, embedder, index, _ = setup_flow("preference", initial)
    first = "Prefers tea without sugar."
    last = "Prefers tea brewed with 40 degree water."
    combined = "Prefers tea without sugar, brewed with 40 degree water."
    llm = CrossSpanProvider(uri, first, last, combined)
    extractor = Extractor(
        llm,
        schema_registry=registry,
        fs=fs,
        vector_index=index,
        embedder=embedder,
        uri_resolver=URIResolver(registry),
    )
    messages = [{"id": "m0", "role": "user", "content": "I prefer tea without sugar."}]
    messages.extend({"role": "assistant", "content": "Understood."} for _ in range(12))
    messages.append(
        {"id": "m13", "role": "user", "content": "I prefer tea brewed with 40 degree water."}
    )
    candidates = extractor._structure_spans(
        [
            _Span(0, 0, categories=["preference"], message_ids=["m0"]),
            _Span(13, 13, categories=["preference"], message_ids=["m13"]),
        ],
        messages,
        "English",
        None,
        "",
        "",
        ctx,
        "archive-tea",
    )
    assert len(candidates) == 1
    assert candidates[0].content == combined
    assert len(llm.reconciliations) == 1
    payload = llm.reconciliations[0]
    assert payload["existing"]["content"] == "Prefers tea."
    assert [item["evidence"] for item in payload["updates"]] == [
        messages[0]["content"],
        messages[13]["content"],
    ]
    assert [item["snapshot"]["content"] for item in payload["updates"]] == [first, last]
    api(fs, llm).write_memory(candidates[0], ctx)
    stored = fs.read_node(uri, ctx)
    assert stored.content == stored.abstract == stored.overview == combined
    assert stored.metadata["version"] == 2
    from core.provenance_resolver import ProvenanceResolver

    assert stored.metadata["provenance_ids"] == [
        "source:original",
        ProvenanceResolver.build_id("archive", "archive-tea", ["m0"]),
        ProvenanceResolver.build_id("archive", "archive-tea", ["m13"]),
    ]
    assert llm.extractions == 4


def test_span_reconciliation_orders_corrections_and_binds_current_version():
    ctx, fs, registry, uri, _, _, _ = setup_flow(
        "preference",
        memory("beverage", category="preference", content="Prefers sweet tea."),
    )
    node = fs.read_node(uri, ctx)
    node.content = "Prefers sweet green tea."
    fs.write_node(node, ctx)
    fs.write_node(node, ctx)
    earlier = replace(
        memory("tea", category="preference", content="Prefers sweet tea."),
        target_uri=uri,
        expected_version=1,
        provenance_ids=["source:early"],
    )
    later = replace(earlier, content="Prefers unsweetened tea.", provenance_ids=["source:late"])

    class CorrectionProvider:
        def complete_json(self, prompt, schema):
            payload = json.loads(prompt.split("\n\n", 1)[1])
            assert payload["existing"]["content"] == "Prefers sweet green tea."
            assert [item["evidence"] for item in payload["updates"]] == [
                "I prefer sweet tea.",
                "I now prefer tea without sugar.",
            ]
            assert "an unchanged old fact is NOT a correction" in prompt
            return dict.fromkeys(
                ("abstract", "overview", "content"), "Prefers unsweetened green tea."
            )

    extractor = Extractor(CorrectionProvider(), schema_registry=registry, fs=fs)
    candidate = extractor._merge_span_results(
        [
            (_Span(13, 13, safe_user_text="I now prefer tea without sugar."), [later]),
            (_Span(0, 0, safe_user_text="I prefer sweet tea."), [earlier]),
        ],
        ctx,
    )[0]
    assert candidate.expected_version == 3
    assert candidate.target_uri == uri
    assert candidate.provenance_ids == ["source:early", "source:late"]
    api(fs, CorrectionProvider()).write_memory(candidate, ctx)
    assert fs.read_node(uri, ctx).content == "Prefers unsweetened green tea."
    assert fs.read_node(uri, ctx).metadata["version"] == 4


@pytest.mark.parametrize(
    "response",
    [
        None,
        {},
        {"abstract": "a", "overview": "b", "content": ""},
        {"abstract": "a", "overview": "b", "content": ["tea"]},
    ],
)
def test_invalid_span_reconciliation_leaves_original_memory_intact(response):
    ctx, fs, registry, uri, _, _, _ = setup_flow(
        "preference",
        memory("beverage", category="preference", content="Prefers tea."),
    )
    candidate = replace(
        memory("tea", category="preference", content="Prefers green tea."),
        target_uri=uri,
        expected_version=1,
    )

    class InvalidProvider:
        def complete_json(self, prompt, schema):
            return response

    extractor = Extractor(InvalidProvider(), schema_registry=registry, fs=fs)
    with pytest.raises(ToolCallProtocolError, match="complete memory snapshot"):
        extractor._merge_span_results(
            [(_Span(0, 0), [candidate]), (_Span(13, 13), [candidate])], ctx
        )
    assert fs.read_node(uri, ctx).content == "Prefers tea."
    assert fs.read_node(uri, ctx).metadata["version"] == 1


def test_independent_targets_do_not_require_reconciliation_calls():
    candidate = replace(
        memory("tea", category="preference", content="Prefers green tea."),
        target_uri="ctx://one",
        expected_version=1,
    )
    other = replace(candidate, target_uri="ctx://two")
    extractor = Extractor(object())
    result = extractor._merge_span_results(
        [(_Span(0, 0), [candidate]), (_Span(13, 13), [other])],
        RequestContext("account", "user", "agent", "session", "trace"),
    )
    assert result == [candidate, other]


def test_low_confidence_span_does_not_suppress_supported_update():
    ctx, fs, registry, uri, _, _, _ = setup_flow(
        "preference",
        memory("beverage", category="preference", content="Prefers tea."),
    )
    supported = replace(
        memory("tea", category="preference", content="Prefers green tea."),
        target_uri=uri,
        expected_version=1,
        confidence=0.9,
    )
    uncertain = replace(supported, content="Maybe prefers sweet tea.", confidence=0.4)

    class Provider:
        def complete_json(self, prompt, schema):
            return dict.fromkeys(("abstract", "overview", "content"), supported.content)

    extractor = Extractor(Provider(), schema_registry=registry, fs=fs)
    result = extractor._merge_span_results(
        [
            (_Span(0, 0), [supported]),
            (_Span(13, 13), [uncertain]),
        ],
        ctx,
    )
    assert result == [supported]
