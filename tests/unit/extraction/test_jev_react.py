"""Exercise JEV at the native Lazy extraction and recovery boundaries."""

import json
from copy import deepcopy
from urllib.error import HTTPError

import pytest

from core.errors import ExtractionIncompleteError, LLMCallError
from core.models import ContextNode, RequestContext
from core.uri_resolver import URIResolver
from extraction.extraction_react_loop import ExtractionReActLoop, create_extraction_loop
from extraction.prefetch import PrefetchResult
from extraction.prompts import PromptManager
from extraction.schemas.registry import SchemaRegistry
from extraction.tools import Extractor, _Span, generate_memory_operations
from providers.decision import JevDecisionClient
from providers.decision.config import DecisionConfig
from providers.token_tracker import UsagePersistenceError, usage_scope
from tests.unit.commit.test_memory_identity import MemoryFS

COFFEE = "ctx://acme/users/alice/memories/preferences/coffee"
TEA = "ctx://acme/users/alice/memories/preferences/tea"
TRAVEL = "ctx://acme/users/alice/memories/preferences/travel"
SOURCE = "Alice: I now prefer coffee without sugar, still with milk. I now take tea without milk."


def context():
    return RequestContext(account_id="acme", user_id="alice", agent_id="assistant",
                          session_id="session", trace_id="trace")


def preference(uri, version, content):
    return ContextNode(
        uri=uri, context_type="MEMORY", category="preference", level=2,
        owner_space="user:alice", abstract=content, overview=content, content=content,
        metadata={"version": version, "routing_key": uri.rsplit("/", 1)[-1], "who": "user", "scope_ref": ""},
    )


def memory_fs():
    fs = MemoryFS()
    fs.nodes[COFFEE] = preference(COFFEE, 7, "I prefer coffee with sugar. I still like milk.")
    fs.nodes[TEA] = preference(TEA, 4, "I take tea with milk. I like tea warm.")
    return fs


def operations():
    return [
        {"tool": "edit_memory", "input": {
            "target_uri": COFFEE, "expected_version": 7, "search": "with sugar", "replacement": "without sugar",
            "abstract": "Coffee without sugar and with milk",
            "overview": "Coffee without sugar, still with milk.", "confidence": 0.95,
            "evidence_quote": "I now prefer coffee without sugar, still with milk.",
        }},
        {"tool": "edit_memory", "input": {
            "target_uri": TEA, "expected_version": 4, "search": "with milk", "replacement": "without milk",
            "abstract": "Warm tea without milk",
            "overview": "Tea without milk, still warm.", "confidence": 0.95,
            "evidence_quote": "I now take tea without milk.",
        }},
    ]


def prefetch(fs, *, read=False):
    return PrefetchResult(
        uri_hints=[{"uri": uri, "abstract": uri.rsplit("/", 1)[-1], "score": 0.9}
                   for uri in (COFFEE, TEA, TRAVEL)],
        is_lazy=True,
        read_uris={COFFEE, TEA} if read else set(),
        read_nodes=dict(fs.nodes) if read else {},
        read_versions={COFFEE: 7, TEA: 4} if read else {},
    )


def scripted_decision(steps):
    calls = []
    remaining = iter(steps)

    def transport(payload):
        calls.append(deepcopy(payload))
        selection = next(remaining)
        if isinstance(selection, Exception):
            raise selection
        answers = {}
        for name, question in payload["questions"].items():
            if question["type"] == "noul":
                answers[name] = {"type": "noul", "noul": selection[name]}
            else:
                probabilities = {key: float(key == selection) for key in question["criteria"]}
                answers[name] = {"type": "choice", "choice": selection, "probabilities": probabilities}
        return {"answers": answers, "usage": {"input_tokens": 10, "output_tokens": 1}}

    return JevDecisionClient(
        DecisionConfig(base_url="https://decision.example", model="jev"), transport=transport,
    ), calls


def create_operation():
    return {"tool": "extract_preference", "input": {
        "routing_key": "coffee", "who": "user", "scope_ref": "",
        "abstract": "Coffee without sugar and with milk",
        "overview": "Coffee without sugar, still with milk.", "confidence": 0.95,
        "evidence_quote": "I now prefer coffee without sugar, still with milk.",
        "content": "I prefer coffee without sugar. I still like milk.",
    }}


class Generator:
    def __init__(self, *, final_operations=None, call_error=None):
        self.requests = []
        self.final_operations = final_operations
        self.call_error = call_error

    def complete_with_tools(self, **kwargs):
        self.requests.append(("tools", kwargs))
        if self.call_error is not None:
            raise self.call_error
        return operations()

    def complete_with_tools_messages(self, **kwargs):
        self.requests.append(("text", kwargs))
        return [], json.dumps(self.final_operations)


def loop_for(decision, generator, fs, *, iterations=3, manager=None):
    registry = SchemaRegistry()
    return create_extraction_loop(
        decision_client=decision, generate_operations=generate_memory_operations, read_probability_threshold=0.70,
        llm=generator, fs=fs, registry=registry, uri_resolver=URIResolver(registry),
        prompt_manager=manager or PromptManager(), max_iterations=iterations,
    )


def run(loop, prepared):
    return loop.run(SOURCE, context(), prepared, categories=frozenset({"preference"}))


def test_batch_reads_and_same_tool_operations_reach_native_commit():
    decision, requests = scripted_decision([
        "EXPLORE_OLD_MEMORY", {"memory_0": .92, "memory_1": .85, "memory_2": .18},
        "FORM_MEMORY_OPERATION", "FINISH_EXTRACTION",
    ])
    fs = memory_fs()
    llm = Generator()
    result = run(loop_for(decision, llm, fs), prefetch(fs))
    assert [call["params"]["uri"] for call in result.tools_used if call["tool_name"] == "read"] == [COFFEE, TEA]
    assert [candidate.target_uri for candidate in result.candidates] == [COFFEE, TEA]
    assert len(llm.requests) == 1
    mode, generation = llm.requests[0]
    assert mode == "tools" and generation["tool_choice"] == "required"
    assert "Explore first" not in generation["prompt"]
    assert "Duplicate/conflict handling" in generation["prompt"]
    assert SOURCE in generation["prompt"] and generation["prompt"].count(SOURCE) == 1
    evidence = json.loads(generation["prompt"].split("## Current evidence and progress\n", 1)[1]
                          .split("\n\n## Current source", 1)[0])
    assert {item["uri"] for item in evidence["memory_index"].values()} == {COFFEE, TEA, TRAVEL}
    assert evidence["memory_index"] == requests[2]["state"]["memory_index"]
    assert all(tool["name"] not in {"read", "list"} for tool in generation["tools"])
    schema = next(t["input_schema"] for t in generation["tools"] if t["name"] == "extract_preference")
    from jsonschema import Draft202012Validator
    validator = Draft202012Validator(schema)
    new_fields = create_operation()["input"]
    assert validator.is_valid(new_fields)
    assert not validator.is_valid({**new_fields, "target_uri": COFFEE})
    assert not validator.is_valid({**new_fields, "expected_version": 7})
    assert not validator.is_valid({**new_fields, "target_uri": "", "expected_version": 7})
    assert validator.is_valid({**new_fields, "target_uri": COFFEE, "expected_version": 7})
    assert len(requests) == decision.token_tracker.snapshot().llm_calls == 4
    assert 'memory_index["memory_0"]' in requests[1]["questions"]["memory_0"]["instructions"]
    assert [read["version"] for read in requests[2]["state"]["read_results"]] == [7, 4]
    assert len(requests[3]["state"]["recorded_operations"]) == 2

    from tests.unit.commit.test_memory_identity import DecisionLLM, api

    for i, candidate in enumerate(result.candidates):
        original_version = fs.nodes[candidate.target_uri].metadata["version"]
        candidate.provenance_ids = [f"prov:1:archive:jev-update:message-{i}"]
        receipt = api(fs, DecisionLLM()).write_memory(candidate, context())
        assert receipt["target_uri"] == candidate.target_uri
        assert fs.nodes[candidate.target_uri].metadata["version"] == original_version + 1
    assert fs.nodes[COFFEE].content == "I prefer coffee without sugar. I still like milk."
    assert fs.nodes[TEA].content == "I take tea without milk. I like tea warm."


@pytest.mark.parametrize("vector_hit", [False, True])
def test_identity_catalog_survives_prefetch_decision_and_generation_without_read_authority(vector_hit):
    from core.models import SeedHit
    from extraction.prefetch import MemoryPrefetcher
    from tests.unit.extraction.test_prefetch import FakeEmbedder, FakeVectorIndex

    class CatalogFS(MemoryFS):
        def list_node_summaries_page(self, directory, ctx, *, limit, after_uri=None):
            return sorted((node for uri, node in self.nodes.items()
                           if uri.rsplit("/", 1)[0] == directory
                           and (after_uri is None or uri > after_uri)), key=lambda node: node.uri)[:limit]

    fs = CatalogFS()
    fs.nodes = memory_fs().nodes
    fs.nodes[COFFEE].metadata.update({"object_name": "Coffee", "object_name_aliases": ["morning drink"],
                                     "routing_key_aliases": ["morning_coffee"], "scope_ref": "breakfast"})
    registry = SchemaRegistry()
    vector = FakeVectorIndex()
    if vector_hit:
        vector.hits = [SeedHit(uri=COFFEE, category="preference", abstract="Coffee", score=.9, level=0)]
    prepared = MemoryPrefetcher(fs, vector, FakeEmbedder(), registry, URIResolver(registry)).prefetch_for_span(
        ["preference"], context(), conversation_text=SOURCE, lazy=True,
    )
    assert bool(prepared.uri_hints) is vector_hit
    assert not prepared.read_uris and not prepared.read_versions
    decision, requests = scripted_decision([
        "EXPLORE_OLD_MEMORY", {"memory_0": .92, "memory_1": .85},
        "FORM_MEMORY_OPERATION", "FINISH_EXTRACTION",
    ])
    llm = Generator()
    result = run(loop_for(decision, llm, fs), prepared)
    initial = requests[0]["state"]
    coffee = next(entry for entry in initial["memory_index"].values() if entry["uri"] == COFFEE)
    assert coffee["abstract"] == "I prefer coffee with sugar. I still like milk."
    assert coffee["who"] == "user" and coffee["scope_ref"] == "breakfast"
    assert coffee["object_name"] == "Coffee" and coffee["routing_key"] == "coffee"
    assert coffee["object_name_aliases"] == ["morning drink"]
    assert coffee["routing_key_aliases"] == ["morning_coffee"]
    if vector_hit:
        assert coffee["score"] == .9
    assert initial["read_results"] == []
    generation = llm.requests[0][1]["prompt"]
    evidence = json.loads(generation.split("## Current evidence and progress\n", 1)[1]
                          .split("\n\n## Current source", 1)[0])
    assert evidence["memory_index"] == initial["memory_index"]
    assert {read["uri"]: read["version"] for read in evidence["read_results"]} == {COFFEE: 7, TEA: 4}
    assert {candidate.target_uri for candidate in result.candidates} == {COFFEE, TEA}


@pytest.mark.parametrize("name", ["legacy_breakfast", "legacy%20breakfast", "%E6%97%A9%E9%A4%90"])
def test_agfs_without_vector_prefetch_reads_alias_target_and_updates_existing_node(monkeypatch, name):
    from commit.outbox_store import OutboxStore
    from fs.agfs_adapter.agfs_context_fs import AGFSContextFS
    from providers.unified_config import OgMemConfig
    from server.memory_service import MemoryService
    from tests.unit.server.test_agent_memory import _MemoryAGFS

    client = _MemoryAGFS()
    fs = AGFSContextFS(client)
    source = "Alice: I now prefer coffee without sugar, still with milk."
    target = f"ctx://acme/users/alice/memories/preferences/{name}"
    node = preference(target, 1, "I prefer coffee with sugar. I still like milk.")
    node.metadata["routing_key"] = "legacy_breakfast"
    node.metadata["object_name_aliases"] = ["coffee", "morning drink"]
    fs.write_node(node, context())
    version = fs.read_node(target, context()).metadata["version"]
    private = "ctx://acme/users/bob/memories/preferences/private_coffee"
    other_ctx = RequestContext(account_id="acme", user_id="bob", agent_id="assistant",
                               session_id="private", trace_id="trace")
    private_node = preference(private, 1, "Bob's private preference")
    private_node.owner_space = "user:bob"
    fs.write_node(private_node, other_ctx)
    decision, requests = scripted_decision([
        "EXPLORE_OLD_MEMORY", {"memory_0": .92}, "FORM_MEMORY_OPERATION", "FINISH_EXTRACTION",
    ])

    class AliasGenerator(Generator):
        def complete_with_tools(self, **kwargs):
            self.requests.append(("tools", kwargs))
            operation = operations()[0]
            operation["input"].update(target_uri=target, expected_version=version)
            return [operation]

    llm = AliasGenerator()
    service = MemoryService(OgMemConfig(
        provider="mock", extraction_mode="lazy", retrieval_mode="agfs", storage_backend="agfs",
        directory_summary_enabled=False, decision=DecisionConfig(base_url="http://localhost:1", model="jev"),
    ))
    monkeypatch.setattr(service, "_get_context_fs", lambda: fs)
    monkeypatch.setattr(service, "_get_outbox_store", lambda fs: OutboxStore(client, fs))
    monkeypatch.setattr(service, "get_llm", lambda: llm)
    monkeypatch.setattr(service, "get_decision_client", lambda: decision)
    monkeypatch.setattr(service, "_start_outbox_worker", lambda: None)
    writer = service.get_write_api()
    extractor = writer._pipeline._extractors[0]
    assert extractor._vector_index is None and extractor._embedder is None
    candidates = extractor._structure_span_lazy(
        _Span(0, 0, categories=["preference"]), source, context(), archive_id="agfs-alias",
        focused_messages=[{"role": "user", "content": source}], categories=frozenset({"preference"}),
    )
    initial = requests[0]["state"]
    assert {entry["uri"] for entry in initial["memory_index"].values()} == {target}
    assert initial["memory_index"]["memory_0"]["object_name_aliases"] == ["coffee", "morning drink"]
    assert initial["read_results"] == []
    assert requests[2]["state"]["read_results"][0]["uri"] == target
    assert "I still like milk" in llm.requests[0][1]["prompt"]
    assert len(candidates) == 1 and candidates[0].target_uri == target
    receipt = writer.write_memory(candidates[0], context())
    assert receipt["target_uri"] == target
    saved = fs.read_node(target, context())
    assert saved.content == "I prefer coffee without sugar. I still like milk."
    assert saved.metadata["version"] == version + 1
    summaries = fs.list_node_summaries_page(target.rsplit("/", 1)[0], context(), limit=10)
    assert {node.uri for node in summaries} == {target}


@pytest.mark.parametrize("iterations", [1, 3])
def test_generation_without_a_complete_read_only_offers_unbound_operations(iterations):
    decision, _ = scripted_decision(["FORM_MEMORY_OPERATION", "FINISH_EXTRACTION"])
    fs = memory_fs()
    del fs.nodes[COFFEE]

    class CreateOnlyGenerator(Generator):
        def complete_with_tools(self, **kwargs):
            self.requests.append(("tools", kwargs))
            return [create_operation()]

    llm = CreateOnlyGenerator(final_operations=[create_operation()])
    prepared = prefetch(fs)
    # A hint and a body without a valid recorded version cannot authorize an update.
    prepared.read_nodes[TEA] = fs.nodes[TEA]
    result = run(loop_for(decision, llm, fs, iterations=iterations), prepared)
    assert len(result.candidates) == 1
    mode, request = llm.requests[0]
    if mode == "text":
        prompt = request["messages"][0]["content"]
        schemas = prompt.split("Allowed operation schemas:\n", 1)[1].split("\n\n", 1)[0]
        tools = json.loads(schemas)
    else:
        prompt = request["prompt"]
        tools = request["tools"]
    assert {tool["name"] for tool in tools} == {"extract_preference"}
    assert all("target_uri" not in tool["input_schema"]["properties"] for tool in tools)
    assert all("expected_version" not in tool["input_schema"]["properties"] for tool in tools)
    assert "Bound upserts:" not in prompt
    assert "Bound upserts:" not in tools[0]["description"]


def test_empty_read_selection_reenters_decision_without_reading():
    decision, requests = scripted_decision([
        "EXPLORE_OLD_MEMORY", {"memory_0": .2, "memory_1": .3, "memory_2": .1}, "FINISH_EXTRACTION",
    ])
    fs = memory_fs()
    llm = Generator()
    result = run(loop_for(decision, llm, fs), prefetch(fs))
    assert result.completed and not result.tools_used and not llm.requests
    assert len(requests) == 3
    assert "EXPLORE_OLD_MEMORY" not in requests[-1]["questions"]["next_step"]["criteria"]
    assert requests[-1]["state"]["read_selection_feedback"]["selected"] == []


@pytest.mark.parametrize("count", [0, 1, 2])
def test_final_text_checks_coverage_before_adopting_candidates(count):
    final_action = "FINISH_EXTRACTION" if count == 2 else "FORM_MEMORY_OPERATION"
    decision, requests = scripted_decision(["FORM_MEMORY_OPERATION", final_action])
    fs = memory_fs()
    llm = Generator(final_operations=operations()[:count])
    loop = loop_for(decision, llm, fs, iterations=1)
    if count == 2:
        assert {c.target_uri for c in run(loop, prefetch(fs, read=True)).candidates} == {COFFEE, TEA}
    else:
        with pytest.raises(ExtractionIncompleteError, match="complete source"):
            run(loop, prefetch(fs, read=True))
    assert len(requests) == 2
    assert len(requests[-1]["state"]["tentative_operations"]) == count
    assert requests[-1]["state"]["recorded_operations"] == []
    assert requests[-1]["state"]["source_span"]["text"] == SOURCE
    assert llm.requests[0][0] == "text"
    assert llm.requests[0][1]["tools"] is None


def test_final_check_combines_recorded_and_tentative_targets(monkeypatch):
    decision, requests = scripted_decision([
        "FORM_MEMORY_OPERATION", "FORM_MEMORY_OPERATION", "FINISH_EXTRACTION",
    ])
    fs = memory_fs()
    llm = Generator(final_operations=operations()[1:])

    def form_coffee(**kwargs):
        return operations()[:1]

    monkeypatch.setattr(llm, "complete_with_tools", form_coffee)
    result = run(loop_for(decision, llm, fs, iterations=2), prefetch(fs, read=True))
    state = requests[-1]["state"]
    assert [op["input"]["target_uri"] for op in state["recorded_operations"]] == [COFFEE]
    assert [op["target_uri"] for op in state["tentative_operations"]] == [TEA]
    assert {candidate.target_uri for candidate in result.candidates} == {COFFEE, TEA}


@pytest.mark.parametrize("stage", ["decision_transport", "decision_usage", "generation_transport", "generation_usage"])
def test_model_failures_escape_lazy_without_eager_retry(stage):
    from providers.token_tracker import TokenTracker

    failure = LLMCallError("upstream unavailable", status_code=503)

    class FailFirstWrite:
        calls = 0
        def record(self, event):
            self.calls += 1
            if self.calls == 1:
                raise OSError("usage store unavailable")

    class RecoveryCapableGenerator(Generator):
        def complete_with_tools(self, **kwargs):
            self.requests.append(("tools", kwargs))
            if len(self.requests) == 1 and stage.startswith("generation"):
                if stage == "generation_usage":
                    tracker.record_llm(10, 2)
                raise failure
            return [create_operation()]

    steps = (
        [HTTPError("https://decision.example", 503, "unavailable", None, None)]
        if stage == "decision_transport" else ["FORM_MEMORY_OPERATION"]
    )
    decision, requests = scripted_decision(steps)
    tracker = TokenTracker("generator")
    store = FailFirstWrite()
    if stage == "decision_usage":
        decision.token_tracker.set_usage_store(store)
    if stage == "generation_usage":
        tracker.set_usage_store(store)
    llm = RecoveryCapableGenerator()
    fs = memory_fs()
    registry = SchemaRegistry()
    extractor = Extractor(llm, mode="lazy", schema_registry=registry, fs=fs,
                          uri_resolver=URIResolver(registry), decision_client=decision)
    exception = UsagePersistenceError if stage.endswith("usage") else LLMCallError
    with usage_scope("acme", "user:alice"), pytest.raises(exception) as caught:
        extractor._structure_span_lazy(
            _Span(0, 0, categories=["preference"], message_ids=["m1"]), SOURCE, context(),
            categories=frozenset({"preference"}),
        )
    assert len(requests) == 1
    assert len(llm.requests) == int(stage.startswith("generation"))
    if isinstance(caught.value, LLMCallError):
        assert caught.value.status_code == 503
    else:
        assert store.calls == 1


def test_incomplete_final_batch_enters_full_source_recovery(monkeypatch):
    decision, _ = scripted_decision(["FORM_MEMORY_OPERATION", "FORM_MEMORY_OPERATION"])
    fs = memory_fs()
    llm = Generator(final_operations=operations()[:1])
    original_factory = create_extraction_loop

    def one_round(**kwargs):
        return original_factory(**kwargs, max_iterations=1)

    monkeypatch.setattr("extraction.extraction_react_loop.create_extraction_loop", one_round)
    registry = SchemaRegistry()
    extractor = Extractor(llm, mode="lazy", schema_registry=registry, fs=fs,
                          uri_resolver=URIResolver(registry), decision_client=decision)
    monkeypatch.setattr(extractor, "_collect_prefetch_context", lambda *a, **kw: prefetch(fs, read=True))
    def with_prefetch(span, prompt, ctx):
        span.prefetch_result = prefetch(fs, read=True)
        return prompt

    monkeypatch.setattr(extractor, "_with_prefetch_context", with_prefetch)
    extractor._prefetcher = object()
    span = _Span(0, 0, categories=["preference"], message_ids=["m1"])
    candidates = extractor._structure_span_lazy(span, SOURCE, context(), categories=frozenset({"preference"}))
    assert {c.target_uri for c in candidates} == {COFFEE, TEA}
    assert [mode for mode, _ in llm.requests] == ["text", "tools"]
    assert SOURCE in llm.requests[1][1]["prompt"]


def test_unconfigured_controller_stays_native():
    loop = loop_for(None, Generator(), memory_fs())
    assert type(loop) is ExtractionReActLoop


@pytest.mark.parametrize("mode,configured", [("lazy", True), ("lazy", False), ("eager", True)])
def test_service_binds_existing_client_only_for_configured_lazy(monkeypatch, mode, configured):
    from providers.llm.mock_llm import MockLLM
    from providers.unified_config import OgMemConfig
    from server.memory_service import MemoryService

    config = OgMemConfig(
        provider="mock",
        vector_db_type="memory",
        extraction_mode=mode,
        read_probability_threshold=0.82,
        decision=DecisionConfig(base_url="https://decision.example", model="jev")
        if configured
        else None,
    )
    service = MemoryService(config)
    fs = MemoryFS()
    monkeypatch.setattr(service, "_get_context_fs", lambda: fs)
    monkeypatch.setattr(service, "_get_outbox_store", lambda _: object())
    monkeypatch.setattr(service, "get_llm", lambda: MockLLM())
    monkeypatch.setattr(service, "_get_shared_vector_index", lambda: None)
    monkeypatch.setattr(service, "_get_shared_embedder", lambda: None)
    monkeypatch.setattr(service, "_start_outbox_worker", lambda: None)
    monkeypatch.setattr(service, "_get_model_usage_store", lambda: None)
    api = service.get_write_api()
    if configured and mode == "lazy":
        assert api._pipeline._extractors[0]._decision_client is service.get_decision_client()
        assert api._pipeline._extractors[0]._read_probability_threshold == 0.82
    else:
        assert api._pipeline._extractors[0]._decision_client is None
        assert service._decision_client is None


@pytest.mark.parametrize("span_count", [1, 2])
@pytest.mark.parametrize("failed_call", [1, 2])
@pytest.mark.parametrize("stage", ["decision", "generation"])
def test_extract_propagates_usage_failure_without_retry(monkeypatch, span_count, failed_call, stage):
    from extraction.tools import _PreparedSpan
    from providers.token_tracker import TokenTracker

    class FailSelectedWrite:
        calls = 0

        def record(self, event):
            self.calls += 1
            if self.calls == failed_call:
                raise OSError("usage store unavailable")

    steps = ["FINISH_EXTRACTION"] * 8 if stage == "decision" else ["FORM_MEMORY_OPERATION", "FINISH_EXTRACTION"] * 4
    decision, requests = scripted_decision(steps)
    store = FailSelectedWrite()
    tracker = decision.token_tracker if stage == "decision" else TokenTracker("generator")
    tracker.set_usage_store(store)

    class AccountedGenerator(Generator):
        def complete_with_tools(self, **kwargs):
            self.requests.append(("tools", kwargs))
            tracker.record_llm(10, 2)
            return [create_operation()]

    fs = memory_fs()
    registry = SchemaRegistry()
    generator = AccountedGenerator()
    extractor = Extractor(
        generator, mode="lazy", schema_registry=registry, fs=fs,
        uri_resolver=URIResolver(registry), decision_client=decision,
        dual_pass_enabled=True, max_workers=1,
    )
    prepared = [
        _PreparedSpan(_Span(i, i, categories=["preference"], message_ids=[f"m{i}"]), SOURCE, "")
        for i in range(span_count)
    ]
    monkeypatch.setattr(extractor, "_identify_spans", lambda *args: [p.span for p in prepared])
    monkeypatch.setattr(extractor, "_prepare_unique_spans", lambda *args: prepared)
    sleeps = []
    monkeypatch.setattr("extraction.tools.time.sleep", sleeps.append)
    with usage_scope("acme", "user:alice"), pytest.raises(UsagePersistenceError):
        extractor.extract([{"id": "m0", "role": "user", "content": SOURCE}], context())
    assert sleeps == []
    if span_count == 1:
        assert store.calls == failed_call
        if stage == "decision":
            assert generator.requests == []
            assert len(requests) == failed_call
        else:
            assert len(generator.requests) == failed_call
            assert len(requests) == failed_call * 2 - 1


def test_extract_reports_unsupported_generation_template(monkeypatch, tmp_path):
    import yaml

    from extraction.assembly_contract import UnsupportedAssemblyTemplateError
    from extraction.tools import _PreparedSpan

    manager = PromptManager()
    template = deepcopy(manager.load("extraction"))
    template.pop("assembly_purposes")
    for section in template.values():
        if isinstance(section, list):
            for block in section:
                if isinstance(block, dict):
                    block.pop("purposes", None)
    (tmp_path / "extraction.yaml").write_text(yaml.safe_dump(template))
    decision, requests = scripted_decision([])
    fs = memory_fs()
    registry = SchemaRegistry()
    extractor = Extractor(
        Generator(), mode="lazy", prompt_manager=PromptManager(tmp_path),
        schema_registry=registry, fs=fs, uri_resolver=URIResolver(registry),
        decision_client=decision,
    )
    prepared = _PreparedSpan(_Span(0, 0, categories=["preference"]), SOURCE, "")
    monkeypatch.setattr(extractor, "_identify_spans", lambda *args: [prepared.span])
    monkeypatch.setattr(extractor, "_prepare_unique_spans", lambda *args: [prepared])
    with pytest.raises(UnsupportedAssemblyTemplateError, match="memory_generation"):
        extractor.extract([{"role": "user", "content": SOURCE}], context())
    assert requests == []
