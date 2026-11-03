"""Exercise trace projections, durable writes, replay and common retrieval surfaces."""

import json
from copy import deepcopy
from unittest.mock import MagicMock

import pytest

from commit.outbox_store import OutboxStore
from core.models import RequestContext
from extraction.agent_memory import (
    FIELDS,
    agent_memory_registry,
    memory_units,
    project_trace,
    tool_memory_key,
)
from fs.agfs_adapter.agfs_trace_store import AGFSTraceEventStore
from index.outbox_worker import OutboxWorker
from providers.embedder.mock_embedder import MockEmbedder
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from server.agent_memory import process_trace_memory, trace_memory_event
from server.mcp_resources import external_to_internal_uri, internal_to_external_uri
from server.mcp_trace import message_events
from tests.unit.commit.test_memory_identity import MemoryFS
from tests.unit.fs.test_agfs_context_fs import _AtomicMemoryAGFSClient

CTX = RequestContext("acct", "alice", "codex", "session", "request")


class _MemoryAGFS(_AtomicMemoryAGFSClient):
    def read(self, path, offset=0, size=-1, stream=False):
        data = super().read(path)
        return data[offset:] if size < 0 else data[offset:offset + size]


class NodeFS(MemoryFS):
    def __init__(self, client):
        super().__init__()
        self.client = client

    def write_node(self, node, ctx):
        super().write_node(node, ctx)
        path = "/local/accounts/" + node.uri.removeprefix("ctx://")
        parts = path.strip("/").split("/")
        for n in range(1, len(parts) + 1):
            self.client.directories.add("/" + "/".join(parts[:n]))


MESSAGES = [
    {"role": "user", "content": "Check the request limit in the docs."},
    {
        "role": "assistant",
        "content": "I will inspect the API docs.",
        "reasoning_content": "The server limit controls batch size.",
        "tool_calls": [
            {"id": "call-1", "function": {"name": "read_file", "arguments": '{"path":"api.md"}'}}
        ],
    },
    {"role": "tool", "tool_call_id": "call-1", "content": "The batch limit is 100 items."},
    {"role": "assistant", "content": "Use batches of at most 100 and verify each response."},
]


class ExtractionLLM:
    def __init__(self):
        self.calls = []

    def complete_with_tools(self, prompt, tools, tool_choice):
        name = tools[0]["name"]
        self.calls.append((name, prompt))
        if name == "summarize_agent_memory":
            result = {
                "abstract": "API batches: 100 items maximum.",
                "overview": ("Batch limit: 100 items. Response check: inspect each batch outcome."
                             if tools[0]["input_schema"]["properties"]["overview"].get("minLength") else ""),
            }
        elif name == "review_agent_memory":
            proposals = json.loads(prompt.split("Proposals:\n")[1].split("\nCaptured evidence:")[0])
            category = prompt.split()[5]
            result = {"decisions": [{"id": p["id"], "keep": True,
                                    "reason": "Retains the documented limit and its applicable API.",
                                    "fields": {k: p["item"][k] for k in FIELDS[category]}} for p in proposals]}
        else:
            category = prompt.split()[2].rstrip(";")
            wire = json.loads(prompt.split("\nCaptured evidence:\n")[1])
            unit = {"key": "batch_limit", "title": "Batch upload limit",
                    "content": "The documented API accepts at most 100 items per batch. Split larger uploads into batches.",
                    "grounding": [{"source_id": wire[-1]["source_id"]}]}
            if category == "tool":
                properties = tools[0]["input_schema"]["properties"]["changes"]["items"]["properties"]
                change = {"routing_key": properties["routing_key"]["enum"][0],
                          "tool_name": properties["tool_name"]["enum"][0], "units": [unit]}
            else:
                unit["content"] = ("Use the documented server limit to choose batch size: the model identifies "
                                   "this limit as controlling request size for this API.")
                change = {"routing_key": "batch_planning", **unit}
            result = {"changes": [{**change, "corrections": [], "removals": []}]}
        return [{"tool": name, "input": result}]


class SameEmbedding(MockEmbedder):
    def embed_texts(self, texts):
        return [[1.0, 0.0] for _ in texts]


def finalized_trace(client, ctx=CTX):
    store = AGFSTraceEventStore(client=client, mount_prefix="/local", max_turn_events=2)
    events = message_events(MESSAGES) + [{"event_type": "turn_end", "status": "ok"}]
    for i, event in enumerate(events):
        result = store.insert(
            event_id=f"{ctx.agent_id}-{ctx.session_id}-{i}",
            account_id=ctx.account_id,
            owner_space=ctx.user_space_name(),
            agent_id=ctx.agent_id,
            session_id=ctx.session_id,
            hook_type=event["event_type"],
            content=event,
        )
    finalized = store.finalize_session(
        account_id=ctx.account_id,
        owner_space=ctx.user_space_name(),
        agent_id=ctx.agent_id,
        session_id=ctx.session_id,
        stop_event_index=result["event_index"],
    )
    return store, finalized


def test_projects_tool_evidence_and_model_content():
    events = message_events(MESSAGES)
    projected = project_trace(
        [{"ordinal": i + 1, "event": event} for i, event in enumerate(events)]
    )
    assert [event["event_type"] for event in projected["tool"]] == ["tool_start", "tool_success"]
    assert projected["tool"][1]["tool_response"] == "The batch limit is 100 items."
    assert projected["tool"][1]["tool_name"] == "read_file"
    assert projected["tool"][1]["tool_use_id"] == "call-1"
    assert [event["ordinal"] for event in projected["experience"]] == [2, 5]
    assert projected["experience"][0]["reasoning"] == "The server limit controls batch size."


@pytest.mark.parametrize(
    "agent", ["dsh", "openclaw", "hermes", "opencode", "codex", "claude-code", "mcp"]
)
def test_finalized_trace_writes_tools_and_experiences_and_replays(agent):
    ctx = RequestContext("acct", "alice", agent, "session", "request")
    client, llm = _MemoryAGFS(), ExtractionLLM()
    fs = NodeFS(client)
    trace_store, finalized = finalized_trace(client, ctx)
    assert finalized["stored_event_count"] == 2
    outbox = OutboxStore(client=client, fs=fs, mount_prefix="/local")
    event = trace_memory_event(finalized["uri"], ctx)
    outbox.register_once(event, ctx)

    def handler(job):
        process_trace_memory(job, trace_store=trace_store, outbox=outbox, fs=fs, llm=llm)

    index = InMemoryVectorIndex()
    worker = OutboxWorker(index, SameEmbedding(), fs=fs, agent_memory_handler=handler)
    assert worker.process_event(event).success
    assert set(node.category for node in fs.nodes.values()) == set(FIELDS)
    for node in fs.nodes.values():
        assert node.owner_space == f"agent:{agent}"
        assert node.context_type == "MEMORY"
        assert len({node.abstract, node.overview, node.content}) == 3
        assert node.overview == ""
        assert "Batch upload limit" in node.content
        assert node.metadata["agent_memory_content_version"] == 3
        units = memory_units(node.category, node.metadata["structured_fields"])
        assert all(set(unit) == {"key", "title", "content", "grounding"} for unit in units)
        assert node.metadata["provenance_ids"] == [finalized["uri"]]
        assert all(g["uri"].startswith(finalized["uri"] + "#event-")
                   for unit in node.metadata["agent_memory"]["units"].values() for g in unit["grounding"])
        public_uri = internal_to_external_uri(node.uri)
        assert external_to_internal_uri(public_uri, ctx) == node.uri
    snapshots = deepcopy(fs.nodes)
    handler(event)
    assert fs.nodes == snapshots
    assert event.payload["cursor"] == finalized["event_count"]
    outbox.mark_done(event, event.uri)
    outbox.register_once(trace_memory_event(finalized["uri"], ctx), ctx)
    assert {queued.event_type for _, queued in outbox.list_pending("acct")} == {"UPSERT_CONTEXT"}
    result = worker.run_once(outbox, ["acct"])
    assert result["succeeded"] == 2
    from core.models import RetrievalConfig
    from tests.unit.retrieval.test_pipeline import _build_pipeline

    pipeline = _build_pipeline(
        index,
        SameEmbedding(),
        RetrievalConfig(expand_hierarchy=False, include_agent_memories=True),
        with_expand=False,
    )
    retrieved = pipeline.run("batch limit", ctx, categories=list(FIELDS), top_k=10)
    assert {hit.category for hit in retrieved.hits} == set(FIELDS)


def test_resumes_reviewed_proposals_after_restart():
    from dataclasses import asdict

    from extraction.agent_memory import AgentMemoryExtractor

    client = _MemoryAGFS()
    fs = NodeFS(client)
    trace_store, finalized = finalized_trace(client)
    outbox = OutboxStore(client=client, fs=fs, mount_prefix="/local")
    event = trace_memory_event(finalized["uri"], CTX)
    outbox.register_once(event, CTX)
    page = trace_store.read_skill_trace_event_page(
        account_id="acct", evidence_uri=finalized["uri"], after_ordinal=0, limit=32
    )
    extractor = AgentMemoryExtractor(fs, ExtractionLLM())
    candidates = extractor.extract_candidates(finalized["uri"], page["events"], CTX)
    event.payload.update(candidates=[asdict(c) for c in candidates],
                         value_reviews=extractor.value_reviews, cursor=page["next_ordinal"])
    outbox.save_payload(event)
    restored = outbox.list_pending("acct")[0][1]
    llm = ExtractionLLM()
    process_trace_memory(restored, trace_store=trace_store, outbox=outbox, fs=fs, llm=llm)
    assert restored.payload["cursor"] == finalized["event_count"]
    assert set(node.category for node in fs.nodes.values()) == set(FIELDS)
    assert [name for name, _ in llm.calls] == ["summarize_agent_memory"] * 2


def test_finalize_enqueues_agent_memory_and_existing_skill_candidates(monkeypatch):
    from providers.unified_config import OgMemConfig
    from server.memory_service import MemoryService

    monkeypatch.setattr(MemoryService, "_start_skill_sedimentation_worker", lambda self: None)
    service = MemoryService(config=OgMemConfig(
        agent_memory_enabled=True, skill_sedimentation_enabled=True, workers=1,
    ))
    store, outbox = MagicMock(), MagicMock()
    store.finalize_session.return_value = {
        "ok": True,
        "uri": "ctx://acct/traces/session/turns/turn1",
        "turn_id": "turn1",
    }
    store.enqueue_skill_candidate_from_trace.return_value = {"candidate_id": "skill-candidate-1"}
    service._get_trace_event_store = lambda: store
    service._get_context_fs = lambda: object()
    service._get_outbox_store = lambda fs: outbox
    service._start_outbox_worker = lambda: None
    result = service.finalize_session_trace(
        {"_ctx": CTX, "sessionId": "session", "stopEventIndex": 6}
    )
    event, context = outbox.register_once.call_args.args
    assert context == CTX
    assert result["agent_memory_event_id"] == event.event_id
    assert event.payload["context"]["user_id"] == "alice"
    assert result["skill_candidate_id"] == "skill-candidate-1"
    store.enqueue_skill_candidate_from_trace.assert_called_once_with(
        account_id="acct", agent_id="codex", session_id="session", turn_id="turn1",
        trace_uri=event.payload["trace_uri"],
    )


def test_anthropic_blocks_preserve_exposed_thinking_and_tool_results():
    events = message_events(
        [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Check the limit first."},
                    {"type": "tool_use", "id": "c1", "name": "read", "input": {"path": "api.md"}},
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "Limit: 100."}],
            },
        ]
    )
    assert events[1]["tool_input"] == {"path": "api.md"}
    assert events[0]["reasoning"] == "Check the limit first."
    assert events[2]["tool_name"] == "read"
    assert events[2]["tool_response"] == "Limit: 100."


def test_procedure_queries_share_existing_skills_and_agent_memories():
    from core.models import RetrievalConfig
    from retrieval.query_planner import QueryPlanner

    planner = QueryPlanner(RetrievalConfig(include_agent_memories=True))
    queries = planner.plan("how to upload a batch", CTX)
    assert [query.context_type for query in queries] == ["SKILL", "MEMORY"]
    queries = planner.plan("how to upload a batch", CTX, categories=["experience"])
    assert [query.context_type for query in queries] == ["MEMORY"]
    queries = planner.plan("how to upload a batch", CTX, categories=["skill"])
    assert [(query.context_type, query.categories) for query in queries] == [("SKILL", ["skill"])]
    queries = planner.plan(
        "how to upload a batch", CTX, categories=["profile", "experience", "skill"]
    )
    assert [(query.context_type, query.categories) for query in queries] == [
        ("MEMORY", ["profile", "experience"]),
        ("SKILL", ["skill"]),
    ]


def test_config_enables_agent_memory_from_yaml_and_environment(monkeypatch):
    from providers import unified_config as uc

    monkeypatch.setattr(uc, "_load_yaml", lambda path: {"agent_memory": {"enabled": True}})
    monkeypatch.delenv("OGMEM_AGENT_MEMORY_ENABLED", raising=False)
    assert uc.OgMemConfig.load().agent_memory_enabled is True
    monkeypatch.setattr(uc, "_load_yaml", lambda path: {})
    monkeypatch.setenv("OGMEM_AGENT_MEMORY_ENABLED", "true")
    assert uc.OgMemConfig.load().agent_memory_enabled is True


def test_updates_existing_nodes_with_a_second_turn():
    client = _MemoryAGFS()
    fs, llm = NodeFS(client), ExtractionLLM()
    outbox = OutboxStore(client=client, fs=fs, mount_prefix="/local")
    turns = []
    for session in ("first", "second"):
        ctx = RequestContext("acct", "alice", "codex", session, "request")
        store, finalized = finalized_trace(client, ctx)
        turns.append(finalized["uri"])
        event = trace_memory_event(finalized["uri"], ctx)
        outbox.register_once(event, ctx)
        process_trace_memory(event, trace_store=store, outbox=outbox, fs=fs, llm=llm)
    assert len(fs.nodes) == 2
    for node in fs.nodes.values():
        assert node.metadata["version"] == 2
        assert len(node.metadata["memory_write_receipts"]) == 2
        unit = memory_units(node.category, node.metadata["structured_fields"])[0]
        assert len(unit["grounding"]) == 2
        assert unit["grounding"] == node.metadata["agent_memory"]["units"][unit["key"]]["grounding"]
        for turn in turns:
            assert any(ref.startswith(turn) for ref in node.metadata["provenance_ids"])
            assert any(source["uri"].startswith(turn) for source in unit["grounding"])


def test_enabled_service_constructs_the_shared_write_api():
    from providers.unified_config import OgMemConfig
    from server.memory_service import MemoryService

    service = MemoryService(config=OgMemConfig(agent_memory_enabled=True))
    client = _MemoryAGFS()
    fs = NodeFS(client)
    service._get_context_fs = lambda: fs
    service._get_outbox_store = lambda fs: OutboxStore(client, fs)
    service._get_shared_vector_index = lambda: InMemoryVectorIndex()
    service._get_shared_embedder = lambda: SameEmbedding()
    service.get_llm = lambda: ExtractionLLM()
    service._start_outbox_worker = lambda: None
    api = service.get_write_api()
    assert api._fs is fs
    assert {schema.owner_scope for schema in api._schema_registry.list_enabled()} == {"user"}


def test_tool_units_keep_typed_history_and_rebuild_all_retrieval_layers():
    from dataclasses import replace

    from commit.agent_memory import AgentMemoryArchiveBuilder, AgentMemoryPolicy
    from core.uri_resolver import URIResolver
    from extraction.agent_memory import AgentMemoryExtractor, render_agent_memory
    from index.index_record_builder import build_index_records

    client, llm = _MemoryAGFS(), ExtractionLLM()
    fs = NodeFS(client)
    store, turn = finalized_trace(client)
    page = store.read_skill_trace_event_page(
        account_id="acct", evidence_uri=turn["uri"], after_ordinal=0, limit=32,
    )
    extractor = AgentMemoryExtractor(fs, llm)
    candidate = next(c for c in extractor.extract_candidates(turn["uri"], page["events"], CTX)
                     if c.category == "tool")
    policy = AgentMemoryPolicy(fs, URIResolver(agent_memory_registry()))
    builder = AgentMemoryArchiveBuilder(llm)
    original = builder.build(candidate, policy.plan(candidate, CTX), CTX)
    fs.write_node(original, CTX)

    second = deepcopy(candidate)
    first_unit = deepcopy(second.structured_fields["units"][0])
    source_id = next(ident for ident in second.memory_context["sources"]
                     if ident != first_unit["grounding"][0]["source_id"])
    source = second.memory_context["sources"][source_id]
    source["locator"] = "https://example.org/api/response-check"
    added = {**deepcopy(first_unit), "key": "response_check",
             "title": "Inspect batch responses", "content": "Inspect the response for each API batch.",
             "grounding": [{"source_id": source_id}]}
    second.structured_fields["units"] = [added]
    second.memory_context.update(
        operation_id="second", base_units={"batch_limit": 1},
        grounding={"response_check": [{**source, "source_id": source_id}]},
    )
    updated = builder.build(second, policy.plan(second, CTX), CTX)
    fs.write_node(updated, CTX)
    assert [item["key"] for item in updated.metadata["structured_fields"]["units"]] == [
        "batch_limit", "response_check",
    ]
    records = build_index_records(updated)
    assert [r.level for r in records] == [0, 1, 2, 2]
    assert [r.metadata["agent_memory_unit_key"] for r in records if r.level == 2] == [
        "batch_limit", "response_check",
    ]
    active_grounding = [g for item in updated.metadata["structured_fields"]["units"] for g in item["grounding"]]
    assert updated.content == render_agent_memory("tool", updated.metadata["structured_fields"], grounding=active_grounding)
    assert updated.abstract == "API batches: 100 items maximum."
    assert "Response check" in updated.overview
    index = InMemoryVectorIndex()
    outbox = OutboxStore(client, fs, mount_prefix="/local")
    worker = OutboxWorker(index, SameEmbedding(), fs=fs)
    first_event = outbox.build_write_event(updated)
    assert worker.process_event(first_event).success
    assert index.count({"account_id": "acct", "owner_space": "agent:codex"}) == 4

    corrected = deepcopy(second)
    corrected.structured_fields["units"][0]["content"] = "Inspect the status and item count for each batch."
    corrected.memory_context.update(
        operation_id="third", base_units={"batch_limit": 1, "response_check": 1},
        corrections=[{"key": "response_check", "reason": "The current evidence specifies both status and count."}],
    )
    revision = builder.build(corrected, policy.plan(corrected, CTX), CTX)
    fs.write_node(revision, CTX)
    assert revision.metadata["agent_memory"]["revisions"][-1]["action"] == "correction"
    assert revision.metadata["agent_memory"]["units"]["response_check"]["revision"] == 2

    withdrawn = replace(corrected, structured_fields={**corrected.structured_fields, "units": []},
                        memory_context={**corrected.memory_context, "operation_id": "fourth",
                            "base_units": {"batch_limit": 1, "response_check": 2}, "corrections": [],
                            "removals": [{"key": "response_check", "reason": "The updated contract replaces this check.",
                                          "grounding": added["grounding"]}]})
    final = builder.build(withdrawn, policy.plan(withdrawn, CTX), CTX)
    fs.write_node(final, CTX)
    assert [i["key"] for i in final.metadata["structured_fields"]["units"]] == ["batch_limit"]
    assert final.metadata["agent_memory"]["revisions"][-1]["action"] == "remove"
    retained = final.metadata["structured_fields"]["units"][0]
    assert final.metadata["derived_from"] == sorted({g["uri"] for g in retained["grounding"]})
    assert final.content == render_agent_memory("tool", final.metadata["structured_fields"], grounding=retained["grounding"])
    assert final.metadata["agent_memory"]["units"]["response_check"]["grounding"][0]["locator"] == source["locator"]
    assert worker.process_event(first_event).success
    assert worker.process_event(outbox.build_write_event(final)).success
    hits = index.search_by_vector([1.0, 0.0], {"account_id": "acct", "owner_space": "agent:codex"}, 10)
    assert {h.metadata.get("agent_memory_unit_key") for h in hits if h.level == 2} == {"batch_limit"}


def test_review_and_summary_calls_follow_source_lanes():
    from extraction.agent_memory import AgentMemoryExtractor

    llm, client = ExtractionLLM(), _MemoryAGFS()
    fs = NodeFS(client)
    store, turn = finalized_trace(client)
    page = store.read_skill_trace_event_page(account_id="acct", evidence_uri=turn["uri"], after_ordinal=0, limit=32)
    extractor = AgentMemoryExtractor(fs, llm)
    candidates = extractor.extract_candidates(turn["uri"], page["events"], CTX)
    assert [name for name, _ in llm.calls] == [
        "maintain_agent_memory", "review_agent_memory",
        "maintain_agent_memory", "review_agent_memory",
    ]
    assert [r["memory_type"] for r in extractor.value_reviews] == ["tool", "experience"]
    for candidate in candidates:
        lane = {"tool": "tool", "experience": "experience"}.get(candidate.category)
        if lane:
            assert {g["lane"] for group in candidate.memory_context["grounding"].values() for g in group} == {lane}
        assert memory_units(candidate.category, candidate.structured_fields)[0]["title"] == "Batch upload limit"
    model_prompt = next(prompt for name, prompt in llm.calls
                        if name == "maintain_agent_memory" and prompt.startswith("Extract only experience;"))
    sources = json.loads(model_prompt.split("\nCaptured evidence:\n")[1])
    for source in sources:
        content = json.loads("".join(line for _, line in source["lines"]))
        assert set(content) <= {"assistant_message", "reasoning", "assistant_messages"}


def test_review_distills_new_tool_fields_and_keeps_source_binding():
    from extraction.agent_memory import AgentMemoryExtractor

    class ReviewedLLM(ExtractionLLM):
        def complete_with_tools(self, prompt, tools, tool_choice):
            response = super().complete_with_tools(prompt, tools, tool_choice)
            if tools[0]["name"] == "review_agent_memory" and prompt.split()[5] == "tool":
                response[0]["input"]["decisions"][0]["fields"]["content"] = (
                    "The documented API batch limit is 100 items."
                )
            return response

    client = _MemoryAGFS()
    fs = NodeFS(client)
    store, turn = finalized_trace(client)
    page = store.read_skill_trace_event_page(account_id="acct", evidence_uri=turn["uri"], after_ordinal=0, limit=32)
    extractor = AgentMemoryExtractor(fs, ReviewedLLM())
    candidate = next(c for c in extractor.extract_candidates(turn["uri"], page["events"], CTX) if c.category == "tool")
    item = candidate.structured_fields["units"][0]
    assert item["content"] == "The documented API batch limit is 100 items."
    assert candidate.memory_context["grounding"][item["key"]][0]["source_id"] == item["grounding"][0]["source_id"]


def test_repeated_calls_group_under_each_tool_identity():
    from extraction.agent_memory import AgentMemoryExtractor

    raw = message_events(MESSAGES)
    raw.extend([
        {"event_type": "tool_success", "tool_name": "read_file", "tool_use_id": "again",
         "tool_response": "The same API batch limit is 100."},
        {"event_type": "tool_success", "tool_name": "inspect_api", "tool_use_id": "inspect",
         "tool_response": "The API batch limit is 100."},
    ])
    events = [{"ordinal": i + 1, "event": event} for i, event in enumerate(raw)]
    extractor = AgentMemoryExtractor(NodeFS(_MemoryAGFS()), ExtractionLLM())
    candidates = extractor.extract_candidates("ctx://acct/traces/session/turns/multi", events, CTX)
    tools = [c for c in candidates if c.category == "tool"]
    assert {c.routing_key for c in tools} == {tool_memory_key(name) for name in ("read_file", "inspect_api")}
    for candidate in tools:
        assert tool_memory_key(candidate.structured_fields["tool_name"]) == candidate.routing_key
        assert candidate.structured_fields["units"][0]["title"] == "Batch upload limit"
        assert {g["tool_name"] for group in candidate.memory_context["grounding"].values() for g in group} == {candidate.structured_fields["tool_name"]}


def test_legacy_node_converts_on_a_supported_unit_update():
    from commit.agent_memory import AgentMemoryArchiveBuilder, AgentMemoryPolicy
    from core.uri_resolver import URIResolver
    from extraction.agent_memory import AgentMemoryExtractor, node_memory_fields

    client, llm = _MemoryAGFS(), ExtractionLLM()
    fs = NodeFS(client)
    store, turn = finalized_trace(client)
    page = store.read_skill_trace_event_page(account_id="acct", evidence_uri=turn["uri"], after_ordinal=0, limit=32)
    candidate = next(c for c in AgentMemoryExtractor(fs, llm).extract_candidates(turn["uri"], page["events"], CTX)
                     if c.category == "tool")
    policy, builder = AgentMemoryPolicy(fs, URIResolver(agent_memory_registry())), AgentMemoryArchiveBuilder(llm)
    original = builder.build(candidate, policy.plan(candidate, CTX), CTX)
    unit = original.metadata["structured_fields"]["units"][0]
    original.metadata["structured_fields"] = {"title": unit["title"], "scope": "Tool: read_file",
        "items": [{"key": unit["key"], "statement": unit["content"]}]}
    original.metadata["agent_memory_content_version"] = 1
    fs.write_node(original, CTX)
    assert node_memory_fields(original)["units"][0] == unit

    update = deepcopy(candidate)
    added = {**deepcopy(update.structured_fields["units"][0]), "key": "response_check",
             "title": "Response verification", "content": "Check each batch response."}
    update.structured_fields["units"] = [added]
    update.memory_context.update(operation_id="legacy-upgrade", base_units={"batch_limit": 1},
                                 grounding={"response_check": candidate.memory_context["grounding"]["batch_limit"]})
    upgraded = builder.build(update, policy.plan(update, CTX), CTX)
    fs.write_node(upgraded, CTX)
    assert upgraded.metadata["agent_memory_content_version"] == 3
    assert set(upgraded.metadata["structured_fields"]) == {"tool_name", "units"}
    assert upgraded.metadata["structured_fields"]["units"][0] == unit
    assert policy.plan(update, CTX).action == "skip"


def test_legacy_checkpoint_reextracts_new_nodes_and_replays():
    client, llm = _MemoryAGFS(), ExtractionLLM()
    fs = NodeFS(client)
    store, turn = finalized_trace(client)
    outbox = OutboxStore(client=client, fs=fs, mount_prefix="/local")
    event = trace_memory_event(turn["uri"], CTX)
    outbox.register_once(event, CTX)
    event.payload["candidates"] = [{"structured_fields": {
        "title": "Batch upload limit", "scope": "Tool: read_file",
        "items": [{"key": "batch_limit", "statement": "The batch limit is 100."}]}}]
    outbox.save_payload(event)
    restored = outbox.list_pending("acct")[0][1]
    process_trace_memory(restored, trace_store=store, outbox=outbox, fs=fs, llm=llm)
    assert {node.category for node in fs.nodes.values()} == {"tool", "experience"}
    assert all(node.metadata["agent_memory_content_version"] == 3 for node in fs.nodes.values())
    assert [name for name, _ in llm.calls].count("maintain_agent_memory") == 2
    snapshot = deepcopy(fs.nodes)
    process_trace_memory(restored, trace_store=store, outbox=outbox, fs=fs, llm=llm)
    assert fs.nodes == snapshot


def test_long_single_conclusion_derives_navigation_from_full_body():
    from extraction.agent_memory import summarize_agent_memory

    fields = {"key": "loader_version", "title": "Loader version constraint",
              "content": "Use src/loader.py:load_v2 for protocol version 2. " * 40, "grounding": []}
    class SummaryLLM(ExtractionLLM):
        def complete_with_tools(self, prompt, tools, tool_choice):
            self.calls.append((tools[0]["name"], prompt))
            assert tools[0]["input_schema"]["properties"]["overview"]["minLength"] == 1
            return [{"tool": "summarize_agent_memory", "input": {
                "abstract": "Protocol v2 requires the v2 loader.",
                "overview": "Loader constraint: src/loader.py:load_v2 applies to protocol version 2."}}]

    llm = SummaryLLM()
    abstract, overview = summarize_agent_memory("experience", fields, llm)
    assert abstract == "Protocol v2 requires the v2 loader."
    assert overview == "Loader constraint: src/loader.py:load_v2 applies to protocol version 2."
    assert fields["content"] in llm.calls[0][1]


def test_long_multibyte_unit_is_available_for_maintenance():
    from core.models import ContextNode
    from extraction.agent_memory import AgentMemoryExtractor

    body = ("版本 2 的批量接口限制为 100 项；实现位于 src/api.py:upload。" * 200)[:8000]
    unit = {"key": "batch_limit", "title": "批量接口约束", "content": body,
            "grounding": [{"uri": "ctx://acct/traces/session/turns/first#event-1"}]}
    uri = "ctx://acct/agents/codex/memories/tools/read_file.md"
    node = ContextNode(uri, "MEMORY", "tool", 4, CTX.agent_space_name(), "批量接口约束", "", body, {
        "routing_key": "read_file", "structured_fields": {"tool_name": "read_file", "units": [unit]},
        "agent_memory": {"version": 1, "units": {"batch_limit": {"revision": 1, "grounding": unit["grounding"]}}}})
    fs = NodeFS(_MemoryAGFS())
    fs.write_node(node, CTX)
    extractor, existing = AgentMemoryExtractor(fs, ExtractionLLM()), {}
    extractor._refresh_maintenance_context(uri, {"batch_limit"}, existing, CTX, "tool")
    supplied = extractor._known_topic(existing[uri])
    assert len(body) == 8000
    assert supplied["fields"]["units"][0]["content"] == body
    assert supplied["unit_revisions"] == {"batch_limit": 1}
