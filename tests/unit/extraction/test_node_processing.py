"""Source-grounded node updates through the public write pipeline."""

from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import json
import pytest

from core.models import RequestContext
from core.errors import LLMOutputTruncatedError
from extraction.node_processing import NodeProcessor
from service.api import MemoryWriteAPI
from tests.unit.commit.test_memory_identity import MemoryFS, memory


def card(key="necklace", **kwargs):
    c = memory(key, object_name=key, content="# Necklace\n\n## Identity\n- Gift from Caroline's grandma.")
    return replace(c, abstract="Caroline's necklace, a gift from her grandmother.",
                   overview="- A necklace given by Caroline's grandmother.", **kwargs)


def operation(ids, *, target="", action="create", fact="Gift from Caroline's grandma."):
    return {"draft_ids": ids, "action": action, "target": target,
            "routing_key": "necklace", "object_name": "necklace",
            "abstract": "Caroline's necklace from her grandmother, with a heart charm.",
            "overview": ["Caroline owns the necklace.", "Her grandmother gave it to her."],
            "edits": [], "sections": [{"heading": "Features", "facts": [fact]}]}


class NodeLLM:
    def __init__(self, decide):
        self.decide, self.calls = decide, []

    def complete_with_tools(self, **kwargs):
        self.calls.append(kwargs)
        return [{"tool": "process_memory_nodes", "input": {"operations": self.decide(kwargs)}}]


def setup(decide, mode="eager"):
    from core.uri_resolver import URIResolver
    from extraction.schemas.registry import SchemaRegistry
    ctx = RequestContext(account_id="node-test", user_id="reader", agent_id="locomo", session_id="s", trace_id="t")
    fs = MemoryFS()
    llm = NodeLLM(decide)
    registry = SchemaRegistry()
    api = MemoryWriteAPI(fs, llm, directory_summary_enabled=False, card_format_enabled=True,
                         node_processing_enabled=True, extraction_mode=mode,
                         schema_registry=registry, uri_resolver=URIResolver(registry))
    return ctx, fs, llm, api


@pytest.mark.parametrize("users,with_event", [(("user69", "user80"), False), (("user70", "user33"), True)])
def test_fallback_locks_complete_concurrent_commits(monkeypatch, users, with_event):
    import threading
    from contextlib import contextmanager
    import commit.identity_resolver as identity

    rendezvous = threading.Barrier(2)
    extracting = threading.Barrier(2)
    local = threading.local()
    original_lock = identity.MemoryIdentityResolver._lock

    @contextmanager
    def coordinated_lock(self, directory, ctx):
        with original_lock(self, directory, ctx):
            if not getattr(local, "entered", False):
                local.entered = True
                try:
                    rendezvous.wait(.2)
                except threading.BrokenBarrierError:
                    pass
            yield

    monkeypatch.setattr(identity.MemoryIdentityResolver, "_lock", coordinated_lock)
    fs = MemoryFS()
    results = []

    def extract(*args, **kwargs):
        if with_event:
            extracting.wait(2)
            return [card(), card("inspection", category="event", content="The technician inspected the necklace.")]
        return [card()]

    def commit(user):
        ctx = RequestContext("acct", user, "agent", user, "trace")
        api = MemoryWriteAPI(fs, NodeLLM(lambda _: [operation([0])]),
                            node_processing_enabled=True, card_format_enabled=True,
                            directory_summary_enabled=False)
        api._pipeline.extract = extract
        results.append(api.commit_session([], ctx))

    threads = [threading.Thread(target=commit, args=(user,), daemon=True)
               for user in users]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(3)
    assert len(results) == 2
    assert len(fs.nodes) == 2 * (1 + with_event)
    assert all(result["writes_completed"] == 1 + with_event for result in results)
    assert all(node.metadata["version"] == 1 for node in fs.nodes.values())


def test_same_object_participants_grouped_and_alias_saved():
    ctx, fs, llm, api = setup(lambda _: [operation([0, 1])])
    drafts = [card("cross_and_heart", who="Caroline; Melanie; Grandma"),
              card("grandmas_necklace", who="Caroline; Grandma; Melanie")]
    api._pipeline.extract = lambda *a, **k: drafts
    result = api.commit_session([{"role": "user", "content": "Grandma gave Caroline the necklace."}], ctx)
    node = next(iter(fs.nodes.values()))
    assert result["writes_completed"] == 1
    assert len(fs.nodes) == 1
    assert set(node.metadata["object_name_aliases"]) >= {"cross_and_heart", "grandmas_necklace"}
    assert llm.calls[0]["temperature"] == 0
    resolver = api._writer._identity_resolver
    assert resolver._names_match(node, drafts[1])


def test_node_rendering_preserves_fact_prefixes():
    facts = ["-5°C is the minimum temperature.", "#123 identifies the part.", "--safe-mode is required."]
    op = operation([0])
    op["sections"][0]["facts"] = facts
    ctx, fs, llm, api = setup(lambda _: [op])
    api._pipeline.extract = lambda *a, **kw: [card()]
    api.commit_session([], ctx)
    stored = next(iter(fs.nodes.values()))
    for fact in facts:
        assert "- " + fact in stored.content


@pytest.mark.parametrize("category,holders", [("entity", (None, None)), ("preference", (None, "user"))])
def test_equivalent_boundaries_commit_all_facts_once(category, holders):
    ctx, fs, llm, api = setup(lambda _: [])
    original = replace(card(), category=category, scope_ref="project_a", who=None)
    if category == "preference":
        original = replace(original, content="# Necklace\n\n- Prefers wearing the necklace.")
    uri = api.write_memory(original, ctx)["target_uri"]
    op = operation([0, 1], target=uri, action="update")
    op["sections"][0]["facts"] = ["First added fact.", "Second added fact."]
    llm.decide = lambda _: [op]
    api._pipeline.extract = lambda *a, **kw: [
        replace(original, scope_ref="Project-A", who=holders[0]),
        replace(original, scope_ref="project_a", who=holders[1]),
    ]
    result = api.commit_session([], ctx)
    stored = fs.read_node(uri, ctx)
    assert len(llm.calls) == result["writes_completed"] == 1
    assert stored.metadata["version"] == 2
    assert "First added fact." in stored.content and "Second added fact." in stored.content


@pytest.mark.parametrize("entity_state", ["absent", "disabled", "incompatible"])
def test_preference_only_registry_processes_its_configured_topic(tmp_path, entity_state):
    from extraction.schemas.registry import SchemaRegistry
    from core.uri_resolver import URIResolver
    ctx, fs, llm, _ = setup(lambda _: [operation([0])])
    defaults = SchemaRegistry()
    registry = SchemaRegistry(str(tmp_path))
    registry.register(defaults.get("preference"))
    if entity_state != "absent":
        registry.register(replace(defaults.get("entity"), enabled=entity_state != "disabled",
                                  version="99.0" if entity_state == "incompatible" else "1.0",
                                  directory="{{ missing_owner.required }}"))
    api = MemoryWriteAPI(fs, llm, schema_registry=registry, uri_resolver=URIResolver(registry),
                        node_processing_enabled=True, card_format_enabled=True, directory_summary_enabled=False)
    api._pipeline.extract = lambda *a, **kw: [replace(card(), category="preference")]
    result = api.commit_session([], ctx)
    assert result["writes_completed"] == 1
    assert next(iter(fs.nodes.values())).category == "preference"


def test_update_retains_old_facts_and_replaces_whole_summary():
    ctx, fs, llm, api = setup(lambda _: [])
    uri = api.write_memory(card(), ctx)["target_uri"]
    llm.decide = lambda _: [operation([0], target=uri, action="update", fact="It has a heart charm.")]
    api._pipeline.extract = lambda *a, **k: [card("grandmas_necklace", who="Caroline; Grandma")]
    api.commit_session([{"role": "user", "content": "The necklace has a heart charm."}], ctx)
    node = fs.read_node(uri, ctx)
    assert "Gift from Caroline's grandma." in node.content
    assert "It has a heart charm." in node.content
    assert "heart charm" in node.abstract
    assert len(llm.calls) == 1
    assert len(fs.nodes) == 1


def test_catalog_beyond_sixteen_recalls_matching_object():
    ctx, fs, llm, api = setup(lambda _: [])
    for i in range(20):
        api.write_memory(card(f"object_{i}"), ctx)
    uri = api.write_memory(card("grandmas_necklace"), ctx)["target_uri"]
    llm.decide = lambda _: [operation([0], target=uri, action="update", fact="A treasured family gift.")]
    drafts = [card("grandmas_necklace")]
    result = api._node_processor().process(drafts, [{"role": "user", "content": "Grandma's necklace is treasured."}], ctx)
    assert result[0].target_uri == uri
    assert len(fs.nodes) == 21
    assert uri in llm.calls[0]["prompt"]


def node_payload(call):
    prompt = call["prompt"]
    return json.JSONDecoder().raw_decode(prompt[prompt.index('\n{"source":') + 1:])[0]


@pytest.mark.parametrize("category", ["entity", "preference"])
@pytest.mark.parametrize("action", ["update", "edit", "noop"])
@pytest.mark.parametrize("budget", ["input", "output"])
def test_split_new_identity_commits_composed_card_once(category, action, budget):
    ctx, fs, llm, api = setup(lambda _: [])
    drafts = [card(f"alias_{i}", category=category, provenance_ids=[f"source:{i}"],
                   content="# Item\n\n## Details\n- " + "Source detail. " * (2300 if budget == "input" else 1)
                   + f"\n- Fact {i}.")
              for i in range(4)]
    api._pipeline.extract = lambda *a, **kw: drafts

    def decide(call):
        payload = node_payload(call)
        if budget == "output" and len(payload["drafts"]) > 2:
            assert len(call["prompt"]) < 110000
            raise LLMOutputTruncatedError("finish_reason=length")
        target = payload["nodes"][0]["uri"] if payload["nodes"] else ""
        op = operation(list(range(len(payload["drafts"]))), target=target,
                       action="update" if target else "create")
        op["sections"][0]["facts"] = [d["content"].splitlines()[-1][2:] for d in payload["drafts"]]
        if target:
            assert "Fact 0." in payload["nodes"][0]["content"]
            if action == "edit":
                op["edits"] = [{"search": "Fact 0.", "replacement": "Verified fact 0."}]
            elif action == "noop":
                op.update(action="noop", sections=[])
        else:
            key = payload["drafts"][0]["routing_key"]
            op.update(routing_key=key, object_name=key)
        return [op]

    llm.decide = decide
    result = api.commit_session([], ctx)
    stored = next(iter(fs.nodes.values()))
    assert len(llm.calls) == (2 if budget == "input" else 3)
    assert len(fs.nodes) == result["writes_completed"] == 1
    assert stored.metadata["version"] == 1
    assert stored.uri.endswith("/alias_0")
    expected = ["Fact 0.", "Fact 1."] if action == "noop" else [f"Fact {i}." for i in range(4)]
    if action == "edit":
        expected[0] = "Verified fact 0."
    assert all(fact in stored.content for fact in expected)
    assert set(stored.metadata["provenance_ids"]) == {f"source:{i}" for i in range(4)}
    for draft in drafts:
        assert api._writer._identity_resolver._names_match(stored, draft)


@pytest.mark.parametrize("binding", ["create", "name", "uri"])
def test_large_optional_history_leaves_room_for_complete_target(binding):
    ctx, fs, llm, api = setup(lambda _: [])
    for i in range(12):
        api.write_memory(card(f"other_{i}", content="# Other\n\n## History\n- " + "Archived detail. " * 1000), ctx)
    original = card(content="# Necklace\n\n## Identity\n- Grandmother's gift.\n- It has one charm.")
    uri = api.write_memory(original, ctx)["target_uri"] if binding != "create" else ""
    draft = card()
    if binding == "uri":
        draft = replace(draft, routing_key="unmatched_alias", object_name="another name",
                        target_uri=uri, expected_version=1)

    def decide(call):
        payload = node_payload(call)
        if uri:
            target = next(n for n in payload["nodes"] if n["uri"] == uri)
            assert target["content"] == original.content
            assert target["version"] == 1
        return [operation([0], target=uri, action="update" if uri else "create", fact="It has a heart charm.")]

    llm.decide = decide
    api._pipeline.extract = lambda *a, **kw: [draft]
    result = api.commit_session([], ctx)
    assert result["writes_completed"] == 1
    assert len(llm.calls) == 1
    assert len(llm.calls[0]["prompt"]) <= 110000
    saved = fs.read_node(result["plans"][0]["target_uri"], ctx)
    assert "heart charm" in saved.content


@pytest.mark.parametrize("last_action", ["append", "edit", "noop"])
@pytest.mark.parametrize("budget", ["input", "output"])
def test_split_drafts_update_one_shared_snapshot(last_action, budget):
    ctx, fs, llm, api = setup(lambda _: [])
    original = card(content="# Necklace\n\n## History\n- " + "Archived attribute. " * 1000)
    uri = api.write_memory(original, ctx)["target_uri"]
    drafts = [replace(original, routing_key=f"alias_{i}", target_uri=uri, expected_version=1,
                      provenance_ids=[f"source:{i}"], verified_object_aliases=[f"name {i}"],
                      content=original.content + "\n- " + "Current detail. " * (450 if budget == "input" else 1)
                      + f"\n- New fact {i % 2 if last_action == 'noop' else i}.")
              for i in range(4)]

    def decide(call):
        payload = node_payload(call)
        if budget == "output" and len(payload["drafts"]) > 2:
            assert len(call["prompt"]) < 110000
            raise LLMOutputTruncatedError("finish_reason=length")
        target = next(n for n in payload["nodes"] if n["uri"] == uri)
        assert target["version"] == 1
        op = operation(list(range(len(payload["drafts"]))), target=uri, action="update")
        op["sections"][0]["facts"] = [d["content"].splitlines()[-1][2:] for d in payload["drafts"]]
        if len(llm.calls) == (2 if budget == "input" else 3):
            assert "New fact 0." in target["content"] and "New fact 1." in target["content"]
            if last_action == "edit":
                op["edits"] = [{"search": "New fact 0.", "replacement": "Corrected fact 0."}]
            elif last_action == "noop":
                op.update(action="noop", sections=[])
        return [op]

    llm.decide = decide
    api._pipeline.extract = lambda *a, **kw: drafts
    result = api.commit_session([], ctx)
    saved = fs.read_node(uri, ctx)
    assert len(llm.calls) == (2 if budget == "input" else 3)
    assert result["writes_completed"] == 1
    assert saved.metadata["version"] == 2
    assert "Archived attribute." in saved.content
    assert "New fact 1." in saved.content
    if last_action != "noop":
        assert "New fact 2." in saved.content and "New fact 3." in saved.content
    assert ("Corrected fact 0." if last_action == "edit" else "New fact 0.") in saved.content
    assert set(saved.metadata["provenance_ids"]) >= {f"source:{i}" for i in range(4)}
    assert set(saved.metadata["routing_key_aliases"]) >= {f"alias_{i}" for i in range(4)}
    assert set(saved.metadata["object_name_aliases"]) >= {f"name {i}" for i in range(4)}


def test_exact_incremental_edit_keeps_other_section():
    ctx, fs, llm, api = setup(lambda _: [])
    uri = api.write_memory(card(content="# Necklace\n\n## Identity\n- Gift from grandma.\n\n## Features\n- One charm."), ctx)["target_uri"]
    op = operation([0], target=uri, action="update")
    op["sections"] = []
    op["edits"] = [{"search": "- One charm.", "replacement": "- Two charms: a cross and a heart."}]
    llm.decide = lambda _: [op]
    result = api._node_processor().process([card()], [], ctx)
    assert "Gift from grandma." in result[0].content
    assert "Two charms: a cross and a heart." in result[0].content


@pytest.mark.parametrize("mode", ["eager", "lazy"])
def test_node_mode_uses_one_native_pass_and_preserves_complementary_drafts(mode):
    from providers.unified_config import OgMemConfig
    config = OgMemConfig(extraction_mode=mode, node_processing_enabled=True)
    assert config.extraction_mode == mode
    _, _, _, api = setup(lambda _: [], mode)
    extractor = api._create_extractors()[0]
    assert extractor._native_temperatures() == (None,)
    first, second = card(), card(content="# Necklace\n\n## Features\n- Heart charm.")
    assert extractor._merge_drafts([first, second]) == [first, second]


def test_lazy_reads_then_processes_topic_before_committing():
    from extraction.tools import _Span
    ctx, fs, llm, api = setup(lambda _: [], "lazy")
    uri = api.write_memory(card(), ctx)["target_uri"]
    llm.decide = lambda _: [operation([0], target=uri, action="update", fact="It has a heart charm.")]
    calls = []

    def react(messages, tools, tool_choice):
        calls.append(messages.copy())
        if messages[-1]["role"] != "tool":
            return [{"name": "read", "input": {"uri": uri}}], ""
        return [], json.dumps([{"name": "extract_entity", "input": {
            "routing_key": "necklace", "object_name": "necklace", "who": "Caroline",
            "owner_scope": "user", "abstract": "Caroline's necklace has a heart charm.",
            "overview": "- Caroline's necklace has a heart charm.",
            "content": "# Necklace\n\n## Features\n- It has a heart charm.", "confidence": 0.95,
        }}])

    llm.complete_with_tools_messages = react
    extractor = api._pipeline._extractors[0]
    extractor._identify_spans = lambda *args: [_Span(0, 0, categories=["entity"])]
    result = api.commit_session([{"role": "user", "content": "Caroline's necklace has a heart charm."}], ctx)
    assert result["writes_completed"] == 1
    assert len(calls) == 2
    assert "TOPIC_DRAFTS" in str(calls[0])
    assert calls[1][-1]["role"] == "tool"
    assert len(llm.calls) == 1
    assert llm.calls[0]["temperature"] == 0
    stored = fs.read_node(uri, ctx)
    assert "Gift from Caroline's grandma." in stored.content
    assert "It has a heart charm." in stored.content
    assert len(fs.nodes) == 1


def test_concurrent_same_owner_sees_committed_node():
    ctx, fs, llm, api = setup(lambda _: [])
    api._pipeline.extract = lambda *a, **k: [card()]
    def decide(_):
        if fs.nodes:
            op = operation([0], target=next(iter(fs.nodes)), action="noop")
            op["sections"] = []
            return [op]
        return [operation([0])]
    llm.decide = decide
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda i: api.commit_session([], replace(ctx, session_id=f"s{i}")), range(2)))
    assert len(results) == 2
    assert len(fs.nodes) == 1
    assert len(llm.calls) == 2


@pytest.mark.parametrize("category", ["entity", "preference"])
def test_checkpoint_revision_commits_against_refreshed_snapshot(category):
    from service.extraction_checkpoint import CandidateCheckpoint
    class Journal:
        def __init__(self, ctx):
            self.ctx, self.batch = ctx, {"archive_id": "revision", "slots": {}}
        def mutate(self, function):
            function(self.batch)
    ctx, fs, llm, api = setup(lambda _: [])
    original = card(category=category)
    if category == "preference":
        original = replace(original, content="# Necklace\n\n- Prefers wearing her grandmother's necklace.")
    uri = api.write_memory(original, ctx)["target_uri"]
    snapshot = fs.read_node(uri, ctx)
    drafts = [replace(original, routing_key=key, object_name=name) for key, name in
              [("grandmas_gift", "Grandmother's gift"), ("heart_necklace", "Heart necklace")]]
    source = [{"role": "user", "content": "It has a heart charm."}]
    llm.decide = lambda _: [operation([0, 1], target=uri, action="update", fact="It has a heart charm.")]
    stale = api._node_processor().process(drafts, source, ctx)
    checkpoint = CandidateCheckpoint(Journal(ctx), "0")
    checkpoint.save_drafts(drafts, source)
    accepted = checkpoint.accept(stale, 2, 0)
    snapshot.content += "\n- It is kept in a blue box."
    fs.write_node(snapshot, ctx)
    llm.decide = lambda _: [operation([0], target=uri, action="update", fact="It has a heart charm.")]
    plans = api._write_checkpointed_candidates(accepted, ctx, checkpoint)
    loaded, _, _ = checkpoint.load()
    assert len(plans) == 1
    assert checkpoint.record["revisions"]["0"] == 1
    assert loaded[0].expected_version == 2
    assert "blue box" in fs.read_node(uri, ctx).content
    assert "heart charm" in fs.read_node(uri, ctx).content
    for draft in drafts:
        assert draft.routing_key in loaded[0].verified_routing_aliases
        assert draft.object_name in loaded[0].verified_object_aliases
        assert api._writer._identity_resolver._names_match(fs.read_node(uri, ctx), draft)
    assert checkpoint.completed(0)["target_uri"] == uri


@pytest.mark.parametrize("failure", ["defer", "truncated"])
def test_stage_b_resume_reuses_saved_stage_a(failure):
    from unittest.mock import Mock
    from tests.unit.server.test_extraction_batch_recovery import setup_service, params
    from tests.unit.server.test_memory_card_recovery import enable
    from tests.unit.server.test_extraction_retry_task import ready
    ctx, fs, _, _ = setup(lambda _: [])
    service, writer, manager = setup_service(fs, ctx, entries=[card()])
    enable(service, writer)
    writer._node_processing_enabled = True
    deferred = operation([0], action="defer")
    def decide(_):
        if failure == "truncated":
            raise LLMOutputTruncatedError("finish_reason=length")
        return [deferred]

    writer._llm = NodeLLM(decide)
    queued = service.after_turn(params(ctx))
    assert queued["status"] == "queued"
    restarted, writer, manager = setup_service(fs, ctx, entries=[card()])
    enable(restarted, writer)
    writer._node_processing_enabled = True
    writer._llm = NodeLLM(lambda _: [operation([0])])
    restarted.wait_until_idle({"sessionId": ctx.session_id, "_ctx": ctx})
    ready(manager, ctx)
    restarted._extraction_retry.run_once(ctx, queued["recovery_task"]["id"])
    assert writer._pipeline.extract.call_count == 0
    assert len([n for n in fs.nodes.values() if n.category == "entity"]) == 1
    assert restarted.after_turn(params(ctx))["status"] == "duplicate"


def test_split_stage_b_resume_commits_one_complete_new_node():
    from tests.unit.server.test_extraction_batch_recovery import setup_service, params
    from tests.unit.server.test_memory_card_recovery import enable
    from tests.unit.server.test_extraction_retry_task import ready

    ctx, fs, _, _ = setup(lambda _: [])
    drafts = [card(f"alias_{i}", content="# Item\n\n## Details\n- " + "Source detail. " * 2300
                   + f"\n- Fact {i}.") for i in range(4)]
    service, writer, manager = setup_service(fs, ctx, entries=drafts)
    enable(service, writer)
    writer._node_processing_enabled = True

    def decide(call, defer=False):
        payload = node_payload(call)
        target = payload["nodes"][0]["uri"] if payload["nodes"] else ""
        op = operation(list(range(len(payload["drafts"]))), target=target,
                       action="defer" if target and defer else "update" if target else "create")
        op["sections"][0]["facts"] = [d["content"].splitlines()[-1][2:] for d in payload["drafts"]]
        return [op]

    writer._llm = NodeLLM(lambda call: decide(call, defer=True))
    queued = service.after_turn(params(ctx))
    assert queued["status"] == "queued"
    restarted, writer, manager = setup_service(fs, ctx, entries=drafts)
    enable(restarted, writer)
    writer._node_processing_enabled = True
    writer._llm = NodeLLM(decide)
    restarted.wait_until_idle({"sessionId": ctx.session_id, "_ctx": ctx})
    ready(manager, ctx)
    restarted._extraction_retry.run_once(ctx, queued["recovery_task"]["id"])
    stored = [node for node in fs.nodes.values() if node.category == "entity"]
    assert len(stored) == 1
    assert stored[0].metadata["version"] == 1
    assert all(f"Fact {i}." in stored[0].content for i in range(4))
    assert writer._pipeline.extract.call_count == 0
    assert restarted.after_turn(params(ctx))["status"] == "duplicate"
