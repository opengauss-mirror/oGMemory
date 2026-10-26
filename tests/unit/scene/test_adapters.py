from copy import deepcopy
from dataclasses import replace
from threading import RLock

import pytest

from core.errors import NodeNotFoundError
from core.models import ContextNode, SeedHit, TypedQuery
from scene.adapters import CanonicalSceneAdapter, _scope_context
from scene.config import InMemorySceneConfigProvider, SceneConfig
from scene.models import SceneCluster, SceneMember, SceneRecord, SceneSnapshot
from session.models import ArchiveEntry


class Repository:
    def __init__(self, scenes=()):
        self.snapshot = SceneSnapshot("acme", "user:alice", 0, scenes)
        self.calls = []
        self.lock = RLock()

    def load_snapshot(self, account_id, owner_space):
        self.calls.append((account_id, owner_space))
        if (account_id, owner_space) == (self.snapshot.account_id, self.snapshot.owner_space):
            return deepcopy(self.snapshot)
        return SceneSnapshot(account_id, owner_space, 0)

    def compare_and_swap(self, expected, replacement):
        if expected.revision != self.snapshot.revision:
            return False
        self.snapshot = SceneSnapshot(replacement.account_id, replacement.owner_space,
                                      replacement.revision, replacement.scenes)
        return True

    def compare_and_swap_checked(self, expected, replacement, check):
        with self.lock:
            return check(None) and self.compare_and_swap(expected, replacement)


class CanonicalFS:
    def __init__(self, nodes):
        self.nodes = {node.uri: node for node in nodes}
        self.reads = []
        self.lock = RLock()

    def read_node(self, uri, ctx):
        self.reads.append(uri)
        if uri not in self.nodes:
            raise NodeNotFoundError(uri)
        return deepcopy(self.nodes[uri])

    def write_node(self, node, ctx):
        with self.lock:
            self.nodes[node.uri] = deepcopy(node)

    def exists(self, uri, ctx):
        return uri in self.nodes


class IndexedVectors:
    def __init__(self, nodes):
        self.vectors = {node.uri: [1.0, 0.0] for node in nodes}
        self.texts = {node.uri: node.abstract for node in nodes}
        self.reads = []
        self.hits = []

    def get_embeddings(self, uris, filters, level=0, *, expected_texts=None):
        self.reads.append((uris, filters, level))
        return {uri: self.vectors[uri] for uri in uris if uri in self.vectors
                and (expected_texts is None or self.texts[uri] == expected_texts.get(uri))}

    def search_by_vector(self, query_vector, filters, top_k):
        self.search_filters = filters
        return self.hits[:top_k]


class Embedder:
    def __init__(self):
        self.calls = []

    def embed_texts(self, texts):
        self.calls.extend(texts)
        return [[1.0, 0.0] for _ in texts]


def node(name, *, owner="user:alice", category="event"):
    kind, identifier = owner.split(":")
    uri = f"ctx://acme/{kind}s/{identifier}/memories/{category}s/{name}"
    return ContextNode(uri, "MEMORY", category, 4, owner, name, "", f"Evidence for {name}",
                       {"version": 1, "status": "ACTIVE"})


def scene(name, nodes, *, centroid=(1.0, 0.0)):
    return SceneRecord(name, SceneCluster(
        nodes[0].owner_space, tuple(SceneMember(item.uri, 1.0, node_version=1) for item in nodes),
        centroid, 0.9, name, f"Description of {name}", "fingerprint",
    ))


def adapter(nodes, scenes=(), **kwargs):
    repo, fs, index, embedder = Repository(scenes), CanonicalFS(nodes), IndexedVectors(nodes), Embedder()
    repo.lock = fs.lock
    return CanonicalSceneAdapter(repo, fs, index, embedder, **kwargs)


def test_snapshot_load_prioritizes_changed_scene_and_keeps_full_snapshot():
    old, fresh, other = node("old"), node("fresh"), node("other")
    source = adapter([old, fresh, other], (scene("changed", [old], centroid=(0, 1)),
                                              scene("similar", [other])),
                     config=replace(SceneConfig(), max_candidate_scenes=1))
    batch = source.load_input("acme", "user:alice", node_uris=[old.uri, fresh.uri])
    assert {item.uri for item in batch.nodes} == {old.uri, fresh.uri}
    assert len(batch.snapshot.scenes) == 2
    assert source.vector_index.reads[0][1]["account_id"] == "acme"
    assert all(read[2] == 0 for read in source.vector_index.reads)
    assert source.embedder.calls == []


@pytest.mark.parametrize("category,suffix,depth", [("profile", "profile", 3), ("preference", "preferences/style", 4)])
def test_scene_reads_full_memory_created_by_context_writer(category, suffix, depth):
    from types import SimpleNamespace

    from commit.context_writer import ContextWriter
    from core.models import CandidateMemory, WritePlan
    from index.index_record_builder import build_index_records
    from providers.llm import MockLLM

    ctx = _scope_context("acme", "user:alice")
    uri = f"ctx://acme/users/alice/memories/{suffix}"
    candidate = CandidateMemory(category, "user", "style", "Planning preferences", "Friday release",
                                "Release reviews happen on Friday.", 0.9)
    fs = CanonicalFS([])
    writer = ContextWriter(fs, MockLLM(), policy_router=SimpleNamespace(
        plan=lambda candidate, ctx: WritePlan("create", uri)))
    writer.write_candidate(candidate, ctx)
    stored = fs.read_node(uri, ctx)
    records = build_index_records(stored)
    source = CanonicalSceneAdapter(Repository(), fs, IndexedVectors([stored]), Embedder())
    batch = source.load_input("acme", "user:alice", node_uris=[uri])
    assert stored.level == depth
    assert [record.level for record in records] == [0, 2]
    assert batch.nodes[0].content == candidate.content
    assert batch.embeddings[uri] == [1.0, 0.0]


def test_selected_scene_members_are_loaded_as_one_bounded_group():
    pending, old, more = node("pending"), node("old"), node("more")
    source = adapter([pending, old, more], (scene("complete", [old, more]),), max_input_nodes=2)
    batch = source.load_input("acme", "user:alice", node_uris=[pending.uri])
    assert [item.uri for item in batch.nodes] == [pending.uri]
    assert len(batch.snapshot.scenes) == 1


def test_deleted_source_archives_affected_scene_at_next_revision():
    pending, old = node("pending"), node("old")
    source = adapter([pending], (scene("outdated", [old]),))
    batch = source.load_input("acme", "user:alice", node_uris=[old.uri, pending.uri])
    assert batch.snapshot.scenes[0].status == "ARCHIVED"
    assert batch.snapshot.revision == 1
    assert source.repository.snapshot.scenes[0].version == 2


def test_publication_commits_refreshed_canonical_evidence():
    pending = node("pending")
    source = adapter([pending])
    source.fs.nodes[pending.uri].content = "Updated evidence"
    refreshed = source.load_input("acme", "user:alice", node_uris=[pending.uri])
    assert refreshed.nodes[0].content == "Updated evidence"
    assert source.compare_and_swap(refreshed.snapshot, replace(refreshed.snapshot, revision=1)) is True
    assert source.repository.snapshot.revision == 1


def test_scene_search_reuses_query_vector_and_resolves_current_filtered_members():
    event, preference = node("release"), node("style", category="preference")
    source = adapter([event, preference], (scene("Release planning", [event, preference]),))
    ctx = _scope_context("acme", "user:alice")
    query = TypedQuery("release", "MEMORY", ["event"], memory_directories=["events"])
    hits = source.search(query.text, ctx, limit=8, query_vector=[1, 0], typed_query=query)
    members = source.resolve_members(hits, ctx, limit=8, query_vector=[1, 0], typed_query=query)
    assert [hit.uri for hit in members] == [event.uri]
    assert members[0].score == 1.0
    assert members[0].metadata["scene_expanded"] is True
    assert source.embedder.calls == []


def test_scene_search_fills_active_quota_after_historical_archive():
    event = node("release")
    history = replace(scene("Old release", [event]), status="ARCHIVED")
    active = scene("Current release", [event])
    source = adapter([event], (history, active),
                     config=replace(SceneConfig(), max_active_scenes_per_space=1))
    ctx = _scope_context("acme", "user:alice")
    hits = source.search("release", ctx, limit=1, query_vector=[1, 0])
    members = source.resolve_members(hits, ctx, limit=1, query_vector=[1, 0])
    assert [hit.scene_id for hit in hits] == [active.scene_id]
    assert [member.uri for member in members] == [event.uri]


def test_agent_scope_uses_agent_identity_and_shared_access_uses_visible_scope():
    agent_node = node("tool", owner="agent:helper", category="tool")
    source = adapter([agent_node])
    source.repository.snapshot = SceneSnapshot("acme", "agent:helper", 0, (scene("Tools", [agent_node]),))
    batch = source.load_input("acme", "agent:helper", node_uris=[agent_node.uri])
    assert batch.nodes[0].owner_space == "agent:helper"
    ctx = replace(_scope_context("acme", "user:alice"), visible_owner_spaces=("user:alice", "agent:helper"))
    hits = source.search("tools", ctx, limit=8, query_vector=[1, 0])
    assert [hit.scene_id for hit in hits] == ["Tools"]


def test_request_scope_intersection_keeps_reads_in_authorized_owner():
    item = node("release")
    source = adapter([item], (scene("Release", [item]),))
    ctx = _scope_context("acme", "user:alice")
    query = TypedQuery("release", "MEMORY", [], owner_space=["user:alice", "user:other"])
    hits = source.search(query.text, ctx, limit=8, query_vector=[1, 0], typed_query=query)
    assert len(hits) == 1
    assert source.repository.calls == [("acme", "user:alice")]


def test_memory_source_reference_is_resolved_through_canonical_scope():
    from core.provenance_resolver import ProvenanceResolver
    item = node("release")
    source = adapter([item])
    reference = ProvenanceResolver.build_id("memory", item.uri)
    cards = source.read_source_context("acme", "user:alice", [reference], 8)
    assert cards[0]["uri"] == item.uri
    assert "Evidence for release" in cards[0]["text"]


class Archives:
    def __init__(self, entries):
        self.entries = {entry.archive_id: entry for entry in entries}
        self.reads = []

    def read_archive_by_id(self, archive_id, ctx):
        self.reads.append((archive_id, ctx.account_id, ctx.user_id, ctx.agent_id))
        return self.entries.get(archive_id)

    def read_archive(self, session_id, archive_id, ctx):
        entry = self.read_archive_by_id(archive_id, ctx)
        return entry if entry and entry.session_id == session_id else None


def archive(archive_id="release", *, user="alice", agent="helper"):
    return ArchiveEntry(archive_id, "planning", "Release planning", "Planning", [
        {"id": "m1", "role": "user", "content": "Release moves to Friday.", "created_at": "2026-10-19"},
        {"id": "m2", "role": "assistant", "content": "I will update the schedule."},
    ], metadata={"owner_user_id": user, "owner_agent_id": agent})


def test_archive_provenance_loads_selected_original_messages():
    from core.provenance_resolver import ProvenanceResolver

    store = Archives([archive()])
    source = adapter([], archive_store=store)
    reference = ProvenanceResolver.build_id("archive", "release", ["m1"])
    cards = source.read_source_context("acme", "user:alice", [reference], 8)
    assert cards[0]["message_ids"] == ["m1"]
    assert cards[0]["text"] == "[m1] 2026-10-19 user: Release moves to Friday.\n"
    assert cards[0]["source_ref"] == reference
    assert store.reads == [("release", "acme", "alice", "")]


@pytest.mark.parametrize("reference_type", ["uri", "memory", "archive"])
def test_raw_chunk_written_through_public_api_is_read_as_source(reference_type):
    from core.provenance_resolver import ProvenanceResolver
    from providers.llm import MockLLM
    from service.api import MemoryWriteAPI

    fs = CanonicalFS([])
    ctx = _scope_context("acme", "user:alice")
    api = MemoryWriteAPI(fs, MockLLM(), directory_summary_enabled=False)
    messages = [{"id": "m1", "role": "user", "content": "Project evidence. " * 100},
                {"id": "m2", "role": "assistant", "content": "Final checkpoint after source excerpt."}]
    result = api.write_raw_chunk(messages, ctx, session_id="release-planning")
    uri = result["target_uri"]
    source = CanonicalSceneAdapter(Repository(), fs, IndexedVectors([]), Embedder())
    reference = uri if reference_type == "uri" else ProvenanceResolver.build_id(reference_type, uri)
    cards = source.read_source_context("acme", "user:alice", [reference], 1)
    stored = fs.read_node(uri, ctx)
    assert stored.category == "session_archive"
    assert stored.level == 4
    assert cards[0]["text"] == stored.content
    assert cards[0]["text"].endswith("Final checkpoint after source excerpt.")


def test_contextfs_session_archive_is_resolved_using_existing_archive_store():
    from session.archive_store import SessionArchiveStore

    fs = CanonicalFS([])
    ctx = _scope_context("acme", "user:alice")
    stored = SessionArchiveStore(fs).write_archive(
        "planning", "Overview", "Abstract", archive().messages, ctx, archive_id="source")
    source = CanonicalSceneAdapter(Repository(), fs, IndexedVectors([]), Embedder(), archive_store=Archives([]))
    cards = source.read_source_context("acme", "user:alice", [stored.uri], 1)
    assert stored.success
    assert cards[0]["message_ids"] == ["m1", "m2"]


def test_explicit_archive_uri_loads_original_conversation():
    source = adapter([], archive_store=Archives([archive()]))
    uri = "ctx://acme/sessions/planning/history/release"
    cards = source.read_source_context("acme", "user:alice", [uri], 8)
    assert cards[0]["uri"] == uri
    assert cards[0]["message_ids"] == ["m1", "m2"]


def test_archive_cards_apply_owner_scope_to_each_resolved_entry():
    from core.provenance_resolver import ProvenanceResolver

    store = Archives([archive("alice"), archive("bob", user="bob"), archive("agent", user="")])
    source = adapter([], archive_store=store)
    refs = [ProvenanceResolver.build_id("archive", key) for key in ("bob", "alice", "agent")]
    user_cards = source.read_source_context("acme", "user:alice", refs, 8)
    agent_cards = source.read_source_context("acme", "agent:helper", refs, 8)
    assert [card["uri"].rsplit("/", 1)[1] for card in user_cards] == ["alice"]
    assert [card["uri"].rsplit("/", 1)[1] for card in agent_cards] == ["agent"]


@pytest.mark.parametrize("content", ["Release moves to Friday.", '"quoted"\ntext\t'])
def test_archive_text_is_bounded_by_configured_evidence_budget(content):
    import json

    entry = archive()
    entry.messages[0]["content"] = content * 5000
    config = replace(SceneConfig(), candidate_card_budget_tokens=256)
    source = adapter([], config=config, archive_store=Archives([entry]))
    cards = source.read_source_context("acme", "user:alice", ["ctx://acme/sessions/planning/history/release"], 1)
    assert cards[0]["message_ids"] == ["m1"]
    assert sum(len(json.dumps(card, ensure_ascii=False)) for card in cards) <= 1024


def test_card_budget_preserves_pending_priority():
    nodes = [node(f"event-{index}") for index in range(12)]
    for item in nodes:
        item.content *= 200
    source = adapter(nodes, config=replace(SceneConfig(), analysis_input_budget_tokens=4096,
                                           analysis_trigger_tokens=2048, candidate_card_budget_tokens=2048))
    batch = source.load_input("acme", "user:alice", node_uris=[item.uri for item in nodes])
    assert 0 < len(batch.nodes) < len(nodes)
    assert batch.nodes[0].uri == nodes[0].uri
    assert sum(source._card_cost(item) for item in batch.nodes) <= 2048 * 4 - 1024


def test_request_config_snapshot_changes_on_next_request():
    class Provider(InMemorySceneConfigProvider):
        reads = 0

        def get_effective(self, account_id, owner_space):
            self.reads += 1
            return super().get_effective(account_id, owner_space)

    item = node("release")
    provider = Provider()
    source = adapter([item], (scene("Release", [item]),), config_provider=provider)
    ctx = _scope_context("acme", "user:alice")
    query = TypedQuery("release", "MEMORY", [])
    request = source.for_request(ctx, [query, query])
    for _ in range(2):
        assert request.search("release", ctx, limit=8, query_vector=[1, 0])
    assert provider.reads == 1
    provider.publish("acme", "user:alice", 0, {"scene_boost": 0.2})
    assert request.config.scene_boost == 0.5
    assert source.for_request(ctx, [query]).config.scene_boost == 0.2
    assert provider.reads == 2


@pytest.mark.parametrize("route", ["member", "lexical", "semantic"])
def test_scene_routes_supply_canonical_evidence_to_ranker(route):
    from scene.candidate_index import LexicalSceneIndex
    from scene.retrieval import SceneRanker
    from tests.unit.scene.test_candidate_index import MemoryIndex

    first, second = node("release"), node("rollback")
    centroid = (1, 0) if route == "semantic" else (0, 1)
    source = adapter([first, second], [scene("deployment", [first, second], centroid=centroid)])
    ctx = _scope_context("acme", "user:alice")
    direct = [SeedHit(first.uri, .8, owner_space="user:alice")] if route == "member" else []
    if route == "lexical":
        source.config = replace(source.config, lexical_enabled=True)
        source.candidate_index = LexicalSceneIndex(MemoryIndex())
        source.candidate_index.publish(source.repository.snapshot)
    hits = source.search_with_evidence("deployment", ctx, limit=3,
                                      query_vector=[1, 0], direct_hits=direct)
    assert hits[0].routes == (route,)
    members = source.resolve_members(hits, ctx, limit=3, query_vector=[1, 0])
    result = SceneRanker().rank(members, hits, limit=3)
    assert {hit.uri for hit in result.hits} == {first.uri, second.uri}
    assert all(hit.metadata["scene_expanded"] for hit in result.hits)


def test_member_limit_counts_matching_canonical_nodes():
    preference = node("first", category="preference")
    first, second = node("release"), node("rollback")
    current = scene("deployment", [preference, first, second])
    current = replace(current, cluster=replace(current.cluster, members=(
        replace(current.cluster.members[0], weight=2), *current.cluster.members[1:],
    )))
    source = adapter([preference, first, second], [current])
    ctx = _scope_context("acme", "user:alice")
    query = TypedQuery("deployment", "MEMORY", ["event"])
    hits = source.search(query.text, ctx, limit=3, query_vector=[1, 0], typed_query=query)
    members = source.resolve_members(hits, ctx, limit=2, query_vector=[1, 0], typed_query=query)
    assert {hit.uri for hit in members} == {first.uri, second.uri}


def test_analysis_tools_return_scoped_node_and_scene_cards():
    first = node("release")
    source = adapter([first], [scene("deployment", [first])])
    source.vector_index.hits = [SeedHit(first.uri, .9, owner_space="user:alice")]
    cards = source.search_related_nodes("acme", "user:alice", "release", 3)
    evidence = source.read_source_context("acme", "user:alice", [first.uri], 1)
    assert cards == [{**source.node_card(first), "node_fingerprint": evidence[0]["node_fingerprint"]}]
    assert source.vector_index.search_filters == {
        "account_id": "acme", "owner_space": ["user:alice", "user_alice"],
        "context_type": "MEMORY", "level": 0,
    }
    scenes = source.search_scene_cards("acme", "user:alice", "release", 3)
    assert scenes[0]["scene_id"] == "deployment"
    assert scenes[0]["member_uris"] == [first.uri]
