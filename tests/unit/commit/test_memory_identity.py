"""Replay real naming and duplication cases through the public write API."""

from copy import deepcopy
from contextlib import contextmanager
from dataclasses import replace

import pytest

from commit.identity_resolver import MemoryIdentityResolver, observation_id
from commit.merge_policies import AggregateTopicPolicy, AppendOnlyPolicy
from core.errors import ConcurrentModificationError, NodeNotFoundError, ValidationError
from core.models import CandidateMemory, RequestContext
from core.uri_resolver import URIResolver
from core.validation import validate_candidate
from extraction.schemas.registry import SchemaRegistry
from service.api import MemoryWriteAPI


class MemoryFS:
    def __init__(self):
        self.nodes = {}

    def list_children(self, uri, ctx):
        return sorted(u for u in self.nodes if u.rsplit("/", 1)[0] == uri.rstrip("/"))

    def read_node(self, uri, ctx):
        if uri not in self.nodes:
            raise NodeNotFoundError(uri)
        return deepcopy(self.nodes[uri])

    def exists(self, uri, ctx):
        return uri in self.nodes

    def archive_node(self, uri, ctx):
        self.nodes.pop(uri)

    def write_node(self, node, ctx):
        current = self.nodes.get(node.uri)
        version = current.metadata["version"] if current else 0
        expected = node.metadata.get("_expected_version", node.metadata.get("expected_version"))
        if expected is not None and expected != version:
            raise ConcurrentModificationError(node.uri, expected, version)
        node = deepcopy(node)
        node.metadata["version"] = version + 1
        node.metadata.pop("_expected_version", None)
        node.metadata.pop("expected_version", None)
        self.nodes[node.uri] = node


class DecisionLLM:
    """Placeholder provider for exercising prepared extraction operations."""
    pass


@pytest.mark.parametrize("category,owner", [("event", "user"), ("case", "agent")])
def test_occurrence_replay_identity_survives_verified_aliases(ctx, category, owner):
    fs = MemoryFS()
    writer = api(fs, DecisionLLM())
    original = replace(memory("receiver_repair", category=category, content="Nora repaired the receiver."),
                       owner_scope=owner)
    uri = writer.write_memory(original, ctx)["target_uri"]
    replay = replace(original, verified_routing_aliases=["radio_repair"],
                     verified_object_aliases=["Nora's receiver"])
    result = writer.write_memory(replay, ctx)
    assert result["action"] == "skip" and result["target_uri"] == uri
    assert len(fs.nodes) == 1


class MergeLLM:
    def complete_json(self, prompt, schema):
        return {"content": "The earlier fact remains. The later fact was added."}


@pytest.fixture
def ctx():
    return RequestContext(account_id="identity-test", user_id="reader", agent_id="agent",
                          session_id="conversation", trace_id="trace")


def memory(key, *, category="entity", content="经验索引文件", **kwargs):
    return CandidateMemory(category=category, owner_scope="user", routing_key=key,
                           abstract=content, overview=content, content=content, confidence=.9,
                           provenance_ids=["prov:1:archive:sample:message-one"], **kwargs)


def api(fs, llm):
    return MemoryWriteAPI(fs, llm, directory_summary_enabled=False)


def test_topic_scope_and_write_share_backend_lock(ctx):
    class LockedFS(MemoryFS):
        acquisitions = 0

        @contextmanager
        def memory_identity_lock(self, directory, request):
            self.acquisitions += 1
            yield

    fs = LockedFS()
    writer = api(fs, DecisionLLM())
    resolver = writer._writer._identity_resolver
    candidate = memory("receiver")
    directory = resolver._directory(candidate, ctx)
    with resolver._lock(directory, ctx):
        result = writer.write_memory(candidate, ctx)
        assert fs.acquisitions == 1
    with resolver._lock(directory, ctx):
        assert fs.acquisitions == 2
    assert fs.read_node(result["target_uri"], ctx).content == candidate.content


def test_fallback_directory_locks_share_ownership_and_recycle_idle_entries(ctx, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from weakref import WeakValueDictionary
    import commit.identity_resolver as identity

    registry = WeakValueDictionary()
    monkeypatch.setattr(identity, "_LOCAL_LOCKS", registry)
    fs = MemoryFS()
    resolvers = [api(fs, DecisionLLM())._writer._identity_resolver for _ in range(2)]
    directory = resolvers[0]._directory(memory("receiver"), ctx)
    waiting = threading.Event()
    observed = []

    def enter():
        waiting.set()
        with resolvers[1]._lock(directory, ctx):
            observed.append(registry[directory])

    with ThreadPoolExecutor(max_workers=1) as pool:
        with resolvers[0]._lock(directory, ctx):
            held = registry[directory]
            waiter = pool.submit(enter)
            assert waiting.wait(2)
            for index in range(100):
                with resolvers[0]._lock(directory + str(index), ctx):
                    assert len(registry) == 2
                assert len(registry) == 1
        waiter.result(timeout=2)
    assert observed == [held]


def test_existing_name_without_read_target_keeps_writes_available(ctx):
    class SummaryFS(MemoryFS):
        def list_node_summaries_page(self, directory, request, *, limit, after_uri=None):
            return [replace(self.read_node(uri, request), content="", overview="")
                    for uri in self.list_children(directory, request) if not after_uri or uri > after_uri][:limit]

    fs = SummaryFS()
    writer = api(fs, MergeLLM())
    first = writer.write_memory(memory("topic", content="The earlier fact remains."), ctx)
    second = writer.write_memory(memory("topic", content="The later fact was added."), ctx)

    assert second["target_uri"] == first["target_uri"]
    stored = fs.read_node(first["target_uri"], ctx)
    assert stored.content == "The earlier fact remains. The later fact was added."
    assert stored.metadata["version"] == 2
    assert len(fs.nodes) == 1


@pytest.mark.parametrize("category", ["entity", "preference", "pattern"])
def test_custom_filename_template_preserves_create_and_update_uri(ctx, category):
    registry = SchemaRegistry()
    schema = registry.get(category)
    registry.register(replace(schema, filename_template="item_{{ routing_key }}.md"))
    fs = MemoryFS()
    writer = MemoryWriteAPI(fs, MergeLLM(), schema_registry=registry,
                            directory_summary_enabled=False)
    first = memory("topic", category=category, content="Initial fact.")
    expected = URIResolver(registry).resolve(category, {"routing_key": "topic"}, ctx)

    created = writer.write_memory(first, ctx)
    updated = writer.write_memory(replace(first, content="Additional fact."), ctx)

    assert created["target_uri"] == updated["target_uri"] == expected
    assert len(fs.nodes) == 1
    stored = fs.read_node(expected, ctx)
    assert stored.metadata["routing_key"] == "topic"
    assert stored.metadata["version"] == 2


def test_custom_filename_template_enforces_rendered_name_limit(ctx):
    registry = SchemaRegistry()
    registry.register(replace(registry.get("entity"),
                              filename_template="item_{{ routing_key }}.md"))
    fs = MemoryFS()
    writer = MemoryWriteAPI(fs, MergeLLM(), schema_registry=registry,
                            directory_summary_enabled=False)

    with pytest.raises(ValidationError, match="Memory name exceeds path segment length limit"):
        writer.write_memory(memory("旅行" * 34 + "记"), ctx)
    assert not fs.nodes


@pytest.mark.parametrize("marker", ["user", "<user>"])
@pytest.mark.parametrize("marker_first", [False, True])
def test_self_preference_role_marker_reuses_topic(ctx, marker, marker_first):
    fs = MemoryFS()
    writer = api(fs, MergeLLM())
    first = memory("pets", category="preference", content="Likes dogs.")
    second = replace(first, who=marker, content="Likes cats.")
    if marker_first:
        first, second = second, first

    uri = writer.write_memory(first, ctx)["target_uri"]
    updated = writer.write_memory(second, ctx)

    assert updated["target_uri"] == uri
    assert uri.endswith("/preferences/pets")
    assert len(fs.nodes) == 1


def test_existing_role_prefixed_preference_keeps_its_uri(ctx):
    fs = MemoryFS()
    writer = api(fs, MergeLLM())
    resolver = writer._writer._identity_resolver
    writer._writer._identity_resolver = None
    legacy = memory("user_pets", category="preference", who="user", content="Likes dogs.")
    uri = writer.write_memory(legacy, ctx)["target_uri"]
    writer._writer._identity_resolver = resolver

    result = writer.write_memory(memory("pets", category="preference", content="Likes cats."), ctx)

    assert result["target_uri"] == uri
    assert len(fs.nodes) == 1


def test_custom_add_only_entity_uses_schema_policy(ctx):
    registry = SchemaRegistry()
    registry.register(replace(registry.get("entity"), operation_mode="add_only"))
    fs = MemoryFS()
    writer = MemoryWriteAPI(fs, MergeLLM(), schema_registry=registry,
                            directory_summary_enabled=False)
    entity = memory("radar", content="Radar installed.")

    first_uri = writer.write_memory(entity, ctx)["target_uri"]
    second_uri = writer.write_memory(replace(entity, content="Radar inspected."), ctx)["target_uri"]

    assert first_uri != second_uri
    assert len(fs.nodes) == 2


def test_custom_registered_policy_takes_precedence(ctx):
    fs = MemoryFS()
    writer = api(fs, MergeLLM())
    writer._writer._policy_router.register_policy(
        "entity", AppendOnlyPolicy(fs, URIResolver(SchemaRegistry())),
    )
    entity = memory("radar", content="Radar installed.")

    first_uri = writer.write_memory(entity, ctx)["target_uri"]
    second_uri = writer.write_memory(replace(entity, content="Radar inspected."), ctx)["target_uri"]

    assert first_uri != second_uri
    assert len(fs.nodes) == 2


def test_registered_aggregate_policy_uses_its_own_uri_resolver(ctx):
    fs = MemoryFS()
    writer = api(fs, MergeLLM())
    registry = SchemaRegistry()
    registry.register(replace(registry.get("entity"),
                              filename_template="custom_{{ routing_key }}.md"))
    writer._writer._policy_router.register_policy(
        "entity", AggregateTopicPolicy(fs, URIResolver(registry)),
    )

    uri = writer.write_memory(memory("radar", content="Radar installed."), ctx)["target_uri"]

    assert uri.endswith("/entities/custom_radar")
    assert len(fs.nodes) == 1


def test_custom_agent_entity_uses_schema_owner(ctx):
    registry = SchemaRegistry()
    registry.register(replace(
        registry.get("entity"), owner_scope="agent",
        directory="ctx://{{ account_id }}/agents/{{ agent_id }}/memories/entities",
    ))
    fs = MemoryFS()
    writer = MemoryWriteAPI(fs, MergeLLM(), schema_registry=registry,
                            directory_summary_enabled=False)
    entity = replace(memory("radar", content="Radar installed."), owner_scope="agent")
    assert MemoryIdentityResolver(fs, URIResolver(registry)).handles(entity)

    first_uri = writer.write_memory(entity, ctx)["target_uri"]
    updated = writer.write_memory(replace(entity, content="Radar inspected."), ctx)

    assert updated["target_uri"] == first_uri
    assert first_uri.startswith("ctx://identity-test/agents/agent/")
    assert len(fs.nodes) == 1


def test_preference_topics_with_shared_object_stay_distinct(ctx):
    fs = MemoryFS()
    writer = api(fs, DecisionLLM())
    brewing = memory("tea_brewing", category="preference", object_name="Tea",
                     content="Uses a clay pot for tea.")
    purchasing = memory("tea_purchasing", category="preference", object_name="Tea",
                       content="Buys loose-leaf tea.")

    first = writer.write_memory(brewing, ctx)["target_uri"]
    second = writer.write_memory(purchasing, ctx)["target_uri"]

    assert first != second
    assert len(fs.nodes) == 2


def test_object_name_does_not_merge_a_different_preference_topic(ctx):
    fs = MemoryFS()
    writer = api(fs, DecisionLLM())
    brewing = memory("tea_brewing", category="preference", object_name="Tea",
                     content="Uses a clay pot for tea.")
    drinking = memory("tea", category="preference", object_name="Tea",
                     content="Drinks tea daily.")

    first_uri = writer.write_memory(brewing, ctx)["target_uri"]
    second_uri = writer.write_memory(drinking, ctx)["target_uri"]

    assert second_uri != first_uri
    assert len(fs.nodes) == 2


def test_distinct_entities_from_one_source_keep_separate_nodes(ctx):
    fs = MemoryFS()
    writer = api(fs, DecisionLLM())
    alice = replace(memory("alice", content="职业：工程师"), abstract="Alice", overview="Alice")
    bob = replace(memory("bob", content="职业：工程师"), abstract="Bob", overview="Bob")

    first_uri = writer.write_memory(alice, ctx)["target_uri"]
    second_uri = writer.write_memory(bob, ctx)["target_uri"]

    assert second_uri != first_uri
    assert len(fs.nodes) == 2


def test_existing_numeric_prefix_name_can_be_updated(ctx):
    fs = MemoryFS()
    writer = api(fs, MergeLLM())
    writer._writer._identity_resolver = None
    old_uri = writer.write_memory(memory("3d_printer", content="Uses a 3D printer."), ctx)["target_uri"]
    writer._writer._identity_resolver = MemoryIdentityResolver(fs, URIResolver(SchemaRegistry()))

    result = writer.write_memory(memory("3d_printer", content="Maintains a 3D printer."), ctx)

    assert result["target_uri"] == old_uri
    assert len(fs.nodes) == 1


def test_existing_synonym_normalization_keeps_one_topic(ctx):
    fs = MemoryFS()
    writer = api(fs, MergeLLM())
    first_uri = writer.write_memory(memory("coffee", category="preference",
                                           content="Likes coffee."), ctx)["target_uri"]

    second_uri = writer.write_memory(memory("tea", category="preference",
                                            content="Likes tea."), ctx)["target_uri"]

    assert first_uri == second_uri
    assert first_uri.endswith("/beverage")
    assert len(fs.nodes) == 1


@pytest.mark.parametrize("category,object_name", [
    ("preference", ""), ("entity", "经验索引文件"), ("pattern", ""),
])
def test_legacy_hash_name_is_reused_for_a_validated_chinese_key(ctx, category, object_name):
    fs = MemoryFS()
    writer = api(fs, MergeLLM())
    candidate = validate_candidate(memory("经验索引文件", category=category, object_name=object_name))
    writer._writer._identity_resolver = None
    old_uri = writer.write_memory(replace(candidate, routing_key=candidate.legacy_routing_key), ctx)["target_uri"]
    writer._writer._identity_resolver = MemoryIdentityResolver(fs, URIResolver(SchemaRegistry()))

    result = writer.write_memory(replace(candidate, content="新的索引经验", abstract="新的索引经验",
                                         overview="新的索引经验"), ctx)

    assert result["target_uri"] == old_uri
    assert len(fs.nodes) == 1


@pytest.mark.parametrize("category,owner", [("event", "user"), ("case", "agent")])
def test_scoped_occurrence_can_be_updated_by_uri(ctx, category, owner):
    fs = MemoryFS()
    writer = api(fs, DecisionLLM())
    original = replace(memory("inspection", category=category, scope_ref="Atlas",
                              content="Inspection started."), owner_scope=owner)
    uri = writer.write_memory(original, ctx)["target_uri"]
    updated = replace(original, target_uri=uri, expected_version=1,
                      content="Inspection finished.", abstract="Inspection finished.",
                      overview="Inspection finished.")

    assert writer.write_memory(updated, ctx)["target_uri"] == uri
    assert fs.read_node(uri, ctx).metadata["scope_ref"] == "Atlas"


def test_composed_memory_name_obeys_storage_byte_limit(ctx):
    candidate = validate_candidate(memory("中" * 60, category="preference", scope_ref="文" * 10))

    with pytest.raises(ValidationError, match="path segment length limit"):
        api(MemoryFS(), DecisionLLM()).write_memory(candidate, ctx)


def test_unversioned_write_rechecks_identity_after_racing_create(ctx):
    class RacingFS(MemoryFS):
        def __init__(self):
            super().__init__()
            self.raced = False

        def write_node(self, node, request):
            if not self.raced:
                self.raced = True
                super().write_node(node, request)
                raise ConcurrentModificationError(node.uri, 0, 1)
            super().write_node(node, request)

    fs = RacingFS()
    result = api(fs, DecisionLLM()).write_memory(memory("topic"), ctx)

    assert result["action"] == "skip"
    assert len(fs.nodes) == 1


@pytest.mark.parametrize("category,owner", [("event", "user"), ("case", "agent")])
def test_occurrence_followup_without_read_target_is_recorded(ctx, category, owner):
    fs = MemoryFS()
    writer = api(fs, DecisionLLM())
    first = replace(memory("valve_inspection", category=category,
                           content="Valve R4 was inspected.", when="2025-03-04", who="Mira"),
                    owner_scope=owner)
    first_uri = writer.write_memory(first, ctx)["target_uri"]
    followup = replace(first, content="Valve R4 passed the inspection.",
                       provenance_ids=["source:followup"])

    second_uri = writer.write_memory(followup, ctx)["target_uri"]

    assert second_uri != first_uri
    assert len(fs.nodes) == 2


def test_exact_topic_updates_and_unsourced_occurrences_avoid_directory_scan(ctx):
    class CountingFS(MemoryFS):
        def __init__(self):
            super().__init__()
            self.scans = 0

        def list_children(self, uri, request):
            self.scans += 1
            return super().list_children(uri, request)

    fs = CountingFS()
    writer = api(fs, MergeLLM())
    first = memory("pets", category="preference", content="Likes dogs.")
    writer.write_memory(first, ctx)
    scans_after_create = fs.scans
    writer.write_memory(replace(first, content="Likes cats."), ctx)
    assert fs.scans == scans_after_create

    occurrence = replace(memory("inspection", category="event", content="Valve was checked."),
                         provenance_ids=[])
    writer.write_memory(occurrence, ctx)
    writer.write_memory(occurrence, ctx)
    assert fs.scans == scans_after_create


def test_historical_preference_content_can_become_current_again(ctx):
    fs = MemoryFS()
    writer = api(fs, MergeLLM())
    first = memory("drink", category="preference", content="Likes tea.")
    writer.write_memory(first, ctx)
    writer.write_memory(replace(first, content="Likes coffee.",
                                provenance_ids=["source:coffee"]), ctx)
    assert writer.write_memory(first, ctx)["action"] == "skip"

    result = writer.write_memory(replace(first, provenance_ids=["source:tea-again"]), ctx)

    assert result["action"] == "merge"
    assert fs.read_node(result["target_uri"], ctx).metadata["version"] == 3


def test_observation_sources_are_tracked_per_content(ctx):
    fs = MemoryFS()
    writer = api(fs, MergeLLM())
    tea = memory("drink", category="preference", content="Likes tea.")
    coffee = replace(tea, content="Likes coffee.", provenance_ids=["source:shared-span"])
    tea_again = replace(tea, provenance_ids=["source:shared-span"])
    uri = writer.write_memory(tea, ctx)["target_uri"]
    writer.write_memory(coffee, ctx)

    result = writer.write_memory(tea_again, ctx)

    assert result["action"] == "merge"
    stored = fs.read_node(uri, ctx)
    assert stored.metadata["version"] == 3
    assert stored.metadata["memory_observations"][-1] == observation_id(tea)
    assert stored.metadata["memory_observation_sources"][observation_id(tea)] == [
        "prov:1:archive:sample:message-one", "source:shared-span",
    ]
    assert stored.metadata["memory_observation_sources"][observation_id(coffee)] == [
        "source:shared-span",
    ]


def test_current_observation_records_new_source_on_replay(ctx):
    fs = MemoryFS()
    writer = api(fs, MergeLLM())
    tea = memory("drink", category="preference", content="Likes tea.")
    uri = writer.write_memory(tea, ctx)["target_uri"]

    replay = writer.write_memory(replace(tea, provenance_ids=["source:second"]), ctx)

    assert replay["action"] == "skip"
    stored = fs.read_node(uri, ctx)
    assert stored.metadata["version"] == 2
    assert stored.metadata["memory_observation_sources"][observation_id(tea)] == [
        "prov:1:archive:sample:message-one", "source:second",
    ]


def test_first_person_quote_binds_to_reporting_speaker(ctx):
    from extraction.source_attribution import bind_source_attribution

    candidate = memory("drink", category="preference", content="Alice likes tea.",
                       who="Bob", evidence_quote="I like tea.")
    source = "Alice: I like tea.\nBob: I like coffee."

    bound = bind_source_attribution(candidate, source, ctx.user_id)

    assert bound.who == "Alice"
    assert bound.attributed_speaker == "Alice"


def test_user_role_turn_keeps_its_own_preference(ctx):
    from extraction.source_attribution import bind_source_attribution

    candidate = memory("drink", category="preference", content="reader likes tea.",
                       who="reader", evidence_quote="I like tea.")
    source = "Alice: Hello.\nuser: I like tea.\nBob: Hi."

    bound = bind_source_attribution(candidate, source, ctx.user_id)

    assert bound.who == ctx.user_id
    assert bound.attributed_speaker == ctx.user_id
    assert bound.scope_ref == ""


def test_assistant_turn_does_not_supply_named_preference_evidence(ctx):
    from extraction.source_attribution import bind_source_attribution

    candidate = memory("drink", category="preference", content="Alice likes tea.",
                       who="Alice", evidence_quote="I like tea.")
    source = "Alice: Hello.\nassistant: I like tea.\nBob: Hi."

    with pytest.raises(ValueError, match="attribution"):
        bind_source_attribution(candidate, source, ctx.user_id)


def test_third_person_quote_keeps_named_preference_subject(ctx):
    from extraction.source_attribution import bind_source_attribution

    candidate = memory("drink", category="preference", content="Bob likes tea.",
                       who="Bob", evidence_quote="Bob likes tea.")
    source = "Alice: Bob likes tea.\nBob: I like coffee."

    bound = bind_source_attribution(candidate, source, ctx.user_id)

    assert bound.who == "Bob"
    assert bound.attributed_speaker == "Alice"


def test_self_preference_who_can_be_filled_after_creation(ctx):
    fs = MemoryFS()
    writer = api(fs, MergeLLM())
    first = memory("pets", category="preference", content="Likes dogs.")
    uri = writer.write_memory(first, ctx)["target_uri"]
    update = replace(first, who=ctx.user_id, content="Likes dogs and cats.",
                     provenance_ids=["source:update"])

    assert writer.write_memory(update, ctx)["target_uri"] == uri
    assert fs.read_node(uri, ctx).metadata["who"] == ctx.user_id


def test_unscoped_preferences_of_different_people_get_distinct_uris(ctx):
    fs = MemoryFS()
    writer = api(fs, DecisionLLM())
    first = memory("pets", category="preference", content="Avery likes dogs.", who="Avery")
    second = memory("pets", category="preference", content="Jordan likes cats.", who="Jordan")

    first_uri = writer.write_memory(first, ctx)["target_uri"]
    second_uri = writer.write_memory(second, ctx)["target_uri"]

    assert first_uri != second_uri
    assert len(fs.nodes) == 2


def test_undated_occurrences_from_different_sources_are_distinct(ctx):
    fs = MemoryFS()
    writer = api(fs, DecisionLLM())
    first = memory("valve_inspection", category="event", content="Valve R4 was inspected.",
                   who="Mira")
    first_uri = writer.write_memory(first, ctx)["target_uri"]
    second = replace(first, provenance_ids=["source:another-inspection"])

    second_uri = writer.write_memory(second, ctx)["target_uri"]

    assert second_uri != first_uri
    assert len(fs.nodes) == 2


def test_replayed_old_observation_does_not_replace_current_order(ctx):
    fs = MemoryFS()
    writer = api(fs, DecisionLLM())
    first = memory("tea", category="preference", content="Likes tea.")
    uri = writer.write_memory(first, ctx)["target_uri"]
    current = replace(first, when="2025-03-04", target_uri=uri, expected_version=1)
    writer.write_memory(current, ctx)
    replay = replace(first, routing_key="tea_alias", target_uri=uri, expected_version=2)

    writer.write_memory(replay, ctx)

    assert fs.read_node(uri, ctx).metadata["memory_observations"][-1] == observation_id(current)


def test_archived_memory_name_reactivates_with_version_check(ctx):
    class ArchivedFS(MemoryFS):
        def __init__(self):
            super().__init__()
            self.archived_uri = "ctx://identity-test/users/reader/memories/preferences/pets"
            self.archived_version = 4

        def archived_node_version(self, uri, request):
            return self.archived_version if uri == self.archived_uri else None

        def write_node(self, node, request):
            if node.uri == self.archived_uri and node.uri not in self.nodes:
                expected = node.metadata.get("_expected_version", node.metadata.get("expected_version"))
                if expected != self.archived_version:
                    raise ConcurrentModificationError(node.uri, expected, self.archived_version)
                self.nodes[node.uri] = deepcopy(node)
                self.nodes[node.uri].metadata["version"] = self.archived_version + 1
                return
            super().write_node(node, request)

    fs = ArchivedFS()
    candidate = memory("pets", category="preference", content="Likes cats.")

    result = api(fs, DecisionLLM()).write_memory(candidate, ctx)

    assert result["action"] == "merge"
    assert result["target_uri"] == fs.archived_uri
    assert fs.read_node(fs.archived_uri, ctx).metadata["version"] == 5


@pytest.mark.parametrize("category,owner", [("event", "user"), ("case", "agent")])
def test_occurrence_key_survives_allocated_filename_and_versioned_updates(ctx, category, owner):
    fs = MemoryFS()
    writer = api(fs, DecisionLLM())
    first = replace(memory("valve_inspection", category=category,
                           content="Valve R4 passed inspection.", when="2025-03-04", who="Mira"),
                    owner_scope=owner)
    uri = writer.write_memory(first, ctx)["target_uri"]
    created = fs.read_node(uri, ctx)
    assert created.metadata["routing_key"] == "valve_inspection"
    assert uri.rsplit("/", 1)[-1].startswith("valve_inspection_")

    for version, supplied_key in enumerate((uri.rsplit("/", 1)[-1], "r4_verified_inspection"), 1):
        text = first.content + f" Report revision {version} verified the result."
        updated = replace(first, routing_key=supplied_key, content=text,
                          target_uri=uri, expected_version=version)
        result = writer.write_memory(updated, ctx)
        stored = fs.read_node(uri, ctx)
        assert result["target_uri"] == uri
        assert stored.content == text
        assert stored.metadata["routing_key"] == "valve_inspection"
        assert stored.metadata["version"] == version + 1
    assert stored.metadata["routing_key_aliases"] == ["valve_inspection", "r4_verified_inspection"]


def test_person_and_preference_coexist_through_preference_update(ctx):
    from extraction.source_attribution import bind_source_attribution

    fs = MemoryFS()
    writer = api(fs, DecisionLLM())
    person = memory("ren_li", object_name="Ren Li", content="Ren Li enjoys jasmine tea.")
    person_uri = writer.write_memory(person, ctx)["target_uri"]
    preference = memory("tea", category="preference", object_name="Tea",
                        content="Ren Li enjoys jasmine tea.", who="Ren Li",
                        evidence_quote="I enjoy jasmine tea.")
    source = "Ren Li: I enjoy jasmine tea.\nAlex: I prefer coffee."
    preference = bind_source_attribution(preference, source, ctx.user_id)
    uri = writer.write_memory(preference, ctx)["target_uri"]
    updated = replace(preference, routing_key="tea_choice", target_uri=uri,
                      expected_version=1, content="Ren Li prefers jasmine tea without sugar.",
                      evidence_quote="I prefer jasmine tea without sugar.",
                      provenance_ids=["source:followup"])
    updated = bind_source_attribution(
        updated, "Ren Li: I prefer jasmine tea without sugar.\nAlex: I like milk.", ctx.user_id,
    )
    result = writer.write_memory(updated, ctx)

    assert result["target_uri"] == uri
    assert uri.endswith("/preferences/ren_li_beverage")
    assert fs.read_node(uri, ctx).content == updated.content
    assert fs.read_node(person_uri, ctx).content == person.content
    assert len(fs.nodes) == 2


def test_same_topic_preferences_of_two_subjects_keep_stable_scoped_keys(ctx):
    from extraction.source_attribution import bind_source_attribution

    fs = MemoryFS()
    writer = api(fs, DecisionLLM())
    source = "Ren Li: I love dogs.\nAlex: I prefer cats."
    first = bind_source_attribution(memory("pets", category="preference", content="Ren Li loves dogs.",
                                           who="Ren Li", evidence_quote="I love dogs."), source, ctx.user_id)
    second = bind_source_attribution(memory("pets", category="preference", content="Alex prefers cats.",
                                            who="Alex", evidence_quote="I prefer cats."), source, ctx.user_id)
    first_uri = writer.write_memory(first, ctx)["target_uri"]
    second_uri = writer.write_memory(second, ctx)["target_uri"]
    assert first_uri.endswith("/preferences/ren_li_pets")
    assert second_uri.endswith("/preferences/alex_pets")

    resolver = MemoryIdentityResolver(fs, URIResolver(SchemaRegistry()))
    remention = bind_source_attribution(
        replace(memory("pets_alex", category="preference", content="Alex prefers cats over dogs.",
                       who="Alex", evidence_quote="I prefer cats."), provenance_ids=["source:later"]),
        source, ctx.user_id)
    assert resolver.match_existing(remention, ctx).uri == second_uri

    own = bind_source_attribution(memory("pets", category="preference", content="I like birds.",
                                         who="reader", evidence_quote="I like birds."),
                                  "reader: I like birds.\nAlex: I prefer cats.", ctx.user_id)
    assert writer.write_memory(own, ctx)["target_uri"].endswith("/preferences/pets")


@pytest.mark.parametrize("holder,document,conversation,quote", [
    ("Review panel", "Meeting notes: The review panel prefers written briefs.",
     "Avery: The review panel prefers written briefs.\nJordan: Noted.",
     "The review panel prefers written briefs."),
    ("评审组", "项目纪要：评审组更喜欢书面提纲。",
     "许岚：评审组更喜欢书面提纲。\n周宁：记下了。", "评审组更喜欢书面提纲。"),
])
def test_document_and_chat_reuse_preference_subject(ctx, holder, document, conversation, quote):
    from extraction.source_attribution import bind_source_attribution

    fs = MemoryFS()
    candidate = memory("brief_format", category="preference", content=quote,
                       who=holder, evidence_quote=quote)
    document_memory = bind_source_attribution(candidate, document, ctx.user_id)
    uri = api(fs, DecisionLLM()).write_memory(document_memory, ctx)["target_uri"]
    chat_memory = bind_source_attribution(candidate, conversation, ctx.user_id)
    resolver = MemoryIdentityResolver(fs, URIResolver(SchemaRegistry()))

    assert resolver.match_existing(chat_memory, ctx).uri == uri
    assert chat_memory.scope_ref == document_memory.scope_ref == holder
    assert api(fs, DecisionLLM()).write_memory(chat_memory, ctx)["target_uri"] == uri
    assert len(fs.nodes) == 1


@pytest.mark.parametrize("holders,scope", [
    (("Review panel", "Field crew"), "Atlas"),
    (("评审组", "现场团队"), "北辰项目"),
])
def test_project_preferences_keep_each_holder_across_creation_order(ctx, holders, scope):
    allocations = []
    for order in (holders, tuple(reversed(holders))):
        fs = MemoryFS()
        writer = api(fs, DecisionLLM())
        resolver = MemoryIdentityResolver(fs, URIResolver(SchemaRegistry()))
        uris = {}
        for holder in order:
            candidate = memory("brief_format", category="preference", content=f"{holder}: written briefs.",
                               who=holder, scope_ref=scope)
            uri = writer.write_memory(candidate, ctx)["target_uri"]
            uris[holder] = uri
            assert fs.read_node(uri, ctx).metadata["scope_ref"] == scope
        for holder in holders:
            candidate = memory("brief_format", category="preference", content=f"{holder}: concise briefs.",
                               who=holder, scope_ref=scope)
            assert resolver.match_existing(candidate, ctx).uri == uris[holder]
            qualified = replace(candidate, routing_key=uris[holder].rsplit("/", 1)[-1])
            assert resolver.match_existing(qualified, ctx).uri == uris[holder]
            updated = replace(candidate, target_uri=uris[holder], expected_version=1)
            assert writer.write_memory(updated, ctx)["target_uri"] == uris[holder]
            assert fs.read_node(uris[holder], ctx).metadata["who"] == holder
        assert len(fs.nodes) == len(set(uris.values())) == 2
        allocations.append(uris)
    assert allocations[0] == allocations[1]


@pytest.mark.parametrize("stored_holder", ["Review panel", ""])
def test_preference_read_target_keeps_legacy_project_key(ctx, stored_holder):
    fs = MemoryFS()
    writer = api(fs, DecisionLLM())
    legacy = memory("brief_format", category="preference", content="Written briefs.", scope_ref="Atlas")
    uri = writer.write_memory(legacy, ctx)["target_uri"]
    fs.nodes[uri].metadata["who"] = stored_holder
    update = replace(legacy, who="Review panel", target_uri=uri, expected_version=1,
                     content="Concise written briefs.")
    resolver = MemoryIdentityResolver(fs, URIResolver(SchemaRegistry()))

    if stored_holder:
        assert resolver.match_existing(replace(update, target_uri="", expected_version=None), ctx).uri == uri
    assert writer.write_memory(update, ctx)["target_uri"] == uri
    stored = fs.read_node(uri, ctx)
    assert stored.metadata["routing_key"] == "atlas_brief_format"
    assert stored.metadata["who"] == "Review panel"
    assert stored.metadata["scope_ref"] == "Atlas"


@pytest.mark.parametrize("category", ["entity", "preference", "pattern"])
def test_scope_qualified_name_can_exceed_extraction_name_limit(ctx, category):
    scope = "project_" + "a" * 52
    key = "subject_" + "b" * 22
    candidate = memory(key, category=category, scope_ref=scope)
    fs = MemoryFS()
    writer = api(fs, DecisionLLM())

    first = writer.write_memory(candidate, ctx)
    second = writer.write_memory(candidate, ctx)

    assert first["target_uri"].rsplit("/", 1)[-1] == scope + "_" + key
    assert second["target_uri"] == first["target_uri"]
    assert len(fs.nodes) == 1
    assert fs.read_node(first["target_uri"], ctx).metadata["scope_ref"] == scope
