"""Optional topic identity work must preserve facts and legal write availability."""
from dataclasses import asdict, replace

import pytest

from commit.topic_aliases import reconcile_topic_aliases
from core.models import RequestContext
from core.operation_deadline import operation_deadline
from tests.unit.commit.test_memory_identity import MemoryFS, MergeLLM, api, memory


class Replies:
    def __init__(self, groups, valid=True, fail=False):
        self.groups, self.valid, self.fail = groups, valid, fail
        self.prompts = []

    def complete_json(self, prompt, schema):
        self.prompts.append(prompt)
        if self.fail:
            raise TimeoutError("optional provider unavailable")
        return {"groups": self.groups} if len(self.prompts) == 1 else {"valid": self.valid}


@pytest.fixture
def setup():
    ctx = RequestContext(account_id="topic-test", user_id="rae", agent_id="agent",
                         session_id="source", trace_id="trace")
    fs = MemoryFS()
    writer = api(fs, MergeLLM())
    return ctx, fs, writer, writer._writer._identity_resolver


def prefs():
    return [memory("unsweetened_drinks", category="preference", who="Rae",
                   content="Rae prefers tea without sugar."),
            memory("sugar_free_beverages", category="preference", who="Rae",
                   content="Rae also orders coffee without sugar.")]


def reconcile(setup, candidates, llm):
    ctx, _, _, resolver = setup
    return reconcile_topic_aliases(candidates, [{"role": "user", "content": "My drink choices"}],
                                   ctx, resolver, llm)


def test_new_aliases_preserve_every_draft_fact_and_receipt_field(setup):
    original = prefs()
    result = reconcile(setup, original, Replies([[0, 1]]))
    assert len(result) == 2
    assert result[1].routing_key == result[0].routing_key
    for before, after in zip(original, result):
        a, b = asdict(before), asdict(after)
        for field in ("routing_key", "verified_routing_aliases", "verified_object_aliases"):
            a.pop(field)
            b.pop(field)
        assert a == b
    assert result[1].verified_routing_aliases == [original[1].routing_key]
    assert original[1].routing_key == "sugar_free_beverages"
    ctx, fs, writer, _ = setup
    for candidate in result:
        writer.write_memory(candidate, ctx)
    assert len(fs.nodes) == 1


def test_cross_source_alias_reuses_uri_without_inventing_read_binding(setup):
    ctx, fs, writer, _ = setup
    first = writer.write_memory(prefs()[0], ctx)
    incoming = replace(prefs()[1], provenance_ids=["source-two"])
    result = reconcile(setup, [incoming], Replies([[0, 1]]))
    assert result[0].target_uri == "" and result[0].expected_version is None
    assert result[0].content == incoming.content
    plan = writer.write_memory(result[0], ctx)
    assert plan["target_uri"] == first["target_uri"]
    assert len(fs.nodes) == 1


@pytest.mark.parametrize("category", ["entity", "preference", "pattern"])
@pytest.mark.parametrize("existing", [False, True])
def test_verified_alias_reuses_identity_after_catalog_growth(setup, category, existing):
    ctx, fs, writer, resolver = setup
    first = memory("field_receiver", category=category, object_name="field receiver", who="Rae")
    incoming = replace(first, routing_key="portable_radio", object_name="portable radio",
                       content="The portable radio has a blue handle.", provenance_ids=["second-source"],
                       verified_routing_aliases=["previous_radio"],
                       verified_object_aliases=["previous receiver"])
    if existing:
        writer.write_memory(first, ctx)
    verified = reconcile(setup, [incoming] if existing else [first, incoming], Replies([[0, 1]]))
    for candidate in verified:
        uri = writer.write_memory(candidate, ctx)["target_uri"]
    stored = fs.read_node(uri, ctx)
    assert resolver._names_match(stored, incoming)
    assert set(stored.metadata["routing_key_aliases"]) >= {"portable_radio", "previous_radio"}
    assert set(stored.metadata["object_name_aliases"]) >= {"portable radio", "previous receiver"}

    for index in range(16):
        writer.write_memory(replace(first, routing_key=f"unrelated_{index}",
                                    object_name=f"unrelated {index}"), ctx)
    latest = replace(incoming, content="The radio has a green carrying case.",
                     provenance_ids=["third-source"], verified_routing_aliases=[], verified_object_aliases=[])
    result = reconcile(setup, [latest], Replies([[0, 1]]))
    plan = writer.write_memory(result[0], ctx)
    assert plan["target_uri"] == uri
    assert len(fs.nodes) == 17
    assert set(fs.read_node(uri, ctx).metadata["provenance_ids"]) >= {"second-source", "third-source"}


@pytest.mark.parametrize("category", ["entity", "pattern"])
@pytest.mark.parametrize("existing", [False, True])
def test_reporters_share_the_same_topic_identity(setup, category, existing):
    ctx, fs, writer, _ = setup
    drafts = [memory("field_receiver", category=category, object_name="field receiver", who="Mira"),
              memory("portable_radio", category=category, object_name="portable radio", who="Theo",
                     content="It has a blue handle.")]
    if existing:
        writer.write_memory(drafts[0], ctx)
        drafts = drafts[1:]
    result = reconcile(setup, drafts, Replies([[0, 1]]))
    assert {c.routing_key for c in result} == {"field_receiver"}
    for candidate in result:
        writer.write_memory(candidate, ctx)
    assert len(fs.nodes) == 1
    assert next(iter(fs.nodes.values())).metadata["who"] == "Theo"


def test_preference_holders_keep_their_own_alias_groups(setup):
    ctx, fs, writer, _ = setup
    drafts = [*prefs(), *(replace(c, who="Noor") for c in prefs())]
    result = reconcile(setup, drafts, Replies([[0, 1], [2, 3]]))
    assert [c.who for c in result] == ["Rae", "Rae", "Noor", "Noor"]
    assert result[0].routing_key == result[1].routing_key
    assert result[2].routing_key == result[3].routing_key
    for candidate in result:
        writer.write_memory(candidate, ctx)
    assert len(fs.nodes) == 2
    assert {node.metadata["who"] for node in fs.nodes.values()} == {"Rae", "Noor"}


@pytest.mark.parametrize("change", [dict(who="Noor"), dict(scope_ref="work"),
                                     dict(category="entity", object_name="Noor")])
def test_hard_boundaries_never_reach_model_together_when_each_is_unique(setup, change):
    candidates = prefs()
    candidates[1] = replace(candidates[1], **change)
    llm = Replies([[0, 1]])
    assert reconcile(setup, candidates, llm) == candidates
    assert not llm.prompts


@pytest.mark.parametrize("groups", [[[0, 0], [1]], [[0]], [[0, True]], [[-1, 0, 1]],
                                    [[0, 1, 2]], "invalid"])
def test_bad_protocol_preserves_all_original_candidates(setup, groups):
    candidates = prefs()
    result = reconcile(setup, candidates, Replies(groups))
    assert result is candidates


@pytest.mark.parametrize("fail,valid", [(True, True), (False, False)])
def test_optional_timeout_or_identity_rejection_does_not_block_write(setup, fail, valid):
    candidates = prefs()
    assert reconcile(setup, candidates, Replies([[0, 1]], valid=valid, fail=fail)) is candidates


def test_near_deadline_keeps_time_for_normal_writes(setup):
    candidates, llm = prefs(), Replies([[0, 1]])
    with operation_deadline(100):
        assert reconcile(setup, candidates, llm) is candidates
    assert not llm.prompts


def test_bound_and_accepted_drafts_never_changed(setup):
    candidates = [replace(prefs()[0], target_uri="ctx://target", expected_version=7),
                  replace(prefs()[1], write_operation_id="already-accepted")]
    llm = Replies([[0, 1]])
    assert reconcile(setup, candidates, llm) is candidates
    assert not llm.prompts


def test_existing_separate_nodes_are_not_consolidated(setup):
    ctx, _, writer, _ = setup
    for candidate in prefs():
        writer.write_memory(candidate, ctx)
    candidate = replace(prefs()[0], routing_key="beverage_choices", provenance_ids=["source-two"])
    original = [candidate]
    assert reconcile(setup, original, Replies([[0, 1, 2]])) is original


def test_exact_existing_identity_uses_original_merge_without_alias_model_calls(setup):
    ctx, _, writer, _ = setup
    writer.write_memory(prefs()[0], ctx)
    incoming = [replace(prefs()[0], content="Rae also avoids sugar in iced tea.")]
    llm = Replies([[0, 1]])
    assert reconcile(setup, incoming, llm) is incoming
    assert not llm.prompts


def test_oversize_source_skips_optional_stage_without_truncation(setup):
    ctx, _, _, resolver = setup
    candidates, llm = prefs(), Replies([[0, 1]])
    assert reconcile_topic_aliases(candidates, [{"role": "user", "content": "x" * 60000}],
                                   ctx, resolver, llm) is candidates
    assert not llm.prompts
