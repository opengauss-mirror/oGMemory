"""Normal Atlas updates reconcile current facts across their stored cards."""

from dataclasses import replace
from unittest.mock import Mock

import pytest

from core.models import CandidateMemory, ContextNode, SeedHit
from extraction.prefetch import MemoryPrefetcher
from tests.unit.commit.test_memory_identity import MemoryFS
from tests.unit.extraction.test_node_processing import node_payload, setup

SOURCE = "更新我的 Atlas 项目发布习惯：从今天起，周四的发布时间由北京时间20点改为21点，包管理工具继续使用 uv。"


def atlas(category="entity", hour=20):
    key = "atlas" if category == "entity" else "atlas_release_schedule"
    name = "Atlas" if category == "entity" else "Atlas 项目发布时间偏好"
    fact = f"Atlas 项目每周四北京时间{hour}点发布，使用 uv。"
    content = f"# {name}\n\n" + ("## 发布\n" if category == "entity" else "") + "- " + fact
    return CandidateMemory(category, "user", key, fact, "- " + fact, content, .95,
                           who="user" if category == "preference" else None,
                           object_name=name, scope_ref="Atlas",
                           provenance_ids=["prov:1:archive:atlas:first"])


def primary(ids, node=None, *, category="entity", hour=21):
    draft = atlas(category, hour)
    return {"draft_ids": ids, "action": "update" if node else "create",
            "target": node["uri"] if node else "",
            "routing_key": node["routing_key"] if node else draft.routing_key,
            "object_name": node["object_name"] if node else draft.object_name,
            "abstract": draft.abstract, "overview": [draft.abstract],
            "edits": [{"search": "北京时间20点", "replacement": "北京时间21点"}] if node else [],
            "sections": [] if node else [{"heading": "发布", "facts": [draft.abstract]}]}


def related(node, *, decision="update", reason="correct_current_fact"):
    op = primary([0], node, category=node["category"])
    op.pop("draft_ids")
    op.pop("action")
    if decision == "unchanged":
        op["edits"] = []
    return {**op, "source_draft_ids": [0], "decision": decision,
            "reason": reason, "source_quote": "周四的发布时间由北京时间20点改为21点"}


class ReconciliationLLM:
    def __init__(self, decide):
        self.decide, self.calls = decide, []

    def complete_with_tools(self, **kwargs):
        self.calls.append(kwargs)
        assert kwargs["tools"][0]["input_schema"]["required"] == ["operations", "related_reconciliations"]
        return [{"tool": "process_memory_nodes", "input": self.decide(node_payload(kwargs))}]


def prepare(entity_scope="Atlas"):
    ctx, fs, _, api = setup(lambda _: [])
    targets = {category: api.write_memory(replace(atlas(category), scope_ref=entity_scope)
                                         if category == "entity" else atlas(category), ctx)["target_uri"]
               for category in ("entity", "preference")}
    return ctx, fs, api, targets


@pytest.mark.parametrize("embedding_error", [True, False], ids=["timeout", "wrong-count"])
def test_commit_uses_identity_catalog_when_semantic_prefetch_is_unavailable(embedding_error):
    ctx, fs, api, targets = prepare()
    source = replace(atlas(hour=21), source_contexts=[SOURCE])
    embedder = Mock()
    if embedding_error:
        embedder.embed_texts.side_effect = TimeoutError("embedding timed out")
    else:
        embedder.embed_texts.return_value = []
    api._embedder = embedder
    api._vector_index = Mock()
    api._pipeline.extract = lambda *args, **kwargs: [source]

    def decide(payload):
        by_uri = {node["uri"]: node for node in payload["nodes"]}
        return {"operations": [primary([0], by_uri[targets["entity"]])],
                "related_reconciliations": [related(by_uri[targets["preference"]])]}

    api._llm = ReconciliationLLM(decide)
    request = [{"role": "user", "content": SOURCE}]
    try:
        result = api.commit_session(request, ctx)
    except Exception as exc:
        print(f"request={request}; embedding_queries={embedder.embed_texts.call_args}; error={exc!r}")
        raise
    print(f"request={request}; embedding_queries={embedder.embed_texts.call_args}; writes={result['writes_completed']}")
    assert result["writes_completed"] == 2
    assert embedder.embed_texts.call_count == 1
    assert api._vector_index.search_by_vector.call_count == 0
    assert all("北京时间21点" in fs.read_node(uri, ctx).content for uri in targets.values())


def test_commit_reconciles_two_large_cards_for_one_draft():
    ctx, fs, _, api = setup(lambda _: [])
    history = "historical observation " * 2600
    targets = {
        category: api.write_memory(
            replace(atlas(category), content=atlas(category).content + "\n- " + history), ctx,
        )["target_uri"]
        for category in ("entity", "preference")
    }
    stored = {uri: fs.read_node(uri, ctx) for uri in targets.values()}
    draft = replace(atlas(hour=21), source_contexts=[SOURCE])
    api._pipeline.extract = lambda *args, **kwargs: [draft]

    def decide(payload):
        by_uri = {node["uri"]: node for node in payload["nodes"]}
        if target := payload.get("related_target"):
            return {"operations": [], "related_reconciliations": [related(by_uri[target])]}
        operations = [primary([0], by_uri[targets["entity"]])]
        reconciliations = [related(by_uri[targets["preference"]])] if targets["preference"] in by_uri else []
        return {"operations": operations, "related_reconciliations": reconciliations}

    llm = api._llm = ReconciliationLLM(decide)
    request = [{"role": "user", "content": SOURCE}]
    sizes = {node.category: len(node.content) for node in stored.values()}
    try:
        result = api.commit_session(request, ctx)
    except Exception as exc:
        print(f"request={request}; stored_chars={sizes}; model_calls={len(llm.calls)}; error={exc!r}")
        raise
    print(f"request={request}; stored_chars={sizes}; prompt_sizes="
          f"{[len(call['prompt']) for call in llm.calls]}; writes={result['writes_completed']}")
    assert result["writes_completed"] == 2
    assert all(len(call["prompt"]) <= 110000 for call in llm.calls)
    for uri, original in stored.items():
        updated = fs.read_node(uri, ctx)
        assert updated.content == original.content.replace("北京时间20点", "北京时间21点")
        assert updated.metadata["version"] == 2


def test_atlas_annotated_owner_preference_updates_its_existing_card_and_alias():
    ctx, fs, _, api = setup(lambda _: [])
    ctx = replace(ctx, account_id="lcphase1-remedy-20261031-f",
                  user_id="lcphase1-remedy-20261031-f-alice", session_id="alice-update")
    key = "atlas_lcphase1_remedy_20261031_f_alice_用户_release_schedule"
    uri = f"ctx://{ctx.account_id}/users/{ctx.user_id}/memories/preferences/{key}"
    original = atlas("preference")
    fs.write_node(ContextNode(uri, "MEMORY", "preference", 2, ctx.user_space_name(),
                              original.abstract, original.overview, original.content,
                              {"who": ctx.user_id + "（用户）", "scope_ref": "Atlas",
                               "routing_key": key, "routing_key_aliases": ["release_schedule"],
                               "object_name": original.object_name, "status": "ACTIVE"}), ctx)
    entity_uri = api.write_memory(atlas(), ctx)["target_uri"]
    ben = replace(ctx, user_id="lcphase1-remedy-20261031-f-ben", session_id="ben-initial")
    ben_uri = api.write_memory(replace(atlas("preference", 19), who=ben.user_id + "（用户）"), ben)["target_uri"]
    source = replace(atlas("preference", 21), who=ctx.user_id, source_contexts=[SOURCE],
                     provenance_ids=["prov:1:archive:atlas:correction"])

    def decide(payload):
        nodes = {node["category"]: node for node in payload["nodes"]}
        assert set(payload["primary_candidate_targets"]) == {uri}
        assert set(payload["related_targets"]) == {entity_uri}
        assert nodes["preference"]["who"] == ctx.user_id + "（用户）"
        return {"operations": [primary([0], nodes["preference"], category="preference")],
                "related_reconciliations": [related(nodes["entity"])]}

    processor = api._node_processor()
    processor.llm = ReconciliationLLM(decide)
    candidates = processor.process([source], [{"role": "user", "content": SOURCE}], ctx)
    assert {candidate.target_uri for candidate in candidates} == {uri, entity_uri}
    written = [api.write_memory(candidate, ctx) for candidate in candidates]
    assert {result["target_uri"] for result in written} == {uri, entity_uri}
    updated = fs.read_node(uri, ctx)
    assert updated.metadata["version"] == 2
    assert updated.metadata["who"] == ctx.user_id
    assert set(updated.metadata["routing_key_aliases"]) >= {"release_schedule", "atlas_release_schedule"}
    assert all("北京时间21点" in value for value in (updated.abstract, updated.overview, updated.content))
    assert "北京时间21点" in fs.read_node(entity_uri, ctx).content
    assert len(fs.nodes) == 3
    assert fs.read_node(ben_uri, ben).metadata["who"] == ben.user_id
    assert "北京时间19点" in fs.read_node(ben_uri, ben).content


@pytest.mark.parametrize("holder", ["user", "<user>", "用户", "{user}（用户）", "{user} (user)", "{user}（<user>）"])
def test_atlas_self_holder_forms_persist_the_current_owner(holder):
    ctx, fs, _, api = setup(lambda _: [])
    candidate = replace(atlas("preference"), who=holder.format(user=ctx.user_id))
    uri = api.write_memory(candidate, ctx)["target_uri"]
    assert uri.endswith("/preferences/atlas_release_schedule")
    assert fs.read_node(uri, ctx).metadata["who"] == ctx.user_id


def test_atlas_named_holders_keep_their_own_annotated_identities():
    ctx, fs, _, api = setup(lambda _: [])
    holders = ("Ben（用户）", "Review panel (Atlas)")
    uris = []
    for holder in holders:
        candidate = replace(atlas("preference"), who=holder,
                            content=f"# Atlas\n\n- {holder} 的 Atlas 项目每周四北京时间20点发布。")
        uri = api.write_memory(candidate, ctx)["target_uri"]
        uris.append(uri)
        assert fs.read_node(uri, ctx).metadata["who"] == holder
        assert holder in fs.read_node(uri, ctx).content
    assert len(set(uris)) == len(fs.nodes) == 2


@pytest.mark.parametrize("entity_scope", ["Atlas", ""])
def test_atlas_current_correction_updates_both_cards_in_one_node_call(entity_scope):
    ctx, fs, api, targets = prepare(entity_scope)
    ben = replace(ctx, user_id="ben", session_id="ben")
    ben_uri = api.write_memory(atlas(hour=19), ben)["target_uri"]
    source = replace(atlas("preference", 21), provenance_ids=["prov:1:archive:atlas:correction"],
                     source_contexts=[SOURCE], node_draft_indices=[4])

    def decide(payload):
        nodes = {node["category"]: node for node in payload["nodes"]}
        assert nodes["entity"]["uri"] == targets["entity"]
        return {"operations": [primary([0], nodes["preference"], category="preference")],
                "related_reconciliations": [related(nodes["entity"])]}

    llm = ReconciliationLLM(decide)
    processor = api._node_processor()
    processor.llm = llm
    candidates = processor.process([source], [{"role": "user", "content": SOURCE}], ctx)
    assert len(llm.calls) == 1
    assert {candidate.target_uri for candidate in candidates} == set(targets.values())
    assert {candidate.node_reconciliation_role for candidate in candidates} == {"primary", "related"}
    assert len({candidate.node_reconciliation_group for candidate in candidates}) == 1
    assert all(candidate.expected_version == 1 and candidate.node_draft_indices == [4]
               for candidate in candidates)
    for candidate in candidates:
        api.write_memory(candidate, ctx)
    for uri in targets.values():
        node = fs.read_node(uri, ctx)
        assert node.metadata["version"] == 2
        assert all("北京时间21点" in value for value in (node.abstract, node.overview, node.content))
        assert "使用 uv" in node.content
    entity = fs.read_node(targets["entity"], ctx)
    assert entity.metadata["routing_key_aliases"] == ["atlas"]
    assert entity.metadata["object_name_aliases"] == ["Atlas"]
    assert "北京时间19点" in fs.read_node(ben_uri, ben).content


@pytest.mark.parametrize("entity_scope", ["Atlas", ""])
def test_atlas_new_entity_and_preference_share_one_subject_plan(entity_scope):
    ctx, fs, _, api = setup(lambda _: [])
    llm = ReconciliationLLM(lambda payload: {
        "operations": [primary([index], category=draft["category"], hour=20)
                       for index, draft in enumerate(payload["drafts"])],
        "related_reconciliations": [],
    })
    api._llm = llm
    api._pipeline.extract = lambda *args, **kwargs: [replace(atlas(), scope_ref=entity_scope), atlas("preference")]
    result = api.commit_session([{"role": "user", "content": "Atlas 每周四20点发布，使用 uv。"}], ctx)
    assert result["writes_completed"] == 2
    assert len(llm.calls) == 1
    assert {node.category for node in fs.nodes.values()} == {"entity", "preference"}


def test_atlas_history_is_retained_when_current_card_already_matches():
    ctx, fs, api, targets = prepare()
    entity = fs.read_node(targets["entity"], ctx)
    entity.abstract = "Atlas 现为周四21点发布；此前20点。"
    entity.overview = "- Atlas 当前北京时间21点发布。\n- 此前发布时间为20点。"
    entity.content = "# Atlas\n\n## 发布\n- 当前每周四北京时间21点发布。\n- 2026-10-28及以前每周四20点发布。"
    fs.write_node(entity, ctx)

    def decide(payload):
        nodes = {node["category"]: node for node in payload["nodes"]}
        return {"operations": [primary([0], nodes["preference"], category="preference")],
                "related_reconciliations": [related(nodes["entity"], decision="unchanged", reason="already_current")]}

    processor = api._node_processor()
    processor.llm = ReconciliationLLM(decide)
    candidates = processor.process([atlas("preference", 21)], [{"role": "user", "content": SOURCE}], ctx)
    assert len(candidates) == 1
    assert candidates[0].target_uri == targets["preference"]
    assert fs.read_node(targets["entity"], ctx).content == entity.content


@pytest.mark.parametrize("reason,source,quote", [
    ("different_fact", "Atlas 项目部署机房位于上海，包管理工具继续使用 uv。", ""),
    ("historical_only", "Atlas 于2026-10-28按当时的周四20点安排发布；它的部署机房位于上海。",
     "Atlas 项目每周四北京时间21点发布，使用 uv。"),
])
def test_atlas_new_details_preserve_unchanged_related_card_and_reuse_stage(reason, source, quote):
    from tests.unit.extraction.test_parallel_node_stages import Stages

    ctx, fs, _, api = setup(lambda _: [])
    targets = {category: api.write_memory(atlas(category, 21), ctx)["target_uri"]
               for category in ("entity", "preference")}
    original = fs.read_node(targets["preference"], ctx)
    draft = replace(atlas(hour=21), content="# Atlas\n\n## 部署\n- " + source,
                    provenance_ids=["prov:1:archive:atlas:deployment"],
                    source_contexts=[source], node_draft_indices=[0])

    def decide(payload):
        nodes = {node["category"]: node for node in payload["nodes"]}
        entity, preference = nodes["entity"], nodes["preference"]
        return {"operations": [{"draft_ids": [0], "action": "update", "target": entity["uri"],
                 "routing_key": entity["routing_key"], "object_name": entity["object_name"],
                 "abstract": "Atlas 每周四21点发布，使用 uv，部署机房位于上海。",
                 "overview": [line[2:] for line in entity["content"].splitlines() if line.startswith("- ")]
                             + ["部署机房位于上海。"],
                 "edits": [], "sections": [{"heading": "部署", "facts": [source]}]}],
                "related_reconciliations": [{"source_draft_ids": [0], "decision": "unchanged",
                 "reason": reason, "source_quote": quote, "target": preference["uri"],
                 "routing_key": preference["routing_key"], "object_name": preference["object_name"],
                 "abstract": "", "overview": [],
                 "edits": [], "sections": []}]}

    processor = api._node_processor()
    processor.llm = ReconciliationLLM(decide)
    stages = Stages()
    messages = [{"role": "user", "content": source}]
    candidates = processor.process([draft], messages, ctx, checkpoint=stages)
    assert processor.process([draft], messages, ctx, checkpoint=stages) == candidates
    assert len(processor.llm.calls) == 1
    candidate, = candidates
    assert candidate.target_uri == targets["entity"] and candidate.expected_version == 1
    assert candidate.source_contexts == [source] and candidate.provenance_ids == draft.provenance_ids
    assert candidate.node_draft_indices == [0]
    assert source in candidate.content and atlas(hour=21).content in candidate.content
    result = api.write_memory(candidate, ctx)
    assert result["target_uri"] == targets["entity"]
    assert fs.read_node(targets["entity"], ctx).metadata["version"] == 2
    assert fs.read_node(targets["preference"], ctx) == original


@pytest.mark.parametrize("category", ["entity", "preference"])
def test_complete_l2_drives_update_summaries_and_noop_retains_the_stored_card(category):
    from copy import deepcopy
    from tests.unit.extraction.test_parallel_node_stages import Stages

    ctx, fs, _, api = setup(lambda _: [])
    runtime_fact = "Atlas 服务运行环境为 Python 3.12。"
    original = replace(atlas(category), content=atlas(category).content + "\n- " + runtime_fact)
    target = api.write_memory(original, ctx)["target_uri"]
    source = replace(atlas(category, 21), source_contexts=[SOURCE],
                     provenance_ids=["prov:1:archive:atlas:current"], node_draft_indices=[0])

    def decide(payload):
        node, = payload["nodes"]
        facts = [line[2:] for line in node["content"].splitlines() if line.startswith("- ")]
        assert runtime_fact in facts and payload["source"] == [SOURCE]
        common = {"draft_ids": [0], "target": target, "routing_key": node["routing_key"],
                  "object_name": node["object_name"], "sections": []}
        if node["version"] == 1:
            search, = [line for line in node["content"].splitlines() if "北京时间20点" in line]
            current_facts = [fact.replace("北京时间20点", "北京时间21点") for fact in facts]
            op = {**common, "action": "update", "abstract": " ".join(current_facts),
                  "overview": current_facts,
                  "edits": [{"search": search, "replacement": search.replace("北京时间20点", "北京时间21点")}]}
        else:
            assert node["version"] == 2 and any("北京时间21点" in fact for fact in facts)
            op = {**common, "action": "noop", "abstract": "", "overview": [], "edits": []}
        return {"operations": [op], "related_reconciliations": []}

    processor = api._node_processor()
    processor.llm = ReconciliationLLM(decide)
    stages = Stages()
    messages = [{"role": "user", "content": SOURCE}]
    updated, = processor.process([source], messages, ctx, checkpoint=stages)
    assert processor.process([source], messages, ctx, checkpoint=stages) == [updated]
    assert len(processor.llm.calls) == 1
    assert updated.expected_version == 1 and updated.target_uri == target
    assert updated.content == original.content.replace("北京时间20点", "北京时间21点")
    assert all("北京时间21点" in text and "uv" in text and runtime_fact in text
               for text in (updated.abstract, updated.overview, updated.content))
    api.write_memory(updated, ctx)
    stored = deepcopy(fs.read_node(target, ctx))
    acknowledged, = processor.process([source], messages, ctx, checkpoint=stages)
    assert len(processor.llm.calls) == 2 and acknowledged.expected_version == 2
    assert (acknowledged.abstract, acknowledged.overview, acknowledged.content) == (
        stored.abstract, stored.overview, stored.content)
    assert acknowledged.provenance_ids == source.provenance_ids
    assert acknowledged.source_contexts == [SOURCE] and acknowledged.node_draft_indices == [0]
    assert api.write_memory(acknowledged, ctx)["action"] == "skip"
    assert fs.read_node(target, ctx) == stored


def test_joint_primary_updates_cover_related_targets_with_an_explicit_empty_array():
    ctx, fs, api, targets = prepare()

    def decide(payload):
        nodes = {node["category"]: node for node in payload["nodes"]}
        assert set(payload["related_targets"]) == set(targets.values())
        return {"operations": [primary([index], nodes[draft["category"]], category=draft["category"])
                               for index, draft in enumerate(payload["drafts"])],
                "related_reconciliations": []}

    processor = api._node_processor()
    processor.llm = ReconciliationLLM(decide)
    candidates = processor.process([atlas(hour=21), atlas("preference", 21)],
                                   [{"role": "user", "content": SOURCE}], ctx)
    assert len(processor.llm.calls) == 1
    assert {candidate.target_uri for candidate in candidates} == set(targets.values())
    for candidate in candidates:
        api.write_memory(candidate, ctx)
    assert all(fs.read_node(uri, ctx).metadata["version"] == 2 for uri in targets.values())
    assert all("北京时间21点" in fs.read_node(uri, ctx).content for uri in targets.values())


def test_atlas_primary_update_and_noops_accept_matching_unchanged_acknowledgements():
    from tests.unit.extraction.test_parallel_node_stages import Stages

    ctx, fs, _, api = setup(lambda _: [])
    source = "请记住我的 Atlas 项目发布习惯：每周四北京时间21点发布，包管理工具统一使用 uv，发布前运行 pytest，部署机房在上海，服务运行环境为 Python 3.12。"
    facts = [("atlas", "entity", "Atlas 每周四北京时间21点发布，使用 uv。"),
             ("atlas_release_schedule", "preference", "Atlas 每周四北京时间21点发布。"),
             ("atlas_package_manager", "preference", "Atlas 包管理工具统一使用 uv。"),
             ("atlas_release_checks", "preference", "Atlas 发布前运行 pytest。"),
             ("atlas_deployment", "preference", "Atlas 部署机房在上海。"),
             ("atlas_runtime", "preference", "Atlas 服务运行环境为 Python 3.12。")]
    drafts = [replace(atlas(category, 21), routing_key=key, abstract=fact, overview="- " + fact,
                      content="# Atlas\n\n" + ("## 发布\n" if category == "entity" else "") + "- " + fact,
                      source_contexts=[source], node_draft_indices=[index])
              for index, (key, category, fact) in enumerate(facts)]
    targets = [api.write_memory(draft, ctx)["target_uri"] for draft in drafts[:3]]

    def decide(payload):
        nodes = {node["routing_key"]: node for node in payload["nodes"]}
        operations = []
        for index, draft in enumerate(drafts):
            node = nodes.get(draft.routing_key)
            op = {"draft_ids": [index], "action": "create", "target": "",
                  "routing_key": draft.routing_key, "object_name": draft.object_name,
                  "abstract": draft.abstract, "overview": [draft.abstract], "edits": [],
                  "sections": [{"heading": "发布", "facts": [draft.abstract]}]}
            if node:
                op.update(action="noop", target=node["uri"], object_name=node["object_name"], sections=[])
            if index == 0:
                op.update(action="update", abstract="Atlas 每周四21点发布，使用 uv，发布前运行 pytest，在上海运行 Python 3.12。",
                          overview=[draft.abstract, *[item[2] for item in facts[3:]]],
                          sections=[{"heading": "发布检查与部署", "facts": [item[2] for item in facts[3:]]}])
            operations.append(op)
        acknowledgements = []
        for index, quote in ((0, source), (2, "包管理工具统一使用 uv"), (1, "每周四北京时间21点发布")):
            node = nodes[drafts[index].routing_key]
            ack = related(node, decision="unchanged", reason="already_current")
            ack.update(source_draft_ids=[index], source_quote=quote)
            acknowledgements.append(ack)
        return {"operations": operations, "related_reconciliations": acknowledgements}

    stages = Stages()
    processor = api._node_processor()
    processor.llm = ReconciliationLLM(decide)
    candidates = processor.process(drafts, [{"role": "user", "content": source}], ctx, checkpoint=stages)
    assert len(processor.llm.calls) == 1
    assert len(candidates) == 6
    assert all(candidate.node_reconciliation_role == "primary" for candidate in candidates)
    assert [candidate.target_uri for candidate in candidates[:3]] == targets
    results = [api.write_memory(candidate, ctx) for candidate in candidates]
    assert len({result["target_uri"] for result in results}) == 6
    entity = fs.read_node(targets[0], ctx)
    assert entity.metadata["version"] == 2
    assert all(fact in entity.content for fact in ("21点", "uv", "pytest", "上海", "Python 3.12"))
    saved = next(iter(stages.values.values()))[0]["input"]
    assert len(saved["operations"]) == 6
    assert len(saved["related_reconciliations"]) == 3


def test_ben_new_command_preserves_acknowledged_existing_preferences():
    from copy import deepcopy
    from tests.unit.extraction.test_parallel_node_stages import Stages

    ctx, fs, _, api = setup(lambda _: [])
    ctx = replace(ctx, user_id="ben")
    source = "补充我的 Atlas 项目发布习惯：每周五北京时间19点发布，仍使用 npm，发布前必须运行 npm test。"

    def preference(key, fact):
        return CandidateMemory("preference", "user", key, fact, "- " + fact,
                               "# Atlas 项目偏好\n\n- " + fact, .95,
                               who="ben", object_name="Atlas 项目偏好", scope_ref="Atlas")

    previous = {}
    for key, fact in (("atlas_release_schedule", "Atlas 每周五北京时间19点发布。"),
                      ("atlas_package_manager", "Atlas 包管理工具使用 npm。")):
        uri = api.write_memory(preference(key, fact), ctx)["target_uri"]
        previous[uri] = deepcopy(fs.read_node(uri, ctx))
    draft = replace(preference("atlas_release_checks", "Atlas 发布前必须运行 npm test。"),
                    source_contexts=[source], node_draft_indices=[0])

    def decide(payload):
        assert payload["related_targets"] == []
        assert set(payload["primary_candidate_targets"]) == set(previous)
        unchanged = [{"source_draft_ids": [0], "decision": "unchanged", "reason": "already_current",
                      "source_quote": source, "target": node["uri"], "routing_key": node["routing_key"],
                      "object_name": node["object_name"], "abstract": "",
                      "overview": [], "edits": [], "sections": []}
                     for node in payload["nodes"]]
        return {"operations": [{"draft_ids": [0], "action": "create", "target": "",
                  "routing_key": draft.routing_key, "object_name": draft.object_name,
                  "abstract": draft.abstract, "overview": [draft.abstract], "edits": [],
                  "sections": [{"heading": "发布检查", "facts": [draft.abstract]}]}],
                "related_reconciliations": unchanged}

    stages = Stages()
    processor = api._node_processor()
    processor.llm = ReconciliationLLM(decide)
    candidates = processor.process([draft], [{"role": "user", "content": source}], ctx, checkpoint=stages)
    assert len(candidates) == len(processor.llm.calls) == 1
    stored = api.write_memory(candidates[0], ctx)["target_uri"]
    assert "npm test" in fs.read_node(stored, ctx).content
    assert all(fs.read_node(uri, ctx) == node for uri, node in previous.items())
    saved = next(iter(stages.values.values()))[0]["input"]
    assert len(saved["operations"]) == 1 and len(saved["related_reconciliations"]) == 2
    assert {entry["target"] for entry in saved["related_reconciliations"]} == set(previous)


def test_atlas_combined_preference_acknowledges_candidates_with_both_roles():
    from copy import deepcopy
    from tests.unit.extraction.test_parallel_node_stages import Stages

    ctx, fs, _, api = setup(lambda _: [])
    source = "请记住我的 Atlas 项目发布习惯：每周四北京时间21点发布，包管理工具统一使用 uv，发布前运行 pytest，部署机房在上海，服务运行环境为 Python 3.12。"
    entity_uri = api.write_memory(atlas(hour=21), ctx)["target_uri"]
    previous = {}
    for key, fact in (("atlas_release_schedule", "Atlas 每周四北京时间21点发布。"),
                      ("atlas_package_manager", "Atlas 包管理工具统一使用 uv。")):
        draft = replace(atlas("preference", 21), routing_key=key, abstract=fact,
                        overview="- " + fact, content="# Atlas 项目偏好\n\n- " + fact)
        uri = api.write_memory(draft, ctx)["target_uri"]
        previous[uri] = deepcopy(fs.read_node(uri, ctx))
    facts = "Atlas 每周四北京时间21点发布，使用 uv，发布前运行 pytest，部署机房在上海，运行 Python 3.12。"
    drafts = [replace(atlas(hour=21), source_contexts=[source]),
              replace(atlas("preference", 21), routing_key="atlas_release_and_toolchain",
                      object_name="Atlas 发布与工具链偏好", abstract=facts, overview="- " + facts,
                      content="# Atlas 发布与工具链偏好\n\n- " + facts, source_contexts=[source])]

    def decide(payload):
        assert set(previous) <= set(payload["related_targets"])
        assert set(previous) <= set(payload["primary_candidate_targets"])
        nodes = {node["uri"]: node for node in payload["nodes"]}
        entity = nodes[entity_uri]
        acknowledgements = [{"source_draft_ids": [1], "decision": "unchanged", "reason": "already_current",
                             "source_quote": source, "target": uri, "routing_key": node["routing_key"],
                             "object_name": node["object_name"], "abstract": "",
                             "overview": [], "edits": [], "sections": []}
                            for uri, node in nodes.items() if uri in previous]
        return {"operations": [
            {"draft_ids": [0], "action": "update", "target": entity_uri,
             "routing_key": entity["routing_key"], "object_name": entity["object_name"],
             "abstract": facts, "overview": [facts], "edits": [],
             "sections": [{"heading": "发布检查与部署", "facts": ["发布前运行 pytest，部署机房在上海，运行 Python 3.12。"]}]},
            {"draft_ids": [1], "action": "create", "target": "", "routing_key": drafts[1].routing_key,
             "object_name": drafts[1].object_name, "abstract": facts, "overview": [facts], "edits": [],
             "sections": [{"heading": "发布与工具链", "facts": [facts]}]}],
            "related_reconciliations": acknowledgements}

    stages = Stages()
    processor = api._node_processor()
    processor.llm = ReconciliationLLM(decide)
    candidates = processor.process(drafts, [{"role": "user", "content": source}], ctx, checkpoint=stages)
    assert len(processor.llm.calls) == 1
    assert len(candidates) == 2
    results = [api.write_memory(candidate, ctx) for candidate in candidates]
    assert all(all(fact in fs.read_node(result["target_uri"], ctx).content
                   for fact in ("21点", "uv", "pytest", "上海", "Python 3.12")) for result in results)
    assert all(fs.read_node(uri, ctx) == node for uri, node in previous.items())
    saved = next(iter(stages.values.values()))[0]["input"]
    assert len(saved["operations"]) == 2 and len(saved["related_reconciliations"]) == 2


def test_accepted_related_target_replans_from_its_latest_complete_snapshot():
    ctx, fs, api, targets = prepare()
    original = replace(atlas("preference", 21), node_draft_indices=[7], source_contexts=[SOURCE])
    accepted = replace(atlas(), target_uri=targets["entity"], expected_version=1,
                       node_draft_indices=[7], node_reconciliation_role="related",
                       node_reconciliation_group="atlas-correction")
    latest = fs.read_node(targets["entity"], ctx)
    latest.content += "\n## 部署\n- 部署在上海，运行 Python 3.12。"
    fs.write_node(latest, ctx)

    def decide(payload):
        node = payload["nodes"][0]
        assert payload["fixed_target"] == targets["entity"]
        assert node["version"] == 2
        assert "上海" in node["content"] and "Python 3.12" in node["content"]
        op = primary([0], node)
        op["overview"].append("部署在上海，运行 Python 3.12。")
        return {"operations": [op], "related_reconciliations": []}

    processor = api._node_processor()
    processor.llm = ReconciliationLLM(decide)
    revised = processor.process([original], [{"role": "user", "content": SOURCE}], ctx,
                                target_candidate=accepted)
    assert len(revised) == 1
    candidate = revised[0]
    assert candidate.category == "entity" and candidate.expected_version == 2
    assert candidate.node_reconciliation_role == "related"
    assert candidate.node_reconciliation_group == "atlas-correction"
    api.write_memory(candidate, ctx)
    stored = fs.read_node(targets["entity"], ctx)
    assert stored.metadata["version"] == 3
    assert all(text in stored.content for text in ("北京时间21点", "上海", "Python 3.12"))


@pytest.mark.parametrize("binding", ["read_target", "source_name"])
def test_atlas_identity_is_recalled_through_the_complete_catalog(binding):
    class PagedFS(MemoryFS):
        def __init__(self):
            super().__init__()
            self.pages, self.reads = [], []

        def list_identity_node_summaries_page(self, directory, ctx, *, limit, after_uri):
            uris = [uri for uri in self.list_children(directory, ctx) if after_uri is None or uri > after_uri]
            self.pages.append((directory, len(uris[:limit])))
            return [replace(self.nodes[uri], content="", overview="") for uri in uris[:limit]]

        def read_node(self, uri, ctx):
            self.reads.append(uri)
            return super().read_node(uri, ctx)

    ctx, original_fs, api, targets = prepare()
    fs = PagedFS()
    template = original_fs.read_node(targets["entity"], ctx)
    directory = targets["entity"].rsplit("/", 1)[0]
    for index in range(500):
        uri = f"{directory}/a_{index:04d}"
        fs.nodes[uri] = replace(template, uri=uri, metadata={
            "version": 1, "scope_ref": f"Project {index}", "routing_key": f"a_{index:04d}",
            "object_name": f"Project {index}",
        })
    target = directory + "/z_atlas"
    fs.nodes[target] = replace(template, uri=target)
    processor = api._node_processor()
    processor.resolver.fs = fs
    draft = replace(atlas(), target_uri=target, expected_version=1) if binding == "read_target" else atlas()
    recalled = processor._recall([draft], ctx, {})
    assert [node.uri for node in recalled] == [target]
    assert sum(count for _, count in fs.pages) == 501
    assert len([page for page in fs.pages if page[0] == directory]) == 21
    assert fs.reads.count(target) == 1


@pytest.mark.parametrize("stored_scope,incoming_scope", [
    ("", "Atlas"), ("Atlas", ""), ("Atlas Workspace", ""),
    ("Atlas 项目", ""), ("", "Atlas 项目"), ("Atlas 项目", "Atlas"),
])
def test_entity_self_scope_and_empty_scope_update_the_same_complete_card(stored_scope, incoming_scope):
    ctx, fs, _, api = setup(lambda _: [])
    original = replace(atlas(hour=21), scope_ref=stored_scope, verified_object_aliases=["Atlas Workspace"])
    target = api.write_memory(original, ctx)["target_uri"]
    processor = api._node_processor()

    for version, (source, fact) in enumerate([
        ("补充 Atlas 项目信息：部署机房在上海。", "Atlas 部署机房在上海。"),
        ("补充 Atlas 项目信息：服务运行环境是 Python 3.12。", "Atlas 服务运行环境是 Python 3.12。"),
    ], start=1):
        draft = replace(atlas(hour=21), scope_ref=incoming_scope, abstract=fact, overview="- " + fact,
                        content="# Atlas\n\n## 部署\n- " + fact, source_contexts=[source],
                        provenance_ids=[f"prov:1:archive:atlas:addition-{version}"])

        def decide(payload):
            node, = payload["nodes"]
            assert node["uri"] == target and node["version"] == version
            assert "北京时间21点" in node["content"] and "uv" in node["content"]
            facts = [line[2:] for line in node["content"].splitlines() if line.startswith("- ")] + [fact]
            resolution = ({"scope_resolution": {"stored_scope_role": "self", "incoming_scope_role": "self",
                           "source_quotes": [source]}} if target in payload["scope_candidate_targets"] else {})
            return {"operations": [{"draft_ids": [0], "action": "update", "target": target,
                     "routing_key": node["routing_key"], "object_name": node["object_name"],
                     "abstract": " ".join(facts), "overview": facts,
                     "edits": [], "sections": [{"heading": "部署", "facts": [fact]}], **resolution}],
                    "related_reconciliations": []}

        processor.llm = ReconciliationLLM(decide)
        candidate, = processor.process([draft], [{"role": "user", "content": source}], ctx)
        assert len(processor.llm.calls) == 1
        assert candidate.target_uri == target and candidate.scope_ref == stored_scope
        assert api.write_memory(candidate, ctx)["target_uri"] == target
        stored = fs.read_node(target, ctx)
        assert stored.metadata["version"] == version + 1
        assert stored.metadata["scope_ref"] == stored_scope

    assert len(fs.nodes) == 1
    assert all(fact in stored.content for fact in ("21点", "uv", "上海", "Python 3.12"))


def test_entity_scope_resolution_combines_source_grounded_drafts_in_one_update():
    ctx, fs, _, api = setup(lambda _: [])
    target = api.write_memory(replace(atlas(hour=21), scope_ref="Atlas 项目"), ctx)["target_uri"]
    sources = ["补充 Atlas 项目信息：部署机房在上海。", "补充 Atlas 项目信息：服务运行环境是 Python 3.12。"]
    facts = ["Atlas 部署机房在上海。", "Atlas 服务运行环境是 Python 3.12。"]
    drafts = [replace(atlas(hour=21), scope_ref="", content="# Atlas\n\n## 部署\n- " + fact,
                      abstract=fact, overview="- " + fact, source_contexts=[source],
                      provenance_ids=[f"prov:1:archive:atlas:detail-{index}"])
              for index, (source, fact) in enumerate(zip(sources, facts))]

    def decide(payload):
        node, = payload["nodes"]
        assert payload["scope_candidate_targets"] == [target]
        existing = [line[2:] for line in node["content"].splitlines() if line.startswith("- ")]
        return {"operations": [{"draft_ids": [0, 1], "action": "update", "target": target,
                 "routing_key": node["routing_key"], "object_name": node["object_name"],
                 "abstract": " ".join(existing + facts), "overview": existing + facts,
                 "edits": [], "sections": [{"heading": "部署", "facts": facts}],
                 "scope_resolution": {"stored_scope_role": "self", "incoming_scope_role": "self",
                                      "source_quotes": sources}}], "related_reconciliations": []}

    processor = api._node_processor()
    processor.llm = ReconciliationLLM(decide)
    candidate, = processor.process(drafts, [{"role": "user", "content": source} for source in sources], ctx)
    assert len(processor.llm.calls) == 1
    assert candidate.scope_ref == "Atlas 项目"
    assert candidate.provenance_ids == [draft.provenance_ids[0] for draft in drafts]
    assert api.write_memory(candidate, ctx)["target_uri"] == target
    stored = fs.read_node(target, ctx)
    assert len(fs.nodes) == 1 and stored.metadata["version"] == 2
    assert all(fact in stored.content for fact in ["21点", "uv", "上海", "Python 3.12"])


def test_same_named_entity_in_a_new_external_project_creates_its_own_card():
    ctx, fs, _, api = setup(lambda _: [])
    target = api.write_memory(replace(atlas(hour=21), scope_ref="Project North"), ctx)["target_uri"]
    original = fs.read_node(target, ctx)
    source = "Project South 新增了另一个 Atlas 服务：每周五19点发布，使用 npm。"
    fact = "Project South 的 Atlas 服务每周五19点发布，使用 npm。"
    draft = replace(atlas(), scope_ref="Project South", content="# Atlas\n\n## 发布\n- " + fact,
                    abstract=fact, overview="- " + fact, source_contexts=[source])

    def decide(payload):
        assert payload["scope_candidate_targets"] == [target]
        return {"operations": [{"draft_ids": [0], "action": "create", "target": "",
                 "routing_key": "atlas", "object_name": "Atlas", "abstract": fact, "overview": [fact],
                 "edits": [], "sections": [{"heading": "发布", "facts": [fact]}]}],
                "related_reconciliations": []}

    processor = api._node_processor()
    processor.llm = ReconciliationLLM(decide)
    candidate, = processor.process([draft], [{"role": "user", "content": source}], ctx)
    created = api.write_memory(candidate, ctx)["target_uri"]
    assert len(fs.nodes) == 2 and len(processor.llm.calls) == 1
    assert fs.read_node(target, ctx) == original
    stored = fs.read_node(created, ctx)
    assert stored.metadata["scope_ref"] == "Project South" and stored.metadata["version"] == 1
    assert stored.content.endswith(fact)


def test_same_named_entities_in_external_project_scopes_update_independently():
    ctx, fs, _, api = setup(lambda _: [])
    scopes = ("Project North", "Project South")
    targets = {scope: api.write_memory(replace(atlas(hour=21), scope_ref=scope), ctx)["target_uri"]
               for scope in scopes}
    processor = api._node_processor()
    for scope, city in zip(scopes, ("上海", "深圳")):
        source = f"补充 {scope} 项目下的 Atlas 服务信息：部署机房在{city}。"
        fact = f"Atlas 部署机房在{city}。"
        draft = replace(atlas(hour=21), scope_ref=scope, abstract=fact, overview="- " + fact,
                        content="# Atlas\n\n## 部署\n- " + fact, source_contexts=[source])

        def decide(payload):
            assert set(payload["primary_candidate_targets"]) == {targets[scope]}
            node, = payload["nodes"]
            assert node["uri"] == targets[scope]
            facts = [line[2:] for line in node["content"].splitlines() if line.startswith("- ")] + [fact]
            return {"operations": [{"draft_ids": [0], "action": "update", "target": node["uri"],
                     "routing_key": node["routing_key"], "object_name": node["object_name"],
                     "abstract": " ".join(facts), "overview": facts,
                     "edits": [], "sections": [{"heading": "部署", "facts": [fact]}]}],
                    "related_reconciliations": []}

        processor.llm = ReconciliationLLM(decide)
        candidate, = processor.process([draft], [{"role": "user", "content": source}], ctx)
        assert api.write_memory(candidate, ctx)["target_uri"] == targets[scope]
    assert len(fs.nodes) == 2
    for scope, city in zip(scopes, ("上海", "深圳")):
        node = fs.read_node(targets[scope], ctx)
        assert node.metadata["scope_ref"] == scope and node.metadata["version"] == 2
        assert node.content.endswith(f"Atlas 部署机房在{city}。")


@pytest.mark.parametrize("scope_resolution", [False, True], ids=["known-scope", "self-scope-fallback"])
def test_same_named_project_catalog_keeps_current_target_with_bounded_scope_candidates(scope_resolution):
    from extraction.node_processing import _PROMPT_LIMIT, _READ_LIMIT

    ctx, fs, _, api = setup(lambda _: [])
    stored_scope = "Atlas 项目" if scope_resolution else "Project Current"
    target = api.write_memory(replace(atlas(hour=21), scope_ref=stored_scope), ctx)["target_uri"]
    other_targets = []
    for index in range(36):
        scope = f"Project {index:02d}"
        history = "\n".join(f"- 发布批次 {batch:03d} 使用 Python 3.12，在上海机房完成部署与验收。"
                            for batch in range(600))
        other = replace(atlas(hour=21), scope_ref=scope, content="# Atlas\n\n## 部署历史\n" + history)
        other_targets.append(api.write_memory(other, ctx)["target_uri"])
    source = ("补充 Atlas 项目信息：部署机房在上海。" if scope_resolution
              else "补充 Project Current 项目下的 Atlas 服务信息：部署机房在上海。")
    fact = "Atlas 部署机房在上海。"
    draft = replace(atlas(hour=21), scope_ref="" if scope_resolution else stored_scope, abstract=fact, overview="- " + fact,
                    content="# Atlas\n\n## 部署\n- " + fact, source_contexts=[source])
    processor = api._node_processor()
    recalled = processor._recall([draft], ctx, {})
    assert target in {node.uri for node in recalled}
    assert len(recalled) <= _READ_LIMIT + 1

    def decide(payload):
        if scope_resolution:
            assert target in payload["scope_candidate_targets"]
            assert 1 <= len(payload["scope_candidate_targets"]) < _READ_LIMIT
        else:
            assert payload["primary_candidate_targets"] == [target]
            assert len(payload["nodes"]) == 1
        node = next(node for node in payload["nodes"] if node["uri"] == target)
        facts = [line[2:] for line in node["content"].splitlines() if line.startswith("- ")] + [fact]
        resolution = ({"scope_resolution": {"stored_scope_role": "self", "incoming_scope_role": "self",
                       "source_quotes": [source]}} if scope_resolution else {})
        return {"operations": [{"draft_ids": [0], "action": "update", "target": target,
                 "routing_key": node["routing_key"], "object_name": node["object_name"],
                 "abstract": " ".join(facts), "overview": facts,
                 "edits": [], "sections": [{"heading": "部署", "facts": [fact]}], **resolution}],
                "related_reconciliations": []}

    processor.llm = ReconciliationLLM(decide)
    candidate, = processor.process([draft], [{"role": "user", "content": source}], ctx)
    assert len(processor.llm.calls) == 1
    assert len(processor.llm.calls[0]["prompt"]) <= _PROMPT_LIMIT
    assert api.write_memory(candidate, ctx)["target_uri"] == target
    stored = fs.read_node(target, ctx)
    assert stored.metadata["version"] == 2 and stored.content.endswith(fact)
    assert all(fs.read_node(uri, ctx).metadata["version"] == 1 for uri in other_targets)
    assert len(fs.nodes) == 37


def test_entity_and_preference_semantic_recall_share_the_embedding_batch():
    ctx, fs, api, targets = prepare()

    class Embedder:
        def __init__(self):
            self.calls = []

        def embed_texts(self, texts):
            self.calls.append(texts)
            return [[1.0, 0.0] for _ in texts]

    class Index:
        def __init__(self):
            self.filters = []

        def search_by_vector(self, *, query_vector, filters, top_k):
            self.filters.append(filters)
            return [SeedHit(uri=targets[filters["category"]], score=.9,
                            category=filters["category"], abstract="Atlas", level=2)]

    embedder, index = Embedder(), Index()
    resolver = api._writer._identity_resolver
    prefetcher = MemoryPrefetcher(fs, index, embedder, resolver.schemas, resolver.uris)
    nodes = prefetcher.prefetch_node_snapshots(["entity", "preference"], ctx, SOURCE)
    assert set(nodes) == set(targets.values())
    assert len(embedder.calls) == 1
    assert {item["category"] for item in index.filters} == {"entity", "preference"}
    assert all(item["account_id"] == ctx.account_id and item["owner_space"] == ctx.user_space_name()
               for item in index.filters)


def test_multiple_projects_keep_their_own_current_facts():
    ctx, fs, api, targets = prepare("")
    other = []
    for name, scope in (("Orion", ""), ("Atlas Docs", "Atlas")):
        draft = replace(atlas(hour=19), routing_key=name.lower().replace(" ", "_"),
                        object_name=name, scope_ref=scope,
                        content=f"# {name}\n\n## 发布\n- 每周五19点发布，使用 npm。")
        uri = api.write_memory(draft, ctx)["target_uri"]
        other.append((uri, fs.read_node(uri, ctx)))

    def decide(payload):
        assert {node["uri"] for node in payload["nodes"]} == set(targets.values())
        nodes = {node["category"]: node for node in payload["nodes"]}
        return {"operations": [primary([0], nodes["preference"], category="preference")],
                "related_reconciliations": [related(nodes["entity"])]}

    processor = api._node_processor()
    processor.llm = ReconciliationLLM(decide)
    candidates = processor.process([atlas("preference", 21)], [{"role": "user", "content": SOURCE}], ctx)
    for candidate in candidates:
        api.write_memory(candidate, ctx)
    assert all(fs.read_node(uri, ctx) == original for uri, original in other)
    assert all("北京时间21点" in fs.read_node(uri, ctx).content for uri in targets.values())


def test_bound_pronoun_uses_the_read_entity_subject_for_related_preferences():
    ctx, fs, api, targets = prepare("")
    source = "把它的周四发布时间从20点改为21点，继续使用 uv。"
    draft = replace(atlas(hour=21), object_name="它", scope_ref="", target_uri=targets["entity"],
                    expected_version=1, source_contexts=[source])

    def decide(payload):
        nodes = {node["category"]: node for node in payload["nodes"]}
        linked = related(nodes["preference"])
        linked["source_quote"] = "周四发布时间从20点改为21点"
        return {"operations": [primary([0], nodes["entity"])], "related_reconciliations": [linked]}

    processor = api._node_processor()
    processor.llm = ReconciliationLLM(decide)
    candidates = processor.process([draft], [{"role": "user", "content": source}], ctx)
    assert {candidate.target_uri for candidate in candidates} == set(targets.values())
    for candidate in candidates:
        api.write_memory(candidate, ctx)
    assert all("北京时间21点" in fs.read_node(uri, ctx).content for uri in targets.values())
