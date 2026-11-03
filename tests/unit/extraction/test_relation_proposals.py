from unittest.mock import Mock

import pytest

from core.models import CandidateMemory, RequestContext
from extraction import Extractor
from extraction.relation_proposals import KEY, attach, parse_proposal, take
from extraction.tools import _Span

PROPOSAL = dict(from_routing_key="alice", to_routing_key="project",
                relation_type="belongs_to", reason="Alice works on the project", weight=.9)


def test_verified_alias_resolves_and_ambiguous_alias_is_rejected():
    from core.models import WritePlan
    from extraction.relation_proposals import resolve_edges
    candidates = [CandidateMemory('entity', 'user', key, key, key, key, .9)
                  for key in ('alicia', 'project')]
    candidates[0].verified_routing_aliases = ['alice']
    plans = [WritePlan('create', 'stored/alicia'), WritePlan('merge', 'stored/project')]
    fs = Mock()
    fs.read_node.return_value = Mock(category='entity')
    edges, rejected = resolve_edges([PROPOSAL], candidates, plans, Mock(), fs=fs)
    assert not rejected and edges[0].from_uri == 'stored/alicia'
    candidates.append(CandidateMemory('entity','user','alice','a','a','a',.9))
    plans.append(WritePlan('create','stored/different-alice'))
    edges, rejected = resolve_edges([PROPOSAL], candidates, plans, Mock(), fs=fs)
    assert not edges and rejected == ['ambiguous_endpoint']


@pytest.mark.parametrize("bad", [dict(weight=True), dict(weight=float("nan")),
    dict(weight=2), dict(reason=""), dict(from_routing_key=None), dict(relation_type="invalid")])
def test_malformed_proposals_are_rejected(bad):
    assert parse_proposal({**PROPOSAL, **bad}) is None


def test_proposals_are_request_local_deduplicated_and_removed_before_storage():
    candidates = [CandidateMemory("entity", "user", key, key, key, key, .9)
                  for key in ("alice", "project")]
    attach(candidates, [parse_proposal(PROPOSAL)])
    assert len(take(candidates)) == 1
    assert all(KEY not in candidate.metadata for candidate in candidates)
    assert take(candidates) == []


def test_eager_extract_relation_is_not_a_memory_and_is_preserved():
    llm = Mock()
    llm.complete_with_tools.return_value = [
        {"tool": "extract_entity", "input": dict(routing_key="alice", abstract="Alice",
            overview="Alice", content="Alice works on a project", confidence=.9,
            update_activation="Alice works on a project.")},
        {"tool": "extract_relation", "input": PROPOSAL},
    ]
    extractor = Extractor(llm, activation_enabled=True)
    candidates = extractor._structure_span_eager(_Span(0, 0, "test", ["entity"]),
        "prompt", RequestContext("acct", "alice", "agent", "session", "trace"), "hash")
    assert len(candidates) == 1 and candidates[0].category == "entity"
    assert take(candidates)[0]["from_routing_key"] == "alice"


def test_lazy_prompt_uses_graph_template():
    from extraction.extraction_react_loop import ExtractionReActLoop, ExtractionTaskContext
    from extraction.schemas.registry import SchemaRegistry
    loop = ExtractionReActLoop(Mock(), Mock(), SchemaRegistry(), Mock(),
                              Extractor(Mock())._prompt_manager, activation_enabled=True)
    messages = loop._build_messages(RequestContext("acct", "alice", "agent", "session", "trace"),
                                    ExtractionTaskContext(conversation_text="facts"))
    assert "UPDATE ACTIVATION — ONE SENTENCE PER MEMORY" in messages[0]["content"]
    assert 'update_activation: "Caroline' in messages[0]["content"]


def test_separate_generator_does_not_reconsider_native_relations():
    from extraction.cross_prompt_relations import build_cross_prompt_relation_prompt
    without = build_cross_prompt_relation_prompt([], [], [], [])
    assert "NODE-EXTRACTION RELATION PROPOSALS" not in without


def test_native_resolution_uses_actual_timestamped_write_uris_and_typed_union():
    from core.models import WritePlan, RelationEdge
    from extraction.relation_proposals import resolve_edges, union_edges
    candidates = [CandidateMemory("entity", "user", key, key, key, key, .9)
                  for key in ("alice", "project")]
    plans = [WritePlan("create", "ctx://stored/alice-20261012"),
             WritePlan("append", "ctx://stored/project-20261012")]
    fs = Mock()
    fs.read_node.return_value = Mock(category="entity")
    edges, rejected = resolve_edges([PROPOSAL], candidates, plans, Mock(), fs=fs)
    assert rejected == [] and len(edges) == 1
    assert edges[0].from_uri == plans[0].target_uri
    assert edges[0].to_uri == plans[1].target_uri
    alternative = RelationEdge(edges[0].from_uri, edges[0].to_uri, "supports", .8, "facts")
    assert union_edges(edges, [edges[0], alternative]) == [edges[0], alternative]


@pytest.mark.parametrize("problem", ["missing", "ambiguous", "alignment", "self", "denied"])
def test_native_resolution_abstains_on_invalid_endpoints(problem):
    from core.models import WritePlan
    from extraction.relation_proposals import resolve_edges
    candidates = [CandidateMemory("entity", "user", key, key, key, key, .9)
                  for key in ("alice", "project")]
    plans = [WritePlan("create", "a"), WritePlan("create", "b")]
    fs = Mock()
    fs.read_node.return_value = Mock(category="entity")
    if problem == "missing": fs.read_node.return_value = None
    if problem == "denied": fs.read_node.side_effect = PermissionError()
    if problem == "self": plans[1].target_uri = "a"
    if problem == "alignment": plans.pop()
    if problem == "ambiguous":
        candidates.append(candidates[0]); plans.append(WritePlan("create", "other-a"))
    edges, rejected = resolve_edges([PROPOSAL], candidates, plans, Mock(), fs=fs)
    assert not edges and len(rejected) == 1


def test_dual_run_merge_retains_proposals_from_both_passes():
    candidates = [CandidateMemory("entity", "user", "alice", "Alice", "Alice", "Alice", .9)
                  for _ in range(2)]
    attach(candidates[:1], [parse_proposal(PROPOSAL)])
    attach(candidates[1:], [parse_proposal({**PROPOSAL, "relation_type": "supports"})])
    merged = Extractor._merge_dual_run(candidates)
    assert len(merged) == 1 and len(take(merged)) == 2


@pytest.mark.parametrize("mode", ["fallback", "lazy-json", "lazy-tools"])
def test_proposals_survive_other_extraction_paths(mode):
    operations = [dict(tool="extract_entity", input=dict(routing_key="alice",
        abstract="Alice", overview="Alice", content="Alice works on a project", confidence=.9,
        update_activation="Alice works on a project.")),
        dict(tool="extract_relation", input=PROPOSAL)]
    ctx = RequestContext("acct", "alice", "agent", "session", "trace")
    llm = Mock()
    extractor = Extractor(llm, activation_enabled=True)
    if mode == "fallback":
        from extraction.tools import _PreparedSpan
        llm.complete_json.return_value = dict(operations=operations)
        candidates = extractor._structure_span_json_fallback(
            _PreparedSpan(_Span(0, 0), "prompt", "hash"), ctx, None)
    else:
        from extraction.extraction_react_loop import ExtractionReActLoop
        from extraction.schemas.registry import SchemaRegistry
        loop = ExtractionReActLoop(llm, Mock(), SchemaRegistry(), Mock(),
                                  extractor._prompt_manager, activation_enabled=True)
        if mode == "lazy-json":
            candidates = loop._candidates_from_json(dict(operations=operations))
        else:
            result = Mock(trace=None, output=[], iterations=1, finish_reason="completed",
                          completed=True, llm_error=None,
                          tools_used=[dict(tool_name=o["tool"], params=o["input"],
                                           result=dict(status="recorded")) for o in operations])
            candidates = loop._to_extraction_react_result(result, ctx).candidates
    assert len(candidates) == 1
    assert len(take(candidates)) == 1
