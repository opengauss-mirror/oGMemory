"""Keep specific code evidence when broad matches dominate several routes."""
import copy

import pytest

import core  # noqa: F401 - initialize the shared service imports.
from retrieval.code_retrieval import fuse_route_hits


def hit(name, score=1.0, identity="repo"):
    return dict(uri=f"ctx://acct/resources/{identity}-{name}", score=score,
                metadata=dict(resource_root_slug="repo", relative_path=f"{name}.py", symbol=name))


def routes():
    others = [hit(f"generic_{i}") for i in range(8)]
    target = hit("TestWidgetParser")
    return dict(semantic=[target, *others], symbolic=[*others[:2], target, *others[2:]],
                bm25=others, graph=others), target


@pytest.mark.parametrize("mode", ["weighted_rrf", "weighted_sum"])
@pytest.mark.parametrize("query", ["WidgetParser tests order", "find parsing implementation"])
def test_route_agreement_survives_fusion_and_keeps_scores(mode, query):
    hits, target = routes()
    before = copy.deepcopy(hits)
    original = fuse_route_hits(hits, mode=mode, limit=50)
    assert next(i for i,x in enumerate(original) if x["uri"] == target["uri"]) >= 5
    result = fuse_route_hits(hits, mode=mode, limit=5, query=query)
    assert result[0]["uri"] == target["uri"]
    assert len(result) == len({x["uri"] for x in result}) == 5
    assert hits == before
    old = next(x for x in original if x["uri"] == target["uri"])
    assert result[0]["score"] == old["score"]
    assert result[0]["metadata"]["route_ranks"] == old["metadata"]["route_ranks"]


@pytest.mark.parametrize("query,symbol", [
    ("WidgetParser tests", "TestWidgetParser"), ("read_value usage", "Cache.read_value"),
    ("use `decode` here", "Decoder.decode"), ("HTTPClient implementation", "HTTPClient.send"),
    ("cache.decode implementation", "Cache.decode"),
    ("HTTP implementation", "HTTP.send"), ("Parser", "Parser.parse"),
    ("use `Find` here", "Tree.find"),
])
def test_identifier_match_survives_without_semantic_route(query, symbol):
    hits, _ = routes()
    target = hit(symbol)
    hits["semantic"] = []
    hits["symbolic"] = hits["bm25"] + [target]
    result = fuse_route_hits(hits, query=query, limit=1)
    assert result[0]["uri"] == target["uri"]


@pytest.mark.parametrize("disabled", ["semantic", "symbolic"])
def test_zero_weight_routes_do_not_protect(disabled):
    hits, target = routes()
    result = fuse_route_hits(hits, query="find parsing implementation", weights={disabled: 0}, limit=5)
    assert target["uri"] not in {x["uri"] for x in result}


def test_unsupported_semantic_hit_and_identifier_substring_do_not_promote():
    hits, target = routes()
    hits["symbolic"] = hits["bm25"]
    assert target["uri"] not in {x["uri"] for x in fuse_route_hits(hits, query="find implementation", limit=5)}
    hits["semantic"] = []
    wrong = hit("Bitmap")
    hits["symbolic"] += [wrong]
    assert fuse_route_hits(hits, query="Map", limit=1)[0]["uri"] != wrong["uri"]


def test_two_slots_identity_isolation_and_zero_limit():
    hits, target = routes()
    explicit = hit("ExactThing")
    twin = hit("ExactThing", identity="other-root")
    hits["symbolic"] = [explicit, twin, target, *hits["bm25"]]
    selected = fuse_route_hits(hits, query="ExactThing parse implementation", limit=2)
    assert [x["uri"] for x in selected] == [explicit["uri"], target["uri"]]
    assert {explicit["uri"], twin["uri"]} <= {x["uri"] for x in fuse_route_hits(hits, query="ExactThing", limit=50)}
    assert fuse_route_hits(hits, query="ExactThing", limit=0) == []


@pytest.mark.parametrize("mode", ["weighted_rrf", "weighted_sum"])
@pytest.mark.parametrize("verb", ["Find", "Show", "Get"])
def test_sentence_capitalization_does_not_protect_instruction_verbs(mode, verb):
    target, incidental = hit("parse_config"), hit("Tree." + verb.lower())
    candidates = dict(semantic=[target], bm25=[target], graph=[target], symbolic=[incidental])
    expected = fuse_route_hits(candidates, mode=mode, limit=1)
    for word in (verb.lower(), verb):
        result = fuse_route_hits(candidates, query=f"{word} the parsing implementation", mode=mode, limit=1)
        assert result == expected
