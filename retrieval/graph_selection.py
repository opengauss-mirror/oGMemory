"""Query-only selection contract used by experiment 33's fresh control.

The control retained the orientation-aware fresh-k3 evidence, not any of the
upstream-inspired experimental selectors. These constants describe that exact
method; changing them is a new retrieval method, not a compatibility repair.
"""
from collections import Counter
import math
import re
from core.vector_math import cosine_similarity

REFERENCE_METHOD = "exp33-fresh-control-orientation-aware-k3-v1"
PREGRAPH_CANDIDATE_LIMIT = 15
MAX_ANCHORS = 5
MAX_HOPS = 2
MAX_EXPANDED_NODES = 30
MAX_DESTINATIONS = 128
LOCAL_LANE_LIMIT = 3
MAX_GRAPH_NODES = 15
SIMILARITY_THRESHOLD = .35
HOP_DECAY = .8

STOP = set("a an the of to for in on with and or is are was were be been being my me i you your how what why when where which can should would could do does did have has had it its this that these those from as at by answer question directly".split())


def cosine(left, right):
    return cosine_similarity(left, right, strict_dimensions=True,
                             require_finite=True, combined_norm=True)


def tokens(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP]


def two_lane_order(records, nodes, query, local_limit=None):
    """URI-unique semantic-first interleave; local cap is per lane, not per edge."""
    if not records:
        return []
    docs = [tokens("\n".join(nodes[r["uri"]].get(k) or "" for k in
                            ("abstract", "overview", "content"))) for r in records]
    frequencies = Counter(t for d in docs for t in set(d))
    average = sum(map(len, docs))/len(docs) or 1.0
    lexical = []
    for record, doc in zip(records, docs):
        counts = Counter(doc)
        score = 0.0
        for term in sorted(set(tokens(query))):
            tf = counts[term]
            if tf:
                score += math.log(1+(len(docs)-frequencies[term]+.5)/(frequencies[term]+.5))*tf*2.5/(tf+1.5*(.25+.75*len(doc)/average))
        lexical.append(dict(record, lexical_score=float(score)))
    lexical.sort(key=lambda r: (-r["lexical_score"], -r["score"], r["uri"]))
    semantic = sorted(records, key=lambda r: (-r["score"], r["uri"]))
    by_uri = {r["uri"]: r for r in lexical}
    sr = {r["uri"]: i for i, r in enumerate(semantic, 1)}
    qr = {r["uri"]: i for i, r in enumerate(lexical, 1)}
    if local_limit is not None:
        semantic, lexical = semantic[:local_limit], lexical[:local_limit]
    lanes = {"semantic": {r["uri"] for r in semantic}, "query": {r["uri"] for r in lexical}}
    seen, result = set(), []
    for i in range(max(len(semantic), len(lexical))):
        for lane in (semantic, lexical):
            if i < len(lane) and lane[i]["uri"] not in seen:
                uri = lane[i]["uri"]
                seen.add(uri)
                result.append(dict(by_uri[uri], semantic_rank=sr[uri], query_rank=qr[uri],
                                   selection_lanes=[name for name, values in lanes.items() if uri in values]))
    return result
