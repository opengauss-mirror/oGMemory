"""Request-local native relations carried through the candidate contract.

Carry these through the existing candidate contract, remove them before storage,
and resolve them to actual stored nodes after writing, without another LLM pass.
No global collector: parallel spans and concurrent sessions remain isolated.
"""
import json
import logging
import math

KEY = "_extraction_relation_proposals"
logger = logging.getLogger(__name__)


def parse_proposal(value):
    if not isinstance(value, dict):
        return None
    result = {}
    for key in ("from_routing_key", "to_routing_key", "relation_type", "reason"):
        item = value.get(key)
        if not isinstance(item, str) or not item.strip():
            logger.warning("Invalid extract_relation argument: %s", key)
            return None
        result[key] = item.strip()
    if result["relation_type"] not in {"alias_of", "supersedes", "belongs_to", "supports", "refers_to"}:
        return None
    for key in ("from_category", "to_category"):
        item = value.get(key, "entity")
        if not isinstance(item, str) or not item.strip():
            return None
        result[key] = item.strip()
    weight = value.get("weight")
    if (isinstance(weight, bool) or not isinstance(weight, (int, float))
            or not math.isfinite(weight) or not 0 <= weight <= 1):
        return None
    result["weight"] = weight
    return result


def attach(candidates, proposals):
    # Every candidate carries a copy so confidence filtering cannot accidentally
    # discard all proposals just because the first candidate was filtered out.
    if proposals:
        for candidate in candidates:
            candidate.metadata[KEY] = [dict(p) for p in proposals]


def take(candidates):
    proposals, seen = [], set()
    for candidate in candidates:
        for proposal in candidate.metadata.pop(KEY, []):
            marker = json.dumps(proposal, sort_keys=True)
            if marker not in seen:
                seen.add(marker)
                proposals.append(proposal)
    return proposals


def resolve_edges(proposals, candidates, plans, ctx, *, fs, uri_resolver=None):
    """Resolve native endpoints using successful write plans and authorized reads.

    Never guess timestamped URIs or invent endpoints. Alignment failure makes
    positional pairing unsafe. Multiple stored nodes for a key are ambiguous.
    """
    from core.models import RelationEdge

    if len(candidates) != len(plans):
        return [], ["candidate_write_alignment_failed"] * len(proposals)
    refs = {}
    for candidate, plan in zip(candidates, plans, strict=True):
        if plan.action in {"create", "merge", "append"}:
            # Reconciliation may rename a candidate after extract_relation
            # emitted its original key. Only verified aliases are accepted;
            # the sets below preserve ambiguity rejection across candidates.
            for key in (candidate.routing_key, *candidate.verified_routing_aliases):
                if key:
                    refs.setdefault((candidate.category, key), set()).add(plan.target_uri)

    def endpoint(category, key):
        choices = refs.get((category, key), set())
        if len(choices) > 1:
            raise ValueError("ambiguous_endpoint")
        if choices:
            uri = next(iter(choices))
        elif uri_resolver is not None:
            uri = uri_resolver.resolve(category, {"routing_key": key}, ctx)
        else:
            raise ValueError("missing_endpoint")
        node = fs.read_node(uri, ctx)
        if node is None or node.category != category:
            raise ValueError("missing_endpoint")
        return uri

    edges, rejected = [], []
    for raw in proposals:
        proposal = parse_proposal(raw)
        if proposal is None:
            rejected.append("invalid_native_relation")
            continue
        try:
            source = endpoint(proposal["from_category"], proposal["from_routing_key"])
            target = endpoint(proposal["to_category"], proposal["to_routing_key"])
            if source == target:
                raise ValueError("self_edge")
            edges.append(RelationEdge(source, target, proposal["relation_type"],
                                      proposal["weight"], proposal["reason"]))
        except Exception as exc:
            rejected.append(str(exc) if isinstance(exc, ValueError) else type(exc).__name__)
    return edges, rejected


def union_edges(native, generated):
    """Historical native-first union, deduplicated by directed typed endpoints."""
    result, seen = [], set()
    for edge in [*native, *generated]:
        key = (edge.from_uri, edge.relation_type, edge.to_uri)
        if key not in seen:
            seen.add(key)
            result.append(edge)
    return result
