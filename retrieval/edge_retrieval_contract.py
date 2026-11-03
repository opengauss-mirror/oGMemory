"""Stable origins and outcomes shared by semantic graph retrieval and tracing."""
from enum import StrEnum

from core.uri_parser import strip_l2_suffix


def canonical_revision(block):
    """Return a concrete canonical revision; missing/unversioned data fails closed."""
    metadata = getattr(block, "metadata", None) or {}
    value = metadata.get("version", metadata.get("canonical_revision"))
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return None
    try:
        revision = int(value)
    except (TypeError, ValueError):
        return None
    return revision if revision > 0 else None


def matches_context_type(block, context_type):
    """Apply the effective query scope to the canonically read node type."""
    if not context_type:
        return True
    return str(getattr(block, "context_type", "")).upper() == str(context_type).upper()


def graph_snapshot_matches(block, metadata, ctx, read_node, *, recheck_target=False):
    """Recheck an edge/path snapshot before mixing its score with canonical text.

    Only native graph hits carry these bindings. Legacy third-party retrievers
    retain their previous contract. Reads share the existing caller ACL path.
    """
    bindings = metadata.get("graph_revision_bindings")
    if bindings is None:
        return True
    if not isinstance(bindings, dict) or not bindings or read_node is None:
        return False
    target = strip_l2_suffix(block.uri)
    context_type = metadata.get("graph_context_type")
    if (target not in bindings or canonical_revision(block) != bindings[target]
            or not matches_context_type(block, context_type)):
        return False
    try:
        for uri, expected in bindings.items():
            if isinstance(expected, bool) or not isinstance(expected, int) or expected < 1:
                return False
            current = block if uri == target and not recheck_target else read_node(uri, ctx)
            status = (current.metadata.get("status") or current.metadata.get("lifecycle_status")
                      or current.metadata.get("visibility") or "")
            if (canonical_revision(current) != expected
                    or str(status).lower() in {"archived", "deleted", "broken", "retired", "inactive",
                                               "superseded", "tombstoned", "expired"}
                    or not matches_context_type(current, context_type)):
                return False
    except Exception:
        return False
    return True


class RetrievalOrigin(StrEnum):
    VECTOR_ANCHOR = "vector_anchor"
    GRAPH_NEIGHBOR = "graph_neighbor"


class GraphExpansionOutcome(StrEnum):
    DISABLED = "disabled"
    NO_EDGES = "no_edges"
    ALL_REJECTED = "all_rejected"
    ADDED_NEIGHBORS = "added_neighbors"
    GRAPH_FAILURE = "graph_failure"
    QUERY_EMBEDDING_FAILURE = "query_embedding_failure"
    HYDRATION_REJECTION = "hydration_rejection"
