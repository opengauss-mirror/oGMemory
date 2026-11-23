"""Check canonical source authority before publishing memory index records."""

from contextlib import ExitStack, contextmanager

from core.models import RequestContext
from service.evidence_store import EvidenceStore


def requires_source_guard(records):
    return any(record.metadata.get("extraction_method") == "deterministic"
               or record.metadata.get("evidence_derived") for record in records)


@contextmanager
def memory_projection_guard(fs, event, records):
    """Fence the final vector write against source revocation, after model work."""
    if not requires_source_guard(records):
        yield True
        return
    from core.errors import NodeNotFoundError
    from service.evidence_directory import directory_lock, source_refs
    from service.evidence_lifecycle import ensure_sources_active

    filters = records[0].filters
    ctx = RequestContext(filters["account_id"], filters["owner_space"].removeprefix("user:"),
                         "", "", "")
    try:
        node = fs.read_node(event.uri, ctx)
    except NodeNotFoundError:
        yield False
        return
    sessions = sorted({ref["session_id"] for ref in source_refs(node)})
    with ExitStack() as stack:
        for session in sessions:
            stack.enter_context(EvidenceStore(fs).session_lock(ctx, session_id=session))
        if node.metadata.get("node_type") == "directory":
            stack.enter_context(directory_lock(fs, node.uri, ctx))
        try:
            current = fs.read_node(event.uri, ctx)
            ensure_sources_active(current, fs, ctx)
            valid = current.metadata.get("version") == records[0].metadata.get("canonical_revision")
        except NodeNotFoundError:
            valid = False
        yield valid
