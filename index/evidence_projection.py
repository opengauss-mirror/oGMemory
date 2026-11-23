"""Check canonical source authority before publishing memory index records."""

from contextlib import contextmanager

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
    from service.evidence_directory import read_publication_revision, source_publication_locks
    from service.evidence_lifecycle import ensure_sources_active

    filters = records[0].filters
    ctx = RequestContext(filters["account_id"], filters["owner_space"].removeprefix("user:"),
                         "", "", "")
    try:
        node = fs.read_node(event.uri, ctx)
    except NodeNotFoundError:
        yield False
        return
    with source_publication_locks(
        EvidenceStore(fs), node, ctx, lock_directory=node.metadata.get("node_type") == "directory",
    ):
        try:
            current = read_publication_revision(fs, event.uri, ctx, records[0].metadata.get("canonical_revision"))
            ensure_sources_active(current, fs, ctx)
        except NodeNotFoundError:
            yield False
            return
        yield True
