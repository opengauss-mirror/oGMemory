"""Normalize scene provenance and resolve selected evidence to known sources."""

from collections.abc import Sequence
from typing import Any
from urllib.parse import urlsplit

from core.models import ContextNode
from core.provenance_resolver import ProvenanceResolver


def node_source_refs(node: ContextNode) -> tuple[str, ...]:
    """Deduplicate complete references, preserving sequence order and sorting sets."""
    metadata = node.metadata or {}
    refs: list[str] = []
    for key in ("provenance_ids", "source_refs", "source_ref"):
        value = metadata.get(key, [])
        if isinstance(value, str):
            value = [value]
        elif isinstance(value, set):
            value = sorted(str(item) for item in value)
        if isinstance(value, (list, tuple)):
            refs.extend(str(item) for item in value)
    return tuple(dict.fromkeys(ref for ref in refs if ref.strip()))


def source_refs(card: dict[str, Any]) -> set[str]:
    refs = set()
    for key in ("source_refs", "provenance_ids"):
        values = card.get(key, [])
        if isinstance(values, str):
            values = [values]
        if isinstance(values, (list, tuple, set)):
            refs.update(ref for ref in values if isinstance(ref, str) and ref.strip())
    # A tool's source_ref locates the card; original provenance identifies its evidence.
    reference = card.get("source_ref")
    if not refs and isinstance(reference, str) and reference.strip():
        refs.add(reference)
    return refs


def independent_sources(refs):
    sources = set()
    archive_ids = set()
    for ref in refs:
        try:
            source = ProvenanceResolver.parse_id(ref)
            kind, identifier = source["source_type"], source["source_id"]
        except ValueError:
            kind, identifier = "reference", ref
        try:
            parts = urlsplit(identifier)
        except ValueError:
            sources.add((kind, identifier))
            continue
        path = parts.path.strip("/").split("/")
        if (kind in ("archive", "memory", "reference") and parts.scheme == "ctx" and parts.netloc
                and len(path) == 4 and path[0] == "sessions" and path[2] == "history"
                and path[1] and path[3]):
            kind, identifier = "archive", f"ctx://{parts.netloc}/{'/'.join(path)}"
            archive_ids.add(path[3])
        elif (kind in ("reference", "memory", "archive") and parts.scheme == "ctx" and parts.netloc
                and len(path) >= 4 and path[0] in {"users", "agents"} and path[2] == "memories"):
            kind = "memory"
        sources.add((kind, identifier))
    # Full archive URIs retain tenant/session scope; matching short IDs are aliases.
    return sources - {("archive", archive_id) for archive_id in archive_ids}


def supporting_sources(cards: Sequence[dict[str, Any]], context: Sequence[dict[str, Any]]) -> set[tuple[str, str]]:
    return resolve_sources(cards, context)[0]


def resolve_sources(
    cards: Sequence[dict[str, Any]], context: Sequence[dict[str, Any]],
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """Return independent roots and all reached identities for selected provenance."""
    parents: dict[tuple[str, str], set[str]] = {}
    for card in context:
        uri = card.get("uri")
        if isinstance(uri, str) and uri:
            for identity in independent_sources([uri]):
                if identity[0] == "memory":
                    parents.setdefault(identity, set()).update(source_refs(card))
    pending = []
    for card in cards:
        refs = source_refs(card)
        uri = card.get("uri")
        if isinstance(uri, str) and uri:
            for identity in independent_sources([uri]):
                refs.update(parents.get(identity, ()))
        pending.extend(refs)
    visited, roots, reached = set(), set(), set()
    while pending:
        ref = pending.pop()
        if ref in visited:
            continue
        visited.add(ref)
        identity = next(iter(independent_sources([ref])))
        reached.add(identity)
        provenance = parents.get(identity, set())
        if independent_sources(provenance) - {identity}:
            pending.extend(provenance)
        else:
            roots.add(ref)
    return independent_sources(roots), reached


def with_source_groups(packet: dict[str, Any]) -> dict[str, Any]:
    """Expose the validator's resolved source identities to the scene model."""
    cards = packet["memories"]
    context = [*cards, *(card for result in packet.get("tool_evidence", [])
                        for card in result.get("results", []) if isinstance(card, dict))]
    roots = [supporting_sources([card], context) for card in cards]
    groups = sorted(set().union(*roots))
    indices = {source: index for index, source in enumerate(groups)}
    aliases: dict[int, list[int]] = {}
    for index, (kind, identifier) in enumerate(groups):
        if kind == "archive" and independent_sources([identifier]) == {("archive", identifier)}:
            short = indices.get(("archive", urlsplit(identifier).path.rsplit("/", 1)[-1]))
            if short is not None and short != index:
                aliases.setdefault(short, []).append(index)
    return {
        **packet,
        "source_groups": groups,
        "source_group_aliases": list(aliases.items()),
        "memories": [{**card, "source_group_ids": sorted(indices[source] for source in sources)}
                     for card, sources in zip(cards, roots)],
    }
