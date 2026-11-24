"""Promote corroborated observations and withdraw derived memories when support changes."""

import json
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from itertools import tee

from commit.dreaming import iter_dreaming_peers
from commit.dreaming_policy import Claim, Evidence, evidence_score, valid_at
from core.dreaming_evidence import (
    archive_evidence_ref,
    archive_proof_matches,
    archive_user_messages,
    matches_node_version,
    node_created_at,
    timestamp,
)
from core.enums import EventType, NodeStatus
from core.errors import AccessDeniedError, NodeBrokenError, NodeNotFoundError
from core.models import ContextNode, OutboxEvent
from core.validation import MAX_ABSTRACT_LEN

PROMOTION = "dreaming_promotion"
DEMOTION = "dreaming_demotion"
_PROMOTION_INPUT = "dreaming_promotion_input"
_PROMOTION_BLOCKED = "dreaming_promotion_blocked"
_MISSING = (AccessDeniedError, NodeBrokenError, NodeNotFoundError)
_TARGETS = {("user", "event"): ("pattern", "patterns"),
            ("agent", "case"): ("experience", "experiences")}


class PromotionArchiveBlockedError(RuntimeError):
    def __init__(self, blocked):
        self.blocked = blocked
        self.terminal = all(item["status"] == "FAILED" for item in blocked.values())
        message = ("Dreaming promotion blocked by failed archive projection" if self.terminal
                   else "Dreaming promotion awaits archive projection")
        super().__init__(message)


def _active(node, now):
    review = node.metadata.get("dreaming_review", {})
    return (
        matches_node_version(node.metadata, review.get("version"), review.get("node_created_at"))
        and review.get("lifecycle", {}).get("eligibility") != "retired"
        and valid_at(now, valid_from=timestamp(review.get("valid_from")),
                     valid_to=timestamp(review.get("valid_to")))
    )


def _facts(node, now):
    if not _active(node, now):
        return {}
    return {
        key: fact for key, fact in node.metadata["dreaming_review"].get("facts", {}).items()
        if fact.get("content")
        and valid_at(now, valid_from=timestamp(fact.get("valid_from")),
                    valid_to=timestamp(fact.get("valid_to")))
        and all(proof.get("supports") is True for proof in fact.get("evidence", []))
    }


def _identity(node, fact):
    metadata = node.metadata
    return Claim(
        subject=metadata.get("object_name") or metadata.get("who") or node.owner_space,
        predicate=fact.get("slot") or node.category,
        value=fact.get("value") or " ".join(fact["content"].split()),
        scope=metadata.get("scope_ref") or "",
        qualifiers=(metadata.get("where") or "",),
    ).identity()


def _promotion_destination(node, identity, directory):
    owner_uri = node.uri.split("/memories/", 1)[0]
    digest = sha256(repr(identity).encode()).hexdigest()[:24]
    routing = f"dream_{digest}"
    return f"{owner_uri}/memories/{directory}/{routing}", routing


def _destination_state(fs, uri, ctx):
    try:
        metadata = fs.read_node_metadata_including_inactive(uri, ctx)
    except NodeNotFoundError:
        return None
    return metadata.get("status"), metadata["version"], node_created_at(metadata)


def plan_promotion(fs, outbox, node, ctx, assessment, now, event):
    target = _TARGETS.get((node.owner_space.split(":", 1)[0], node.category))
    if target is None:
        return False
    current = replace(node, metadata={**node.metadata, "dreaming_review": assessment})
    ready = [(key, _identity(current, fact), fact["evidence"])
             for key, fact in sorted(_facts(current, now).items()) if fact.get("evidence")]
    if not ready:
        return False
    destinations = sorted({_promotion_destination(node, identity, target[1])[0] for _, identity, _ in ready})
    states = [_destination_state(fs, uri, ctx) for uri in destinations]
    signature = sha256(json.dumps(
        [node_created_at(node.metadata), node.metadata["version"], ready, states], sort_keys=True,
    ).encode()).hexdigest()
    if event.payload.get(_PROMOTION_INPUT) != signature:
        previous = node.metadata.get("dreaming_review", {}).get(_PROMOTION_INPUT)
        if (isinstance(previous, dict) and previous.get("signature") == signature
                and (previous_event := previous.get("event_id"))
                and outbox.get_event_status(node.uri, previous_event) == "DONE"):
            assessment[_PROMOTION_INPUT] = previous
            return False
        event.payload[_PROMOTION_INPUT] = signature
        outbox.save_payload(event)
    assessment[_PROMOTION_INPUT] = {"signature": signature, "event_id": event.event_id}
    return True


def _read_archive_source(fs, archive_id, owner_space, ctx, source_uri):
    kind, owner = owner_space.split(":", 1)
    uri = source_uri or fs.find_dreaming_archive_uri(archive_id, ctx, owner_kind=kind)
    prefix = f"ctx://{ctx.account_id}/sessions/"
    if not uri or not uri.startswith(prefix) or not uri.endswith(f"/history/{archive_id}"):
        return None
    try:
        archive = fs.read_node(uri, ctx)
        messages = json.loads(archive.content)
    except _MISSING:
        return None
    except (ValueError, TypeError):
        return None
    metadata = archive.metadata
    version = metadata.get("version")
    if (metadata.get(f"owner_{kind}_id") != owner or metadata.get("status") != NodeStatus.ACTIVE.value
            or metadata.get("archive_status") == "MERGED" or not isinstance(version, int)
            or not isinstance(messages, list)):
        return None
    return uri, version, archive_user_messages(messages)


def _witnesses(fs, node, key, fact, ctx, *, known=None, archives=None):
    if archives is None:
        archives = {}
    provenance = {ref for ref in node.metadata.get("provenance_ids") or () if isinstance(ref, str)}
    references = {}
    source_uri = known["source_uri"] if known else None
    created_at = node_created_at(node.metadata)
    for proof in fact["evidence"]:
        if not (proof.get("weight") and proof.get("quote") and proof.get("occurrence_id")):
            continue
        if known and (proof["source_ref"] != known["source_ref"]
                      or proof["occurrence_id"] != known["occurrence"]):
            continue
        reference = proof.get("source_ref")
        if not isinstance(reference, str) or reference not in provenance:
            continue
        if reference not in references:
            references[reference] = archive_evidence_ref(reference)
        if not (parsed := references[reference]):
            continue
        archive_id, message_ids = parsed
        cache_key = node.owner_space, archive_id, source_uri
        if cache_key not in archives:
            archives[cache_key] = _read_archive_source(fs, archive_id, node.owner_space, ctx, source_uri)
        source = archives[cache_key]
        if source is not None and archive_proof_matches(source[2], proof, message_ids):
            yield {
                "uri": node.uri, "version": node.metadata["version"], "key": key,
                "node_created_at": created_at,
                "source_ref": proof["source_ref"], "source_uri": source[0],
                "source_version": source[1], "occurrence": proof["occurrence_id"],
                "weight": proof["weight"],
            }


def _peers(fs, node, ctx, now):
    for peer in iter_dreaming_peers(fs, node, ctx):
        if peer.owner_space == node.owner_space and peer.category == node.category and _active(peer, now):
            yield peer


def _independent_pair(left, right):
    left = iter(left)
    if (first := next(left, None)) is None:
        return None
    # Two distinct alternatives per shared source or occurrence cover later matches.
    by_occurrence, by_source = {}, {}
    for second in right:
        if _independent(first, second):
            return first, second
        if second["source_uri"] == first["source_uri"]:
            group, key = by_occurrence, second["occurrence"]
        else:
            group, key = by_source, second["source_uri"]
        if len(group) < 2:
            group.setdefault(key, second)
    representatives = (*by_occurrence.values(), *by_source.values())
    for first in left:
        for second in representatives:
            if _independent(first, second):
                return first, second
    return None


def _independent(first, second):
    return first["source_uri"] != second["source_uri"] and first["occurrence"] != second["occurrence"]


def _supports(fs, promotion, owner, ctx):
    parents = promotion.get("parents") if isinstance(promotion, dict) else None
    if not isinstance(parents, list) or len(parents) != 2:
        return False
    now = datetime.now(UTC)
    archives = {}
    for witness in parents:
        try:
            parent = fs.read_node(witness["uri"], ctx)
        except _MISSING:
            return False
        if (parent.owner_space != owner or not matches_node_version(
                parent.metadata, witness["version"], witness.get("node_created_at"),
        )):
            return False
        fact = _facts(parent, now).get(witness["key"])
        if fact is None or witness not in _witnesses(fs, parent, witness["key"], fact, ctx, known=witness, archives=archives):
            return False
    return True


def promotion_is_supported(fs, node, ctx):
    promotion = node.metadata.get(PROMOTION)
    return promotion is None or _supports(fs, promotion, node.owner_space, ctx)


def _publish(fs, outbox, node, fact, witnesses, target, ctx):
    category, directory = target
    uri, routing = _promotion_destination(node, _identity(node, fact), directory)
    promotion = {"parents": sorted(witnesses, key=lambda witness: witness["uri"])}
    with fs.memory_identity_lock(node.parent_uri.rstrip("/"), ctx):
        if not _supports(fs, promotion, node.owner_space, ctx):
            return False
        expected = 0
        try:
            existing = fs.read_node(uri, ctx)
        except NodeNotFoundError:
            existing = None
            if fs.archived_node_version(uri, ctx) is not None:
                archived = fs.read_node_metadata_including_inactive(uri, ctx)
                archived_event = archived.get(DEMOTION)
                if not archived_event:
                    return True
                status = outbox.get_event_status(uri, archived_event) or "PENDING"
                if status != "DONE":
                    raise PromotionArchiveBlockedError({uri: {"event_id": archived_event, "status": status}})
                expected = archived["version"]
        if existing is not None and promotion_is_supported(fs, existing, ctx):
            if PROMOTION not in existing.metadata:
                return True
            published = existing
        else:
            if existing is not None:
                expected = existing.metadata["version"]
            score = evidence_score(Evidence(w["occurrence"], w["weight"]) for w in witnesses)
            abstract = fact.get("abstract") or fact["content"]
            published = ContextNode(
                uri, "MEMORY", category, 0, node.owner_space, abstract[:MAX_ABSTRACT_LEN],
                f"Observed in two independent sources. {abstract}"[:MAX_ABSTRACT_LEN],
                fact["content"], {
                    "routing_key": routing, "scope_ref": node.metadata.get("scope_ref") or "",
                    "who": node.metadata.get("who"), "where": node.metadata.get("where"),
                    "confidence": score, "provenance_ids": [w["source_ref"] for w in witnesses],
                    PROMOTION: promotion, "_expected_version": expected,
                },
            )
            fs.write_node(published, ctx)
            published = fs.read_node(uri, ctx)
    projection = outbox.build_write_event(published)
    created_at, version = node_created_at(published.metadata), published.metadata["version"]
    projection.event_id = sha256(f"dream-promote:{uri}:{created_at}:{version}".encode()).hexdigest()[:32]
    outbox.register_once(projection, ctx)
    return True


def resume_promotion(fs, outbox, event, ctx):
    version = event.payload[DEMOTION]
    created_at = event.payload.get("node_created_at") or ""
    archive_id = sha256(f"dream-demote:{event.uri}:{created_at}:{version}".encode()).hexdigest()[:32]
    outbox.ensure_processing_owned(event, event.uri)
    try:
        node = fs.read_node(event.uri, ctx)
    except NodeNotFoundError:
        node = None
    if node is not None:
        if not matches_node_version(node.metadata, version, created_at) or promotion_is_supported(fs, node, ctx):
            return
        fs.demote_dreaming_node(node, ctx, archive_id)
    try:
        metadata = fs.read_node_metadata_including_inactive(event.uri, ctx)
    except NodeNotFoundError:
        return
    if (metadata.get("status") != NodeStatus.ARCHIVED.value or metadata.get(DEMOTION) != archive_id
            or (created_at and node_created_at(metadata) != created_at)):
        return
    outbox.register_once(OutboxEvent(
        archive_id, EventType.ARCHIVE_CONTEXT.value, event.uri,
        {"dreaming_archive_revision": {
            "created_at": node_created_at(metadata), "version": metadata["version"],
        }},
        "PENDING",
    ), ctx)


def reconcile_promotion(fs, outbox, node, ctx, *, event=None):
    if PROMOTION in node.metadata:
        if event is not None and not promotion_is_supported(fs, node, ctx):
            outbox.ensure_processing_owned(event, event.uri)
            event.payload[DEMOTION] = node.metadata["version"]
            outbox.save_payload(event)
            resume_promotion(fs, outbox, event, ctx)
        return
    target = _TARGETS.get((node.owner_space.split(":", 1)[0], node.category))
    if target is None:
        return
    now = datetime.now(UTC)
    facts = {key: fact for key, fact in _facts(node, now).items() if fact.get("evidence")}
    if not facts:
        return
    archives = {}
    remaining = set(facts)
    unconfirmed = set()
    blocked = {}

    def publish(key, fact, independent):
        try:
            if not _publish(fs, outbox, node, fact, independent, target, ctx):
                unconfirmed.add(key)
                return
        except PromotionArchiveBlockedError as error:
            blocked.update(error.blocked)
        remaining.remove(key)

    for key, fact in facts.items():
        if independent := _independent_pair(*tee(_witnesses(fs, node, key, fact, ctx, archives=archives))):
            publish(key, fact, independent)
    if remaining:
        for peer in _peers(fs, node, ctx, now):
            for other_key, other in _facts(peer, now).items():
                for key in sorted(remaining):
                    fact = facts[key]
                    if _identity(node, fact) != _identity(peer, other):
                        continue
                    independent = _independent_pair(
                        _witnesses(fs, node, key, fact, ctx, archives=archives),
                        _witnesses(fs, peer, other_key, other, ctx, archives=archives),
                    )
                    if independent:
                        publish(key, fact, independent)
            if not remaining:
                break
    if blocked and event is not None and event.payload.get(_PROMOTION_BLOCKED) != blocked:
        event.payload[_PROMOTION_BLOCKED] = blocked
        outbox.save_payload(event)
    if unconfirmed & remaining:
        raise RuntimeError("Dreaming promotion evidence changed during publication")
    if blocked:
        raise PromotionArchiveBlockedError(blocked)
    if event is not None:
        event.payload.pop(_PROMOTION_BLOCKED, None)
