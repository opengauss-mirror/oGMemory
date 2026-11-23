"""Source-bound assertions consumed by Dreaming."""

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256

from core.memory_names import canonical_object_name, canonical_scope
from core.provenance_resolver import ProvenanceResolver

EVIDENCE_KEY = "dreaming_evidence_v1"


def timestamp(value):
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
        return (parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)).astimezone(UTC)
    except (OverflowError, ValueError):
        return None


def claim_text(value):
    return " ".join(value.split())


@dataclass(frozen=True)
class Assertion:
    claim: str
    subject: str
    relation: str
    quote: str
    slot: str = ""
    value: str = ""
    valid_from: str = ""
    valid_to: str = ""
    effect: str = ""
    retain: bool = False
    scope_ref: str = ""

    @classmethod
    def parse(cls, row):
        if not isinstance(row, dict):
            return None
        fields = {key: row.get(key, "") for key in (
            "claim", "subject", "relation", "quote", "slot", "value",
            "valid_from", "valid_to", "effect", "scope_ref",
        )}
        if any(not isinstance(value, str) for value in fields.values()):
            return None
        if not all(fields[key].strip() for key in ("claim", "subject", "quote")):
            return None
        assertion = cls(**fields, retain=row.get("retain") is True)
        start, end = timestamp(assertion.valid_from), timestamp(assertion.valid_to)
        if (assertion.relation not in ("supports", "refutes", "unknown")
                or (assertion.valid_from and start is None)
                or (assertion.valid_to and end is None)
                or (start and end and start >= end)
                or bool(assertion.slot) != bool(assertion.value)):
            return None
        if assertion.effect == "correction":
            if not (assertion.relation == "supports" and start and assertion.slot.strip() and assertion.value.strip()):
                return None
        elif assertion.effect == "revocation":
            if not (assertion.relation == "refutes" and start and end is None):
                return None
        elif assertion.effect:
            return None
        return assertion


def _archive_ref(value):
    if not isinstance(value, str):
        return None
    try:
        ref = ProvenanceResolver.parse_id(value)
    except ValueError:
        return None
    if (ref["version"] == 1 and ref["source_type"] == "archive" and ref["source_id"] and ref["detail"]
            and ProvenanceResolver.build_id("archive", ref["source_id"], ref["detail"]) == value):
        return ref["source_id"], set(ref["detail"])
    return None


def _user_messages(messages, ids):
    users, seen = {}, set()
    for message in messages:
        if not isinstance(message, dict) or not isinstance(msg_id := message.get("id"), str) or msg_id not in ids:
            continue
        if msg_id in seen:
            users.pop(msg_id, None)
        elif message.get("role") == "user" and isinstance(content := message.get("content"), str):
            users[msg_id] = content
        seen.add(msg_id)
    return users


def _revision(content):
    return sha256(content.encode("utf-8")).hexdigest()


def bind_archive_evidence(rows, source_ref, messages):
    """Resolve a verbatim quote in exactly one authorized user message."""
    if not (ref := _archive_ref(source_ref)):
        return []
    users = _user_messages(messages, ref[1])
    bound = []
    for row in rows:
        if (assertion := Assertion.parse(row)) is None:
            continue
        matches = [msg_id for msg_id, content in users.items() if assertion.quote in content]
        if len(matches) == 1:
            msg_id = matches[0]
            bound.append({
                **asdict(assertion), "source_ref": source_ref, "message_id": msg_id,
                "source_revision": _revision(users[msg_id]),
            })
    return bound


def verified_assertions(node, read_source):
    if read_source is None:
        return
    subject = canonical_object_name(node.metadata.get("object_name") or node.metadata.get("who")
                                    or node.owner_space.split(":", 1)[-1])
    scope = canonical_scope(node.metadata.get("scope_ref") or "")
    refs = node.metadata.get("provenance_ids") or ()
    rows = node.metadata.get(EVIDENCE_KEY)
    if not subject or not isinstance(rows, list) or not isinstance(refs, (list, tuple)):
        return
    archives, messages, revisions = {}, {}, {}
    for row in rows:
        assertion = Assertion.parse(row)
        if (assertion is None or assertion.relation == "unknown"
                or canonical_object_name(assertion.subject) != subject
                or canonical_scope(assertion.scope_ref) != scope):
            continue
        source_ref, msg_id = row.get("source_ref"), row.get("message_id")
        if (not isinstance(source_ref, str) or source_ref not in refs or not (ref := _archive_ref(source_ref))
                or not isinstance(msg_id, str) or msg_id not in ref[1]):
            continue
        archive_id, ids = ref
        if archive_id not in archives:
            archives[archive_id] = read_source(archive_id)
        archive = archives[archive_id]
        if getattr(archive, "archive_id", None) != archive_id:
            continue
        if source_ref not in messages:
            messages[source_ref] = _user_messages(archive.messages, ids)
        content = messages[source_ref].get(msg_id)
        if content is None or assertion.quote not in content:
            continue
        message_key = archive_id, msg_id
        if message_key not in revisions:
            revisions[message_key] = _revision(content)
        if row.get("source_revision") != revisions[message_key]:
            continue
        occurrence = (archive.session_id or archive_id, msg_id)
        yield assertion, {
            "source_ref": source_ref,
            "message_id": msg_id,
            "source_revision": row["source_revision"],
            "occurrence_id": sha256(repr(occurrence).encode()).hexdigest(),
            "quote": assertion.quote,
            "weight": 4,
            "supports": assertion.relation == "supports",
        }
