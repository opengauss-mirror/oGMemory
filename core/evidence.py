"""Immutable message sources and evidence units, independent of NLP and storage."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from core.errors import ValidationError
from core.models import RequestContext

EVIDENCE_KINDS = frozenset({"sources", "windows", "sessions", "lifecycle"})
_DIGEST = re.compile(r"[0-9a-f]{64}")
_EVIDENCE_URI = re.compile(
    r"ctx://(?P<account>[^/]+)/users/(?P<owner_id>[^/]+)/evidence/"
    r"(?P<kind>sources|windows|sessions|lifecycle)(?:/(?P<slug>[0-9a-f]{64}))?/?"
)
_TURN = re.compile(r"^\[([^\[\]\r\n:]{1,80})\]:[ \t]*", re.MULTILINE)


def digest(*values: object) -> str:
    encoded = json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _identity(value: str) -> str:
    if (not isinstance(value, str) or not value or value in {".", ".."}
            or any(c in value for c in "/\\%?#") or any(c.isspace() or ord(c) < 32 for c in value)):
        raise ValueError("evidence requires an exact, safe owner identity")
    return value


def parse_evidence_uri(uri: str) -> dict | None:
    """Parse only the strict user evidence family; never accept account resources."""
    match = _EVIDENCE_URI.fullmatch(uri)
    if match is None:
        return None
    fields = match.groupdict()
    _identity(fields["account"])
    _identity(fields["owner_id"])
    return {"account": fields["account"], "owner_type": "users",
            "owner_id": fields["owner_id"], "category": "evidence/" + fields["kind"],
            "slug": fields["slug"] or ""}


def evidence_uri(ctx: RequestContext, kind: str, key: str = "") -> str:
    if kind not in EVIDENCE_KINDS or key and not _DIGEST.fullmatch(key):
        raise ValueError("invalid evidence kind or key")
    account, user = _identity(ctx.account_id), _identity(ctx.user_id)
    return f"ctx://{account}/users/{user}/evidence/{kind}" + (f"/{key}" if key else "")


@dataclass(frozen=True)
class EvidenceUnit:
    evidence_unit_id: str
    parent_span: tuple[int, int]
    text: str
    role: str
    speaker: str


@dataclass(frozen=True)
class MessageSource:
    source_key: str
    account_id: str
    owner_space: str
    session_id: str
    message_id: str
    sequence: int
    text: str
    content_hash: str
    role: str
    speaker: str
    recorded_at: str | None
    timezone: str | None
    time_basis: str
    units: tuple[EvidenceUnit, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def message_source(message: dict, ctx: RequestContext) -> MessageSource:
    """Freeze identity and original character offsets; no inferred message IDs/time."""
    evidence_uri(ctx, "sources")
    message_id = message.get("id") or message.get("message_id")
    if not isinstance(message_id, str) or not message_id or not ctx.session_id:
        raise ValidationError("source", "stable message ID and session context are required")
    text, role = message.get("content"), message.get("role")
    if not isinstance(text, str) or role not in {"user", "assistant"}:
        raise ValidationError("source", "expected original user/assistant text")
    if not text.strip() or len(text.encode("utf-8")) > 2 * 1024 * 1024:
        raise ValidationError("source", "message must contain 1 to 2097152 UTF-8 bytes")
    sequence = message.get("sequence")
    if type(sequence) is not int or sequence < 0:
        raise ValidationError("sequence", "original non-negative message sequence is required")
    recorded_at = message.get("recorded_at") or message.get("timestamp")
    if isinstance(recorded_at, datetime):
        recorded_at = recorded_at.isoformat()
    if recorded_at is not None:
        try:
            datetime.fromisoformat(recorded_at)
        except (ValueError, TypeError) as exc:
            raise ValidationError("recorded_at", "expected an ISO timestamp") from exc
    timezone = message.get("timezone")
    if timezone is not None and not isinstance(timezone, str):
        raise ValidationError("timezone", "expected a timezone name")
    if timezone is not None:
        try:
            ZoneInfo(timezone)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValidationError("timezone", "expected a resolvable IANA timezone name") from exc
    speaker = message.get("speaker", "")
    if not isinstance(speaker, str):
        raise ValidationError("speaker", "expected a speaker name")
    key = digest(ctx.account_id, ctx.user_space_name(), ctx.session_id, message_id)
    matches = list(_TURN.finditer(text))
    # Only a whole message consisting of the documented wrapper is parsed.
    if matches and not text[:matches[0].start()].strip():
        spans = [(m.end(), matches[i + 1].start() if i + 1 < len(matches) else len(text),
                  m.group(1).strip()) for i, m in enumerate(matches)]
    else:
        spans = [(0, len(text), speaker)]
    units = tuple(EvidenceUnit(digest(key, start, end), (start, end), text[start:end], role, name)
                  for start, end, name in spans if text[start:end].strip())
    return MessageSource(key, ctx.account_id, ctx.user_space_name(), ctx.session_id,
                         message_id, sequence, text, content_hash(text), role, speaker,
                         recorded_at, timezone, "message" if recorded_at else "unknown", units)


def reassemble_messages(messages: list[dict]) -> list[dict]:
    """Reassemble only explicitly positioned fragments, rejecting gaps and conflicts."""
    complete, fragments = {}, {}
    order = []
    for message in messages:
        key = message.get("id") or message.get("message_id")
        if not isinstance(key, str) or not key:
            raise ValidationError("source", "stable message ID is required")
        if key not in order:
            order.append(key)
        if "parent_offset" in message:
            fragments.setdefault(key, []).append(message)
        elif key in complete and complete[key] != message:
            raise ValidationError("source", "conflicting message identity")
        else:
            complete[key] = dict(message)
    for key, pieces in fragments.items():
        first = pieces[0]
        length = first.get("parent_length")
        if type(length) is not int or not 0 < length <= 2 * 1024 * 1024:
            raise ValidationError("parent_length", "bounded original length is required")
        chars: list[str | None] = [None] * length
        metadata = {k: v for k, v in first.items() if k not in {"content", "parent_offset"}}
        for piece in pieces:
            start, text = piece["parent_offset"], piece.get("content")
            if (type(start) is not int or not isinstance(text, str) or start < 0
                    or start + len(text) > length
                    or metadata != {k: v for k, v in piece.items() if k not in {"content", "parent_offset"}}):
                raise ValidationError("source", "inconsistent fragment offsets or metadata")
            for i, char in enumerate(text, start):
                if chars[i] is not None and chars[i] != char:
                    raise ValidationError("source", "conflicting fragment overlap")
                chars[i] = char
        if any(char is None for char in chars):
            raise ValidationError("source", "incomplete message fragments")
        assembled = {k: v for k, v in metadata.items() if k != "parent_length"}
        assembled["content"] = "".join(char for char in chars if char is not None)
        if key in complete and complete[key] != assembled:
            raise ValidationError("source", "fragments differ from original message")
        complete[key] = assembled
    return [complete[key] for key in order]


def fact_source_refs(records):
    return [ref for record in records for ref in record["source_refs"]]


def deterministic_payload(records, *, existing=None):
    fields = {"schema_version": 1, "extraction_method": "deterministic"} if existing is None else existing
    return {**fields, "fact_records": records, "source_refs": fact_source_refs(records)}
