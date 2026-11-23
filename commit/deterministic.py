"""Source-verified deterministic writes through the existing writer and receipts."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import replace

from core.errors import NodeNotFoundError, ValidationError
from core.evidence import DETERMINISTIC_MEMORY_TYPES, deterministic_payload, digest, fact_source_refs
from core.models import WritePlan
from core.operation_deadline import check_deadline
from core.validation import validate_candidate
from extraction.deterministic.extractor import render_records

_TRUSTED_BATCH: ContextVar[frozenset[str] | None] = ContextVar("deterministic_batch", default=None)


@contextmanager
def deterministic_batch(verified_candidates=()):
    """Only the server's deterministic extraction/recovery path may enter this scope."""
    token = _TRUSTED_BATCH.set(frozenset(
        digest(candidate.metadata.get("deterministic")) for candidate in verified_candidates))
    try:
        yield
    finally:
        _TRUSTED_BATCH.reset(token)


def validate_payload(candidate, store, ctx, *, verify_structure=True, resolve=None, verify=None):
    check_deadline()
    data = candidate.metadata.get("deterministic")
    if (not isinstance(data, dict) or data.get("schema_version") != 1
            or data.get("extraction_method") != "deterministic"
            or candidate.category not in DETERMINISTIC_MEMORY_TYPES
            or candidate.owner_scope != "user" or not candidate.routing_key.startswith("det_")):
        raise ValidationError("deterministic", "invalid deterministic candidate")
    records, refs = data.get("fact_records"), data.get("source_refs")
    if not isinstance(records, list) or not records or not isinstance(refs, list) or not refs:
        raise ValidationError("deterministic", "records and source references are required")
    if refs != fact_source_refs(records):
        raise ValidationError("deterministic", "fact references differ from candidate sources")
    from extraction.deterministic.verification import record_verifier
    trusted = _TRUSTED_BATCH.get()
    check_structure = verify_structure and not (trusted is not None and digest(data) in trusted)
    if resolve is None:
        resolve = store.reference_reader(ctx)
    if check_structure and verify is None:
        verify = record_verifier(store, ctx, resolve)
    for record in records:
        for ref in record["source_refs"]:
            if (ref.get("session_id") != ctx.session_id or ref.get("quote") != record.get("quote")
                    or resolve(ref) is None):
                raise ValidationError("deterministic", "unverified or revoked fact source")
        if check_structure and not verify(record):
            raise ValidationError("deterministic", "unverified structured assertion")
    check_deadline()
    return deepcopy(data)


def require_deterministic_batch():
    if _TRUSTED_BATCH.get() is None:
        raise ValidationError("deterministic", "reserved for source-verified extraction batches")


def plan_deterministic_write(candidate, uri, data, current):
    if current is not None and not current.metadata.get("deterministic"):
        raise ValidationError("deterministic", "cannot overwrite another extraction engine")
    records = {record["fact_id"]: record for record in (
        current.metadata["deterministic"]["fact_records"] if current else [])}
    for record in data["fact_records"]:
        if record["fact_id"] in records and records[record["fact_id"]] != record:
            raise ValidationError("deterministic", "immutable fact identity changed")
        records[record["fact_id"]] = record
    data = deterministic_payload([records[key] for key in sorted(records)], existing=data)
    abstract, overview, content = render_records(candidate.object_name or candidate.title, data["fact_records"])
    validate_candidate(replace(candidate, abstract=abstract, overview=overview, content=content))
    same = current is not None and data == current.metadata["deterministic"]
    return WritePlan("skip" if same else "merge" if current else "create", uri,
                     {"identity_metadata": {"deterministic": data},
                      "expected_version": current.metadata.get("version", 0) if current else 0,
                      "identity_create": current is None, "resolved_abstract": abstract,
                      "resolved_overview": overview, "resolved_content": content})


def check_engine_target(fs, candidate, uri, ctx):
    """Final protection also covers custom policies and explicit LLM update targets."""
    try:
        current = fs.read_node(uri, ctx)
    except NodeNotFoundError:
        return
    if bool(current.metadata.get("deterministic")) != bool(candidate.metadata.get("deterministic")):
        raise ValidationError("target_uri", "automatic writes cannot cross extraction engines")
