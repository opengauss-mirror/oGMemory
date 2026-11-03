"""Orchestration for selecting, storing, and safely rewriting tool results."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
import threading
import time
from collections import OrderedDict
from copy import deepcopy
from datetime import UTC, datetime
from typing import Iterable
from uuid import uuid4

from core.models import RequestContext

from .fragments import (
    best_query_match,
    build_query_plan,
    precise_fragment_from_window,
    query_occurrences,
    rank_structural_fragments,
)
from .models import (
    ArtifactMetadata,
    ExternalizationResult,
    GovernedMessages,
    NormalizedMessage,
    SourceCollectionReuse,
    SourceFragmentPassthrough,
    ToolResultPart,
)
from .multires import (
    ToolResultMultiresolutionBuilder,
    ranking_context_from_messages,
)
from .normalizer import ToolResultNormalizer, canonical_tool_call_id
from .policy import SelectedPart, ToolResultPolicy
from .records import (
    extract_record_candidates,
    is_hard_critical_record,
    record_covers_fact,
    retained_record_values,
)
from .store_base import ArtifactNotVisibleError, ToolResultArtifactStore
from .synopsis import ToolResultSynopsisBuilder

logger = logging.getLogger("ogmem.tool_results")

_STUB_START_V2 = "[ogmemory-tool-result v=2]"
_STUB_START = _STUB_START_V2
_STUB_END = "[/ogmemory-tool-result]"
_STUB_START_PATTERN = re.compile(re.escape(_STUB_START_V2))
_REF_PATTERN = re.compile(r"^ogmemory://tool-results/(tr_[a-f0-9_]+)$")
_MIN_SUMMARY_JSON_CHARS = 8
_MAX_FRAGMENT_WINDOW_BYTES = 64 * 1024
_MAX_FRAGMENT_WINDOW_CACHE = 64


class InvalidSourceArtifactError(ValueError):
    """A source-aware envelope was recognized but could not be verified."""


class ExternalizationTimeoutError(TimeoutError):
    """The bounded Artifact write/read verification deadline expired."""


class BoundedDaemonIORunner:
    """Run blocking store operations with bounded outstanding daemon threads.

    Python cannot safely cancel an arbitrary synchronous filesystem/database
    call.  On timeout the caller returns immediately, while the daemon retains
    its semaphore slot until the underlying operation actually exits.  This
    bounds leaked work during a backend outage instead of spawning unbounded
    timeout threads.

    An active SQL mutation guard instead executes on its owning thread. Its
    capacity/deadline accounting is retained, but an in-progress synchronous
    call cannot be cancelled or bounded by the former daemon caller timeout.
    """

    def __init__(self, max_inflight: int = 8) -> None:
        if max_inflight < 1:
            raise ValueError("max_inflight must be positive")
        self.max_inflight = int(max_inflight)
        self._slots = threading.BoundedSemaphore(self.max_inflight)
        self._inflight_lock = threading.Lock()
        self._inflight_by_key: dict[str, int] = {}

    def run(self, operation, *, deadline: float, operation_key: str | None = None):
        from fs.sql_adapter.session_mutation import active_session_guard
        guard = active_session_guard()
        if guard is not None:
            guard.ensure_owned()
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not self._slots.acquire(timeout=remaining):
            raise ExternalizationTimeoutError("artifact I/O capacity deadline expired")

        if operation_key:
            with self._inflight_lock:
                self._inflight_by_key[operation_key] = (
                    self._inflight_by_key.get(operation_key, 0) + 1
                )

        done = threading.Event()
        outcome: dict[str, object] = {}

        def worker() -> None:
            try:
                if guard is not None:
                    guard.ensure_owned()
                    if time.monotonic() >= deadline:
                        raise ExternalizationTimeoutError("artifact I/O deadline expired before execution")
                outcome["value"] = operation()
            except BaseException as exc:  # surfaced in the requesting thread
                outcome["error"] = exc
            finally:
                if operation_key:
                    with self._inflight_lock:
                        remaining_count = self._inflight_by_key.get(operation_key, 1) - 1
                        if remaining_count > 0:
                            self._inflight_by_key[operation_key] = remaining_count
                        else:
                            self._inflight_by_key.pop(operation_key, None)
                self._slots.release()
                done.set()

        if guard is not None:
            # SQL ownership cannot cross threads or reconnect on a worker.
            # Inline I/O keeps the connection fence; the deadline cannot
            # cancel a blocking owning query or arbitrary synchronous call.
            worker()
            guard.ensure_owned()
        else:
            try:
                thread = threading.Thread(
                    target=worker,
                    daemon=True,
                    name="ogmem-tool-result-io",
                )
                thread.start()
            except BaseException:
                if operation_key:
                    with self._inflight_lock:
                        remaining_count = self._inflight_by_key.get(operation_key, 1) - 1
                        if remaining_count > 0:
                            self._inflight_by_key[operation_key] = remaining_count
                        else:
                            self._inflight_by_key.pop(operation_key, None)
                self._slots.release()
                raise

        remaining = deadline - time.monotonic()
        if remaining <= 0 or not done.wait(remaining):
            raise ExternalizationTimeoutError("artifact write/read verification timed out")
        error = outcome.get("error")
        if isinstance(error, BaseException):
            raise error
        return outcome.get("value")

    def has_inflight(self, operation_key: str) -> bool:
        with self._inflight_lock:
            return self._inflight_by_key.get(operation_key, 0) > 0


def _contains_stub_marker(value: str) -> bool:
    return (
        isinstance(value, str)
        and ("[ogmemory-tool-result" in value or "[/ogmemory-tool-result" in value)
    )


def _attestation_payload(envelope: dict, ctx: RequestContext, session_id: str) -> bytes:
    fields = [
        envelope.get("schema_version"),
        envelope.get("source_tool"),
        envelope.get("artifact_ref"),
        envelope.get("offset_bytes"),
        envelope.get("actual_length_bytes"),
        envelope.get("artifact_sha256"),
        envelope.get("content_sha256"),
        ctx.account_id,
        ctx.user_id,
        ctx.agent_id,
        session_id,
    ]
    try:
        schema_version = int(envelope.get("schema_version", 1) or 1)
    except (TypeError, ValueError):
        schema_version = 1
    if schema_version >= 2:
        fields.extend([
            envelope.get("fragment_type"),
            envelope.get("fragment_complete"),
            envelope.get("match_offset_bytes"),
            envelope.get("match_length_bytes"),
            envelope.get("context_before"),
            envelope.get("context_before_offset_bytes"),
            envelope.get("context_before_sha256"),
            envelope.get("context_after"),
            envelope.get("context_after_offset_bytes"),
            envelope.get("context_after_sha256"),
        ])
    if envelope.get("source_tool") == "list_tool_result_artifacts":
        listing = {
            "_ogmemory_artifact_list_source": envelope.get(
                "_ogmemory_artifact_list_source"
            ),
            "schema_version": envelope.get("schema_version"),
            "source_tool": envelope.get("source_tool"),
            "artifacts": envelope.get("artifacts"),
            "truncated": envelope.get("truncated"),
            "next_cursor": envelope.get("next_cursor"),
        }
        fields.append(hashlib.sha256(json.dumps(
            listing,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest())
    return json.dumps(
        fields,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def attest_source_envelope(
    envelope: dict,
    key: bytes,
    ctx: RequestContext,
    session_id: str,
) -> dict:
    result = dict(envelope)
    result["source_attestation"] = hmac.new(
        key,
        _attestation_payload(result, ctx, session_id),
        hashlib.sha256,
    ).hexdigest()
    return result


def parse_stub(value: str) -> dict | None:
    """Parse only a complete versioned stub, never a loose ref-like substring."""
    if not isinstance(value, str):
        return None
    first_line, separator, _remainder = value.partition("\n")
    marker = _STUB_START_PATTERN.fullmatch(first_line)
    if not separator or marker is None:
        return None
    if not value.endswith("\n" + _STUB_END):
        return None
    fields: dict[str, str] = {}
    for line in value.splitlines()[1:-1]:
        if ": " not in line:
            return None
        key, field_value = line.split(": ", 1)
        if key in fields:
            return None
        fields[key] = field_value
    if not _REF_PATTERN.fullmatch(fields.get("ref", "")):
        return None
    fields["version"] = first_line.removeprefix("[ogmemory-tool-result v=").removesuffix("]")
    return fields


def _parse_stub_blocks(value: str) -> tuple[tuple[str, dict], ...] | None:
    if not isinstance(value, str):
        return None
    parsed: list[tuple[str, dict]] = []
    position = 0
    end_token = "\n" + _STUB_END
    while position < len(value):
        while position < len(value) and value[position].isspace():
            position += 1
        if position == len(value):
            break
        start_match = _STUB_START_PATTERN.match(value, position)
        if start_match is None or not value.startswith("\n", start_match.end()):
            return None
        end_position = value.find(end_token, start_match.end())
        if end_position < 0:
            return None
        block_end = end_position + len(end_token)
        block = value[position:block_end]
        fields = parse_stub(block)
        if fields is None or not re.fullmatch(r"[a-f0-9]{64}", fields.get("sha256", "")):
            return None
        parsed.append((block, fields))
        position = block_end
    return tuple(parsed) if parsed else None


def _parse_embedded_stub_blocks(value: str) -> tuple[tuple[str, dict], ...] | None:
    """Parse every canonical stub embedded in an otherwise flat projection.

    Buffer persistence flattens structured provider messages.  A message with
    one externalized result and one inline result therefore legitimately
    contains both ordinary text and a canonical stub.  Treat every marker as
    structural: stray, nested, or incomplete markers invalidate the complete
    projection instead of being accepted as ordinary text.
    """
    if not isinstance(value, str):
        return None
    parsed: list[tuple[str, dict]] = []
    position = 0
    while position < len(value):
        start_match = _STUB_START_PATTERN.search(value, position)
        start = start_match.start() if start_match is not None else -1
        stray_end = value.find(_STUB_END, position)
        if start < 0:
            return None if stray_end >= 0 else (tuple(parsed) if parsed else None)
        if 0 <= stray_end < start:
            return None
        end = value.find(_STUB_END, start_match.end())
        if end < 0:
            return None
        nested_start = _STUB_START_PATTERN.search(value, start_match.end(), end)
        if nested_start is not None:
            return None
        block_end = end + len(_STUB_END)
        block = value[start:block_end]
        fields = parse_stub(block)
        if fields is None or not re.fullmatch(r"[a-f0-9]{64}", fields.get("sha256", "")):
            return None
        parsed.append((block, fields))
        position = block_end
    return tuple(parsed) if parsed else None


def parse_stub_sequence(value: str) -> tuple[dict, ...] | None:
    """Parse content made exclusively of complete stubs separated by whitespace."""
    parsed = _parse_stub_blocks(value)
    if parsed is None:
        return None
    return tuple(fields for _, fields in parsed)


def render_safe_message_content(message: dict) -> str:
    """Flatten a governed message without dropping stubs in result blocks."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, (dict, tuple)):
        return json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if not isinstance(content, list):
        return str(content) if content is not None else ""
    parts: list[str] = []
    message_is_result = str(message.get("role", "")).casefold() in {
        "tool",
        "tool_result",
        "toolresult",
    }
    for block in content:
        if isinstance(block, str):
            if block:
                parts.append(block)
            continue
        if not isinstance(block, dict):
            continue
        block_is_result = str(block.get("type", "")).casefold() in {
            "tool_result",
            "toolresult",
            "tool_call",
            "toolcall",
            "tool_use",
            "tooluse",
        }
        result_like = message_is_result or block_is_result
        keys = (
            ("tool_output", "toolOutput", "output", "result", "content", "text")
            if result_like
            else ("text", "tool_output", "toolOutput", "output", "result", "content")
        )
        for key in keys:
            value = block.get(key)
            if isinstance(value, str) and value:
                parts.append(value)
                if not result_like:
                    break
            if isinstance(value, (dict, list)):
                parts.append(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
                if not result_like:
                    break
    return "\n".join(parts).strip()


class ToolResultExternalizer:
    def __init__(
        self,
        normalizer: ToolResultNormalizer,
        policy: ToolResultPolicy,
        synopsis_builder: ToolResultSynopsisBuilder,
        store: ToolResultArtifactStore,
        *,
        retry_attempts: int = 2,
        write_timeout_ms: int = 3_000,
        summary_per_artifact_chars: int = 2_000,
        summary_per_turn_chars: int = 12_000,
        multires_builder: ToolResultMultiresolutionBuilder | None = None,
        source_attestation_key: bytes | None = None,
        io_runner: BoundedDaemonIORunner | None = None,
    ):
        self.normalizer = normalizer
        self.policy = policy
        self.synopsis_builder = synopsis_builder
        self.store = store
        self.retry_attempts = max(1, retry_attempts)
        self.write_timeout_ms = max(1, write_timeout_ms)
        self.summary_per_artifact_chars = max(
            _MIN_SUMMARY_JSON_CHARS, summary_per_artifact_chars,
        )
        self.summary_per_turn_chars = max(
            _MIN_SUMMARY_JSON_CHARS, summary_per_turn_chars,
        )
        self.multires_builder = multires_builder or ToolResultMultiresolutionBuilder()
        self.source_attestation_key = source_attestation_key
        self.io_runner = io_runner or BoundedDaemonIORunner()

    def validates_stub_sequence(
        self,
        value: str,
        ctx: RequestContext,
        session_id: str,
        *,
        deadline: float | None = None,
    ) -> bool:
        """Verify every stub is canonical and references a visible artifact."""
        return self.stub_sequence_status(
            value,
            ctx,
            session_id,
            deadline=deadline,
        ) == "valid"

    def validates_source_attestation(
        self,
        envelope: dict,
        ctx: RequestContext,
        session_id: str,
    ) -> bool:
        """Accept only a complete owner-bound HMAC issued by this service."""
        if self.source_attestation_key is None or not isinstance(envelope, dict):
            return False
        actual = envelope.get("source_attestation")
        if not isinstance(actual, str) or not re.fullmatch(r"[a-f0-9]{64}", actual):
            return False
        expected = hmac.new(
            self.source_attestation_key,
            _attestation_payload(envelope, ctx, session_id),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(actual, expected)

    def validates_artifact_list_result(
        self,
        part: ToolResultPart,
        ctx: RequestContext,
        session_id: str,
        *,
        deadline: float | None = None,
    ) -> bool:
        """Verify one ephemeral list result before terminal lifecycle removal."""
        if (
            not part.tool_call_id
            or str(part.tool_name or "").rsplit("__", 1)[-1].casefold()
            != "list_tool_result_artifacts"
        ):
            return False
        candidate = part.output_text
        first_line, separator, remainder = candidate.partition("\n")
        if separator and first_line.rstrip("\r") == "structuredContent:":
            candidate = remainder
        try:
            envelope = json.loads(candidate)
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        allowed_fields = {
            "_ogmemory_artifact_list_source",
            "schema_version",
            "source_tool",
            "artifacts",
            "truncated",
            "next_cursor",
            "source_attestation",
        }
        if (
            not isinstance(envelope, dict)
            or frozenset(envelope) != frozenset(allowed_fields)
            or envelope.get("_ogmemory_artifact_list_source") is not True
            or envelope.get("schema_version") != 1
            or envelope.get("source_tool") != "list_tool_result_artifacts"
            or not isinstance(envelope.get("artifacts"), list)
            or not isinstance(envelope.get("truncated"), bool)
            or not self.validates_source_attestation(envelope, ctx, session_id)
        ):
            return False
        artifacts = envelope["artifacts"]
        next_cursor = envelope.get("next_cursor")
        if envelope["truncated"]:
            if not artifacts or next_cursor != artifacts[-1].get("artifact_id"):
                return False
        elif next_cursor is not None:
            return False
        refs = [row.get("ref") for row in artifacts if isinstance(row, dict)]
        if len(refs) != len(artifacts) or len(set(refs)) != len(refs):
            return False
        deadline = deadline or (time.monotonic() + self.write_timeout_ms / 1000)
        try:
            for row, ref in zip(artifacts, refs):
                match = _REF_PATTERN.fullmatch(str(ref or ""))
                if match is None:
                    return False
                metadata = self.io_runner.run(
                    lambda artifact_id=match.group(1): self.store.read_metadata(
                        artifact_id, session_id, ctx,
                    ),
                    deadline=deadline,
                    operation_key=session_id,
                )
                if row != metadata.public_dict() | {"ref": metadata.ref}:
                    return False
        except Exception:
            return False
        return True

    def _stub_matches_metadata(self, block: str, metadata: ArtifactMetadata) -> bool:
        fields = parse_stub(block)
        if fields is None:
            return False
        if fields.get("version") != "2" or metadata.schema_version < 2:
            return False
        expected = {
            "ref": metadata.ref,
            "session_id": self._safe_stub_field(metadata.session_id),
            "tool": self._safe_stub_field((metadata.tool_name or "unknown")[:200]),
            "type": metadata.detected_type,
            "chars": str(metadata.original_chars),
            "bytes": str(metadata.original_bytes),
            "sha256": metadata.sha256,
        }
        if set(fields) != {
            "version",
            *expected,
            "synopsis",
            "key_facts",
            "selected_evidence",
            "retained_records",
            "coverage",
            "retrieval_hint",
        }:
            return False
        if any(fields.get(key) != value for key, value in expected.items()):
            return False
        try:
            synopsis = json.loads(fields["synopsis"])
            facts = json.loads(fields["key_facts"])
            evidence = json.loads(fields["selected_evidence"])
            retained_records = json.loads(fields.get("retained_records", "null"))
            coverage = json.loads(fields["coverage"])
            retrieval_hint = json.loads(fields["retrieval_hint"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False
        if not isinstance(synopsis, str) or not (
            synopsis == metadata.synopsis
            or (synopsis.endswith("…") and metadata.synopsis.startswith(synopsis[:-1]))
            or synopsis == ""
        ):
            return False
        if not isinstance(facts, list) or not isinstance(evidence, list):
            return False
        fact_pool = [dict(item) for item in metadata.fact_candidates]
        evidence_pool = [dict(item) for item in metadata.evidence_candidates]
        record_pool = [dict(item) for item in metadata.record_candidates]
        if len(facts) != len({json.dumps(item, sort_keys=True) for item in facts}):
            return False
        if len(evidence) != len({json.dumps(item, sort_keys=True) for item in evidence}):
            return False
        if any(item not in fact_pool for item in facts) or any(
            item not in evidence_pool for item in evidence
        ):
            return False
        retained_values = retained_record_values(retained_records, metadata.record_layout)
        if retained_records is not None and retained_values is None:
            return False
        retained_values = retained_values or []
        candidate_by_content = {
            json.dumps(item.get("value"), ensure_ascii=False, sort_keys=True, separators=(",", ":")):
            item
            for item in record_pool
        }
        retained_rows: list[dict] = []
        retained_hashes: set[str] = set()
        for value in retained_values:
            canonical = json.dumps(
                value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            )
            candidate = candidate_by_content.get(canonical)
            if candidate is None or canonical in retained_hashes:
                return False
            retained_hashes.add(canonical)
            retained_rows.append(candidate)
        retained_indices = [int(item.get("index", -1)) for item in retained_rows]
        if retained_indices != sorted(retained_indices):
            return False
        base_critical = {
            int(item.get("index", -1))
            for item in record_pool
            if is_hard_critical_record(item)
        }
        if not base_critical.issubset(retained_indices):
            return False
        if not isinstance(coverage, dict) or not isinstance(metadata.coverage, dict):
            return False
        selected_offsets = {int(item.get("offset_bytes", -1)) for item in facts}
        if any(
            int(item.get("fact_offset_bytes", -2)) not in selected_offsets
            for item in evidence
        ):
            return False
        eligible_evidence = sum(
            1 for item in evidence_pool
            if int(item.get("fact_offset_bytes", -2)) in selected_offsets
        )
        covered_offsets = {
            int(fact.get("offset_bytes", -1))
            for fact in fact_pool
            if any(
                record_covers_fact(row.get("value"), fact)
                for row in retained_rows
            )
        }
        coverage_tracks_dedup = bool(
            {"covered_key_facts", "covered_evidence"} & set(coverage)
        )
        if coverage_tracks_dedup:
            if selected_offsets & covered_offsets:
                return False
            uncovered_pool = [
                fact
                for fact in fact_pool
                if int(fact.get("offset_bytes", -1)) not in covered_offsets
            ]
            omitted_facts = max(0, len(uncovered_pool) - len(facts))
            covered_evidence = sum(
                int(item.get("fact_offset_bytes", -2)) in covered_offsets
                for item in evidence_pool
            )
            if coverage.get("covered_key_facts") != len(covered_offsets):
                return False
            if covered_evidence and coverage.get("covered_evidence") != covered_evidence:
                return False
            if not covered_evidence and "covered_evidence" in coverage:
                return False
        else:
            # Existing v2 stubs may predate retained-record/fact deduplication.
            omitted_facts = max(0, len(fact_pool) - len(facts))
        omitted_evidence = max(0, eligible_evidence - len(evidence))
        evidence_truncated = any(
            bool(item.get("truncated"))
            or item.get("fact_complete") is False
            for item in evidence
        )
        summary_truncated = bool(
            omitted_facts
            or omitted_evidence
            or evidence_truncated
            or metadata.coverage.get("candidate_pool_truncated")
            or synopsis != metadata.synopsis
            or int((metadata.record_layout or {}).get("record_count", 0))
            - int((metadata.record_layout or {}).get("duplicate_records", 0))
            > len(retained_indices)
        )
        if not coverage:
            return self._retrieval_hint_is_accepted(retrieval_hint)
        total_records = int((metadata.record_layout or {}).get("record_count", 0))
        duplicate_records = int((metadata.record_layout or {}).get("duplicate_records", 0))
        omitted_records = max(0, total_records - duplicate_records - len(retained_indices))
        record_pattern = str((metadata.record_layout or {}).get("pattern", "array"))
        critical_count = coverage.get("critical_record_count")
        record_fields = {
            "record_pattern",
            "record_count",
            "retained_record_count",
            "critical_record_count",
            "deduplicated_records",
            "omitted_records",
            "critical_records_over_budget",
        }
        if metadata.record_layout is None:
            record_fields_valid = not retained_indices and not (record_fields & set(coverage))
        else:
            record_fields_valid = (
                coverage.get("record_pattern") == record_pattern
                and coverage.get("record_count") == total_records
                and coverage.get("retained_record_count") == len(retained_indices)
                and coverage.get("deduplicated_records") == duplicate_records
                and coverage.get("omitted_records") == omitted_records
                and isinstance(critical_count, int)
                and len(base_critical) <= critical_count <= len(retained_indices)
                and isinstance(coverage.get("critical_records_over_budget"), bool)
            )
        if not record_fields_valid:
            return False
        if coverage.get("coverage_compact") is True:
            expected_compact_fields = {
                "scan_complete",
                "candidate_pool_truncated",
                "evidence_truncated",
                "summary_truncated",
                "coverage_compact",
            }
            if metadata.record_layout is not None:
                expected_compact_fields.update(record_fields)
            return (
                set(coverage) == expected_compact_fields
                and coverage.get("scan_complete")
                is bool(metadata.coverage.get("scan_complete", False))
                and coverage.get("candidate_pool_truncated")
                is bool(metadata.coverage.get("candidate_pool_truncated", False))
                and coverage.get("evidence_truncated") is evidence_truncated
                and coverage.get("summary_truncated") is True
                and self._retrieval_hint_is_accepted(retrieval_hint)
            )
        for key in (
            "scanned_start_bytes",
            "scanned_end_bytes",
            "scan_complete",
            "candidate_pool_truncated",
            "uncovered_candidates",
        ):
            if coverage.get(key) != metadata.coverage.get(key):
                return False
        expected_uncovered = (
            "lower-ranked facts, evidence, records, or ordinary content omitted"
            if summary_truncated
            else "ordinary content is represented by retained records and synopsis"
        )
        if coverage.get("omitted_key_facts") != omitted_facts:
            return False
        if coverage.get("omitted_evidence") != omitted_evidence:
            return False
        if coverage.get("evidence_truncated") is not evidence_truncated:
            return False
        if coverage.get("summary_truncated") is not summary_truncated:
            return False
        if coverage.get("uncovered") != expected_uncovered:
            return False
        return self._retrieval_hint_is_accepted(retrieval_hint)

    def stub_sequence_status(
        self,
        value: str,
        ctx: RequestContext,
        session_id: str,
        *,
        deadline: float | None = None,
    ) -> str:
        """Return valid, invalid, or unavailable for a canonical stub sequence."""
        status, _metadata = self._stub_sequence_resolution(
            value,
            ctx,
            session_id,
            deadline=deadline,
        )
        return status

    def _stub_sequence_resolution(
        self,
        value: str,
        ctx: RequestContext,
        session_id: str,
        *,
        deadline: float | None = None,
    ) -> tuple[str, tuple[ArtifactMetadata, ...]]:
        parsed = _parse_stub_blocks(value)
        if parsed is None:
            return "invalid", ()
        deadline = deadline or (time.monotonic() + self.write_timeout_ms / 1000)
        resolved: list[ArtifactMetadata] = []
        try:
            for block, fields in parsed:
                match = _REF_PATTERN.fullmatch(fields["ref"])
                if match is None:
                    return "invalid", ()
                metadata = self.io_runner.run(
                    lambda artifact_id=match.group(1): self.store.read_metadata(
                        artifact_id, session_id, ctx,
                    ),
                    deadline=deadline,
                    operation_key=session_id,
                )
                if not self._stub_matches_metadata(block, metadata):
                    return "invalid", ()
                resolved.append(metadata)
            return "valid", tuple(resolved)
        except (ArtifactNotVisibleError, FileNotFoundError, KeyError):
            return "invalid", ()
        except Exception:
            return "unavailable", ()

    def embedded_stub_projection_status(
        self,
        value: str,
        ctx: RequestContext,
        session_id: str,
        *,
        deadline: float | None = None,
        expected_source_message_id: str | None = None,
    ) -> str:
        """Verify all canonical stubs inside a flattened mixed projection."""
        parsed = _parse_embedded_stub_blocks(value)
        if parsed is None:
            return "invalid"
        residual = value
        for block, _fields in parsed:
            residual = residual.replace(block, "", 1)
        mixed_projection = bool(residual.strip())
        if mixed_projection and not expected_source_message_id:
            return "invalid"
        deadline = deadline or (time.monotonic() + self.write_timeout_ms / 1000)
        try:
            for block, fields in parsed:
                match = _REF_PATTERN.fullmatch(fields["ref"])
                if match is None:
                    return "invalid"
                metadata = self.io_runner.run(
                    lambda artifact_id=match.group(1): self.store.read_metadata(
                        artifact_id, session_id, ctx,
                    ),
                    deadline=deadline,
                    operation_key=session_id,
                )
                if not self._stub_matches_metadata(block, metadata):
                    return "invalid"
                if mixed_projection and (
                    not metadata.source_message_id
                    or metadata.source_message_id != expected_source_message_id
                ):
                    # Mixed projections need a server-issued Buffer binding for
                    # every embedded Artifact. Canonical syntax and visibility
                    # alone cannot distinguish a governed structured result
                    # from an arbitrary "valid stub + residual text" splice.
                    return "invalid"
            return "valid"
        except (ArtifactNotVisibleError, FileNotFoundError, KeyError):
            return "invalid"
        except Exception:
            return "unavailable"

    def govern_messages(
        self,
        messages: list,
        *,
        ctx: RequestContext,
        session_id: str,
        persistence: bool,
        ranking_prompt: str = "",
        fallback_original: bool = False,
    ) -> GovernedMessages:
        deadline = time.monotonic() + self.write_timeout_ms / 1000
        normalized = self.normalizer.normalize(messages)
        if fallback_original:
            originals = [deepcopy(item.original) for item in normalized]
            return GovernedMessages(
                messages=originals,
                source_message_ids=[item.source_message_id for item in normalized],
                fallback_original_contents=[
                    render_safe_message_content(message) for message in originals
                ],
                removed_message_ids=[],
                removed_fallback_messages=[],
                preserved_message_ids=[],
                fallback_original=True,
            )
        ranking_context = ranking_context_from_messages(
            normalized,
            prompt=ranking_prompt,
        )
        summary_turn_remaining = self.summary_per_turn_chars
        summary_slots_remaining = 1
        rendered_metadata_cache: dict[str, str] = {}

        def render_metadata(metadata: ArtifactMetadata) -> str:
            nonlocal summary_slots_remaining, summary_turn_remaining
            cached = rendered_metadata_cache.get(metadata.ref)
            if cached is not None:
                # Multiple attested read/search results can point back to the
                # same Artifact in one Turn. They need the same safe stub, not
                # another share of the bounded summary budget.
                return cached
            if metadata.schema_version < 2:
                stub = self.build_stub(metadata)
                summary_turn_remaining = max(
                    0,
                    summary_turn_remaining - self._stub_summary_chars(stub),
                )
                summary_slots_remaining = max(0, summary_slots_remaining - 1)
                rendered_metadata_cache[metadata.ref] = stub
                return stub
            if summary_turn_remaining < _MIN_SUMMARY_JSON_CHARS:
                raise ValueError(
                    "remaining Turn summary budget cannot represent another v2 stub"
                )
            fair_share = max(
                _MIN_SUMMARY_JSON_CHARS,
                summary_turn_remaining // max(1, summary_slots_remaining),
            )
            allowance = min(self.summary_per_artifact_chars, fair_share)
            stub = self.build_stub(
                metadata,
                ranking_context=ranking_context,
                summary_budget_chars=allowance,
            )
            summary_turn_remaining = max(
                0,
                summary_turn_remaining - self._stub_summary_chars(stub),
            )
            summary_slots_remaining = max(0, summary_slots_remaining - 1)
            rendered_metadata_cache[metadata.ref] = stub
            return stub

        def render_result(
            result: ExternalizationResult | SourceCollectionReuse,
        ) -> str:
            if isinstance(result, SourceCollectionReuse):
                if result.metadata:
                    return "\n\n".join(render_metadata(item) for item in result.metadata)
                return result.stub
            return render_metadata(result.metadata)
        valid_stub_sequences: set[tuple[int, str]] = set()
        valid_stub_metadata: dict[tuple[int, str], tuple[ArtifactMetadata, ...]] = {}
        invalid_stub_sequences: list[tuple[int, ToolResultPart]] = []
        for normalized_message in normalized:
            for part in normalized_message.tool_results:
                if not _contains_stub_marker(part.output_text):
                    continue
                key = (normalized_message.message_index, part.part_path)
                status, resolved_metadata = self._stub_sequence_resolution(
                    part.output_text,
                    ctx,
                    session_id,
                    deadline=deadline,
                )
                if status == "valid":
                    valid_stub_sequences.add(key)
                    valid_stub_metadata[key] = resolved_metadata
                else:
                    invalid_stub_sequences.append((normalized_message.message_index, part))

        selected = [
            item for item in self.policy.select(normalized)
            if (item.message_index, item.part.part_path) not in valid_stub_sequences
            and not (item.part.source_collection and not persistence)
        ]
        selected_keys = {
            (item.message_index, item.part.part_path)
            for item in selected
        }
        for message_index, part in invalid_stub_sequences:
            key = (message_index, part.part_path)
            if key not in selected_keys:
                selected.append(SelectedPart(
                    message_index,
                    part,
                    "invalid_stub_sequence",
                ))
                selected_keys.add(key)
        prevalidated_sources: dict[
            tuple[int, str], ExternalizationResult | SourceCollectionReuse
        ] = {}
        source_passthrough_keys: set[tuple[int, str]] = set()
        for key, metadata_rows in valid_stub_metadata.items():
            if metadata_rows and all(row.schema_version >= 2 for row in metadata_rows):
                prevalidated_sources[key] = SourceCollectionReuse(
                    stub="",
                    source_count=0,
                    metadata=metadata_rows,
                )
                if key not in selected_keys:
                    selected.append(SelectedPart(
                        key[0],
                        next(
                            part
                            for message in normalized
                            if message.message_index == key[0]
                            for part in message.tool_results
                            if part.part_path == key[1]
                        ),
                        "verified_stub_reprojection",
                    ))
                    selected_keys.add(key)
        invalid_source_keys: set[tuple[int, str]] = set()
        source_artifact_cache: dict[str, tuple[ArtifactMetadata, str]] = {}
        for normalized_message in normalized:
            for part in normalized_message.tool_results:
                key = (normalized_message.message_index, part.part_path)
                if key in valid_stub_sequences:
                    continue
                if not part.source_ref and not part.source_collection:
                    continue
                try:
                    if part.source_collection:
                        source_result = self._validated_source_collection(
                            part,
                            ctx,
                            session_id,
                            deadline=deadline,
                            artifact_cache=source_artifact_cache,
                        )
                    else:
                        source_result = self._validated_source_artifact(
                            part,
                            ctx,
                            session_id,
                            deadline=deadline,
                            artifact_cache=source_artifact_cache,
                        )
                except Exception:
                    source_result = None
                if source_result is None:
                    invalid_source_keys.add(key)
                    continue
                if isinstance(source_result, SourceFragmentPassthrough):
                    if not persistence:
                        source_passthrough_keys.add(key)
                        continue
                    source_result = ExternalizationResult(
                        source_result.metadata,
                        self.build_stub(source_result.metadata),
                        reused=True,
                    )
                prevalidated_sources[key] = source_result
                if key not in selected_keys and (
                    persistence or not part.source_collection
                ):
                    selected.append(SelectedPart(
                        normalized_message.message_index,
                        part,
                        "source_artifact_reuse",
                    ))
                    selected_keys.add(key)
        if source_passthrough_keys:
            selected = [
                item
                for item in selected
                if (item.message_index, item.part.part_path)
                not in source_passthrough_keys
            ]
            selected_keys.difference_update(source_passthrough_keys)
        selected.sort(key=lambda item: (item.message_index, item.part.part_path))
        unique_prevalidated_refs: set[str] = set()
        new_artifact_slots = 0
        for item in selected:
            source = prevalidated_sources.get(
                (item.message_index, item.part.part_path)
            )
            if isinstance(source, SourceCollectionReuse) and source.metadata:
                unique_prevalidated_refs.update(
                    metadata.ref for metadata in source.metadata
                )
            elif isinstance(source, ExternalizationResult):
                unique_prevalidated_refs.add(source.metadata.ref)
            elif item.part.source_collection and item.part.source_refs:
                unique_prevalidated_refs.update(item.part.source_refs)
            else:
                new_artifact_slots += 1
        summary_slots_remaining = max(
            1,
            len(unique_prevalidated_refs) + new_artifact_slots,
        )
        selected_by_message: dict[int, list[SelectedPart]] = {}
        for item in selected:
            selected_by_message.setdefault(item.message_index, []).append(item)

        output_messages: list[dict] = []
        source_ids: list[str | None] = []
        fallback_original_contents: list[str] = []
        removed_ids: list[str] = []
        removed_fallback_messages: list[tuple[str, str]] = []
        preserved_ids: list[str] = []
        externalized_count = 0
        failed_count = 0
        for normalized_message in normalized:
            rewritten = deepcopy(normalized_message.original)
            original_content = render_safe_message_content(normalized_message.original)
            preserve_existing = False
            message_externalized_count = 0
            message_created_artifact = False
            for item in selected_by_message.get(normalized_message.message_index, []):
                try:
                    result = prevalidated_sources.get(
                        (normalized_message.message_index, item.part.part_path)
                    )
                    if result is None:
                        if (
                            normalized_message.message_index,
                            item.part.part_path,
                        ) in invalid_source_keys:
                            raise InvalidSourceArtifactError(
                                "source artifact envelope is not verifiable"
                            )
                        result = self._externalize(
                            normalized_message,
                            item.part,
                            item.reason,
                            ctx,
                            session_id,
                            ranking_context=ranking_context,
                            deadline=deadline,
                        )
                    self.normalizer.set_payload(
                        rewritten,
                        item.part.payload_path,
                        render_result(result),
                    )
                    message_externalized_count += (
                        result.source_count
                        if isinstance(result, SourceCollectionReuse)
                        else 1
                    )
                    message_created_artifact = bool(
                        message_created_artifact
                        or (
                            isinstance(result, ExternalizationResult)
                            and not result.reused
                        )
                    )
                except Exception as exc:
                    failed_count += 1
                    logger.warning(
                        "tool-result externalization failed session=%s part=%s: %s",
                        session_id[:8], item.part.part_path, exc,
                    )
                    if (
                        persistence
                        and normalized_message.identity_is_stable
                        and normalized_message.source_message_id
                    ):
                        if message_created_artifact:
                            # Do not discard a newly written Artifact by
                            # preserving the previous whole-message projection.
                            # Commit the successful Stub parts and remove this
                            # failed raw part fail-closed. A retry can then reuse
                            # the stable Artifact id without creating orphans.
                            self.normalizer.remove_payload(
                                rewritten,
                                item.part.payload_path,
                            )
                            continue
                        # A stable source may already have a verified stub in the
                        # Buffer.  Keep that last-known-good projection instead
                        # of removing or downgrading it during a transient store
                        # failure.  If this is a brand-new source id, skipping it
                        # still preserves the fail-closed rule: raw output is not
                        # persisted.
                        preserve_existing = True
                        preserved_ids.append(normalized_message.source_message_id)
                        break
                    if persistence:
                        self.normalizer.remove_payload(rewritten, item.part.payload_path)

            if preserve_existing:
                continue

            if persistence and not render_safe_message_content(rewritten):
                if normalized_message.source_message_id:
                    removed_ids.append(normalized_message.source_message_id)
                elif original_content:
                    removed_fallback_messages.append(
                        (normalized_message.role, original_content)
                    )
                continue
            output_messages.append(rewritten)
            source_ids.append(normalized_message.source_message_id)
            fallback_original_contents.append(original_content)
            externalized_count += message_externalized_count

        return GovernedMessages(
            messages=output_messages,
            source_message_ids=source_ids,
            fallback_original_contents=fallback_original_contents,
            removed_message_ids=removed_ids,
            removed_fallback_messages=removed_fallback_messages,
            preserved_message_ids=preserved_ids,
            externalized_count=externalized_count,
            failed_count=failed_count,
        )

    def _externalize(
        self,
        message: NormalizedMessage,
        part: ToolResultPart,
        reason: str,
        ctx: RequestContext,
        session_id: str,
        *,
        ranking_context: str = "",
        deadline: float | None = None,
    ) -> ExternalizationResult:
        deadline = deadline or (time.monotonic() + self.write_timeout_ms / 1000)
        source_result = self._validated_source_artifact(
            part,
            ctx,
            session_id,
            deadline=deadline,
        )
        if source_result is not None:
            return source_result
        if part.source_ref or part.source_collection:
            # Never create an Artifact whose payload is merely an unverified
            # read/search envelope for another Artifact.  Model-facing callers
            # retain the envelope (fail-open); persistent callers fail closed in
            # govern_messages().
            raise InvalidSourceArtifactError("source artifact envelope is not verifiable")

        encoded = part.output_text.encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        identity_strength = (
            "strong"
            if message.identity_is_stable
            and (message.source_message_id or (message.turn_id and part.tool_call_id))
            else "fallback"
        )
        identity_fields = (
            ctx.account_id,
            ctx.user_id,
            ctx.agent_id,
            session_id,
            message.turn_id or "",
            message.source_message_id or "",
            canonical_tool_call_id(part.tool_call_id),
            part.part_path,
            str(part.occurrence_index),
        )
        if identity_strength == "strong":
            artifact_id = self.strong_artifact_id(*identity_fields)
        else:
            artifact_id = "tr_" + uuid4().hex

        detected_type, synopsis = self.synopsis_builder.build(part.output_text, part.mime_type)
        fact_candidates, evidence_candidates, coverage = self.multires_builder.extract(
            part.output_text,
            detected_type=detected_type,
            ranking_context=ranking_context,
        )
        record_candidates, record_layout = extract_record_candidates(
            part.output_text,
            detected_type=detected_type,
            tool_name=part.tool_name,
            ranking_context=ranking_context,
            config=self.multires_builder.record_config,
        )
        metadata = ArtifactMetadata(
            schema_version=2,
            artifact_id=artifact_id,
            account_id=ctx.account_id,
            session_id=session_id,
            owner_user_id=ctx.user_id,
            owner_agent_id=ctx.agent_id,
            turn_id=message.turn_id,
            source_message_id=message.source_message_id,
            tool_call_id=part.tool_call_id,
            tool_name=part.tool_name,
            part_path=part.part_path,
            occurrence_index=part.occurrence_index,
            mime_type=part.mime_type,
            detected_type=detected_type,
            serialization=part.serialization,
            original_chars=part.original_chars,
            original_bytes=part.original_bytes,
            sha256=digest,
            synopsis=synopsis,
            externalization_reason=reason,
            identity_strength=identity_strength,
            created_at=datetime.now(UTC).isoformat(),
            fact_candidates=fact_candidates,
            evidence_candidates=evidence_candidates,
            record_candidates=record_candidates,
            record_layout=record_layout,
            coverage=coverage,
        )
        last_error: Exception | None = None

        def write_and_verify(
            target_metadata: ArtifactMetadata,
            target_artifact_id: str,
            verification_error: str,
        ) -> ExternalizationResult:
            reused = self.store.write(target_metadata, part.output_text, ctx)
            verified_metadata, verified_output = self.store.read_all(
                target_artifact_id,
                session_id,
                ctx,
            )
            if verified_metadata.sha256 != digest or verified_output != part.output_text:
                raise IOError(verification_error)
            return ExternalizationResult(
                metadata=verified_metadata,
                stub=self.build_stub(
                    verified_metadata,
                    ranking_context=ranking_context,
                ),
                reused=reused,
            )

        for attempt in range(self.retry_attempts):
            try:
                return self.io_runner.run(
                    lambda: write_and_verify(
                        metadata,
                        artifact_id,
                        "artifact read-after-write verification failed",
                    ),
                    deadline=deadline,
                    operation_key=session_id,
                )
            except ExternalizationTimeoutError:
                # The timed-out daemon still owns its capacity slot until the
                # backend call exits. Retrying would duplicate blocking work.
                raise
            except FileExistsError:
                try:
                    existing_metadata, existing_output = self.io_runner.run(
                        lambda: self.store.read_all(artifact_id, session_id, ctx),
                        deadline=deadline,
                        operation_key=session_id,
                    )
                    if (
                        existing_metadata.identity_strength == "strong"
                        and existing_metadata.account_id == ctx.account_id
                        and existing_metadata.owner_user_id == ctx.user_id
                        and existing_metadata.owner_agent_id == ctx.agent_id
                        and existing_metadata.session_id == session_id
                        and existing_metadata.turn_id == message.turn_id
                        and existing_metadata.source_message_id == message.source_message_id
                        and canonical_tool_call_id(existing_metadata.tool_call_id)
                        == canonical_tool_call_id(part.tool_call_id)
                        and existing_metadata.part_path == part.part_path
                        and existing_metadata.occurrence_index == part.occurrence_index
                        and existing_metadata.sha256 == digest
                        and existing_output == part.output_text
                    ):
                        return ExternalizationResult(
                            existing_metadata,
                            self.build_stub(
                                existing_metadata,
                                ranking_context=ranking_context,
                            ),
                            reused=True,
                        )
                except ExternalizationTimeoutError:
                    raise
                except Exception:
                    pass
                conflict_id = f"{artifact_id}_{digest[:12]}"
                metadata = ArtifactMetadata.from_dict(
                    metadata.to_dict() | {"artifact_id": conflict_id},
                )
                artifact_id = conflict_id
                try:
                    return self.io_runner.run(
                        lambda: write_and_verify(
                            metadata,
                            artifact_id,
                            "artifact conflict read-after-write verification failed",
                        ),
                        deadline=deadline,
                        operation_key=session_id,
                    )
                except ExternalizationTimeoutError:
                    raise
                except Exception as exc:
                    last_error = exc
            except Exception as exc:
                last_error = exc
            if attempt + 1 >= self.retry_attempts or time.monotonic() >= deadline:
                break
            time.sleep(min(0.02 * (attempt + 1), max(0.0, deadline - time.monotonic())))
        assert last_error is not None
        raise last_error

    @staticmethod
    def strong_artifact_id(
        account_id: str,
        user_id: str,
        agent_id: str,
        session_id: str,
        turn_id: str,
        source_message_id: str,
        tool_call_key: str,
        part_path: str,
        occurrence_index: str,
    ) -> str:
        fields = (
            account_id,
            user_id,
            agent_id,
            session_id,
            turn_id,
            source_message_id,
            tool_call_key,
            part_path,
            occurrence_index,
        )
        return "tr_" + hashlib.sha256("\0".join(fields).encode()).hexdigest()[:32]

    def _validated_source_artifact(
        self,
        part: ToolResultPart,
        ctx: RequestContext,
        session_id: str,
        *,
        deadline: float | None = None,
        artifact_cache: dict[str, tuple[ArtifactMetadata, str]] | None = None,
    ) -> ExternalizationResult | SourceFragmentPassthrough | None:
        if not part.source_ref or not _REF_PATTERN.fullmatch(part.source_ref):
            return None
        deadline = deadline or (time.monotonic() + self.write_timeout_ms / 1000)
        try:
            artifact_id = _REF_PATTERN.fullmatch(part.source_ref).group(1)
            cached = artifact_cache.get(artifact_id) if artifact_cache is not None else None
            if cached is None:
                metadata, output = self.io_runner.run(
                    lambda: self.store.read_all(artifact_id, session_id, ctx),
                    deadline=deadline,
                    operation_key=session_id,
                )
                if artifact_cache is not None:
                    artifact_cache[artifact_id] = (metadata, output)
            else:
                metadata, output = cached
            existing_stub = parse_stub(part.output_text)
            if existing_stub is not None:
                if (
                    existing_stub.get("ref") != metadata.ref
                    or existing_stub.get("sha256") != metadata.sha256
                    or int(existing_stub.get("chars", -1)) != metadata.original_chars
                ):
                    return None
                return ExternalizationResult(
                    metadata,
                    self.build_stub(metadata),
                    reused=True,
                )
            envelope = json.loads(part.output_text)
            if (
                not isinstance(envelope, dict)
                or envelope.get("schema_version") not in {1, 2}
            ):
                return None
            source_tool = envelope.get("source_tool")
            if source_tool not in {
                "read_tool_result_artifact",
                "search_tool_result_artifacts",
            }:
                return None
            content = envelope.get("content")
            if not isinstance(content, str):
                return None
            content_bytes = content.encode("utf-8")
            if envelope.get("artifact_sha256") != metadata.sha256:
                return None
            if envelope.get("content_sha256") != hashlib.sha256(content_bytes).hexdigest():
                return None
            offset = int(envelope.get("offset_bytes", -1))
            actual_length = int(envelope.get("actual_length_bytes", -1))
            if offset < 0 or actual_length < 0 or actual_length != len(content_bytes):
                return None
            expected = output.encode("utf-8")[offset:offset + actual_length]
            if expected != content_bytes:
                return None
            if envelope.get("schema_version") == 2:
                if source_tool != "search_tool_result_artifacts":
                    return None
                if envelope.get("actual_offset_bytes", offset) != offset:
                    return None
                if envelope.get("next_offset_bytes", offset + actual_length) != (
                    offset + actual_length
                ):
                    return None
                if envelope.get("decoded_chars", len(content)) != len(content):
                    return None
                if not isinstance(envelope.get("fragment_complete"), bool):
                    return None
                if not isinstance(envelope.get("fragment_type"), str):
                    return None
                context_before = envelope.get("context_before")
                context_after = envelope.get("context_after")
                if not isinstance(context_before, str) or not isinstance(context_after, str):
                    return None
                before_bytes = context_before.encode("utf-8")
                after_bytes = context_after.encode("utf-8")
                before_offset = int(envelope.get("context_before_offset_bytes", -1))
                after_offset = int(envelope.get("context_after_offset_bytes", -1))
                if before_offset < 0 or before_offset + len(before_bytes) != offset:
                    return None
                if after_offset != offset + actual_length:
                    return None
                output_bytes = output.encode("utf-8")
                if output_bytes[before_offset:offset] != before_bytes:
                    return None
                if output_bytes[after_offset:after_offset + len(after_bytes)] != after_bytes:
                    return None
                if envelope.get("context_before_sha256") != hashlib.sha256(
                    before_bytes
                ).hexdigest():
                    return None
                if envelope.get("context_after_sha256") != hashlib.sha256(
                    after_bytes
                ).hexdigest():
                    return None
                match_offset = int(envelope.get("match_offset_bytes", -1))
                match_length = int(envelope.get("match_length_bytes", -1))
                if (
                    match_offset < offset
                    or match_length < 1
                    or match_offset + match_length > offset + actual_length
                ):
                    return None
                if envelope.get("returned_chars") not in {
                    None,
                    len(context_before) + len(content) + len(context_after),
                }:
                    return None
            if source_tool == "read_tool_result_artifact":
                return SourceFragmentPassthrough(metadata)
            return ExternalizationResult(metadata, self.build_stub(metadata), reused=True)
        except (ArtifactNotVisibleError, TypeError, ValueError, AttributeError, json.JSONDecodeError):
            return None

    def _validated_source_collection(
        self,
        part: ToolResultPart,
        ctx: RequestContext,
        session_id: str,
        *,
        deadline: float,
        artifact_cache: dict[str, tuple[ArtifactMetadata, str]],
    ) -> SourceCollectionReuse | None:
        if not part.source_collection:
            return None
        try:
            envelope = json.loads(part.output_text)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(envelope, dict):
            return None
        results = envelope.get("results")
        metadata_listing = False
        if not isinstance(results, list) and isinstance(envelope.get("artifacts"), list):
            legacy_fields = {"artifacts", "truncated", "next_cursor"}
            marked_fields = legacy_fields | {
                "_ogmemory_artifact_list_source",
                "schema_version",
                "source_tool",
            }
            envelope_fields = frozenset(envelope)
            if envelope_fields not in {
                frozenset(legacy_fields),
                frozenset(marked_fields),
            }:
                return None
            if envelope_fields == frozenset(marked_fields) and (
                envelope.get("_ogmemory_artifact_list_source") is not True
                or envelope.get("schema_version") != 1
                or envelope.get("source_tool") != "list_tool_result_artifacts"
            ):
                return None
            results = envelope["artifacts"]
            metadata_listing = True
        if (
            not isinstance(results, list)
            or not results
            or len(results) != len(part.source_refs)
        ):
            return None
        if metadata_listing:
            truncated = envelope.get("truncated")
            next_cursor = envelope.get("next_cursor")
            if not isinstance(truncated, bool):
                return None
            if truncated:
                if next_cursor != results[-1].get("artifact_id"):
                    return None
            elif next_cursor is not None:
                return None
            refs = [row.get("ref") for row in results if isinstance(row, dict)]
            if len(refs) != len(results) or len(set(refs)) != len(refs):
                return None

        verified: list[ExternalizationResult] = []
        for index, (row, expected_ref) in enumerate(zip(results, part.source_refs)):
            if metadata_listing:
                if not isinstance(row, dict) or row.get("ref") != expected_ref:
                    return None
                match = _REF_PATTERN.fullmatch(expected_ref)
                if match is None:
                    return None
                metadata = self.io_runner.run(
                    lambda artifact_id=match.group(1): self.store.read_metadata(
                        artifact_id, session_id, ctx,
                    ),
                    deadline=deadline,
                    operation_key=session_id,
                )
                if row != metadata.public_dict() | {"ref": metadata.ref}:
                    return None
                source = ExternalizationResult(
                    metadata=metadata,
                    stub=self.build_stub(metadata),
                    reused=True,
                )
            else:
                if self.normalizer._source_ref(row) != expected_ref:
                    return None
                row_text = json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                row_part = ToolResultPart(
                    part_path=f"{part.part_path}/results/{index}",
                    payload_path=part.payload_path,
                    tool_call_id=part.tool_call_id,
                    tool_name=part.tool_name,
                    output_text=row_text,
                    mime_type=part.mime_type,
                    serialization="canonical_json",
                    source_ref=expected_ref,
                    original_chars=len(row_text),
                    original_bytes=len(row_text.encode("utf-8")),
                    occurrence_index=part.occurrence_index,
                )
                source = self._validated_source_artifact(
                    row_part,
                    ctx,
                    session_id,
                    deadline=deadline,
                    artifact_cache=artifact_cache,
                )
            if not isinstance(source, ExternalizationResult):
                return None
            verified.append(source)

        unique_stubs: list[str] = []
        unique_metadata: list[ArtifactMetadata] = []
        seen_refs: set[str] = set()
        for source in verified:
            if source.metadata.ref in seen_refs:
                continue
            seen_refs.add(source.metadata.ref)
            unique_stubs.append(source.stub)
            unique_metadata.append(source.metadata)
        return SourceCollectionReuse(
            stub="\n\n".join(unique_stubs),
            source_count=len(unique_stubs),
            metadata=tuple(unique_metadata),
        )

    @staticmethod
    def _safe_stub_field(value: str) -> str:
        return json.dumps(value, ensure_ascii=False)[1:-1]

    @staticmethod
    def _retrieval_hint() -> str:
        return (
            "Answer from retained_records, key_facts, selected_evidence, and existing "
            "assistant analysis first. "
            "Search the Artifact only when those are insufficient. Search returns precise "
            "fragments; use interval reads only if a precise fragment is still insufficient. "
            "Never guess a missing ID, number, status, error, count, order, first, or last value. "
            "Do not mention oGMemory or any related terminology. Keep oGMemory, retrieval, "
            "and other internal mechanisms transparent to the user, and decide autonomously "
            "when retrieval is necessary. Do not add caveats merely because the evidence is "
            "a compact representation."
        )

    @classmethod
    def _retrieval_hint_is_accepted(cls, value: object) -> bool:
        return value == cls._retrieval_hint()

    @staticmethod
    def _stub_summary_chars(stub: str) -> int:
        fields = parse_stub(stub) or {}
        total = sum(len(fields.get(key, "")) for key in (
            "synopsis", "key_facts", "selected_evidence", "coverage",
        ))
        retained_records = fields.get("retained_records", "")
        if retained_records not in {"", "null"}:
            total += len(retained_records)
        return total

    def build_stub(
        self,
        metadata: ArtifactMetadata,
        *,
        ranking_context: str = "",
        summary_budget_chars: int | None = None,
    ) -> str:
        if metadata.schema_version != 2 or not isinstance(metadata.coverage, dict):
            raise ValueError("only v2 tool-result metadata can be rendered as a stub")
        projection = self.multires_builder.project(
            synopsis=metadata.synopsis,
            fact_candidates=metadata.fact_candidates,
            evidence_candidates=metadata.evidence_candidates,
            base_coverage=metadata.coverage,
            ranking_context=ranking_context,
            max_chars=(
                self.summary_per_artifact_chars
                if summary_budget_chars is None
                else max(_MIN_SUMMARY_JSON_CHARS, summary_budget_chars)
            ),
            record_candidates=metadata.record_candidates,
            record_layout=metadata.record_layout,
        )
        tool_name = self._safe_stub_field((metadata.tool_name or "unknown")[:200])
        session_id = self._safe_stub_field(metadata.session_id)
        encoded = lambda value: json.dumps(  # noqa: E731 - keeps field rendering uniform
            value,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return "\n".join((
            _STUB_START_V2,
            f"ref: {metadata.ref}",
            f"session_id: {session_id}",
            f"tool: {tool_name}",
            f"type: {metadata.detected_type}",
            f"chars: {metadata.original_chars}",
            f"bytes: {metadata.original_bytes}",
            f"sha256: {metadata.sha256}",
            f"synopsis: {encoded(projection.synopsis)}",
            f"key_facts: {encoded(list(projection.key_facts))}",
            f"selected_evidence: {encoded(list(projection.selected_evidence))}",
            f"retained_records: {encoded(projection.retained_records)}",
            f"coverage: {encoded(projection.coverage)}",
            f"retrieval_hint: {encoded(self._retrieval_hint())}",
            _STUB_END,
        ))

    def build_stub_sequence(
        self,
        metadata_rows: Iterable[ArtifactMetadata],
        *,
        ranking_context: str = "",
    ) -> tuple[str, ...]:
        """Render one ordered set under the configured shared Turn budget."""
        rows = tuple(metadata_rows)
        if any(
            metadata.schema_version != 2 or not isinstance(metadata.coverage, dict)
            for metadata in rows
        ):
            raise ValueError("only v2 tool-result metadata can be rendered as a stub")
        if self.summary_per_turn_chars < _MIN_SUMMARY_JSON_CHARS * len(rows):
            raise ValueError(
                "Turn summary budget cannot represent all requested v2 stubs"
            )
        remaining = self.summary_per_turn_chars
        slots = len(rows)
        stubs: list[str] = []
        for metadata in rows:
            allowance = max(
                _MIN_SUMMARY_JSON_CHARS,
                min(
                    self.summary_per_artifact_chars,
                    remaining // max(1, slots),
                ),
            )
            stub = self.build_stub(
                metadata,
                ranking_context=ranking_context,
                summary_budget_chars=allowance,
            )
            stubs.append(stub)
            remaining = max(0, remaining - self._stub_summary_chars(stub))
            slots -= 1
        return tuple(stubs)


def search_artifacts(
    store: ToolResultArtifactStore,
    *,
    session_id: str,
    query: str,
    ctx: RequestContext,
    top_k: int,
    max_chars: int,
    max_artifacts: int,
    chunk_bytes: int,
    overlap_bytes: int,
    deadline_ms: int,
    artifact_ids: set[str] | None = None,
    io_runner: BoundedDaemonIORunner | None = None,
) -> dict:
    """Locate multiple query terms, merge structural fragments, then rank globally."""
    if not query.strip():
        raise ValueError("query must not be empty")
    plan = build_query_plan(query)
    deadline = time.monotonic() + deadline_ms / 1000
    deadline_reached = False
    listing_complete = False
    rows: list[dict] = []
    cursor = None
    while time.monotonic() < deadline:
        try:
            page = (
                io_runner.run(
                    lambda: store.list(session_id, ctx, limit=100, cursor=cursor),
                    deadline=deadline,
                    operation_key=session_id,
                )
                if io_runner is not None
                else store.list(session_id, ctx, limit=100, cursor=cursor)
            )
        except ExternalizationTimeoutError:
            deadline_reached = True
            break
        page_rows = page.get("artifacts", [])
        if artifact_ids is not None:
            page_rows = [
                row for row in page_rows
                if row.get("artifact_id") in artifact_ids
            ]
        rows.extend(page_rows)
        if not page.get("truncated"):
            listing_complete = True
            break
        next_cursor = page.get("next_cursor")
        if not isinstance(next_cursor, str) or not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor
    def metadata_score(row: dict) -> float:
        searchable = " ".join((
            str(row.get("tool_name", "")),
            str(row.get("synopsis", "")),
            json.dumps(row.get("fact_candidates", []), ensure_ascii=False),
        ))
        match = best_query_match(searchable, query)
        return match.score if match is not None else 0.0

    metadata_scores = {
        str(row.get("artifact_id", "")): metadata_score(row)
        for row in rows
    }
    candidates = sorted(
        rows,
        key=lambda row: (
            -metadata_scores.get(str(row.get("artifact_id", "")), 0.0),
            str(row.get("artifact_id", "")),
        ),
    )[:max_artifacts]
    artifact_limit_reached = len(rows) > max_artifacts
    scan_intervals: dict[str, list[tuple[int, int]]] = {}
    fully_scanned_artifacts: set[str] = set()

    def record_scanned_interval(artifact_id: str, start: int, end: int) -> None:
        if end <= start:
            return
        intervals = sorted([*scan_intervals.get(artifact_id, []), (start, end)])
        merged: list[tuple[int, int]] = []
        for interval_start, interval_end in intervals:
            if not merged or interval_start > merged[-1][1]:
                merged.append((interval_start, interval_end))
                continue
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, interval_end))
        scan_intervals[artifact_id] = merged

    locators: list[dict] = []
    seen_locators: set[tuple[str, int, int, str, bool]] = set()
    for row in candidates:
        if time.monotonic() >= deadline:
            break
        artifact_id = row["artifact_id"]
        metadata = ArtifactMetadata.from_dict(row)
        total_bytes = metadata.original_bytes
        step = max(1, chunk_bytes - overlap_bytes)
        artifact_eof = total_bytes == 0
        for offset in range(0, total_bytes, step):
            if time.monotonic() >= deadline:
                break
            try:
                page = (
                    io_runner.run(
                        lambda: store.read_range(
                            artifact_id, session_id, ctx, offset, chunk_bytes,
                        ),
                        deadline=deadline,
                        operation_key=session_id,
                    )
                    if io_runner is not None
                    else store.read_range(
                        artifact_id, session_id, ctx, offset, chunk_bytes,
                    )
                )
            except ExternalizationTimeoutError:
                deadline_reached = True
                break
            page_start = max(0, int(page.get("actual_offset_bytes", offset)))
            page_end = min(
                total_bytes,
                int(page.get(
                    "next_offset_bytes",
                    page_start + int(page.get("actual_length_bytes", 0)),
                )),
            )
            record_scanned_interval(artifact_id, page_start, page_end)
            content = page["content"]
            for occurrence in query_occurrences(
                content,
                plan,
                deadline=deadline,
            ):
                match_prefix = content[:occurrence.char_start].encode("utf-8")
                match_bytes = content[
                    occurrence.char_start:occurrence.char_end
                ].encode("utf-8")
                match_offset = page["actual_offset_bytes"] + len(match_prefix)
                locator_key = (
                    artifact_id,
                    match_offset,
                    max(1, len(match_bytes)),
                    occurrence.value.casefold(),
                    occurrence.exact_phrase,
                )
                if locator_key in seen_locators:
                    continue
                seen_locators.add(locator_key)
                locators.append({
                    "ref": metadata.ref,
                    "artifact_id": artifact_id,
                    "artifact_sha256": metadata.sha256,
                    "detected_type": metadata.detected_type,
                    "match_offset_bytes": match_offset,
                    "match_length_bytes": max(1, len(match_bytes)),
                    "_query_term": occurrence.value,
                    "_query_term_kind": occurrence.kind,
                    "_exact_phrase_hits": int(occurrence.exact_phrase),
                    "_metadata_score": metadata_scores.get(artifact_id, 0.0),
                })
            if page["eof"]:
                artifact_eof = True
                break
        intervals = scan_intervals.get(artifact_id, [])
        if total_bytes == 0 or (
            artifact_eof
            and len(intervals) == 1
            and intervals[0][0] == 0
            and intervals[0][1] >= total_bytes
        ):
            fully_scanned_artifacts.add(artifact_id)

    metadata_by_artifact = {
        str(row.get("artifact_id", "")): ArtifactMetadata.from_dict(row)
        for row in candidates
    }
    fragment_pool: dict[tuple[str, int, int, str], dict] = {}
    window_cache: OrderedDict[tuple[str, int, int], dict] = OrderedDict()

    def window_bounds(
        match_offset: int,
        match_length: int,
        total_bytes: int,
        window_bytes: int,
    ) -> tuple[int, int]:
        size = min(total_bytes, max(match_length, window_bytes))
        quantum = max(1, size // 4)
        start = max(0, match_offset - size // 2)
        start = (start // quantum) * quantum
        start = min(start, max(0, total_bytes - size))
        match_end = min(total_bytes, match_offset + max(1, match_length))
        if match_end > start + size:
            start = min(max(0, total_bytes - size), match_end - size)
        return start, max(1, min(size, total_bytes - start))

    def read_fragment_window(
        artifact_id: str,
        offset_bytes: int,
        length_bytes: int,
    ) -> dict:
        key = (artifact_id, offset_bytes, length_bytes)
        cached = window_cache.get(key)
        if cached is not None:
            window_cache.move_to_end(key)
            return cached
        page = (
            io_runner.run(
                lambda: store.read_range(
                    artifact_id,
                    session_id,
                    ctx,
                    offset_bytes,
                    length_bytes,
                ),
                deadline=deadline,
                operation_key=session_id,
            )
            if io_runner is not None
            else store.read_range(
                artifact_id,
                session_id,
                ctx,
                offset_bytes,
                length_bytes,
            )
        )
        window_cache[key] = page
        while len(window_cache) > _MAX_FRAGMENT_WINDOW_CACHE:
            window_cache.popitem(last=False)
        return page

    def materialize_fragment(locator: dict, *, max_fragment_chars: int) -> dict:
        artifact_id = locator["artifact_id"]
        metadata = metadata_by_artifact[artifact_id]
        maximum_window = min(metadata.original_bytes, _MAX_FRAGMENT_WINDOW_BYTES)
        window_size = min(
            maximum_window,
            max(
                4_096,
                min(chunk_bytes, _MAX_FRAGMENT_WINDOW_BYTES),
                int(locator["match_length_bytes"]) + 512,
            ),
        )
        fragment = None
        page = None
        while window_size > 0:
            if time.monotonic() >= deadline:
                raise ExternalizationTimeoutError(
                    "tool-result fragment materialization timed out"
                )
            window_start, window_length = window_bounds(
                int(locator["match_offset_bytes"]),
                int(locator["match_length_bytes"]),
                metadata.original_bytes,
                window_size,
            )
            page = read_fragment_window(
                artifact_id,
                window_start,
                window_length,
            )
            fragment = precise_fragment_from_window(
                str(page["content"]),
                window_offset_bytes=int(page["actual_offset_bytes"]),
                artifact_total_bytes=metadata.original_bytes,
                match_offset_bytes=int(locator["match_offset_bytes"]),
                match_length_bytes=int(locator["match_length_bytes"]),
                detected_type=metadata.detected_type,
                max_chars=max_fragment_chars,
            )
            if (
                not fragment["_needs_window_expansion"]
                or window_size >= maximum_window
            ):
                break
            window_size = min(maximum_window, window_size * 2)
        if fragment is None or page is None:
            raise ExternalizationTimeoutError(
                "tool-result fragment materialization timed out"
            )
        fragment.update({
            "ref": metadata.ref,
            "artifact_id": artifact_id,
            "artifact_sha256": metadata.sha256,
            "_metadata_score": locator.get("_metadata_score", 0.0),
            "_window_request_offset_bytes": window_start,
            "_window_request_length_bytes": window_length,
            "_artifact_total_bytes": metadata.original_bytes,
            "_detected_type": metadata.detected_type,
        })
        return fragment

    for locator in locators:
        if time.monotonic() >= deadline:
            break
        try:
            fragment = materialize_fragment(locator, max_fragment_chars=4_000)
        except ExternalizationTimeoutError:
            deadline_reached = True
            break
        fragment_key = (
            fragment["ref"],
            fragment["offset_bytes"],
            fragment["actual_length_bytes"],
            fragment["fragment_type"],
        )
        existing = fragment_pool.get(fragment_key)
        if existing is None:
            fragment_pool[fragment_key] = fragment
        else:
            existing["_metadata_score"] = max(
                float(existing.get("_metadata_score", 0.0)),
                float(fragment.get("_metadata_score", 0.0)),
            )

    ranked = rank_structural_fragments(
        list(fragment_pool.values()),
        plan,
        deadline=deadline,
    )

    selected: list[dict] = []
    chars = 0
    for hit in ranked:
        if (
            len(selected) >= top_k
            or chars >= max_chars
            or time.monotonic() >= deadline
        ):
            break
        remaining = max_chars - chars
        if int(hit.get("returned_chars", 0)) > remaining:
            try:
                page = read_fragment_window(
                    hit["artifact_id"],
                    hit["_window_request_offset_bytes"],
                    hit["_window_request_length_bytes"],
                )
            except ExternalizationTimeoutError:
                deadline_reached = True
                break
            hit = {
                **hit,
                **precise_fragment_from_window(
                    str(page["content"]),
                    window_offset_bytes=int(page["actual_offset_bytes"]),
                    artifact_total_bytes=hit["_artifact_total_bytes"],
                    match_offset_bytes=hit["match_offset_bytes"],
                    match_length_bytes=hit["match_length_bytes"],
                    detected_type=hit["_detected_type"],
                    max_chars=remaining,
                ),
            }
        chars += int(hit.get("returned_chars", len(hit.get("content", ""))))
        for field in (
            "_exact_phrase_hits",
            "_covered_query_terms",
            "_metadata_score",
            "_needs_window_expansion",
            "_window_request_offset_bytes",
            "_window_request_length_bytes",
            "_artifact_total_bytes",
            "_detected_type",
        ):
            hit.pop(field, None)
        selected.append(hit)
    deadline_reached = deadline_reached or time.monotonic() >= deadline
    candidate_artifact_ids = {
        str(row.get("artifact_id", ""))
        for row in candidates
        if row.get("artifact_id")
    }
    scanned_bytes = sum(
        end - start
        for intervals in scan_intervals.values()
        for start, end in intervals
    )
    search_complete = (
        listing_complete
        and not artifact_limit_reached
        and candidate_artifact_ids.issubset(fully_scanned_artifacts)
        and not deadline_reached
    )
    return {
        "query": query,
        "results": selected,
        "search_complete": search_complete,
        "no_match_confirmed": search_complete and not ranked,
        "scanned_artifacts": len(fully_scanned_artifacts),
        "scanned_bytes": scanned_bytes,
        "results_truncated": len(selected) < len(ranked),
        "deadline_reached": deadline_reached,
    }
