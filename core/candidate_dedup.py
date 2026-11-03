"""Conservative equality for occurrence drafts before reconciliation."""

import json
from collections import defaultdict, deque
from dataclasses import asdict, replace

from core.models import CandidateMemory


def candidate_draft_fields(candidate: CandidateMemory) -> dict:
    """Preserve legacy serialization when optional internal partition identity is absent."""
    fields = asdict(candidate)
    for name in ("source_occurrence_id", "source_contexts", "node_draft_indices"):
        if not fields.get(name):
            fields.pop(name, None)
    return fields


def occurrence_replay_fields(candidate: CandidateMemory) -> dict:
    """Complete occurrence facts and partition identity used for exact replay."""
    fields = candidate_draft_fields(candidate)
    for name in ("routing_key", "legacy_routing_key", "target_uri", "expected_version",
                 "confidence", "provenance_ids", "complete_occurrence_context", "write_operation_id",
                 "verified_routing_aliases", "verified_object_aliases"):
        fields.pop(name)
    return fields


def event_draft_signature(candidate: CandidateMemory) -> str:
    """Only confidence and source IDs may differ in an exact duplicate.

    A shared routing key is not evidence of a shared occurrence. Keep different
    bodies, occurrence fields and attribution for the later semantic review.
    """
    fields = candidate_draft_fields(candidate)
    fields.pop("confidence")
    fields.pop("provenance_ids")
    fields.pop("write_operation_id")
    return json.dumps(fields, ensure_ascii=False, sort_keys=True)


def merge_duplicate_event_sources(
    existing: CandidateMemory, candidate: CandidateMemory,
) -> CandidateMemory:
    """Merge evidence IDs for equal drafts without mutating either input."""
    winner = candidate if candidate.confidence > existing.confidence else existing
    return replace(winner, provenance_ids=list(dict.fromkeys(
        [*existing.provenance_ids, *candidate.provenance_ids])))


def case_draft_signature(candidate: CandidateMemory) -> str:
    """Compare complete case outputs, excluding evidence and write bookkeeping."""
    fields = candidate_draft_fields(candidate)
    for name in ("confidence", "provenance_ids", "write_operation_id"):
        fields.pop(name, None)
    return json.dumps(fields, ensure_ascii=False, sort_keys=True)


def normalize_same_span_case_runs(
    first: list[CandidateMemory], second: list[CandidateMemory],
) -> list[CandidateMemory]:
    """Pair exact case outputs one-to-one between two known runs of one span.

    This normalizes repeated extraction output, not real occurrence identity.
    Each run's multiplicity survives, and different complete facts never pair.
    Callers must supply the original separate run batches from one prepared span.
    """
    return normalize_same_span_occurrence_runs(first, second, categories=frozenset({"case"}))


def normalize_same_span_occurrence_runs(
    first: list[CandidateMemory], second: list[CandidateMemory],
    *, categories: frozenset[str] = frozenset({"event", "case"}),
) -> list[CandidateMemory]:
    """Pair exact outputs from known native runs without losing multiplicity."""
    result = list(first)
    unmatched: dict[str, deque[int]] = defaultdict(deque)
    for index, candidate in enumerate(first):
        if candidate.category in categories and not candidate.target_uri and not candidate.write_operation_id:
            unmatched[case_draft_signature(candidate)].append(index)
    for candidate in second:
        if candidate.category in categories and not candidate.target_uri and not candidate.write_operation_id:
            matches = unmatched.get(case_draft_signature(candidate))
            if matches:
                index = matches.popleft()
                result[index] = merge_duplicate_event_sources(result[index], candidate)
                continue
        result.append(candidate)
    return result
