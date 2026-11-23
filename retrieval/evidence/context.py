"""Pure caller-side rendering of source-checked message evidence."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

from core.evidence_query import MessageEvidenceResult
from retrieval.evidence.ranking import fact_matches, query_topics, terms


@dataclass
class EvidenceContext:
    text: str
    blocks: list[dict]
    diagnostics: dict

    def to_dict(self):
        return asdict(self)


def fit_evidence_payload(payload, remaining_chars):
    """Fit complete groups and their supporting facts in the query response."""
    units = payload["evidence"]
    facts = payload["facts"]
    groups = payload["diagnostics"]["evidence_groups"]
    fitted = None
    selected = set()
    selected_groups = []
    for group in groups:
        keys = selected | set(group)
        trial = {
            "evidence": [unit for unit in units if unit["evidence_unit_id"] in keys],
            "facts": [fact for fact in facts if all(ref["evidence_unit_id"] in keys for ref in fact["source_refs"])],
            "groups": [*selected_groups, group],
            "partial": payload["diagnostics"]["partial"],
            "truncated": payload["diagnostics"]["truncated"] or len(keys) < len(units),
        }
        size = len(json.dumps({"message_evidence": trial}, ensure_ascii=False))
        # The enclosing object contributes a comma and a space in place of braces.
        if size <= remaining_chars:
            fitted = trial
            selected = keys
            selected_groups.append(group)
    return fitted


def _context_spans(text, quotes, topics):
    """Keep each quote's neighbouring sentences, merging overlapping excerpts."""
    boundaries = [0] + [match.end() for match in re.finditer(r"[.!?](?=\s|$)|[。！？]|\n+", text)]
    if boundaries[-1] != len(text):
        boundaries.append(len(text))
    # Stored facts are not exhaustive. Other topical source sentences must
    # survive even if extraction did not create a record for them.
    anchors = list(quotes) + [(start, end) for start, end in zip(boundaries, boundaries[1:])
                             if topics & set(terms(text[start:end]))]
    spans = []
    for start, end in anchors:
        first = next(i for i in range(len(boundaries) - 1) if boundaries[i] <= start < boundaries[i + 1])
        last = next(i for i in range(first, len(boundaries) - 1) if end <= boundaries[i + 1])
        spans.append((boundaries[max(0, first - 1)], boundaries[min(len(boundaries) - 1, last + 2)]))
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def _quote_spans(payload, units):
    """The service verifies facts; rendering also checks exact source alignment.

    This is not fresh ACL/version verification. A saved response can become stale when source authority changes.
    """
    topics = query_topics(payload["query"], units)
    quotes = {key: [] for key in units}
    for record in payload.get("facts", []):
        if not (topics & set(terms(record["quote"])) and fact_matches(payload["query"], record)):
            continue
        refs = record.get("source_refs", [])
        aligned = []
        for ref in refs:
            unit = units.get(ref["evidence_unit_id"])
            if unit is None:
                break
            start, end = ref["span"]
            offset = unit["parent_span"][0]
            if (not offset <= start < end <= unit["parent_span"][1]
                    or unit["text"][start - offset:end - offset] != record["quote"]
                    or ref["quote"] != record["quote"]
                    or any(ref[name] != unit[name] for name in ("source_key", "content_hash", "session_id", "message_id"))
                    or any(record.get(name) != unit.get(name) for name in ("role", "speaker", "recorded_at"))):
                break
            aligned.append((ref["evidence_unit_id"], start - offset, end - offset))
        else:
            for key, start, end in aligned:
                quotes[key].append((start, end))
    return quotes


def render_evidence_context(result: MessageEvidenceResult | dict, *, max_chars=30000):
    """Render bounded historical data without splitting required dependencies.

    Accept the shared Python result or an HTTP/MCP payload. Independent units
    may use relevant verified quotes with neighbouring sentences; multi-unit
    dependencies keep full text. Older responses lacking group diagnostics are
    conservatively treated as one atomic group. No model or storage calls occur.
    """
    if type(max_chars) is not int or max_chars < 1:
        raise ValueError("max_chars must be a positive integer")
    payload = result.to_dict() if isinstance(result, MessageEvidenceResult) else result
    units = {unit["evidence_unit_id"]: unit for unit in payload.get("evidence", [])}
    quotes = _quote_spans(payload, units)
    topics = query_topics(payload["query"], units)
    groups = payload.get("diagnostics", {}).get("evidence_groups")
    legacy = groups is None
    if legacy:
        groups = [list(units)] if units else []
    # Keep the notice even when an adapter consumes blocks instead of text.
    notice = payload.get("untrusted_data_notice", MessageEvidenceResult.untrusted_data_notice)

    def block(key, full):
        unit = units[key]
        text = unit["text"]
        spans = _context_spans(text, quotes[key], topics) if quotes[key] and not full else [[0, len(text)]]
        absolute = [[unit["parent_span"][0] + start, unit["parent_span"][0] + end] for start, end in spans]
        header = {"unit": key, "source": unit.get("source_uri", ""), "session": unit["session_id"],
                  "message": unit["message_id"], "role": unit["role"], "speaker": unit.get("speaker", ""),
                  "recorded_at": unit.get("recorded_at"), "spans": absolute}
        body = ("[…]\n" if spans[0][0] else "") + "\n[…]\n".join(text[start:end] for start, end in spans)
        if spans[-1][1] < len(text):
            body += "\n[…]"
        return {"evidence_unit_id": key, "text": json.dumps(header, ensure_ascii=False) + "\n" + body,
                "spans": absolute, "parent_span": list(unit["parent_span"])}

    selected, skipped = {}, []
    for group in groups:
        # Incomplete groups cannot become singleton evidence.
        if not group or any(key not in units for key in group):
            skipped.append({"unit_ids": list(group), "reason": "missing_dependency"})
            continue
        trial = dict(selected)
        for key in group:
            # A later dependent may require upgrading an earlier short excerpt
            # to full context. Commit that upgrade atomically with the group.
            if key not in trial or len(group) > 1:
                trial[key] = block(key, full=len(group) > 1)
        text = notice + "\n\n" + "\n\n".join(item["text"] for item in trial.values())
        if len(text) > max_chars:
            skipped.append({"unit_ids": list(group), "reason": "character_budget"})
            continue
        selected = trial
    blocks = list(selected.values())
    if blocks:
        blocks[0] = {**blocks[0], "text": notice + "\n\n" + blocks[0]["text"]}
    text = "\n\n".join(item["text"] for item in blocks)
    shortened = [key for key, item in selected.items() if item["spans"] != [units[key]["parent_span"]]]
    return EvidenceContext(text, blocks, {
        "context_chars": len(text), "selected_unit_ids": list(selected),
        "omitted_unit_ids": [key for key in units if key not in selected],
        "shortened_unit_ids": shortened, "skipped_groups": skipped, "legacy_grouping": legacy,
    })
