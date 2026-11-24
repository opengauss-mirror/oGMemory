"""Pure caller-side rendering of source-checked message evidence."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

from core.evidence_query import MessageEvidenceResult, facts_supported_by, source_refs_covered
from retrieval.evidence.ranking import fact_matches, query_topics, terms


@dataclass
class EvidenceContext:
    text: str
    blocks: list[dict]
    diagnostics: dict

    def to_dict(self):
        return asdict(self)


def evidence_is_incomplete(diagnostics) -> bool:
    return diagnostics["partial"] or diagnostics["truncated"]


def fit_evidence_payload(payload, remaining_chars):
    """Fit complete groups and their supporting facts in the query response."""
    units = payload["evidence"]
    facts = payload["facts"]
    diagnostics = payload["diagnostics"]
    groups = diagnostics["evidence_groups"]

    def fits(candidate):
        # The enclosing object contributes a comma and a space in place of braces.
        return len(json.dumps({"message_evidence": candidate}, ensure_ascii=False)) <= remaining_chars

    def build_payload(included_units, included_facts, included_groups):
        return {
            "evidence": included_units,
            "facts": included_facts,
            "groups": included_groups,
            "partial": diagnostics["partial"],
            "truncated": diagnostics["truncated"] or len(included_units) < len(units),
        }

    fitted = None
    selected = set()
    selected_groups = []
    for group in groups:
        keys = selected | set(group)
        trial = build_payload(
            [unit for unit in units if unit["evidence_unit_id"] in keys],
            facts_supported_by(facts, keys),
            [*selected_groups, group],
        )
        if fits(trial):
            fitted = trial
            selected = keys
            selected_groups.append(group)
    if fitted is None and evidence_is_incomplete(diagnostics):
        empty = build_payload([], [], [])
        if fits(empty):
            fitted = empty
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


def _quote_spans(query, facts, topics, units):
    """The service verifies facts; rendering also checks exact source alignment.

    This is not fresh ACL/version verification. A saved response can become stale when source authority changes.
    """
    quotes = {key: [] for key in units}
    for record in facts:
        quote = record.get("quote")
        if not isinstance(quote, str) or not (topics & set(terms(quote)) and fact_matches(query, record)):
            continue
        refs = record.get("source_refs", [])
        if not source_refs_covered(refs, units):
            continue
        aligned = []
        for ref in refs:
            key = ref["evidence_unit_id"]
            unit = units[key]
            span = ref.get("span")
            if not isinstance(span, (list, tuple)) or len(span) != 2:
                break
            start, end = span
            offset = unit["parent_span"][0]
            if (type(start) is not int or type(end) is not int or not offset <= start < end <= unit["parent_span"][1]
                    or unit["text"][start - offset:end - offset] != quote or ref.get("quote") != quote
                    or any(ref.get(name) != unit[name] for name in ("source_key", "content_hash", "session_id", "message_id"))
                    or any(record.get(name) != unit.get(name) for name in ("role", "speaker", "recorded_at"))):
                break
            aligned.append((key, start - offset, end - offset))
        else:
            for key, start, end in aligned:
                quotes[key].append((start, end))
    return quotes


def render_evidence_context(result: MessageEvidenceResult | dict, *, query: str | None = None, max_chars=30000):
    """Render bounded historical data without splitting required dependencies.

    Accept a Python evidence result, an HTTP/MCP query response, or its
    message_evidence object. Pass the query separately with a standalone
    message_evidence object to enable quote-based excerpts. Multi-unit
    dependencies keep full text. Results without groups are kept together.
    """
    if type(max_chars) is not int or max_chars < 1:
        raise ValueError("max_chars must be a positive integer")
    payload = result.to_dict() if isinstance(result, MessageEvidenceResult) else result
    if query is None:
        query = payload.get("query", "")
    if "hits" in payload or "message_evidence" in payload:
        payload = payload.get("message_evidence") or {}
    units = {unit["evidence_unit_id"]: unit for unit in payload.get("evidence", [])}
    topics = query_topics(query, units)
    quotes = _quote_spans(query, payload.get("facts", []), topics, units)
    groups = payload.get("groups", payload.get("diagnostics", {}).get("evidence_groups"))
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
