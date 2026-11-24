"""Recheck structured assertions against their frozen original unit and rule version."""

from bisect import bisect_right
from functools import lru_cache, partial
from types import SimpleNamespace

from core.evidence import has_source_refs
from core.evidence_timing import timed
from extraction.deterministic.extractor import (
    RULE_VERSION,
    build_assertion,
    fact_id,
    is_assertion_quote,
    overlaps_quotation,
    quotation_ranges,
)


def record_verifier(store, ctx, resolve=None):
    """Reuse validated bodies and quotation offsets during one verification pass."""
    ranges = lru_cache(maxsize=8)(quotation_ranges)

    def quoted(text, start, end):
        starts, ends = ranges(text)
        position = bisect_right(ends, start)
        return position < len(starts) and starts[position] < end

    return partial(verify_record, store=store, ctx=ctx,
                   resolve=resolve if resolve is not None else store.reference_reader(ctx), quoted=quoted)


@timed("fact_verify")
def verify_record(record, store, ctx, *, resolve=None, quoted=overlaps_quotation):
    refs = record.get("source_refs") if isinstance(record, dict) else None
    if not has_source_refs(refs) or record.get("rule_version") != RULE_VERSION:
        return False
    if resolve is None:
        resolve = store.reference_reader(ctx)
    for ref in refs:
        resolved = resolve(ref)
        if resolved is None:
            return False
        source, unit = resolved
        offset = unit["parent_span"][0]
        if quoted(unit["text"], ref["span"][0] - offset, ref["span"][1] - offset):
            return False
        quote = record.get("quote", "")
        if (quote != ref["quote"] or record.get("role") != unit["role"]
                or record.get("speaker") != unit["speaker"] or record.get("recorded_at") != source["recorded_at"]
                or not is_assertion_quote(quote)):
            return False
        doc = None
        signature = record.get("nlp_model")
        if signature:
            if (not isinstance(signature, dict) or not isinstance(signature.get("name"), str)
                    or not isinstance(signature.get("version"), str)):
                return False
            from extraction.deterministic.nlp import load_model, verified_document
            try:
                model = load_model(signature["name"])
            except (ImportError, OSError):
                return False
            if model.meta.get("version") != signature["version"]:
                return False
            doc = verified_document(model, signature, quote)
        assertion = build_assertion(quote, unit["speaker"], unit["role"], SimpleNamespace(**source), doc)
        if (assertion is None or any(record.get(key) != value for key, value in assertion.items())
                or record.get("fact_id") != fact_id(ref, assertion)):
            return False
    return True
