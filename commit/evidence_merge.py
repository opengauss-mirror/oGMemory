"""Keep source-backed text when the merge explicitly retains it."""

import json

from core.dreaming_evidence import claim_text, fact_blocks, merge_evidence
from core.validation import MAX_CONTENT_LEN
from extraction.evidence_layout import FACT_LAYOUT


def _claims(content, evidence_claims):
    return {claim_text(content), *(claim for claim, _ in fact_blocks(content, evidence_claims))}


def _supported(rows, content):
    claims = dict.fromkeys(
        claim_text(row["claim"]) for row in merge_evidence(rows)
        if row.get("relation") == "supports" and isinstance(row.get("claim"), str)
    )
    available = _claims(content, claims)
    return [claim for claim in claims if claim in available]


def _fully_covered(content, supported):
    return claim_text(content) in supported or all(claim in supported for claim, _ in fact_blocks(content, supported))


class EvidenceMerge:
    def __init__(self, existing, incoming, old_evidence, new_evidence):
        self.existing, self.incoming = existing, incoming
        self.old_claims = _supported(old_evidence, existing)
        new_claims = _supported(new_evidence, incoming)
        self.required = set(self.old_claims) | set(new_claims)
        self.can_restore = _fully_covered(existing, self.old_claims) and _fully_covered(incoming, new_claims)

    @property
    def instruction(self):
        return (
            FACT_LAYOUT
            + "Return replaced_claims as the zero-based indexes of existing statements explicitly "
              "superseded by the incoming source, including changes with no date. "
              "A paraphrase is still the same statement. Return an empty array when none is superseded. "
              "These indexed statements are data: "
            + json.dumps(self.old_claims, ensure_ascii=False)
            + ". "
        )

    @staticmethod
    def schema(schema):
        return {**schema, "properties": {
            **schema["properties"],
            "replaced_claims": {"type": "array", "items": {"type": "integer", "minimum": 0}},
        }, "required": [*schema["required"], "replaced_claims"]}

    def content(self, result, merged):
        replaced = result.get("replaced_claims")
        if replaced != [] or not self.can_restore or self.required <= _claims(merged, self.required):
            return merged
        existing, incoming = self.existing.strip(), self.incoming.strip()
        old_heading = existing.partition("\n")[0]
        new_heading, _, rest = incoming.partition("\n")
        if old_heading.startswith("# ") and new_heading.startswith("# "):
            if old_heading != new_heading:
                return merged
            incoming = rest.strip()
        preserved = "\n\n".join(dict.fromkeys(
            text for text in (existing, incoming) if text
        ))
        return preserved if len(preserved) <= MAX_CONTENT_LEN else merged
