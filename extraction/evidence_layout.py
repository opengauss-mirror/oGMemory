"""Preserve evidence claim boundaries during existing model reconciliation."""

from core.dreaming_evidence import EVIDENCE_KEY

FACT_LAYOUT = "Put each retained source-backed fact on its own line in content, reusing its exact draft text. "


def evidence_layout(candidates):
    if any(candidate.metadata.get(EVIDENCE_KEY) for candidate in candidates):
        return FACT_LAYOUT
    return ""
