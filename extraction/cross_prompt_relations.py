"""Graph-on historical recommendation and cross-prompt relation generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from core.interfaces import LLM
from core.models import CandidateMemory, RelationEdge, RequestContext, WritePlan
from commit.write_receipts import has_stored_target
from index.activation_projection import build_activation_record

def run_integrated_relations(messages, candidates, plans, ctx, *, fs,
                             vector_index, embedder, llm,
                             historical_candidates=None,
                             historical_diagnostics=None):
    """Post-write hook. Lookup failure preserves new-to-new generation only."""
    from dataclasses import asdict, replace

    diagnostics = {"status": "completed", "relation_accepted": 0}
    if len(candidates) != len(plans):
        # The legacy writer omits failed writes. Positional pairing is no longer
        # trustworthy; retain successful memories but abstain from graph linking.
        return [], {**diagnostics, "status": "abstained",
                    "reason": "candidate_write_alignment_failed"}
    stored_candidates, stored_plans = [], []
    for candidate, plan in zip(candidates, plans, strict=True):
        if not has_stored_target(plan):
            continue
        try:
            node = fs.read_node(plan.target_uri, ctx)
            if node is None or build_activation_record(node, {}) is None:
                continue
        except Exception:
            continue
        stored_candidates.append(replace(candidate, abstract=node.abstract,
                                         provenance_ids=list(node.metadata.get("provenance_ids", []))))
        stored_plans.append(plan)
    if not stored_candidates:
        return [], {**diagnostics, "status": "no_stored_candidates"}
    historical = list(historical_candidates or [])
    if historical_diagnostics is not None:
        diagnostics.update(asdict(historical_diagnostics))
        diagnostics["selected_scores"] = [
            {"node_ref": item.node_ref, "score": item.score} for item in historical
        ]
    elif historical_candidates is None:
        # Production callers must compose the authorized retrieval service.
        # A second private executor/vector/ACL implementation is deliberately
        # not retained as a fallback.
        diagnostics["historical_recommendation_failure"] = "not_composed"
    try:
        edges, rejected, count = generate_cross_prompt_relations(
            messages, stored_candidates, stored_plans, historical, llm=llm)
        diagnostics.update(relation_accepted=len(edges), relation_proposals=count,
                           relation_rejections=list(rejected))
        return edges, diagnostics
    except Exception as exc:
        return [], {**diagnostics, "status": "abstained", "model_error": type(exc).__name__}


@dataclass(frozen=True)
class HistoricalNodeRecommendation:
    node_ref: str
    category: str
    name: str
    abstract: str
    source_ids: tuple[str, ...]
    score: float

    def to_model_dict(self) -> dict[str, Any]:
        """Return the score-free, activation-free model representation."""
        return {
            "node_ref": self.node_ref,
            "category": self.category,
            "name": self.name,
            "abstract": self.abstract,
            "source_ids": list(self.source_ids),
        }


@dataclass(frozen=True)
class CrossPromptRelationDiagnostics:
    eligible_new_node_count: int
    historical_candidates_before_threshold: int
    historical_candidates_selected: int
    missing_activation_count: int
    relation_proposals: int
    relation_accepted: int
    rejection_reasons: tuple[str, ...] = ()


def build_cross_prompt_relation_prompt(
    messages: list[dict],
    candidates: list[CandidateMemory],
    plans: list[WritePlan],
    historical: list[HistoricalNodeRecommendation],
) -> str:
    """Build the question-blind mixed new/historical relation prompt."""
    current_turns = [
        {
            "turn_id": str(message.get("id", index)),
            "speaker": str(message.get("speaker") or message.get("role", "user")),
            "text": str(message.get("content", "")),
        }
        for index, message in enumerate(messages)
        if message.get("role") != "assistant" and str(message.get("content", "")).strip()
    ]
    new_nodes = [
        {
            "node_ref": plan.target_uri,
            "category": candidate.category,
            "name": candidate.routing_key,
            "abstract": candidate.abstract,
            "source_ids": list(candidate.provenance_ids),
        }
        for candidate, plan in zip(candidates, plans, strict=True)
        if has_stored_target(plan)
    ]
    turns_json = json.dumps(
        current_turns, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    new_nodes_json = json.dumps(
        new_nodes, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    historical_nodes_json = json.dumps(
        [item.to_model_dict() for item in historical],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return """You are a relation extraction assistant. Extract precise directed relationships between the supplied memory nodes.

SOURCE DISCIPLINE:
- new_nodes are grounded in the current non-assistant turns.
- recommended_historical_nodes are stored source-grounded facts selected only as possible comparison candidates.
- Recommendation, similarity, shared words, and co-occurrence are not evidence of a relationship.
- Use only the supplied turns and node fields. Never invent missing dialogue or facts.

GRAPH RULES:
- Consider new-to-new and new-to-historical pairs.
- At least one endpoint of every relation must be from new_nodes.
- Never create historical-to-historical relations.
- Copy node_ref values exactly. Endpoints must be different supplied nodes.
- Use a short, specific, open-text directed relation label.
- A relation must make semantic sense for both endpoint categories and make the target useful one-hop context from the source.
- For a cross-prompt relation, evidence_ids must include supporting source IDs from both the new and historical endpoints.
- If evidence is insufficient, omit the relation. An empty list is valid.

Do not create nodes, merge identities, resolve conflicts, change lifecycle state, persist data, or perform retrieval.
Return only JSON:
{"relations":[{"source_node_ref":"exact supplied node_ref","relation":"short directed open-text label","target_node_ref":"different exact supplied node_ref","evidence_ids":["supporting supplied source ID"]}]}

ORIGINAL NON-ASSISTANT MESSAGES:
These are the original messages that produced the newly extracted nodes.
""" + turns_json + """

NEWLY EXTRACTED MEMORY NODES:
These nodes were extracted from the original messages above.
""" + new_nodes_json + """

POTENTIALLY RELEVANT EXISTING MEMORY NODES:
These existing nodes were retrieved because they may concern similar subjects. Retrieval does not prove that they are related to the newly extracted nodes. Decide whether the supplied evidence supports a directed relationship. If it does not, create no edge.
""" + historical_nodes_json


def relation_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["relations"],
        "properties": {
            "relations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "source_node_ref",
                        "relation",
                        "target_node_ref",
                        "evidence_ids",
                    ],
                    "properties": {
                        "source_node_ref": {"type": "string"},
                        "relation": {"type": "string", "maxLength": 256},
                        "target_node_ref": {"type": "string"},
                        "evidence_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
            }
        },
    }


def generate_cross_prompt_relations(
    messages: list[dict],
    candidates: list[CandidateMemory],
    plans: list[WritePlan],
    historical: list[HistoricalNodeRecommendation],
    *,
    llm: LLM,
) -> tuple[list[RelationEdge], tuple[str, ...], int]:
    """Generate and strictly validate mixed relations; malformed siblings abstain."""
    prompt = build_cross_prompt_relation_prompt(messages, candidates, plans, historical)
    data = llm.complete_json(prompt, relation_response_schema())
    items = data.get("relations", []) if isinstance(data, dict) else []
    if not isinstance(items, list):
        return [], ("relations_not_array",), 0

    new_sources_by_ref = {
        plan.target_uri: {
            str(source_id)
            for source_id in candidate.provenance_ids
            if str(source_id).strip()
        }
        for candidate, plan in zip(candidates, plans, strict=True)
        if has_stored_target(plan)
    }
    new_refs = set(new_sources_by_ref)
    historical_by_ref = {item.node_ref: item for item in historical}
    allowed_refs = new_refs | set(historical_by_ref)
    historical_sources_by_ref = {
        item.node_ref: set(item.source_ids) for item in historical
    }
    allowed_ids = {
        source_id
        for sources in (*new_sources_by_ref.values(), *historical_sources_by_ref.values())
        for source_id in sources
    }
    accepted: list[RelationEdge] = []
    rejected: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    cross_counts: dict[str, int] = {}
    cross_total = 0
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            rejected.append(f"{index}:not_object")
            continue
        source_ref = item.get("source_node_ref")
        target_ref = item.get("target_node_ref")
        relation = item.get("relation")
        evidence = item.get("evidence_ids")
        if (
            not isinstance(source_ref, str)
            or not isinstance(target_ref, str)
            or source_ref not in allowed_refs
            or target_ref not in allowed_refs
            or source_ref == target_ref
        ):
            rejected.append(f"{index}:invalid_endpoint")
            continue
        if source_ref not in new_refs and target_ref not in new_refs:
            rejected.append(f"{index}:historical_to_historical")
            continue
        if not isinstance(relation, str) or not relation.strip() or len(relation.strip()) > 256:
            rejected.append(f"{index}:invalid_relation")
            continue
        if (
            not isinstance(evidence, list)
            or not evidence
            or not all(isinstance(value, str) and value in allowed_ids for value in evidence)
        ):
            rejected.append(f"{index}:invalid_evidence")
            continue
        is_cross_prompt = source_ref in historical_by_ref or target_ref in historical_by_ref
        if is_cross_prompt:
            new_ref = source_ref if source_ref in new_refs else target_ref
            historical_ref = (
                source_ref if source_ref in historical_by_ref else target_ref
            )
            evidence_ids = set(evidence)
            if not (
                evidence_ids & new_sources_by_ref[new_ref]
                and evidence_ids & historical_sources_by_ref[historical_ref]
            ):
                rejected.append(f"{index}:incomplete_cross_prompt_evidence")
                continue
        key = (source_ref, relation.strip(), target_ref)
        if key in seen:
            rejected.append(f"{index}:duplicate")
            continue
        if is_cross_prompt:
            if cross_total >= 24 or cross_counts.get(new_ref, 0) >= 3:
                rejected.append(f"{index}:cross_prompt_cap")
                continue
            cross_total += 1
            cross_counts[new_ref] = cross_counts.get(new_ref, 0) + 1
        seen.add(key)
        accepted.append(
            RelationEdge(
                from_uri=source_ref,
                to_uri=target_ref,
                relation_type=relation.strip(),
                weight=1.0,
                reason="evidence:" + ",".join(evidence),
            )
        )
    return accepted, tuple(rejected), len(items)
