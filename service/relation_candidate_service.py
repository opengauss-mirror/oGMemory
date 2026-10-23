"""Authorized activation retrieval for cross-prompt relation generation."""

from __future__ import annotations

import json
import time

from core.models import CandidateMemory, RequestContext, WritePlan
from extraction.cross_prompt_relations import (
    CrossPromptRelationDiagnostics,
    HistoricalNodeRecommendation,
)
from index.activation_projection import ACTIVATION_VERSION, activation_hit_is_compatible
from retrieval.authorized_vector_search import AuthorizedVectorSearch
from retrieval.owner_scope import candidate_owner_spaces


class RelationCandidateService:
    """Retrieve bounded historical relation candidates through read policy."""

    def __init__(self, authorized_search: AuthorizedVectorSearch) -> None:
        self._authorized_search = authorized_search

    def recommend(
        self,
        candidates: list[CandidateMemory],
        plans: list[WritePlan],
        ctx: RequestContext,
        *,
        per_node_limit: int = 5,
        total_limit: int = 20,
        minimum_similarity: float = 0.40,
        deadline_seconds: float = 15.0,
    ) -> tuple[list[HistoricalNodeRecommendation], CrossPromptRelationDiagnostics]:
        started = time.monotonic()
        selected: dict[str, HistoricalNodeRecommendation] = {}
        current_uris = {plan.target_uri for plan in plans}
        eligible = [
            (candidate, plan)
            for candidate, plan in zip(candidates, plans, strict=True)
            if plan.action in {"create", "merge", "append"}
        ]
        if getattr(ctx, "acl_enforced", False) and not ctx.visible_space_ids:
            return [], CrossPromptRelationDiagnostics(
                len(eligible), 0, 0, 0, 0, 0, ("no_visible_spaces",)
            )
        queries = [
            f"{candidate.category} | "
            f"{candidate.routing_key.replace('_', ' ')} | {candidate.abstract}"
            for candidate, _ in eligible
        ]
        owners = [candidate_owner_spaces(candidate.owner_scope, ctx) for candidate, _ in eligible]
        search_results = self._authorized_search.search_many(
            queries,
            ctx,
            owner_spaces=owners,
            filters={
                "level": 3,
                "record_kind": "update_activation",
                "representation_version": ACTIVATION_VERSION,
            },
            top_k=25,
            minimum_similarity=minimum_similarity,
            excluded_uris=current_uris,
        )
        before_threshold = 0
        rejected: list[str] = []
        for (candidate, _), result in zip(eligible, search_results, strict=True):
            if time.monotonic() - started > deadline_seconds:
                raise TimeoutError("historical recommendation deadline")
            before_threshold += result.raw_count
            rejected.extend(result.rejection_reasons)
            valid: list[HistoricalNodeRecommendation] = []
            for authorized in result.hits:
                hit = authorized.hit
                block = authorized.block
                metadata = dict(block.metadata or {})
                if not activation_hit_is_compatible(
                    hit,
                    block,
                    embedding_model=authorized.embedding_model,
                    embedding_dimension=authorized.embedding_dimension,
                ):
                    rejected.append("incompatible_or_stale_activation")
                    continue
                valid.append(
                    HistoricalNodeRecommendation(
                        node_ref=hit.uri,
                        category=block.category,
                        name=str(metadata.get("routing_key") or hit.uri.rsplit("/", 1)[-1]),
                        abstract=(block.abstract or "")[:500],
                        source_ids=tuple(
                            str(value)
                            for value in metadata.get("provenance_ids", ())
                            if isinstance(value, str) and value.strip()
                        ),
                        score=float(hit.score),
                    )
                )
            for item in sorted(valid, key=lambda value: (-value.score, value.node_ref))[
                :per_node_limit
            ]:
                existing = selected.get(item.node_ref)
                if existing is None or existing.score < item.score:
                    selected[item.node_ref] = item
        ordered: list[HistoricalNodeRecommendation] = []
        budget = 0
        for item in sorted(selected.values(), key=lambda value: (-value.score, value.node_ref)):
            size = len(json.dumps(item.to_model_dict(), ensure_ascii=False).encode("utf-8"))
            if budget + size > 4000:
                continue
            ordered.append(item)
            budget += size
            if len(ordered) == total_limit:
                break
        return ordered, CrossPromptRelationDiagnostics(
            len(eligible),
            before_threshold,
            len(ordered),
            rejected.count("incompatible_or_stale_activation"),
            0,
            0,
            tuple(rejected),
        )
