"""Stage 4 — Result ranker.

Filters to L2 leaf nodes only, deduplicates by URI, sorts by score,
truncates to top_k, and converts into RetrievedBlock list.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Protocol

from core.errors import AccessDeniedError, NodeBrokenError, NodeNotFoundError
from core.models import (
    LeafHit,
    RequestContext,
    RetrievalConfig,
    RetrievedBlock,
    SeedHit,
    SeedResult,
    TypedQuery,
)
from retrieval.identity import memory_identity, memory_unit_key
from scene.retrieval import select_scene_evidence

logger = logging.getLogger(__name__)


_TEMPORAL_KEYWORDS = re.compile(
    r"\b("
    r"when|date|day|before|after|since|until|during|last|next|previous|"
    r"following|weekend|month|year|today|yesterday|tomorrow"
    r")\b",
    re.IGNORECASE,
)
_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


def is_temporal_query(query: str | None) -> bool:
    text = query or ""
    return bool(_TEMPORAL_KEYWORDS.search(text) or _ISO_DATE.search(text))


def _date_tokens(text: str | None) -> set[str]:
    return set(_ISO_DATE.findall(text or ""))


def _temporal_bonus(hit: LeafHit, query_dates: set[str]) -> float:
    metadata = hit.metadata or {}
    when = str(metadata.get("when") or "")
    searchable = " ".join(
        str(value)
        for value in (
            when,
            hit.abstract,
        )
        if value
    )

    bonus = 0.0
    if when:
        bonus += 0.03
    if hit.category == "event":
        bonus += 0.02
    if query_dates:
        hit_dates = _date_tokens(searchable)
        if hit_dates & query_dates:
            bonus += 0.18
        elif hit_dates:
            bonus -= 0.04
    return bonus


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return " ".join(str(item) for item in value if item)
    return str(value)


def _metadata_values(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _contains_token(query: str, value: object) -> bool:
    query_text = query.lower()
    for item in _metadata_values(value):
        phrase = item.lower()
        if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", query_text):
            return True
    return False


def _metadata_bonus(hit: LeafHit, query: str, query_dates: set[str]) -> float:
    metadata = hit.metadata or {}
    bonus = 0.0

    when = metadata.get("when")
    if when:
        bonus += 0.02
        hit_dates = _date_tokens(_as_text(when))
        if query_dates and hit_dates & query_dates:
            bonus += 0.08
        elif query_dates and hit_dates:
            bonus -= 0.04

    if _contains_token(query, metadata.get("who")):
        bonus += 0.03
    if _contains_token(query, metadata.get("where")):
        bonus += 0.03
    if hit.category == "event" and is_temporal_query(query):
        bonus += 0.02

    return max(-0.04, min(0.12, bonus))


@dataclass(frozen=True)
class RankedHit:
    hit: LeafHit
    rank_score: float
    reason: str


class CandidateRanker(Protocol):
    def rank(self, query: TypedQuery, hits: list[LeafHit]) -> list[RankedHit]:
        ...


class SemanticRanker:
    def rank(self, query: TypedQuery, hits: list[LeafHit]) -> list[RankedHit]:
        query_dates = _date_tokens(query.text)
        temporal = is_temporal_query(query.text)

        ranked = [
            RankedHit(
                hit=hit,
                rank_score=hit.score + (_temporal_bonus(hit, query_dates) if temporal else 0.0),
                reason=f"vector score {hit.score:.4f}",
            )
            for hit in hits
        ]
        ranked.sort(key=lambda item: (item.rank_score, item.hit.score), reverse=True)
        return ranked


class MetadataAwareRanker:
    def rank(self, query: TypedQuery, hits: list[LeafHit]) -> list[RankedHit]:
        query_dates = _date_tokens(query.text)
        ranked = []
        for hit in hits:
            bonus = _metadata_bonus(hit, query.text, query_dates)
            ranked.append(
                RankedHit(
                    hit=hit,
                    rank_score=hit.score + bonus,
                    reason=(
                        f"metadata-aware rank {hit.score + bonus:.4f} "
                        f"(vector score {hit.score:.4f})"
                    ),
                )
            )
        ranked.sort(key=lambda item: (item.rank_score, item.hit.score), reverse=True)
        return ranked


def _make_ranker(strategy: str | None) -> CandidateRanker:
    if (strategy or "semantic").lower() == "metadata_aware":
        return MetadataAwareRanker()
    return SemanticRanker()


def _vec_to_leaf(v: SeedHit) -> LeafHit:
    """Convert a SeedHit to a LeafHit.

    This is a shared utility used by ResultRanker and RetrievalPipeline.
    """
    return LeafHit(
        uri=v.uri,
        score=v.score,
        level=v.level,
        parent_uri=v.parent_uri,
        category=v.category,
        owner_space=v.owner_space,
        abstract=v.abstract,
        has_overview=v.has_overview,
        has_content=v.has_content,
        active_count=v.active_count,
        updated_at=v.updated_at,
        metadata=dict(v.metadata or {}),
    )


class ResultRanker:
    def __init__(self, config: RetrievalConfig | None = None, relation_store=None) -> None:
        self.cfg = config or RetrievalConfig()
        self._relation_store = relation_store
        self._ranker = _make_ranker(getattr(self.cfg, "result_ranker_strategy", "semantic"))

    def assemble(
        self,
        typed_query: TypedQuery,
        leaf_hits: list[LeafHit],
        seed_result: SeedResult | None,
        ctx: RequestContext | None = None,
        *,
        fill_content_for_top_k: int = 0,
        context_reader=None,
        direct_evidence_reserve: float | None = None,
    ) -> list[RetrievedBlock]:
        candidates = list(leaf_hits) if leaf_hits else []

        if not candidates and seed_result:
            candidates = self._l2_from_seed(seed_result)

        l2_only = [h for h in candidates if h.level == 2]

        # Keep the highest-scored hit for each node or knowledge unit.
        seen_uris: dict[tuple[str, str], LeafHit] = {}
        for hit in l2_only:
            identity = memory_identity(hit.uri, hit.metadata)
            if identity not in seen_uris or hit.score > seen_uris[identity].score:
                seen_uris[identity] = hit
        l2_only = list(seen_uris.values())

        is_visible = getattr(context_reader, "is_visible", None)
        if ctx is not None and callable(is_visible):
            l2_only = [hit for hit in l2_only if is_visible(hit.uri, ctx)]

        ranked = self._ranker.rank(typed_query, l2_only)
        top = select_scene_evidence(
            ranked, typed_query.top_k, direct_evidence_reserve,
            lambda item: bool(item.hit.metadata.get("scene_expanded")),
        )
        if direct_evidence_reserve is not None:
            top.sort(key=lambda item: (item.rank_score, item.hit.score), reverse=True)

        blocks: list[RetrievedBlock] = []
        for item in top:
            hit = item.hit
            blocks.append(
                RetrievedBlock(
                    uri=hit.uri,
                    level_hit="L2",
                    score=hit.score,
                    ranking_score=item.rank_score,
                    category=hit.category,
                    owner_space=hit.owner_space,
                    abstract=hit.abstract or None,
                    has_overview=hit.has_overview,
                    has_content=hit.has_content,
                    match_reason=item.reason,
                    metadata=dict(hit.metadata or {}),
                )
            )

        # Fill content_excerpt for top-K results (read full content from AGFS)
        unavailable = set()
        if fill_content_for_top_k > 0 and context_reader and ctx:
            top_k_to_fill = min(fill_content_for_top_k, len(blocks))
            for i in range(top_k_to_fill):
                block = blocks[i]
                try:
                    unit_key = memory_unit_key(block.metadata)
                    content_block = context_reader.read(
                        block.uri, ctx, **({"unit_key": unit_key} if unit_key else {}))
                    block.metadata.update(content_block.metadata or {})
                    if content_block.overview:
                        block.overview = content_block.overview
                    if content_block.content_excerpt:
                        block.content_excerpt = content_block.content_excerpt
                except (AccessDeniedError, PermissionError, NodeNotFoundError, NodeBrokenError, FileNotFoundError):
                    unavailable.add(i)
                except Exception as exc:
                    logger.debug("content fill failed for %s: %s", block.uri, exc)

        if unavailable:
            blocks = [block for index, block in enumerate(blocks) if index not in unavailable]

        # Fill relations from relation store
        if self._relation_store and ctx:
            for block in blocks:
                try:
                    block.relations = self._relation_store.get_one_hop(block.uri, ctx, limit=3)
                except Exception as exc:
                    logger.debug("relation fetch failed for %s: %s", block.uri, exc)

        return blocks


    @staticmethod
    def _l2_from_seed(seed: SeedResult) -> list[LeafHit]:
        out: list[LeafHit] = []
        for v in seed.initial_candidates:
            if v.level == 2:
                out.append(_vec_to_leaf(v))
        return out
