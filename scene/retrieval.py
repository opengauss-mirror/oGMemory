"""Bounded scene-aware ranking over an existing memory result set.

Scene evidence above the query threshold boosts canonical memory hits.
Direct evidence keeps a configurable score and result-slot reserve.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from math import ceil
from typing import Protocol, TypeVar

from core.models import RequestContext, SeedHit, TypedQuery
from retrieval.identity import canonical_uri
from retrieval.owner_scope import expand_owner_space_aliases
from scene.config import SceneConfig

_T = TypeVar("_T")


def select_scene_evidence(
    ranked: Sequence[_T], limit: int, reserve: float | None,
    is_expanded: Callable[[_T], bool],
) -> list[_T]:
    """Apply evidence quotas to an already ranked candidate pool."""
    if reserve is None:
        return list(ranked[:limit])
    if limit <= 0:
        return []
    direct, expanded = [], []
    for item in ranked:
        (expanded if is_expanded(item) else direct).append(item)
    reserved = min(len(direct), ceil(limit * reserve))
    selected = direct[:reserved] + expanded[:limit - reserved]
    selected.extend(direct[reserved:reserved + limit - len(selected)])
    return selected


@dataclass(frozen=True)
class SceneHit:
    scene_id: str
    score: float
    owner_space: str
    member_uris: tuple[str, ...]
    title: str = ""
    description: str = ""
    member_versions: tuple[tuple[str, int], ...] = ()
    query_threshold: float | None = None
    routes: tuple[str, ...] = ()


@dataclass(frozen=True)
class SceneRanking:
    hits: tuple[SeedHit, ...]
    matched_scenes: tuple[SceneHit, ...]


class SceneSearchSource(Protocol):
    """Authorized scene-card lookup used by the retrieval adapter."""

    def search(
        self, query: str, ctx: RequestContext, *, limit: int,
        query_vector: Sequence[float] | None = None,
        typed_query: TypedQuery | None = None,
    ) -> Sequence[SceneHit]: ...


class SceneMemberSource(Protocol):
    """Resolve additional scene members from canonical/indexed memory.

    Implementations use their existing canonical/indexed memory store.  The
    resolver and ``SceneSearchSource`` form independent persistence ports.
    """

    def resolve_members(
        self,
        scenes: Sequence[SceneHit],
        ctx: RequestContext,
        *,
        limit: int,
        query_vector: Sequence[float] | None = None,
        typed_query: TypedQuery | None = None,
    ) -> Sequence[SeedHit]: ...


class SceneEvidenceSearchSource(SceneSearchSource, Protocol):
    def search_with_evidence(
        self, query: str, ctx: RequestContext, *, direct_hits: Sequence[SeedHit],
        limit: int, query_vector: Sequence[float] | None = None,
        typed_query: TypedQuery | None = None,
    ) -> Sequence[SceneHit]: ...


class SceneRanker:
    """Blend bounded scene evidence with reserved direct memory evidence."""

    def __init__(self, config: SceneConfig | None = None) -> None:
        self.config = config or SceneConfig()
        self.config.validate()

    def rank(
        self,
        direct_hits: Iterable[SeedHit],
        scene_hits: Iterable[SceneHit],
        *,
        owner_spaces: set[str] | None = None,
        limit: int | None = None,
    ) -> SceneRanking:
        hits = list(direct_hits)
        if limit is not None and limit <= 0:
            return SceneRanking((), ())
        allowed_spaces = set(expand_owner_space_aliases(list(owner_spaces or ())))
        matched = tuple(
            scene
            for scene in sorted(scene_hits, key=lambda item: (-item.score, item.scene_id))
            if scene.score >= (scene.query_threshold if scene.query_threshold is not None else self.config.query_threshold)
            and (
                not allowed_spaces
                or scene.owner_space in allowed_spaces
            )
        )
        scene_scores: dict[str, float] = {}
        scene_ids: dict[str, list[str]] = {}
        for scene in matched:
            for position, uri in enumerate(scene.member_uris):
                uri = canonical_uri(uri)
                contribution = (
                    self.config.scene_boost
                    * scene.score
                    * (self.config.member_decay**position)
                )
                if contribution > scene_scores.get(uri, 0.0):
                    scene_scores[uri] = contribution
                scene_ids.setdefault(uri, []).append(scene.scene_id)

        ranked: list[SeedHit] = []
        max_direct = max((max(hit.score, 0.0) for hit in hits
                          if not hit.metadata.get("scene_expanded")), default=0.0)
        scale = max_direct or 1.0
        for hit in hits:
            expanded = hit.metadata.get("scene_expanded", False)
            base_score = (
                scale * min(1.0, max(hit.score, 0.0)) * (1 - self.config.direct_evidence_reserve)
                if expanded and max_direct else hit.score
            )
            identity = canonical_uri(hit.uri)
            scene_score = scene_scores.get(identity, 0.0)
            score = base_score + scale * (1 - self.config.direct_evidence_reserve) * scene_score
            metadata = dict(hit.metadata)
            if scene_score:
                metadata["scene_score"] = scene_score
                metadata["scene_ids"] = tuple(scene_ids[identity])
                metadata["scene_match"] = True
            ranked.append(replace(hit, score=score, metadata=metadata))
        ranked.sort(key=lambda item: (-item.score, item.uri))
        if limit is not None:
            reserve = (self.config.direct_evidence_reserve
                       if any("member" in s.routes or "lexical" in s.routes for s in matched) else None)
            ranked = select_scene_evidence(
                ranked, limit, reserve, lambda hit: bool(hit.metadata.get("scene_expanded")))
        return SceneRanking(tuple(ranked), matched)
