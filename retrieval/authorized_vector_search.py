"""Authorized vector search and strict canonical hydration."""

from __future__ import annotations

import math
from dataclasses import dataclass

from core.models import RequestContext, RetrievedBlock, SeedHit
from retrieval.context_reader import ContextReader
from retrieval.owner_scope import validate_owner_filter


@dataclass(frozen=True)
class AuthorizedVectorHit:
    hit: SeedHit
    block: RetrievedBlock
    embedding_model: str
    embedding_dimension: int


@dataclass(frozen=True)
class AuthorizedVectorSearchResult:
    hits: tuple[AuthorizedVectorHit, ...]
    raw_count: int
    rejection_reasons: tuple[str, ...]


class AuthorizedVectorSearch:
    """Own embedding, tenant filters, vector search, and strict hydration."""

    def __init__(self, vector_index, embedder, reader: ContextReader) -> None:
        self._vector_index = vector_index
        self._embedder = embedder
        self._reader = reader

    def search(
        self,
        query: str,
        ctx: RequestContext,
        *,
        owner_space: str | list[str] | None,
        filters: dict,
        top_k: int,
        minimum_similarity: float,
        excluded_uris: set[str] | None = None,
    ) -> AuthorizedVectorSearchResult:
        vectors = self._embedder.embed_texts([query])
        if len(vectors) != 1:
            raise ValueError("authorized query embedding count mismatch")
        return self._search_vector(
            vectors[0], ctx, owner_space=owner_space, filters=filters,
            top_k=top_k, minimum_similarity=minimum_similarity,
            excluded_uris=excluded_uris,
        )

    def search_many(
        self,
        queries: list[str],
        ctx: RequestContext,
        *,
        owner_spaces: list[str | list[str] | None],
        filters: dict,
        top_k: int,
        minimum_similarity: float,
        excluded_uris: set[str] | None = None,
    ) -> list[AuthorizedVectorSearchResult]:
        """Embed a recommendation batch once, then authorize every hit."""
        if len(queries) != len(owner_spaces):
            raise ValueError("query and owner-space counts differ")
        vectors = self._embedder.embed_texts(queries) if queries else []
        if len(vectors) != len(queries):
            raise ValueError("authorized query embedding count mismatch")
        return [
            self._search_vector(
                vector, ctx, owner_space=owner_space, filters=filters,
                top_k=top_k, minimum_similarity=minimum_similarity,
                excluded_uris=excluded_uris,
            )
            for vector, owner_space in zip(vectors, owner_spaces, strict=True)
        ]

    def _search_vector(
        self,
        vector: list[float],
        ctx: RequestContext,
        *,
        owner_space: str | list[str] | None,
        filters: dict,
        top_k: int,
        minimum_similarity: float,
        excluded_uris: set[str] | None,
    ) -> AuthorizedVectorSearchResult:
        owners = validate_owner_filter(owner_space, ctx)
        effective = dict(filters)
        effective.update(account_id=ctx.account_id, owner_space=owners)
        if getattr(ctx, "acl_enforced", False):
            if not ctx.visible_space_ids:
                return AuthorizedVectorSearchResult((), 0, ("no_visible_spaces",))
            effective["space_id"] = list(ctx.visible_space_ids)

        raw = self._vector_index.search_by_vector(vector, effective, top_k)
        excluded = excluded_uris or set()
        accepted: list[AuthorizedVectorHit] = []
        rejected: list[str] = []
        owner_values = {owners} if isinstance(owners, str) else set(owners)
        account_prefix = f"ctx://{ctx.account_id}/"
        for hit in raw:
            if hit.uri in excluded or not math.isfinite(hit.score):
                continue
            if hit.score < minimum_similarity:
                continue
            if not hit.uri.startswith(account_prefix):
                rejected.append("wrong_account")
                continue
            try:
                block = self._reader.read_authorized(hit.uri, ctx)
            except Exception:
                rejected.append("unavailable_or_unauthorized")
                continue
            if block.owner_space not in owner_values:
                rejected.append("unavailable_or_unauthorized")
                continue
            if (
                getattr(ctx, "acl_enforced", False)
                and hit.metadata.get("space_id") not in ctx.visible_space_ids
            ):
                rejected.append("unavailable_or_unauthorized")
                continue
            accepted.append(
                AuthorizedVectorHit(
                    hit=hit,
                    block=block,
                    embedding_model=getattr(
                        self._embedder, "model", type(self._embedder).__name__
                    ),
                    embedding_dimension=len(vector),
                )
            )
        return AuthorizedVectorSearchResult(
            tuple(accepted), len(raw), tuple(rejected)
        )
