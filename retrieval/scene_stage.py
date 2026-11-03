"""Optional scene evidence expansion between recall and result assembly."""

from dataclasses import dataclass, replace

from core.models import LeafHit, RequestContext, SeedHit, TypedQuery
from retrieval.identity import memory_identity
from retrieval.owner_scope import allowed_owner_space_filter, expand_owner_space_aliases
from retrieval.result_ranker import _vec_to_leaf
from retrieval.trace import RetrievalTrace, StageTrace, TraceTimer
from scene.retrieval import SceneRanker, SceneSearchSource


@dataclass(frozen=True)
class SceneCandidates:
    hits: list[LeafHit]
    direct_evidence_reserve: float | None = None


class SceneRetrievalStage:
    def __init__(self, source: SceneSearchSource, ranker: SceneRanker):
        self.source = source
        self.ranker = ranker

    def bind(self, ctx, queries, trace):
        """Pin configuration once for all memory subqueries of this request."""
        memory_queries = [query for query in queries if self._is_memory(query)]
        if not memory_queries:
            return None
        bind_request = getattr(self.source, "for_request", None)
        if not callable(bind_request):
            return self
        try:
            source = bind_request(ctx, memory_queries)
            return SceneRetrievalStage(source, SceneRanker(source.config))
        except Exception as exc:
            trace.add_stage(StageTrace(
                "scene_configuration", warnings=[f"scene configuration unavailable: {type(exc).__name__}"]
            ))
            return None

    def apply(
        self, leaves: list[LeafHit], query: TypedQuery, query_vector: list[float],
        ctx: RequestContext, trace: RetrievalTrace,
    ) -> SceneCandidates:
        if not self._is_memory(query):
            return SceneCandidates(leaves)
        result = SceneCandidates(leaves)
        with TraceTimer("scene_ranking") as stage:
            stage.input_count = len(leaves)
            try:
                result = self._expand(leaves, query, query_vector, ctx)
            except Exception as exc:
                stage.warnings.append(f"scene evidence unavailable: {type(exc).__name__}")
            stage.output_count = len(result.hits)
        trace.add_stage(stage)
        return result

    def _expand(self, leaves, query, query_vector, ctx):
        spaces = allowed_owner_space_filter(query, ctx, enforce_visible_scope=True)
        owners = set([spaces] if isinstance(spaces, str) else spaces or ["__inaccessible__"])
        original = {}
        for hit in leaves:
            identity = memory_identity(hit.uri, hit.metadata)
            previous = original.get(identity)
            if previous is None or hit.score > previous.score:
                original[identity] = hit
        direct = [self._to_seed(hit) for hit in original.values()]
        evidence_search = getattr(self.source, "search_with_evidence", None)
        search = evidence_search if callable(evidence_search) else self.source.search
        options = {"direct_hits": tuple(direct)} if callable(evidence_search) else {}
        candidates = search(
            query.text, ctx, limit=self.ranker.config.max_candidate_scenes,
            query_vector=query_vector, typed_query=query, **options,
        )
        matched = self.ranker.rank([], candidates, owner_spaces=owners).matched_scenes
        matched = matched[:self.ranker.config.max_candidate_scenes]
        if not matched:
            return SceneCandidates(leaves)
        known = set(original)
        known_nodes = {uri for uri, _ in known}
        resolve = getattr(self.source, "resolve_members", None)
        if callable(resolve):
            members = resolve(
                matched, ctx, limit=self.ranker.config.max_members_per_scene + len(known),
                query_vector=query_vector, typed_query=query,
            )
            accepted = 0
            member_uris = {memory_identity(uri, {})[0] for scene in matched for uri in scene.member_uris}
            for member in members:
                identity = memory_identity(member.uri, member.metadata)
                if (
                    member.level == 2 and identity not in known and identity[0] in member_uris
                    and (identity[1] or identity[0] not in known_nodes)
                    and member.uri.startswith(f"ctx://{ctx.account_id}/")
                    and set(expand_owner_space_aliases([member.owner_space])) & owners
                    and (not query.categories or member.category in query.categories)
                ):
                    direct.append(replace(member, metadata={**member.metadata, "scene_expanded": True}))
                    known.add(identity)
                    known_nodes.add(identity[0])
                    accepted += 1
                    if accepted >= self.ranker.config.max_members_per_scene:
                        break
        ranked = self.ranker.rank(direct, matched, owner_spaces=owners)
        result = []
        for hit in ranked.hits:
            previous = original.get(memory_identity(hit.uri, hit.metadata))
            result.append(
                replace(previous, score=hit.score, metadata=hit.metadata)
                if previous is not None else _vec_to_leaf(hit)
            )
        reserve = (self.ranker.config.direct_evidence_reserve
                   if any("member" in scene.routes or "lexical" in scene.routes for scene in matched) else None)
        return SceneCandidates(result, reserve)

    @staticmethod
    def _is_memory(query):
        return not query.context_type or query.context_type.upper() == "MEMORY"

    @staticmethod
    def _to_seed(hit: LeafHit) -> SeedHit:
        return SeedHit(
            uri=hit.uri, score=hit.score, level=hit.level, parent_uri=hit.parent_uri,
            context_type="MEMORY",
            category=hit.category, owner_space=hit.owner_space, abstract=hit.abstract,
            has_overview=hit.has_overview, has_content=hit.has_content,
            active_count=hit.active_count, updated_at=hit.updated_at,
            metadata=dict(hit.metadata or {}),
        )
