"""Canonical memory and indexed-vector adapters for scene analysis and search."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from copy import copy, deepcopy
from dataclasses import dataclass, field, replace
from itertools import chain, islice
from urllib.parse import urlsplit

from core.errors import AccessDeniedError, NodeBrokenError, NodeNotFoundError
from core.interfaces import ContextFS, Embedder, VectorIndex
from core.memory_names import memory_directory
from core.models import ContextNode, RequestContext, SeedHit, TypedQuery
from core.provenance_resolver import ProvenanceResolver
from retrieval.context_reader import canonical_agent_node
from retrieval.identity import canonical_uri
from retrieval.owner_scope import allowed_owner_space_filter, expand_owner_space_aliases
from retrieval.owner_scope import canonical_owner_space as _canonical_space
from scene.candidate_index import SceneCandidateIndex
from scene.config import SceneConfig, SceneConfigProvider
from scene.models import SceneSnapshot
from scene.retrieval import SceneHit
from scene.store import SceneDecisionInput, SceneEvidenceRepository


@dataclass(frozen=True)
class _EvidenceSnapshot(SceneSnapshot):
    evidence_fingerprints: tuple[tuple[str, str], ...] = field(default=(), compare=False)


def _scope_context(account_id: str, owner_space: str) -> RequestContext:
    kind, separator, identifier = _canonical_space(owner_space).partition(":")
    if not account_id or not separator or not identifier or kind not in {"user", "agent"}:
        raise ValueError("scene scope requires an account and user or agent owner")
    return RequestContext(
        account_id=account_id, user_id=identifier if kind == "user" else "",
        agent_id=identifier if kind == "agent" else "", session_id="", trace_id="scene",
        visible_owner_spaces=(f"{kind}:{identifier}",),
    )


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    if any(not math.isfinite(value) for value in (*left, *right)):
        return 0.0
    denominator = math.sqrt(sum(value * value for value in left) * sum(value * value for value in right))
    return sum(a * b for a, b in zip(left, right)) / denominator if denominator else 0.0


def _fingerprint(node: ContextNode) -> str:
    data = (node.uri, node.owner_space, node.category, node.abstract, node.overview,
            node.content, node.metadata)
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


def _json_text_prefix(text: str, budget: int) -> str:
    """Longest prefix fitting a JSON string-content budget, excluding quotes."""
    low, high = 0, min(len(text), max(0, budget))
    while low < high:
        middle = (low + high + 1) // 2
        if len(json.dumps(text[:middle], ensure_ascii=False)) - 2 <= budget:
            low = middle
        else:
            high = middle - 1
    return text[:low]


class CanonicalSceneAdapter:
    """Resolve scene evidence through the same scoped canonical store as memory reads."""

    def __init__(
        self, repository: SceneEvidenceRepository, context_fs: ContextFS,
        vector_index: VectorIndex, embedder: Embedder,
        *, config_provider: SceneConfigProvider | None = None,
        config: SceneConfig | None = None,
        context_factory: Callable[[str, str], RequestContext] | None = None,
        max_input_nodes: int = 128,
        archive_store=None,
        candidate_index: SceneCandidateIndex | None = None,
    ) -> None:
        if max_input_nodes <= 0:
            raise ValueError("max_input_nodes must be positive")
        self.repository = repository
        self.fs = context_fs
        self.vector_index = vector_index
        self.embedder = embedder
        self.config_provider = config_provider
        self.config = config or SceneConfig()
        self.config.validate()
        self.context_factory = context_factory or _scope_context
        self.max_input_nodes = max_input_nodes
        self.archive_store = archive_store
        self.candidate_index = candidate_index
        self._request_configs: dict[tuple[str, str], SceneConfig] | None = None

    def _config(self, account_id: str, owner_space: str) -> SceneConfig:
        if self._request_configs is not None:
            return self._request_configs[(account_id, _canonical_space(owner_space))]
        if self.config_provider is not None:
            return self.config_provider.get_effective(account_id, owner_space).config
        return self.config

    def for_request(self, ctx: RequestContext, queries: Sequence[TypedQuery]):
        """Pin effective configurations for one retrieval request."""
        queries = queries or (TypedQuery("", "MEMORY", []),)
        spaces = {space for query in queries for space in self._spaces(query, ctx)}
        primary = ctx.user_space_name() if ctx.user_id else ctx.agent_space_name()
        spaces.add(primary)
        bound = copy(self)
        bound._request_configs = {
            (ctx.account_id, space): self._config(ctx.account_id, space) for space in sorted(spaces)
        }
        bound.config = bound._request_configs[(ctx.account_id, primary)]
        return bound

    def _read(self, uri: str, ctx: RequestContext, owner_spaces: Sequence[str],
              read_node=None) -> ContextNode | None:
        parsed = urlsplit(uri)
        allowed = {_canonical_space(space) for space in owner_spaces}
        if parsed.scheme != "ctx" or parsed.netloc != ctx.account_id:
            return None
        parts = parsed.path.strip("/").split("/")
        if len(parts) < 4 or parts[0] not in {"users", "agents"} or parts[2] != "memories":
            return None
        uri_owner = f"{parts[0][:-1]}:{parts[1]}"
        if uri_owner not in allowed:
            return None
        try:
            node = (read_node or self.fs.read_node)(uri, ctx)
        except (NodeNotFoundError, NodeBrokenError, AccessDeniedError):
            return None
        if (node.uri != uri or node.context_type != "MEMORY"
                or node.metadata.get("node_type") == "directory"
                or _canonical_space(node.owner_space) != uri_owner
                or node.metadata.get("status", "ACTIVE") != "ACTIVE"):
            return None
        return deepcopy(canonical_agent_node(node))

    def _vectors(self, nodes: Sequence[ContextNode], ctx: RequestContext) -> dict[str, list[float]]:
        if not nodes:
            return {}
        filters = {
            "account_id": ctx.account_id, "context_type": "MEMORY",
            "owner_space": expand_owner_space_aliases(sorted({node.owner_space for node in nodes})),
        }
        vectors = self.vector_index.get_embeddings(
            [node.uri for node in nodes], filters, level=0,
            expected_texts={node.uri: node.abstract for node in nodes},
        )
        return {node.uri: list(vectors[node.uri]) for node in nodes if node.uri in vectors}

    def load_nodes(self, account_id: str, owner_space: str, node_uris: Sequence[str]):
        nodes, vectors, _ = self._load_nodes(account_id, owner_space, node_uris)
        return nodes, vectors

    def _load_nodes(self, account_id, owner_space, node_uris):
        ctx = self.context_factory(account_id, owner_space)
        if ctx.account_id != account_id:
            raise ValueError("scene context account mismatch")
        requested = list(dict.fromkeys(node_uris))[:self.max_input_nodes]
        nodes = [node for uri in requested
                 if (node := self._read(uri, ctx, [owner_space])) is not None]
        vectors = self._vectors(nodes, ctx)
        if any(node.uri not in vectors for node in nodes):
            raise RuntimeError("scene evidence is awaiting vector indexing")
        available = {node.uri for node in nodes}
        return nodes, vectors, tuple(uri for uri in requested if uri not in available)

    def load_input(self, account_id: str, owner_space: str, *, node_uris=None) -> SceneDecisionInput:
        config = self._config(account_id, owner_space)
        offered, vectors, unavailable_uris = self._load_nodes(account_id, owner_space, node_uris or ())
        budget = max(0, (config.analysis_input_budget_tokens - config.candidate_card_budget_tokens) * 4 - 1024)
        accepted = []
        remaining = budget
        for node in offered:
            cost = self._card_cost(node)
            if cost <= remaining:
                accepted.append(node)
                remaining -= cost
        offered = accepted
        vectors = {node.uri: vectors[node.uri] for node in offered}
        ctx = self.context_factory(account_id, owner_space)
        for _ in range(config.retry_limit):
            snapshot = self.repository.load_snapshot(account_id, owner_space)
            active = [record for record in snapshot.scenes if record.status == "ACTIVE"]
            offered_uris = set(node_uris or ())
            active.sort(key=lambda record: (
                -bool(offered_uris.intersection(member.context_uri for member in record.cluster.members)),
                -max((_cosine(vector, record.cluster.centroid) for vector in vectors.values()), default=0.0),
                record.scene_id,
            ))
            selected = {node.uri: node for node in offered}
            used = sum(self._card_cost(node) for node in offered)
            unavailable = set(unavailable_uris)
            for record in active[:config.max_candidate_scenes]:
                members = {member.context_uri for member in record.cluster.members}
                if len(members | selected.keys()) > self.max_input_nodes:
                    continue
                resolved = [self._read(uri, ctx, [owner_space]) for uri in sorted(members)]
                if any(node is None for node in resolved):
                    unavailable.update(uri for uri, node in zip(sorted(members), resolved) if node is None)
                    continue
                nodes = [node for node in resolved if node is not None]
                cost = sum(self._card_cost(node) for node in nodes if node.uri not in selected)
                if used + cost > budget:
                    continue
                indexed = self._vectors(nodes, ctx)
                if len(indexed) != len(nodes):
                    raise RuntimeError("scene evidence is awaiting vector indexing")
                selected.update((node.uri, node) for node in nodes)
                vectors.update(indexed)
                used += cost
            stale = {record.scene_id for record in active
                     if any(member.context_uri in unavailable for member in record.cluster.members)}
            if stale:
                replacement = replace(snapshot, revision=snapshot.revision + 1, scenes=tuple(
                    replace(record, status="ARCHIVED", version=record.version + 1)
                    if record.scene_id in stale else record for record in snapshot.scenes
                ))
                if not self.repository.compare_and_swap_checked(
                    snapshot, replacement,
                    lambda reader: all(self._read(uri, ctx, [owner_space], reader) is None for uri in unavailable),
                ):
                    continue
                snapshot = replacement
            nodes = tuple(selected.values())
            guarded = _EvidenceSnapshot(
                snapshot.account_id, snapshot.owner_space, snapshot.revision, snapshot.scenes,
                tuple((node.uri, _fingerprint(node)) for node in nodes),
            )
            return SceneDecisionInput(guarded, nodes, {node.uri: vectors[node.uri] for node in nodes},
                                      unavailable_uris=unavailable_uris)
        raise RuntimeError("scene source changed during bounded snapshot read")

    def compare_and_swap(
        self, expected: SceneSnapshot, replacement: SceneSnapshot,
        *, evidence_nodes: Sequence[ContextNode] = (),
        source_fingerprints: Sequence[tuple[str, str]] = (),
    ) -> bool:
        if not isinstance(expected, _EvidenceSnapshot):
            raise ValueError("scene publication requires a loaded evidence snapshot")
        ctx = self.context_factory(expected.account_id, expected.owner_space)
        fingerprints = dict(expected.evidence_fingerprints)
        evidence = ((node.uri, _fingerprint(node)) for node in evidence_nodes)
        for uri, fingerprint in chain(evidence, source_fingerprints):
            if fingerprints.setdefault(uri, fingerprint) != fingerprint:
                return False

        def check(reader):
            return all(
                (node := self._read(uri, ctx, [expected.owner_space], reader)) is not None
                and _fingerprint(node) == fingerprint
                for uri, fingerprint in fingerprints.items()
            )
        return self.repository.compare_and_swap_checked(expected, replacement, check)

    @staticmethod
    def _spaces(query: TypedQuery, ctx: RequestContext) -> list[str]:
        spaces = allowed_owner_space_filter(query, ctx, enforce_visible_scope=True)
        return sorted({_canonical_space(space) for space in ([spaces] if isinstance(spaces, str) else spaces or [])
                       if space != "__inaccessible__"})

    def search_with_evidence(self, query, ctx, *, direct_hits, **kwargs):
        return self.search(query, ctx, direct_hits=direct_hits, **kwargs)

    def search(self, query: str, ctx: RequestContext, *, limit: int, query_vector=None,
               typed_query: TypedQuery | None = None, direct_hits=()) -> list[SceneHit]:
        typed_query = typed_query or TypedQuery(query, "MEMORY", [])
        if typed_query.context_type != "MEMORY":
            return []
        spaces = self._spaces(typed_query, ctx)
        if not spaces or limit <= 0:
            return []
        vector = query_vector if query_vector is not None else self.embedder.embed_texts([query])[0]
        hits = []
        for space in spaces:
            config = self._config(ctx.account_id, space)
            snapshot = self.repository.load_snapshot(ctx.account_id, space)
            active = [record for record in snapshot.scenes if record.status == "ACTIVE"]
            lexical = (self.candidate_index.search(query, snapshot, config.max_active_scenes_per_space)
                       if config.lexical_enabled and self.candidate_index is not None else [])
            lexical_ranks = {scene_id: rank for rank, scene_id in enumerate(lexical, 1)}
            anchors = {canonical_uri(hit.uri): rank for rank, hit in
                       enumerate(sorted(direct_hits, key=lambda h: (-h.score, h.uri))[:config.neighbor_k], 1)
                       if _canonical_space(hit.owner_space) == space}
            for record in active[:config.max_active_scenes_per_space]:
                score = _cosine(vector, record.cluster.centroid)
                member_ranks = [anchors[canonical_uri(member.context_uri)]
                                for member in record.cluster.members
                                if canonical_uri(member.context_uri) in anchors]
                routes = []
                if score >= config.query_threshold:
                    routes.append("semantic")
                if record.scene_id in lexical_ranks:
                    routes.append("lexical")
                if member_ranks:
                    routes.append("member")
                if not routes:
                    continue
                if "lexical" in routes or "member" in routes:
                    ranks = ([lexical_ranks[record.scene_id]] if "lexical" in routes else [])
                    if member_ranks:
                        ranks.append(min(member_ranks))
                    score = max(score, sum(1.0 / rank for rank in ranks) / len(ranks))
                members = sorted(record.cluster.members, key=lambda item: (-item.weight, item.context_uri))
                hits.append(SceneHit(record.scene_id, score, space,
                                     tuple(member.context_uri for member in members),
                                     record.cluster.label, record.cluster.summary,
                                     tuple((member.context_uri, member.node_version) for member in members),
                                     config.query_threshold if routes == ["semantic"] else 0.0,
                                     tuple(routes)))
        matching = []
        for hit in sorted(hits, key=lambda hit: (-hit.score, hit.scene_id)):
            nodes = list(self._member_nodes([hit], ctx, typed_query))
            vectors = self._vectors(nodes, ctx)
            if not vectors:
                continue
            matching.append(replace(
                hit, member_uris=tuple(node.uri for node in nodes if node.uri in vectors),
                member_versions=tuple((uri, version) for uri, version in hit.member_versions if uri in vectors),
            ))
            if len(matching) == limit:
                break
        return matching

    @staticmethod
    def _matches(node: ContextNode, query: TypedQuery) -> bool:
        if node.context_type != query.context_type:
            return False
        if query.categories and node.category not in query.categories:
            return False
        if query.memory_directories is not None and memory_directory(node.uri) not in query.memory_directories:
            return False
        if query.target_uri and node.uri != query.target_uri and not node.uri.startswith(query.target_uri.rstrip("/") + "/"):
            return False
        for bound, relation in (("time_start", "before"), ("time_end", "after")):
            value = getattr(query, bound, None)
            timestamp = node.metadata.get("event_time") or node.metadata.get("updated_at")
            if value and (not timestamp or (timestamp < value if relation == "before" else timestamp > value)):
                return False
        return True

    def _member_nodes(self, scenes, ctx, query):
        spaces = self._spaces(query, ctx)
        seen = set()
        ordered = [(scene, dict(scene.member_versions)) for scene in
                   sorted(scenes, key=lambda scene: (-scene.score, scene.scene_id))]
        for position in range(max((len(scene.member_uris) for scene, _ in ordered), default=0)):
            for scene, versions in ordered:
                if _canonical_space(scene.owner_space) not in spaces or position >= len(scene.member_uris):
                    continue
                uri = scene.member_uris[position]
                if uri in seen:
                    continue
                node = self._read(uri, ctx, [scene.owner_space])
                if node is None or not self._matches(node, query):
                    continue
                expected_version = versions.get(uri, 0)
                version = node.metadata.get("version", node.metadata.get("node_version", 0))
                if expected_version and expected_version != version:
                    continue
                seen.add(uri)
                yield node

    def resolve_members(self, scenes, ctx: RequestContext, *, limit: int, query_vector=None,
                        typed_query: TypedQuery | None = None) -> list[SeedHit]:
        candidates = self._member_nodes(scenes, ctx, typed_query or TypedQuery("", "MEMORY", []))
        hits = []
        while len(hits) < limit:
            nodes = list(islice(candidates, limit - len(hits)))
            if not nodes:
                break
            vectors = self._vectors(nodes, ctx)
            for node in nodes:
                if node.uri in vectors:
                    score = max(0.0, _cosine(query_vector or (), vectors[node.uri]))
                    hits.append(SeedHit(
                        node.uri, score, level=2, parent_uri=node.parent_uri, context_type="MEMORY",
                        category=node.category, owner_space=node.owner_space, abstract=node.abstract,
                        has_overview=bool(node.overview), has_content=bool(node.content),
                        updated_at=node.metadata.get("updated_at"),
                        metadata={**node.metadata, "scene_expanded": True, "scene_member_similarity": score},
                    ))
        return sorted(hits, key=lambda hit: (-hit.score, hit.uri))

    def search_related_nodes(self, account_id: str, owner_space: str, query: str, limit: int):
        config = self._config(account_id, owner_space)
        if limit <= 0:
            return []
        vector = self.embedder.embed_texts([query])[0]
        hits = self.vector_index.search_by_vector(vector, {
            "account_id": account_id, "owner_space": expand_owner_space_aliases([owner_space]),
            "context_type": "MEMORY", "level": 0,
        }, min(limit, config.neighbor_k))
        nodes, _ = self.load_nodes(account_id, owner_space, [hit.uri for hit in hits])
        return [{**self.node_card(node), "node_fingerprint": _fingerprint(node)} for node in nodes]

    def search_scene_cards(self, account_id: str, owner_space: str, query: str, limit: int):
        ctx = self.context_factory(account_id, owner_space)
        hits = self.search(query, ctx, limit=min(limit, self._config(account_id, owner_space).max_candidate_scenes),
                           typed_query=TypedQuery(query, "MEMORY", [], owner_space=owner_space))
        return [{"scene_id": hit.scene_id, "title": hit.title, "description": hit.description,
                 "member_uris": list(hit.member_uris), "score": hit.score} for hit in hits]

    def read_source_context(self, account_id: str, owner_space: str, source_refs: Sequence[str], limit: int):
        ctx = self.context_factory(account_id, owner_space)
        config = self._config(account_id, owner_space)
        references = list(dict.fromkeys(source_refs))[:max(0, min(limit, config.neighbor_k))]
        budget = config.candidate_card_budget_tokens * 4
        cards = []
        for reference in references:
            source_id, message_ids, source_type = reference, [], "memory"
            if reference.startswith("prov:"):
                try:
                    parsed = ProvenanceResolver.parse_id(reference)
                except ValueError:
                    continue
                source_id, source_type = parsed["source_id"], parsed["source_type"]
                message_ids = parsed["detail"] if source_type == "archive" else []
            parts = urlsplit(source_id)
            path = parts.path.strip("/").split("/")
            archive_path = parts.scheme == "ctx" and len(path) == 4 and path[0] == "sessions" and path[2] == "history"
            if source_type == "archive" or archive_path:
                card = self._archive_card(source_id, message_ids, reference, ctx, owner_space, budget)
            elif source_type == "memory":
                node = self._read(source_id, ctx, [owner_space])
                card = self._source_node_card(node, reference, budget) if node is not None else None
            else:
                card = None
            if card is not None:
                cost = len(json.dumps(card, ensure_ascii=False))
                if cost > budget:
                    break
                cards.append(card)
                budget -= cost
        return cards

    @staticmethod
    def _source_node_card(node, reference, budget):
        card = {"uri": node.uri, "owner_space": node.owner_space, "category": node.category,
                "source_ref": reference, "source_refs": CanonicalSceneAdapter.node_card(node)["source_refs"],
                "node_fingerprint": _fingerprint(node), "text": ""}
        available = budget - len(json.dumps(card, ensure_ascii=False))
        if available <= 0:
            return None
        text = _json_text_prefix(node.content or node.overview or node.abstract, available)
        card["text"] = text
        return card if text else None

    def _archive_card(self, source_id, message_ids, reference, ctx, owner_space, budget):
        if budget <= 0:
            return None
        parts = urlsplit(source_id)
        if parts.scheme:
            node = self._read(source_id, ctx, [owner_space])
            if node is not None:
                return self._source_node_card(node, reference, budget)
            path = parts.path.strip("/").split("/")
            if parts.scheme != "ctx" or parts.netloc != ctx.account_id or len(path) != 4 or path[0] != "sessions" or path[2] != "history":
                return None
            session_id, archive_id = path[1], path[3]
            archive = self.archive_store.read_archive(session_id, archive_id, ctx) if self.archive_store else None
            if archive is None:
                from session.archive_store import SessionArchiveStore

                archive = SessionArchiveStore(self.fs).read_archive(session_id, archive_id, ctx)
            if archive is not None and archive.session_id != session_id:
                return None
        else:
            archive_id = source_id
            from session.archive_store import SessionArchiveStore

            lookup = getattr(self.archive_store or SessionArchiveStore(self.fs), "read_archive_by_id", None)
            archive = lookup(archive_id, ctx) if lookup else None
        if archive is None or archive.archive_id != archive_id:
            return None
        meta = archive.metadata
        if not (ctx.user_id or ctx.agent_id) or meta.get("owner_user_id", "") != ctx.user_id:
            return None
        if ctx.agent_id and meta.get("owner_agent_id", "") != ctx.agent_id:
            return None
        card = {"uri": f"ctx://{ctx.account_id}/sessions/{archive.session_id}/history/{archive_id}",
                "owner_space": owner_space, "category": "archive", "source_ref": reference,
                "message_ids": [], "text": ""}
        selected = set(message_ids)
        for message in archive.messages:
            if not isinstance(message, dict) or (selected and message.get("id") not in selected):
                continue
            message_id = str(message.get("id", ""))
            if message_id:
                card["message_ids"].append(message_id)
            available = budget - len(json.dumps(card, ensure_ascii=False))
            if available <= 0:
                if message_id:
                    card["message_ids"].pop()
                break
            line = f"[{message_id}] {message.get('created_at', '')} {message.get('role', '')}: {message.get('content', '')}\n"
            excerpt = _json_text_prefix(line, available)
            if not excerpt:
                if message_id:
                    card["message_ids"].pop()
                break
            card["text"] += excerpt
            if len(excerpt) < len(line):
                break
        return card if card["text"] else None

    @staticmethod
    def _card_cost(node: ContextNode) -> int:
        return len(json.dumps(CanonicalSceneAdapter.node_card(node), ensure_ascii=False)) + len(node.uri) + 32

    @staticmethod
    def node_card(node: ContextNode) -> dict:
        source_refs = []
        for key in ("provenance_ids", "source_refs", "source_ref"):
            values = node.metadata.get(key, [])
            if isinstance(values, str):
                source_refs.append(values)
            elif isinstance(values, (list, tuple)):
                source_refs.extend(str(value) for value in values)
        return {"uri": node.uri, "owner_space": node.owner_space, "category": node.category,
                "text": "\n".join((node.abstract, node.overview, node.content))[:800],
                "source_refs": list(dict.fromkeys(source_refs))[:16]}
