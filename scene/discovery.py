"""Generate scene candidates from vector similarity, shared sources, and entity mentions."""

from __future__ import annotations

import hashlib
import math
import re
from array import array
from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass
from heapq import nlargest

from core.models import ContextNode
from core.provenance_resolver import ProvenanceResolver
from scene.models import SceneCandidate, SceneMember

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,8}")
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
_STOP = {
    "the", "and", "for", "that", "with", "this", "from", "was", "were",
    "are", "have", "has", "about", "用户", "记忆", "相关", "进行", "一个",
}


@dataclass(frozen=True)
class CandidateConfig:
    """Bounds for one call; max_owner_nodes applies to each owner's embedded nodes."""

    neighbor_k: int = 8
    similarity_threshold: float = 0.72
    strong_pair_threshold: float = 0.86
    min_candidate_size: int = 3
    max_candidate_size: int = 12
    min_quality: float = 0.78
    entity_candidate_max_chunks: int = 32
    max_owner_nodes: int = 512

    def validate(self) -> None:
        """Require positive integer bounds and finite thresholds in their score ranges."""
        for name in (
            "neighbor_k",
            "min_candidate_size",
            "max_candidate_size",
            "entity_candidate_max_chunks",
            "max_owner_nodes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_candidate_size > self.max_candidate_size:
            raise ValueError("min_candidate_size must be <= max_candidate_size")
        for name, lower in (
            ("similarity_threshold", -1.0),
            ("strong_pair_threshold", -1.0),
            ("min_quality", 0.0),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not lower <= value <= 1.0
            ):
                raise ValueError(f"{name} must be a finite number in [{lower}, 1.0]")


def _normalize(vector: list[float]) -> tuple[float, ...]:
    norm = math.sqrt(sum(value * value for value in vector))
    if not norm:
        return tuple(0.0 for _ in vector)
    return tuple(value / norm for value in vector)


def _validated_vectors(
    nodes: list[ContextNode],
    embeddings: dict[str, list[float]],
) -> list[tuple[float, ...]]:
    expected_dimension = len(embeddings[nodes[0].uri])
    vectors: list[tuple[float, ...]] = []
    for node in nodes:
        vector = embeddings[node.uri]
        dimension = len(vector)
        if dimension != expected_dimension:
            raise ValueError(
                f"embedding dimension mismatch for {node.uri}: "
                f"expected {expected_dimension}, got {dimension}"
            )
        try:
            finite = all(math.isfinite(value) for value in vector)
        except TypeError:
            finite = False
        if not finite:
            raise ValueError(f"embedding for {node.uri} must contain finite numeric values")
        vectors.append(_normalize(vector))
    return vectors


def cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def _centroid(indices: Collection[int], vectors: list[tuple[float, ...]]) -> tuple[float, ...]:
    dimension = len(vectors[0])
    values = [0.0] * dimension
    for index in indices:
        for position, value in enumerate(vectors[index]):
            values[position] += value
    return _normalize([value / len(indices) for value in values])


def _quality(indices: Collection[int], vectors: list[tuple[float, ...]]) -> float:
    if len(indices) < 2:
        return 0.0
    center = _centroid(indices, vectors)
    return sum(max(0.0, cosine(vectors[index], center)) for index in indices) / len(indices)


def _minimum_similarity(indices: set[int], vectors: list[tuple[float, ...]]) -> float:
    ordered = sorted(indices)
    return min(
        cosine(vectors[left], vectors[right])
        for position, left in enumerate(ordered)
        for right in ordered[position + 1:]
    )


def _label_hint(nodes: list[ContextNode], indices: Collection[int]) -> str:
    terms: Counter[str] = Counter()
    for index in indices:
        text = f"{nodes[index].abstract} {nodes[index].content[:300]}".lower()
        terms.update(token for token in _WORD.findall(text) if token not in _STOP)
    common = [term for term, _ in terms.most_common(3)]
    return " / ".join(common) or "semantic candidate"


def _summary_hint(
    nodes: list[ContextNode],
    indices: Collection[int],
    vectors: list[tuple[float, ...]],
) -> str:
    center = _centroid(indices, vectors)
    ranked = sorted(indices, key=lambda index: (-cosine(vectors[index], center), nodes[index].uri))
    parts = [text for index in ranked if (text := nodes[index].abstract.strip())]
    return " | ".join(parts[:3])[:800]


def _archive_ids(node: ContextNode) -> set[str]:
    values = (node.metadata or {}).get("provenance_ids", [])
    if isinstance(values, str):
        values = [values]
    output: set[str] = set()
    for value in values:
        try:
            parsed = ProvenanceResolver.parse_id(str(value))
        except ValueError:
            continue
        if parsed["source_type"] == "archive":
            output.add(parsed["source_id"])
    return output


def _entity_aliases(node: ContextNode) -> tuple[str, ...]:
    if node.category.lower() != "entity":
        return ()
    slug = node.uri.rstrip("/").rsplit("/", 1)[-1]
    metadata = node.metadata or {}
    for value in [metadata.get(key) for key in ("entity_name", "name", "who", "routing_key")] + [
        slug
    ]:
        if not isinstance(value, str) or re.fullmatch(r"auto[-_][0-9a-f]+", value):
            continue
        alias = " ".join(re.findall(r"[^\W_]+", value.replace("-", " ").lower()))
        if len(alias) >= 2 and alias not in _STOP:
            return (alias,)
    return ()


def _bounded_groups(ordered: list[int], capacity: int, minimum: int) -> list[set[int]]:
    """Balance chunks, sharing boundary evidence when minimum sizes require overlap."""
    minimum = min(minimum, capacity)
    count = math.ceil(len(ordered) / capacity)
    size, extra = divmod(len(ordered), count)
    groups = []
    offset = 0
    for position in range(count):
        end = offset + size + int(position < extra)
        start = max(0, min(offset, end - minimum))
        groups.append(set(ordered[start : max(end, minimum)]))
        offset = end
    return groups


def _semantic_group_valid(
    indices: set[int], vectors: list[tuple[float, ...]], cfg: CandidateConfig
) -> bool:
    if len(indices) < 2:
        return False
    minimum = _minimum_similarity(indices, vectors)
    return (
        (
            len(indices) >= cfg.min_candidate_size
            or (len(indices) == 2 and minimum >= cfg.strong_pair_threshold)
        )
        and minimum >= cfg.similarity_threshold
        and _quality(indices, vectors) >= cfg.min_quality
    )


def _rebalance_semantic_groups(
    clusters: dict[int, set[int]], vectors: list[tuple[float, ...]], cfg: CandidateConfig
) -> None:
    for small_id in sorted(clusters):
        small = clusters[small_id]
        if _semantic_group_valid(small, vectors, cfg):
            continue
        for donor_id in sorted(clusters):
            donor = clusters[donor_id]
            if donor_id == small_id or len(donor) <= cfg.min_candidate_size:
                continue
            combined = small | donor
            if _minimum_similarity(combined, vectors) < cfg.similarity_threshold:
                continue
            groups = _bounded_groups(
                sorted(combined), cfg.max_candidate_size, cfg.min_candidate_size
            )
            if len(groups) != 2 or sum(map(len, groups)) != len(combined):
                continue
            if all(_semantic_group_valid(group, vectors, cfg) for group in groups):
                clusters[small_id], clusters[donor_id] = groups
                break


def _node_mentions(node: ContextNode, mention_pattern: re.Pattern[str]) -> set[str]:
    """Match exact metadata names or the longest known name at each text position."""
    metadata = node.metadata or {}
    names = [
        value
        for key in ("who", "attributed_speaker", "routing_key", "subject")
        if isinstance(value := metadata.get(key), str)
    ]
    text = f"{node.abstract} {node.overview} {node.content[:2000]} {' '.join(names)}".lower()
    text = re.sub(r"[-_\s]+", " ", text)
    return {re.sub(r"[-_\s]+", " ", name.lower()).strip() for name in names} | {
        match.group() for match in mention_pattern.finditer(text)
    }


def _account_id(uri: str) -> str:
    match = re.fullmatch(r"ctx://([^/?#\\\s]+)/[^?#\\\s]+", uri)
    if match is None or match[1] in {".", ".."}:
        raise ValueError(f"invalid account-scoped context URI: {uri}")
    return match[1]


def _candidate(
    nodes: list[ContextNode],
    indices: Collection[int],
    vectors: list[tuple[float, ...]],
    *,
    basis: str,
    quality: float | None = None,
    source_archive_ids: tuple[str, ...] | None = None,
    anchor_index: int | None = None,
) -> SceneCandidate:
    scopes = {(_account_id(nodes[index].uri), nodes[index].owner_space) for index in indices}
    if len(scopes) != 1:
        raise ValueError("candidate members must belong to exactly one account and owner space")
    account_id, owner_space = next(iter(scopes))
    center = _centroid(indices, vectors)
    anchor = (
        anchor_index
        if anchor_index in indices
        else min(indices, key=lambda index: (-cosine(vectors[index], center), nodes[index].uri))
    )
    members = tuple(
        SceneMember(
            context_uri=nodes[index].uri,
            weight=max(0.0, min(1.0, cosine(vectors[index], center))),
            role="anchor" if index == anchor else "member",
        )
        for index in sorted(indices, key=lambda item: nodes[item].uri)
    )
    digest = hashlib.sha256()
    digest.update(basis.encode("ascii"))
    for member in members:
        digest.update(member.context_uri.encode("utf-8"))
        digest.update(f"{member.weight:.6f}".encode("ascii"))
    return SceneCandidate(
        account_id=account_id,
        owner_space=owner_space,
        members=members,
        centroid=center,
        quality=_quality(indices, vectors) if quality is None else quality,
        label_hint=_label_hint(nodes, indices),
        summary_hint=_summary_hint(nodes, indices, vectors),
        source_fingerprint=digest.hexdigest(),
        basis=basis,
        source_archive_ids=(
            tuple(sorted(set(source_archive_ids)))
            if source_archive_ids is not None
            else tuple(sorted({
                archive_id
                for index in indices
                for archive_id in _archive_ids(nodes[index])
            }))
        ),
    )


def _pairwise_similarities(
    vectors: list[tuple[float, ...]],
    indices: set[int],
) -> dict[int, array]:
    """Compute each unordered pair once in a packed double-precision buffer."""
    scores = {index: array("d", [0.0]) * len(vectors) for index in indices}
    ordered = sorted(indices)
    for position, left in enumerate(ordered):
        for right in ordered[position + 1 :]:
            scores[left][right] = scores[right][left] = cosine(vectors[left], vectors[right])
    return scores


def _nearest_neighbors(
    nodes: list[ContextNode],
    similarities: dict[int, array],
    indices: set[int],
    neighbor_k: int,
) -> dict[int, dict[int, float]]:
    """Select exact active top-k neighbors from reusable pairwise scores."""
    nearest: dict[int, dict[int, float]] = {}
    for index in sorted(indices):
        ranked = nlargest(
            neighbor_k,
            (
                (similarities[index][other], nodes[other].uri, other)
                for other in indices
                if other != index
            ),
        )
        nearest[index] = {other: similarity for similarity, _, other in ranked}
    return nearest


def _generate_owner_candidates(
    available: list[ContextNode],
    embeddings: dict[str, list[float]],
    cfg: CandidateConfig,
) -> list[SceneCandidate]:
    if len(available) < 2:
        return []

    available = sorted(available, key=lambda node: node.uri)
    vectors = _validated_vectors(available, embeddings)
    semantic_indices = {
        index for index, node in enumerate(available) if node.category.lower() != "entity"
    }
    clusters: dict[int, set[int]] = {index: {index} for index in semantic_indices}
    membership = list(range(len(available)))
    similarities = _pairwise_similarities(vectors, semantic_indices)
    active = semantic_indices
    while active:
        nearest = _nearest_neighbors(available, similarities, active, cfg.neighbor_k)
        edges: set[tuple[float, int, int]] = set()
        for left, neighbors in nearest.items():
            for right, similarity in neighbors.items():
                if similarity >= cfg.similarity_threshold:
                    first, second = sorted((left, right), key=lambda i: available[i].uri)
                    edges.add((similarity, first, second))
        for _, left, right in sorted(
            edges, key=lambda edge: (-edge[0], available[edge[1]].uri, available[edge[2]].uri)
        ):
            left_id, right_id = membership[left], membership[right]
            if left_id == right_id:
                continue
            combined = clusters[left_id] | clusters[right_id]
            if len(combined) > cfg.max_candidate_size:
                continue
            if _minimum_similarity(combined, vectors) < cfg.similarity_threshold:
                continue
            if _quality(combined, vectors) < cfg.min_quality:
                continue
            keep, remove = min(left_id, right_id), max(left_id, right_id)
            clusters[keep] = combined
            del clusters[remove]
            for index in combined:
                membership[index] = keep

        remaining = {
            index
            for indices in clusters.values()
            if len(indices) < cfg.max_candidate_size
            for index in indices
        }
        if remaining == active:
            break
        active = remaining

    _rebalance_semantic_groups(clusters, vectors, cfg)
    output: list[SceneCandidate] = []
    for indices in clusters.values():
        if not _semantic_group_valid(indices, vectors, cfg):
            continue
        output.append(_candidate(available, indices, vectors, basis="semantic"))

    by_archive: dict[str, set[int]] = {}
    for index, node in enumerate(available):
        for archive_id in _archive_ids(node):
            by_archive.setdefault(archive_id, set()).add(index)
    existing_members = {
        frozenset(member.context_uri for member in candidate.members)
        for candidate in output
    }
    episodic_sources: dict[frozenset[int], list[str]] = {}
    for archive_id, source_indices in sorted(by_archive.items()):
        if len(source_indices) < cfg.min_candidate_size and len(source_indices) != 2:
            continue
        center = _centroid(source_indices, vectors)
        ordered = sorted(
            source_indices,
            key=lambda index: (cosine(vectors[index], center), available[index].uri),
            reverse=True,
        )
        for selected in _bounded_groups(
            ordered, cfg.max_candidate_size, cfg.min_candidate_size
        ):
            member_ids = frozenset(available[index].uri for index in selected)
            if member_ids in existing_members:
                continue
            episodic_sources.setdefault(frozenset(selected), []).append(archive_id)
    for selected_indices, archive_ids in episodic_sources.items():
        selected = set(selected_indices)
        semantic_quality = _quality(selected, vectors)
        contextual_quality = max(cfg.min_quality, 0.5 + 0.5 * semantic_quality)
        output.append(
            _candidate(
                available,
                selected,
                vectors,
                basis="episodic",
                quality=min(1.0, contextual_quality),
                source_archive_ids=tuple(archive_ids),
            )
        )
        existing_members.add(frozenset(available[index].uri for index in selected))

    if cfg.max_candidate_size < 3:
        return output

    entity_anchors = [
        (index, aliases)
        for index, node in enumerate(available)
        if (aliases := _entity_aliases(node))
    ]
    if not entity_anchors:
        return output
    known_names = {alias for _, aliases in entity_anchors for alias in aliases}
    known_names.update(
        re.sub(r"[-_\s]+", " ", value.lower()).strip()
        for node in available
        for key in ("who", "attributed_speaker")
        if isinstance(value := (node.metadata or {}).get(key), str) and _CJK.search(value)
    )
    mention_pattern = re.compile(
        "|".join(
            re.escape(name) if _CJK.search(name) else rf"(?<!\w){re.escape(name)}(?!\w)"
            for name in sorted(known_names, key=lambda name: (-len(name), name))
        )
    )
    mentions = [_node_mentions(node, mention_pattern) for node in available]
    minimum_related = max(2, cfg.min_candidate_size - 1)
    for anchor_index, aliases in entity_anchors:
        remaining = {
            index
            for index, names in enumerate(mentions)
            if index != anchor_index and any(alias in names for alias in aliases)
        }
        chunks = 0
        while len(remaining) >= minimum_related and chunks < cfg.entity_candidate_max_chunks:
            seed = max(
                remaining,
                key=lambda index: (
                    cosine(vectors[anchor_index], vectors[index]),
                    available[index].uri,
                ),
            )
            selected_related = set(
                sorted(
                    remaining,
                    key=lambda index: (
                        cosine(vectors[seed], vectors[index]),
                        cosine(vectors[anchor_index], vectors[index]),
                        available[index].uri,
                    ),
                    reverse=True,
                )[: cfg.max_candidate_size - 1]
            )
            selected = {anchor_index, *selected_related}
            member_ids = frozenset(available[index].uri for index in selected)
            if member_ids not in existing_members:
                semantic_quality = _quality(selected, vectors)
                output.append(
                    _candidate(
                        available,
                        selected,
                        vectors,
                        basis="entity_anchor",
                        quality=min(1.0, max(cfg.min_quality, 0.5 + 0.5 * semantic_quality)),
                        anchor_index=anchor_index,
                    )
                )
                existing_members.add(member_ids)
            remaining.difference_update(selected_related)
            chunks += 1
    return output


def generate_scene_candidates(
    nodes: list[ContextNode],
    embeddings: dict[str, list[float]],
    config: CandidateConfig | None = None,
) -> list[SceneCandidate]:
    """Group account- and owner-isolated evidence for subsequent scene decisions.

    Candidate hints contain keywords and representative abstracts. The scene
    decision stage uses these groups and their source nodes to establish scene
    membership, titles, and descriptions.

    Neighbor scores cost O(N²D) once and use O(N²) packed-double storage
    (2 MiB at 512 nodes). Later passes select active top-k neighbors from these
    scores as full candidates leave the active set.
    Each account/owner pair is bounded by max_owner_nodes (default 512).
    Exceeding the bound raises ValueError before vector processing. Callers
    select bounded inputs using changed nodes and relevant existing evidence.
    """
    cfg = config or CandidateConfig()
    cfg.validate()
    nodes_by_owner: dict[tuple[str, str], list[ContextNode]] = {}
    for node in nodes:
        if embeddings.get(node.uri):
            scope = (_account_id(node.uri), node.owner_space)
            owner_nodes = nodes_by_owner.setdefault(scope, [])
            owner_nodes.append(node)
            if len(owner_nodes) > cfg.max_owner_nodes:
                raise ValueError(
                    f"account {scope[0]} owner {node.owner_space} exceeds max_owner_nodes={cfg.max_owner_nodes}; "
                    "select a bounded set of memory nodes"
                )

    output = [
        candidate
        for scope in sorted(nodes_by_owner)
        for candidate in _generate_owner_candidates(nodes_by_owner[scope], embeddings, cfg)
    ]
    output.sort(key=lambda candidate: tuple(member.context_uri for member in candidate.members))
    return output
