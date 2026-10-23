"""Live, request-local fresh-k3 traversal and prefix-protected final delivery.

No frozen dataset, global query cache, monkey-patch, LLM or graph re-embedding.
The native one-hop/assembly path remains authoritative; this lane only appends.
"""
from collections import defaultdict, deque
import hashlib
import json
import math

from core.errors import AccessDeniedError, NodeBrokenError, NodeNotFoundError
from core.models import IndexRecord, RelationNeighbor
from core.uri_parser import strip_l2_suffix
from retrieval.graph_retriever import OneHopRetriever
from retrieval.graph_selection import (
    HOP_DECAY, LOCAL_LANE_LIMIT, MAX_ANCHORS, MAX_DESTINATIONS,
    MAX_EXPANDED_NODES, MAX_GRAPH_NODES, MAX_HOPS, PREGRAPH_CANDIDATE_LIMIT,
    REFERENCE_METHOD, SIMILARITY_THRESHOLD, cosine, two_lane_order,
)

INACTIVE = {"archived", "deleted", "broken", "retired", "inactive", "superseded", "tombstoned", "expired"}
EXPECTED_REJECTIONS = (AccessDeniedError, NodeBrokenError, NodeNotFoundError, PermissionError)
PROTOCOL = "live-fresh-k3-v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def text_digest(value):
    """Match the archived evidence receipt: hash rendered text, not JSON text."""
    return hashlib.sha256(value.encode()).hexdigest()


def binding(ctx):
    return {key: getattr(ctx, key) for key in ("account_id", "user_id", "agent_id", "session_id", "trace_id")}


def body(block):
    return dict(abstract=block.abstract or "", overview=block.overview or "", content=block.content_excerpt or "")


def revision(block):
    return int(block.metadata.get("version", block.metadata.get("canonical_revision", -1)))


class GraphSupplement:
    """Uses existing GraphService, ContextReader and vector index contracts."""
    def __init__(self, graph_service, context_reader, vector_index, embedder, profile):
        self.graph = graph_service
        self.reader = context_reader
        self.index = vector_index
        self.embedder = embedder
        self.profile = dict(profile)
        self.gate = OneHopRetriever(None, embedding_profile=self.profile)

    def read(self, uri, ctx):
        allowed = (f"ctx://{ctx.account_id}/users/{ctx.user_id}/",
                   f"ctx://{ctx.account_id}/agents/{ctx.agent_id}/")
        if not uri.startswith(allowed):
            raise PermissionError("Graph supplement outside current user/agent scope")
        block = self.reader.read_authorized(uri, ctx)
        lifecycle = block.metadata.get("status") or block.metadata.get("lifecycle_status") or block.metadata.get("visibility")
        if str(lifecycle or "").lower() in INACTIVE:
            raise NodeNotFoundError(uri)
        return block

    def _vectors(self, uri, block, ctx):
        """Read stored L0/L1/L2 vectors only; never synthesize a missing vector."""
        getter = getattr(self.index, "get_record", None)
        if getter is None:
            return []
        result = []
        identities = {"embedding_profile": self.profile["profile_id"],
            "profile_id": self.profile["profile_id"], "embedding_profile_id": self.profile["profile_id"],
            "model": self.profile["model"], "embedding_model": self.profile["model"],
            "dimension": self.profile["dimension"], "embedding_dimension": self.profile["dimension"],
            "representation_version": self.profile["representation_version"]}
        ids = [IndexRecord.generate_id(u, level) for u, level in
               ((uri, 0), (uri, 1), (uri.rstrip("/")+"/content.md", 2))]
        batch = getattr(self.index, "get_records", None)
        records = batch(ids) if batch is not None else [getter(i) for i in ids]
        for record in records:
            if record is None:
                continue
            meta = record.metadata
            vector = meta.get("_embedding")
            if (strip_l2_suffix(record.uri) != uri or record.filters.get("account_id") != ctx.account_id
                or record.filters.get("owner_space") != block.owner_space
                or meta.get("record_kind") not in ("node_abstract", "node_overview", "node_content")
                or str(meta.get("canonical_revision", -1)) != str(revision(block))
                or meta.get("embedding_model") != self.profile["model"]
                or meta.get("embedding_dimension") != self.profile["dimension"]
                or any(k in meta and meta[k] != value for k, value in identities.items())
                or not isinstance(vector, (list, tuple)) or len(vector) != self.profile["dimension"]
                or not all(math.isfinite(x) for x in vector)):
                continue
            result.append(vector)
        return result

    def select(self, query, pregraph, ordinary, ctx, *, query_vector=None, categories=None,
               memory_directories=None):
        audit = dict(protocol=PROTOCOL, reference_method=REFERENCE_METHOD,
            embedding_profile=dict(self.profile), query_sha256=digest(query), scope=binding(ctx),
            anchors=[], selected=[], decisions=[], expanded_nodes=[], authorization_rejections=0,
            candidate_uris=[strip_l2_suffix(c.uri) for c in pregraph],
            ordinary_uris=[strip_l2_suffix(c.uri) for c in ordinary],
            local_limit_per_lane=LOCAL_LANE_LIMIT, local_max_unique_candidates=2*LOCAL_LANE_LIMIT,
            max_hops=MAX_HOPS, max_expanded_nodes=MAX_EXPANDED_NODES, max_graph_nodes=MAX_GRAPH_NODES,
            final_selection="URI-unique semantic/query interleaving, semantic first",
            additional_embedding_calls=0, generative_llm_calls=0,
            categories=sorted(categories) if categories is not None else None,
            memory_directories=list(memory_directories) if memory_directories is not None else None)
        if not pregraph:
            return audit
        if query_vector is None:
            vectors = self.embedder.embed_texts([query])
            if len(vectors) != 1:
                raise ValueError("Graph query embedding count mismatch")
            query_vector = vectors[0]
            audit["additional_embedding_calls"] = 1
        if (len(query_vector) != self.profile["dimension"]
                or not all(math.isfinite(x) for x in query_vector)):
            raise ValueError("Graph query embedding dimension mismatch")
        # All caches below belong to this call, never to the shared pipeline.
        blocks, nodes, adjacent, node_vectors = {}, {}, {}, {}

        def read(uri):
            if uri not in blocks:
                try:
                    blocks[uri] = self.read(uri, ctx)
                    if categories is not None and blocks[uri].category not in categories:
                        raise NodeNotFoundError(uri)
                    from core.memory_names import memory_directory
                    if memory_directories is not None and memory_directory(uri) not in memory_directories:
                        raise NodeNotFoundError(uri)
                    nodes[uri] = body(blocks[uri])
                except EXPECTED_REJECTIONS:
                    blocks[uri] = None
                    audit["authorization_rejections"] += 1
            return blocks[uri]

        def neighbors(uri):
            """Cache compatible structure, not lifecycle-filtered destinations.

            The reference counts an active seed as connected before hydrating
            its neighbors. Hydrating here would silently replace its first five
            anchors when their only neighbors are inactive. SQL still enforces
            endpoint ACLs; live content/revision checks occur during expansion.
            """
            if uri not in adjacent:
                adjacent[uri] = []
                for edge in self.graph.get_traversal_neighbors(
                        uri, ctx, limit=MAX_DESTINATIONS, embedding_profile=self.profile):
                    source, target = edge["source_uri"], edge["target_uri"]
                    if uri not in (source, target) or edge["relation_type"].lower() in ("supersedes", "conflict_alert"):
                        continue
                    allowed = (f"ctx://{ctx.account_id}/users/{ctx.user_id}/",
                               f"ctx://{ctx.account_id}/agents/{ctx.agent_id}/")
                    if not source.startswith(allowed) or not target.startswith(allowed):
                        continue
                    vector = self.gate._compatible_vector(RelationNeighbor(
                        uri=target, edge_projection=edge["edge_projection"],
                        source_revision=edge["source_revision"], target_revision=edge["target_revision"]))
                    weight = edge["weight"]
                    if vector is None or not math.isfinite(weight) or weight < 0 or not all(math.isfinite(x) for x in vector):
                        continue
                    adjacent[uri].append(dict(edge_source=source, edge_target=target,
                        uri=source if uri == target else target, incoming=uri == target,
                        relation=edge["relation_type"], vector=vector, weight=weight,
                        source_revision=edge["source_revision"], target_revision=edge["target_revision"]))
            return adjacent[uri]

        for candidate in pregraph[:PREGRAPH_CANDIDATE_LIMIT]:
            uri = strip_l2_suffix(candidate.uri)
            if uri not in audit["anchors"] and read(uri) and neighbors(uri):
                audit["anchors"].append(uri)
            if len(audit["anchors"]) == MAX_ANCHORS:
                break
        frontier = deque((uri, [uri], 0, 1.0) for uri in audit["anchors"])
        expanded, records = set(), {}
        while frontier and len(expanded) < MAX_EXPANDED_NODES:
            anchor, path, hop, quality = frontier.popleft()
            if anchor in expanded or hop >= MAX_HOPS or not read(anchor):
                continue
            expanded.add(anchor)
            grouped = defaultdict(list)
            for edge in neighbors(anchor):
                grouped[edge["uri"]].append(edge)
            destinations = sorted(grouped, key=lambda u: (-max(e["weight"] for e in grouped[u]), u))[:MAX_DESTINATIONS]
            best = {}
            for uri in destinations:
                for edge in grouped[uri]:
                    # Lifecycle is an expansion gate AFTER structural anchor
                    # selection and the unique-destination cap, as in exp33.
                    source, target = read(edge["edge_source"]), read(edge["edge_target"])
                    if (not source or not target or revision(source) != edge["source_revision"]
                            or revision(target) != edge["target_revision"]):
                        continue
                    similarity = cosine(query_vector, edge["vector"])
                    qualification = similarity
                    endpoint_similarity = None
                    if edge["incoming"]:
                        if uri not in node_vectors:
                            node_vectors[uri] = self._vectors(uri, read(uri), ctx)
                        if node_vectors[uri]:
                            endpoint_similarity = max(cosine(query_vector, v) for v in node_vectors[uri])
                            qualification = max(qualification, endpoint_similarity)
                    # Incoming endpoint similarity adds eligibility, not a rank boost.
                    score = min(quality, similarity*edge["weight"])*(HOP_DECAY**hop)
                    accepted = qualification >= SIMILARITY_THRESHOLD and uri not in path
                    record = {k: v for k, v in edge.items() if k != "vector"}
                    record.update(score=score, forward_similarity=similarity,
                        visited_endpoint_similarity=endpoint_similarity, qualification_similarity=qualification,
                        original_projected_target=edge["edge_target"], accepted=accepted,
                        hop=hop+1, path=path+[uri], anchor_uri=path[0], local_selected=False)
                    audit["decisions"].append(record)
                    if accepted and (uri not in best or score > best[uri]["score"]):
                        best[uri] = record
            for rank, record in enumerate(two_lane_order(list(best.values()), nodes, query,
                                                        local_limit=LOCAL_LANE_LIMIT), 1):
                uri = record["uri"]
                record.update(local_interleaved_rank=rank, local_query_rank=record["query_rank"],
                              local_semantic_rank=record["semantic_rank"], local_selected=True)
                best[uri].update({key: record[key] for key in (
                    "local_interleaved_rank", "local_query_rank", "local_semantic_rank",
                    "lexical_score", "selection_lanes", "local_selected")})
                if uri not in records or record["score"] > records[uri]["score"]:
                    records[uri] = record
                if hop+1 < MAX_HOPS:
                    frontier.append((uri, record["path"], hop+1,
                                     min(quality, record["forward_similarity"]*record["weight"])))
        excluded = {strip_l2_suffix(c.uri) for c in ordinary}
        selected = two_lane_order([r for u, r in records.items() if u not in excluded], nodes, query)[:MAX_GRAPH_NODES]
        audit["selected"] = [dict(r, selection_rank=rank, revision=revision(blocks[r["uri"]]),
                                 body_sha256=digest(body(blocks[r["uri"]])))
                             for rank, r in enumerate(selected, 1)]
        audit["expanded_nodes"] = sorted(expanded)
        return audit

    def deliver(self, items, query, stats, ctx, budget, formatter):
        audit = (stats.get("graph_expansion") or {}).get("two_hop_supplement")
        if audit is None:
            return items
        if (audit.get("protocol") != PROTOCOL or audit.get("reference_method") != REFERENCE_METHOD
                or audit.get("embedding_profile") != self.profile
                or audit.get("query_sha256") != digest(query) or audit.get("scope") != binding(ctx)):
            raise ValueError("Cross-request graph supplement state")
        baseline = formatter(items, budget)
        # Legacy native formatter disables clipping at zero. This is only a
        # whole-body size probe; the real allocation is never increased.
        # dev's native formatter treats zero as an empty budget, not unlimited.
        # Probe with a bounded full-body budget, never increase actual delivery.
        def full_budget(values):
            chars = 4096 + sum(sum(len(str(i.get(k) or "")) for k in
                ("uri", "abstract", "overview", "content", "category")) + 1024 for i in values)
            return {"working_set": max(int(budget.get("working_set", 20000)), chars // 4 + 1)}
        full = formatter(items, full_budget(items))
        limit = int(budget.get("working_set", 20000))
        graph_count = len({strip_l2_suffix(i["uri"]) for i in items
                           if (i.get("metadata") or {}).get("retrieval_origin") == "graph_neighbor"})
        marker = dict(reference_method=REFERENCE_METHOD, embedding_profile=dict(self.profile),
                      added_uris=[], rejected_additions=[], baseline_prefix_unchanged=True,
                      baseline_sha256=digest(baseline), baseline_format_sha256=text_digest(baseline),
                      max_graph_nodes=MAX_GRAPH_NODES, actual_budget=dict(budget), actual_character_limit=limit*4,
                      baseline_characters=len(baseline), baseline_uris=[i["uri"] for i in items],
                      native_graph_count=graph_count, working_set_budget=limit,
                      final_sha256=digest(baseline), final_format_sha256=text_digest(baseline),
                      final_characters=len(baseline), no_reordering=True, no_new_clipping=True,
                      conflict_management_enabled=False, duplication_interpretation_enabled=False)
        stats["graph_supplement_delivery"] = marker
        if "working_set" not in budget or limit <= 100 or len(full) > limit*4 or full != baseline:
            marker["rejected_additions"].append("no_unused_native_budget")
            return items
        delivered, seen = list(items), {strip_l2_suffix(i["uri"]) for i in items}
        for selected in audit["selected"]:
            uri = selected["uri"]
            if uri in seen or graph_count + len(marker["added_uris"]) >= MAX_GRAPH_NODES:
                continue
            try:
                block = self.read(uri, ctx)  # Re-authorize at the final injection boundary.
            except EXPECTED_REJECTIONS:
                marker["rejected_additions"].append("unavailable_or_unauthorized")
                continue
            if audit.get("categories") is not None and block.category not in audit["categories"]:
                marker["rejected_additions"].append("category_outside_request")
                continue
            from core.memory_names import memory_directory
            if (audit.get("memory_directories") is not None
                    and memory_directory(uri) not in audit["memory_directories"]):
                marker["rejected_additions"].append("directory_outside_request")
                continue
            if revision(block) != selected["revision"] or digest(body(block)) != selected["body_sha256"]:
                marker["rejected_additions"].append("node_changed_during_request")
                continue
            item = dict(uri=uri, **body(block), category=block.category, score=selected["score"],
                when=block.metadata.get("when"), who=block.metadata.get("who"), where=block.metadata.get("where"),
                metadata={**block.metadata, "retrieval_origin": "graph_neighbor", "graph_supplemental": True,
                          "graph_hop_distance": selected["hop"], "graph_path": selected["path"]})
            text = formatter(delivered+[item], full_budget(delivered+[item]))
            if len(text) > limit*4:
                marker["rejected_additions"].append("whole_node_exceeds_unused_budget")
                continue
            if baseline and not text.startswith(baseline):
                raise AssertionError("Graph supplement changed the native evidence prefix")
            delivered.append(item)
            seen.add(uri)
            marker["added_uris"].append(uri)
        final = formatter(delivered, budget)
        if baseline and not final.startswith(baseline):
            raise AssertionError("Graph supplement changed the delivered evidence prefix")
        marker["final_sha256"] = digest(final)
        marker["final_format_sha256"] = text_digest(final)
        marker["final_characters"] = len(final)
        return delivered
