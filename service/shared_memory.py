"""Shared retrieval with authoritative SQL authorization before response assembly."""

from __future__ import annotations

from server.space_rules import SpaceError
from service.migration.maintenance import MaintenanceError


class SharedMemoryRetrievalService:
    def __init__(self, repository, vector_index=None, embedder=None):
        self.repository = repository
        self.vector_index = vector_index
        self.embedder = embedder

    def _candidates(self, ctx, query, space_ids, mode, kind, top_k):
        keyword = []
        semantic = []
        semantic_scores = {}
        if mode in ('keyword', 'hybrid'):
            keyword = self.repository.visible_items(
                ctx, space_ids, query=query, kind=kind, limit=top_k, hydrate=False)
        if mode in ('semantic', 'hybrid'):
            if self.vector_index is not None and self.embedder is not None:
                vector = self.embedder.embed_texts([query])[0]
                filters = {'account_id': ctx.account_id,
                           'owner_space': [f'space:{space}' for space in space_ids]}
                if kind:
                    filters['category'] = kind
                hits = self.vector_index.search_by_vector(vector, filters=filters, top_k=top_k * 12)
                for hit in hits:
                    uri = hit.uri.removesuffix('/content.md')
                    semantic_scores[uri] = max(semantic_scores.get(uri, float('-inf')), float(hit.score))
                semantic = self.repository.visible_items(
                    ctx, space_ids, uris=list(semantic_scores), kind=kind, limit=2000, hydrate=False)
                semantic.sort(key=lambda item: (-semantic_scores[item['uri']], item['uri']))
                semantic = semantic[:top_k]
            elif mode == 'semantic':
                raise SpaceError(503, 'SEMANTIC_UNAVAILABLE', 'Semantic search requires an embedding and vector index')
        if mode == 'semantic':
            return semantic, semantic_scores
        if mode == 'keyword':
            return keyword, {item['uri']: 1 / rank for rank, item in enumerate(keyword, 1)}
        candidates = {}
        scores = {}
        for ranked in (keyword, semantic):
            for rank, item in enumerate(ranked, 1):
                uri = item['uri']
                candidates[uri] = item
                scores[uri] = scores.get(uri, 0.0) + 1 / (60 + rank)
        ordered = sorted(candidates, key=lambda uri: (-scores[uri], uri))[:top_k]
        return [candidates[uri] for uri in ordered], scores

    def search(self, ctx, query, space_ids=None, top_k=10, kind=None, max_context_tokens=4096,
               mode='hybrid', limit=None, cursor=None):
        if not isinstance(query, str) or not 1 <= len(query.strip()) <= 4096:
            raise SpaceError(400, "INVALID_QUERY", "query must contain 1 to 4096 characters")
        if (not isinstance(space_ids, list) or not 1 <= len(space_ids) <= 20
                or any(not isinstance(space, str) or not space or len(space) > 64 for space in space_ids)):
            raise SpaceError(400, "INVALID_SPACES", "space_ids must select 1 to 20 spaces")
        if type(top_k) is not int or not 1 <= top_k <= 100:
            raise SpaceError(400, "INVALID_LIMIT", "top_k must be between 1 and 100")
        if type(max_context_tokens) is not int or not 128 <= max_context_tokens <= 8192:
            raise SpaceError(400, "INVALID_BUDGET", "max_context_tokens must be between 128 and 8192")
        if kind not in (None, "experience"):
            raise SpaceError(400, "INVALID_KIND", "kind must be experience")
        kind = 'experience'
        if mode not in ('hybrid', 'keyword', 'semantic'):
            raise SpaceError(400, 'INVALID_MODE', 'mode must be hybrid, keyword or semantic')
        if limit is not None and (type(limit) is not int or not 1 <= limit <= 50):
            raise SpaceError(400, 'INVALID_LIMIT', 'limit must be between 1 and 50')
        if cursor is not None and (not isinstance(cursor, str) or len(cursor) > 4096 or limit is None):
            raise SpaceError(400, 'INVALID_CURSOR', 'A search cursor requires a page limit')
        query = query.strip()
        space_ids = list(dict.fromkeys(space_ids))
        offset, binding = 0, None
        if limit is not None:
            scope = [query, sorted(space_ids), mode, kind, top_k, limit, max_context_tokens]
            offset, binding = self.repository.search_page_context(ctx, scope, cursor)
        candidates, scores = self._candidates(ctx, query, space_ids, mode, kind, top_k)
        total = len(candidates)
        candidates = candidates[offset:offset + (limit or top_k)]
        # Recheck the selected batch after recall, immediately before assembly.
        visible = {item["uri"]: item for item in self.repository.visible_items(
            ctx, space_ids, uris=[item["uri"] for item in candidates], kind=kind, limit=limit or top_k)}
        if getattr(self.repository, "content_store", None) is not None and visible:
            current = {item["uri"]: item for item in self.repository.visible_items(
                ctx, space_ids, uris=list(visible), kind=kind, limit=limit or top_k, hydrate=False)}
            visible = {uri: item for uri, item in visible.items()
                       if uri in current and current[uri]["version"] == item["version"]}
        budget = max_context_tokens
        results = []
        consumed = 0
        for candidate in candidates:
            item = visible.get(candidate["uri"])
            if item is None:
                consumed += 1
                continue
            if budget <= 0:
                break
            # UTF-8 bytes give a conservative bound independent of model tokenization.
            content = item["content"].encode()[:budget].decode("utf-8", errors="ignore")
            if item['content'] and not content:
                break
            budget -= len(content.encode())
            consumed += 1
            results.append({"item_id": item["item_id"], "space_id": item["space_id"], "uri": item["uri"],
                            "kind": item["kind"], "content": content, "abstract": item["abstract"],
                            "score": scores.get(item["uri"], 1.0), "rule_id": item["rule_id"],
                            "rule_rev": item["rule_rev"], "content_rev": item["version"],
                            "source_count": item.get("source_count", 0), "index_state": "READY",
                            "updated_at": item["updated_at"]})
        if results:
            try:
                with self.repository.transaction(ctx.account_id, write=True) as cur:
                    cur.execute("""UPDATE space_items SET last_used_at=NOW()
                        WHERE account_id=%s AND uri=ANY(%s) AND
                        (last_used_at IS NULL OR last_used_at<CURRENT_DATE)""",
                        (ctx.account_id, [item["uri"] for item in results]))
            except MaintenanceError:
                # Usage statistics are optional; a source snapshot keeps reads
                # available while atomically rejecting this write admission.
                pass
        next_cursor = None
        if limit is not None and offset + consumed < total:
            next_cursor = self.repository.cursor_after(offset + consumed, binding)
        return {"items": results, "total": total if limit is not None else len(results),
                "next_cursor": next_cursor, "mode": mode, "query": query, "space_ids": space_ids}
