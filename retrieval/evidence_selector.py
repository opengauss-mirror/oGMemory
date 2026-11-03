"""Bounded, source-preserving L2 evidence for composed memory context."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace

from retrieval.identity import canonical_uri, memory_identity, memory_unit_key


@dataclass(frozen=True)
class EvidenceLimits:
    max_candidates: int = 15
    max_node_bytes: int = 64 * 1024
    max_total_bytes: int = 256 * 1024
    max_blocks: int = 256
    max_block_bytes: int = 1024
    max_embedding_blocks: int = 96
    max_embedding_bytes: int = 64 * 1024
    max_query_bytes: int = 4096


LIMITS = EvidenceLimits()


def _size(text: str) -> int:
    return len(text.encode("utf-8"))


def _clip(text: str, limit: int) -> str:
    return text.encode("utf-8")[:max(0, limit)].decode("utf-8", errors="ignore")


def hydrate_evidence(items, read_memory, ctx, limits=LIMITS):
    """Read canonical candidates with caller ACL before selecting passages."""
    hydrated, diagnostics, seen = [], [], set()
    total_bytes = 0
    reads = 0
    for item in items:
        if item.get("category") == "code":
            hydrated.append(item)
            continue
        uri = canonical_uri(item.get("uri", ""))
        identity = memory_identity(uri, item.get("metadata"))
        unit_key = identity[1]
        if identity in seen:
            continue
        seen.add(identity)
        if reads >= limits.max_candidates:
            continue
        entry = {"uri": uri, "selected_ranges": []}
        if unit_key:
            entry["unit_key"] = unit_key
        diagnostics.append(entry)
        if not uri:
            entry["reason"] = "missing_uri"
            continue
        reads += 1
        try:
            block = read_memory(uri, ctx, **({"unit_key": unit_key} if unit_key else {}))
        except Exception as exc:
            entry["reason"] = "read_failed"
            entry["error_kind"] = type(exc).__name__
            continue
        if all(getattr(block, field, None) is None for field in ("abstract", "overview", "content_excerpt")):
            entry["reason"] = "read_failed"
            continue
        hydrated_item = {**item, "uri": uri}
        for key, field in (("abstract", "abstract"), ("overview", "overview"),
                           ("content", "content_excerpt"), ("category", "category")):
            value = getattr(block, field, None)
            if isinstance(value, str):
                hydrated_item[key] = value
        metadata = getattr(block, "metadata", None)
        hydrated_item["metadata"] = {
            **(item.get("metadata") or {}),
            **(metadata if isinstance(metadata, dict) else {}),
        }
        content = hydrated_item.get("content", "")
        length = _size(content)
        entry.update(content_chars=len(content), content_bytes=length)
        if length > limits.max_node_bytes:
            hydrated_item["content"] = ""
            entry["reason"] = "node_byte_limit"
        elif total_bytes + length > limits.max_total_bytes:
            hydrated_item["content"] = ""
            entry["reason"] = "total_byte_limit"
        else:
            total_bytes += length
            entry["reason"] = "ready" if content else "summary_only"
        hydrated_item["_evidence_diagnostic"] = entry
        hydrated.append(hydrated_item)
    return hydrated, {
        "read_count": reads,
        "candidate_limit": limits.max_candidates,
        "candidate_limit_skipped": max(0, len(seen) - reads),
        "admitted_content_bytes": total_bytes,
        "sources": diagnostics,
    }


@dataclass(frozen=True)
class _Passage:
    start: int
    end: int
    heading: str


def _passages(content: str, limits: EvidenceLimits) -> list[_Passage]:
    """Group nearby lines, preserving headings and original character offsets."""
    passages = []
    headings: list[tuple[int, str]] = []
    start = end = 0
    heading = ""
    for line in content.splitlines(keepends=True):
        match = re.match(r"^(#{1,6})\s+(.+)", line)
        if match:
            if end > start:
                passages.append(_Passage(start, end, heading))
            level = len(match[1])
            headings = [(depth, title) for depth, title in headings if depth < level]
            headings.append((level, line.strip()))
            heading = _clip("\n".join(title for _, title in headings), 256)
            start = end
        line_end = end + len(line)
        while _size(content[start:line_end]) > limits.max_block_bytes:
            if end > start:
                passages.append(_Passage(start, end, heading))
                start = end
            else:
                piece = _clip(content[start:line_end], limits.max_block_bytes)
                boundary = max(piece.rfind(". "), piece.rfind("。"), piece.rfind("; "))
                cut = start + (boundary + 1 if boundary > len(piece) // 2 else len(piece))
                passages.append(_Passage(start, cut, heading))
                start = end = cut
        end = line_end
    if end > start:
        passages.append(_Passage(start, end, heading))
    return passages


def _terms(text: str) -> set[str]:
    text = text.casefold()
    return set(re.findall(r"\w+", text)) | {
        text[index:index + 2] for index in range(len(text) - 1)
        if text[index:index + 2].isalnum()
    }


def _cosine(left, right) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    norm = math.sqrt(sum(a * a for a in left) * sum(b * b for b in right))
    return dot / norm if norm else 0.0


def _heading(item: dict) -> str:
    abstract = _clip(str(item.get("abstract") or ""), 512)
    lines = [f"- [{item.get('category') or 'memory'}] {abstract}",
             f"  URI: {item.get('uri', '')}"]
    metadata = item.get("metadata") or {}
    if memory_unit_key(metadata):
        lines.append(f"  unit_key: {memory_unit_key(metadata)}")
        if metadata.get("unit_revision") is not None:
            lines.append(f"  unit_revision: {metadata['unit_revision']}")
    for key in ("when", "source_uri", "source_session_id"):
        if metadata.get(key):
            lines.append(f"  {key}: {_clip(str(metadata[key]), 256)}")
    return "\n".join(lines)


def _render_passage(content, block, start, end):
    prefix = block.heading + "\n" if block.heading else ""
    return f"{prefix}[chars {start}:{end}]\n{content[start:end]}"


def select_evidence(items, query, max_chars, *, embedder_factory=None, limits=LIMITS):
    """Select complete source spans within Compose's character budget."""
    entries, stats = [], {"embedding_text_count": 0, "embedding_input_bytes": 0,
                          "delivered_uris": [], "delivered_units": []}
    if max_chars <= 0:
        for item in items:
            item.get("_evidence_diagnostic", {}).update(reason="output_budget")
        return "", {**stats, "output_chars": 0, "output_bytes": 0,
                    "output_budget_chars": 0, "estimated_tokens": 0}
    query = _clip(query, limits.max_query_bytes)
    query_terms = _terms(query)
    remaining = max(0, max_chars)
    plans = []
    candidates = []
    admitted = []
    for item in items[:limits.max_candidates]:
        if len(_heading(item)) + 1 <= max_chars:
            admitted.append(item)
        else:
            item.get("_evidence_diagnostic", {}).update(reason="output_budget")
    items = admitted
    for index, item in enumerate(items):
        if item.get("category") == "skill" and item.get("overview"):
            items[index] = {**item, "content": item["overview"]}
            item.get("_evidence_diagnostic", {}).update(source_field="overview")
    full_chars = sum(
        len(_heading(item)) + (1 + len(item.get("content") or item.get("overview") or "")
                              if item.get("content") or item.get("overview") else 0)
        for item in items
    ) + max(0, len(items) - 1) * 2
    all_fit = full_chars <= max_chars
    minimums = []
    for index, item in enumerate(items):
        header = _heading(item)
        content = item.get("content") or ""
        separator_chars = 2 if index else 0
        # Rendering spends at most each entry's share, so later shares can only grow.
        allowance = max(0, max_chars // len(items) - len(header) - separator_chars - 1)
        block_limits = replace(
            limits, max_block_bytes=max(64, min(limits.max_block_bytes, allowance // 2)),
        )
        blocks = _passages(content, block_limits) if not all_fit and content else []
        plans.append((item, header, content, blocks))
        body_cost = min(64, len(item.get("overview") or ""))
        if content:
            body_cost = min(len(content), min(
                (len(_render_passage(content, b, b.start, b.end)) for b in blocks), default=len(content)))
        minimums.append(len(header) + separator_chars + 1 + body_cost)
        for block in blocks:
            text = block.heading + "\n" + content[block.start:block.end]
            overlap = len(query_terms & _terms(text)) / max(1, len(query_terms))
            candidates.append((index, block, text, overlap))
    # Admit a ranked prefix with room for complete, already prepared source spans.
    while not all_fit and len(plans) > 1 and max(minimums) > max_chars // len(plans):
        item, header, content, _ = plans.pop()
        minimums.pop()
        body = content or item.get("overview") or ""
        full_chars -= len(header) + (1 + len(body) if body else 0) + 2
        all_fit = full_chars <= max_chars
        item.get("_evidence_diagnostic", {}).update(reason="output_budget")
    candidates = [c for c in candidates if c[0] < len(plans)] if not all_fit else []
    if len(candidates) > limits.max_blocks:
        ranked = sorted(candidates, key=lambda candidate: -candidate[3])
        focused = ranked[:limits.max_blocks // 2]
        rest = [candidate for candidate in candidates if candidate not in focused]
        count = limits.max_blocks - len(focused)
        candidates = focused + [rest[index * len(rest) // count] for index in range(count)]
    scores = {(index, block.start): overlap for index, block, _, overlap in candidates}
    if candidates and query and embedder_factory:
        ranked = sorted(candidates, key=lambda candidate: -candidate[3])
        focused = ranked[:limits.max_embedding_blocks // 2]
        rest = [candidate for candidate in candidates if candidate not in focused]
        count = min(len(rest), limits.max_embedding_blocks - len(focused))
        ranked = focused + [rest[index * len(rest) // count] for index in range(count)]
        semantic_candidates, used_bytes = [], _size(query)
        for candidate in ranked:
            text = _clip(candidate[2], limits.max_block_bytes)
            if len(semantic_candidates) >= limits.max_embedding_blocks:
                break
            if used_bytes + _size(text) > limits.max_embedding_bytes:
                break
            semantic_candidates.append((*candidate[:2], text, candidate[3]))
            used_bytes += _size(text)
        try:
            embedder = embedder_factory()
            texts = [query] + [candidate[2] for candidate in semantic_candidates]
            stats.update(embedding_text_count=len(texts), embedding_input_bytes=used_bytes)
            vectors = embedder.embed_texts(texts)
            if len(vectors) != len(texts):
                raise ValueError("embedding_count")
            semantic_scores = {key: overlap * 0.1 for key, overlap in scores.items()}
            for candidate, vector in zip(semantic_candidates, vectors[1:]):
                index, block, _, overlap = candidate
                similarity = max(-1.0, min(1.0, _cosine(vectors[0], vector)))
                semantic_scores[index, block.start] = (similarity + 1.0) * 0.45 + overlap * 0.1
            scores = semantic_scores
            stats["ranking"] = "semantic"
        except Exception as exc:
            stats.update(ranking="lexical", embedding_error=type(exc).__name__)
    else:
        stats["ranking"] = "lexical" if candidates else "full_content"
    stats["passage_count"] = len(candidates)
    for index, (item, header, content, blocks) in enumerate(plans):
        entry = item.get("_evidence_diagnostic", {})
        allowance = remaining if all_fit else remaining // (len(plans) - index)
        separator_chars = 2 if entries else 0
        body_budget = allowance - len(header) - separator_chars - 1
        if body_budget < 0:
            entry["reason"] = "output_budget"
            continue
        ranges = []
        if content and len(content) <= body_budget:
            body = content
            ranges = [[0, len(content)]]
        elif content:
            selected = []
            eligible = [block for block in blocks if (index, block.start) in scores]
            for block in sorted(eligible, key=lambda block: -scores[index, block.start]):
                start = max(0, block.start - 80)
                end = min(len(content), block.end + 80)
                if any(start < old_end and end > old_start for old_start, old_end in ranges):
                    continue
                passage = _render_passage(content, block, start, end)
                if len(passage) + (2 if selected else 0) > body_budget:
                    start, end = block.start, block.end
                    passage = _render_passage(content, block, start, end)
                cost = len(passage) + (2 if selected else 0)
                if cost <= body_budget:
                    selected.append((start, passage))
                    ranges.append([start, end])
                    body_budget -= cost
            body = "\n\n".join(text for _, text in sorted(selected))
        else:
            body = (item.get("overview") or "")[:body_budget]
        rendered = header + ("\n" + body if body else "")
        remaining -= len(rendered) + separator_chars
        entry.update(selected_ranges=sorted(ranges), delivered_bytes=_size(body),
                     delivered_chars=sum(end - start for start, end in ranges))
        if content:
            entry["reason"] = "full_content" if ranges == [[0, len(content)]] else "selected_passages" if ranges else "output_budget"
        entries.append(rendered)
        if body or not (content or item.get("overview")):
            stats["delivered_uris"].append(canonical_uri(item.get("uri", "")))
            unit_key = memory_unit_key(item.get("metadata"))
            if unit_key:
                stats["delivered_units"].append({"uri": canonical_uri(item.get("uri", "")),
                                                 "unit_key": unit_key})
    text = "\n\n".join(entries)
    stats.update(output_bytes=_size(text), output_chars=len(text),
                 output_budget_chars=max(0, max_chars), estimated_tokens=math.ceil(len(text) / 4))
    return text, stats
