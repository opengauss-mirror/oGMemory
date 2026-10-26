"""Opt-in M3 context admission: bounded requests, extractive evidence, no LLM.

The default counter charges one unit per UTF-8 byte plus framing. An optional
pinned local tokenizer gives a less pessimistic estimate with a multiplier.
Neither claims to know every remote template/tokenizer: validate provider usage.
Raw tool results stay outside this model-facing projection.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path

from core.errors import LLMCallError


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class ContextBudget:
    context_window: int = 32768
    max_input_tokens: int = 24000
    max_output_tokens: int = 4096
    safety_tokens: int = 2048
    max_top_k: int = 16
    max_candidates: int = 32
    max_result_nodes: int = 24
    max_neighbor_clips: int = 64
    max_search_calls: int = 8
    max_query_bytes: int = 2048
    max_selected_nodes: int = 32
    max_passages_per_node: int = 32
    max_scan_passages_per_node: int = 256
    max_admission_checks: int = 64
    max_node_bytes: int = 65536
    tokenizer_path: str | None = None
    tokenizer_multiplier: float = 1.15

    def __post_init__(self):
        for field in fields(self):
            if field.name in {"tokenizer_path", "tokenizer_multiplier"}:
                continue
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"qa_context.{field.name} must be a positive integer")
        if self.tokenizer_path is not None and (
            not isinstance(self.tokenizer_path, str) or not self.tokenizer_path.strip()
        ):
            raise ValueError("qa_context.tokenizer_path must be a local tokenizer JSON path")
        if (isinstance(self.tokenizer_multiplier, bool)
                or not isinstance(self.tokenizer_multiplier, (float, int))
                or not math.isfinite(self.tokenizer_multiplier) or self.tokenizer_multiplier < 1):
            raise ValueError("qa_context.tokenizer_multiplier must be finite and >= 1")
        if self.max_candidates < self.max_top_k:
            raise ValueError("qa_context.max_candidates must be >= max_top_k")
        if self.context_window <= self.max_output_tokens + self.safety_tokens:
            raise ValueError("qa_context has no input capacity after output/safety reservation")

    @property
    def input_limit(self):
        return min(self.max_input_tokens,
                   self.context_window - self.max_output_tokens - self.safety_tokens)


def load_context_budget(config_path=None):
    from m3.config import _load_yaml_config, load_m3_runtime_config

    block = _load_yaml_config(config_path).get("m3", {}).get("qa_context", {})
    if not isinstance(block, dict):
        raise ValueError("m3.qa_context must be a mapping")
    enabled = block.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("m3.qa_context.enabled must be a boolean")
    if not enabled:
        return None
    settings = {key: value for key, value in block.items() if key != "enabled"}
    if settings.keys() - {f.name for f in fields(ContextBudget)}:
        raise ValueError("Unknown m3.qa_context setting")
    if "max_output_tokens" in settings:
        raise ValueError("Configure output tokens in m3.qa_llm, not qa_context")
    settings["max_output_tokens"] = load_m3_runtime_config(config_path).qa_llm.max_tokens
    return ContextBudget(**settings)


@lru_cache(maxsize=4)
def _local_tokenizer(path):
    try:
        from tokenizers import Tokenizer
    except ImportError:
        raise ValueError("Install the m3-budget extra to use a local tokenizer") from None
    tokenizer = Tokenizer.from_file(path)
    # A saved tokenizer may enable truncation/padding. Counting must never
    # truncate the very input whose size we are validating.
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer, hashlib.sha256(Path(path).read_bytes()).hexdigest()


def request_units(messages, tools, budget=None):
    """Include complete history, tool schema, JSON escaping and framing."""
    wire_tools = [
        {"type": "function", "function": {
            "name": t["name"], "description": t.get("description", ""),
            "parameters": t.get("input_schema", {}),
        }} for t in (tools or [])
    ]
    serialized = dumps({"messages": messages, "tools": wire_tools})
    if budget is not None and budget.tokenizer_path:
        tokenizer, _ = _local_tokenizer(budget.tokenizer_path)
        units = math.ceil(len(tokenizer.encode(serialized, add_special_tokens=False).ids)
                          * budget.tokenizer_multiplier)
    else:
        units = len(serialized.encode("utf-8"))
    return units + 256 + 32 * len(messages)


_GROUPS = ("hits", "context_hits", "boundary_hits")
_RANK_FIELDS = {"score", "source_scores", "retrieval_sources", "evidence_role",
                "context_for", "context_distance"}
_CONTENT_FIELDS = ("description", "speech_transcript", "events", "characters", "entities")
_STOP = set("the a an is are was were of to in on and or did does do what where how who "
            "when which this that it its for from with by have has had were video".split())


def terms(text):
    return {word for word in re.findall(r"\w+", text.casefold()) if word not in _STOP}


def passages(body):
    """Whole original list entries or exact description slices, never paraphrase."""
    text = body.get("description", "")
    if isinstance(text, str):
        for match in re.finditer(r"[^.!?。！？\n]+(?:[.!?。！？]+|$)", text):
            yield {"field": "description", "start_char": match.start(),
                   "end_char": match.end(), "text": match.group()}
    for field in _CONTENT_FIELDS[1:]:
        value = body.get(field, [])
        if isinstance(value, list):
            for index, item in enumerate(value):
                yield {"field": field, "index": index, "value": item}


def coherent_units(body, max_items):
    """Keep a clip's description and nearby speech together before supplements.

    Global sentence ranking can retain "preparing the trunk" but discard the
    following counted instructions. Prefer whole short clip narratives; very
    large clips fall back to contiguous source windows, not isolated sentences.
    """
    description = body.get("description", "")
    primary = ([{"field": "description", "start_char": 0, "end_char": len(description),
                 "text": description}] if isinstance(description, str) and description else [])
    transcript = body.get("speech_transcript", [])
    if isinstance(transcript, list):
        primary += [{"field": "speech_transcript", "index": i, "value": item}
                    for i, item in enumerate(transcript)]
    if primary and len(primary) <= max_items and len(dumps(primary).encode()) <= 4096:
        yield 0, primary
    else:
        primary = [p for p in passages(body) if p["field"] in {"description", "speech_transcript"}]
        width = min(4, max_items)
        for offset in range(0, len(primary), width):
            yield 0, primary[offset:offset + width]
    for passage in passages(body):
        if passage["field"] not in {"description", "speech_transcript"}:
            yield (1 if primary else 0), [passage]


def pack_context(messages, tools, question, budget: ContextBudget):
    """Rebuild a bounded view each call; refs only target admitted evidence.

    Versions include the complete projected body and temporal bounds, excluding
    ranking metadata. State is local to this request, so no account/run leakage.
    Nothing is removed from the caller's messages or audit/cache.
    """
    view = deepcopy(messages)
    documents, records = {}, []
    latest_query = ""
    oversized = 0
    for index, message in enumerate(messages):
        if message.get("role") != "tool" or not isinstance(message.get("content"), str):
            continue
        try:
            payload = json.loads(message["content"])
        except (ValueError, TypeError):
            continue
        if not isinstance(payload, dict) or not all(isinstance(payload.get(g), list) for g in _GROUPS):
            continue
        latest_query = str(payload.get("query", latest_query))
        occurrences = []
        for group in _GROUPS:
            for rank, hit in enumerate(payload[group]):
                if not isinstance(hit, dict) or not hit.get("node_id"):
                    continue
                body = {k: v for k, v in hit.items() if k not in _RANK_FIELDS}
                encoded = dumps(body).encode("utf-8")
                if len(encoded) > budget.max_node_bytes:
                    oversized += 1
                    continue
                version = hashlib.sha256(encoded).hexdigest()
                if version not in documents:
                    documents[version] = {
                        "id": f"e{len(documents) + 1}", "body": body,
                        "first": index, "rank": rank, "passages": [],
                    }
                documents[version]["rank"] = min(documents[version]["rank"], rank)
                relation = {k: hit[k] for k in ("context_for", "context_distance") if k in hit}
                occurrences.append((version, group, relation))
        records.append((index, payload, occurrences))

    notice = ("Budgeted original excerpts, not complete memory. Omitted evidence is not "
              "evidence of absence or a complete count. Evidence IDs refer only to this "
              "request. Search a specific missing fact to select different excerpts.")
    selected = {}

    def render():
        for index, payload, occurrences in records:
            refs, bodies = [], {}
            for version, role, relation in occurrences:
                if version not in selected:
                    continue
                doc = documents[version]
                refs.append({"id": doc["id"], "role": role, **relation})
                if doc["first"] == index:
                    bodies[doc["id"]] = selected[version]
            packet = {
                "query": payload.get("query", ""),
                "retrieved_nodes": sum(len(payload[g]) for g in _GROUPS),
                "omitted_nodes": (sum(len(payload[g]) for g in _GROUPS)
                                  - sum(1 for version, _, _ in occurrences if version in selected)),
                "evidence_refs": refs, "evidence": bodies, "budgeted": True,
            }
            if index == records[0][0]:
                packet["notice"] = notice
            view[index]["content"] = dumps(packet)
        return request_units(view, tools, budget)

    original_units = request_units(messages, tools, budget)
    used = render()
    if used > budget.input_limit:
        raise LLMCallError(
            f"M3 non-evidence input exceeds budget: estimated={used} limit={budget.input_limit}; "
            "request not sent"
        )
    query_terms = terms(question)
    recent_terms = terms(latest_query)
    ranked = []
    for version, doc in documents.items():
        for index, (phase, bundle) in enumerate(coherent_units(doc["body"], budget.max_passages_per_node)):
            if index >= budget.max_scan_passages_per_node:
                break
            content = " ".join(p.get("text") or dumps(p.get("value")) for p in bundle)
            words = terms(content)
            score = (3 * len(words & query_terms) / max(1, len(query_terms))
                     + len(words & recent_terms) / max(1, len(recent_terms))
                     + .2 / (1 + doc["rank"]))
            ranked.append((phase, score, version, bundle))
    # Primary narratives get capacity before redundant entity/event descriptions.
    ranked.sort(key=lambda item: (item[0], -item[1]))
    admitted = 0
    checks = 0
    for phase, _, version, bundle in ranked:
        # Bound packing CPU rather than chasing the last few tokens with
        # hundreds of full-request tokenizer passes. Final rendering still
        # verifies the complete request.
        if checks >= budget.max_admission_checks or budget.input_limit - used < 128:
            break
        doc = documents[version]
        new = version not in selected
        if phase and new:
            continue
        if new and len(selected) >= budget.max_selected_nodes:
            continue
        if len(bundle) + (0 if new else len(selected[version]["passages"])) > budget.max_passages_per_node:
            continue
        if new:
            selected[version] = {
                **{k: v for k, v in doc["body"].items() if k not in _CONTENT_FIELDS},
                "content_sha256": version, "passages": [],
            }
        selected[version]["passages"].extend(bundle)
        checks += 1
        cost = render()
        if cost > budget.input_limit:
            del selected[version]["passages"][-len(bundle):]
            if new:
                del selected[version]
        else:
            used = cost
            admitted += len(bundle)
    used = render()
    assert used <= budget.input_limit
    occurrences = sum(len(record[2]) for record in records)
    return view, {
        "counter": ("local_tokenizer_serialized_json_plus_framing" if budget.tokenizer_path
                    else "serialized_utf8_bytes_plus_framing"),
        "tokenizer_sha256": (_local_tokenizer(budget.tokenizer_path)[1]
                             if budget.tokenizer_path else None),
        "estimated_input_tokens": used, "input_limit": budget.input_limit,
        "unpacked_input_units": original_units,
        "retrieved_occurrences": occurrences, "unique_evidence_versions": len(documents),
        "repeated_occurrences": occurrences - len(documents),
        "selected_nodes": len(selected), "selected_passages": admitted,
        "admission_checks": checks,
        "admission_limit_reached": checks >= budget.max_admission_checks,
        "oversized_node_occurrences": oversized,
        "selected_sources": [
            {"id": documents[version]["id"], "node_id": doc.get("node_id"),
             "video_id": doc.get("video_id"), "clip_id": doc.get("clip_id"),
             "content_sha256": version,
             "ranges": [{k: v for k, v in p.items() if k not in {"text", "value"}}
                        for p in doc["passages"]]}
            for version, doc in selected.items()
        ],
    }
