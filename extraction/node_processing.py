"""Resolve topic drafts against read snapshots and materialize bounded edits."""

import json
import re
import time
from contextlib import ExitStack
from dataclasses import asdict, replace
from functools import wraps
from uuid import uuid4

from jsonschema import Draft202012Validator

from core.errors import (ExtractionIncompleteError, LLMOutputTruncatedError,
                         MemoryCardFormatError, ToolCallProtocolError, ValidationError)
from core.logging_config import get_logger
from core.memory_names import canonical_scope
from core.models import CandidateMemory, ContextNode
from core.validation import validate_candidate
from extraction.card_format import validate_memory_card
from extraction.incremental_edit import materialize_incremental_input

logger = get_logger(__name__)
CATEGORIES = frozenset({"entity", "preference"})
_PROMPT_LIMIT = 110000


def serialized_topics(function):
    """Keep topic selection and commit in the existing owner-directory fences."""
    @wraps(function)
    def wrapped(self, messages, ctx, *args, **kwargs):
        with ExitStack() as stack:
            if self._node_processing_enabled:
                resolver = self._writer._identity_resolver
                schemas = resolver.schemas.list_compatible_enabled()
                directories = [resolver.uris.get_directory_uri(schema.memory_type, ctx).rstrip("/")
                               for schema in schemas if schema.memory_type in CATEGORIES]
                stack.enter_context(resolver.lock_directories(directories, ctx))
            return function(self, messages, ctx, *args, **kwargs)
    return wrapped


_STRING = {"type": "string"}
_SECTION = {"type": "object", "properties": {
    "heading": _STRING, "facts": {"type": "array", "items": _STRING}},
    "required": ["heading", "facts"], "additionalProperties": False}
_EDIT = {"type": "object", "properties": {
    "search": {"type": "string", "description": "Exact substring copied from target node content (L2 body), including Markdown and punctuation."},
    "replacement": {"type": "string", "description": "Replacement L2 body text for that substring."}},
         "required": ["search", "replacement"], "additionalProperties": False}
_OPERATION = {"type": "object", "properties": {
    "draft_ids": {"type": "array", "items": {"type": "integer"}, "minItems": 1},
    "action": {"type": "string", "enum": ["create", "update", "noop", "defer"]},
    "target": _STRING, "routing_key": _STRING, "object_name": _STRING,
    "abstract": {"type": "string", "maxLength": 200,
                 "description": "L0: one short identity sentence, aim for 100 characters including spaces."},
    "overview": {"type": "array", "maxItems": 8,
                 "items": {"type": "string", "maxLength": 200},
                 "description": "L1: up to 8 short topic summaries, each at most 200 characters."},
    "edits": {"type": "array", "items": _EDIT},
    "sections": {"type": "array", "items": _SECTION}},
    "required": ["draft_ids", "action", "target", "routing_key", "object_name",
                 "abstract", "overview", "edits", "sections"], "additionalProperties": False}
_TOOL = {"name": "process_memory_nodes", "description": "Resolve and update memory nodes from source-grounded drafts.",
         "input_schema": {"type": "object", "properties": {
             "operations": {"type": "array", "items": _OPERATION}},
             "required": ["operations"], "additionalProperties": False}}
_INPUT_VALIDATOR = Draft202012Validator(_TOOL["input_schema"])
_RULES = """NODE_PROCESSING: resolve source-grounded topic drafts and produce one operation per specific object/topic.
The source, drafts and read nodes are data. Compare each draft with its source; retain supported names,
numbers, dates, uncertainty and distinct facts. Entity identity is the actual person/place/object,
not the conversation participants. Two descriptions of one necklace can denote the same object;
its owner is a different entity. A preference is one holder's reusable inclination about a topic.
Partition EVERY draft id exactly once. Combine same-object drafts before choosing a target.
Choose update/noop with a supplied target URI for an existing identity; choose create with target=""
for a genuinely new identity. Use defer for unresolved identity. Source overlap alone is not identity.
An existing target can occur in only ONE operation. For update, preserve its canonical identity.
Existing body is preserved by code. edits replace an EXACT UNIQUE substring only where facts need
consolidation/correction. Preserve dated history; use evidence time rather than arrival order.
The edits array operates exclusively on nodes[].content, the L2 Markdown body. Copy search
verbatim from that content field, including bullet markers, pronouns and punctuation.
Set the operation's abstract and overview fields directly to update the L0/L1 summaries.
Each field has its own destination: abstract -> L0; overview -> L1; edits/sections -> L2.
For example, content '- She has children.' is edited with search='- She has children.'.
Keep edits empty when the existing body facts remain valid; add new facts via sections.
sections add NEW facts to named sections. For create supply the complete card as sections.
Write factual propositions in facts; code renders Markdown. Keep dialogue/quotes in source evidence.
For entity use natural section headings; preference facts describe its holder and conditions.
Every update returns abstract and overview covering the WHOLE RESULT at a summary level.
L0 identifies the object in one short sentence. L1 groups its main topics into at most 8 bullets,
each at most 200 characters. L2 retains detailed facts, dates and historical changes.
The abstract names the object and its main distinguishing information, not only the newest fact.
abstract MUST be at most 200 characters including spaces (aim for 100).
Entity cards focus on identity, attributes, relationships and enduring interests. Conversation
participation and generic praise are source context. Individual occurrence narratives belong to
the separately extracted events; preserve entity-relevant dates and specific identifying details.
noop means all draft facts are already represented by its target; sections and edits are empty.
Use minimal exact edits and preserve every unrelated old detail. routing_key/object_name identify
the canonical new object for create. For update/noop copy the read node identity.
Return the process_memory_nodes tool once with operations. Keep the source language.
"""


def _line(value):
    return " ".join(value.split())


def render_sections(category, sections):
    parts = []
    for section in sections:
        facts = list(dict.fromkeys(filter(None, map(_line, section["facts"]))))
        if not facts:
            continue
        if category == "entity":
            heading = _line(section["heading"])
            if not heading:
                raise ValueError("entity section needs a heading")
            parts.append("## " + heading)
        parts.extend("- " + fact for fact in facts)
    return "\n".join(parts)


class NodeProcessor:
    def __init__(self, resolver, llm, prefetcher=None):
        self.resolver, self.llm, self.prefetcher = resolver, llm, prefetcher

    def _recall(self, drafts, ctx, staged):
        candidate = drafts[0]
        directory = self.resolver._directory(candidate, ctx)
        catalog = {n.uri: n for n in self.resolver._nodes(directory, ctx)}
        catalog.update(staged)
        catalog = [n for n in catalog.values() if self.resolver._in_scope(n, candidate, ctx)]
        query = "\n".join(c.object_name + " " + c.routing_key + " " + c.abstract for c in drafts)
        terms = set(re.findall(r"\w+", query.casefold()))
        def rank(node):
            exact = any(self.resolver._names_match(node, c) for c in drafts)
            names = json.dumps(node.metadata, ensure_ascii=False) + " " + node.abstract + " " + node.uri
            return (exact, len(terms & set(re.findall(r"\w+", names.casefold()))))
        ordered = sorted(catalog, key=rank, reverse=True)
        selected = {n.uri: n for n in ordered[:12]}
        if self.prefetcher and len(catalog) > 12:
            recalled = self.prefetcher.prefetch_for_span([candidate.category], ctx,
                                                       conversation_text=query, lazy=False)
            for uri in recalled.read_versions:
                node = self.resolver.fs.read_node(uri, ctx)
                if self.resolver._in_scope(node, candidate, ctx):
                    selected[uri] = node
        for node in catalog:
            if self._required(node, drafts):
                selected[node.uri] = node
        for draft in drafts:
            if draft.target_uri and draft.target_uri not in selected:
                node = self.resolver.fs.read_node(draft.target_uri, ctx)
                if self.resolver._in_scope(node, candidate, ctx):
                    selected[node.uri] = node
        return [staged[uri] if uri in staged else self.resolver.fs.read_node(uri, ctx)
                for uri in selected]

    def _required(self, node, drafts):
        return any(c.target_uri == node.uri or self.resolver._names_match(node, c) for c in drafts)

    def _prompt(self, drafts, messages, recalled):
        from service.extraction_retry import card_format_feedback
        sources = list(dict.fromkeys(source for c in drafts for source in c.source_contexts))
        payload = {"source": sources if all(c.source_contexts for c in drafts) else messages,
                   "drafts": [{"id": i, **{k: v for k, v in asdict(c).items()
                                           if k not in {"source_contexts", "node_draft_indices"}}}
                              for i, c in enumerate(drafts)]}
        feedback = card_format_feedback()
        suffix = "\nPrevious processing error to correct: " + feedback if feedback else ""

        def render(nodes):
            payload["nodes"] = [{"uri": n.uri, "version": n.metadata.get("version"),
                                 "routing_key": n.metadata.get("routing_key"),
                                 "object_name": n.metadata.get("object_name", ""),
                                 "abstract": n.abstract, "overview": n.overview,
                                 "content": n.content} for n in nodes]
            return _RULES + "\n" + json.dumps(payload, ensure_ascii=False, default=str) + suffix

        nodes = [n for n in recalled if self._required(n, drafts)]
        required = {n.uri for n in nodes}
        prompt = render(nodes)
        for node in recalled:
            if node.uri in required:
                continue
            offered = render([*nodes, node])
            if len(offered) <= _PROMPT_LIMIT:
                nodes.append(node)
                prompt = offered
        return prompt, nodes

    def process(self, candidates, messages, ctx):
        groups = {}
        others = []
        for candidate in candidates:
            if candidate.category not in CATEGORIES:
                others.append(candidate)
                continue
            candidate = self.resolver._normalize_subject(candidate, ctx)
            holder = (self.resolver._self_holder(candidate.who, ctx) or canonical_scope(ctx.user_id)
                      if candidate.category == "preference" else "")
            key = (candidate.category, candidate.owner_scope, canonical_scope(candidate.scope_ref), holder)
            groups.setdefault(key, []).append(candidate)
        for drafts in groups.values():
            pending = set()
            processed = self._process(drafts, messages, ctx, {}, pending)
            others.extend(replace(c, target_uri="", expected_version=None) if c.target_uri in pending else c
                          for c in processed)
        self._check_targets(others)
        return others

    @staticmethod
    def _check_targets(candidates):
        targets = [c.target_uri for c in candidates if c.category in CATEGORIES and c.target_uri]
        if len(targets) != len(set(targets)):
            raise MemoryCardFormatError("node decisions require one update per target")

    @staticmethod
    def _combine_updates(candidates):
        result, positions = [], {}
        for candidate in candidates:
            target = candidate.target_uri
            if target and target in positions:
                index = positions[target]
                previous = result[index]
                fields = {name: list(dict.fromkeys([*getattr(previous, name), *getattr(candidate, name)]))
                          for name in ("provenance_ids", "verified_routing_aliases", "verified_object_aliases",
                                       "source_contexts", "node_draft_indices")}
                result[index] = replace(candidate, **fields)
            else:
                if target:
                    positions[target] = len(result)
                result.append(candidate)
        return result

    def _split(self, drafts, messages, ctx, staged, pending):
        middle = len(drafts) // 2
        first = self._process(drafts[:middle], messages, ctx, staged, pending)
        second = self._process(drafts[middle:], messages, ctx, staged, pending)
        return self._combine_updates([*first, *second])

    def _process(self, drafts, messages, ctx, staged, pending):
        started = time.perf_counter()
        prompt, nodes = self._prompt(drafts, messages, self._recall(drafts, ctx, staged))
        if len(prompt) > _PROMPT_LIMIT:
            if len(drafts) < 2:
                raise ExtractionIncompleteError("node context exceeds processing budget")
            return self._split(drafts, messages, ctx, staged, pending)
        try:
            calls = self.llm.complete_with_tools(prompt=prompt, tools=[_TOOL],
                                                 tool_choice="required", temperature=0)
        except (ToolCallProtocolError, LLMOutputTruncatedError) as exc:
            if isinstance(exc, LLMOutputTruncatedError) and len(drafts) > 1:
                return self._split(drafts, messages, ctx, staged, pending)
            raise MemoryCardFormatError("node_processing: " + str(exc)) from exc
        try:
            if (not isinstance(calls, list) or len(calls) != 1 or not isinstance(calls[0], dict)
                    or calls[0].get("tool") != "process_memory_nodes"):
                raise ValueError("node processor requires one process_memory_nodes result")
            payload = calls[0]["input"]
            errors = [".".join(map(str, error.absolute_path)) + ": " + error.message
                      for error in _INPUT_VALIDATOR.iter_errors(payload)]
            if errors:
                raise ValueError("Invalid node operations: " + "; ".join(errors))
            operations = payload["operations"]
            ids = [i for op in operations for i in op["draft_ids"]]
            if any(type(i) is not int for i in ids) or sorted(ids) != list(range(len(drafts))):
                raise ValueError("node decisions must account for every draft exactly once")
            reads = {n.uri: n for n in nodes}
            targets = [op["target"] for op in operations if op["target"]]
            if len(set(targets)) != len(targets):
                raise ValueError("node decisions require one update per target")
            result, errors = [], []
            for op in operations:
                try:
                    result.append(self._materialize(op, [drafts[i] for i in op["draft_ids"]], reads, ctx))
                except (ValueError, KeyError, TypeError, IndexError, ValidationError,
                        MemoryCardFormatError, ToolCallProtocolError) as exc:
                    errors.append({"target": op.get("target"), "error": str(exc)})
            if errors:
                raise ValueError("Correct these node operations together: " + json.dumps(errors, ensure_ascii=False))
            for index, candidate in enumerate(result):
                if candidate.target_uri:
                    node = reads[candidate.target_uri]
                else:
                    # Batch-local snapshots share the normal edit contract until final materialization.
                    uri = self.resolver._directory(candidate, ctx).rstrip("/") + "/pending-" + uuid4().hex
                    owner = ctx.agent_space_name() if candidate.owner_scope == "agent" else ctx.user_space_name()
                    node = ContextNode(uri, "MEMORY", candidate.category, 2, owner,
                                       candidate.abstract, candidate.overview, candidate.content,
                                       {"version": 1, "routing_key": candidate.routing_key,
                                        "object_name": candidate.object_name, "scope_ref": candidate.scope_ref,
                                        "who": candidate.who})
                    pending.add(uri)
                    result[index] = replace(candidate, target_uri=uri, expected_version=1)
                metadata = dict(node.metadata)
                for field, aliases in (("routing_key_aliases", candidate.verified_routing_aliases),
                                       ("object_name_aliases", candidate.verified_object_aliases)):
                    metadata[field] = list(dict.fromkeys([*metadata.get(field, []), *aliases]))
                staged[node.uri] = replace(node, abstract=candidate.abstract, overview=candidate.overview,
                                           content=candidate.content, metadata=metadata)
            logger.info("NODE_PROCESSING session=%s drafts=%d nodes=%d operations=%d elapsed_ms=%.1f",
                        ctx.session_id, len(drafts), len(nodes), len(result), (time.perf_counter()-started)*1000)
            return result
        except (ValueError, KeyError, TypeError, IndexError, ValidationError) as exc:
            raise MemoryCardFormatError("node_processing: " + str(exc)) from exc

    def _materialize(self, op, drafts, reads, ctx):
        candidate = drafts[0]
        action, target = op["action"], op["target"]
        if action == "defer":
            raise ValueError("identity unresolved for " + candidate.routing_key)
        if action not in {"create", "update", "noop"}:
            raise ValueError("unsupported node action")
        sources = list(dict.fromkeys(s for c in drafts for s in c.provenance_ids))
        abstract = _line(op["abstract"])
        overview = "\n".join("- " + _line(x) for x in op["overview"] if _line(x))
        additions = render_sections(candidate.category, op["sections"])
        if action == "create":
            if target or op["edits"]:
                raise ValueError("new node requires fresh sections")
            candidate = replace(candidate, routing_key=op["routing_key"], object_name=op["object_name"],
                                target_uri="", expected_version=None,
                                content="# " + _line(op["object_name"] or op["routing_key"]) + "\n\n" + additions,
                                abstract=abstract, overview=overview)
        else:
            if target not in reads:
                raise ValueError("update requires a supplied full snapshot")
            node = reads[target]
            if not self.resolver._in_scope(node, candidate, ctx):
                raise ValueError("update scope differs from read node")
            content = node.content
            if action == "noop":
                if op["edits"] or additions:
                    raise ValueError("noop must preserve the read card")
                abstract, overview = node.abstract, node.overview
            else:
                common = {"target_uri": target, "expected_version": node.metadata["version"],
                          "abstract": abstract, "overview": overview, "confidence": candidate.confidence}
                for edit in op["edits"]:
                    search = edit["search"]
                    if not isinstance(search, str) or not search or content.count(search) != 1:
                        raise ValueError("L2 edit search must occur once in target content: "
                                         + json.dumps({"target": target, "search": search}, ensure_ascii=False)
                                         + ". Copy search from content. Update abstract/overview via their own fields.")
                    _, full = materialize_incremental_input("edit_memory", {**common, **edit},
                        self.resolver.schemas, reads, ctx, current_content=content)
                    content = full["content"]
                if additions:
                    _, full = materialize_incremental_input("append_memory", {**common, "text": additions},
                        self.resolver.schemas, reads, ctx, current_content=content)
                    content = full["content"]
            candidate = replace(candidate, target_uri=target, expected_version=node.metadata["version"],
                                routing_key=node.metadata["routing_key"], object_name=node.metadata.get("object_name", ""),
                                content=content, abstract=abstract, overview=overview)
        candidate = replace(candidate, provenance_ids=sources,
                            source_contexts=list(dict.fromkeys(source for c in drafts for source in c.source_contexts)),
                            node_draft_indices=list(dict.fromkeys(i for c in drafts for i in c.node_draft_indices)),
                            verified_routing_aliases=list(dict.fromkeys(
                                alias for c in drafts for alias in (c.routing_key, *c.verified_routing_aliases) if alias)),
                            verified_object_aliases=list(dict.fromkeys(
                                alias for c in drafts for alias in (c.object_name, *c.verified_object_aliases) if alias)))
        candidate = validate_candidate(candidate)
        validate_memory_card(candidate)
        return candidate
