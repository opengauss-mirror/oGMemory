"""Resolve topic drafts against read snapshots and materialize bounded edits."""

import json
import re
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack
from contextvars import copy_context
from dataclasses import asdict, replace
from functools import wraps
from uuid import NAMESPACE_URL, uuid5

from jsonschema import Draft202012Validator

from core.errors import (
    ExtractionIncompleteError,
    LLMOutputTruncatedError,
    MemoryCardFormatError,
    NodeNotFoundError,
    ToolCallProtocolError,
    ValidationError,
)
from core.logging_config import get_logger
from core.memory_names import canonical_memory_key, canonical_scope
from core.models import ContextNode
from core.validation import validate_candidate
from extraction.card_format import validate_memory_card
from extraction.incremental_edit import materialize_incremental_input
from providers.llm_observation import observe_call

logger = get_logger(__name__)
CATEGORIES = frozenset({"entity", "preference"})
_PROMPT_LIMIT = 110000
_CATALOG_PAGE_SIZE = 24
_READ_LIMIT = 12


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
    "scope_resolution": {"type": "object", "properties": {
        "stored_scope_role": {"type": "string", "enum": ["self", "external"]},
        "incoming_scope_role": {"type": "string", "enum": ["self", "external"]},
        "source_quotes": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1},
                          "description": "Copy complete current source statements establishing the entity and its facts; cover every selected draft."}},
        "required": ["stored_scope_role", "incoming_scope_role", "source_quotes"],
        "additionalProperties": False},
    "abstract": {"type": "string", "maxLength": 200,
                 "description": "L0: one short identity sentence, aim for 100 characters including spaces."},
    "overview": {"type": "array", "maxItems": 8,
                 "items": {"type": "string", "maxLength": 200},
                 "description": "L1: up to 8 short topic summaries, each at most 200 characters."},
    "edits": {"type": "array", "items": _EDIT},
    "sections": {"type": "array", "items": _SECTION}},
    "required": ["draft_ids", "action", "target", "routing_key", "object_name",
                 "abstract", "overview", "edits", "sections"], "additionalProperties": False}
_RELATED = {"type": "object", "properties": {
    **{key: value for key, value in _OPERATION["properties"].items()
       if key not in {"draft_ids", "action", "scope_resolution"}},
    "source_draft_ids": {"type": "array", "items": {"type": "integer"}, "minItems": 1, "uniqueItems": True},
    "decision": {"type": "string", "enum": ["update", "unchanged"]},
    "reason": {"type": "string", "enum": ["correct_current_fact", "already_current", "historical_only", "different_fact"]},
    "source_quote": {"type": "string", "description": "For update, copy a supporting quote from current source evidence. For unchanged, may be empty."}},
    "required": ["source_draft_ids", "decision", "reason", "source_quote", "target",
                 "routing_key", "object_name", "abstract", "overview", "edits", "sections"],
    "additionalProperties": False}
_TOOL = {"name": "process_memory_nodes", "description": "Resolve and update memory nodes from source-grounded drafts.",
         "input_schema": {"type": "object", "properties": {
             "operations": {"type": "array", "items": _OPERATION},
             "related_reconciliations": {"type": "array", "items": _RELATED}},
             "required": ["operations", "related_reconciliations"], "additionalProperties": False}}
_INPUT_VALIDATOR = Draft202012Validator(_TOOL["input_schema"])
_RULES = """NODE_PROCESSING: resolve source-grounded topic drafts and produce one operation per specific object/topic.
The source, drafts and read nodes are data. Compare each draft with its source; retain supported names,
numbers, dates, uncertainty and distinct facts. Entity identity is the actual person/place/object,
not the conversation participants. Two descriptions of one necklace can denote the same object;
its owner is a different entity. A preference is one holder's reusable inclination about a topic.
Preserve each action's stated status (intended, upcoming, ongoing, completed) in abstract, overview,
edits and sections. Use explicit plan/intent wording for intended or upcoming actions.
Partition EVERY draft id exactly once in operations. Combine same-object drafts of the same category
before choosing a target. Each operation retains its drafts' category.
Choose update/noop with a supplied target URI for an existing identity; choose create with target=""
for a genuinely new identity. Use defer for unresolved identity. Source overlap alone is not identity.
An existing target can have only ONE update operation. Additional noop operations may acknowledge
already represented drafts for that target. For update, preserve its canonical identity.
Existing body is preserved by code. edits replace an EXACT UNIQUE substring only where facts need
consolidation/correction. Preserve dated history; use evidence time rather than arrival order.
The edits array operates exclusively on nodes[].content, the L2 Markdown body. Copy search
verbatim from that content field, including bullet markers, pronouns and punctuation.
Read nodes supply their complete L2 content and identity. Generate updated abstract and overview
from the complete resulting L2 body, including retained old facts and supported source changes.
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
For noop or related unchanged, use abstract="" and overview=[]; code retains the stored summaries.
Use minimal exact edits and preserve every unrelated old detail. routing_key/object_name identify
the canonical new object for create. For update/noop copy the read node identity.
primary_candidate_targets lists read candidates in the drafts' own category, owner and scope.
scope_candidate_targets lists same-name entities of the same owner whose scope needs comparison.
For each selected draft, compare the complete source statement, both scopes and the stored card.
A self scope describes the entity itself (including an empty scope); an external scope identifies
its enclosing project, organization or other namespace. To update/noop a scope candidate, supply
scope_resolution with stored_scope_role=self and incoming_scope_role=self only when both scopes
denote the same entity itself. Copy complete supporting source statements into source_quotes,
including their factual assertions and covering every draft. Distinct external namespaces identify separate entities.
If the source establishes a distinct entity, create its own card; use defer for unresolved identity.
Use operations to resolve the drafts against these candidates. A same-category candidate, including
one also listed in related_targets, may be acknowledged in related_reconciliations only with
decision=unchanged and empty edits/sections. For a target used by a
primary operation, source_draft_ids belong to that operation. The acknowledgement consumes no draft again.
related_targets explicitly lists cross-category nodes of the same owner and confirmed subject.
Compare their current facts against this source. For each related target outside
the primary operations, return exactly one related_reconciliations entry. source_draft_ids reference
the supporting drafts without consuming them again.
Use decision=update and reason=correct_current_fact when the source corrects a fact also asserted
as current in this node; copy a supporting source_quote verbatim from current source evidence,
and update its abstract, overview and body together, keeping dated history.
Use decision=unchanged with already_current, historical_only or different_fact for valid content.
For unchanged, keep edits/sections empty to retain the complete read card; source_quote may be empty.
Same scope or similar names select candidates; compare the actual fact, holder, conditions and time.
Related updates retain the read target's category, identity and unrelated facts.
When fixed_target is supplied, return one primary update/noop for that exact target containing all
draft ids; reconcile the source against its latest snapshot. Return related_reconciliations=[].
Return the process_memory_nodes tool once with operations and related_reconciliations. Keep the source language.
"""


def _line(value):
    return " ".join(value.split())


def _text_values(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _text_values(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _text_values(nested)


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

    def _recall(self, drafts, ctx, staged, cache=None):
        cache = cache if cache is not None else {"catalogs": {}, "reads": {}}
        categories = {c.category for c in drafts}
        if any((c.scope_ref or c.category == "entity" and c.object_name)
               and self._own_holder(c, ctx) for c in drafts):
            categories.update(CATEGORIES)
        categories = sorted(category for category in categories
                            if (schema := self.resolver.schemas.get_compatible(category)) and schema.enabled)
        full = {**cache["reads"], **staged}
        for draft in drafts:
            direct = [draft.target_uri, self.resolver.uris.resolve(draft.category, asdict(draft), ctx)]
            if draft.scope_ref and "entity" in categories:
                direct.append(self.resolver.uris.resolve("entity", {
                    "routing_key": canonical_memory_key(draft.scope_ref)}, ctx))
            for uri in filter(None, direct):
                if uri in full:
                    continue
                try:
                    full[uri] = self.resolver.fs.read_node(uri, ctx)
                except NodeNotFoundError:
                    if uri == draft.target_uri:
                        raise
                    continue
        query = "\n".join(c.object_name + " " + c.routing_key + " " + c.abstract for c in drafts)
        if self.prefetcher:
            full.update(self.prefetcher.prefetch_node_snapshots(categories, ctx, query, read_nodes=full))
        cache["reads"].update((uri, node) for uri, node in full.items() if uri not in staged)
        catalog = {}
        for category in categories:
            directory = self.resolver.uris.get_directory_uri(category, ctx).rstrip("/")
            if directory not in cache["catalogs"]:
                cache["catalogs"][directory] = list(self._catalog(directory, ctx, cache["reads"]))
            catalog.update((node.uri, node) for node in cache["catalogs"][directory])
        full = {**cache["reads"], **full}
        catalog.update(full)
        scope_fallbacks = [c for c in drafts if c.category == "entity" and not any(
            self.resolver._in_scope(node, c, ctx)
            and (node.uri == c.target_uri or self.resolver._names_match(node, c))
            for node in catalog.values())]
        def eligible(node):
            return (any(self.resolver._in_scope(node, c, ctx)
                        or self._related_in_scope(node, c, ctx, full) for c in drafts)
                    or any(self._scope_candidate(node, c, ctx) for c in scope_fallbacks))
        catalog = [node for node in catalog.values() if eligible(node)]
        terms = set(re.findall(r"\w+", query.casefold()))
        def rank(node):
            exact = any(self.resolver._names_match(node, c) for c in drafts)
            names = json.dumps(node.metadata, ensure_ascii=False) + " " + node.abstract + " " + node.uri
            return (exact, len(terms & set(re.findall(r"\w+", names.casefold()))))
        selected = {node.uri: node for category in categories
                    for node in sorted((n for n in catalog if n.category == category),
                                       key=rank, reverse=True)[:_READ_LIMIT]}
        selected.update((node.uri, node) for node in catalog if self._required(node, drafts, ctx)
                        or any(self._related_in_scope(node, c, ctx, full) for c in drafts))
        for uri in selected:
            if uri not in full:
                full[uri] = self.resolver.fs.read_node(uri, ctx)
        cache["reads"].update((uri, node) for uri, node in full.items() if uri not in staged)
        return [full[uri] for uri in selected]

    def _catalog(self, directory, ctx, reads=None):
        method = getattr(type(self.resolver.fs), "list_identity_node_summaries_page", None)
        if not callable(method):
            method = getattr(type(self.resolver.fs), "list_node_summaries_page", None)
        if callable(method):
            order_key = getattr(self.resolver.fs, "list_node_order_key", lambda uri: uri)
            after_uri = None
            while True:
                page = list(method(self.resolver.fs, directory, ctx,
                                   limit=_CATALOG_PAGE_SIZE, after_uri=after_uri))
                yield from page
                if len(page) < _CATALOG_PAGE_SIZE:
                    break
                after_uri = order_key(page[-1].uri)
            return
        uris = sorted(self.resolver.fs.list_children(directory, ctx))
        for uri in uris:
            try:
                node = reads.get(uri) if reads is not None else None
                if node is None:
                    node = self.resolver.fs.read_node(uri, ctx)
                    if reads is not None:
                        reads[uri] = node
                yield node
            except NodeNotFoundError:
                continue

    def _own_holder(self, candidate, ctx):
        return (candidate.category != "preference"
                or self.resolver._self_holder(candidate.who, ctx) in {"", canonical_scope(ctx.user_id)})

    def _scope_candidate(self, node, candidate, ctx):
        return (candidate.category == "entity" and node.category == "entity"
                and self.resolver._names_match(node, candidate)
                and self.resolver._in_scope(node, replace(candidate, scope_ref=node.metadata.get("scope_ref", "")), ctx))

    def _related_in_scope(self, node, candidate, ctx, reads=None):
        if (node.category not in CATEGORIES or node.category == candidate.category
                or not self._own_holder(candidate, ctx)):
            return False
        if candidate.category == "preference":
            subject = canonical_scope(candidate.scope_ref)
            names = [node.metadata.get("object_name", ""), *node.metadata.get("object_name_aliases", [])]
            entity_scope = canonical_scope(node.metadata.get("scope_ref", ""))
        else:
            subject = canonical_scope(node.metadata.get("scope_ref", ""))
            names = [candidate.object_name, *candidate.verified_object_aliases]
            entity_scope = canonical_scope(candidate.scope_ref)
            target = (reads or {}).get(candidate.target_uri)
            if target is not None and self.resolver._in_scope(target, candidate, ctx):
                names.extend([target.metadata.get("object_name", ""), *target.metadata.get("object_name_aliases", [])])
        if (not subject or entity_scope not in {"", subject}
                or subject not in {canonical_scope(name) for name in names if name}):
            return False
        counterpart = replace(candidate, category=node.category, who=ctx.user_id,
                              scope_ref=node.metadata.get("scope_ref", ""))
        return self.resolver._in_scope(node, counterpart, ctx)

    def _required(self, node, drafts, ctx):
        return any(c.target_uri == node.uri or (self.resolver._in_scope(node, c, ctx)
                   and self.resolver._names_match(node, c))
                   for c in drafts)

    def _prompt(self, drafts, messages, recalled, ctx, *, fixed_target="", related_target=""):
        reads = {node.uri: node for node in recalled}
        sources = list(dict.fromkeys(source for c in drafts for source in c.source_contexts))
        payload = {"source": sources if all(c.source_contexts for c in drafts) else messages,
                   "drafts": [{"id": i, **{k: v for k, v in asdict(c).items()
                                           if k not in {"source_contexts", "node_draft_indices",
                                                        "node_reconciliation_role", "node_reconciliation_group"}}}
                              for i, c in enumerate(drafts)]}
        rules = _RULES
        if related_target:
            rules += ("This request only reconciles the supplied related_target. "
                      "Return operations=[] and one related_reconciliations entry for it. "
                      "Use the source drafts as evidence without resolving their primary targets.\n")
            payload["related_target"] = related_target
        prefix = rules + "\n" + json.dumps(payload, ensure_ascii=False, default=str)[:-1]
        suffix = ', "fixed_target": ' + json.dumps(fixed_target, ensure_ascii=False) + "}"
        fragments = {"nodes": [], "related_targets": [], "primary_candidate_targets": [],
                     "scope_candidate_targets": []}
        length = len(prefix) + len(suffix) + sum(len(', "' + name + '": []') for name in fragments)
        entries = []
        for node in recalled:
            related = (node.uri == related_target if related_target else
                       any(self._related_in_scope(node, c, ctx, reads) for c in drafts))
            primary = not related_target and any(self.resolver._in_scope(node, c, ctx) for c in drafts)
            encoded = json.dumps({"uri": node.uri, "version": node.metadata.get("version"),
                                  "category": node.category, "scope_ref": node.metadata.get("scope_ref", ""),
                                  "who": node.metadata.get("who"),
                                  "routing_key": node.metadata.get("routing_key"),
                                  "object_name": node.metadata.get("object_name", ""),
                                  "content": node.content}, ensure_ascii=False, default=str)
            uri = json.dumps(node.uri, ensure_ascii=False)
            values = {"nodes": encoded}
            if related:
                values["related_targets"] = uri
            if primary:
                values["primary_candidate_targets"] = uri
            elif not related_target and any(self._scope_candidate(node, c, ctx) for c in drafts):
                values["scope_candidate_targets"] = uri
            entries.append((self._required(node, drafts, ctx) or related, node, values))
        nodes = []
        for required, node, values in sorted(entries, key=lambda item: not item[0]):
            added = sum(len(value) + (2 if fragments[name] else 0) for name, value in values.items())
            if required or length + added <= _PROMPT_LIMIT:
                nodes.append(node)
                length += added
                for name, value in values.items():
                    fragments[name].append(value)
        prompt = prefix + "".join(', "' + name + '": [' + ", ".join(values) + "]"
                                   for name, values in fragments.items()) + suffix
        return prompt, nodes

    def process(self, candidates, messages, ctx, *, checkpoint=None, reconcile=None, target_candidate=None):
        from core.model_stage import collect_stage_results, stage_collector_scope
        if target_candidate is not None:
            return self._reprocess_target(candidates, messages, ctx, target_candidate)
        from extraction.relation_proposals import attach, take
        proposal_sources = [replace(c, metadata=dict(c.metadata)) for c in candidates]
        recall_cache = {"catalogs": {}, "reads": {}}
        groups = {}
        others = []
        preference_subjects = {canonical_scope(candidate.scope_ref) for candidate in candidates
                               if candidate.category == "preference" and candidate.scope_ref
                               and self._own_holder(candidate, ctx)}
        for candidate in candidates:
            if candidate.category not in CATEGORIES:
                others.append(candidate)
                continue
            candidate = self.resolver._normalize_subject(candidate, ctx)
            holder = (self.resolver._self_holder(candidate.who, ctx) or canonical_scope(ctx.user_id)
                      if candidate.category == "preference" else "")
            scope = canonical_scope(candidate.scope_ref)
            subject = (scope if candidate.category == "preference" else
                       next((name for value in [candidate.object_name, *candidate.verified_object_aliases]
                             if (name := canonical_scope(value)) in preference_subjects
                             and scope in {"", name}), ""))
            key = (("subject", candidate.owner_scope, subject) if subject and self._own_holder(candidate, ctx)
                   else (candidate.category, candidate.owner_scope, scope, holder))
            groups.setdefault(key, []).append(candidate)
        chains = []
        for drafts in groups.values():
            recalled = self._recall(drafts, ctx, {}, recall_cache)
            targets = {n.uri for n in recalled}
            matches = [chain for chain in chains if chain[1] & targets]
            batches = [(drafts, recalled)]
            for chain in reversed(matches):
                batches = [*chain[0], *batches]
                targets |= chain[1]
                chains.remove(chain)
            chains.append((batches, targets))
        results = [None] * len(chains)
        failures = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {}

            def advance(index, generator, value=None, error=None):
                try:
                    prompt = generator.throw(error) if error is not None else generator.send(value)
                except StopIteration as done:
                    results[index] = done.value
                    return
                except Exception as exc:
                    failures.append(exc)
                    return
                future = pool.submit(copy_context().run, observe_call, "postprocess", self.llm.complete_with_tools,
                                     prompt=prompt, tools=[_TOOL],
                                     tool_choice="required", temperature=0)
                futures[future] = (index, generator)

            for index, (batches, _) in enumerate(chains):
                advance(index, self._process_chain(batches, messages, ctx, checkpoint, recall_cache))
            def collect_nodes():
                for future in [future for future in futures if future.done()]:
                    index, generator = futures.pop(future)
                    try:
                        value = future.result()
                    except Exception as exc:
                        advance(index, generator, error=exc)
                    else:
                        advance(index, generator, value=value)

            with stage_collector_scope(collect_nodes):
                if reconcile is not None:
                    try:
                        others = reconcile()
                    except Exception as exc:
                        failures.append(exc)
                while futures:
                    wait(futures, timeout=.1, return_when=FIRST_COMPLETED)
                    collect_stage_results()
            if failures:
                raise failures[0]
        for processed in results:
            others.extend(processed)
        self._check_targets(others)
        attach(others, take([*proposal_sources, *others]))
        return others

    def _reprocess_target(self, drafts, messages, ctx, accepted):
        node = self.resolver.fs.read_node(accepted.target_uri, ctx)
        if not self.resolver._in_scope(node, accepted, ctx):
            raise MemoryCardFormatError("node revision requires its accepted owner and subject")
        rebound = [replace(draft, category=accepted.category, owner_scope=accepted.owner_scope,
                           scope_ref=accepted.scope_ref, who=accepted.who,
                           target_uri=node.uri, expected_version=node.metadata["version"],
                           routing_key=node.metadata["routing_key"],
                           object_name=node.metadata.get("object_name", ""),
                           verified_routing_aliases=accepted.verified_routing_aliases,
                           verified_object_aliases=accepted.verified_object_aliases)
                   for draft in drafts]
        generator = self._process(rebound, messages, ctx, {}, set(), None,
                                  recalled=[node], fixed_target=node.uri)
        response = None
        while True:
            try:
                prompt = generator.send(response)
            except StopIteration as done:
                return [replace(candidate, node_reconciliation_role=accepted.node_reconciliation_role,
                                node_reconciliation_group=accepted.node_reconciliation_group)
                        for candidate in done.value]
            response = observe_call("postprocess", self.llm.complete_with_tools, prompt=prompt, tools=[_TOOL],
                                                    tool_choice="required", temperature=0)

    def _process_chain(self, batches, messages, ctx, checkpoint, recall_cache=None):
        staged, pending, result = {}, set(), []
        for drafts, recalled in batches:
            result.extend((yield from self._process(drafts, messages, ctx, staged, pending, checkpoint,
                                                   recalled=recalled if not staged else None,
                                                   recall_cache=recall_cache)))
        return [replace(c, target_uri="", expected_version=None) if c.target_uri in pending else c
                for c in self._combine_updates(result)]

    @staticmethod
    def _check_targets(candidates):
        targets = [c.target_uri for c in candidates if c.category in CATEGORIES and c.target_uri]
        if len(targets) != len(set(targets)):
            raise MemoryCardFormatError("node decisions require one update per target")

    @staticmethod
    def _combine_updates(candidates, *, preferred=None):
        result, positions = [], {}
        for candidate in candidates:
            target = candidate.target_uri
            if target and target in positions:
                index = positions[target]
                previous = result[index]
                fields = {name: list(dict.fromkeys([*getattr(previous, name), *getattr(candidate, name)]))
                          for name in ("provenance_ids", "verified_routing_aliases", "verified_object_aliases",
                                       "source_contexts", "node_draft_indices")}
                result[index] = replace((preferred or {}).get(target, candidate), **fields)
            else:
                if target:
                    positions[target] = len(result)
                result.append(candidate)
        return result

    def _split(self, drafts, messages, ctx, staged, pending, checkpoint, *, recalled=None, fixed_target="",
               recall_cache=None):
        middle = len(drafts) // 2
        result = []
        for batch in (drafts[:middle], drafts[middle:]):
            snapshots = [staged.get(node.uri, node) for node in recalled] if fixed_target else None
            result.extend((yield from self._process(
                batch, messages, ctx, staged, pending, checkpoint,
                recalled=snapshots, fixed_target=fixed_target, recall_cache=recall_cache)))
        return self._combine_updates(result)

    def _split_related(self, drafts, messages, ctx, staged, pending, checkpoint, recalled):
        reads = {node.uri: node for node in recalled}
        related = [node for node in recalled
                   if self._related_in_scope(node, drafts[0], ctx, reads)]
        if not related:
            raise ExtractionIncompleteError("node context exceeds processing budget")
        related_uris = {node.uri for node in related}
        primary = [node for node in recalled if node.uri not in related_uris]
        batch_staged, batch_pending = dict(staged), set(pending)
        result = yield from self._process(
            drafts, messages, ctx, batch_staged, batch_pending, checkpoint, recalled=primary,
        )
        for node in related:
            result.extend((yield from self._process(
                drafts, messages, ctx, batch_staged, batch_pending, checkpoint,
                recalled=[node], related_target=node.uri, scope_reads=reads,
            )))
        staged.update(batch_staged)
        pending.update(batch_pending)
        return result

    def _process(self, drafts, messages, ctx, staged, pending, checkpoint, *, recalled=None, fixed_target="",
                 recall_cache=None, related_target="", scope_reads=None):
        started = time.perf_counter()
        if recalled is None:
            recalled = self._recall(drafts, ctx, staged, recall_cache)
        prompt, nodes = self._prompt(drafts, messages, recalled, ctx, fixed_target=fixed_target,
                                     related_target=related_target)
        from core.model_stage import model_signature
        cached = None
        if checkpoint is not None and len(prompt) <= _PROMPT_LIMIT:
            inputs = {"prompt": prompt, "tool": _TOOL, "model": model_signature(self.llm, temperature=0),
                      "drafts": [asdict(c) for c in drafts], "nodes": [asdict(n) for n in recalled],
                      "schemas": [asdict(schema) for c in sorted(CATEGORIES)
                                  if (schema := self.resolver.schemas.get_compatible(c)) is not None]}
            cached = checkpoint.load_stage("node_processing", inputs)
        from service.extraction_retry import card_format_feedback
        feedback = card_format_feedback()
        request_prompt = prompt + ("\nPrevious processing error to correct: " + feedback if feedback else "")
        if cached is None and len(request_prompt) > _PROMPT_LIMIT:
            if len(drafts) > 1:
                return (yield from self._split(drafts, messages, ctx, staged, pending, checkpoint,
                                              recalled=recalled, fixed_target=fixed_target, recall_cache=recall_cache))
            if not fixed_target and not related_target:
                return (yield from self._split_related(drafts, messages, ctx, staged, pending, checkpoint, recalled))
            raise ExtractionIncompleteError("node context exceeds processing budget")
        try:
            calls = cached if cached is not None else (yield request_prompt)
        except (ToolCallProtocolError, LLMOutputTruncatedError) as exc:
            if isinstance(exc, LLMOutputTruncatedError) and len(drafts) > 1:
                return (yield from self._split(drafts, messages, ctx, staged, pending, checkpoint,
                                              recalled=recalled, fixed_target=fixed_target, recall_cache=recall_cache))
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
            expected = [] if related_target else list(range(len(drafts)))
            if any(type(i) is not int for i in ids) or sorted(ids) != expected:
                counts = Counter(ids)
                raise ValueError("node decisions must account for every draft exactly once. "
                                 "Use integer draft_ids and preserve all source facts: " +
                                 json.dumps({"expected": expected,
                                             "missing": [i for i in expected if i not in counts],
                                             "repeated": sorted(i for i, count in counts.items() if count > 1),
                                             "out_of_range": sorted(i for i in counts if i not in expected)},
                                            ensure_ascii=False, separators=(",", ":")))
            reads = {n.uri: n for n in nodes}
            targets = [op["target"] for op in operations if op["target"]]
            update_targets = [op["target"] for op in operations if op["action"] == "update" and op["target"]]
            if len(set(update_targets)) != len(update_targets):
                duplicates = [target for target, count in Counter(update_targets).items() if count > 1]
                raise ValueError("node decisions require one update per target. "
                                 "Combine same-object drafts into one complete operation preserving all facts "
                                 "and whole-result abstract/overview: " +
                                 json.dumps([{"target": target, "operations": [
                                     {"draft_ids": op["draft_ids"], "action": op["action"]}
                                     for op in operations if op["target"] == target]}
                                     for target in duplicates], ensure_ascii=False, separators=(",", ":")))
            related = payload.get("related_reconciliations", [])
            offered = ({related_target} if related_target else
                       {n.uri for n in nodes if any(self._related_in_scope(n, c, ctx, reads) for c in drafts)})
            related_targets = [op["target"] for op in related]
            required_related = offered - set(targets)
            primary_drafts = {}
            for op in operations:
                if op["target"]:
                    primary_drafts.setdefault(op["target"], set()).update(op["draft_ids"])
            unchanged_candidates = {op["target"] for op in related if (op["target"] not in primary_drafts
                                         or set(op["source_draft_ids"]) <= primary_drafts[op["target"]])
                                    and self._unchanged_primary_candidate(op, drafts, reads, ctx)}
            missing = required_related - set(related_targets)
            unexpected = set(related_targets) - required_related - unchanged_candidates
            duplicates = sorted(uri for uri, count in Counter(related_targets).items() if count > 1)
            if duplicates or missing or unexpected:
                raise ValueError("related decisions must cover every supplied related target once: " +
                                 json.dumps({"related_decisions": [
                                                {**{key: op[key] for key in
                                                    ("target", "decision", "reason", "source_draft_ids")},
                                                 "edits_count": len(op["edits"]),
                                                 "sections_count": len(op["sections"]),
                                                 "source_quote_chars": len(op["source_quote"])}
                                                for op in related if op["target"] in unexpected or op["target"] in duplicates],
                                             "missing": sorted(missing),
                                             "unexpected": sorted(unexpected),
                                             "duplicates": duplicates,
                                             "primary_targets": sorted(targets),
                                             "primary_operations": [{key: op[key] for key in
                                                ("target", "action", "draft_ids")} for op in operations]}, ensure_ascii=False))
            if fixed_target and (len(operations) != 1 or targets != [fixed_target]
                                 or operations[0]["action"] not in {"update", "noop"} or related):
                raise ValueError(f"node revision requires one operation for its fixed target: {fixed_target}")
            group = uuid5(NAMESPACE_URL, json.dumps({
                "account": ctx.account_id, "owner": (ctx.agent_space_name() if drafts[0].owner_scope == "agent"
                                                       else ctx.user_space_name()),
                "scope": canonical_scope(drafts[0].scope_ref), "source": messages,
                "provenance": sorted({source for c in drafts for source in c.provenance_ids}),
            }, sort_keys=True, ensure_ascii=False, default=str)).hex
            result, errors, updates = [], [], {}
            for op in operations:
                try:
                    selected = [drafts[i] for i in op["draft_ids"]]
                    if len({c.category for c in selected}) != 1:
                        raise ValueError("primary operation drafts must share their category")
                    candidate = self._materialize(op, selected, reads, ctx, messages=messages)
                    candidate = replace(candidate, node_reconciliation_group=group)
                    result.append(candidate)
                    if op["action"] == "update":
                        updates[candidate.target_uri] = candidate
                except (ValueError, KeyError, TypeError, IndexError, ValidationError,
                        MemoryCardFormatError, ToolCallProtocolError) as exc:
                    errors.append({"target": op.get("target"), "error": str(exc)})
            result = self._combine_updates(result, preferred=updates)
            for op in related:
                try:
                    candidate = self._materialize_related(
                        op, drafts, messages, scope_reads if related_target else reads, ctx,
                        allow_same_category=op["target"] in unchanged_candidates)
                    if candidate is not None:
                        result.append(replace(candidate, node_reconciliation_group=group))
                except (ValueError, KeyError, TypeError, IndexError, ValidationError,
                        MemoryCardFormatError, ToolCallProtocolError) as exc:
                    errors.append({"target": op.get("target"), "error": str(exc)})
            if errors:
                raise ValueError("Correct these node operations together: " + json.dumps(errors, ensure_ascii=False))
            if checkpoint is not None and cached is None:
                checkpoint.save_stage("node_processing", inputs, calls)
            for index, candidate in enumerate(result):
                if candidate.target_uri:
                    node = reads[candidate.target_uri]
                else:
                    # Batch-local snapshots share the normal edit contract until final materialization.
                    uri = self.resolver._directory(candidate, ctx).rstrip("/") + "/pending-" + uuid5(NAMESPACE_URL, prompt + str(index)).hex
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

    def _unchanged_primary_candidate(self, op, drafts, reads, ctx):
        if (op["decision"] != "unchanged" or op["reason"] == "correct_current_fact"
                or op["edits"] or op["sections"]):
            return False
        node = reads.get(op["target"])
        ids = op["source_draft_ids"]
        return (node is not None and bool(ids) and all(type(index) is int and 0 <= index < len(drafts)
                and self.resolver._in_scope(node, drafts[index], ctx) for index in ids))

    def _materialize_related(self, op, drafts, messages, reads, ctx, *, allow_same_category=False):
        ids = op["source_draft_ids"]
        if any(type(index) is not int or not 0 <= index < len(drafts) for index in ids):
            raise ValueError("related decision requires supplied source draft ids")
        sources = [drafts[index] for index in ids]
        node = reads[op["target"]]
        if (not all(self._related_in_scope(node, source, ctx, reads) for source in sources)
                and not (allow_same_category and self._unchanged_primary_candidate(op, drafts, reads, ctx))):
            raise ValueError("related decision requires the same owner and confirmed subject scope")
        if op["decision"] == "unchanged":
            if op["reason"] == "correct_current_fact" or op["edits"] or op["sections"]:
                raise ValueError("unchanged related decision must retain the complete current card")
            return None
        evidence = [source for draft in sources for source in draft.source_contexts]

        if not all(source.source_contexts for source in sources):
            evidence.extend(_text_values(messages))
        if not op["source_quote"].strip() or not any(op["source_quote"] in value for value in evidence):
            raise ValueError("related decision source quote must occur in current source evidence")
        if op["reason"] != "correct_current_fact":
            raise ValueError("related update requires a supported current-fact correction")
        source = sources[0]
        schema = self.resolver.schemas.get_compatible(node.category)
        seed = replace(source, category=node.category, owner_scope=schema.owner_scope,
                       scope_ref=node.metadata.get("scope_ref", ""), who=node.metadata.get("who"),
                       routing_key=node.metadata["routing_key"], object_name=node.metadata.get("object_name", ""),
                       target_uri=node.uri, expected_version=node.metadata["version"],
                       when=node.metadata.get("when"), where=node.metadata.get("where"),
                       evidence_quote=op["source_quote"],
                       verified_routing_aliases=list(node.metadata.get("routing_key_aliases", [])),
                       verified_object_aliases=list(node.metadata.get("object_name_aliases", [])))
        candidate = self._materialize({**op, "action": "update"}, [seed], reads, ctx)
        return replace(candidate, node_reconciliation_role="related",
                       provenance_ids=list(dict.fromkeys(s for c in sources for s in c.provenance_ids)),
                       source_contexts=list(dict.fromkeys(s for c in sources for s in c.source_contexts)),
                       node_draft_indices=list(dict.fromkeys(i for c in sources for i in c.node_draft_indices)))

    def _materialize(self, op, drafts, reads, ctx, *, messages=()):
        candidate = drafts[0]
        action, target = op["action"], op["target"]
        if action == "defer":
            raise ValueError("identity unresolved for " + candidate.routing_key)
        if action not in {"create", "update", "noop"}:
            raise ValueError("unsupported node action")
        sources = list(dict.fromkeys(s for c in drafts for s in c.provenance_ids))
        abstract = _line(op["abstract"])
        overview = "\n".join("- " + _line(x).removeprefix("- ") for x in op["overview"] if _line(x))
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
            for draft in drafts:
                if self.resolver._in_scope(node, draft, ctx):
                    continue
                resolution = op.get("scope_resolution", {})
                quotes = [quote.strip() for quote in resolution.get("source_quotes", [])]
                evidence = draft.source_contexts or list(_text_values(messages))
                identity_labels = {canonical_scope(value) for value in (
                    draft.object_name, draft.scope_ref, node.metadata.get("object_name", ""),
                    node.metadata.get("scope_ref", ""), *draft.verified_object_aliases,
                    *node.metadata.get("object_name_aliases", []))}
                if not (self._scope_candidate(node, draft, ctx)
                        and resolution.get("stored_scope_role") == "self"
                        and resolution.get("incoming_scope_role") == "self"
                        and any(quote and canonical_scope(quote) not in identity_labels
                                and any(quote in source for source in evidence) for quote in quotes)):
                    raise ValueError("update requires a source-grounded self-scope resolution for each draft")
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
                                scope_ref=node.metadata.get("scope_ref", "") if candidate.category == "entity" else candidate.scope_ref,
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
