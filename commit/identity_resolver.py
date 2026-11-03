"""Resolve memory identity before allocating a public node name."""

from __future__ import annotations

import hashlib
import json
import threading
import unicodedata
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from datetime import date, datetime
from weakref import WeakValueDictionary

from commit.merge_policies import AggregateTopicPolicy, AppendOnlyPolicy
from commit.policy_router import PolicyRouter
from commit.routing_key import normalize_routing_key
from commit.write_receipts import receipt_plan
from core.candidate_dedup import occurrence_replay_fields
from core.errors import (
    ConcurrentModificationError,
    ExtractionIncompleteError,
    NodeNotFoundError,
    ValidationError,
)
from core.interfaces import ContextFS
from core.memory_names import (
    canonical_memory_key,
    canonical_object_name,
    canonical_scope,
    is_meaningful_memory_key,
    qualify_memory_key,
    unscoped_memory_key,
)
from core.models import CandidateMemory, ContextNode, RequestContext, WritePlan
from core.uri_resolver import URIResolver
from core.validation import MAX_MEMORY_KEY_BYTES
from extraction.schemas.registry import SchemaRegistry

MEMORY_CATEGORIES = frozenset({"entity", "preference", "pattern"})
OCCURRENCE_CATEGORIES = frozenset({"event", "case"})
_USER_ROLE_MARKERS = frozenset({"user", "<user>"})
_LOCAL_LOCKS: WeakValueDictionary[str, threading.RLock] = WeakValueDictionary()
_LOCAL_LOCKS_GUARD = threading.Lock()


def observation_id(candidate: CandidateMemory) -> str:
    """Identify exact content independently of its generated name and sources."""
    payload = {
        "category": candidate.category,
        "scope": canonical_scope(candidate.scope_ref),
        "content": unicodedata.normalize("NFKC", candidate.content).strip(),
        "when": candidate.when,
        "who": candidate.who,
        "where": candidate.where,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def occurrence_replay_id(candidate: CandidateMemory) -> str:
    """Fingerprint complete occurrence evidence, excluding generated write identity.

    This is deliberately stricter than semantic occurrence matching. Even identical
    evidence is a replay only when every incoming source was recorded for it.
    """
    payload = occurrence_replay_fields(candidate)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class MemoryIdentityResolver:
    """Validate extraction-selected targets and serialize their writes."""

    def __init__(self, fs: ContextFS, uri_resolver: URIResolver,
                 schema_registry: SchemaRegistry | None = None,
                 policy_router: PolicyRouter | None = None):
        self.fs = fs
        self.uris = uri_resolver
        self.schemas = schema_registry or uri_resolver.schema_registry
        self.policies = policy_router or PolicyRouter(fs, registry=self.schemas, uri_resolver=uri_resolver)
        self._lock_state = threading.local()

    def handles(self, candidate: CandidateMemory) -> bool:
        if candidate.category not in MEMORY_CATEGORIES | OCCURRENCE_CATEGORIES:
            return False
        schema = self.schemas.get_compatible(candidate.category)
        if (schema is None or schema.owner_scope != candidate.owner_scope
                or self.policies.has_custom_policy(candidate.category)):
            return False
        policy = self.policies.route(candidate)
        if candidate.category in MEMORY_CATEGORIES:
            return (schema.operation_mode == "upsert" and schema.is_multi_file
                    and type(policy) is AggregateTopicPolicy)
        if candidate.category in OCCURRENCE_CATEGORIES:
            return schema.operation_mode == "add_only" and type(policy) is AppendOnlyPolicy
        return False

    def _directory(self, candidate, ctx):
        schema = self.schemas.get_compatible(candidate.category)
        if schema is None or candidate.owner_scope != schema.owner_scope:
            raise ValueError("memory owner must match the category owner")
        return self.uris.get_directory_uri(candidate.category, ctx).rstrip("/")

    @staticmethod
    def _self_holder(value, ctx):
        holder = canonical_scope(value or "")
        return canonical_scope(ctx.user_id) if holder in _USER_ROLE_MARKERS else holder

    @staticmethod
    def _normalize_subject(candidate, ctx):
        if (candidate.category == "preference" and candidate.who
                and canonical_scope(candidate.who) in _USER_ROLE_MARKERS):
            return replace(candidate, who=ctx.user_id)
        return candidate

    @contextmanager
    def lock_directories(self, directories, ctx):
        """Acquire each directory once in a consistent order."""
        with ExitStack() as stack:
            for directory in sorted(set(directories)):
                stack.enter_context(self._lock(directory, ctx))
            yield

    @contextmanager
    def _lock(self, directory, ctx):
        key = (ctx.account_id, directory)
        held = getattr(self._lock_state, "held", frozenset())
        if key in held:
            yield
            return
        method = getattr(type(self.fs), "memory_identity_lock", None)
        if callable(method):
            lock = self.fs.memory_identity_lock(directory, ctx)
        else:
            # Hold a strong reference through acquisition and release; idle locks are reclaimed.
            with _LOCAL_LOCKS_GUARD:
                lock = _LOCAL_LOCKS.get(directory)
                if lock is None:
                    lock = threading.RLock()
                    _LOCAL_LOCKS[directory] = lock
        with lock:
            self._lock_state.held = held | {key}
            try:
                yield
            finally:
                self._lock_state.held = held

    def _nodes(self, directory, ctx):
        page_method = getattr(type(self.fs), "list_identity_node_summaries_page", None)
        if not callable(page_method):
            page_method = getattr(type(self.fs), "list_node_summaries_page", None)
        if callable(page_method):
            nodes = []
            cursor = None
            order_key = getattr(self.fs, "list_node_order_key", lambda uri: uri)
            while True:
                page = page_method(self.fs, directory, ctx, limit=100, after_uri=cursor)
                nodes.extend(page)
                if len(page) < 100:
                    return nodes
                cursor = order_key(page[-1].uri)
        nodes = []
        for uri in self.fs.list_children(directory, ctx):
            try:
                nodes.append(self.fs.read_node(uri, ctx))
            except NodeNotFoundError:
                continue
        return nodes

    @staticmethod
    def _in_scope(node, candidate, ctx):
        owner = ctx.agent_space_name() if candidate.owner_scope == "agent" else ctx.user_space_name()
        if (node.category != candidate.category or node.owner_space != owner
                or canonical_scope(node.metadata.get("scope_ref", ""))
                != canonical_scope(candidate.scope_ref)):
            return False
        if candidate.category != "preference":
            return True
        stored = MemoryIdentityResolver._self_holder(node.metadata.get("who"), ctx)
        incoming = MemoryIdentityResolver._self_holder(candidate.who, ctx)
        return (stored == incoming
                or {stored, incoming} <= {"", canonical_scope(ctx.user_id)}
                or (bool(candidate.target_uri) and not stored))

    @staticmethod
    def _names_match(node, candidate):
        routing_values = (
            node.uri.rsplit("/", 1)[-1], node.metadata.get("routing_key", ""),
            *node.metadata.get("routing_key_aliases", []),
        )
        legacy_key = (normalize_routing_key(candidate.legacy_routing_key, candidate.category)
                      if candidate.legacy_routing_key else "")
        legacy_match = bool(legacy_key and routing_values[0] == legacy_key)
        if candidate.category == "entity" and candidate.object_name:
            object_names = {canonical_object_name(value) for value in (
                node.metadata.get("object_name", ""), *node.metadata.get("object_name_aliases", []),
            ) if value}
            object_name = canonical_object_name(candidate.object_name)
            if object_names:
                return bool(object_name and object_name in object_names)
            legacy_names = {canonical_object_name(value) for value in routing_values if value}
            scoped_name = qualify_memory_key(object_name.replace(" ", "_"),
                                             canonical_memory_key(candidate.scope_ref))
            return bool(object_name and ({object_name, scoped_name} & legacy_names)) or legacy_match
        if legacy_match:
            return True
        scope = canonical_memory_key(candidate.scope_ref)
        holder = canonical_memory_key(candidate.who or "") if candidate.category == "preference" else ""
        legacy_user_prefix = (candidate.category == "preference"
                              and canonical_scope(node.metadata.get("who") or "") in _USER_ROLE_MARKERS)

        def topic_key(value):
            key = canonical_memory_key(value)
            if legacy_user_prefix:
                key = key.removeprefix("user_")
            unscoped = unscoped_memory_key(unscoped_memory_key(key, scope), holder)
            return normalize_routing_key(unscoped, candidate.category)

        routing_names = {topic_key(value) for value in routing_values if value}
        return bool(candidate.routing_key and topic_key(candidate.routing_key) in routing_names)

    @staticmethod
    def _exact_time(value):
        value = (value or "").strip()
        try:
            if len(value) == 10:
                return date.fromisoformat(value).isoformat()
            if len(value) > 10 and value[10] in {"T", " "}:
                return datetime.fromisoformat(value).isoformat()
        except ValueError:
            pass
        return ""

    @staticmethod
    def _replays_observation(node, candidate, observation):
        history = node.metadata.get("memory_observations", [])
        if observation not in history:
            return False
        same_source = (bool(candidate.provenance_ids)
                       and set(candidate.provenance_ids)
                       <= set(node.metadata.get("memory_observation_sources", {}).get(observation, [])))
        if candidate.category in OCCURRENCE_CATEGORIES:
            when = MemoryIdentityResolver._exact_time(candidate.when)
            if not when or not canonical_object_name(candidate.who or ""):
                return same_source
        return observation == history[-1] or same_source

    @staticmethod
    def _is_changed_current_update(node, candidate):
        if (not candidate.target_uri
                or candidate.expected_version != node.metadata.get("version", 0)):
            return False
        if any(getattr(candidate, name) != getattr(node, name)
               for name in ("content", "overview", "abstract")):
            return True
        if (candidate.category in OCCURRENCE_CATEGORIES and candidate.complete_occurrence_context
                and any(getattr(candidate, name) != node.metadata.get(name)
                        for name in ("when", "who", "where"))):
            return True
        names = ("when", "who", "where", "title") if candidate.category in OCCURRENCE_CATEGORIES else (
            "when", "who", "where")
        return any(getattr(candidate, name)
                   and getattr(candidate, name) != node.metadata.get(name)
                   for name in names)

    def _read_target(self, candidate, directory, ctx):
        if candidate.target_uri.rsplit("/", 1)[0] != directory:
            raise ValueError("update target must be in the current owner/category directory")
        try:
            node = self.fs.read_node(candidate.target_uri, ctx)
        except NodeNotFoundError as exc:
            raise ValueError("update target must exist in the candidate scope") from exc
        if not self._in_scope(node, candidate, ctx):
            raise ValueError("update target must exist in the candidate scope")
        return node

    def _match(self, candidate, nodes, directory, ctx):
        if candidate.target_uri:
            if candidate.target_uri.rsplit("/", 1)[0] != directory:
                raise ValueError("update target must be in the current owner/category directory")
            matches = [node for node in nodes if node.uri == candidate.target_uri
                       and self._in_scope(node, candidate, ctx)]
            if not matches:
                raise ValueError("update target must exist in the candidate scope")
            return matches[0]
        matches = []
        for node in nodes:
            if not self._in_scope(node, candidate, ctx) or not self._names_match(node, candidate):
                continue
            if candidate.category in OCCURRENCE_CATEGORIES:
                when = self._exact_time(candidate.when)
                who = canonical_object_name(candidate.who or "")
                if (not when or not who or when != self._exact_time(node.metadata.get("when"))
                        or who != canonical_object_name(node.metadata.get("who") or "")):
                    continue
            matches.append(node)
        if len(matches) > 1:
            raise ValueError("memory identity is ambiguous; select a read target explicitly")
        return matches[0] if matches else None

    def match_existing(self, candidate: CandidateMemory, ctx: RequestContext) -> ContextNode | None:
        """Return one exact scoped identity with content for an extraction update.

        Matching an occurrence additionally requires its explicit time and actor.
        The writer checks the version after extraction prepares the complete update.
        """
        if not self.handles(candidate):
            return None
        candidate = self._normalize_subject(candidate, ctx)
        directory = self._directory(candidate, ctx)
        if candidate.target_uri:
            return self._read_target(candidate, directory, ctx)
        match = self._match(candidate, self._nodes(directory, ctx), directory, ctx)
        return self.fs.read_node(match.uri, ctx) if match else None

    def preflight_new_batch(self, candidates, ctx):
        """Reject ambiguous allocations before a new durable batch is accepted.

        Project only names, never writes or content merges. The real writer still
        resolves identity and CAS under its lock; this is not write authority.
        """
        directories = {}
        loaded_directories = set()
        for incoming in candidates:
            if incoming.category not in MEMORY_CATEGORIES or not self.handles(incoming):
                continue
            candidate = self._normalize_subject(incoming, ctx)
            directory = self._directory(candidate, ctx)
            projected = directories.setdefault(directory, {})
            try:
                if candidate.target_uri:
                    if candidate.target_uri not in projected:
                        projected[candidate.target_uri] = self._read_target(candidate, directory, ctx)
                elif directory not in loaded_directories:
                    for node in self._nodes(directory, ctx):
                        projected.setdefault(node.uri, node)
                    loaded_directories.add(directory)
                nodes = list(projected.values())
                existing = self._match(candidate, self._candidate_nodes(candidate, directory, ctx, nodes=nodes), directory, ctx)
                uri = existing.uri if existing else self._new_uri(candidate, nodes, ctx)
            except ValueError as exc:
                raise ExtractionIncompleteError("Memory batch identity needs re-extraction: " + str(exc)) from exc
            metadata = self._name_metadata(candidate, existing, uri, ctx)
            if candidate.who:
                metadata["who"] = candidate.who
            elif existing:
                metadata["who"] = existing.metadata.get("who")
            projected[uri] = ContextNode(
                uri=uri, context_type="MEMORY", category=candidate.category,
                level=0, abstract="", overview="", content="",
                owner_space=ctx.agent_space_name() if candidate.owner_scope == "agent" else ctx.user_space_name(),
                metadata=metadata,
            )

    def _proposed_key(self, candidate, ctx):
        proposed = normalize_routing_key(canonical_memory_key(candidate.routing_key), candidate.category)
        if candidate.category == "entity" and candidate.object_name:
            object_key = canonical_memory_key(candidate.object_name)
            if is_meaningful_memory_key(object_key):
                proposed = object_key
        if (candidate.category == "preference" and candidate.who
                and canonical_scope(candidate.who) != canonical_scope(ctx.user_id)):
            proposed = qualify_memory_key(proposed, canonical_memory_key(candidate.who))
        if candidate.scope_ref:
            scope_key = canonical_memory_key(candidate.scope_ref)
            if not scope_key:
                raise ValueError("memory scope needs a descriptive name")
            proposed = qualify_memory_key(proposed, scope_key)
        return proposed

    def _uri_for_key(self, candidate, key, ctx):
        return self.uris.resolve(candidate.category, {"routing_key": key}, ctx)

    def _new_uri(self, candidate, nodes, ctx):
        if candidate.category in OCCURRENCE_CATEGORIES:
            uri = AppendOnlyPolicy(self.fs, self.uris).plan(candidate, ctx).target_uri
        else:
            proposed = self._proposed_key(candidate, ctx)
            if (candidate.category == "entity" and candidate.object_name
                    and any(node.uri == self._uri_for_key(candidate, proposed, ctx) for node in nodes)):
                proposed = qualify_memory_key(normalize_routing_key(
                    canonical_memory_key(candidate.routing_key), candidate.category),
                                              canonical_memory_key(candidate.scope_ref))
            # Resolved names include scope/subject prefixes; their limit is the
            # storage path budget, not the extraction prompt's short-name limit.
            if not is_meaningful_memory_key(proposed, max_length=MAX_MEMORY_KEY_BYTES):
                raise ValueError("memory needs a descriptive routing key")
            if len(proposed.encode("utf-8")) > MAX_MEMORY_KEY_BYTES:
                raise ValidationError("routing_key", "Memory name exceeds path segment length limit")
            uri = self._uri_for_key(candidate, proposed, ctx)
        if len(uri.rsplit("/", 1)[-1].encode("utf-8")) > MAX_MEMORY_KEY_BYTES:
            raise ValidationError("routing_key", "Memory name exceeds path segment length limit")
        if any(node.uri == uri for node in nodes):
            raise ValueError("memory name is already used; identity needs disambiguation")
        return uri

    def _archived_version(self, uri, ctx):
        if callable(getattr(type(self.fs), "archived_node_version", None)):
            return self.fs.archived_node_version(uri, ctx)
        return None

    def _candidate_nodes(self, candidate, directory, ctx, *, nodes=None):
        if candidate.target_uri:
            return ([self._read_target(candidate, directory, ctx)] if nodes is None else
                    [self._match(candidate, nodes, directory, ctx)])
        proposed = self._proposed_key(candidate, ctx)
        resolved = self._uri_for_key(candidate, proposed, ctx) if proposed else ""
        uris = ([resolved] if resolved and
                len(resolved.rsplit("/", 1)[-1].encode("utf-8")) <= MAX_MEMORY_KEY_BYTES else [])
        if candidate.legacy_routing_key:
            legacy_key = normalize_routing_key(candidate.legacy_routing_key, candidate.category)
            legacy_uri = self._uri_for_key(candidate, legacy_key, ctx)
            if legacy_uri not in uris:
                uris.append(legacy_uri)
        for uri in uris:
            try:
                direct = (self.fs.read_node(uri, ctx) if nodes is None else
                          next((node for node in nodes if node.uri == uri), None))
            except NodeNotFoundError:
                continue
            if direct is not None and self._in_scope(direct, candidate, ctx) and self._names_match(direct, candidate):
                return [direct]
        return self._nodes(directory, ctx) if nodes is None else nodes

    def _name_metadata(self, candidate, existing, uri, ctx):
        """Project the same scoped names for validation and the real write plan."""
        meta = existing.metadata if existing else {}
        allocated_key = uri.rsplit("/", 1)[-1]
        if candidate.category in OCCURRENCE_CATEGORIES:
            routing_key = meta.get("routing_key") or canonical_memory_key(candidate.routing_key)
        elif self._uri_for_key(candidate, allocated_key, ctx) == uri:
            routing_key = allocated_key
        else:
            routing_key = meta.get("routing_key") or self._proposed_key(candidate, ctx)
        routing_aliases = list(dict.fromkeys(filter(None, [
            *meta.get("routing_key_aliases", []),
            *candidate.verified_routing_aliases,
            candidate.routing_key if candidate.routing_key != allocated_key else routing_key,
        ])))
        object_names = [meta.get("object_name", ""), *meta.get("object_name_aliases", [])]
        object_names.extend(candidate.verified_object_aliases)
        if not existing or candidate.target_uri:
            object_names.append(candidate.object_name)
        object_aliases = list(dict.fromkeys(filter(None, object_names)))
        return dict(identity_version=2, routing_key=routing_key,
                    routing_key_aliases=routing_aliases, object_name_aliases=object_aliases,
                    scope_ref=candidate.scope_ref,
                    object_name=meta.get("object_name") or candidate.object_name)

    def _plan(self, candidate, existing, nodes, directory, ctx, *, duplicate=False):
        uri = existing.uri if existing else self._new_uri(candidate, nodes, ctx)
        archived_version = (self._archived_version(uri, ctx) if not existing
                            and candidate.category not in OCCURRENCE_CATEGORIES else None)
        meta = dict(existing.metadata) if existing else {}
        for transient in ("expected_version", "_expected_version", "has_content"):
            meta.pop(transient, None)
        observation = observation_id(candidate)
        if not duplicate:
            previous = dict.fromkeys(meta.get("memory_observations", []))
            previous.pop(observation, None)
            meta["memory_observations"] = [*previous, observation]
        if candidate.provenance_ids:
            sources = dict(meta.get("memory_observation_sources", {}))
            sources[observation] = list(dict.fromkeys([
                *sources.get(observation, []), *candidate.provenance_ids,
            ]))
            meta["memory_observation_sources"] = sources
        meta["provenance_ids"] = list(dict.fromkeys([*meta.get("provenance_ids", []), *candidate.provenance_ids]))
        if candidate.category in OCCURRENCE_CATEGORIES and candidate.provenance_ids:
            replays = dict(meta.get("occurrence_replay_sources", {}))
            replay = occurrence_replay_id(candidate)
            replays[replay] = list(dict.fromkeys([*replays.get(replay, []), *candidate.provenance_ids]))
            meta["occurrence_replay_sources"] = replays
        meta.update(self._name_metadata(candidate, existing, uri, ctx))
        if duplicate:
            content, overview, abstract = existing.content, existing.overview, existing.abstract
        else:
            fields = [candidate.content, candidate.overview, candidate.abstract]
            if not all(isinstance(value, str) and value.strip() for value in fields):
                raise ValueError("identity result needs complete content, overview and abstract")
            content, overview, abstract = fields
        meta["identity_reason"] = ("exact content observed again" if duplicate else
                                   "target selected during extraction" if candidate.target_uri else
                                   "existing subject matched by name" if existing else
                                   "new subject selected during extraction")
        for name in ("when", "who", "where"):
            value = getattr(candidate, name)
            if value and not duplicate:
                meta[name] = value
            elif (not duplicate and candidate.category in OCCURRENCE_CATEGORIES
                  and candidate.target_uri and candidate.complete_occurrence_context):
                meta.pop(name, None)
        if candidate.category in OCCURRENCE_CATEGORIES and candidate.title and not duplicate:
            meta["title"] = candidate.title
        return WritePlan(
            action="merge" if existing or archived_version is not None else "create", target_uri=uri,
            merged_fields={"identity_metadata": meta, "resolved_content": content,
                           "resolved_overview": overview, "resolved_abstract": abstract,
                           **({"existing_overview": overview} if duplicate else {}),
                           "expected_version": (existing.metadata.get("version", 0) if existing
                                                else archived_version if archived_version is not None else 0),
                           "identity_create": existing is None and archived_version is None},
        )

    def write(self, candidate, ctx, commit):
        if candidate.owner_scope == "agent" and not ctx.agent_id:
            return commit(candidate, ctx)
        candidate = self._normalize_subject(candidate, ctx)
        for attempt in range(3):
            try:
                with self._lock(self._directory(candidate, ctx), ctx):
                    if candidate.write_operation_id:
                        committed = receipt_plan(self.fs, self.uris, candidate, ctx)
                        if committed is not None:
                            return committed
                    return self._write_once(candidate, ctx, commit)
            except ConcurrentModificationError:
                if candidate.target_uri or attempt == 2:
                    raise

    def _write_occurrence(self, candidate, ctx, commit):
        """Skip durable exact-source replays without guessing an occurrence by name."""
        directory = self._directory(candidate, ctx)
        replay = occurrence_replay_id(candidate)
        nodes = self._nodes(directory, ctx)
        matches = [node for node in nodes if self._in_scope(node, candidate, ctx)
                   and set(candidate.provenance_ids) <= set(
                       node.metadata.get("occurrence_replay_sources", {}).get(replay, []))]
        if len(matches) > 1:
            raise ValueError("occurrence replay has ambiguous targets; select a read target explicitly")
        if matches:
            if candidate.write_operation_id:
                existing = self.fs.read_node(matches[0].uri, ctx)
                commit(candidate, ctx, self._plan(candidate, existing, nodes, directory, ctx, duplicate=True))
            return WritePlan(action="skip", target_uri=matches[0].uri,
                             merged_fields={"status": "duplicate"})
        return commit(candidate, ctx, self._plan(candidate, None, nodes, directory, ctx))

    def _write_once(self, candidate, ctx, commit):
        if candidate.category in OCCURRENCE_CATEGORIES and not candidate.target_uri:
            if not candidate.provenance_ids:
                return commit(candidate, ctx)
            return self._write_occurrence(candidate, ctx, commit)
        directory = self._directory(candidate, ctx)
        observation = observation_id(candidate)
        nodes = self._candidate_nodes(candidate, directory, ctx)
        matched = self._match(candidate, nodes, directory, ctx)
        if matched and not candidate.target_uri:
            matched = self.fs.read_node(matched.uri, ctx)
        repeats = [node for node in nodes if self._in_scope(node, candidate, ctx)
                   and (node.uri == matched.uri if matched else
                        self._names_match(node, candidate))
                   and not self._is_changed_current_update(node, candidate)
                   and self._replays_observation(node, candidate, observation)]
        if len(repeats) > 1 and candidate.category not in OCCURRENCE_CATEGORIES:
            raise ValueError("duplicate content has ambiguous identities; select a read target explicitly")
        if repeats:
            existing = repeats[0] if candidate.target_uri else self.fs.read_node(repeats[0].uri, ctx)
            plan = self._plan(candidate, existing, nodes, directory, ctx, duplicate=True)
            metadata = plan.merged_fields["identity_metadata"]
            changed = any(metadata.get(name) != existing.metadata.get(name) for name in
                          ("provenance_ids", "memory_observation_sources",
                           "routing_key_aliases", "object_name_aliases"))
            if changed or candidate.write_operation_id:
                commit(candidate, ctx, plan)
            return WritePlan(action="skip", target_uri=existing.uri,
                             merged_fields={"status": "duplicate"})
        if matched:
            if not candidate.target_uri:
                plan = self._plan(candidate, matched, nodes, directory, ctx)
                fields = plan.merged_fields
                for name in ("resolved_content", "resolved_overview", "resolved_abstract"):
                    fields.pop(name)
                fields.update(existing_content=matched.content, content_append=candidate.content,
                              existing_overview=matched.overview, overview_append=candidate.overview)
                return commit(candidate, ctx, plan)
            if candidate.expected_version != matched.metadata.get("version", 0):
                raise ValueError("memory changed since extraction; refresh and re-extract the update")
        plan = self._plan(candidate, matched, nodes, directory, ctx)
        return commit(candidate, ctx, plan)
