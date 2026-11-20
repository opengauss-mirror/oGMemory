"""Prepare a canonical merge without changing the target business storage."""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

from service.migration.engine import Inventory, record_key
from service.migration.package import DATASETS, PackageError
from service.migration.records import identity, node_object, normalize_node


def fingerprint(record):
    # Storage revisions do not make a new memory occurrence. Business timestamps
    # such as metadata.when remain part of the content identity.
    business = {key: value for key, value in record.items()
                if key not in {'version', 'created_at', 'updated_at'}}
    return hashlib.sha256(json.dumps(business, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def lookup(inv, kind, key):
    row = inv.db.execute('SELECT body FROM records WHERE kind=? AND key=?', (kind, key)).fetchone()
    return json.loads(row[0]) if row else None


def merge_identity_metadata(existing, incoming):
    """Preserve both stores' observation history and verified naming aliases."""
    from commit.merge_policies import _merge_provenance_ids

    def union(left, right):
        if any(not isinstance(values, list) or any(not isinstance(v, str) for v in values)
               for values in (left, right)):
            raise PackageError('invalid memory identity metadata')
        return _merge_provenance_ids(left, right)

    result = {}
    for key in ('memory_observations', 'routing_key_aliases', 'object_name_aliases'):
        if key in existing or key in incoming:
            result[key] = union(existing.get(key, []), incoming.get(key, []))
    for key in ('memory_observation_sources', 'occurrence_replay_sources'):
        if key not in existing and key not in incoming:
            continue
        left, right = existing.get(key, {}), incoming.get(key, {})
        if not isinstance(left, dict) or not isinstance(right, dict):
            raise PackageError('invalid memory identity source metadata')
        result[key] = {item: union(left.get(item, []), right.get(item, []))
                       for item in dict.fromkeys([*left, *right])}
    # Historical explanation does not change the selected target identity.
    if 'identity_reason' in existing:
        result['identity_reason'] = existing['identity_reason']
    return result


class PolicyConflictResolver:
    """Use normal memory policies against an isolated inventory, including LLM merge."""
    def __init__(self, target, llm, *, registry=None, card_format_enabled=False):
        from commit.archive_builder import ArchiveBuilder
        from commit.policy_router import PolicyRouter

        self.target = target
        self.router = PolicyRouter(self, registry=registry)
        self.builder = ArchiveBuilder(llm, schema_registry=registry,
                                      card_format_enabled=card_format_enabled)

    def exists(self, uri, ctx):
        return identity(uri)[0] == ctx.account_id and lookup(self.target, 'nodes', uri.rstrip('/')) is not None

    def read_node(self, uri, ctx):
        record = lookup(self.target, 'nodes', uri.rstrip('/'))
        if record is None or record['account_id'] != ctx.account_id:
            raise PackageError('merge policy selected an unavailable node')
        return node_object(record)

    def __call__(self, existing, incoming):
        from core.models import CandidateMemory, RequestContext

        for key in ('account_id', 'owner_space', 'category', 'context_type', 'status'):
            if existing[key] != incoming[key]:
                raise PackageError(f'conflicting node {key}')
        if incoming['uri'].endswith('/') and existing['uri'].endswith('/'):
            return existing  # Summaries are rebuilt from the final child set.
        if incoming['status'] != 'ACTIVE' or incoming['context_type'] not in {'MEMORY', 'SKILL'}:
            raise PackageError('conflicting resource or inactive node cannot be merged')
        owner, owner_id = incoming['owner_space'].split(':', 1)
        ctx = RequestContext(incoming['account_id'], owner_id if owner == 'user' else '',
                             owner_id if owner == 'agent' else '', '', 'migration')
        metadata = incoming['metadata']
        candidate = CandidateMemory(
            category=incoming['category'], owner_scope=owner,
            routing_key=metadata.get('routing_key') or identity(incoming['uri'])[1][-1],
            content=incoming['content'], abstract=incoming['abstract'],
            overview=incoming['overview'], confidence=1.0,
            **{key: metadata[key] for key in ('when', 'who', 'where', 'tool_stats',
                 'provenance_ids', 'scope_ref', 'title', 'evidence_quote') if key in metadata},
        )
        plan = self.router.plan(candidate, ctx)
        if plan.action == 'skip':
            return existing
        if plan.action not in {'create', 'merge'}:
            raise PackageError('unsupported memory merge action')
        if plan.action == 'merge' and plan.target_uri.rstrip('/') != existing['uri'].rstrip('/'):
            raise PackageError('merge policy selected a different existing node')
        from commit.merge_policies import AppendOnlyPolicy
        if plan.action == 'create' and not isinstance(self.router.route(candidate), AppendOnlyPolicy):
            raise PackageError('conflicting identity cannot be rerouted to another memory')
        if plan.action == 'create' and lookup(self.target, 'nodes', plan.target_uri.rstrip('/')):
            raise PackageError('merge policy allocated an occupied identity')
        if identity(plan.target_uri)[0] != incoming['account_id']:
            raise PackageError('merge policy crossed account boundary')
        built = self.builder.build(candidate, plan, ctx)
        # Preserve unhandled business metadata; reject conflicting values rather
        # than silently picking a source. Policy-owned values use policy output.
        merged = dict(existing['metadata']) if plan.action == 'merge' else {}
        identity_metadata = merge_identity_metadata(merged, metadata)
        for key, value in metadata.items():
            # This incarnation belongs to the target source node. Imported
            # shared evidence keeps the source epoch only for inserted nodes.
            if key == 'shared_source_epoch' and plan.action == 'merge':
                continue
            if key in merged and merged[key] != value and key not in built.metadata and key not in identity_metadata:
                raise PackageError(f'unsupported metadata conflict: {key}')
            merged[key] = value
        merged.update(built.metadata)
        merged.update(identity_metadata)
        now = datetime.now(UTC).isoformat()
        return normalize_node({**incoming, 'uri': built.uri, 'content': built.content,
                               'abstract': built.abstract, 'overview': built.overview,
                               'metadata': merged, 'updated_at': now,
                               'created_at': existing['created_at'] if plan.action == 'merge' else now,
                               'version': existing['version'] + 1 if plan.action == 'merge' else 1})


class MergePlan:
    """Disk-backed result and changed records; callers publish only after backup.

    ``applied`` maps a canonical source record digest to the previously allocated
    URI. Persist ``receipts`` together with successful data and include that ledger
    in target backup. This preserves append-only policy identities across packages.
    """
    def __init__(self, path, target, source, resolve, *, applied=lambda digest: None):
        self.inventory = Inventory(path)
        self.db = self.inventory.db
        self.db.execute('CREATE TABLE changes(kind TEXT,key TEXT,PRIMARY KEY(kind,key))')
        self.db.execute('CREATE TABLE mappings(source TEXT PRIMARY KEY,target TEXT)')
        self.db.execute('CREATE TABLE receipts(digest TEXT PRIMARY KEY,target TEXT)')
        self.counts = dict(inserted=0, merged=0, skipped=0)
        try:
            self._prepare(target, source, resolve, applied)
        except BaseException:
            self.close()
            raise

    def close(self):
        self.inventory.close()

    def _put(self, kind, record, *, changed=False):
        key = record_key(kind, record)
        self.db.execute('INSERT OR REPLACE INTO records VALUES(?,?,?)',
                        (kind, key, json.dumps(record, ensure_ascii=False, sort_keys=True)))
        if changed:
            self.db.execute('INSERT OR IGNORE INTO changes VALUES(?,?)', (kind, key))

    def _rewrite(self, value):
        if isinstance(value, dict):
            return {key: self._rewrite(child) for key, child in value.items()}
        if isinstance(value, list):
            return [self._rewrite(child) for child in value]
        if not isinstance(value, str):
            return value
        if value.startswith('prov:'):
            from core.provenance_resolver import ProvenanceResolver
            parsed = ProvenanceResolver.parse_id(value)
            if parsed['source_type'] == 'memory':
                target = self._rewrite(parsed['source_id'])
                if target != parsed['source_id']:
                    return ProvenanceResolver.build_id('memory', target, parsed['detail'])
        row = self.db.execute('SELECT target FROM mappings WHERE source=?', (value.rstrip('/'),)).fetchone()
        return row[0] if row else value

    def _rewrite_history(self, old_ids, incoming_ids, combined):
        return list(dict.fromkeys([
            *old_ids, *(self._rewrite(pid) for pid in incoming_ids),
            *(self._rewrite(pid) for pid in combined if pid not in old_ids and pid not in incoming_ids),
        ]))

    def _prepare(self, target, source, resolve, applied):
        for kind in DATASETS:
            for record in target.records(kind):
                self._put(kind, record)
        # Resolve nodes first, so all imported relations can use final identities.
        for record in source.records('nodes'):
            key, digest = record_key('nodes', record), fingerprint(record)
            previous = applied(digest)
            existing = lookup(target, 'nodes', key)
            if previous is not None:
                if lookup(target, 'nodes', previous.rstrip('/')) is None:
                    raise PackageError('previous migration receipt points to missing data')
                result, action = lookup(target, 'nodes', previous.rstrip('/')), 'skipped'
            elif existing is not None and fingerprint(existing) == digest:
                result, action = existing, 'skipped'
            elif existing is None:
                result, action = record, 'inserted'
            else:
                result, action = resolve(existing, record), 'merged'
            result_key = record_key('nodes', result)
            if result_key != key and previous is None and (lookup(self.inventory, 'nodes', result_key) is not None or lookup(source, 'nodes', result_key) is not None):
                raise PackageError('allocated migration identity already exists')
            self.counts[action] += 1
            self.db.execute('INSERT INTO mappings VALUES(?,?)', (key, result['uri']))
            self.db.execute('INSERT INTO receipts VALUES(?,?)', (digest, result['uri']))
            if action != 'skipped':
                self._put('nodes', result, changed=True)
        for kind in ('archives', 'relations'):
            for record in source.records(kind):
                record = dict(record)
                if kind == 'relations':
                    for endpoint in ('from_uri', 'to_uri'):
                        record[endpoint] = self._rewrite(record[endpoint])
                # Archive metadata is opaque session data, not memory references.
                # Preserve notes and historical identifiers exactly as exported.
                key = record_key(kind, record)
                existing = lookup(self.inventory, kind, key)
                if existing is not None and fingerprint(existing) != fingerprint(record):
                    raise PackageError(f'conflicting {kind} cannot be merged')
                self.counts['skipped' if existing else 'inserted'] += 1
                if existing is None:
                    self._put(kind, record, changed=True)
        # Rewrite only imported/merged nodes. Original target references keep
        # pointing at the original occurrence when the source was appended.
        for (key,) in self.db.execute("SELECT key FROM changes WHERE kind='nodes'"):
            record = lookup(self.inventory, 'nodes', key)
            rewritten = self._rewrite(record['metadata'])
            original = lookup(target, 'nodes', key)
            if original is not None:
                # Existing provenance belongs to the target occurrence. Only
                # incoming evidence is remapped to a newly allocated identity.
                source_node = lookup(source, 'nodes', key)
                old_ids = original['metadata'].get('provenance_ids', [])
                incoming_ids = source_node['metadata'].get('provenance_ids', []) if source_node else []
                ids = record['metadata'].get('provenance_ids', [])
                if ids:
                    rewritten['provenance_ids'] = self._rewrite_history(old_ids, incoming_ids, ids)
                source_fields = ('memory_observation_sources', 'occurrence_replay_sources')
                for field in source_fields:
                    if field not in record['metadata']:
                        continue
                    old_sources = original['metadata'].get(field, {})
                    incoming_sources = source_node['metadata'].get(field, {}) if source_node else {}
                    rewritten[field] = {
                        observation: self._rewrite_history(old_sources.get(observation, []),
                            incoming_sources.get(observation, []), values)
                        for observation, values in record['metadata'][field].items()
                    }
                for field, value in original['metadata'].items():
                    if field not in ('provenance_ids', *source_fields) and record['metadata'].get(field) == value:
                        if (source_node and source_node['metadata'].get(field) == value
                                and self._rewrite(value) != value):
                            raise PackageError(f'ambiguous remapped metadata reference: {field}')
                        rewritten[field] = value
            record['metadata'] = rewritten
            self._put('nodes', record)
        self._directories(resolve)
        # Rebuild references using Inventory's shared validation, not a second
        # partial interpretation of the protocol.
        self.db.execute('ALTER TABLE records RENAME TO staged_records')
        self.db.execute('CREATE TABLE records(kind TEXT,key TEXT,body TEXT,PRIMARY KEY(kind,key))')
        for kind in DATASETS:
            for (body,) in self.db.execute('SELECT body FROM staged_records WHERE kind=? ORDER BY key', (kind,)):
                record = json.loads(body)
                dataset = 'space_nodes' if kind == 'nodes' and record['owner_space'].startswith('space:') else kind
                self.inventory.add(dataset, record)
        self.db.execute('DROP TABLE staged_records')
        self.db.execute("CREATE INDEX record_scope ON records "
                        "(kind,json_extract(body,'$.account_id'),json_extract(body,'$.owner_space'),key)")
        self.inventory.finish()

    def _directories(self, resolve):
        from index.directory_summarizer import DirectorySummarizer, DirectorySummary

        self.db.execute('CREATE TABLE directories(key TEXT PRIMARY KEY)')
        for record in self.changes('nodes'):
            key = record_key('nodes', record)
            if record['uri'].endswith('/'):
                self.db.execute('INSERT OR IGNORE INTO directories VALUES(?)', (key,))
            account, parts = identity(record['uri'])
            for length in range(2, len(parts)):
                parent = f'ctx://{account}/' + '/'.join(parts[:length])
                existing = lookup(self.inventory, 'nodes', parent)
                if existing and existing['uri'].endswith('/'):
                    self.db.execute('INSERT OR IGNORE INTO directories VALUES(?)', (parent,))
        summarizer = DirectorySummarizer(None, getattr(getattr(resolve, 'builder', None), '_llm', None))
        # Children first; ancestors read the already-updated child summaries.
        for (key,) in self.db.execute('SELECT key FROM directories ORDER BY length(key) DESC,key'):
            record = lookup(self.inventory, 'nodes', key)
            if record['status'] != 'ACTIVE' or record['metadata'].get('indexing_disabled'):
                continue
            prefix = key + '/'
            children, categories, count = [], set(), 0
            for (body,) in self.db.execute(
                    "SELECT body FROM records WHERE kind='nodes' AND key>=? AND key<? "
                    "AND instr(substr(key,?),'/')=0 ORDER BY key", (prefix, key + '0', len(prefix) + 1)):
                child = json.loads(body)
                if child['status'] != 'ACTIVE' or child['metadata'].get('indexing_disabled'):
                    continue
                if child['account_id'] != record['account_id'] or child['owner_space'] != record['owner_space']:
                    raise PackageError('directory children cross owner boundary')
                count += 1
                categories.add(child['category'])
                if child['abstract'] and len(children) < 50:
                    children.append({name: child[name] for name in ('uri', 'abstract', 'category')})
            summary = (summarizer._generate_summary(record['uri'], children) if children
                       else DirectorySummary('空目录', '', 0, []))
            record.update(abstract=summary.abstract, overview=summary.overview, content='',
                          version=record['version'] + 1, updated_at=datetime.now(UTC).isoformat())
            record['metadata'].update(node_type='directory', child_count=count, categories=sorted(categories))
            self._put('nodes', record, changed=True)

    def changes(self, kind):
        for (body,) in self.db.execute('SELECT r.body FROM records r JOIN changes c '
                                      'ON r.kind=c.kind AND r.key=c.key WHERE r.kind=? ORDER BY r.key', (kind,)):
            yield json.loads(body)

    def receipts(self):
        yield from self.db.execute('SELECT digest,target FROM receipts ORDER BY digest')
