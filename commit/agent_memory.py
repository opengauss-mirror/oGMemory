"""Typed Agent units with validated source bindings and revision-bound updates."""
import json
import re
from copy import deepcopy

from core.interfaces import ContextFS, MergePolicy
from core.models import CandidateMemory, RequestContext, WritePlan


class AgentMemoryConflictError(ValueError):
    """Re-extract against the current unit revision."""


_ITEM_FIELDS = {
    'tool': {'key', 'title', 'content', 'grounding'},
    'experience': {'key', 'title', 'content', 'grounding'},
}
_SHA = re.compile(r'[0-9a-f]{64}')
MAX_UNITS = 64  # Accumulated keys, including withdrawals.


def _need(condition, code):
    if not condition:
        raise ValueError(code)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _revision(value):
    return type(value) is int and value >= 0


def _unique(values):
    return list(dict.fromkeys(values))


def _same_content(before, after):
    return ({name: value for name, value in before.items() if name != 'grounding'} ==
            {name: value for name, value in after.items() if name != 'grounding'})


def _merge_grounding(previous, current):
    unique = {}
    for source in [*previous, *current]:
        unique[json.dumps(source, sort_keys=True, ensure_ascii=False)] = deepcopy(source)
    return list(unique.values())


def _citations(values):
    _need(isinstance(values, list) and bool(values), 'AGENT_MEMORY_EVIDENCE_REQUIRED')
    keys = []
    for value in values:
        _need(isinstance(value, dict) and set(value) in ({'source_id'}, {'source_id', 'start_line', 'end_line'})
              and _text(value['source_id']), 'AGENT_MEMORY_CITATION_INVALID')
        if 'start_line' in value:
            _need(type(value['start_line']) is int and type(value['end_line']) is int
                  and 1 <= value['start_line'] <= value['end_line'], 'AGENT_MEMORY_CITATION_INVALID')
        keys.append((value['source_id'], value.get('start_line'), value.get('end_line')))
    _need(len(set(keys)) == len(keys), 'AGENT_MEMORY_CITATION_DUPLICATE')
    return keys


def _fields(fields, category, *, stored=False):
    from extraction.agent_memory import memory_units
    _need(isinstance(fields, dict), 'AGENT_MEMORY_STRUCTURED_FIELDS_INVALID')
    if category == 'tool':
        _need(set(fields) == {'tool_name', 'units'} and _text(fields['tool_name'])
              and isinstance(fields['units'], list), 'AGENT_MEMORY_STRUCTURED_FIELDS_INVALID')
    else:
        _need(not fields or set(fields) == _ITEM_FIELDS[category], 'AGENT_MEMORY_STRUCTURED_FIELDS_INVALID')
    units = memory_units(category, fields)
    _need(len(units) <= MAX_UNITS, 'AGENT_MEMORY_UNIT_LIMIT')
    items = {}
    for item in units:
        expected = _ITEM_FIELDS[category]
        _need(isinstance(item, dict) and set(item) == expected
              and all(_text(item[name]) for name in expected-{'grounding'})
              and isinstance(item['grounding'], list) and bool(item['grounding']),
              'AGENT_MEMORY_UNIT_INVALID')
        _need(item['key'] not in items, 'AGENT_MEMORY_UNIT_KEY_DUPLICATE')
        if not stored:
            _citations(item['grounding'])
        items[item['key']] = deepcopy(item)
    return items


def _grounding(key, evidence, context, ctx, category, tool_name=None):
    """Match exact citations to the server's evidence reconstruction."""
    citations = _citations(evidence)
    groups = context['grounding'].get(key)
    _need(isinstance(groups, list) and len(groups) == len(citations), 'AGENT_MEMORY_GROUNDING_MISSING')
    matched, result = set(), []
    for source_id, first, last in citations:
        source = context['sources'].get(source_id)
        _need(isinstance(source, dict) and _text(source.get('uri'))
              and source['uri'].startswith(f'ctx://{ctx.account_id}/')
              and isinstance(source.get('content_hash'), str) and _SHA.fullmatch(source['content_hash'])
              and _text(source.get('resource_version')) and _text(source.get('role'))
              and _text(source.get('origin')), 'AGENT_MEMORY_SOURCE_BINDING_INVALID')
        _need(source.get('lane') == category, 'AGENT_MEMORY_SOURCE_LANE_MISMATCH')
        if category == 'tool':
            _need(source.get('tool_name') == tool_name, 'AGENT_MEMORY_TOOL_IDENTITY_MISMATCH')
        found = [i for i, group in enumerate(groups) if isinstance(group, dict)
                 and group.get('uri') == source['uri']
                 and group.get('content_hash') == source['content_hash']
                 and group.get('resource_version') == source['resource_version']
                 and group.get('start_line') == first and group.get('end_line') == last
                 and (_text(group.get('quote')) if first is not None else 'quote' not in group)
                 and group.get('source_id', source_id) == source_id]
        _need(len(found) == 1 and found[0] not in matched, 'AGENT_MEMORY_GROUNDING_MISMATCH')
        matched.add(found[0])
        group = deepcopy(groups[found[0]])
        for name in ('role', 'origin', 'locator', 'captured_at', 'lane', 'tool_name', 'tool_use_id', 'sequence'):
            if name in group and name in source:
                _need(group[name] == source[name], 'AGENT_MEMORY_SOURCE_METADATA_MISMATCH')
            if name in source:
                group[name] = deepcopy(source[name])
        group['source_id'] = source_id
        result.append(group)
    return result


def _state(existing, category, fields, ctx):
    if existing is None:
        return {}, dict(version=1, units={}, applied_operations=[], revisions=[], producer_agent_ids=[])
    _need(existing.context_type == 'MEMORY' and existing.category == category
          and existing.owner_space == ctx.agent_space_name(), 'AGENT_MEMORY_EXISTING_OWNER_OR_CATEGORY_MISMATCH')
    from extraction.agent_memory import node_memory_fields
    old_fields = node_memory_fields(existing)
    old_items = _fields(old_fields, category, stored=True)
    if category == 'tool':
        _need(old_fields['tool_name'] == fields['tool_name'], 'AGENT_MEMORY_TOOL_IDENTITY_MISMATCH')
    state = deepcopy(existing.metadata.get('agent_memory'))
    _need(isinstance(state, dict) and type(state.get('version')) is int and state['version'] == 1
          and isinstance(state.get('units'), dict) and isinstance(state.get('applied_operations'), list)
          and isinstance(state.get('revisions'), list) and isinstance(state.get('producer_agent_ids'), list),
          'AGENT_MEMORY_EXISTING_STATE_INVALID')
    _need(len(state['units']) <= MAX_UNITS, 'AGENT_MEMORY_UNIT_LIMIT')
    _need(all(_text(value) for value in state['applied_operations']+state['producer_agent_ids'])
          and len(set(state['applied_operations'])) == len(state['applied_operations']),
          'AGENT_MEMORY_EXISTING_OPERATIONS_INVALID')
    for key, value in state['units'].items():
        _need(_text(key) and isinstance(value, dict) and _revision(value.get('revision'))
              and value['revision'] > 0 and isinstance(value.get('grounding'), list)
              and bool(value['grounding']), 'AGENT_MEMORY_EXISTING_UNIT_STATE_INVALID')
    _need(set(old_items) <= set(state['units']), 'AGENT_MEMORY_EXISTING_UNIT_REVISION_MISSING')
    return old_items, state


class AgentMemoryPolicy(MergePolicy):
    def __init__(self, fs: ContextFS, uri_resolver):
        self._fs = fs
        self._uri_resolver = uri_resolver

    def plan(self, candidate: CandidateMemory, ctx: RequestContext) -> WritePlan:
        _need(candidate.category in _ITEM_FIELDS and candidate.owner_scope == 'agent', 'AGENT_MEMORY_CATEGORY_OR_OWNER_INVALID')
        fields = getattr(candidate, 'structured_fields', None)
        incoming = _fields(fields, candidate.category)
        context = getattr(candidate, 'memory_context', None)
        _need(isinstance(context, dict)
              and set(context) == {'operation_id', 'base_units', 'removals', 'corrections', 'sources', 'grounding'}
              and _text(context['operation_id']) and isinstance(context['base_units'], dict)
              and isinstance(context['removals'], list) and isinstance(context['corrections'], list)
              and isinstance(context['sources'], dict)
              and isinstance(context['grounding'], dict), 'AGENT_MEMORY_SERVER_CONTEXT_REQUIRED')
        _need(all(_text(key) and _revision(value) for key, value in context['base_units'].items()),
              'AGENT_MEMORY_BASE_UNIT_INVALID')
        target = self._uri_resolver.resolve(candidate.category,
            {'routing_key': candidate.routing_key}, ctx)
        _need(isinstance(target, str) and target.startswith(f'ctx://{ctx.account_id}/') and '..' not in target,
              'AGENT_MEMORY_TARGET_INVALID')
        existing = self._fs.read_node(target, ctx) if self._fs.exists(target, ctx) else None
        old_items, state = _state(existing, candidate.category, fields, ctx)
        if context['operation_id'] in state['applied_operations']:
            return WritePlan(action='skip', target_uri=target)
        removals = {}
        for removal in context['removals']:
            _need(isinstance(removal, dict) and set(removal) == {'key', 'reason', 'grounding'}
                  and _text(removal['key']) and _text(removal['reason']), 'AGENT_MEMORY_REMOVAL_INVALID')
            key = removal['key']
            _need(key not in incoming and key not in removals, 'AGENT_MEMORY_MUTATION_KEY_CONFLICT')
            removals[key] = removal
        corrections = {}
        for correction in context['corrections']:
            _need(isinstance(correction, dict) and set(correction) == {'key', 'reason'}
                  and _text(correction['key']) and _text(correction['reason'])
                  and len(correction['reason']) <= 300
                  and correction['key'] in incoming and correction['key'] not in corrections,
                  'AGENT_MEMORY_CORRECTION_INVALID')
            corrections[correction['key']] = correction['reason']
        affected = [*incoming, *removals]
        groundings = {key: _grounding(key, (incoming.get(key) or removals[key])['grounding'], context, ctx,
                                      candidate.category, fields.get('tool_name'))
                      for key in affected}
        for key, item in incoming.items():
            item['grounding'] = deepcopy(groundings[key])
        _need(len(set(state['units']) | set(incoming)) <= MAX_UNITS, 'AGENT_MEMORY_UNIT_LIMIT')
        revision_keys = (set(context['base_units']) | set(state['units']) | set(affected)
                         if candidate.category == 'experience' else affected)
        for key in revision_keys:
            current = state['units'].get(key, {}).get('revision', 0)
            if context['base_units'].get(key, 0) != current:
                raise AgentMemoryConflictError('AGENT_MEMORY_STALE_UNIT_REVISION')
        required_corrections = {key for key, after in incoming.items()
                                if key in state['units'] and
                                (key not in old_items or not _same_content(old_items[key], after))}
        missing = sorted(required_corrections - corrections.keys())
        _need(not missing, 'AGENT_MEMORY_CORRECTION_REQUIRED: ' + ', '.join(missing))
        extra = sorted(corrections.keys() - required_corrections)
        _need(not extra, 'AGENT_MEMORY_CORRECTION_NOT_APPLICABLE: ' + ', '.join(extra))
        merged = deepcopy(old_items)
        changed = False
        for key in affected:
            before = old_items.get(key)
            old_grounding = state['units'].get(key, {}).get('grounding', [])
            if key in removals and before is None:
                continue
            after = incoming.get(key)
            new_grounding = groundings[key]
            action = 'remove' if key in removals else 'correction' if key in corrections else 'insert'
            if before is not None and after is not None and _same_content(before, after):
                after = deepcopy(after)
                action = 'evidence_extend'
                new_grounding = _merge_grounding(old_grounding, new_grounding)
                after['grounding'] = deepcopy(new_grounding)
            if after is not None and before == after and old_grounding == new_grounding:
                continue
            revision = state['units'].get(key, {}).get('revision', 0) + 1
            state['units'][key] = {'revision': revision, 'grounding': deepcopy(new_grounding), 'withdrawn': after is None}
            entry = dict(key=key, revision=revision, action=action,
                         operation_id=context['operation_id'], producer_agent_id=ctx.agent_id)
            if action == 'evidence_extend':
                entry['added_grounding'] = [deepcopy(source) for source in groundings[key]
                                           if source not in old_grounding]
            else:
                entry.update(before=deepcopy(before), after=deepcopy(after),
                             before_grounding=deepcopy(old_grounding), after_grounding=deepcopy(new_grounding))
                if key in removals:
                    entry['reason'] = removals[key]['reason']
                elif key in corrections:
                    entry['reason'] = corrections[key]
            state['revisions'].append(entry)
            if after is None:
                merged.pop(key, None)
            else:
                merged[key] = deepcopy(after)
            changed = True
        if not changed:
            return WritePlan(action='skip', target_uri=target)
        state['applied_operations'].append(context['operation_id'])
        _need(_text(ctx.agent_id), 'AGENT_MEMORY_PRODUCER_REQUIRED')
        state['producer_agent_ids'] = _unique([*state['producer_agent_ids'], ctx.agent_id])
        if candidate.category == 'tool':
            full = {'tool_name': fields['tool_name'], 'units': list(merged.values())}
        else:
            _need(len(merged) <= 1, 'AGENT_MEMORY_EXPERIENCE_REQUIRES_ONE_CONCLUSION')
            full = next(iter(merged.values()), {})
        archived_version = None
        if existing is None and callable(getattr(type(self._fs), 'archived_node_version', None)):
            archived_version = self._fs.archived_node_version(target, ctx)
        version = existing.metadata.get('version', 0) if existing else (archived_version or 0)
        _need(_revision(version), 'AGENT_MEMORY_NODE_REVISION_INVALID')
        previous = existing.metadata.get('provenance_ids', []) if existing else []
        _need(isinstance(previous, list) and isinstance(candidate.provenance_ids, list)
              and all(_text(value) for value in previous+candidate.provenance_ids), 'AGENT_MEMORY_PROVENANCE_INVALID')
        return WritePlan(action='merge' if existing or archived_version is not None else 'create', target_uri=target,
            merged_fields={'agent_memory_fields': full, 'agent_memory_context': state,
                           'expected_version': version,
                           'provenance_ids': _unique([*previous, *candidate.provenance_ids])},
            relation_edges=[])


class AgentMemoryArchiveBuilder:
    def __init__(self, llm):
        self.llm = llm

    def build(self, candidate, plan, ctx):
        from datetime import UTC, datetime

        from core.models import ContextNode
        from extraction.agent_memory import (
            memory_units,
            render_agent_memory,
            summarize_agent_memory,
        )
        fields = deepcopy(plan.merged_fields["agent_memory_fields"])
        state = deepcopy(plan.merged_fields["agent_memory_context"])
        grounding = [g for unit in memory_units(candidate.category, fields) for g in unit["grounding"]]
        abstract, overview = (summarize_agent_memory(candidate.category, fields, self.llm)
                              if memory_units(candidate.category, fields) else ('', ''))
        metadata = {
            "routing_key": candidate.routing_key, "agent_id": ctx.agent_id,
            "agent_memory_version": 1, "agent_memory_content_version": 3,
            "agent_memory_layers_version": 2,
            "structured_fields": fields, "agent_memory": state,
            "agent_memory_value_review": deepcopy(candidate.value_review),
            "provenance_ids": plan.merged_fields["provenance_ids"],
            "derived_from": sorted({g["uri"] for g in grounding}),
            "_expected_version": plan.merged_fields["expected_version"],
            "updated_at": datetime.now(UTC).isoformat(),
        }
        return ContextNode(
            plan.target_uri, "MEMORY", candidate.category, 4, ctx.agent_space_name(),
            abstract, overview, render_agent_memory(candidate.category, fields, grounding=grounding), metadata,
        )
