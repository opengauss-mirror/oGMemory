"""Validation and canonical snapshots for shared-memory rules."""
from __future__ import annotations

import copy
import hashlib
import json
import re


class SpaceError(ValueError):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code


def identifier(value: str, name: str = 'identifier', maximum: int = 64) -> str:
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value) or len(value) > maximum:
        raise SpaceError(400, 'INVALID_ARGUMENT', f'Invalid {name}')
    return value


def principal(value: str) -> tuple[str, str]:
    kind, separator, identity = str(value).partition(':')
    if not separator or kind not in ('user', 'agent') or not identity or len(identity) > 128:
        raise SpaceError(400, 'INVALID_ARGUMENT', 'Expected user:id or agent:id')
    return kind, identity


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _number(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
        raise SpaceError(400, 'INVALID_ARGUMENT', f'{name} must be between {low} and {high}')
    return value


def validate_acl(value: dict) -> dict:
    if isinstance(value, dict) and 'acl_mode' in value:
        value = {**value, 'mode': value['acl_mode']}
    if not isinstance(value, dict) or value.get('mode', 'inherit') not in ('inherit', 'restricted'):
        raise SpaceError(400, 'INVALID_ARGUMENT', 'Invalid read ACL')
    readers = value.get('readers', [])
    if not isinstance(readers, list) or len(readers) > 1000:
        raise SpaceError(400, 'INVALID_ARGUMENT', 'readers must contain at most 1000 principals')
    for reader in readers:
        principal(reader)
    return {'mode': value.get('mode', 'inherit'), 'readers': sorted(set(readers))}


def memory_definition(spec: dict) -> dict:
    """Read text rules while preserving historical snapshots."""
    return {'description': spec.get('description', spec.get('instructions', '')),
            'privacy_policy': copy.deepcopy(spec.get('privacy_policy', {'redact_terms': []}))}


def validate_rule(spec: dict) -> dict:
    if not isinstance(spec, dict):
        raise SpaceError(400, 'INVALID_ARGUMENT', 'spec must be an object')
    result = copy.deepcopy(spec)
    if 'source_subjects' in result:
        sources = result['source_subjects']
        if not isinstance(sources, list) or not 1 <= len(sources) <= 100:
            raise SpaceError(400, 'INVALID_ARGUMENT', 'source_subjects requires 1 to 100 principals')
        for source in sources:
            principal(source)
        result['source_subjects'] = sorted(set(sources))
    types = result.get('memory_types', [])
    if not isinstance(types, list) or not 1 <= len(types) <= 32:
        raise SpaceError(400, 'INVALID_ARGUMENT', 'memory_types requires 1 to 32 types')
    for memory_type in types:
        identifier(memory_type, 'memory type')
    result['memory_types'] = sorted(set(types))
    if result.get('output_kind', 'experience') != 'experience':
        raise SpaceError(400, 'INVALID_ARGUMENT', 'Team rules currently generate text memories only')
    result.setdefault('output_kind', 'experience')
    definition = memory_definition(result)
    description = definition['description']
    if not isinstance(description, str) or not description.strip() or len(description.encode()) > 16384:
        raise SpaceError(400, 'INVALID_ARGUMENT', 'description requires 1 to 16384 bytes')
    result.update(definition)
    for key in ('instructions', 'output_schema', 'fields'):
        result.pop(key, None)
    result.setdefault('merge_policy', 'by_topic')
    if result['merge_policy'] not in ('by_topic', 'append'):
        raise SpaceError(400, 'INVALID_ARGUMENT', 'merge_policy must be by_topic or append')
    privacy = result['privacy_policy']
    if not isinstance(privacy, dict) or set(privacy) - {'redact_terms'}:
        raise SpaceError(400, 'INVALID_ARGUMENT', 'privacy_policy supports redact_terms')
    terms = privacy.get('redact_terms', [])
    if (not isinstance(terms, list) or len(terms) > 100
            or any(not isinstance(term, str) or not 1 <= len(term.strip()) <= 128 for term in terms)):
        raise SpaceError(400, 'INVALID_ARGUMENT', 'redact_terms requires up to 100 terms of 1 to 128 characters')
    result['privacy_policy'] = {'redact_terms': sorted(set(term.strip() for term in terms))}
    policies: dict[str, dict[str, tuple[float, float, float]]] = {
        'trigger_policy': {'debounce_seconds': (30, 0, 86400), 'min_interval_seconds': (60, 0, 86400), 'reconcile_seconds': (300, 1, 86400)},
        'maintenance_policy': {'retention_days': (180, 1, 36500), 'archive_keep_days': (30, 0, 36500), 'half_life_days': (90, 1, 36500), 'min_age_days': (30, 0, 36500), 'archive_threshold': (0.2, 0, 1)},
        'quality_policy': {'min_sources': (1, 1, 100), 'min_score': (0.8, 0, 1)},
    }
    for group, definitions in policies.items():
        values = result.setdefault(group, {})
        if not isinstance(values, dict):
            raise SpaceError(400, 'INVALID_ARGUMENT', f'{group} must be an object')
        for name, (default, low, high) in definitions.items():
            values[name] = _number(values.get(name, default), name, low, high)
    result['output_acl'] = validate_acl(result.get('output_acl', {}))
    return result
