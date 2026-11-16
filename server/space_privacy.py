"""Deterministic redaction for generated team-memory text."""
from __future__ import annotations

import re
from functools import lru_cache
from heapq import merge

_PATTERNS = (
    (re.compile(r"(?<![A-Za-z0-9.!#$%&'*+/=?^_`{|}~-])[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+"), '[邮箱]'),
    (re.compile(r'(?<!\d)(?:\+?86[ -]?)?1[3-9]\d[ -]?\d{4}[ -]?\d{4}(?!\d)'), '[手机号]'),
    (re.compile(r'(?<![A-Za-z0-9])[1-9]\d{5}(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?![A-Za-z0-9])'), '[身份证号]'),
)
_MARKERS = re.compile(r'(\[(?:邮箱|手机号|身份证号|已遮蔽)\])')


@lru_cache(maxsize=256)
def _terms_pattern(terms):
    alternatives = '|'.join(re.escape(term) for term in sorted(terms, key=len, reverse=True))
    return re.compile(f'(?=({alternatives}))', re.IGNORECASE)


def _matches(pattern, text, replacement, group=0):
    for match in pattern.finditer(text):
        yield (*match.span(group), replacement)


def _custom_matches(text, custom):
    markers = iter(_MARKERS.finditer(text))
    marker = next(markers, None)
    for start, stop, label in _matches(custom, text, '[已遮蔽]', 1):
        while marker is not None and marker.end() <= start:
            marker = next(markers, None)
        if marker is None or start < marker.start() or stop > marker.end():
            yield start, stop, label


def _redact_segment(text, custom):
    streams = [_matches(pattern, text, label) for pattern, label in _PATTERNS]
    streams.append(_custom_matches(text, custom))
    streams.append(((*match.span(), match.group()) for match in _MARKERS.finditer(text)))
    parts, left, end, label = [], 0, 0, ''
    # Match original text and merge overlapping ranges before emitting replacements.
    for start, stop, replacement in merge(*streams, key=lambda span: span[0]):
        if start < end:
            if start == left and stop >= end:
                label = replacement
            elif stop > end:
                label = '[已遮蔽]'
            end = max(end, stop)
        else:
            parts.extend((label, text[end:start]))
            left, end, label = start, stop, replacement
    parts.extend((label, text[end:]))
    return ''.join(parts)


def redact_text(text: str, policy: dict | None = None) -> str:
    terms = tuple((policy or {}).get('redact_terms', ()))
    if terms:
        return _redact_segment(text, _terms_pattern(terms))
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _redact_value(value, policy):
    if isinstance(value, str):
        return redact_text(value, policy)
    if isinstance(value, list):
        return [_redact_value(entry, policy) for entry in value]
    if isinstance(value, dict):
        return {redact_text(key, policy): _redact_value(entry, policy) for key, entry in value.items()}
    return value


def redact_item(item: dict, policy: dict | None = None) -> dict:
    result = dict(item)
    for key in ('content', 'abstract', 'overview', 'title', 'topic'):
        if key in result:
            result[key] = _redact_value(result[key], policy)
    if isinstance(result.get('metadata'), dict):
        metadata = dict(result['metadata'])
        for key in ('fields', 'description', 'topic', 'title'):
            if key in metadata:
                metadata[key] = _redact_value(metadata[key], policy)
        result['metadata'] = metadata
    return result
