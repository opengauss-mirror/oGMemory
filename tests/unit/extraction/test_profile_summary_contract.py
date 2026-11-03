"""Profile reconciliation must honor the same bounded summary contract as native output."""

from copy import deepcopy

import pytest

from core.errors import ToolCallProtocolError
from core.uri_resolver import URIResolver
from extraction.tools import Extractor
from tests.unit.commit.test_memory_identity import api
from tests.unit.extraction.test_profile_batch_recovery import NEW, OLD, fixture


class SummaryProvider:
    def __init__(self, field, value, repair=None):
        self.field = field
        self.value = value
        self.repair = repair
        self.calls = []

    def complete_json(self, prompt, schema):
        self.calls.append((prompt, schema))
        if 'repairs' in schema['properties']:
            return (self.repair if self.repair is not None else
                    {'repairs': [{'operation_index': 0, self.field: 'Nora profile'}]})
        return {**dict.fromkeys(('abstract', 'overview', 'content'), NEW), self.field: self.value}


@pytest.mark.parametrize('field,value', [('abstract', 'Nora profile detail. ' * 13),
                                        ('overview', 'Nora profile detail. ' * 260)])
def test_profile_merge_repairs_only_oversized_field_and_preserves_identity(field, value):
    ctx, fs, registry, uri, delta, snapshot, span = fixture()
    before = deepcopy((delta, snapshot))
    llm = SummaryProvider(field, value)
    extractor = Extractor(llm, fs=fs, schema_registry=registry, uri_resolver=URIResolver(registry))
    candidate, = extractor._merge_span_results([(span, [delta, snapshot])], ctx)
    assert len(llm.calls) == 2
    assert getattr(candidate, field) == 'Nora profile'
    assert value in candidate.content and NEW in candidate.content
    assert candidate.target_uri == uri and candidate.expected_version == 1
    assert set(candidate.provenance_ids) == {'source:delta', 'source:snapshot'}
    assert (delta, snapshot) == before
    api(fs, llm).write_memory(candidate, ctx)
    assert fs.read_node(uri, ctx).metadata['version'] == 2
    assert fs.read_node(uri, ctx).content == candidate.content


@pytest.mark.parametrize('repair', [
    {'repairs': [{'operation_index': 0, 'abstract': 'x' * 201}]},
    {'repairs': [{'operation_index': 0, 'abstract': 'short', 'content': 'changed'}]},
    {'repairs': [{'operation_index': 1, 'abstract': 'short'}]},
])
def test_bad_profile_repair_fails_without_writing(repair):
    ctx, fs, registry, uri, delta, snapshot, span = fixture()
    llm = SummaryProvider('abstract', 'a' * 249, repair)
    extractor = Extractor(llm, fs=fs, schema_registry=registry, uri_resolver=URIResolver(registry))
    with pytest.raises(ToolCallProtocolError):
        extractor._merge_span_results([(span, [delta, snapshot])], ctx)
    assert len(llm.calls) == 2
    assert fs.read_node(uri, ctx).content == OLD
