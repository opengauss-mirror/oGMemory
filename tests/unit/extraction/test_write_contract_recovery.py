"""Write-time summary and read-scope contracts fail before storage or acceptance."""
import pytest

from core.errors import ToolCallProtocolError
from core.uri_resolver import URIResolver
from extraction.extraction_react_loop import ExtractionReActLoop
from extraction.tools import Extractor, _Span
from tests.unit.commit.test_memory_identity import MemoryFS, api, memory
from tests.unit.commit.test_profile_merge import profile, profile_context
from tests.unit.extraction.test_memory_update_flow import ExtractionProvider, operation, setup_flow

LONG = "Nora lives in Utrecht and works as an engineer. " * 6


class SummaryProvider:
    def __init__(self, bad=False):
        self.calls = []
        self.bad = bad

    def complete_json(self, prompt, schema):
        self.calls.append(prompt)
        if 'repairs' in schema.get('properties', {}):
            return {'repairs': [{'operation_index': 0, 'abstract': LONG if self.bad else 'Nora in Utrecht: engineer.'}]}
        return {'abstract': LONG, 'overview': 'Nora lives in Utrecht; she is an engineer.',
                'content': 'Nora lives in Utrecht. Nora is an engineer.'}


def profile_write(fs, ctx, provider):
    writer = api(fs, provider)
    uri = writer.write_memory(profile('identity', 'Nora lives in Utrecht.'), ctx)['target_uri']
    writer.write_memory(profile('occupation', 'Nora is an engineer.'), ctx)
    return fs.read_node(uri, ctx)


def test_write_time_profile_summary_recovers_once_without_losing_original():
    provider = SummaryProvider()
    node = profile_write(MemoryFS(), profile_context(), provider)
    assert len(provider.calls) == 2
    assert len(node.abstract) <= 200 and LONG in node.content
    assert node.metadata['version'] == 2
    assert node.metadata['provenance_ids'] == ['source:identity', 'source:occupation']


def test_invalid_write_time_repair_never_writes():
    fs, ctx, provider = MemoryFS(), profile_context(), SummaryProvider(bad=True)
    with pytest.raises(Exception):
        profile_write(fs, ctx, provider)
    node = next(iter(fs.nodes.values()))
    assert node.content == 'Nora lives in Utrecht.' and node.metadata['version'] == 1
    assert len(provider.calls) == 2


@pytest.mark.parametrize('mode', ['eager', 'lazy'])
def test_changed_read_scope_rejected_during_extraction(mode):
    ctx, fs, registry, uri, embedder, index, prefetcher = setup_flow('entity', memory('atlas', category='entity', content='Atlas is blue.'))
    op = operation('entity', 'atlas', 'Atlas is blue and fast.', uri)
    op['input']['scope_ref'] = 'New conversation date'
    provider = ExtractionProvider(op)
    extractor = Extractor(provider, mode=mode, fs=fs, schema_registry=registry,
                          uri_resolver=URIResolver(registry), embedder=embedder, vector_index=index)
    if mode == 'eager':
        prepared = extractor._prepare_span(_Span(0, 0, categories=['entity']),
            [{'role': 'user', 'content': 'Atlas is fast.'}], 'en', None, '', '', ctx)
        with pytest.raises(ToolCallProtocolError, match='scope'):
            extractor._structure_prepared_span(prepared, ctx)
    else:
        loop = ExtractionReActLoop(provider, fs, uri_resolver=URIResolver(registry), registry=registry, prompt_manager=extractor._prompt_manager)
        loop._read_versions = {uri: 1}
        loop._read_nodes = {uri: fs.read_node(uri, ctx)}
        result = loop._execute_tool(op['name'], op['input'], ctx)
        assert 'scope' in result.get('error', '')
    assert fs.read_node(uri, ctx).metadata['version'] == 1
