"""Exercise real identity resolution, not just a mock write policy."""
from dataclasses import replace
from tests.unit.commit.test_memory_identity import MemoryFS, DecisionLLM, api, memory
from core.models import RequestContext
from index.activation_projection import ACTIVATION_VERSION

def meta(text):
    return dict(update_activation=text, update_activation_version=ACTIVATION_VERSION)

def test_identity_update_keeps_new_descriptor():
    ctx = RequestContext('test', 'u', 'a', 's', 't')
    fs = MemoryFS()
    writer = api(fs, DecisionLLM())
    first = memory('topic', content='First fact.', metadata=meta('Old descriptor.'))
    uri = writer.write_memory(first, ctx)['target_uri']
    writer.write_memory(replace(first, target_uri=uri, expected_version=1,
        content='New fact.', overview='New fact.', abstract='New fact.',
        metadata=meta('New descriptor.')), ctx)
    stored = fs.read_node(uri, ctx)
    assert stored.content == 'New fact.'
    assert stored.metadata['update_activation'] == 'New descriptor.'

def test_descriptor_only_refresh_preserves_content_and_remains_idempotent():
    ctx = RequestContext('test', 'u', 'a', 's', 't')
    fs = MemoryFS()
    writer = api(fs, DecisionLLM())
    first = memory('topic', content='Same fact.', metadata=meta('Old descriptor.'))
    uri = writer.write_memory(first, ctx)['target_uri']
    update = replace(first, metadata=meta('New descriptor.'))
    assert writer.write_memory(update, ctx)['action'] == 'merge'
    stored = fs.read_node(uri, ctx)
    assert stored.content == first.content and stored.metadata['version'] == 2
    assert stored.metadata['update_activation'] == 'New descriptor.'
    assert writer.write_memory(update, ctx)['action'] == 'skip'
    assert fs.read_node(uri, ctx).metadata['version'] == 2
    # Missing or malformed descriptors must not erase/replace valid metadata.
    for metadata in ({}, meta('bad\nmultiline')):
        assert writer.write_memory(replace(first, metadata=metadata), ctx)['action'] == 'skip'
    assert fs.read_node(uri, ctx).metadata['update_activation'] == 'New descriptor.'
