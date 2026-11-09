"""Request-scoped node snapshots preserve staged and subsequent versions."""

from dataclasses import replace
import threading

from tests.unit.extraction.test_node_processing import card, operation, setup

def record_storage_threads(fs):
    calls = []
    for name in ("list_children", "read_node", "write_node", "exists"):
        original = getattr(fs, name)

        def recorded(*args, method=original, operation=name, **kwargs):
            calls.append((operation, threading.get_ident()))
            return method(*args, **kwargs)

        setattr(fs, name, recorded)
    return calls


def test_independent_groups_share_one_scan_per_subject_category():
    ctx, fs, llm, api = setup(lambda _: [operation([0])])
    calls = record_storage_threads(fs)
    drafts = [card(scope_ref=scope) for scope in ("project_a", "project_b")]
    result = api._node_processor().process(drafts, [], ctx)
    assert len(result) == 2
    assert len([name for name, _ in calls if name == "list_children"]) == 2
    assert {thread_id for _, thread_id in calls} == {threading.get_ident()}


def test_request_snapshots_reuse_reads_and_preserve_staged_and_next_request_versions():
    ctx, fs, llm, api = setup(lambda _: [])
    uri = api.write_memory(card(), ctx)["target_uri"]
    processor = api._node_processor()
    draft = replace(card(), target_uri=uri, expected_version=1)
    reads = []
    read_node = fs.read_node

    def read(target, request):
        reads.append(target)
        return read_node(target, request)

    fs.read_node = read
    cache = {"catalogs": {}, "reads": {}}
    first = processor._recall([draft], ctx, {}, cache)
    assert processor._recall([draft], ctx, {}, cache) == first
    staged = replace(first[0], content=first[0].content + "\n- It has a heart charm.")
    assert processor._recall([draft], ctx, {uri: staged}, cache)[0].content == staged.content
    assert reads.count(uri) == 1
    fs.write_node(staged, ctx)
    refreshed = processor._recall([replace(draft, expected_version=2)], ctx, {})
    assert refreshed[0].metadata["version"] == 2
    assert refreshed[0].content == staged.content
    assert reads.count(uri) == 2
