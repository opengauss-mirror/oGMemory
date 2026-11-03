from pathlib import Path
import importlib.util
import pytest


def smoke():
    path = Path(__file__).resolve().parents[3] / 'scripts/compose_memory_smoke.py'
    assert path.is_file(), 'memory smoke verifier is required'
    spec = importlib.util.spec_from_file_location('memory_smoke', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.parametrize('result', [
    {'ok': True, 'status': 'accepted'},
    {'ok': True, 'status': 'completed', 'writes_completed': 0},
    {'ok': True, 'status': 'completed', 'writes_completed': 1, 'writes_failed': 1},
    {'ok': False, 'status': 'completed', 'writes_completed': 1},
])
def test_acceptance_does_not_count_as_completed_write(result):
    with pytest.raises(RuntimeError):
        smoke().verify_write(result)


def test_completed_atomic_write_is_accepted():
    smoke().verify_write({'ok': True, 'status': 'completed', 'writes_completed': 1, 'writes_failed': 0})


def test_recall_waits_for_index_and_reads_persisted_content():
    module = smoke()
    calls = []
    queries = 0

    def post(route, payload):
        nonlocal queries
        calls.append(route)
        if route == 'memories/query':
            queries += 1
            return {'hits': [] if queries == 1 else [{'uri': 'agfs://atomic-memory'}]}
        if route == 'memories/read':
            assert payload['level'] == 'content'
            return {'content': 'marker-test 使用 Python'}
        pytest.fail('unsupported HTTP route: ' + route)

    uri = module.wait_for_recall(post, 'marker-test', timeout=1, interval=0)
    assert uri == 'agfs://atomic-memory'
    assert calls == ['memories/query', 'memories/query', 'memories/read']


def test_query_echo_is_not_persistent_recall():
    module = smoke()
    with pytest.raises(RuntimeError):
        module.wait_for_recall(lambda route, payload: {'query': 'marker-test Python', 'hits': []},
                               'marker-test', timeout=.01, interval=0)
