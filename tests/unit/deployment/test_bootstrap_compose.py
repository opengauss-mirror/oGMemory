from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import pytest


def bootstrap():
    spec = spec_from_file_location('bootstrap', Path(__file__).resolve().parents[3] / 'scripts/bootstrap_compose.py')
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_identity_setup_does_not_rotate_keys_or_reassign_agent():
    m = bootstrap()
    writes = []
    def call(method, route, payload=None):
        if method == 'GET':
            return {'users': [{'user_id': 'user'}]} if route.endswith('users') else {
                'agents': [{'agent_id': 'agent', 'owner_user_id': 'user'}]}
        writes.append(payload)
    m.ensure_identity(call, 'account', 'user', 'agent')
    assert writes == []


def test_new_identity_is_registered_in_owner_order():
    writes = []
    def call(method, route, payload=None):
        if method == 'GET': return {'users': [], 'agents': []}
        writes.append(payload)
    bootstrap().ensure_identity(call, 'account', 'user', 'agent')
    assert writes == [{'user_id': 'user', 'role': 'user'}, {'agent_id': 'agent', 'owner_user_id': 'user'}]


def test_existing_agent_owner_mismatch_is_rejected():
    def call(method, route, payload=None):
        assert method == 'GET'
        return {'users': [{'user_id': 'user'}], 'agents': [{'agent_id': 'agent', 'owner_user_id': 'other'}]}
    with pytest.raises(ValueError, match='owner'):
        bootstrap().ensure_identity(call, 'account', 'user', 'agent')
