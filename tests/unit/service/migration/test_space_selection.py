from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from service.migration.package import PackageError
from service.migration.space_selection import confirm_selection, select_spaces


@pytest.mark.parametrize(('case', 'reason'), [
    ('disabled', 'target_shared_spaces_disabled'),
    ('backend', 'content_backend_differs'),
    ('database', 'database_type_differs'),
    ('schema', 'shared_schema_differs'),
    ('business', 'target_business_not_empty'),
    ('catalog', 'target_business_not_empty'),
    ('vector', 'target_business_not_empty'),
    ('identity', 'target_identity_missing'),
    ('eligible', None),
])
def test_whole_space_selection(case, reason, monkeypatch):
    source = {'database_type': 'opengauss', 'schema_sha256': 'same'}
    target = dict(source)
    if case == 'database':
        target['database_type'] = 'postgresql'
    if case == 'schema':
        target['schema_sha256'] = 'different'
    catalog = Mock()
    catalog.description.return_value = target
    catalog.records.return_value = iter([{}] if case == 'catalog' else [])
    monkeypatch.setattr('service.migration.space_selection.SpaceCatalog', lambda adapter: catalog)
    spaces = Mock()
    spaces.count.return_value = 2
    spaces.db.execute.return_value.fetchone.return_value = (3,)
    spaces.target_identity_error.return_value = 'target_identity_missing' if case == 'identity' else None
    existing = Mock()
    existing.db.execute.return_value.fetchone.return_value = (1,) if case == 'business' else None
    vector = Mock()
    vector.count.return_value = int(case == 'vector')
    result = select_spaces({'source_backend': 'sql', 'shared_spaces': source}, spaces,
        SimpleNamespace(shared_spaces_enabled=case != 'disabled'),
        SimpleNamespace(backend='agfs' if case == 'backend' else 'sql', conn=None), existing, vector)
    assert result == {'spaces': 2, 'nodes': 3, 'action': 'skipped' if reason else 'import', 'reason': reason}


@pytest.mark.parametrize(('current', 'confirmed'), [('import', 'skipped'), ('skipped', 'import')])
def test_selection_changes_require_confirmation_in_both_directions(current, confirmed):
    with pytest.raises(PackageError, match='preflight'):
        confirm_selection({'spaces': 1, 'action': current}, {'spaces': 1, 'action': confirmed})
