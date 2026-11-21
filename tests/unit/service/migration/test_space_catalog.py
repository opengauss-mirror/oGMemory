from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from service.migration.package import PackageError
from service.migration.space_catalog import normalize
from service.migration.space_inventory import SpaceInventory


def space(**changes):
    return dict(account_id='acme', space_id='team', name='Team', owner_user_id='alice',
                state='ACTIVE', authz_rev=1, created_at='2026-01-01T00:00:00+00:00',
                updated_at='2026-01-01T00:00:00+00:00', deleted_at=None, **changes)


def entry(table, row):
    return {'table': table, 'row': row}


def test_space_snapshot_normalizes_dates_without_mutating_source():
    row = space()
    row['created_at'] = datetime(2026, 1, 1, tzinfo=UTC)
    result = normalize('spaces', row)
    assert result['created_at'] == '2026-01-01T00:00:00+00:00'
    assert isinstance(row['created_at'], datetime)


@pytest.mark.parametrize('change', [
    {'password': 'must not be copied'}, {'name': 'a\0b'}, {'state': 'UNKNOWN'},
    {'authz_rev': True}, {'authz_rev': 0}, {'created_at': None}, {'space_id': ''},
])
def test_invalid_space_values_are_rejected(change):
    with pytest.raises(PackageError):
        normalize('spaces', {**space(), **change})


@pytest.mark.parametrize('table', ['cp_users', 'space_jobs', 'spaces;DROP TABLE cp_users', []])
def test_only_business_catalog_tables_are_accepted(table):
    with pytest.raises(PackageError):
        normalize(table, space())


def test_duplicate_spaces_are_rejected(tmp_path):
    with pytest.raises(PackageError, match='duplicate'):
        SpaceInventory(tmp_path / 'space.db', [entry('spaces', space())] * 2)


def test_orphan_member_is_rejected_before_restore(tmp_path):
    member = dict(account_id='acme', space_id='missing', principal='user:alice',
                  user_id='alice', agent_id=None, role='member',
                  created_at=space()['created_at'], updated_at=space()['updated_at'])
    inv = SpaceInventory(tmp_path / 'space.db', [entry('space_members', member)])
    try:
        with pytest.raises(PackageError, match='missing shared-space dependency'):
            inv.validate_dependencies(SimpleNamespace(db=None))
    finally:
        inv.close()


def test_complete_empty_space_is_valid_and_repeatably_readable(tmp_path):
    inv = SpaceInventory(tmp_path / 'space.db', [entry('spaces', space())])
    try:
        inv.validate_dependencies(SimpleNamespace(db=None))
        assert inv.count() == 1
        assert list(inv.records('spaces')) == [entry('spaces', space())]
        assert list(inv.records('spaces')) == list(inv.records('spaces'))
    finally:
        inv.close()
