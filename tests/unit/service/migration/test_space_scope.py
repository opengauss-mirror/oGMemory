import pytest

from core.provenance_resolver import ProvenanceResolver
from service.migration.engine import inventory
from service.migration.package import PackageError
from service.migration.space_scope import discard_shared
from tests.unit.service.migration.test_engine import node


@pytest.mark.parametrize('referenced', [False, True])
def test_skipping_shared_content_cannot_leave_ordinary_evidence_dangling(tmp_path, referenced):
    shared = {**node(), 'uri': 'ctx://acme/spaces/team/memories/experience/item',
              'owner_space': 'space:team', 'category': 'experience'}
    ordinary = node()
    if referenced:
        ordinary['metadata']['provenance_ids'] = [ProvenanceResolver.build_id('memory', shared['uri'], 'L2')]
    with inventory(tmp_path / 'scope.db', lambda kind: [ordinary] if kind == 'nodes' else (),
                   shared_records=[shared]) as inv:
        if referenced:
            with pytest.raises(PackageError, match='depend on skipped'):
                discard_shared(inv)
        else:
            discard_shared(inv)
            assert list(inv.records('nodes')) == [ordinary]


def test_shared_records_cannot_enter_the_ordinary_package_dataset(tmp_path):
    shared = {**node(), 'uri': 'ctx://acme/spaces/team/memories/experience/item',
              'owner_space': 'space:team'}
    with pytest.raises(PackageError, match='owner'):
        with inventory(tmp_path / 'scope.db', lambda kind: [shared] if kind == 'nodes' else ()):
            pass
