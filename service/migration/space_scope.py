"""Validated shared datasets joined only for common merge and index processing."""
from contextlib import contextmanager

from service.migration.engine import inventory
from service.migration.package import PackageError
from service.migration.shared_storage import portable_space_node
from service.migration.space_inventory import SpaceInventory


@contextmanager
def incoming_scope(package):
    spaces = SpaceInventory(package.root / 'spaces.sqlite', package.records('spaces'),
                            package.records('space_nodes'))
    try:
        nodes = (portable_space_node(node) for node in spaces.records('space_nodes'))
        with inventory(package.root / 'incoming.sqlite', package.records, shared_records=nodes) as incoming:
            spaces.validate_dependencies(incoming)
            # Recovery reopens the original package after a crash. Keep its root
            # scope stable even when the import later skips all shared records.
            incoming.recovery_accounts = frozenset(record['account_id'] for kind in ('nodes', 'archives')
                                                   for record in incoming.records(kind))
            yield incoming, spaces
    finally:
        spaces.close()


def shared_records(cfg, adapter, assets):
    if getattr(cfg, 'shared_spaces_enabled', False):
        for node in adapter.records('space_nodes'):
            yield assets.capture(node, copy_bytes=False)


def discard_shared(incoming):
    """Skip the entire shared scope, but never strand an ordinary dependency."""
    predicate = "kind='nodes' AND json_extract(body,'$.owner_space') LIKE 'space:%'"
    incoming.db.execute('DELETE FROM refs WHERE source IN (SELECT key FROM records WHERE ' + predicate + ')')
    incoming.db.execute('DELETE FROM records WHERE ' + predicate)
    try:
        incoming.finish()
    except PackageError as exc:
        raise PackageError('ordinary memories depend on skipped shared-space content') from exc
