"""Run against a dedicated SQL instance; never modify application tables."""
import os
import uuid

import pytest

from service.migration.package import PackageError
from service.migration.sql_backup import SQLBackup


def test_sql_recovery_preserves_types_sequences_and_neighbor(tmp_path):
    dsn = os.environ.get("MIGRATION_TEST_ADMIN_DSN")
    if not dsn:
        pytest.skip("MIGRATION_TEST_ADMIN_DSN is required")
    import psycopg2
    from psycopg2 import sql

    schema = "recovery_" + uuid.uuid4().hex
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
            cur.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(schema)))
            cur.execute("CREATE TABLE parent (id BIGSERIAL PRIMARY KEY, content TEXT, metadata JSONB, bytes BYTEA, created TIMESTAMPTZ)")
            cur.execute("CREATE TABLE child (id INT PRIMARY KEY, parent_id BIGINT REFERENCES parent(id), value DOUBLE PRECISION)")
            cur.execute("CREATE TABLE neighbor (content TEXT)")
            cur.execute("INSERT INTO parent(content,metadata,bytes,created) VALUES(%s,%s,%s,%s)",
                        ("原始\n内容", '{"items":[null,1,"中文"]}', psycopg2.Binary(b"\x00\xff"), "2026-01-01T01:02:03+08:00"))
            cur.execute("INSERT INTO child VALUES(1,1,0.125)")
            cur.execute("INSERT INTO neighbor VALUES('untouched')")
        conn.commit()

        def identity():
            with conn.cursor() as cur:
                cur.execute("SELECT current_database(),oid FROM pg_namespace WHERE nspname=%s", (schema,))
                database, oid = cur.fetchone()
                return {"database": database, "schema_oid": oid}

        backup = SQLBackup(conn, schema, ["parent", "child"], identity)
        destination = tmp_path / "backup"
        manifest = backup.capture(destination)
        with conn.cursor() as cur:
            cur.execute("DELETE FROM child")
            cur.execute("UPDATE parent SET content='changed'")
            cur.execute("INSERT INTO parent(content) VALUES('imported')")
        conn.commit()
        backup.restore(destination, manifest)
        with conn.cursor() as cur:
            cur.execute("SELECT content,metadata,bytes FROM parent")
            row = cur.fetchone()
            assert row[0] == "原始\n内容" and row[1] == {"items": [None, 1, "中文"]}
            assert bytes(row[2]) == b"\x00\xff"
            cur.execute("SELECT content FROM neighbor")
            assert cur.fetchone()[0] == "untouched"
            cur.execute("INSERT INTO parent(content) VALUES('after restore') RETURNING id")
            assert cur.fetchone()[0] == 2
        conn.commit()
        backup.restore(destination, manifest)
        backup.identity = lambda: {"database": "wrong"}
        with pytest.raises(PackageError, match="target"):
            backup.restore(destination, manifest)
    finally:
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
        conn.commit()
        conn.close()


@pytest.mark.parametrize('client_encoding', ['UTF8', 'LATIN1'])
@pytest.mark.parametrize('csv_c_long_limit', [2**31 - 1, 2**63 - 1])
def test_sql_recovery_sets_each_account_and_preserves_csv_nulls(tmp_path, monkeypatch, client_encoding, csv_c_long_limit):
    dsn = os.environ.get('MIGRATION_TEST_ADMIN_DSN')
    if not dsn:
        pytest.skip('MIGRATION_TEST_ADMIN_DSN is required')
    import psycopg2
    from psycopg2 import sql

    schema = 'recovery_' + uuid.uuid4().hex
    conn = psycopg2.connect(dsn)
    conn.set_client_encoding(client_encoding)
    import csv
    original_limit = csv.field_size_limit
    def portable_limit(value):
        if value > csv_c_long_limit:
            raise OverflowError('Python int too large to convert to C long')
        return original_limit(value)
    monkeypatch.setattr(csv, 'field_size_limit', portable_limit)
    try:
        with conn.cursor() as cur:
            cur.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(schema)))
            cur.execute(sql.SQL('SET search_path TO {}').format(sql.Identifier(schema)))
            cur.execute("CREATE TABLE tenant_records (id INT PRIMARY KEY, account_id TEXT NOT NULL, "
                        "content TEXT, optional TEXT, CHECK(account_id=current_setting('app.account_id')))")
            for number, account in enumerate(('alpha', 'beta', 'alpha')):
                cur.execute("SELECT set_config('app.account_id',%s,true)", (account,))
                cur.execute('INSERT INTO tenant_records VALUES(%s,%s,%s,%s)',
                            (number, account, ('café,"quoted"\nsecond line' if client_encoding == 'LATIN1' else '中文,"quoted"\nsecond line'), '' if number else None))
        conn.commit()
        backup = SQLBackup(conn, schema, ['tenant_records'], lambda: {'schema': schema})
        destination = tmp_path / 'backup'
        manifest = backup.capture(destination)
        with conn.cursor() as cur:
            cur.execute('DELETE FROM tenant_records')
        conn.commit()
        backup.restore(destination, manifest)
        with conn.cursor() as cur:
            cur.execute('SELECT account_id,content,optional FROM tenant_records ORDER BY id')
            assert cur.fetchall() == [(account, ('café,"quoted"\nsecond line' if client_encoding == 'LATIN1' else '中文,"quoted"\nsecond line'), '' if number else None)
                                      for number, account in enumerate(('alpha', 'beta', 'alpha'))]
    finally:
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(schema)))
        conn.commit()
        conn.close()


def test_shared_source_epoch_and_cleanup_queue_survive_recovery(tmp_path, monkeypatch):
    """Use the real shared-space triggers; recovery must not replay their effects."""
    from pathlib import Path

    dsn = os.environ.get('MIGRATION_TEST_ADMIN_DSN')
    if not dsn:
        pytest.skip('MIGRATION_TEST_ADMIN_DSN is required')
    import psycopg2
    from psycopg2 import sql

    schema = 'recovery_' + uuid.uuid4().hex
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(schema)))
            cur.execute(sql.SQL('SET search_path TO {}').format(sql.Identifier(schema)))
            cur.execute("CREATE TABLE context_nodes (uri TEXT PRIMARY KEY, account_id TEXT, "
                        "status TEXT, content TEXT, abstract TEXT, overview TEXT, version INT, metadata JSONB)")
            cur.execute('CREATE TABLE relation_edges (id INT)')
            cur.execute('CREATE TABLE edge_projection_jobs (id INT)')
            cur.execute("CREATE TABLE outbox_events (event_id TEXT,event_type TEXT,uri TEXT,account_id TEXT,status TEXT,payload JSONB DEFAULT '{}')")
            for filename, function, trigger in (
                ('server/space_schema.sql', 'ogmemory_space_source_revision', 'ogmemory_space_source_revision'),
                ('server/space_schema.sql', 'ogmemory_queue_agfs_cleanup', 'ogmemory_queue_agfs_cleanup'),
                ('fs/sql_adapter/space_events.sql', 'ogmemory_space_source_changed', 'ogmemory_space_source_change'),
            ):
                source = (Path(__file__).parents[2] / filename).read_text()
                begin = source.index('CREATE OR REPLACE FUNCTION ' + function + '()')
                end = source.index('$$ LANGUAGE plpgsql;', begin) + len('$$ LANGUAGE plpgsql;')
                cur.execute(source[begin:end])
                begin = source.index('CREATE TRIGGER ' + trigger)
                end = source.index('();', begin) + 3
                cur.execute(source[begin:end])
            uri = 'ctx://acme/users/alice/memories/preferences/original'
            cur.execute("INSERT INTO context_nodes VALUES(%s,'acme','ACTIVE','original','','',1,%s)",
                        (uri, '{"agfs_content":{"uri":"' + uri + '"}}'))
        conn.commit()
        backup = SQLBackup(conn, schema, ['context_nodes', 'relation_edges', 'edge_projection_jobs', 'outbox_events'],
                           lambda: {'schema': schema})
        destination = tmp_path / 'backup'
        manifest = backup.capture(destination)
        with conn.cursor() as cur:
            cur.execute('SELECT metadata,version FROM context_nodes')
            original = cur.fetchone()
            cur.execute("UPDATE context_nodes SET content='changed'")
        conn.commit()
        with monkeypatch.context() as patch:
            def failed_copy(*args):
                raise RuntimeError('injected restore failure')
            patch.setattr(backup, '_copy_in', failed_copy)
            with pytest.raises(RuntimeError, match='injected restore failure'):
                backup.restore(destination, manifest)
        with conn.cursor() as cur:
            cur.execute('SELECT content FROM context_nodes')
            assert cur.fetchone()[0] == 'changed'
            cur.execute("SELECT tgenabled FROM pg_trigger WHERE tgrelid='context_nodes'::regclass AND NOT tgisinternal")
            assert all(row[0] == 'O' for row in cur)
        conn.rollback()
        backup.restore(destination, manifest)
        with conn.cursor() as cur:
            cur.execute('SELECT metadata,version FROM context_nodes')
            assert cur.fetchone() == original
            cur.execute('SELECT count(*) FROM outbox_events')
            assert cur.fetchone()[0] == 1  # Only the original SOURCE_CHANGED event.
            cur.execute("SELECT tgenabled FROM pg_trigger WHERE tgname IN "
                        "('ogmemory_space_source_revision','ogmemory_queue_agfs_cleanup') AND tgrelid='context_nodes'::regclass")
            assert [row[0] for row in cur] == ['O', 'O']
            cur.execute("UPDATE context_nodes SET content='after recovery' RETURNING version")
            assert cur.fetchone()[0] == original[1] + 1
        conn.rollback()
    finally:
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(schema)))
        conn.commit()
        conn.close()
