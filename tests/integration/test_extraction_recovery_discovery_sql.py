"""Session recovery discovery against disposable PostgreSQL storage."""

import os
from dataclasses import replace
from uuid import uuid4

import psycopg2
import pytest
from psycopg2 import sql

from core.models import RequestContext
from service.extraction_recovery_discovery import ExtractionRecoveryDiscovery
from tests.integration.test_occurrence_replay_sql import close_stores, stores
from tests.unit.service.test_extraction_recovery_discovery import state


@pytest.mark.parametrize("upgrade", [False, True])
def test_normal_role_discovers_account_directory_after_restart(upgrade):
    dsn = os.environ.get("OGMEM_IDENTITY_TEST_DSN")
    if not dsn:
        pytest.skip("set OGMEM_IDENTITY_TEST_DSN to a disposable PostgreSQL database")
    from fs.sql_adapter.schema import _schema_ensured, ensure_schema
    from fs.sql_adapter.sql_context_fs import SQLContextFS
    role = "recovery_role_" + uuid4().hex
    admin = psycopg2.connect(dsn)
    admin.autocommit = True
    fs = None
    try:
        with admin.cursor() as cursor:
            cursor.execute(sql.SQL("CREATE ROLE {} NOSUPERUSER NOBYPASSRLS").format(sql.Identifier(role)))
            cursor.execute(sql.SQL("CREATE SCHEMA {} AUTHORIZATION {}").format(sql.Identifier(role), sql.Identifier(role)))
        tenant_dsn = psycopg2.extensions.make_dsn(dsn, options=f"-c role={role} -c search_path={role}")
        fs = SQLContextFS(tenant_dsn)
        ctx = RequestContext("discovery-" + uuid4().hex, "reader", "agent", "source", "trace")
        fs.write_node(state(ctx), ctx)
        connection = fs._get_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT current_user, rolsuper, rolbypassrls FROM pg_roles WHERE rolname=current_user")
                assert cursor.fetchone() == (role, False, False)
                cursor.execute("SELECT count(*) FROM context_nodes")
                assert cursor.fetchone()[0] == 0
                if upgrade:
                    cursor.execute("DROP TABLE extraction_work_accounts")
            connection.commit()
            if upgrade:
                _schema_ensured.clear()
                ensure_schema(connection)
            with connection.cursor() as cursor:
                cursor.execute("SELECT relforcerowsecurity FROM pg_class WHERE oid='context_nodes'::regclass")
                assert cursor.fetchone()[0] is True
            connection.rollback()
        finally:
            fs._return_connection(connection)
        discover = ExtractionRecoveryDiscovery(lambda: fs, fs.recovery_account_ids)
        [(recovered, task)] = discover()
        assert recovered.account_id == ctx.account_id
        assert task["id"] == "archive-source"
    finally:
        if fs is not None:
            while fs._pool:
                fs._pool.pop().close()
        with admin.cursor() as cursor:
            cursor.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(role)))
            cursor.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role)))
        admin.close()


@pytest.mark.parametrize("legacy", [False, True])
def test_sql_metadata_pages_restore_every_source_owner(legacy):
    ctx = RequestContext("discovery-" + uuid4().hex, "reader", "agent", "source", "trace")
    fs, outbox = stores()
    try:
        for index in range(5):
            current = replace(ctx, session_id=f"source-{index}")
            fs.write_node(state(current, legacy=legacy), current)
        assert ctx.account_id in fs.recovery_account_ids()
        discover = ExtractionRecoveryDiscovery(lambda: fs, lambda: [ctx.account_id], page_size=2)

        pages = [discover(), discover(), discover()]

        assert [len(page) for page in pages] == [2, 2, 1]
        assert [item[0].session_id for page in pages for item in page] == [f"source-{index}" for index in range(5)]
        assert {item[0].account_id for page in pages for item in page} == {ctx.account_id}
    finally:
        close_stores(fs, outbox)
