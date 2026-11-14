"""Service restart releases physical openGauss sessions after normal reads and writes."""

import os
import time
from uuid import uuid4

import psycopg2
import pytest

from scripts.shared_space_agfs_qa import create_service, load_environment


def test_service_restart_releases_sql_and_vector_sessions(record_property):
    if not os.environ.get("OGMEM_SPACE_TEST_CONFIG"):
        pytest.skip("OGMEM_SPACE_TEST_CONFIG selects the openGauss + AGFS deployment")
    environment = load_environment()
    for cycle in range(2):
        service = create_service(environment, "resource-test-" + uuid4().hex[:12])
        pool = service._get_shared_sql_pool()
        vector = None
        sessions, pids = [], []
        try:
            conn = pool.get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute("CREATE TEMP TABLE service_resource_probe (value INTEGER)")
                    cur.execute("INSERT INTO service_resource_probe(value) VALUES(42)")
                    cur.execute("SELECT value FROM service_resource_probe")
                    assert cur.fetchone()[0] == 42
                    cur.execute("SELECT pg_backend_pid()")
                    pids.append(cur.fetchone()[0])
                sessions.append(conn)
                conn.commit()
            finally:
                pool.return_connection(conn)
            vector = service._get_shared_vector_index()
            assert vector.count({"account_id": service._default_account_id}) == 0
            vector_conn = vector._get_connection()
            try:
                with vector_conn.cursor() as cur:
                    cur.execute("SELECT pg_backend_pid()")
                    pids.append(cur.fetchone()[0])
                sessions.append(vector_conn)
                vector_conn.rollback()
            finally:
                vector._return_connection(vector_conn)
            observer = psycopg2.connect(environment["dsn"], connect_timeout=5)
            try:
                observer.autocommit = True
                with observer.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM pg_stat_activity WHERE pid=ANY(%s)", (pids,))
                    before = cur.fetchone()[0]
                    assert before == len(set(pids))
                    service.shutdown(10)
                    assert all(session.closed for session in sessions)
                    deadline = time.monotonic() + 3
                    while True:
                        cur.execute("SELECT COUNT(*) FROM pg_stat_activity WHERE pid=ANY(%s)", (pids,))
                        after = cur.fetchone()[0]
                        if after == 0 or time.monotonic() >= deadline:
                            break
                        time.sleep(0.02)
                    assert after == 0
                    record_property(f"cycle_{cycle + 1}", {"pids": pids, "before": before, "after": after})
            finally:
                observer.close()
        finally:
            service.shutdown(10)
            if vector is not None:
                vector.close()
            pool.close_all()
