#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Opt-in PostgreSQL+pgvector benchmark in a disposable, automatically removed schema.

Set VECTOR_DB_DSN; optionally pass --baseline-file containing a previous version
of providers/vector_index/opengauss_index.py. Does not touch application tables.
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from uuid import uuid4

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    import psycopg2
    from psycopg2.extensions import make_dsn

    from providers.vector_index.opengauss_index import OpenGaussVectorIndex

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=int, default=1_000_000)
    parser.add_argument('--baseline-file')
    args = parser.parse_args()
    dsn = os.environ['VECTOR_DB_DSN']
    schema = 'vector_startup_bench_' + uuid4().hex
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
            cur.execute(f'CREATE SCHEMA {schema}')
            cur.execute(f'SET search_path TO {schema}, public')
        isolated = make_dsn(dsn, options=f'-c search_path={schema},public')

        def construct(cls):
            start = time.perf_counter()
            vi = cls(isolated, dimension=3)
            elapsed = time.perf_counter() - start
            for c in vi._pool:
                c.close()
            return elapsed

        construct(OpenGaussVectorIndex)
        with conn.cursor() as cur:
            # Seed before timing; this benchmark measures existing-table startup,
            # not initial HNSW/GIN index construction.
            cur.execute("""INSERT INTO vector_index
                (id, uri, level, text, embedding, filters, metadata)
                SELECT lpad(i::text, 16, '0'), 'ctx://legacy/' || i, 0,
                       repeat('legacy text ', 10), '[1,0,0]'::vector,
                       '{"account_id":"bench","owner_space":"user:u"}'::jsonb, '{}'::jsonb
                FROM generate_series(1, %s) i""", (args.rows,))
            cur.execute('CREATE TABLE context_nodes (uri text PRIMARY KEY, node_id uuid, account_id text)')
            cur.execute('ANALYZE vector_index')
        results = {'rows': args.rows, 'dimension': 3,
                   'current_seconds': [construct(OpenGaussVectorIndex) for _ in range(3)]}
        with conn.cursor() as cur:
            cur.execute('SELECT count(*) FROM vector_index WHERE account_id IS NULL')
            assert cur.fetchone()[0] == args.rows, 'Startup unexpectedly backfilled data'
        if args.baseline_file:
            spec = importlib.util.spec_from_file_location('vector_baseline', args.baseline_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            results['baseline_seconds'] = [construct(module.OpenGaussVectorIndex) for _ in range(3)]
        print(json.dumps(results, indent=2), flush=True)
    finally:
        with conn.cursor() as cur:
            cur.execute(f'DROP SCHEMA IF EXISTS {schema} CASCADE')
        conn.close()


if __name__ == '__main__':
    main()
