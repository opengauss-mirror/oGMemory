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

"""Explicit one-time vector column backfill; see deploy/README.standalone-upgrade.md."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    from providers.vector_index.opengauss_index import OpenGaussVectorIndex

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dsn-env', default='VECTOR_DB_DSN', help='Environment variable holding DSN')
    parser.add_argument('--table', default='vector_index')
    parser.add_argument('--dimension', type=int, default=1536)
    parser.add_argument('--dialect', choices=('opengauss', 'postgresql', 'auto'), default='opengauss')
    args = parser.parse_args()
    dsn = os.environ.get(args.dsn_env)
    if not dsn:
        parser.error(f'Set {args.dsn_env} to the vector database DSN')
    index = OpenGaussVectorIndex(dsn, table_name=args.table, dimension=args.dimension, dialect=args.dialect)
    try:
        changed = index.migrate_legacy_rows()
    finally:
        index.close()
    print('Legacy vector backfill completed.' if changed else 'Already migrated; no rows scanned.')


if __name__ == '__main__':
    main()
