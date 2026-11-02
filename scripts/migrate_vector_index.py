#!/usr/bin/env python3
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
