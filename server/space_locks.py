"""Shared transaction lock for team tasks and identity changes."""

import hashlib


def lock_space_account(cur, account_id):
    digest = hashlib.sha256(f"shared-space:{account_id}".encode()).digest()
    key = int.from_bytes(digest[:8], "big", signed=True)
    cur.execute("SELECT pg_advisory_xact_lock(%s)", (key,))
