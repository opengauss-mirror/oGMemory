from core.errors import NodeNotFoundError


def lock_relation_endpoints(cursor, account_id: str, uris: set[str]) -> int:
    """Lock existing endpoints in the same URI order as Dreaming merges."""
    cursor.execute(
        "SELECT uri, status, metadata->>'dreaming_canonical_uri' "
        "FROM context_nodes WHERE account_id=%s AND uri=ANY(%s) ORDER BY uri FOR UPDATE",
        (account_id, sorted(uris)),
    )
    endpoints = cursor.fetchall()
    for uri, status, canonical_uri in endpoints:
        if status == "ARCHIVED" and canonical_uri:
            raise NodeNotFoundError(uri)
    return len(endpoints)
