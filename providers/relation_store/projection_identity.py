"""Stable identity and representation metadata for relation projections."""

from __future__ import annotations

import hashlib

from core.uri_parser import strip_l2_suffix

REPRESENTATION_VERSION = "relation-target-v2"


def canonical_edge_id(
    account_id: str,
    source_uri: str,
    target_uri: str,
    relation: str,
) -> str:
    raw = "\0".join(
        (
            account_id,
            strip_l2_suffix(source_uri),
            strip_l2_suffix(target_uri),
            relation,
        )
    )
    return hashlib.sha256(raw.encode()).hexdigest()
