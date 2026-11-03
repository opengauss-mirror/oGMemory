"""Verify configured Milvus permissions before replacing a deployment.

The short-lived probe uses an unregistered, UUID-scoped account and is removed
before success is reported. The selected collection is initialized through the
same provider factory as the application; existing records are never rewritten.
"""

from __future__ import annotations

import logging
import os
import sys
from uuid import uuid4

from core.models import IndexRecord
from providers.config import ProviderConfig


def check_milvus(config: ProviderConfig) -> None:
    probe = uuid4().hex
    account = f"ogmemory-preflight-{probe}"
    uri = f"ctx://{account}/users/probe/memories/entities/check"
    vector = [1.0] + [0.0] * (config.embedding_dimension - 1)
    record = IndexRecord(
        id=f"preflight-{probe}", uri=uri, level=0,
        text="Milvus deployment preflight",
        filters={"account_id": account, "owner_space": "user:probe"},
        metadata={"_embedding": vector},
    )
    index = config.create_vector_index()
    try:
        index.upsert([record])
        saved = index.get_record(record.id)
        if saved is None or saved.text != record.text:
            raise RuntimeError("Milvus preflight read did not match the written record")
        hits = index.search_by_vector(vector, {"account_id": account}, 1)
        if not hits or hits[0].uri != uri:
            raise RuntimeError("Milvus preflight search did not return the written record")
    finally:
        try:
            # Even an unsuccessful write may have committed before timing out.
            index.delete([record.id])
            if index.get_record(record.id) is not None:
                raise RuntimeError("Milvus preflight record was not removed")
        finally:
            close = getattr(index, "close", None)
            if callable(close):
                close()


def main() -> int:
    # SDK exceptions/logs can contain connection details. Report a failure type
    # without printing credentials from the configured URI or token.
    logging.disable(logging.CRITICAL)
    try:
        check_milvus(ProviderConfig(
            vector_db_type="milvus",
            embedding_dimension=int(os.environ["VECTOR_DB_DIMENSION"]),
            milvus_uri=os.environ["MILVUS_URI"],
            milvus_token=os.environ.get("MILVUS_TOKEN", ""),
            milvus_collection_name=os.environ["MILVUS_COLLECTION"],
            milvus_db_name=os.environ["MILVUS_DB_NAME"],
            milvus_timeout=float(os.environ["MILVUS_TIMEOUT"]),
        ))
    except Exception as exc:
        print(f"Milvus preflight failed ({type(exc).__name__})", file=sys.stderr)
        return 1
    print("Milvus schema/index and write/read/search/delete preflight passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
