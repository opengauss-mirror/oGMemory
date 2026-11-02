"""Chroma adapter contracts for batching and scoped deletion."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from core.models import IndexRecord
from providers.vector_index import chroma_index
from providers.vector_index.chroma_index import ChromaVectorIndex


class FakeClient:
    def __init__(self, max_batch_size=5_000):
        self.max_batch_size = max_batch_size

    def get_max_batch_size(self):
        return self.max_batch_size


class FakeCollection:
    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}
        self.upsert_sizes: list[int] = []
        self.delete_sizes: list[int] = []

    def count(self):
        return len(self.rows)

    def upsert(self, *, ids, embeddings, documents, metadatas):
        self.upsert_sizes.append(len(ids))
        for record_id, embedding, document, metadata in zip(ids, embeddings, documents, metadatas):
            self.rows[record_id] = {
                "embedding": list(embedding),
                "document": document,
                "metadata": dict(metadata),
            }

    def get(self, *, where, limit, include):
        del include
        selected = [
            record_id
            for record_id, row in self.rows.items()
            if self._matches(row["metadata"], where)
        ][:limit]
        return {"ids": selected}

    def delete(self, *, ids):
        self.delete_sizes.append(len(ids))
        for record_id in ids:
            self.rows.pop(record_id, None)

    @classmethod
    def _matches(cls, metadata, where):
        if "$and" in where:
            return all(cls._matches(metadata, condition) for condition in where["$and"])
        return all(metadata.get(key) == expected for key, expected in where.items())


def _index(max_batch_size=5_000):
    index = ChromaVectorIndex.__new__(ChromaVectorIndex)
    index._dimension = 3
    index._client = FakeClient(max_batch_size)
    index._collection = FakeCollection()
    index._lock_path = str(Path(tempfile.mkdtemp()) / ".ogmemory-chroma.lock")
    return index


def _record(record_id, *, account="tenant", owner="user:alice", embedding=None):
    return IndexRecord(
        id=record_id,
        uri=f"ctx://{account}/memory/{record_id}",
        level=0,
        text=record_id,
        filters={"account_id": account, "owner_space": owner},
        metadata={"_embedding": embedding or [1.0, 0.0, 0.0]},
    )


def test_upsert_and_delete_respect_provider_batch_limit():
    index = _index(max_batch_size=2)
    records = [
        _record(f"record-{number}", embedding=[float(number), 1.0, 0.0]) for number in range(5)
    ]

    index.upsert(records)
    assert {
        record_id: (
            row["embedding"],
            row["document"],
            row["metadata"]["uri"],
        )
        for record_id, row in index._collection.rows.items()
    } == {record.id: (record.metadata["_embedding"], record.text, record.uri) for record in records}

    index.delete([record.id for record in records])

    assert index._collection.upsert_sizes == [2, 2, 1]
    assert index._collection.delete_sizes == [2, 2, 1]
    assert index.count() == 0


def test_scope_delete_is_tenant_and_owner_isolated():
    index = _index()
    index.upsert(
        [
            _record("target"),
            _record("other-owner", owner="agent:bot"),
            _record("other-account", account="other"),
        ]
    )

    assert index.delete_by_owner_space("tenant", "user:alice") == 1
    assert set(index._collection.rows) == {"other-owner", "other-account"}
    assert index.delete_account_data("tenant") == 1
    assert set(index._collection.rows) == {"other-account"}


def test_scope_delete_pages_at_provider_limit():
    index = _index(max_batch_size=2)
    index.upsert([_record(f"record-{number}") for number in range(5)])

    assert index.delete_account_data("tenant") == 5
    assert index._collection.delete_sizes == [2, 2, 1]


def test_real_chroma_round_trip_when_provider_is_installed(tmp_path):
    pytest.importorskip("chromadb")
    index = ChromaVectorIndex(
        collection_name="adapter_contract",
        persist_directory=str(tmp_path),
        dimension=3,
    )
    assert (tmp_path / ".ogmemory-chroma.lock").exists()
    index.upsert([_record("one"), _record("two", owner="agent:bot")])
    assert index.count() == 2

    assert index.delete_by_owner_space("tenant", "user:alice") == 1
    assert index.delete_account_data("tenant") == 1
    assert index.count() == 0


def test_windows_lock_retries_until_available(monkeypatch):
    attempts = {"count": 0}

    class FakeMsvcrt:
        LK_NBLCK = 1
        LK_UNLCK = 2

        def locking(self, _fd, mode, _size):
            if mode == self.LK_NBLCK and attempts["count"] < 2:
                attempts["count"] += 1
                raise PermissionError(13, "locked")

    fake_file = Mock()
    fake_file.fileno.return_value = 3
    monkeypatch.setitem(__import__("sys").modules, "msvcrt", FakeMsvcrt())
    monkeypatch.setattr(chroma_index.time, "sleep", Mock())

    chroma_index._lock_windows_file(fake_file)

    assert attempts["count"] == 2
    assert chroma_index.time.sleep.call_count == 2


def test_windows_lock_propagates_access_denied(monkeypatch):
    access_denied = OSError(13, "Permission denied")
    access_denied.winerror = 5
    fake_msvcrt = Mock(LK_NBLCK=1)
    fake_msvcrt.locking.side_effect = access_denied
    fake_file = Mock()
    fake_file.fileno.return_value = 3
    monkeypatch.setitem(__import__("sys").modules, "msvcrt", fake_msvcrt)
    monkeypatch.setattr(chroma_index.time, "sleep", Mock())

    with pytest.raises(OSError) as exc_info:
        chroma_index._lock_windows_file(fake_file)

    assert exc_info.value is access_denied
    chroma_index.time.sleep.assert_not_called()
