from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from core.lexical import HybridLexicalIndex, LexicalRecord
from providers.lexical_index.sqlite_index import SQLiteLexicalIndex


def _record(record_id, text, *, account="academy", owner="user:alice", directory="events", level=2):
    return LexicalRecord(
        id=record_id,
        uri=f"ctx://{account}/users/{owner.split(':')[1]}/memories/{directory}/{record_id}",
        level=level,
        text=text,
        account_id=account,
        owner_space=owner,
        metadata={"context_type": "MEMORY", "category": "event"},
    )


def _ids(index, query, **filters):
    return [hit.id for hit in index.search(query, 5, {"account_id": "academy", **filters})]


def test_sqlite_connections_open_and_write_shared_file_concurrently(tmp_path):
    path = str(tmp_path / "index.db")
    record_ids = {f"memory-{number}" for number in range(8)}
    ready = Barrier(len(record_ids), timeout=5)

    def publish(record_id):
        ready.wait()
        index = SQLiteLexicalIndex(path)
        try:
            index.upsert_many([_record(record_id, "stalagmite")])
        finally:
            index.close()

    with ThreadPoolExecutor(max_workers=len(record_ids)) as executor:
        list(executor.map(publish, record_ids))
    index = SQLiteLexicalIndex(path)
    try:
        hits = index.search("stalagmite", len(record_ids), {"account_id": "academy"})
        assert {hit.id for hit in hits} == record_ids
    finally:
        assert index.close()


def test_sqlite_fts_projections_share_live_records_and_checkpoints(tmp_path):
    path = str(tmp_path / "index.db")
    writer, reader = SQLiteLexicalIndex(path), SQLiteLexicalIndex(path)
    try:
        assert isinstance(writer, HybridLexicalIndex)
        records = [
            _record("alice", "Alpine hut in 杭州西湖"),
            _record("overview", "Alpine", level=0),
            _record("library", "Alpine", directory="knowledge"),
            _record("bob", "Alpine", owner="user:bob"),
            _record("different_account", "Alpine", account="other"),
        ]
        writer.upsert_many(records)
        assert _ids(
            reader,
            '"Alpine" OR',
            owner_space=["user:alice"],
            level=2,
            context_type="MEMORY",
            category="event",
            memory_directory=["events"],
        ) == ["alice"]
        (hit,) = reader.search("西湖", 1, {"account_id": "academy", "owner_space": "user:alice"})
        assert (hit.id, hit.metadata["uri"], hit.metadata["owner_space"]) == (
            "alice",
            records[0].uri,
            "user:alice",
        )
        assert hit.score > 0
        assert writer.replace_scopes(
            rebuild_id=writer.begin_rebuild(),
            account_id="academy",
            owner_spaces=["user:alice"],
            records=iter([_record("alice", "Coastal cabin")]),
        )
        assert reader.ready_scopes("academy", ["user:alice"]) == {"user:alice"}
        assert _ids(reader, "cabin") == ["alice"]
        assert _ids(reader, "Alpine") == ["bob"]
        assert writer.close()
        restarted = SQLiteLexicalIndex(path)
        try:
            assert restarted.ready_scopes("academy", ["user:alice"]) == {"user:alice"}
            restarted.upsert_many([_record("alice", "New mountain cabin")])
            assert _ids(reader, "mountain") == ["alice"]
            assert restarted.delete_scope(account_id="academy", owner_space="user:bob") == 1
            restarted.delete(["alice"])
            assert reader.health()["records"] == 1
        finally:
            assert restarted.close()
    finally:
        assert reader.close()
        assert writer.close()


def test_sqlite_rebuild_keeps_newer_publications_and_retries(tmp_path):
    path = str(tmp_path / "index.db")
    writer, reader = SQLiteLexicalIndex(path), SQLiteLexicalIndex(path)
    try:
        writer.upsert_many([_record("alice", "original marker")])
        rebuild_id = reader.begin_rebuild()
        writer.mark_scopes_dirty([("academy", "user:alice")])
        writer.upsert_many([_record("alice", "latest marker")])
        reader.replace_scopes(
            rebuild_id=rebuild_id,
            account_id="academy",
            owner_spaces=["user:alice"],
            records=iter([_record("alice", "stale marker")]),
        )
        assert _ids(reader, "latest") == ["alice"]
        assert reader.dirty_generations("academy", ["user:alice"])["user:alice"] > 0

        assert reader.replace_scopes(
            rebuild_id=reader.begin_rebuild(),
            account_id="academy",
            owner_spaces=["user:alice"],
            records=iter([_record("alice", "latest marker")]),
        )
        assert writer.ready_scopes("academy", ["user:alice"]) == {"user:alice"}
        assert _ids(writer, "latest") == ["alice"]
    finally:
        assert writer.close()
        assert reader.close()
