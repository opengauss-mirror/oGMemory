import pytest

from core.models import IndexRecord
from providers.composite_index import CompositeProjectionWriter


class _Vector:
    def __init__(self, fail=False):
        self.fail = fail
        self.upserts = []
        self.deletes = []
        self.closed = False

    def upsert(self, records):
        if self.fail:
            raise RuntimeError("vector failed")
        self.upserts.append(records)

    def delete(self, ids):
        self.deletes.append(ids)

    def delete_account_data(self, account_id):
        return 1

    def delete_by_owner_space(self, account_id, owner_space):
        return 1

    def close(self):
        self.closed = True
        return True


class _Lexical:
    def __init__(self, fail=False, delete_fail=False):
        self.fail = fail
        self.delete_fail = delete_fail
        self.upserts = []
        self.deletes = []
        self.deleted_scopes = []

    def upsert_many(self, records):
        if self.fail:
            raise RuntimeError("lexical failed")
        self.upserts.append(records)

    def delete(self, ids):
        if self.delete_fail:
            raise RuntimeError("lexical delete failed")
        self.deletes.append(ids)

    def delete_scope(self, **scope):
        self.deleted_scopes.append(scope)
        return 1

    def close(self, timeout=5.0):
        return True


def _record():
    return IndexRecord(
        id="id-1",
        uri="ctx://acct/users/user/memories/a",
        level=2,
        text="database native bm25",
        filters={"account_id": "acct", "owner_space": "user:user"},
        metadata={"context_type": "MEMORY", "category": "fact"},
    )


def test_same_outbox_record_updates_vector_and_lexical_projection():
    vector, lexical = _Vector(), _Lexical()
    writer = CompositeProjectionWriter(vector, lexical)

    writer.upsert([_record()])

    assert vector.upserts[0][0].id == "id-1"
    assert lexical.upserts[0][0].id == "id-1"
    assert lexical.upserts[0][0].account_id == "acct"


def test_connection_descriptor_is_vector_only_not_a_lexical_document():
    from dataclasses import replace
    vector,lexical=_Vector(),_Lexical()
    writer=CompositeProjectionWriter(vector,lexical)
    descriptor=replace(_record(),id='descriptor',level=3,
        metadata={'record_kind':'update_activation'})
    writer.upsert([descriptor])
    assert vector.upserts==[[descriptor]] and lexical.upserts==[]
    writer.upsert([_record(),descriptor])
    assert [r.id for r in lexical.upserts[0]]==['id-1']


def test_required_lexical_failure_prevents_vector_commit():
    vector = _Vector()
    writer = CompositeProjectionWriter(vector, _Lexical(fail=True))

    with pytest.raises(RuntimeError, match="lexical failed"):
        writer.upsert([_record()])

    assert vector.upserts == []


def test_vector_failure_compensates_required_lexical_projection():
    lexical = _Lexical()
    writer = CompositeProjectionWriter(_Vector(fail=True), lexical)

    with pytest.raises(RuntimeError, match="vector failed"):
        writer.upsert([_record()])

    assert lexical.deletes == [["id-1"]]


def test_invalidation_failure_does_not_mask_vector_error_or_skip_compensation():
    lexical = _Lexical()
    writer = CompositeProjectionWriter(_Vector(fail=True), lexical)
    writer.set_failure_callbacks(
        records_failed=lambda _records: (_ for _ in ()).throw(RuntimeError("dirty failed"))
    )

    with pytest.raises(RuntimeError, match="vector failed"):
        writer.upsert([_record()])

    assert lexical.deletes == [["id-1"]]


def test_delete_updates_both_projections():
    vector, lexical = _Vector(), _Lexical()
    writer = CompositeProjectionWriter(vector, lexical)

    writer.delete(["id-1"])

    assert vector.deletes == [["id-1"]]
    assert lexical.deletes == [["id-1"]]


def test_close_closes_both_owned_projections():
    vector, lexical = _Vector(), _Lexical()
    writer = CompositeProjectionWriter(vector, lexical)

    assert writer.close(timeout=1.0) is True
    assert vector.closed is True


def test_account_and_owner_deletes_update_both_projections():
    vector, lexical = _Vector(), _Lexical()
    writer = CompositeProjectionWriter(vector, lexical)

    assert writer.delete_account_data("acct") == 1
    assert writer.delete_by_owner_space("acct", "user:u1") == 1

    assert lexical.deleted_scopes == [
        {"account_id": "acct"},
        {"account_id": "acct", "owner_space": "user:u1"},
    ]


def test_compensation_failure_preserves_vector_error_and_marks_scope_unknown():
    dirty = []
    writer = CompositeProjectionWriter(
        _Vector(fail=True),
        _Lexical(delete_fail=True),
    )
    writer.set_failure_callbacks(scope_unknown_failed=lambda: dirty.append(True))

    with pytest.raises(RuntimeError, match="vector failed"):
        writer.upsert([_record()])

    assert dirty == [True]

def test_takeover_keeps_successor_lexical_projection():
    from dataclasses import replace
    from core.projection_ownership import projection_owner, check_projection_owner

    class Lexical(_Lexical):
        def upsert_many(self, records):
            self.text = records[0].text

        def delete(self, ids):
            self.text = None

    lexical = Lexical()
    successor = CompositeProjectionWriter(_Vector(), lexical)
    old = replace(_record(), text="Monday")
    new = replace(old, text="Friday")
    owned = True

    def validate():
        if not owned:
            raise RuntimeError("claim transferred")

    class DelayedVector(_Vector):
        def upsert(self, records):
            nonlocal owned
            owned = False
            with projection_owner(lambda: None):
                successor.upsert([new])
            check_projection_owner()

    with projection_owner(validate), pytest.raises(RuntimeError, match="claim transferred"):
        CompositeProjectionWriter(DelayedVector(), lexical).upsert([old])
    print("REQUEST Monday -> Friday; RESPONSE lexical:", lexical.text)
    assert lexical.text == "Friday"
