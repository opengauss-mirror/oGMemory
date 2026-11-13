from dataclasses import replace
from unittest.mock import MagicMock, Mock

from index.edge_projection import (
    AcceptedEdgeProjectionInput,
    EdgeProjectionWorker,
    InMemoryEdgeProjectionStore,
    bounded_projection_batch,
    build_searchable_text,
)
from providers.relation_store.sql_projection import SQLEdgeProjectionStore
from providers.token_tracker import distributed_usage_scope, embedding_scope_weights

PROFILE = {"profile_id": "bge", "model": "test", "dimension": 2}


def edge(name, text="Shared target"):
    return AcceptedEdgeProjectionInput(
        name, f"ctx://acct/users/alice/memories/{name}",
        "ctx://acct/users/alice/memories/target", "supports", "evidence", text, 1, 2,
        claim_token=f"claim-{name}",
    )


def test_shared_target_is_embedded_once_and_each_edge_keeps_its_revision():
    inputs = [edge("a"), replace(edge("b"), source_revision=3)]
    embedder = Mock()
    embedder.embed_texts.return_value = [[1.0, 0.0]]
    store = InMemoryEdgeProjectionStore()
    projected = EdgeProjectionWorker(embedder, store, PROFILE).project_accepted_sync(inputs)
    embedder.embed_texts.assert_called_once_with([build_searchable_text(inputs[0])])
    assert [item.source_revision for item in projected] == [1, 3]
    assert [item.claim_token for item in projected] == ["claim-a", "claim-b"]
    assert store.get("a").vector == store.get("b").vector == (1.0, 0.0)


def test_distinct_owner_inputs_retain_usage_attribution():
    from providers.token_tracker import _embedding_inputs

    inputs = [edge("a"), edge("b"), replace(edge("c"),
        source_uri="ctx://acct/users/bob/memories/c")]
    usage = [(("acct", "user:alice"), 10), (("acct", "user:alice"), 10),
             (("acct", "user:bob"), 10)]
    observed = []
    embedder = Mock()

    def embed(texts):
        observed.extend(_embedding_inputs.get())
        return [[1.0, 0.0] for _ in texts]

    embedder.embed_texts.side_effect = embed
    with distributed_usage_scope(embedding_scope_weights(usage), inputs=usage):
        projected = EdgeProjectionWorker(embedder, InMemoryEdgeProjectionStore(),
                                         PROFILE).project_accepted_sync(inputs)
    assert len(projected) == 3
    assert observed == [usage[0], usage[2]]


def test_byte_budget_groups_complete_unique_texts_in_order():
    inputs = [edge("a", "甲" * 40), edge("b", "甲" * 40), edge("c", "乙" * 40)]
    budget = len(build_searchable_text(inputs[0]).encode("utf-8"))
    assert bounded_projection_batch(inputs, budget) == inputs[:2]
    assert bounded_projection_batch(inputs[2:], budget) == inputs[2:]


def test_oversized_first_text_is_claimed_intact():
    inputs = [edge("a", "Long content " * 1000), edge("b", "Next target")]
    selected = bounded_projection_batch(inputs)
    assert selected == inputs[:1]
    assert selected[0].target_text == "Long content " * 1000


def test_repeated_oversized_input_shares_one_claim():
    inputs = [edge("a", "Long content " * 1000), edge("b", "Long content " * 1000),
              edge("c", "Next target")]
    assert bounded_projection_batch(inputs) == inputs[:2]


def test_sql_claim_updates_the_bounded_prefix():
    inputs = [edge(str(i), str(i) * 4000) for i in range(4)]
    rows = [(item.edge_id, item.source_uri, item.target_uri, item.relation, item.reason,
             item.target_text, item.source_revision, item.target_revision) for item in inputs]
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = rows
    store = object.__new__(SQLEdgeProjectionStore)
    store._get_connection = Mock(return_value=conn)
    store._return_connection = Mock()
    store.bind_tenant = Mock()
    store._dialect = Mock(supports_batch_claim=True)
    store._claimed_accounts, store._claimed_tokens = {}, {}
    claimed = store.claim_pending("acct")
    assert [item.edge_id for item in claimed] == ["0", "1"]
    update = next(call for call in cursor.execute.call_args_list
                  if "SET status = 'PROCESSING'" in call.args[0])
    assert update.args[1][2] == ["0", "1"]
    assert all(item.claim_token for item in claimed)
