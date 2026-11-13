"""Prepared model results survive a fresh SQL session owner."""

from service.extraction_checkpoint import ExtractionBatchJournal
from session.session_state import SessionState
from tests.integration.test_session_mutation_sql import database, ctx
from tests.unit.service.test_model_stage_checkpoint import journal_for, restore


def test_sql_stage_groups_restore_and_finish_under_new_owner(database, ctx):
    _, make = database
    fs, _, _ = make()
    state_uri = SessionState.state_uri(ctx.account_id, ctx.session_id)
    with fs.session_mutation_lock(state_uri, ctx):
        _, journal = journal_for(fs, ctx)
        for group in ("painting", "travel"):
            journal.checkpoint(0).save_stage("node_processing", {"group": group},
                                              {"content": group + " complete facts"})
        artifacts = list(journal.batch["stage_result_uris"])
    resumed_fs, _, _ = make()
    with resumed_fs.session_mutation_lock(state_uri, ctx):
        resumed = restore(resumed_fs, ctx)
        for group in ("painting", "travel"):
            assert resumed.checkpoint(0).load_stage("node_processing", {"group": group}) == {
                "content": group + " complete facts"}
        resumed.finish()
        assert all(resumed_fs.exists(uri, ctx) is False for uri in artifacts)
        manager = resumed.manager
        assert manager.get_or_create(ctx.session_id, ctx).request_completed("source-request")
