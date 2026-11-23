"""Selecting an extraction engine preserves the public commit result contract."""

from unittest.mock import Mock

import pytest

from extraction.deterministic.nlp import DeterministicConfig
from service.deterministic_ingest import extraction_configuration
from service.extraction_checkpoint import RecoveryNeedsConfirmationError
from tests.unit.evidence.test_sources import CTX, message


@pytest.mark.parametrize("engine", ["llm", "deterministic"])
@pytest.mark.parametrize("wait", [False, True])
@pytest.mark.parametrize("archive_id", [None, "caller-archive"])
def test_commit_result_and_task_contract(setup, monkeypatch, engine, wait, archive_id):
    _, api, store, llm = setup
    monkeypatch.setattr("extraction.deterministic.nlp.load_model", Mock(side_effect=OSError("rules-only test")))
    api._extraction_engine = engine
    api._deterministic_config = DeterministicConfig(on_model_missing="rules_only")
    api._pipeline.extract = Mock(return_value=[])
    result = api.commit_session([message()], CTX, wait=wait, archive_id=archive_id)
    assert result["archive_id"]
    if archive_id is not None:
        assert result["archive_id"] == archive_id
    if wait:
        assert "task_id" not in result and "status" not in result
    else:
        assert result["status"] == "completed"
        task = api.get_task_status(result["task_id"])
        assert task["status"] == "completed" and task["result"] == result
    if engine == "deterministic":
        source = store.read_source(store.manifest(CTX)[0]["sources"][0], CTX)
        assert result["archive_id"] in source["archive_ids"]
        api._pipeline.extract.assert_not_called()
    else:
        api._pipeline.extract.assert_called_once()
        assert store.manifest(CTX)[0] is None
    llm.complete_json.assert_not_called()


@pytest.mark.parametrize("archive_id", [None, "checkpoint-archive", "different-archive"])
def test_deterministic_commit_preserves_checkpoint_archive(setup, monkeypatch, archive_id):
    _, api, store, _ = setup
    monkeypatch.setattr("extraction.deterministic.nlp.load_model", Mock(side_effect=OSError("rules-only test")))
    api._extraction_engine = "deterministic"
    api._deterministic_config = DeterministicConfig(on_model_missing="rules_only")
    checkpoint = Mock(slot="user")
    checkpoint.journal.batch = {"archive_id": "checkpoint-archive",
                                "extraction_config": extraction_configuration(api)}
    checkpoint.load.return_value = ([], 0, 0)
    if archive_id == "different-archive":
        with pytest.raises(RecoveryNeedsConfirmationError, match="archive_id differs"):
            api.commit_session([message()], CTX, archive_id=archive_id, candidate_checkpoint=checkpoint)
        assert store.manifest(CTX)[0] is None
    else:
        result = api.commit_session([message()], CTX, archive_id=archive_id, candidate_checkpoint=checkpoint)
        assert result["archive_id"] == "checkpoint-archive" and result["evidence"]["replayed"]
        assert store.read_source(store.manifest(CTX)[0]["sources"][0], CTX)["archive_ids"] == ["checkpoint-archive"]
