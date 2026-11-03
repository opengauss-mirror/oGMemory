"""Missing AGFS directories are empty recovery roots, not scanner failures."""

import logging
from unittest.mock import Mock

import pytest
from pyagfs.exceptions import AGFSClientError

from core.models import RequestContext
from fs.agfs_adapter.agfs_context_fs import AGFSContextFS, _is_not_found_error
from service.extraction_recovery_discovery import ExtractionRecoveryDiscovery


@pytest.fixture
def context_fs():
    fs = AGFSContextFS.__new__(AGFSContextFS)
    fs._client = Mock()
    fs._mount_prefix = "/local"
    return fs


@pytest.mark.parametrize(
    "message",
    [
        "no such directory: /accounts/isolated/sessions",
        "NO SUCH DIRECTORY: /accounts/isolated/sessions",
        "no such file: /accounts/isolated/sessions/state.json",
        "404 Not Found",
    ],
)
def test_absent_recovery_roots_are_empty(context_fs, message):
    context_fs._client.ls.side_effect = AGFSClientError(message)
    ctx = RequestContext("isolated", "user", "agent", "session", "trace")
    assert _is_not_found_error(AGFSClientError(message))
    assert AGFSContextFS._is_not_found_error(AGFSClientError(message))
    assert context_fs.scan_session_recovery("isolated") == ([], None)
    assert context_fs.recovery_account_ids() == []
    assert context_fs.find_archive_uri("archive", ctx) is None


@pytest.mark.parametrize(
    "message", ["permission denied", "connection timed out", "internal server error"]
)
def test_real_storage_failures_still_propagate(context_fs, message):
    context_fs._client.ls.side_effect = AGFSClientError(message)
    assert not _is_not_found_error(AGFSClientError(message))
    assert not AGFSContextFS._is_not_found_error(AGFSClientError(message))
    with pytest.raises(AGFSClientError, match=message):
        context_fs.scan_session_recovery("isolated")


def test_missing_session_directory_does_not_log_a_recovery_error(context_fs, caplog):
    context_fs._client.ls.side_effect = AGFSClientError(
        "no such directory: /accounts/isolated/sessions"
    )
    discovery = ExtractionRecoveryDiscovery(lambda: context_fs, lambda: ["isolated"])
    with caplog.at_level(logging.ERROR):
        assert discovery() == []
    assert not caplog.records
