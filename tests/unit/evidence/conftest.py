"""Shared in-memory writer and evidence store for evidence unit tests."""

from unittest.mock import Mock

import pytest

from fs.agfs_adapter import AGFSContextFS
from service.api import MemoryWriteAPI
from service.evidence_store import EvidenceStore
from tests.unit.fs.test_agfs_context_fs import _AtomicMemoryAGFSClient


@pytest.fixture
def setup():
    fs = AGFSContextFS(_AtomicMemoryAGFSClient(), mount_prefix="/local")
    llm = Mock()
    llm.complete_json.side_effect = AssertionError("deterministic writes cannot call the LLM")
    api = MemoryWriteAPI(fs, llm, directory_summary_enabled=False)
    return fs, api, EvidenceStore(fs), llm
