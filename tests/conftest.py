"""Pytest configuration and shared fixtures for ContextEngine tests.

This module provides fixtures for testing with AGFS server.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.models import RequestContext, ContextNode
from core.interfaces import ContextFS, VectorIndex, Embedder
from pyagfs import AGFSClient
from fs.agfs_adapter import AGFSContextFS
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from providers.embedder.mock_embedder import MockEmbedder


# ============================================================================
# Configuration
# ============================================================================

AGFS_BASE_URL = os.environ.get("AGFS_BASE_URL", "http://localhost:1833")
AGFS_DATA_DIR = os.environ.get("AGFS_DATA_DIR", str(Path(__file__).parent.parent / ".agfs_test_data"))


# ============================================================================
# AGFS Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def agfs_server_info():
    """Check if AGFS server is available."""
    import requests

    try:
        response = requests.get(f"{AGFS_BASE_URL}/api/v1/health", timeout=2)
        if response.status_code == 200:
            return {"available": True, "url": AGFS_BASE_URL}
    except Exception:
        pass

    pytest.skip(f"AGFS server not available at {AGFS_BASE_URL}. Start it with:\n"
                "  cd agfs/agfs-server && ./build/agfs-server -c config.yaml")


@pytest.fixture(scope="function")
def agfs_client(agfs_server_info):
    """Create AGFS client for testing."""
    return AGFSClient(api_base_url=agfs_server_info["url"])


@pytest.fixture(scope="function")
def agfs(agfs_client, tmp_path):
    """Create AGFS ContextFS for testing.

    Each test gets an isolated mount point under /local/test_{test_name}/
    """
    # Create unique test directory in localfs
    test_name = tmp_path.name[:16]  # Use part of temp dir name as unique ID
    mount_prefix = f"/local/test_{test_name}"

    fs = AGFSContextFS(client=agfs_client, mount_prefix=mount_prefix)

    # Create test base directory
    try:
        agfs_client.mkdir(mount_prefix)
    except Exception:
        pass  # Directory may already exist

    yield fs

    # Cleanup: remove test directory
    try:
        # Note: AGFS rm doesn't work recursively via HTTP, so we clean files individually
        # For now, we leave test data in place
        pass
    except Exception:
        pass


@pytest.fixture(scope="function")
def sample_context():
    """Create a sample RequestContext for testing."""
    import uuid
    return RequestContext(
        account_id="test-account",
        user_id="test-user",
        agent_id="test-agent",
        session_id=str(uuid.uuid4()),
        trace_id=str(uuid.uuid4()),
    )


@pytest.fixture(scope="function")
def sample_context_a():
    """Context for user A in multi-tenant tests."""
    import uuid
    return RequestContext(
        account_id="account-a",
        user_id="alice",
        agent_id="agent",
        session_id=str(uuid.uuid4()),
        trace_id=str(uuid.uuid4()),
    )


@pytest.fixture(scope="function")
def sample_context_b():
    """Context for user B in multi-tenant tests."""
    import uuid
    return RequestContext(
        account_id="account-b",
        user_id="bob",
        agent_id="agent",
        session_id=str(uuid.uuid4()),
        trace_id=str(uuid.uuid4()),
    )


@pytest.fixture(scope="function")
def vector_index():
    """Create InMemoryVectorIndex for testing."""
    # Use dimension compatible with text-embedding-ada-002
    return InMemoryVectorIndex(dimension=1536)


@pytest.fixture(scope="function")
def mock_embedder():
    """Create mock embedder for testing."""
    return MockEmbedder(dimension=1536)


# ============================================================================
# Legacy fixture aliases (for backward compatibility)
# ============================================================================

@pytest.fixture(scope="function")
def mock_agfs(agfs):
    """Alias for agfs (legacy name)."""
    return agfs


@pytest.fixture(scope="function")
def mock_vector_index(vector_index):
    """Alias for vector_index (legacy name)."""
    return vector_index


# ============================================================================
# Test data helpers
# ============================================================================

@pytest.fixture(scope="function")
def sample_profile_node():
    """Create a sample profile ContextNode."""
    import uuid
    return ContextNode(
        uri="ctx://test-account/users/test-user/memories/profile",
        context_type="MEMORY",
        category="profile",
        level=0,
        owner_space="user:test-user",
        abstract="Test user is a software engineer",
        overview="## Profile\n\nName: Test User\nRole: Software Engineer",
        content="Full profile content here...",
        metadata={"status": "ACTIVE"},
    )


@pytest.fixture(scope="function")
def sample_preference_node():
    """Create a sample preference ContextNode."""
    import uuid
    return ContextNode(
        uri="ctx://test-account/users/test-user/memories/preferences/coding",
        context_type="MEMORY",
        category="preferences",
        level=0,
        owner_space="user:test-user",
        abstract="Prefers PEP8 coding style",
        overview="## Coding Preferences\n\n- PEP8 compliant\n- Type hints",
        content="Detailed coding preferences...",
        metadata={"status": "ACTIVE"},
    )


@pytest.fixture(scope="function")
def mock_llm():
    """Create a mock LLM for testing."""
    from providers.llm import MockLLM
    return MockLLM()


@pytest.fixture
def service_threading(monkeypatch):
    import threading
    from types import SimpleNamespace
    monkeypatch.setattr("server.memory_service.threading", SimpleNamespace(**vars(threading)))
