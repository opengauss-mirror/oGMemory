"""Deterministic tokenization for code-search tests, including MCP children."""
from contextlib import contextmanager
import re
from unittest.mock import patch

import pytest


@contextmanager
def local_code_tokenizer():
    with patch("retrieval.bm25_index._tokenize", lambda text: re.findall(r"[a-z0-9]+", text.lower())):
        yield


@pytest.fixture(autouse=True)
def deterministic_code_tokenizer():
    with local_code_tokenizer():
        yield
