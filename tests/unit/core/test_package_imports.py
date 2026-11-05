"""Public entry points load in fresh worker processes."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("entry", ["commit.context_writer", "core.react_loop", "service.api"])
def test_public_entry_points_in_fresh_process(entry):
    result = subprocess.run(
        [sys.executable, "-c", f"import {entry}; from core import ReactLoop; "
         "from core.react_loop import ReactLoop as Direct; "
         "assert ReactLoop is Direct; print('worker imports ready')"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "worker imports ready"
