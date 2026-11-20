"""Persist recovery-point directory entries before recording backup readiness."""
import os
from pathlib import Path


def sync_directory(path):
    if os.name == "nt":
        # Python does not expose a portable directory FlushFileBuffers handle.
        # SQLite and backup files still flush their own contents on Windows.
        return
    path = Path(path).resolve()
    for directory in (path, path.parent):
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
