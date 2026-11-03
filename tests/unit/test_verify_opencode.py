"""Check that verification credentials are private before publication."""

import json
import os
import stat
from pathlib import Path

import pytest

from scripts.verify_opencode import _write_private_config


@pytest.mark.skipif(os.name == "nt", reason="POSIX file permissions")
@pytest.mark.parametrize("existing", [False, True])
def test_config_is_private_before_atomic_publication(tmp_path, monkeypatch, existing):
    path = tmp_path / "ogmemory-config.json"
    if existing:
        path.write_text("old content", encoding="utf-8")
        path.chmod(0o644)
    replace = os.replace
    observed = []

    def inspect_replace(source, destination):
        assert stat.S_IMODE(os.stat(source).st_mode) == 0o600
        observed.append(json.loads(Path(source).read_text(encoding="utf-8")))
        replace(source, destination)

    monkeypatch.setattr(os, "replace", inspect_replace)
    previous_umask = os.umask(0o022)
    try:
        _write_private_config(path, {"apiKey": "test-key"})
    finally:
        os.umask(previous_umask)
    assert observed == [{"apiKey": "test-key"}]
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(path.read_text(encoding="utf-8")) == {"apiKey": "test-key"}
    assert list(tmp_path.iterdir()) == [path]
