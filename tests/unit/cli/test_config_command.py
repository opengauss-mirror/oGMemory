import argparse

from cli.commands.config import run


def test_config_init_creates_missing_config_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    rc = run(argparse.Namespace(config_action="init"))

    assert rc == 0
    assert (tmp_path / "config" / "ogmem.yaml").is_file()
