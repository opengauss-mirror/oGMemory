"""Normal native reads through the standalone and migration entrypoints."""

import os
from uuid import uuid4

import pytest
import yaml

from fs.agfs_adapter.client import create_agfs_client
from providers.unified_config import OgMemConfig
from scripts.run_index_service import _create_storage
from service.migration.storage import AGFSStorage


def test_native_entrypoints_share_written_file(tmp_path):
    library = os.environ.get("AGFS_NATIVE_LIBRARY")
    if not library:
        pytest.skip("AGFS_NATIVE_LIBRARY supplies the integration library")
    mount = "/entrypoints-" + uuid4().hex
    config = tmp_path / "agfs.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "plugins": {
                    "localfs": {
                        "enabled": True,
                        "path": mount,
                        "config": {"local_dir": str(tmp_path)},
                    }
                }
            }
        )
    )
    cfg = OgMemConfig(
        storage_backend="agfs",
        agfs_transport="native",
        agfs_native_library=library,
        agfs_native_config=str(config),
        agfs_mount_prefix=mount,
    )
    path = mount + "/shared.txt"
    expected = "native transport 数据".encode()
    create_agfs_client(cfg).write(path, expected)
    fs, outbox, pool = _create_storage(cfg)
    assert fs._client.read(path) == expected
    assert AGFSStorage(cfg).client.read(path) == expected
