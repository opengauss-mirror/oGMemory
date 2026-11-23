#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Read ogmemory YAML config and print shell export statements.

Used by entrypoint-standalone.sh to make service.http_port and service
concurrency available before gunicorn starts.

Usage (in shell):
    eval $(python3 /opt/ogmem/scripts/yaml_to_env.py)
"""

import os
import sys
from urllib.parse import urlsplit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from providers.unified_config import OgMemConfig
except Exception:
    print("Cannot import standalone configuration loader", file=sys.stderr)
    sys.exit(1)

try:
    cfg = OgMemConfig.load()
except Exception:
    print("Cannot load standalone configuration; refusing to start", file=sys.stderr)
    sys.exit(1)

print(f'export OGMEM_HTTP_PORT="{cfg.http_port}"')
# Only the legacy local default is managed by this image. External AGFS
# endpoints and SQL deployments must not spawn an unrelated local server.
agfs_url = urlsplit(cfg.agfs_base_url)
bundled_agfs = (
    cfg.storage_backend == "agfs"
    and agfs_url.scheme == "http"
    and agfs_url.hostname in {"127.0.0.1", "localhost"}
    and agfs_url.port == 1833
)
print(f'export OGMEM_START_BUNDLED_AGFS="{str(bundled_agfs).lower()}"')
print(f'export OGMEM_WORKERS="{cfg.workers}"')
print(f'export OGMEM_THREADS="{cfg.threads}"')
print(f'export OGMEM_MCP_ENABLED="{str(cfg.mcp_enabled).lower()}"')
print(f'export OGMEM_MCP_HOST="{cfg.mcp_host}"')
print(f'export OGMEM_MCP_PORT="{cfg.mcp_port}"')
print(f'export OGMEM_MCP_PATH="{cfg.mcp_path}"')
