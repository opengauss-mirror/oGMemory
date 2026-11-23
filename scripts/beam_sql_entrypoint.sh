#!/bin/sh
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

# BEAM-only SQL deployment. MemoryService owns native SQL outbox indexing and
# graph projection. Never start a competing standalone index-service process.
set -eu
python3 <<'PY'
import os
from pathlib import Path
import yaml

source = Path(os.environ['OGMEM_CONFIG'])
cfg = yaml.safe_load(source.read_text())
assert cfg['storage']['backend'] == 'sql'
assert cfg['storage']['connection_string'] == cfg['vector_db']['connection_string']
assert not cfg['mcp']['enabled'] and not cfg['cache']['enabled']
destination = Path('/tmp/ogmem.resolved.yaml')
destination.write_text(yaml.safe_dump(cfg))
destination.chmod(0o600)
print('[entrypoint] BEAM private SQL config resolved; MemoryService owns indexing', flush=True)
PY
export OGMEM_CONFIG=/tmp/ogmem.resolved.yaml
cd /opt/ogmem
exec python3 -m gunicorn --bind "127.0.0.1:${OGMEM_HTTP_PORT}" \
    --workers 1 --timeout 900 --access-logfile - --error-logfile - server.app:app
