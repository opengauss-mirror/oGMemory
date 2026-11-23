#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:?请指定输出目录}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
cd "$ROOT"
COPYFILE_DISABLE=1 tar --no-xattrs --no-acls --exclude=__pycache__ --exclude="*.pyc" -czf "$OUT/ogmemory-compose.tar.gz" compose.yaml .env.example deploy/compose/README_zh.md deploy/compose/deploy.sh deploy/compose/ogmemory.yaml deploy/compose/compose.network.yaml deploy/compose/compose.model-network.yaml deploy/compose/scripts docs/integrations
printf '%s\n' "$OUT/ogmemory-compose.tar.gz"
