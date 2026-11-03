#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:?请指定输出目录}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
cd "$ROOT"
COPYFILE_DISABLE=1 tar --no-xattrs --no-acls --exclude=__pycache__ --exclude="*.pyc" -czf "$OUT/ogmemory-compose.tar.gz" compose.yaml .env.example deploy/compose/README_zh.md deploy/compose/deploy.sh deploy/compose/ogmemory.yaml deploy/compose/compose.network.yaml deploy/compose/compose.model-network.yaml deploy/compose/scripts docs/integrations
printf '%s\n' "$OUT/ogmemory-compose.tar.gz"
