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

# Stop/PreCompact wrapper — detaches the Python child so Claude Code's `-p`
# mode shutdown does not kill it. Background:
#
#   Plugin-loaded Stop hooks (--plugin-dir) are async-backgrounded by Claude
#   Code. In -p mode Claude exits ~130ms after spawning the hook subprocess —
#   before python (~50-100ms cold start) can run, leaving `outcome: cancelled`
#   in the stream JSONL and never invoking call_after_turn.py.
#
#   This wrapper exits in ~5ms (so the stream records `outcome: success`) and
#   launches call_after_turn.py inside a fresh session via `setsid`, which
#   moves it out of Claude's process group and immunizes it from teardown.
#
# stdin (hook JSON) is written to a temp file and read by python from that path.
# We never pass hook JSON through bash -c string expansion (avoids injection
# from single quotes or other shell-sensitive bytes in tool/content fields).
set -u

# Match Python's boolean parser. Unset preserves capture; any explicit value
# other than 1/true/yes disables it before reading stdin or spawning a child.
if [[ -v OGMEM_AUTO_CAPTURE ]] && ! [[ "$OGMEM_AUTO_CAPTURE" =~ ^[[:space:]]*(1|true|yes)[[:space:]]*$ ]]; then
  exit 0
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$HERE/call_after_turn.py"

INPUT="$(cat)"
TMPFILE="$(mktemp)"
printf '%s' "$INPUT" >"$TMPFILE"

# `setsid ... &` + `disown` fully detach grandchild from this process group.
# Only SCRIPT and TMPFILE (paths) are passed into bash -c, not the JSON body.
setsid bash -c 'python3 "$1" <"$2"; rm -f "$2"' _ "$SCRIPT" "$TMPFILE" </dev/null >/dev/null 2>&1 &
disown 2>/dev/null || true

exit 0
