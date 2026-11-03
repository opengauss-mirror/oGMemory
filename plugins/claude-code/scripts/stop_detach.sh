#!/usr/bin/env bash
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

# The code-only profile intentionally owns no generic after-turn lifecycle.
# Keep this ahead of stdin, mktemp, and detached-child setup so Stop is inert.
shopt -s nocasematch
if [[ "${OGMEM_CODE_MEMORY_ONLY:-}" =~ ^[[:space:]]*(1|true|yes)[[:space:]]*$ ]]; then
  exit 0
fi

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
