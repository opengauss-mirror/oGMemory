# oGMemory — Claude Code plugin (English)

Sync Claude Code sessions to oGMemory, inject retrieved memory on each user prompt, report successful tool I/O to the same session buffer as the transcript, and **reuse one HTTP/identity module** for all hook scripts.

## What the hooks do

| Hook | Script | API | Notes |
|------|--------|-----|--------|
| `PostToolUse` | `call_add_session_message.py` | `POST /api/v1/sessions/…/messages` | Only **mutating** tools (`Write`, `Edit`, `MultiEdit`, `Bash`, `NotebookEdit`) — matcher + script allowlist so Read/Glob/Grep noise does not flood the session buffer. Appends a `role: tool` summary (truncated). |
| `UserPromptSubmit` | `call_compose.py` | `POST /api/v1/compose` | RAG + profile → `additionalContext` for the model. |
| `Stop` | `call_after_turn.py` (async) or `stop_detach.sh` | `POST /api/v1/after_turn` | Incremental JSONL → server (byte offset sidecar). Supports headless `-p` mode. |
| `PreCompact` | `call_after_turn.py` | `POST /api/v1/after_turn` | Same script; syncs before compaction. |

Shared code: `scripts/ogm_plugin_request.py` (`base_api_url`, `http_plugin_headers`, `base_ctx`) used by the three `call_*.py` entrypoints. Scripts also prepend the `scripts` directory to `sys.path` so `import ogm_plugin_request` works when the CWD is not the scripts folder.

## Config / auth (optional)

- **Base URL:** `OG_MEMORY_URL` (e.g. `http://127.0.0.1:8090`).

- **Same key as OpenClaw** (`plugins/openclaw` → `authApiKey` / `process.env.OG_AUTH_API_KEY`):

  Set **`OG_AUTH_API_KEY`** in the **environment of the `claude` process** (shell export, service env, etc.).  
  When set, requests also send `X-API-Key`, `X-Account-ID`, and `X-User-ID` (identity from `OgMemConfig` when the repo is on `PYTHONPATH`, else `OG_*` / `OG_MEMORY_*` as implemented in `ogm_plugin_request.py`).

  If your server has no key-based auth, leave it unset. **Do not commit secrets.**

### Automatic capture and recall

`OGMEM_AUTO_CAPTURE` and `OGMEM_AUTO_RECALL` are independent environment
switches, both enabled by default, matching the OpenClaw switch names.
To recall existing memory without automatically writing turns:

```bash
export OGMEM_AUTO_CAPTURE=false
export OGMEM_AUTO_RECALL=true
claude
```

Disabling capture skips transcript reads and `/api/v1/after_turn` on Stop and
PreCompact, as well as PostToolUse session-message writes. The Stop wrapper
creates no temporary file or detached capture worker. No ingest offset is
created or advanced, and no after-turn retry is attempted. Re-enabling capture
resumes from the last successful offset, so conversations from the disabled
period may be imported later.

Disabling recall skips automatic UserPromptSubmit compose and context injection.
Values `1`, `true`, and `yes` enable a switch (case-insensitive, whitespace
trimmed); other explicit values disable it. Restart Claude after changing them.

Trace uploads continue, including message content unless
`OGMEM_TRACE_CAPTURE_CONTENT=0` selects metadata only. Read/Bash output filters
keep their own settings. Explicit `/og-add-history`, `/og-compose`, and MCP
tools remain available. `OGMEM_CODE_MEMORY_ONLY=true` still takes precedence
over generic Trace and session persistence.

### Optional MCP prerequisite for original-output recovery

Read/Bash filtering uses the oGMemory HTTP API. Recovering a filtered original
requires Claude to also connect to the oGMemory MCP server. The plugin does not
start MCP or embed a user-specific URL or credential in the package.

1. Start oGMemory MCP (or use an already running service), for example:

   ```bash
   ogmem mcp --transport streamable-http --config /absolute/path/ogmem.yaml
   ```

2. Add `ogmemory` through Claude Code's supported MCP command. Use the actual
   MCP address and keep `X-Agent-ID` aligned with the plugin's Agent identity:

   ```bash
   claude mcp add --transport http --scope user ogmemory \
     http://127.0.0.1:8091/mcp \
     --header "X-API-Key: <tenant-key>" \
     --header "X-Account-ID: acct-1" \
     --header "X-User-ID: user-1" \
     --header "X-Agent-ID: claude-code"
   ```

3. Run `claude mcp list` to confirm that `ogmemory` is connected, then use
   `/mcp` inside a Claude session to confirm that `ogm_read_memory` is available.
   Set this variable in the same environment that launches Claude:

   ```bash
   export OGMEM_CLAUDE_MCP_RECOVERY_ENABLED=true
   ```

Without this opt-in, filtering still works, but the Hook omits the
`ogm_read_memory(..., recover_original=true)` recovery instruction.

- Unit tests: `pytest tests/unit/claude_plugin/test_plugin_scripts.py -q` (from repo root).

## Skills (slash commands in Claude Code)

| Command | Role |
|--------|------|
| `/og-compose <query>` | Calls compose and prints full context (see skill). |
| `/og-add-history` | Imports project chat history into oGMemory (see skill). |

## Install

**Marketplace:** e.g. add the opengauss `oGMemory` repo, install `og-memory@og-memory`, then `/reload-plugins` (see marketplace docs in repo).

**Local dev:**

```bash
claude --plugin-dir ./plugins/claude-code
# optional: --debug
```

**Prereqs:** reachable oGMemory HTTP API; Python 3; hook scripts are stdlib-only.

## Debugging

- `/reload-plugins` — should show the plugin and hook count.
- With `--debug`, search for `[call_compose]`, `[call_after_turn]`, and `[call_add_session_message]` in hook output.
- Ingest success is reflected in `*.jsonl.ingest-offset` next to the session transcript (see `call_after_turn` sidecar).
