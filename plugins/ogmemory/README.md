# oGMemory Codex plugin

Codex plugin integration over the oGMemory HTTP API.

## Lifecycle coverage

| Codex hook | oGMemory use |
| --- | --- |
| `SessionStart` | Bootstrap the session and inject durable workspace context |
| `UserPromptSubmit` | Capture the prompt and inject relevant memory |
| `PreToolUse` | Capture tool intent and inject relevant constraints and prior outcomes |
| `PermissionRequest` | Capture permission request context as lifecycle trace data |
| `PostToolUse` | Capture bounded tool input and output |
| `PreCompact` | Prepare extraction and compaction state |
| `PostCompact` | Commit oGMemory compaction state |
| `SubagentStart` | Prepare the subagent and inject relevant context |
| `SubagentStop` | Capture the subagent result and close its lifecycle |
| `Stop` | Persist the completed user/assistant turn |
| `SessionEnd` | Finalize trace data and dispose the session |

Every event is also sent to `/api/v1/trace/events` for turn reconstruction and reusable-skill extraction. Trace content is bounded and common credential fields are redacted.

## Automatic memory controls

`OGMEM_AUTO_CAPTURE` and `OGMEM_AUTO_RECALL` default to `true` and independently
control automatic writes and recall. Set `OGMEM_AUTO_CAPTURE=false` to skip prompt
caching, turn writes, bootstrap, compaction, subagent writes and disposal while
keeping recall and lifecycle traces. Existing pending state is preserved and can
be submitted after capture is re-enabled. Trace content remains enabled.

The user configuration also accepts `auto_capture` and `auto_recall` booleans;
environment variables take precedence. Values `1`, `true` and `yes` enable the
feature (case and surrounding whitespace are ignored); other explicit values
disable it. Restart Codex after changing its environment.

## Service connection

The plugin can connect to an existing remote oGMemory service. Create `.ogmemory/codex-plugin.json` in the user home directory with the connection details supplied by the service administrator. Use an empty `api_key` when authentication is disabled:

```json
{
  "url": "https://memory.example.com",
  "api_key": "<tenant-key>",
  "account_id": "<account-id>",
  "user_id": "<user-id>",
  "agent_id": "codex"
}
```

For local development, install and start oGMemory with `ogmem onboard --mode headless`. The default local service URL is `http://127.0.0.1:8090`; local default settings require no configuration file.

Automatic Skill recall requires authenticated Skill HTTP access. The administrator
must enable `auth.role_control_enabled` and configure privileged API keys in the
server configuration. Set the issued `api_key`, `account_id`, `user_id`, and
`agent_id` in `~/.ogmemory/codex-plugin.json` above, using credentials authorized
for that agent. Empty-key configurations support ordinary memory hooks; Skill
HTTP authentication failures produce a diagnostic on hook stderr. A 401 means
authentication configuration needs attention; a 403 means access needs attention.

## Install from this repository

```text
codex plugin marketplace add https://gitcode.com/opengauss/oGMemory.git --ref dev --sparse .agents/plugins --sparse plugins/ogmemory
codex plugin add ogmemory@ogmemory
```

Refresh Codex, install `ogmemory`, then open `/hooks` and trust its lifecycle hooks.

## Verify

- `/hooks` lists all 11 lifecycle events as trusted.
- A new session receives recalled workspace context.
- Tool calls appear in the oGMemory session trace.
- A completed turn becomes recallable in a later Codex session.
- Compaction, subagent, and session-end events appear in the trace.

The bridge is fail-open: an unavailable memory service does not block Codex.
