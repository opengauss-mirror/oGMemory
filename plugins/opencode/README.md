# oGMemory OpenCode Plugin

This package connects OpenCode directly to oGMemory lifecycle APIs:

- retrieve memory with `POST /api/v1/compose`;
- write user and assistant messages with `POST /api/v1/after_turn`;
- record canonical session, message, tool, compaction, and error Trace events with `POST /api/v1/trace/events`;
- include `parent_session_id` when OpenCode reports a parent session.

It operates above MCP. MCP remains available for explicit memory tools, but normal recall and write-back do not depend on the model choosing a tool.

## Requirements

- OpenCode 1.15.11 or later
- Node.js 18 or later
- A running oGMemory HTTP service

The plugin targets one current OpenCode lifecycle contract instead of probing and emulating older SDK versions.

## Install from npm

Add the package to `~/.config/opencode/opencode.json`:

```json
{
  "plugin": ["@opengauss/ogmemory-opencode-plugin"]
}
```

OpenCode installs configured npm plugins automatically. Restart OpenCode after changing the configuration.

## Install from source

Copy this directory and its wrapper into OpenCode's global plugin directory:

```text
~/.config/opencode/plugins/
├── ogmemory.js
└── ogmemory/
    ├── index.mjs
    ├── package.json
    └── lib/
```

On Unix-like systems:

```bash
mkdir -p ~/.config/opencode/plugins/ogmemory
cp plugins/opencode/wrappers/ogmemory.js ~/.config/opencode/plugins/ogmemory.js
cp plugins/opencode/index.mjs plugins/opencode/package.json ~/.config/opencode/plugins/ogmemory/
cp -r plugins/opencode/lib ~/.config/opencode/plugins/ogmemory/
```

See [INSTALL-ZH.md](INSTALL-ZH.md) for Windows instructions.

## Configuration

Create `~/.config/opencode/ogmemory-config.json`:

```json
{
  "enabled": true,
  "autoCapture": true,
  "autoRecall": true,
  "endpoint": "http://127.0.0.1:8090",
  "apiKey": "",
  "accountId": "",
  "userId": "",
  "agentId": "opencode",
  "composeTimeoutMs": 5000,
  "writeTimeoutMs": 15000,
  "traceTimeoutMs": 3000,
  "traceCaptureContent": true
}
```

Environment variables override identity and endpoint fields:

- `OG_MEMORY_URL`
- `OG_AUTH_API_KEY`
- `OG_AUTH_ACCOUNT_ID`
- `OG_AUTH_USER_ID`
- `OG_AUTH_AGENT_ID`
- `OGMEM_TRACE_CAPTURE_CONTENT`

Plaintext HTTP is accepted only for loopback endpoints. Remote oGMemory services must use HTTPS.

## Runtime behavior

OpenCode message events are buffered in memory until a completed assistant message, idle boundary, compaction, error, deletion, or plugin shutdown. A failed `after_turn` call remains in memory and is retried at the next boundary while the OpenCode process is alive. Synthetic and compaction-summary messages are excluded.

Trace delivery is asynchronous and never waits in a user or tool hook. Trace payloads are bounded and locally redacted before delivery. Failed Trace requests are logged and dropped; this first release intentionally has no disk spool, cross-process lock, crash recovery, or compatibility emulation.

All oGMemory failures are fail-open for the OpenCode conversation. At most 500 unsent messages are held per session; when the limit is exceeded, the oldest message is dropped with a warning.

Set `enabled` to `false` to disable all hooks and network calls.

### Automatic capture and recall

`autoCapture` and `autoRecall` are independent and
default to `true`. `autoCapture=false` prevents automatic message buffering,
write-back and retries through `/api/v1/after_turn`, including completion, idle,
compaction, error, deletion and shutdown boundaries. `autoRecall=false` skips
`/api/v1/compose` and system-prompt memory injection while leaving capture enabled.

To read existing memory without writing new turns, merge these fields into your
configuration:

```json
{
  "enabled": true,
  "autoCapture": false,
  "autoRecall": true
}
```

`OGMEM_AUTO_CAPTURE` and `OGMEM_AUTO_RECALL` supply defaults when the corresponding
JSON field is absent; explicit JSON `true` or `false` takes priority. Environment
values `0`, `false`, `off` and `no` disable the corresponding feature (case-insensitive).
Restart OpenCode after changing these settings.

These switches control the native plugin's automatic memory workflow. Session,
tool, compaction and error Trace events continue to be uploaded; message and
`turn_end` Trace events depend on the capture buffer and are skipped when capture
is disabled. Explicit MCP memory tools retain their own behavior.

## Verify

After one OpenCode turn, confirm:

1. `/api/v1/compose` receives the user prompt.
2. `/api/v1/after_turn` receives both the user and assistant messages.
3. `/api/v1/trace/events` contains `user_message`, `assistant_message`, and `turn_end`.
4. Tool use produces `tool_start` and a terminal tool event.

## Development

```bash
npm run check
npm test
npm pack --dry-run
```
