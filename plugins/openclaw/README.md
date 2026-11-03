# oG-Memory Context Engine - OpenClaw Plugin

OpenClaw context-engine plugin backed by oG-Memory. It connects OpenClaw lifecycle calls to the oG-Memory HTTP API, injects layered memory context before model inference, persists new memories after each turn, and supports oG-Memory-owned compact takeover.

## Prerequisites

- Node.js >= 18
- OpenClaw installed and running
- An oG-Memory HTTP service reachable from the OpenClaw gateway, or local-mode prerequisites for starting AGFS + ContextEngine from this plugin

## Install

```bash
cd plugins/openclaw
openclaw plugins install -l .
```

## Configure

Set the context engine slot in `~/.openclaw/openclaw.json`.

### Remote Mode

Use `remote` when oG-Memory is already running.

```jsonc
{
  "plugins": {
    "slots": {
      "contextEngine": "og-memory-context-engine"
    },
    "entries": {
      "og-memory-context-engine": {
        "enabled": true,
        "config": {
          "mode": "remote",
          "memoryApiBaseUrl": "http://127.0.0.1:8090",
          "traceTimeoutMs": 3000,
          "traceCaptureContent": true,
          "prefetchEnabled": true,
          "prefetchTopK": 5,
          "toolResultExternalizationEnabled": false,
          "toolResultFirstViewExternalizationEnabled": true,
          "toolResultFirstViewMinChars": 20000,
          "semanticTurnMode": "user_message",
          "compactTakeoverEnabled": true,
          "summaryMaxChars": 4000,
          "shortTermIndexMode": "sync"
        }
      }
    }
  }
}
```

`toolResultExternalizationEnabled=false` preserves the original OpenClaw
message and lifecycle path. Enable it only when the oGMemory server's
`tool_result_externalization.enabled` setting is also enabled.

With first-view externalization enabled, OpenClaw's trusted tool-result
middleware sends each result at or above `toolResultFirstViewMinChars` to
oGMemory before its first model request. oGMemory persists the complete result
as an owner/session/Turn/tool-call-bound Artifact and returns a bounded
multiresolution stub. The stub retains deterministic synopsis, key facts,
selected evidence, critical records and exact byte ranges; callers can search
Artifact indexes and recall only the precise matching fragments when more
detail is required. Store, transport or identity failures fail open and retain
the original result.

For OpenClaw 2026.6.10, keep `semanticTurnMode=user_message`. The final
`afterTurn()` carries the complete user-to-terminal-assistant delta and
`prePromptMessageCount`. oGMemory's owner-bound HMAC Turn Ledger validates that
ordered delta, provides stable source identities across retries/restarts and
reuses any first-view Artifact without duplicating it. Individual results
below the configured single-result threshold remain inline.

### Local Mode

Use `local` when the OpenClaw gateway should start AGFS and ContextEngine.

```jsonc
{
  "plugins": {
    "slots": {
      "contextEngine": "og-memory-context-engine"
    },
    "entries": {
      "og-memory-context-engine": {
        "enabled": true,
        "config": {
          "mode": "local",
          "projectRoot": "/path/to/oG-Memory",
          "agfsServerDir": "/path/to/oG-Memory/agfs",
          "agfsDataDir": "/path/to/ogmem-data",
          "contextEnginePort": 8090,
          "toolResultExternalizationEnabled": false,
          "toolResultFirstViewExternalizationEnabled": true,
          "toolResultFirstViewMinChars": 20000,
          "semanticTurnMode": "user_message",
          "llmProvider": "openai",
          "llmApiKey": "sk-...",
          "llmBaseUrl": "https://api.openai.com/v1",
          "llmModel": "gpt-4o-mini",
          "llmReasoningEffort": "xhigh",
          "llmIsReasoningModel": true,
          "llmExtraBody": { "thinking": { "type": "enabled" } }
        }
      }
    }
  }
}
```

Restart OpenClaw after changing the config.

```bash
openclaw restart
```

### Conversation Hook Permission

OpenClaw treats conversation-bearing hooks as host-level permissions. For a non-bundled plugin, add `hooks` beside `config` under the plugin entry (not inside the plugin config schema):

```json
{
  "plugins": {
    "entries": {
      "og-memory-context-engine": {
        "hooks": { "allowConversationAccess": true }
      }
    }
  }
}
```

Without this setting, OpenClaw host policy blocks raw conversation payloads for `before_agent_run` and `agent_end`, so those trace events cannot capture the user prompt or final assistant text.

### Trace Capture

The plugin registers OpenClaw typed hooks and converts them to the same canonical event shape used by other agents before posting to `POST /api/v1/trace/events`. After hook2Event conversion, oG-Memory's internal trace flow does not branch on OpenClaw-specific hook names.

| OpenClaw hook | Canonical event |
| --- | --- |
| `session_start` | `session_start` |
| `before_agent_run` | `user_message` |
| `before_tool_call` | `tool_start` |
| successful `after_tool_call` | `tool_success` |
| failed `after_tool_call` | `tool_failure` |
| `agent_end` | `turn_end` |
| `before_compaction` | `pre_compact` |
| `session_end` | `session_end` |

Trace upload is non-blocking for the OpenClaw turn. Each event is attempted at most twice, and each attempt uses `traceTimeoutMs` (default: 3000 milliseconds). If the service is unavailable or both attempts fail, the plugin logs the error and lets the agent continue. Set `traceCaptureContent=false` to keep size metadata while omitting redacted prompt, tool, error, and assistant content fields.

The OpenClaw hook2Event adapter redacts sensitive values before transport. Emails, phone numbers, government IDs, bank cards, IP addresses, bearer tokens, API keys, cookies, and credential-bearing connection strings are replaced with irreversible placeholders such as `{{email_1}}` or `{{connection_string_1}}`. Placeholder numbering is stable only inside one event; no cross-event secret map is retained.

## How It Works

```
OpenClaw gateway
  |
  | lifecycle call
  v
plugins/openclaw/index.js
  |
  | HTTP
  v
oG-Memory /api/v1/{prefetch,compose,after_turn,prepare_compaction,compact,dispose}
```

Main lifecycle behavior:

| OpenClaw method | oG-Memory calls | Behavior |
| --- | --- | --- |
| `assemble()` | optional `/prefetch`, then `/compose` | Stages quick retrieval hits, then returns layered messages for Profile, Archive History, Session State, and Working Set. |
| `afterTurn()` | `/after_turn` | Sends completed turn messages for extraction, persistence and trusted tool-result identity reconciliation. |
| `compact()` | `/prepare_compaction`, then `/compact` when takeover is enabled | Runs pre-compaction extraction, validates a one-time prepare token, commits/archive synchronously, and returns a compact summary. |
| `dispose()` | none | Releases per-run resources without ending the memory session. The `/dispose` call runs on every OpenClaw `session_end`; only an explicit `reason=deleted` additionally deletes the oGMemory Session and its tool-result Artifacts. |

When Artifact persistence is unavailable, an attested `after_turn` checkpoint is
sent once and retried at most once. After that retry fails, the adapter
uses the server-issued, owner- and checkpoint-bound HMAC fallback token to
persist that checkpoint's original messages through the default SessionBuffer
and memory-extraction path. The fallback applies only to that failed checkpoint;
later checkpoints continue attempting normal externalization. If the oGMemory
service itself is unreachable, no authorized fallback can be completed.

Session lifecycle conflicts use typed HTTP 409 responses. A transient
`session_replaced` conflict is retried only when the server explicitly marks it
retryable. A `session_deleting` conflict is never replayed, preventing a stale
`after_turn` checkpoint from recreating a Session after explicit deletion.

`compactTakeoverEnabled=false` keeps OpenClaw responsible for final compact generation, but the plugin still calls `prepare_compaction` so oG-Memory can extract the delta before OpenClaw compacts the local transcript.

## Project Structure

```
plugins/openclaw/
├── index.js                 # Plugin entry, config resolver, HTTP bridge, lifecycle handlers
├── compaction-boundary.js   # Persists OpenClaw compaction boundary entries during takeover
├── process.js               # Local-mode health checks, port probing, process helpers
├── compact.test.js          # Node tests for config, prefetch, and compact takeover behavior
├── trace.test.js            # Node tests for trace hooks, redaction, and upload behavior
├── source-identity.test.js  # Trusted lifecycle identity and fail-open tests
├── first-view-externalization.test.js # Tool-result middleware tests
├── openclaw.plugin.json     # Plugin manifest and config schema
├── package.json
├── ENV.md                   # Environment variable reference
└── README.md
```

## Verify

```bash
openclaw doctor
npm test --prefix plugins/openclaw
```

For a remote service, also verify oG-Memory directly:

```bash
curl http://127.0.0.1:8090/api/v1/health
```

Gateway logs should show `[og-memory]` lines for `compose`, `after_turn`, and compact control flow.

## Switching Back

To revert to the built-in context engine:

```jsonc
{
  "plugins": {
    "slots": {
      "contextEngine": "legacy"
    }
  }
}
```

Then restart the gateway.


## Evaluation Phase Control

The native plugin supports independent `autoCapture` and `autoRecall` booleans
under `plugins.entries.og-memory-context-engine.config`. Both default to `true`.

| Phase | autoCapture | autoRecall |
| --- | --- | --- |
| Normal use | true | true |
| Historical ingestion | true | false |
| QA | false | true |

`autoCapture=false` skips `ingest` buffering and `afterTurn` capture, including
creation of new after-turn retry entries. `autoRecall=false` returns the current
messages unchanged without prefetch/compose. It does not prevent capture.

These are automatic behavior switches, not a global read-only mode. Explicit
compact still processes previously buffered content; dispose/session_end can
finish existing work. They do not cancel backend jobs already in progress or
prevent writes through other clients, trace hooks, or tool-result externalization.

For controlled text QA:

1. Ingest history with capture enabled, then wait for extraction and indexing.
2. Set capture off and recall on, then **restart the dedicated Gateway**, or use
   a new `--local` CLI process. The running plugin does not hot-reload these flags.
3. Use fresh QA sessions and disable explicit memory-writing tools. Separately
   disable tool-result externalization and trace-driven skill sedimentation in
   the test deployment if they are enabled. `traceCaptureContent=false` removes
   trace content but does not turn off trace uploads.
4. Restore the original configuration after evaluation and reload it again.

Example (merge these fields into the existing plugin config):

```json
{
  "autoCapture": false,
  "autoRecall": true
}
```

These settings do not modify the backend `memory.after_turn_threshold`.
