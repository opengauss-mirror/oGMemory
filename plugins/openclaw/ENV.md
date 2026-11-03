# OpenClaw Plugin Environment Variables

This file documents variables consumed by `plugins/openclaw/index.js`. Core oG-Memory server configuration is documented in [`../docs/OGMEMORY_ENV.md`](../../docs/OGMEMORY_ENV.md).

`openclaw.json` plugin config has priority over environment variables. When neither is set, the defaults below apply.

## Plugin Runtime

| Variable | Config key | Default | Description |
| --- | --- | --- | --- |
| `OGMEM_MODE` | `mode` | `remote` | `remote` connects to an existing oG-Memory service. `local` starts AGFS + ContextEngine from the plugin. |
| `OGMEM_API_URL` | `memoryApiBaseUrl` | `http://127.0.0.1:8090` | oG-Memory HTTP API base URL. |
| `OG_AUTH_API_KEY` | `authApiKey` | empty | API key sent as `X-API-Key` when oG-Memory role control is enabled. |
| `OG_AUTH_ACCOUNT_ID` | `authAccountId` | empty | Account ID sent with authenticated requests. |

## Local Mode

These are only needed when `mode=local`.

| Variable | Config key | Default | Description |
| --- | --- | --- | --- |
| `AGFS_SERVER_DIR` | `agfsServerDir` | `<repo>/agfs` | AGFS Go server source directory. |
| `AGFS_PORT` | `agfsPort` | `1833` | Local AGFS HTTP port. |
| `AGFS_DATA_DIR` | `agfsDataDir` | `<repo>/../data` | Local AGFS data directory. |
| `CONTEXTENGINE_PORT` | `contextEnginePort` | `8090` | Local ContextEngine HTTP port. |
| `CONTEXTENGINE_PROJECT_ROOT` | `projectRoot` | plugin parent directory | oG-Memory project root used as the working directory for `python server/app.py`. |

## Automatic Capture and Recall

| Variable | Config key | Default | Description |
| --- | --- | --- | --- |
| `OGMEM_AUTO_CAPTURE` | `autoCapture` | `true` | Allow automatic buffering/capture through `ingest` and `afterTurn`. |
| `OGMEM_AUTO_RECALL` | `autoRecall` | `true` | Allow `assemble` to call prefetch/compose and inject memory. |

Explicit JSON `false` overrides the environment. These settings are read when the
plugin registers; restart the Gateway or use a new local CLI process after changes.
For the Docker deployment, the startup renderer writes nonempty
`OGMEM_AUTO_CAPTURE` / `OGMEM_AUTO_RECALL` values into the plugin config, overriding
persisted values for these two keys. Unset or empty variables leave existing
config unchanged. Compose defaults both to `false`; set them explicitly to
`true` to enable automatic capture/recall. Recreate the container after changing
its environment, rather than only restarting it. The entrypoint also refreshes
the image-managed plugin files in the state volume; do not edit that bundled
extension in place.
`autoCapture=false` is not a global write ban: explicit compact, existing-buffer
cleanup, trace upload and tool-result externalization retain their own behavior.
See the evaluation procedure in [README.md](README.md#evaluation-phase-control).

## Compose and Prefetch

| Variable | Config key | Default | Description |
| --- | --- | --- | --- |
| `OGMEM_PREFETCH_ENABLED` | `prefetchEnabled` | `false` | Call `/api/v1/prefetch` before `/api/v1/compose`. |
| `OGMEM_PREFETCH_TOP_K` | `prefetchTopK` | `5` | Maximum hits staged in the session topic buffer. |
| `OGMEM_TOOL_RESULT_EXTERNALIZATION_ENABLED` | `toolResultExternalizationEnabled` | `false` | Enable opt-in Artifact externalization and trusted lifecycle transport. Keep aligned with the server setting. |
| `OGMEM_TOOL_RESULT_FIRST_VIEW_ENABLED` | `toolResultFirstViewExternalizationEnabled` | `true` | Intercept qualifying results after tool execution and before their first model request. |
| `OGMEM_TOOL_RESULT_FIRST_VIEW_MIN_CHARS` | `toolResultFirstViewMinChars` | `20000` | Local first-view prefilter. Keep aligned with server `single_result_chars`; the server remains authoritative. |
| `OGMEM_TOOL_RESULT_FIRST_VIEW_STATE_DIR` | `toolResultFirstViewStateDir` | under `OPENCLAW_STATE_DIR` | Private directory for the HMAC-attested active `user_message` Turn lease shared across plugin registry passes. |
| `OGMEM_TOOL_RESULT_SEMANTIC_TURN_MODE` | `semanticTurnMode` | `user_message` | Externalization lifecycle. The only currently supported value is `user_message`; the option is retained for future extensions. |

OpenClaw 2026.6.10 uses one outer user message as the semantic Turn. It calls
`assemble()` before the agent/tool loop and sends the complete Turn to
`afterTurn()` with `prePromptMessageCount`, so `user_message` is the default.

## Compact Takeover

| Variable | Config key | Default | Description |
| --- | --- | --- | --- |
| `OGMEM_COMPACT_TAKEOVER_ENABLED` | `compactTakeoverEnabled` | `true` | Let oG-Memory generate the final compact result. When disabled, the plugin still calls `prepare_compaction` before delegating compact back to OpenClaw. |
| `OGMEM_SUMMARY_MAX_CHARS` | `summaryMaxChars` | `4000` | Maximum compact summary length requested from oG-Memory. |
| `OGMEM_SHORT_TERM_INDEX_MODE` | `shortTermIndexMode` | `sync` | Short-term index update mode during compact takeover: `sync`, `async`, or `off`. |

## Identity Defaults

These can also be provided in OpenClaw lifecycle params.

| Variable | Config key | Default | Description |
| --- | --- | --- | --- |
| none | `accountId` | empty | Default account ID. |
| none | `userId` | empty | Default user ID. |
| none | `agentId` | empty | Fallback agent ID. Trace and CE share the session runtime agent ID; OpenClaw session keys supply the agent ID before hooks arrive. Register and authorize that runtime agent in oGMemory. |

## Local ContextEngine Provider

When `mode=local`, the plugin passes these values to the spawned ContextEngine process.

| Variable | Config key | Default | Description |
| --- | --- | --- | --- |
| `CONTEXTENGINE_PROVIDER` | `llmProvider` | `mock` | LLM provider for ContextEngine, for example `mock` or `openai`. |
| `OPENAI_API_KEY` | `llmApiKey` | empty | LLM API key passed to ContextEngine. |
| `OPENAI_BASE_URL` | `llmBaseUrl` | empty | OpenAI-compatible LLM base URL passed to ContextEngine. |
| `OPENAI_LLM_MODEL` | `llmModel` | `gpt-4o-mini` | LLM model passed to ContextEngine. |
| `EMBEDDING_PROVIDER` | `embeddingProvider` | `llmProvider` | Embedding provider. |
| `OPENAI_EMBEDDING_API_KEY` | `embeddingApiKey` | `llmApiKey` | Embedding API key. |
| `OPENAI_EMBEDDING_BASE_URL` | `embeddingBaseUrl` | `llmBaseUrl` | Embedding API base URL. |
| `OPENAI_EMBEDDING_MODEL` | `embeddingModel` | `text-embedding-ada-002` | Embedding model. |

Note: the local-mode provider variables intentionally use the `OPENAI_*` names because `index.js` passes those names to the spawned ContextEngine process. For a standalone oG-Memory server, prefer the unified `OGMEM_*` variables documented in `docs/OGMEMORY_ENV.md`.

## Example

```bash
export OGMEM_MODE=remote
export OGMEM_API_URL=http://127.0.0.1:8090
export OGMEM_PREFETCH_ENABLED=true
export OGMEM_PREFETCH_TOP_K=5
export OGMEM_TOOL_RESULT_EXTERNALIZATION_ENABLED=false
export OGMEM_TOOL_RESULT_SEMANTIC_TURN_MODE=user_message
export OGMEM_COMPACT_TAKEOVER_ENABLED=true
export OGMEM_SHORT_TERM_INDEX_MODE=sync
```

## Verification

```bash
openclaw doctor
curl http://127.0.0.1:8090/api/v1/health
node plugins/openclaw/compact.test.js
```

Look for `[og-memory]` lines in OpenClaw gateway logs. Compose should call `/compose`; compact takeover should call `/prepare_compaction` before `/compact`.
