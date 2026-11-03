# oGMemory plugin for DeepSeek Harness

This plugin connects a user's DeepSeek Harness installation to an available
oGMemory HTTP service and enables automatic cross-session memory recall.

Prerequisites:

- DSH `0.1.5-rc.2` is installed with a working model provider configuration.
- The oGMemory HTTP service address is known and reachable from DSH.

Plugin `0.1.1` supports DSH CLI `0.1.5-rc.2`, core packages `0.1.5-rc.2`,
and Cordis `4.0.2`. It automatically writes completed turns to oGMemory
and recalls relevant memories across sessions.

## Automatic memory controls

`autoCapture` and `autoRecall` default to `true`. Environment variables
`OGMEM_AUTO_CAPTURE` and `OGMEM_AUTO_RECALL` override them independently.
Values `1`, `true`, `yes` enable the feature (case and surrounding whitespace
are ignored); other explicit values disable it. Restart DSH after changes.

With capture disabled, no memory queue starts: completed turns and disposal
are skipped, and existing pending writes stay on disk until capture is enabled
again. Recall and Trace remain independent. Explicit tools retain their existing
permissions, including `writeToolsEnabled`. The existing `composeEnabled`,
`writebackEnabled` and `disposeEnabled` flags still apply.

## Install

```powershell
dsh plugin --profile web add https://gitcode.com/opengauss/oGMemory.git#dev
```

## Connect DSH to oGMemory

Set the oGMemory service address and the memory identity used by DSH:

```powershell
$env:OGMEM_URL='http://127.0.0.1:8090'
$env:OGMEM_ACCOUNT_ID='acct-demo'
$env:OGMEM_USER_ID='user-dsh'
$env:OGMEM_AGENT_ID='deepseek-harness'
dsh --profile web
```

If the oGMemory HTTP API requires authentication, also set:

```powershell
$env:OGMEM_DSH_API_KEY='<oGMemory API key>'
```

`OGMEM_DSH_API_KEY` is the oGMemory HTTP API credential, not a model-provider
credential.

Retryable write failures are persisted before delivery under
`~/.ogmemory/dsh-pending`, retried in session order while DSH remains running,
and replayed at the next start if shutdown happens first. Set
`OGMEM_DSH_PENDING_DIR` to use another private runtime directory. Session
identities use separate hashed subdirectories. Session shutdown drains queued
turns before calling oGMemory `/api/v1/dispose`.

Completed turns trigger background extraction by default, including short
conversations. Keep DSH running until write-back, extraction, and indexing
finish. Once processing completes, the plugin automatically recalls relevant
memories in later sessions. Set `OGMEM_DSH_FORCE_EXTRACT=false` to use
server-side threshold batching.

## Verify the connection

Lifecycle traces are enabled by default. Completed turns upload ordered user,
assistant, tool-start, tool-outcome, and turn-end events to `/api/v1/trace/events`.
Session boundaries are captured as well. A separate durable queue in the identity
directory's `traces` subdirectory preserves event IDs across retries and restarts.
Trace content is redacted and bounded to 10,000 characters per field.

Configure `OGMEM_DSH_TRACE` and `OGMEM_DSH_TRACE_CAPTURE_CONTENT` in the launch
environment, or set `traceEnabled`, `traceCaptureContent`, and `traceTimeoutMs`
(default 3000) in the plugin config. `captureToolResults` controls memory
write-back; trace tool capture uses the trace settings.

Confirm that DSH loaded the plugin:

```powershell
dsh --profile web --dump-config
```

The output must include the `@opengauss/ogmemory-dsh-plugin` layer and the
`ogmemory-lifecycle` entry.

Then verify cross-session recall in the DSH UI:

1. Create a session and send: `Remember my test marker OGMEM-DSH-TEST-001.
   Reply only "Remembered".` Use a fresh marker for each test.
2. After the reply, keep DSH running for background extraction and indexing.
   Allow 1–2 minutes initially; processing time depends on the model service
   and indexing speed.
3. Create a new session with the same Account ID, User ID, and Agent ID. Send:
   `What test marker did I ask you to remember in the previous session?
   Reply with the marker only.`
4. Confirm that DSH returns `OGMEM-DSH-TEST-001` (or your chosen marker).
   Expand the reply's process section to see
   `Context injection ogmemory-lifecycle`. The `Trajectory` tab also shows
   the injected `<ogmemory-context>` content.

If background processing is still underway, allow more time and repeat step 3
in another new session.

Both the oGMemory context injection and the correct cross-session answer must
be present for the connection to be considered successful.

Run unit and contract tests with `npm install && npm test`. Set
`OGMEM_E2E=1` to include the real-backend marker write/recall gate.

For Chinese instructions, see [INSTALL-ZH.md](./INSTALL-ZH.md).

## Uninstall

```powershell
dsh plugin --profile web remove @opengauss/ogmemory-dsh-plugin
```
