# Hermes Agent integration

This directory provides a native Hermes `MemoryProvider` for oGMemory. It uses
oGMemory's own context composition, session buffering, memory extraction, and
commit lifecycle instead of emulating OpenViking's filesystem API.

## Automatic memory controls

`OGMEM_AUTO_CAPTURE` and `OGMEM_AUTO_RECALL` default to `true`. They override
`auto_capture` and `auto_recall` in `$HERMES_HOME/ogmemory.json` independently.
Values `1`, `true`, `yes` enable the feature (case and surrounding whitespace
are ignored); other explicit values disable it. Restart Hermes after changes.

With capture disabled, completed turns and native Hermes memory write mirroring
are skipped, the background writer and recovery replay do not start, and automatic
compaction and session commits do not run. Recovery files remain untouched until
capture is enabled again. Explicit `ogmemory_remember` and `ogmemory_search`
tools remain available regardless of the automatic controls.

## Install

Start oGMemory's HTTP server, then copy the provider into the active Hermes
profile:

```bash
mkdir -p "${HERMES_HOME:-$HOME/.hermes}/plugins/ogmemory"
cp plugins/hermes/ogmemory/* "${HERMES_HOME:-$HOME/.hermes}/plugins/ogmemory/"
```

Configure `${HERMES_HOME:-$HOME/.hermes}/config.yaml`:

```yaml
memory:
  provider: ogmemory
```

Configure `${HERMES_HOME:-$HOME/.hermes}/.env`:

```dotenv
OGMEMORY_ENDPOINT=http://127.0.0.1:8090
OGMEMORY_ACCOUNT_ID=default
OGMEMORY_USER_ID=default
OGMEMORY_AGENT_ID=hermes
# OGMEMORY_API_KEY=tenant-key
```

Verify:

```bash
hermes memory status
```

The provider recalls context before each turn, sends completed turns to
`/api/v1/after_turn`, and commits the oGMemory session on Hermes session end,
session switch, compression, or shutdown. It exposes `ogmemory_search` and
`ogmemory_remember` tools.
