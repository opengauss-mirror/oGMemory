# oGMemory Agent Plugin

Portable Agent Plugins 1.0 package for model-driven memory recall and persistence.
Requires Node.js 22+ and a separately deployed oGMemory Streamable HTTP MCP server.
There are no npm dependencies and no lifecycle hooks.

## 1. Configure

Copy `config.example.json` to `~/.ogmemory/agent-plugin.json` and replace all
placeholders with administrator-issued connection details. Alternatively set
`OGMEM_AGENT_PLUGIN_CONFIG` to the absolute path of your private JSON file.
Do not put credentials in this package. The full MCP URL is required; REST and
MCP may use different ports. Remote endpoints require HTTPS; loopback HTTP is
supported for local development. HTTP MCP requires server-side authentication.

Environment overrides (including empty values) take priority over JSON fields:

| Environment | JSON field |
| --- | --- |
| `OGMEM_MCP_URL` | `mcp_url` |
| `OGMEM_MCP_API_KEY` | `api_key` |
| `OGMEM_MCP_ACCOUNT_ID` | `account_id` |
| `OGMEM_MCP_USER_ID` | `user_id` |
| `OGMEM_MCP_AGENT_ID` | `agent_id` |
| `OGMEM_MCP_TIMEOUT_MS` | `timeout_ms` |

Restart the client connection after changing configuration. Default timeout is
30 seconds; use 120000 for slower synchronous extraction and increase the host's
tool timeout accordingly (maximum proxy timeout: 300000).

## 2. Connect

A conforming Agent Plugins client can load this directory's `plugin.json` and
`mcp.json`, then discover `skills/ogmemory-memory/SKILL.md`. Installation commands
depend on the host. Do not assume a client's proprietary plugin command loads
the portable format.

For a generic MCP client, register this stdio command instead:

```json
{
  "mcpServers": {
    "ogmemory": {
      "command": "node",
      "args": ["/absolute/path/plugins/agent-plugins/servers/mcp-proxy.mjs"],
      "env": {"OGMEM_AGENT_PLUGIN_CONFIG": "/absolute/path/private/agent-plugin.json"}
    }
  }
}
```

Load `skills/ogmemory-memory/SKILL.md` through the client's Skill mechanism, or
add its instructions to the host's system/project instructions. The supported
surface is exactly `ogm_query_memory`, `ogm_read_memory`, `ogm_write_memory`.
The proxy filters discovery and rejects other tool calls, resource reads and
prompts. The write tool appears only when the upstream service enables it.
Reading is limited to memory item URIs; this is not a general resource browser.
Server authorization remains authoritative; possession of this plugin grants no
extra permissions. Do not configure a second direct upstream connection if you
intend to keep the same limited tool surface.

## 3. Verify

```bash
node /absolute/path/plugins/agent-plugins/scripts/doctor.mjs
```

Exit 0: all three tools found. Exit 2: connected but missing tools (for example a
read-only service). Exit 1: configuration, connection or protocol error. Doctor
never reads or writes memories and does not prove model behavior.

Ask the Agent to remember a unique test preference, verify the write result,
start a new conversation, query it and read the exact returned URI. Preserve the
same account/user/authorized agent; use a different business `session_id`.
`accepted` is not `completed`. On write timeout, the outcome is unknown: inspect
memory before retrying. The proxy never automatically replays writes.

## Client guides and development

See [Chinese guide](README_zh.md) and the repository's
[four-client tutorial](../../docs/integrations/agent-plugins.md). Release ZIPs include
that tutorial as `CLIENT-GUIDE.md`.

```bash
node --test plugins/agent-plugins/tests/proxy.test.mjs
python -m pytest tests/integration/test_agent_plugin_protocol.py
python plugins/agent-plugins/scripts/package.py --output dist
```

The automated protocol fixture uses production MCP handlers with deterministic
service doubles. For a live TEST deployment, set `OGMEM_AGENT_PLUGIN_CONFIG` and
run `python tests/e2e/run_agent_plugin_smoke.py`; this writes a unique test fact.
The ZIP uses an explicit allowlist and excludes private configuration and logs.

Requests are serialized per proxy process and bounded to 1 MiB. There is no
background upstream event subscription, automatic reconnect, task execution,
sampling or elicitation. Cancellation cannot undo an accepted write. Reconnect
after transport-session expiry. Diagnostics go to stderr; stdout is MCP only.
