# MCP protocol compatibility

oGMemory uses MCP Python SDK 2.x and supports both MCP protocol eras.

| Protocol revision | Era | stdio | Streamable HTTP |
| --- | --- | --- | --- |
| `2024-11-05` | Legacy handshake | Supported | Use the legacy HTTP+SSE endpoint |
| `2025-03-26` | Legacy handshake | Supported | Supported |
| `2025-06-18` | Legacy handshake | Supported | Supported |
| `2025-11-25` | Legacy handshake | Supported | Supported |
| `2026-07-28` | Per-request metadata | Supported | Supported |

The SDK selects the protocol behavior automatically. Modern requests carry
`2026-07-28` metadata on each request. Older clients continue to use the
`initialize` handshake and protocol sessions. oGMemory also keeps its separate
SSE transport for clients that still use the deprecated `2024-11-05` HTTP+SSE
binding.

Legacy Streamable HTTP sessions are stored in the serving process. Deployments
with multiple workers must use sticky routing for legacy clients. The
`2026-07-28` protocol is stateless and does not require stickiness.

Run the compatibility tests with:

```bash
pytest tests/unit/server/test_mcp_server.py tests/unit/server/test_mcp_security.py -q
```

See the upstream [versioning and compatibility specification](https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning)
and the SDK guide for [serving legacy clients](https://py.sdk.modelcontextprotocol.io/run/legacy-clients/).
