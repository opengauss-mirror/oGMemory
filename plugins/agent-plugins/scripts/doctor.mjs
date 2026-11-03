import { MemoryBridge, TOOL_NAMES } from '../servers/mcp-proxy.mjs';
import { loadConfig } from '../servers/config.mjs';

let bridge;
try {
  bridge = new MemoryBridge(loadConfig());
  await bridge.handle({method: 'initialize', params: {protocolVersion: '2025-03-26', capabilities: {}, clientInfo: {name: 'ogmemory-doctor', version: '0.1.0'}}});
  await bridge.handle({method: 'notifications/initialized'});
  const {tools} = await bridge.handle({method: 'tools/list'});
  const names = tools.map(t => t.name);
  const missing = TOOL_NAMES.filter(name => !names.includes(name));
  console.log(JSON.stringify({connected: true, tools: names, missing, writable: names.includes('ogm_write_memory'), note: 'No memory was read or written; this does not verify model behavior.'}, null, 2));
  if (missing.length) process.exitCode = 2;
} catch (error) {
  console.error(`oGMemory doctor: ${error.message}`);
  process.exitCode = 1;
} finally { await bridge?.close(); }
