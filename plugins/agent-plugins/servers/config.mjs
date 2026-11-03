import { readFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';

export function loadConfig(env = process.env, home = homedir()) {
  const path = env.OGMEM_AGENT_PLUGIN_CONFIG || join(home, '.ogmemory', 'agent-plugin.json');
  let file = {};
  try { file = JSON.parse(readFileSync(path, 'utf8').replace(/^\uFEFF/, '')); }
  catch (error) {
    if (error.code !== 'ENOENT' || env.OGMEM_AGENT_PLUGIN_CONFIG) {
      throw new Error('Cannot read agent-plugin configuration; check its path and JSON syntax.');
    }
  }
  if (!file || Array.isArray(file) || typeof file !== 'object') throw new Error('Configuration must be an object.');
  const names = {
    mcp_url: 'OGMEM_MCP_URL', api_key: 'OGMEM_MCP_API_KEY',
    account_id: 'OGMEM_MCP_ACCOUNT_ID', user_id: 'OGMEM_MCP_USER_ID',
    agent_id: 'OGMEM_MCP_AGENT_ID', timeout_ms: 'OGMEM_MCP_TIMEOUT_MS',
  };
  const c = Object.fromEntries(Object.entries(names).map(([key, name]) => [key, env[name] ?? file[key]]));
  for (const key of ['mcp_url', 'api_key', 'account_id', 'user_id', 'agent_id']) {
    if (typeof c[key] !== 'string' || !c[key].trim() || /[\x00-\x1f\x7f]/.test(c[key]) || /[<>]/.test(c[key])) {
      throw new Error(`Missing or invalid ${key}; configure the service-issued connection details.`);
    }
  }
  let url;
  try { url = new URL(c.mcp_url); } catch { throw new Error('Invalid mcp_url.'); }
  if (!['https:', 'http:'].includes(url.protocol) || url.username || url.password || url.search || url.hash || /\s|\\/.test(c.mcp_url)) {
    throw new Error('mcp_url must be HTTP(S), without credentials, query or fragment.');
  }
  if (url.protocol === 'http:' && !['localhost', '127.0.0.1', '[::1]'].includes(url.hostname)) {
    throw new Error('Remote MCP endpoints require HTTPS.');
  }
  c.timeout_ms = Number(c.timeout_ms ?? 30000);
  if (!Number.isInteger(c.timeout_ms) || c.timeout_ms < 100 || c.timeout_ms > 300000) throw new Error('timeout_ms must be 100..300000.');
  return c;
}
