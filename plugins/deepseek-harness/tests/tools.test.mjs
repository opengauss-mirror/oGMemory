import assert from 'node:assert/strict'
import test from 'node:test'

import { resolveConfig } from '../lib/config.js'
import { registerOgMemoryTools } from '../lib/tools.js'

test('registers native read tools by default and keeps mutation tools opt-in', () => {
  const tools = register(resolveConfig())
  assert.deepEqual([...tools.keys()], ['ogm_query_memory', 'ogm_read_memory'])
  assert.deepEqual(tools.get('ogm_query_memory').parameters.required, ['query'])
  assert.equal('session_id' in tools.get('ogm_query_memory').parameters.properties, false)
  assert.equal('agent_id' in tools.get('ogm_query_memory').parameters.properties, false)
})

test('allows native tool registration to be disabled for MCP-backed profiles', () => {
  assert.equal(register(resolveConfig({ toolsEnabled: false })).size, 0)
})

test('native tools use the calling session and configured client identity', async () => {
  const calls = []
  const client = {
    queryMemory: async input => {
      calls.push(input)
      return { hits: [{ uri: 'ogmemory://memories/preference/1' }] }
    },
  }
  const tools = register(resolveConfig(), client)
  const output = await tools.get('ogm_query_memory').execute({
    query: 'editor preference',
    top_k: 100,
  }, execution('session-7'))
  assert.deepEqual(calls, [{
    sessionId: 'session-7',
    query: 'editor preference',
    types: undefined,
    topK: 20,
    scoreThreshold: undefined,
  }])
  assert.match(output, /ogmemory:\/\/memories\/preference\/1/)
})

test('write and delete tools require explicit plugin opt-in', async () => {
  const calls = []
  const client = {
    writeMemory: async input => { calls.push(['write', input]); return { ok: true } },
    deleteMemory: async input => { calls.push(['delete', input]); return { deleted: true } },
  }
  const tools = register(resolveConfig({
    writeToolsEnabled: true,
    deleteToolsEnabled: true,
  }), client)
  assert.deepEqual([...tools.keys()], [
    'ogm_query_memory',
    'ogm_read_memory',
    'ogm_write_memory',
    'ogm_delete_memory',
  ])
  await tools.get('ogm_write_memory').execute({ text: 'Remember blue.' }, execution('s1'))
  await tools.get('ogm_delete_memory').execute({
    uri: 'ogmemory://memories/preference/1',
  }, execution('s1'))
  assert.equal(calls[0][1].sessionId, 's1')
  assert.equal(calls[0][1].text, 'Remember blue.')
  assert.match(calls[0][1].clientRequestId, /^[0-9a-f-]{36}$/)
  assert.equal(calls[1][1].uri, 'ogmemory://memories/preference/1')
})

function register(config, client = {}) {
  const tools = new Map()
  registerOgMemoryTools({ tools: { register: tool => tools.set(tool.name, tool) } }, client, config)
  return tools
}

function execution(sessionId) {
  return { agent: { session: { id: sessionId } } }
}
