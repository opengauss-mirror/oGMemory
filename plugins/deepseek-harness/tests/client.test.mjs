import assert from 'node:assert/strict'
import test from 'node:test'

import { OgMemoryClient } from '../lib/client.js'
import { resolveConfig } from '../lib/config.js'

test('dispose marks unavailable server cleanup as retryable', async t => {
  const originalFetch = globalThis.fetch
  globalThis.fetch = async () => Response.json({
    disposed: true,
    flushed: false,
    reason: 'thread_unavailable',
  })
  t.after(() => { globalThis.fetch = originalFetch })

  const client = new OgMemoryClient({
    credentials: { resolve: async () => undefined },
  }, resolveConfig({ disposeTimeoutMs: 1000 }))
  await assert.rejects(
    client.dispose({ sessionId: 's1' }),
    error => error.retryable === true,
  )
})

test('request diagnostics omit memory content', async t => {
  const originalFetch = globalThis.fetch
  let requestBody
  globalThis.fetch = async (_url, init) => {
    requestBody = JSON.parse(init.body)
    return Response.json({ hits: [] })
  }
  t.after(() => { globalThis.fetch = originalFetch })
  const logs = []
  const client = new OgMemoryClient({
    credentials: { resolve: async () => undefined },
    logger: { debug: line => logs.push(line) },
  }, resolveConfig({ toolTimeoutMs: 1000 }))

  await client.queryMemory({
    sessionId: 'session-safe',
    query: 'private-memory-query',
    topK: 5,
  })
  assert.match(logs[0], /path=\/api\/v1\/memories\/query/)
  assert.match(logs[0], /session=session-safe/)
  assert.equal(logs[0].includes('private-memory-query'), false)
  assert.equal(requestBody.accountId, 'acct-demo')
  assert.equal(requestBody.userId, 'user-dsh')
  assert.equal(requestBody.agentId, 'deepseek-harness')
  assert.equal(requestBody.sessionId, 'session-safe')
})

test('native tool clients use dedicated oGMemory HTTP contracts', async t => {
  const originalFetch = globalThis.fetch
  const requests = []
  globalThis.fetch = async (url, init) => {
    requests.push({ path: new URL(url).pathname, body: JSON.parse(init.body) })
    return Response.json({ ok: true })
  }
  t.after(() => { globalThis.fetch = originalFetch })
  const client = new OgMemoryClient({
    credentials: { resolve: async () => undefined },
    logger: { debug() {} },
  }, resolveConfig({ toolTimeoutMs: 1000 }))

  await client.readMemory({ sessionId: 's1', uri: 'ogmemory://memories/profile/self', level: 'all', maxChars: 1000 })
  await client.writeMemory({ sessionId: 's1', text: 'Remember blue.', clientRequestId: 'req-1' })
  await client.deleteMemory({ sessionId: 's1', uri: 'ogmemory://memories/preference/1', clientRequestId: 'req-2' })
  assert.deepEqual(requests.map(item => item.path), [
    '/api/v1/memories/read',
    '/api/v1/memories/write',
    '/api/v1/memories/delete',
  ])
  assert.equal(requests[1].body.force_extract, true)
  assert.equal(requests[2].body.uri, 'ogmemory://memories/preference/1')
})

test('after-turn forwards a stable idempotency key and waits for completion', async t => {
  const originalFetch = globalThis.fetch
  let body
  globalThis.fetch = async (_url, init) => {
    body = JSON.parse(init.body)
    return Response.json({ ok: true, status: 'completed' })
  }
  t.after(() => { globalThis.fetch = originalFetch })
  const client = new OgMemoryClient({
    credentials: { resolve: async () => undefined },
    logger: { debug() {} },
  }, resolveConfig({ afterTurnTimeoutMs: 1000 }))

  await client.afterTurn({
    sessionId: 's1',
    pendingId: 'pending-1',
    messages: [{ role: 'user', content: 'remember this' }],
    forceExtract: true,
    wait: true,
  })

  assert.equal(body.client_request_id, 'pending-1')
  assert.equal(body.wait, true)
})
