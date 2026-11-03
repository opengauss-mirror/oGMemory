import assert from 'node:assert/strict'
import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test, { beforeEach } from 'node:test'
import { createAssistantMessage, createUserMessage } from '@deepseek-ai/dsh-llm'
import { Session } from '@deepseek-ai/dsh-session'

import { apply } from '../index.js'
import { PendingStore, scopedPendingDir } from '../lib/pending-queue.js'
import { resolveConfig } from '../lib/config.js'
import { isolateAutoMemoryEnv } from './auto-memory-env.mjs'

beforeEach(isolateAutoMemoryEnv)

for (const autoCapture of [true, false]) {
  for (const autoRecall of [true, false]) {
    test(`automatic controls capture=${autoCapture} recall=${autoRecall}`, async t => {
      const pendingDir = await mkdtemp(join(tmpdir(), 'ogmemory-dsh-controls-'))
      t.after(() => rm(pendingDir, { recursive: true, force: true }))
      const config = resolveConfig({ pendingDir, autoCapture, autoRecall })
      const store = new PendingStore(scopedPendingDir(pendingDir, config))
      await store.put('old-session', { kind: 'afterTurn', sessionId: 'old-session', messages: [{ role: 'user', content: 'old' }] })
      const requests = []
      const originalFetch = globalThis.fetch
      globalThis.fetch = async (url, init) => {
        requests.push({ url: String(url), body: JSON.parse(init.body) })
        return Response.json({ ok: true, identityContext: 'memory' })
      }
      t.after(() => { globalThis.fetch = originalFetch })
      const harness = fakeContext()
      apply(harness.ctx, config)
      const events = [
        { type: 'turn/start', data: { turn: 1 } },
        { type: 'user/message', data: { source: { kind: 'user' }, content: [{ type: 'text', text: 'hello' }] } },
        { type: 'assistant/message', data: { turn: 1, message: {
          source: { kind: 'model' }, content: [{ type: 'text', text: 'answer' }],
        } } },
        { type: 'turn/end', data: { turn: 1, reason: { kind: 'completed' } } },
      ].map((event, seq) => ({ ...event, seq, time: 1750000000000 + seq }))
      const session = { id: 'controls-session', snapshotEvents: () => events }
      harness.listener('agent/session-start')({ agent: { session } })
      const decision = { kind: 'enter', messages: [{ source: { kind: 'user' }, content: [{ type: 'text', text: 'question' }] }] }
      const result = await harness.listener('agent/pre-step')(
        { agent: { session }, step: 1, signal: new AbortController().signal }, async () => decision,
      )
      assert.equal(result.messages.length, autoRecall ? 2 : 1)
      harness.listener('session/event')(session, events.at(-1))
      await harness.listener('session/flush')(session)
      await harness.listener('session/disposed')(session)
      await harness.dispose()
      assert.equal(requests.filter(item => item.url.endsWith('/compose')).length, autoRecall ? 1 : 0)
      const writes = requests.filter(item => item.url.endsWith('/after_turn'))
      const replayed = writes.filter(item => item.body.sessionId === 'old-session')
      const current = writes.filter(item => item.body.sessionId === session.id)
      assert.equal(writes.length, autoCapture ? 2 : 0)
      assert.equal(replayed.length, autoCapture ? 1 : 0)
      assert.equal(current.length, autoCapture ? 1 : 0)
      if (autoCapture) {
        assert.deepEqual(replayed[0].body.messages, [{ role: 'user', content: 'old' }])
        assert.deepEqual(current[0].body.messages.map(({ role, content }) => [role, content]), [
          ['user', 'hello'], ['assistant', 'answer'],
        ])
      }
      const disposals = requests.filter(item => item.url.endsWith('/dispose'))
      assert.equal(disposals.length, autoCapture ? 1 : 0)
      if (autoCapture) assert.equal(disposals[0].body.sessionId, session.id)
      const traces = requests.filter(item => item.url.endsWith('/trace/events'))
      assert.equal(traces.length, 5)
      assert.ok(traces.every(item => item.body.session_id === session.id))
      assert.deepEqual(traces.map(item => item.body.event.event_type), [
        'session_start', 'user_message', 'assistant_message', 'turn_end', 'session_end',
      ])
      assert.equal(traces[1].body.event.prompt, 'hello')
      assert.equal(traces[2].body.event.assistant_message, 'answer')
      assert.equal(traces[3].body.event.assistant_message, 'answer')
      assert.ok(traces.slice(1, 4).every(item => item.body.event.turn_id === '1'))
      assert.equal((await store.list()).length, autoCapture ? 0 : 1)
    })
  }
}

test('integrates compose context and after-turn writeback through lifecycle events', async t => {
  const pendingDir = await mkdtemp(join(tmpdir(), 'ogmemory-dsh-plugin-'))
  t.after(() => rm(pendingDir, { recursive: true, force: true }))
  const requests = []
  const originalFetch = globalThis.fetch
  globalThis.fetch = async (url, init) => {
    const body = JSON.parse(init.body)
    requests.push({ url: String(url), init, body })
    if (String(url).endsWith('/compose')) {
      return Response.json({
        identityContext: 'profile-memory',
        retrievedEvidence: 'retrieved-memory',
      })
    }
    return Response.json({ ok: true, status: 'accumulating' })
  }
  t.after(() => { globalThis.fetch = originalFetch })

  const harness = fakeContext()
  apply(harness.ctx, {
    baseUrl: 'http://127.0.0.1:8090',
    accountId: 'acct-test',
    userId: 'user-test',
    agentId: 'agent-test',
    pendingDir,
    composeTimeoutMs: 1000,
    afterTurnTimeoutMs: 1000,
    writebackInitialDelayMs: 1,
    writebackMaxDelayMs: 2,
  })

  const session = Session.create('session-1')
  const agent = { session }
  harness.agents.set(session.id, agent)
  harness.listener('agent/session-start')({ agent })
  const decision = {
    kind: 'enter',
    messages: [{
      source: { kind: 'user' },
      content: [{ type: 'text', text: 'what did we decide?' }],
    }],
  }
  const recalledDecision = await harness.listener('agent/pre-step')(
    { agent, step: 1, signal: new AbortController().signal },
    async () => decision,
  )
  const context = recalledDecision.messages.at(-1).content[0].text
  assert.match(context, /profile-memory/)
  assert.match(context, /retrieved-memory/)
  assert.equal(recalledDecision.messages.at(-1).source.kind, 'plugin')
  assert.equal(recalledDecision.messages.at(-1).source.plugin, 'ogmemory-lifecycle')
  assert.equal(recalledDecision.messages.at(-1).role, 'user')
  assert.match(recalledDecision.messages.at(-1).id, /^[0-9a-f-]{36}$/)
  assert.equal(Object.isFrozen(recalledDecision.messages.at(-1)), true)

  session.append('turn/start', { turn: 1 })
  session.append('user/message', createUserMessage(decision.messages[0]), { surfaceOp: 'append' })
  session.append('assistant/message', {
    turn: 1,
    step: 1,
    message: createAssistantMessage({
      source: { kind: 'model', provider: 'p', model: 'm' },
      content: [{ type: 'text', text: 'the decision' }],
    }),
    stream: [],
  }, { surfaceOp: 'append' })
  const end = session.append('turn/end', { turn: 1, reason: { kind: 'completed' } })
  harness.listener('session/event')(session, end)
  await harness.listener('session/flush')(session)

  const writeRequests = requests.filter(item => item.url.endsWith('/after_turn'))
  const traceRequests = requests.filter(item => item.url.endsWith('/trace/events'))
  assert.equal(requests.length, 6)
  const compose = requests.find(item => item.url.endsWith('/compose'))
  assert.equal(compose.body.sessionId, 'session-1')
  assert.equal(compose.init.headers['X-API-Key'], 'test-key')
  assert.deepEqual(
    writeRequests[0].body.messages.map(message => [message.role, message.content]),
    [['user', 'what did we decide?'], ['assistant', 'the decision']],
  )
  assert.equal(writeRequests[0].body.forceExtract, true)
  assert.match(writeRequests[0].body.client_request_id, /^[0-9a-f-]{36}$/)
  assert.equal(writeRequests[0].body.wait, true)
  assert.deepEqual(traceRequests.map(item => item.body.event.event_type), [
    'session_start', 'user_message', 'assistant_message', 'turn_end',
  ])
  assert.equal(traceRequests[1].body.event.prompt, 'what did we decide?')
  assert.equal(traceRequests[3].body.event.assistant_message, 'the decision')
  assert.equal(traceRequests[0].body.account_id, 'acct-test')
  assert.equal(traceRequests[0].body.session_id, 'session-1')
  await harness.listener('session/disposed')(session)
  assert.equal(requests.find(item => item.url.endsWith('/dispose')).body.sessionId, 'session-1')
  assert.equal(requests.filter(item => item.url.endsWith('/trace/events')).at(-1).body.event.event_type, 'session_end')
  await harness.dispose()
})

function fakeContext() {
  const listeners = new Map()
  const effects = []
  const agents = new Map()
  const tools = new Map()
  const ctx = {
    agents: { get: id => agents.get(id) },
    credentials: { resolve: async ref => ref === 'OGMEM_DSH_API_KEY' ? { value: 'test-key', source: 'test' } : undefined },
    sessions: {},
    tools: { register: tool => tools.set(tool.name, tool) },
    logger: { debug() {}, warn() {}, error() {} },
    on(name, listener) {
      listeners.set(name, listener)
      return () => listeners.delete(name)
    },
    effect(register) {
      effects.push(register())
    },
  }
  return {
    ctx,
    agents,
    tools,
    listener: name => listeners.get(name),
    async dispose() {
      await Promise.all(effects.map(dispose => dispose()))
    },
  }
}


test('replays a complete durable trace turn after process exit during a blocked request', { timeout: 10000 }, async t => {
  const pendingDir = await mkdtemp(join(tmpdir(), 'ogmemory-dsh-restart-'))
  t.after(() => rm(pendingDir, { recursive: true, force: true }))
  const config = { pendingDir, writebackEnabled: false, disposeEnabled: false }
  const events = [
    { type: 'turn/start', data: { turn: 1 } },
    { type: 'user/message', data: { source: { kind: 'user' }, content: [{ type: 'text', text: 'question' }] } },
    { type: 'assistant/message', data: { turn: 1, message: { source: { kind: 'model' }, content: [{ type: 'text', text: 'answer' }] } } },
    { type: 'turn/end', data: { turn: 1, reason: { kind: 'completed' } } },
  ].map((event, seq) => ({ ...event, seq, time: 1750000000000 + seq }))
  const child = `
    import { apply, resolveConfig } from ${JSON.stringify(new URL('../index.js', import.meta.url).href)};
    import { PendingStore, scopedPendingDir } from ${JSON.stringify(new URL('../lib/pending-queue.js', import.meta.url).href)};
    import { join } from 'node:path';
    ${fakeContext.toString()}
    globalThis.fetch = () => new Promise(() => {});
    const config = resolveConfig(${JSON.stringify(config)});
    const harness = fakeContext();
    apply(harness.ctx, config);
    const events = ${JSON.stringify(events)};
    const session = { id: 'restart-session', snapshotEvents: () => events };
    harness.listener('agent/session-start')({ agent: { session } });
    harness.listener('session/event')(session, events.at(-1));
    const store = new PendingStore(join(scopedPendingDir(config.pendingDir, config), 'traces'));
    for (let attempt = 0; attempt < 200; attempt++) {
      const entries = await store.list();
      if (entries.length === 2 && entries[1].payload.envelopes?.length === 3) process.exit(0);
      await new Promise(resolve => setTimeout(resolve, 10));
    }
    throw new Error('complete trace turn was not persisted');
  `
  await promisify(execFile)(process.execPath, ['--input-type=module', '--eval', child], { timeout: 5000 })
  const requests = []
  let finish
  const replayed = new Promise(resolve => { finish = resolve })
  const originalFetch = globalThis.fetch
  globalThis.fetch = async (_url, init) => {
    requests.push(JSON.parse(init.body))
    if (requests.length === 4) finish()
    return Response.json({ ok: true })
  }
  t.after(() => { globalThis.fetch = originalFetch })
  const harness = fakeContext()
  apply(harness.ctx, config)
  await replayed
  await harness.dispose()
  assert.deepEqual(requests.map(item => item.event.event_type), [
    'session_start', 'user_message', 'assistant_message', 'turn_end',
  ])
  assert.equal(requests[1].event.prompt, 'question')
  assert.equal(requests[3].event.assistant_message, 'answer')
})
