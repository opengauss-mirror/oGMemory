import assert from 'node:assert/strict'
import { mkdtemp, readdir, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'

import { scopedPendingDir } from '../lib/pending-queue.js'
import { collectTurnMessages, OrderedWriteQueue } from '../lib/writeback.js'

const OPTIONS = {
  captureToolResults: true,
  maxMessageChars: 1000,
  maxToolResultChars: 1000,
  maxTurnMessages: 20,
}

test('collects direct user, final assistant, and bounded tool result for one turn', () => {
  const events = [
    event('turn/start', { turn: 1 }, 0),
    event('user/message', message('user', 'question'), 1),
    event('user/message', {
      ...message('plugin', 'recalled secret'),
      source: { kind: 'plugin', plugin: 'ogmemory' },
    }, 2),
    event('tool/call', { turn: 1, step: 1, callId: 'call-1', name: 'read_file', arguments: '{"path":"a"}' }, 3),
    event('tool/result', {
      turn: 1,
      step: 1,
      message: {
        source: { kind: 'tool', callId: 'call-1' },
        content: [{ type: 'tool-result', content: [{ type: 'text', text: 'file body' }] }],
      },
    }, 4),
    event('assistant/message', {
      turn: 1,
      step: 1,
      message: message('model', 'answer'),
    }, 5),
    event('assistant/message', {
      turn: 1,
      step: 1,
      message: {
        source: { kind: 'plugin', plugin: 'other-plugin' },
        content: [{ type: 'text', text: 'synthetic assistant context' }],
      },
    }, 6),
    event('turn/end', { turn: 1, reason: { kind: 'completed' } }, 7),
  ]
  const messages = collectTurnMessages(events, 1, OPTIONS)
  assert.deepEqual(messages.map(item => item.role), ['user', 'tool', 'assistant'])
  assert.equal(messages[0].content, 'question')
  assert.match(messages[1].content, /tool=read_file/)
  assert.match(messages[1].content, /result=file body/)
  assert.equal(messages[2].content, 'answer')
  assert.equal(messages.some(item => item.content.includes('recalled secret')), false)
  assert.equal(messages.some(item => item.content.includes('synthetic assistant context')), false)
})

test('write queue preserves session order and retries retryable failures', async t => {
  const pendingDir = await temporaryPendingDir(t)
  const calls = []
  let failures = 1
  const logs = []
  const queue = new OrderedWriteQueue({
    logger: logger(logs),
    maxAttempts: 3,
    initialDelayMs: 1,
    maxDelayMs: 2,
    pendingDir,
    send: async payload => {
      calls.push(payload.turn)
      if (payload.turn === 1 && failures-- > 0) {
        const error = new Error('temporary')
        error.retryable = true
        throw error
      }
    },
  })
  queue.enqueue('s1', { turn: 1 })
  queue.enqueue('s1', { turn: 2 })
  await queue.flush('s1')
  assert.deepEqual(calls, [1, 1, 2])
  assert.equal(logs.some(line => line.includes('writeback retry')), true)
  await queue.dispose()
})

test('write queue durably replays retryable failures in session order', async t => {
  const pendingDir = await temporaryPendingDir(t)
  const unavailable = new OrderedWriteQueue({
    logger: logger([]),
    maxAttempts: 1,
    initialDelayMs: 1,
    maxDelayMs: 1,
    pendingDir,
    send: async () => {
      const error = new Error('offline')
      error.retryable = true
      throw error
    },
  })
  await unavailable.enqueue('s1', { turn: 1 })
  await unavailable.enqueue('s1', { turn: 2 })
  assert.equal((await readdir(pendingDir)).filter(name => name.endsWith('.json')).length, 2)
  await unavailable.dispose()

  const calls = []
  const recovered = new OrderedWriteQueue({
    logger: logger([]),
    maxAttempts: 1,
    initialDelayMs: 1,
    maxDelayMs: 1,
    pendingDir,
    send: async payload => { calls.push(payload.turn) },
  })
  await recovered.start()
  assert.deepEqual(calls, [1, 2])
  assert.equal((await readdir(pendingDir)).filter(name => name.endsWith('.json')).length, 0)
  await recovered.dispose()
})

test('replay keeps the pending record id after a post-send crash window', async t => {
  const pendingDir = await temporaryPendingDir(t)
  const requestIds = []
  const first = new OrderedWriteQueue({
    logger: logger([]),
    maxAttempts: 1,
    initialDelayMs: 1,
    maxDelayMs: 1,
    pendingDir,
    send: async payload => { requestIds.push(payload.pendingId) },
  })
  first.store.remove = async () => {}
  await first.enqueue('s1', { kind: 'afterTurn', turn: 1 })
  await first.dispose()

  const replay = new OrderedWriteQueue({
    logger: logger([]),
    maxAttempts: 1,
    initialDelayMs: 1,
    maxDelayMs: 1,
    pendingDir,
    send: async payload => { requestIds.push(payload.pendingId) },
  })
  await replay.start()

  assert.equal(requestIds.length, 2)
  assert.match(requestIds[0], /^[0-9a-f-]{36}$/)
  assert.equal(requestIds[1], requestIds[0])
  await replay.dispose()
})

test('blocked session retries at runtime and preserves pending order', async t => {
  const pendingDir = await temporaryPendingDir(t)
  const calls = []
  let available = false
  let resolveRecovered
  const recovered = new Promise(resolve => { resolveRecovered = resolve })
  const queue = new OrderedWriteQueue({
    logger: logger([]),
    maxAttempts: 1,
    initialDelayMs: 1,
    maxDelayMs: 2,
    pendingDir,
    send: async payload => {
      calls.push(payload.turn)
      if (!available) {
        const error = new Error('offline')
        error.retryable = true
        throw error
      }
      if (payload.turn === 2) resolveRecovered()
    },
  })
  t.after(() => queue.dispose())

  await queue.enqueue('s1', { turn: 1 })
  available = true
  await queue.enqueue('s1', { turn: 2 })
  await Promise.race([
    recovered,
    new Promise((_, reject) => setTimeout(() => reject(new Error('runtime retry timed out')), 200)),
  ])
  await queue.flush('s1')

  assert.deepEqual(calls, [1, 1, 2])
  assert.equal((await readdir(pendingDir)).filter(name => name.endsWith('.json')).length, 0)
})

test('bounded shutdown leaves an in-flight write durable', async t => {
  const pendingDir = await temporaryPendingDir(t)
  const queue = new OrderedWriteQueue({
    logger: logger([]),
    maxAttempts: 1,
    initialDelayMs: 1,
    maxDelayMs: 1,
    pendingDir,
    shutdownTimeoutMs: 20,
    send: () => new Promise(() => {}),
  })
  void queue.enqueue('s1', { turn: 1 })
  await queue.dispose()
  assert.equal((await readdir(pendingDir)).filter(name => name.endsWith('.json')).length, 1)
})

test('write queue keeps retrying without sending while durable persistence fails', async () => {
  let sends = 0
  let signalAttempt
  const attempted = new Promise(resolve => { signalAttempt = resolve })
  const queue = new OrderedWriteQueue({
    logger: logger([]),
    maxAttempts: 1,
    initialDelayMs: 1,
    maxDelayMs: 1,
    send: async () => { sends += 1 },
  })
  queue.store.put = async () => {
    signalAttempt()
    throw new Error('disk full')
  }

  let settled = false
  const pending = queue.enqueue('s1', { turn: 1 }).then(() => { settled = true })
  await attempted

  assert.equal(sends, 0)
  assert.equal(settled, false)
  await queue.dispose()
  await pending
})

test('write queue resumes after a transient durable persistence failure', async t => {
  const pendingDir = await temporaryPendingDir(t)
  const calls = []
  let attempts = 0
  const queue = new OrderedWriteQueue({
    pendingDir,
    logger: logger([]),
    maxAttempts: 1,
    initialDelayMs: 1,
    maxDelayMs: 1,
    send: async payload => { calls.push(payload.turn) },
  })
  const put = queue.store.put.bind(queue.store)
  queue.store.put = async (sessionId, payload) => {
    attempts += 1
    if (attempts === 1) throw new Error('temporary disk error')
    return put(sessionId, payload)
  }

  await queue.enqueue('s1', { turn: 1 })

  assert.equal(attempts, 2)
  assert.deepEqual(calls, [1])
  await queue.dispose()
})

function event(type, data, seq) {
  return { type, data, seq, time: 1700000000000 + seq }
}

function message(kind, text) {
  return {
    source: kind === 'user' ? { kind: 'user' } : { kind: 'model', provider: 'p', model: 'm' },
    content: [{ type: 'text', text }],
  }
}

function logger(lines) {
  return {
    debug: value => lines.push(value),
    warn: value => lines.push(value),
    error: value => lines.push(value),
  }
}

test('pending directories isolate memory identities without exposing identifiers', () => {
  const root = join(tmpdir(), 'pending')
  const aliceIdentity = { accountId: 'acct', userId: 'alice', agentId: 'dsh' }
  const alice = scopedPendingDir(root, {
    baseUrl: 'https://memory.example.com', ...aliceIdentity,
  })
  const bob = scopedPendingDir(root, {
    baseUrl: 'https://memory.example.com', accountId: 'acct', userId: 'bob', agentId: 'dsh',
  })
  const staging = scopedPendingDir(root, {
    baseUrl: 'https://staging-memory.example.com', ...aliceIdentity,
  })
  assert.notEqual(alice, bob)
  assert.notEqual(alice, staging)
  assert.equal(alice.includes('alice'), false)
})

async function temporaryPendingDir(t) {
  const directory = await mkdtemp(join(tmpdir(), 'ogmemory-dsh-queue-'))
  t.after(() => rm(directory, { recursive: true, force: true }))
  return directory
}

test('runtime replay and queued callbacks deliver each durable operation once', async t => {
  const pendingDir = await temporaryPendingDir(t)
  const calls = []
  const queue = new OrderedWriteQueue({
    logger: logger([]), maxAttempts: 1, initialDelayMs: 1, maxDelayMs: 1,
    pendingDir, send: async payload => { calls.push(payload.kind) },
  })
  t.after(() => queue.dispose())
  await queue.start()
  await queue.store.put('s1', { kind: 'old' })
  queue.blockedSessions.add('s1')
  let resumeList, signalList
  const paused = new Promise(resolve => { resumeList = resolve })
  const listing = new Promise(resolve => { signalList = resolve })
  const list = queue.store.list.bind(queue.store)
  queue.store.list = async () => { signalList(); await paused; return list() }
  queue.scheduleReplay('s1')
  await Promise.race([listing, new Promise((_, reject) => setTimeout(() => reject(new Error('replay timeout')), 1000))])
  const turn = queue.enqueue('s1', { kind: 'afterTurn' })
  const trace = queue.enqueue('s1', { kind: 'trace', envelopes: [{ event: { event_type: 'turn_end' } }] })
  const dispose = queue.enqueue('s1', { kind: 'dispose' })
  await queue.persistence
  resumeList()
  await Promise.all([turn, trace, dispose])
  assert.deepEqual(calls, ['old', 'afterTurn', 'trace', 'dispose'])
  assert.deepEqual(await readdir(pendingDir), [])
})

test('pending read recovery preserves queued write order', async t => {
  const pendingDir = await temporaryPendingDir(t)
  const calls = []
  let recovered
  const completed = new Promise(resolve => { recovered = resolve })
  const queue = new OrderedWriteQueue({
    logger: logger([]), maxAttempts: 1, initialDelayMs: 1, maxDelayMs: 2,
    pendingDir, send: async payload => { calls.push(payload.turn); if (payload.turn === 2) recovered() },
  })
  t.after(() => queue.dispose())
  await queue.start()
  const read = queue.store.read.bind(queue.store)
  let failed = false
  queue.store.read = async path => {
    if (!failed) { failed = true; throw new Error('temporary read failure') }
    return read(path)
  }
  await queue.enqueue('s1', { turn: 1 })
  await queue.enqueue('s1', { turn: 2 })
  await Promise.race([completed, new Promise((_, reject) => setTimeout(() => reject(new Error('recovery timeout')), 1000))])
  await queue.flush('s1')
  assert.deepEqual(calls, [1, 2])
  assert.deepEqual(await readdir(pendingDir), [])
})
