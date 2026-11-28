import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import test from 'node:test'

// Against a staged package this resolves the real SDKs installed for that target.
const packageDir = resolve(process.env.OGMEM_TEST_PACKAGE_DIR ?? join(import.meta.dirname, '..'))
const requireTarget = createRequire(join(packageDir, 'package.json'))
const sdk = name => import(pathToFileURL(requireTarget.resolve(name)).href)
const { createToolResultMessage, createUserMessage } = await sdk('@deepseek-ai/dsh-llm')
const { Session } = await sdk('@deepseek-ai/dsh-session')
const { apply, name } = await import(pathToFileURL(join(packageDir, 'index.js')).href)

test('records actual DSH tool outcomes in trace and writeback and preserves recall decision flags', async t => {
  const pendingDir = await mkdtemp(join(tmpdir(), 'ogmemory-adapter-'))
  t.after(() => rm(pendingDir, { recursive: true, force: true }))
  const session = Session.create('adapter-session')
  const listeners = new Map()
  const tools = new Map()
  const requests = []
  t.mock.method(globalThis, 'fetch', async (url, init) => {
    requests.push({ path: new URL(url).pathname, body: JSON.parse(init.body) })
    return Response.json({ ok: true, identityContext: 'adapter test memory' })
  })
  apply({
    get: () => undefined,
    logger: { debug() {}, warn() {}, error() {} },
    tools: { register: tool => tools.set(tool.name, tool) },
    on: (name, handler) => listeners.set(name, handler),
    effect() {},
  }, { pendingDir, captureToolResults: true })
  t.after(() => listeners.get('session/disposed')(session))
  const agent = { session }
  const question = createUserMessage({ source: { kind: 'user' }, content: [{ type: 'text', text: 'query' }] })
  const decision = { kind: 'enter', messages: [question], startsRequestSeries: true }
  const recalled = await listeners.get('agent/pre-step')(
    { agent, step: 1, signal: new AbortController().signal }, async () => decision,
  )
  const newSdk = createToolResultMessage({ callId: 'probe', content: [], isError: false }).role === 'tool'
  const memory = recalled.messages.at(-1)
  assert.deepEqual(memory.source, {
    ...(newSdk ? { kind: name } : { kind: 'plugin', plugin: name }),
    form: 'snapshot',
    sections: [{ name: 'ogmemory', text: memory.content[0].text }],
  })
  assert.equal(recalled.startsRequestSeries, true)
  assert.match(memory.content[0].text, /adapter test memory/)
  assert.ok(tools.has('ogm_query_memory'))

  session.append('turn/start', { turn: 1 })
  session.append('user/message', question, { surfaceOp: 'append' })
  const appendResult = (id, content, isError) => session.append('tool/result', {
    turn: 1, step: 1,
    message: createToolResultMessage({ callId: id, content: [{ type: 'text', text: content }], isError }),
  }, { surfaceOp: 'append' })
  for (const id of ['a', 'b']) {
    session.append('tool/call', { turn: 1, step: 1, callId: id, name: 'example', arguments: '{}' })
  }
  appendResult('b', 'failure b', true)
  appendResult('a', 'success a', false)
  const end = session.append('turn/end', { turn: 1, reason: { kind: 'completed' } })
  listeners.get('session/event')(session, end)
  await listeners.get('session/flush')(session)
  const trace = requests.filter(r => r.path.endsWith('/trace/events')).map(r => r.body.event)
  assert.deepEqual(trace.filter(e => e.event_type.startsWith('tool_')).map(e =>
    [e.event_type, e.tool_use_id, e.error ?? e.tool_response]), [
    ['tool_start', 'a', undefined],
    ['tool_start', 'b', undefined],
    ['tool_failure', 'b', 'failure b'],
    ['tool_success', 'a', 'success a'],
  ])
  const write = requests.find(r => r.path.endsWith('/after_turn'))
  assert.deepEqual(write.body.messages.map(m => [m.role, m.content]), [
    ['user', 'query'], ['tool', 'tool=example\narguments={}\nresult=failure b'],
    ['tool', 'tool=example\narguments={}\nresult=success a'],
  ])
})
