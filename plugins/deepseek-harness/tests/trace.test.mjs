import assert from 'node:assert/strict'
import test from 'node:test'
import { readFileSync } from 'node:fs'
import { collectTurnTraces, traceEnvelope } from '../lib/trace.js'
import { resolveConfig } from '../lib/config.js'

const config = resolveConfig()
const message = (kind, value) => ({ source: { kind }, content: [{ type: 'text', text: value }] })
const event = (type, data, seq) => ({ type, data, seq, time: 1750000000000 + seq })

function turnEvents() {
  return [
    event('turn/start', { turn: 1 }, 0),
    event('user/message', message('user', 'Inspect the configuration'), 1),
    event('tool/call', { turn: 1, callId: 'call-1', name: 'read_file', arguments: '{"path":"config.json","password":"example-secret"}' }, 2),
    event('tool/result', {
      turn: 1,
      message: {
        source: { kind: 'tool', callId: 'call-1' },
        content: [{ type: 'tool-result', content: [{ type: 'text', text: 'Configuration valid' }] }],
      },
    }, 3),
    event('assistant/message', { turn: 1, message: message('model', 'Configuration validated') }, 4),
    event('turn/end', { turn: 1, reason: { kind: 'completed' } }, 5),
  ]
}

test('preserves exposed model reasoning in the assistant event', () => {
  const input = turnEvents()
  input[4].data.message.content.unshift({ type: 'reasoning', text: 'Check the documented batch limit.' })
  const result = collectTurnTraces(input, 1, 's1', config)
  const model = result.find(item => item.event.event_type === 'assistant_message').event
  assert.equal(model.reasoning, 'Check the documented batch limit.')
  assert.equal(model.assistant_message, 'Configuration validated')
})

for (const reasoning of ['\u0001'.repeat(10000), '😀'.repeat(10000), '字'.repeat(10000)]) {
  test(`bounds the complete trace request with ${JSON.stringify(reasoning.slice(0, 2))} reasoning`, () => {
    const input = turnEvents()
    const answer = '答'.repeat(10000)
    input[4].data.message.content = [{ type: 'text', text: answer }, { type: 'reasoning', text: reasoning }]
    const envelopes = collectTurnTraces(input, 1, 's1', config)
    assert.ok(envelopes.every(item => Buffer.byteLength(JSON.stringify(item)) <= 60000))
    const model = envelopes.find(item => item.event.event_type === 'assistant_message').event
    assert.equal(model.assistant_message, answer)
    assert.ok(model.reasoning.length > 0)
    assert.equal(envelopes.at(-1).event.assistant_message, answer)
    assert.equal(envelopes.at(-1).event.event_type, 'turn_end')
    assert.equal(input[4].data.message.content[1].text, reasoning)
  })
}

test('captures ordered DSH messages and paired tool execution with redacted input', () => {
  const envelopes = collectTurnTraces(turnEvents(), 1, 's1', config)
  const events = envelopes.map(item => item.event)
  assert.deepEqual(events.map(item => item.event_type), [
    'user_message', 'tool_start', 'tool_success', 'assistant_message', 'turn_end',
  ])
  assert.equal(events[1].tool_use_id, events[2].tool_use_id)
  assert.equal(events[1].tool_name, 'read_file')
  assert.equal(JSON.parse(events[1].tool_input).password, '[REDACTED]')
  assert.equal(events[2].tool_response, 'Configuration valid')
  assert.equal(events[4].assistant_message, 'Configuration validated')
  assert.equal(events[4].status, 'ok')
  assert.equal(envelopes[0].timestamp, new Date(1750000000001).toISOString())
  assert.deepEqual(
    collectTurnTraces(turnEvents(), 1, 's1', config).map(item => item.request_id),
    envelopes.map(item => item.request_id),
  )
})

test('captures observed tool failure and interrupted turn outcome', () => {
  const events = turnEvents()
  events[3].data.message.content[0].isError = true
  events[3].data.message.content[0].content[0].text = 'File unavailable'
  events[5].data.reason.kind = 'aborted'
  const trace = collectTurnTraces(events, 1, 's1', config).map(item => item.event)
  assert.equal(trace[2].event_type, 'tool_failure')
  assert.equal(trace[2].error, 'File unavailable')
  assert.equal(trace.at(-1).status, 'error')
})

test('bounds content and preserves metadata in metadata capture mode', () => {
  const events = turnEvents()
  events[1].data.content[0].text = 'x'.repeat(100000)
  const trace = collectTurnTraces(events, 1, 's1', config)
  assert.equal(trace[0].event.prompt.length, 10000)
  assert.equal(trace[0].event.prompt_size, 100000)
  assert.ok(trace.every(item => Buffer.byteLength(JSON.stringify(item)) < 65536))
  const metadata = collectTurnTraces(events, 1, 's1', { ...config, traceCaptureContent: false })
  assert.deepEqual(metadata[1].event, {
    event_type: 'tool_start', tool_name: 'read_file', tool_use_id: 'call-1',
    tool_input_size: events[2].data.arguments.length, turn_id: '1',
  })
})

test('lifecycle boundaries carry authenticated identity and stable event ids', () => {
  const first = traceEnvelope(config, 's1', 'session_end', { event_type: 'session_end' }, 1)
  const replay = traceEnvelope(config, 's1', 'session_end', { event_type: 'session_end' }, 2)
  assert.equal(first.request_id, replay.request_id)
  assert.equal(first.account_id, config.accountId)
  assert.equal(first.agent_id, config.agentId)
  assert.equal(first.session_id, 's1')
})

test('retains the original tool outcome when DSH rewrites its displayed result', () => {
  const events = turnEvents()
  const replacement = structuredClone(events[3])
  replacement.seq = 6
  replacement.surfaceOp = { op: 'replace', startSeq: 3, endSeq: 3 }
  replacement.data.message.content[0].content[0].text = 'Condensed result'
  events.splice(4, 0, replacement)
  const outcomes = collectTurnTraces(events, 1, 's1', config)
    .map(item => item.event).filter(item => item.event_type === 'tool_success')
  assert.deepEqual(outcomes.map(item => item.tool_response), ['Configuration valid'])
})

for (const [label, wrap] of [
  ['prefixed', text => `Config: ${text}`],
  ['fenced', text => `\`\`\`json\n${text}\n\`\`\``],
]) {
  test(`redacts ${label} JSON in prompts, assistant messages, and tool output`, () => {
    const events = turnEvents()
    const value = wrap('{"password":"hunter2","nested":{"apiKey":"example-key"},"label":"db"}')
    events[1].data.content[0].text = value
    events[3].data.message.content[0].content[0].text = value
    events[4].data.message.content[0].text = value
    const trace = collectTurnTraces(events, 1, 's1', config).map(item => item.event)
    const expected = wrap('{"password":"[REDACTED]","nested":{"apiKey":"[REDACTED]"},"label":"db"}')
    assert.equal(trace[0].prompt, expected)
    assert.equal(trace[2].tool_response, expected)
    assert.equal(trace[3].assistant_message, expected)
    assert.equal(trace[4].assistant_message, expected)
  })
}


for (const key of ['credentials', 'credential', 'privateKey', 'private_key']) {
  test(`redacts ${key} in tool input and output`, () => {
    const events = turnEvents()
    const credential = key === 'credentials' ? { value: 'opaque-example' } : 'opaque-example'
    const value = JSON.stringify({ [key]: credential, label: 'db' })
    events[2].data.arguments = value
    events[3].data.message.content[0].content[0].text = value
    const trace = collectTurnTraces(events, 1, 's1', config).map(item => item.event)
    assert.deepEqual(JSON.parse(trace[1].tool_input), { [key]: '[REDACTED]', label: 'db' })
    assert.equal(trace[2].tool_response, JSON.stringify({ [key]: '[REDACTED]', label: 'db' }))
  })
}

for (const wrap of [text => `Config: ${text}`, text => `\`\`\`json\n${text}\n\`\`\``]) {
  for (const depth of [1, 3]) {
    test(`redacts wrapped JSON encoded ${depth} times across trace messages`, () => {
      let value = { password: 'hunter2' }
      let expected = { password: '[REDACTED]' }
      for (let i = 0; i < depth; i++) {
        value = JSON.stringify(value)
        expected = JSON.stringify(expected)
      }
      const events = turnEvents()
      const input = wrap(JSON.stringify({ payload: value }))
      events[1].data.content[0].text = input
      events[3].data.message.content[0].content[0].text = input
      events[4].data.message.content[0].text = input
      const trace = collectTurnTraces(events, 1, 's1', config).map(item => item.event)
      const output = wrap(JSON.stringify({ payload: expected }))
      assert.equal(trace[0].prompt, output)
      assert.equal(trace[2].tool_response, output)
      assert.equal(trace[3].assistant_message, output)
      assert.equal(trace[4].assistant_message, output)
    })
  }
}


for (const [label, wrap] of [
  ['prefixed', text => `Config: ${text}`],
  ['fenced', text => `\`\`\`json\n${text}\n\`\`\``],
]) {
  for (const credential of [{ value: 'opaque-example' }, ['opaque-example'], 12345, true, null]) {
    test(`redacts ${label} sensitive JSON value ${JSON.stringify(credential)}`, () => {
      const events = turnEvents()
      const input = wrap(JSON.stringify({ credentials: credential, label: 'db', count: 3 }))
      events[1].data.content[0].text = input
      events[2].data.arguments = input
      events[3].data.message.content[0].content[0].text = input
      events[4].data.message.content[0].text = input
      const trace = collectTurnTraces(events, 1, 's1', config).map(item => item.event)
      const expected = wrap(JSON.stringify({ credentials: '[REDACTED]', label: 'db', count: 3 }))
      assert.equal(trace[0].prompt, expected)
      assert.equal(trace[1].tool_input, expected)
      assert.equal(trace[2].tool_response, expected)
      assert.equal(trace[3].assistant_message, expected)
      assert.equal(trace[4].assistant_message, expected)
    })
  }
}

for (const [input, expected] of [
  ['DB_PASSWORD=hunter2', 'DB_PASSWORD=[REDACTED]'],
  ['OGMEM_DSH_API_KEY=ogm_example_credential', 'OGMEM_DSH_API_KEY=[REDACTED]'],
  ["export DB_PASSWORD = 'hunter 2'", 'export DB_PASSWORD=[REDACTED]'],
  ['Config: DB_PASSWORD: "hunter 2"', 'Config: DB_PASSWORD=[REDACTED]'],
  ['Config:\nDB_PASSWORD=hunter2\nAPP_MODE=dev', 'Config:\nDB_PASSWORD=[REDACTED]\nAPP_MODE=dev'],
]) {
  test(`redacts environment assignment ${input.split('=')[0]}`, () => {
    const events = turnEvents()
    events[3].data.message.content[0].content[0].text = input
    const trace = collectTurnTraces(events, 1, 's1', config).map(item => item.event)
    assert.equal(trace[2].tool_response, expected)
  })
}


for (const [label, wrap] of [
  ['prefixed', text => `Config: ${text}`],
  ['fenced', text => `\`\`\`json\n${text}\n\`\`\``],
]) {
  test(`redacts ${label} oversized JSON before truncation`, () => {
    const events = turnEvents()
    const input = wrap(JSON.stringify({ credentials: { value: 'opaque-example' }, padding: 'x'.repeat(20000) }))
    events[1].data.content[0].text = input
    events[2].data.arguments = input
    events[3].data.message.content[0].content[0].text = input
    events[4].data.message.content[0].text = input
    const trace = collectTurnTraces(events, 1, 's1', config).map(item => item.event)
    assert.equal(trace[0].prompt, '[TRUNCATED_JSON]')
    assert.equal(trace[1].tool_input, '[TRUNCATED_JSON]')
    assert.equal(trace[2].tool_response, '[TRUNCATED_JSON]')
    assert.equal(trace[3].assistant_message, '[TRUNCATED_JSON]')
    assert.equal(trace[4].assistant_message, '[TRUNCATED_JSON]')
  })
  for (const depth of [1, 3]) {
    test(`redacts ${label} top-level JSON string encoded ${depth} times`, () => {
      let value = { password: 'hunter2', label: 'db' }
      let expected = { password: '[REDACTED]', label: 'db' }
      for (let i = 0; i < depth; i++) {
        value = JSON.stringify(value)
        expected = JSON.stringify(expected)
      }
      const events = turnEvents()
      const input = wrap(JSON.stringify(value))
      events[1].data.content[0].text = input
      events[2].data.arguments = input
      events[3].data.message.content[0].content[0].text = input
      events[4].data.message.content[0].text = input
      const trace = collectTurnTraces(events, 1, 's1', config).map(item => item.event)
      const output = wrap(JSON.stringify(expected))
      assert.equal(trace[0].prompt, output)
      assert.equal(trace[1].tool_input, output)
      assert.equal(trace[2].tool_response, output)
      assert.equal(trace[3].assistant_message, output)
      assert.equal(trace[4].assistant_message, output)
    })
  }
}


for (const [name, input, expected] of [
  ['truncated password', '{"password":"hunter2', '[REDACTED]'],
  ['truncated credential container', 'Config: {"credentials":{"value":"opaque-example"', '[REDACTED]'],
  ['truncated fenced JSON', '```json\n{"password":"hunter2\n```', '[REDACTED]'],
  ['truncated field', '"password":"hunter2', '[REDACTED]'],
  ['escaped sensitive key', '{"pass\\u0077ord":"hunter2', '[REDACTED]'],
  ['scan budget', '['.repeat(14000), '[REDACTED_SCAN_LIMIT]'],
]) {
  test(`persists a safe placeholder for ${name}`, () => {
    const events = turnEvents()
    events[1].data.content[0].text = input
    events[2].data.arguments = input
    events[3].data.message.content[0].content[0].text = input
    events[4].data.message.content[0].text = input
    const trace = collectTurnTraces(events, 1, 's1', config).map(item => item.event)
    assert.equal(trace[0].prompt, expected)
    assert.equal(trace[1].tool_input, expected)
    assert.equal(trace[2].tool_response, expected)
    assert.equal(trace[3].assistant_message, expected)
    assert.equal(trace[4].assistant_message, expected)
  })
}


for (const [label, wrap] of [
  ['standalone', text => text],
  ['prefixed', text => `Config: ${text}`],
  ['fenced', text => `\`\`\`json\n${text}\n\`\`\``],
]) {
  for (const credential of [{ value: 'opaque-example' }, ['opaque-example'], 'opaque-example', 123, true, null]) {
    test(`redacts ${label} sensitive field with value ${JSON.stringify(credential)}`, () => {
      const input = wrap('"credentials":' + JSON.stringify(credential))
      const events = turnEvents()
      events[1].data.content[0].text = input
      events[2].data.arguments = input
      events[3].data.message.content[0].content[0].text = input
      events[4].data.message.content[0].text = input
      const trace = collectTurnTraces(events, 1, 's1', config).map(item => item.event)
      assert.equal(trace[0].prompt, '[REDACTED]')
      assert.equal(trace[1].tool_input, '[REDACTED]')
      assert.equal(trace[2].tool_response, '[REDACTED]')
      assert.equal(trace[3].assistant_message, '[REDACTED]')
      assert.equal(trace[4].assistant_message, '[REDACTED]')
    })
  }
}


for (const input of [
  "{'password': 'hunter2'}",
  "Config: {'credentials': {'value': 'opaque-example'}}",
  "```python\n{'api_key': ['opaque-example']}\n```",
  "'privateKey': 'opaque-example'",
  "{'password': 123}",
  "{'credentials': None}",
]) {
  test(`redacts single-quoted sensitive fields in ${input}`, () => {
    const events = turnEvents()
    events[1].data.content[0].text = input
    events[2].data.arguments = input
    events[3].data.message.content[0].content[0].text = input
    events[4].data.message.content[0].text = input
    const trace = collectTurnTraces(events, 1, 's1', config).map(item => item.event)
    assert.equal(trace[0].prompt, '[REDACTED]')
    assert.equal(trace[1].tool_input, '[REDACTED]')
    assert.equal(trace[2].tool_response, '[REDACTED]')
    assert.equal(trace[3].assistant_message, '[REDACTED]')
    assert.equal(trace[4].assistant_message, '[REDACTED]')
  })
}


for (const key of ['pwd', 'passwd', 'credentials', 'credential', 'private-key', 'API_KEY']) {
  test(`redacts whitespace-separated --${key} credentials`, () => {
    const input = `command --mode safe --${key} \t\n"opaque example" --count 3`
    const expected = `command --mode safe --${key} [REDACTED] --count 3`
    const events = turnEvents()
    events[1].data.content[0].text = input
    events[2].data.arguments = input
    events[3].data.message.content[0].content[0].text = input
    events[4].data.message.content[0].text = input
    const trace = collectTurnTraces(events, 1, 's1', config).map(item => item.event)
    assert.equal(trace[0].prompt, expected)
    assert.equal(trace[1].tool_input, expected)
    assert.equal(trace[2].tool_response, expected)
    assert.equal(trace[3].assistant_message, expected)
    assert.equal(trace[4].assistant_message, expected)
  })
}


for (const scheme of ['Bearer', 'bearer', 'Basic']) {
  test(`redacts the complete ${scheme} credential in labeled text`, () => {
    const events = turnEvents()
    const input = `token: ${scheme} opaque-example`
    events[1].data.content[0].text = input
    events[2].data.arguments = input
    events[3].data.message.content[0].content[0].text = input
    events[4].data.message.content[0].text = input
    const trace = collectTurnTraces(events, 1, 's1', config).map(item => item.event)
    assert.equal(trace[0].prompt, '[REDACTED_SECRET]')
    assert.equal(trace[1].tool_input, '[REDACTED_SECRET]')
    assert.equal(trace[2].tool_response, '[REDACTED_SECRET]')
    assert.equal(trace[3].assistant_message, '[REDACTED_SECRET]')
    assert.equal(trace[4].assistant_message, '[REDACTED_SECRET]')
  })
}

const piiCases = JSON.parse(readFileSync(new URL('./fixtures/pii_cases.json', import.meta.url), 'utf8'))
for (const fixture of piiCases) {
  for (const wrap of [text => text, text => `Config: ${text}`, text => `\`\`\`json\n${text}\n\`\`\``]) {
    test(`redacts shared PII case ${fixture.name} in ${wrap('JSON')}`, () => {
      const input = wrap(typeof fixture.input === 'string' ? fixture.input : JSON.stringify(fixture.input))
      const expected = wrap(typeof fixture.expected === 'string' ? fixture.expected : JSON.stringify(fixture.expected))
      const events = turnEvents()
      events[1].data.content[0].text = input
      events[2].data.arguments = input
      events[3].data.message.content[0].content[0].text = input
      events[4].data.message.content[0].text = input
      const trace = collectTurnTraces(events, 1, 's1', config).map(item => item.event)
      assert.equal(trace[0].prompt, expected)
      assert.equal(trace[1].tool_input, expected)
      assert.equal(trace[2].tool_response, expected)
      assert.equal(trace[3].assistant_message, expected)
      assert.equal(trace[4].assistant_message, expected)
    })
  }
}

test('preserves entries when private keys map to the same redacted name', () => {
  const input = '{"alice@example.com":"admin","bob@example.com":"reader","__proto__":"metadata"}'
  const events = turnEvents()
  events[2].data.arguments = input
  const trace = collectTurnTraces(events, 1, 's1', config).map(item => item.event)
  assert.deepEqual(JSON.parse(trace[1].tool_input), JSON.parse(
    '{"[REDACTED_EMAIL]":"admin","[REDACTED_EMAIL]_2":"reader","__proto__":"metadata"}',
  ))
})

const yamlCases = JSON.parse(readFileSync(new URL('./fixtures/yaml_secret_cases.json', import.meta.url), 'utf8'))
for (const text of yamlCases) {
  for (const asJson of [false, true]) {
    test(`redacts YAML multiline credentials ${JSON.stringify(text)} json=${asJson}`, () => {
      const input = asJson ? JSON.stringify({ output: text }) : text
      const expected = asJson ? JSON.stringify({ output: '[REDACTED_SECRET]' }) : '[REDACTED_SECRET]'
      const events = turnEvents()
      events[1].data.content[0].text = input
      events[2].data.arguments = input
      events[3].data.message.content[0].content[0].text = input
      events[4].data.message.content[0].text = input
      const trace = collectTurnTraces(events, 1, 's1', config).map(item => item.event)
      assert.equal(trace[0].prompt, expected)
      assert.equal(trace[1].tool_input, expected)
      assert.equal(trace[2].tool_response, expected)
      assert.equal(trace[3].assistant_message, expected)
      assert.equal(trace[4].assistant_message, expected)
    })
  }
}

for (const [text, cleaned] of [
  ["dbPassword: 'abc''def'", 'dbPassword=[REDACTED]'],
  ["dbPassword: 'abc''''def'", 'dbPassword=[REDACTED]'],
  ["password: 'abc''def'", '[REDACTED_SECRET]'],
  ["--dbPassword 'abc''def'", '--dbPassword [REDACTED]'],
  ['DB_PASSWORD="prefix actual-secret', 'DB_PASSWORD=[REDACTED]'],
  ["DB_PASSWORD='prefix actual-secret", 'DB_PASSWORD=[REDACTED]'],
  ["DB_PASSWORD=\"prefix \"'actual-secret'", 'DB_PASSWORD=[REDACTED]'],
  ['DB_PASSWORD=prefix"actual secret"tail', 'DB_PASSWORD=[REDACTED]'],
  ['DB_PASSWORD="prefix actual-secret\\', 'DB_PASSWORD=[REDACTED]'],
  ['DB_PASSWORD=prefix\\ actual-secret', 'DB_PASSWORD=[REDACTED]'],
]) {
  for (const asJson of [false, true]) {
    test(`redacts complete quoted credentials ${text} json=${asJson}`, () => {
      const input = asJson ? JSON.stringify({ output: text }) : text
      const expected = asJson ? JSON.stringify({ output: cleaned }) : cleaned
      const events = turnEvents()
      events[1].data.content[0].text = input
      events[2].data.arguments = input
      events[3].data.message.content[0].content[0].text = input
      events[4].data.message.content[0].text = input
      const trace = collectTurnTraces(events, 1, 's1', config).map(item => item.event)
      assert.equal(trace[0].prompt, expected)
      assert.equal(trace[1].tool_input, expected)
      assert.equal(trace[2].tool_response, expected)
      assert.equal(trace[3].assistant_message, expected)
      assert.equal(trace[4].assistant_message, expected)
    })
  }
}
