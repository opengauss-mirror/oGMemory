import { createHash } from 'node:crypto'
import { redactString } from './redaction.js'

const CONTENT_CHARS = 10_000
const TRACE_BODY_BYTES = 60_000

export function traceEnvelope(config, sessionId, identity, event, time = Date.now()) {
  const scope = [config.accountId, config.userId, config.agentId, sessionId, identity]
  const envelope = {
    request_id: createHash('sha256').update(JSON.stringify(scope)).digest('hex'),
    timestamp: new Date(time).toISOString(),
    account_id: config.accountId,
    user_id: config.userId,
    agent_id: config.agentId,
    session_id: sessionId,
    event: { ...event },
  }
  for (const field of ['reasoning', 'assistant_message', 'prompt', 'tool_input', 'tool_response', 'error']) {
    if (Buffer.byteLength(JSON.stringify(envelope)) <= TRACE_BODY_BYTES) break
    const value = envelope.event[field]
    if (typeof value !== 'string') continue
    envelope.event.truncated = true
    let low = 0, high = value.length
    while (low < high) {
      const middle = Math.ceil((low + high) / 2)
      envelope.event[field] = value.slice(0, middle)
      if (Buffer.byteLength(JSON.stringify(envelope)) <= TRACE_BODY_BYTES) low = middle
      else high = middle - 1
    }
    envelope.event[field] = value.slice(0, low)
  }
  return envelope
}

export function collectTurnTraces(events, turn, sessionId, config) {
  const end = events.findLastIndex(item => item.type === 'turn/end' && item.data.turn === turn)
  if (end < 0) return []
  const start = events.findLastIndex((item, index) =>
    index < end && item.type === 'turn/start' && item.data.turn === turn)
  if (start < 0) return []
  const traces = []
  const calls = new Map()
  let assistant = ''
  const addContent = (event, field, value) => {
    event[`${field}_size`] = value.length
    if (config.traceCaptureContent) event[field] = redactString(value, CONTENT_CHARS)
    return event
  }
  for (let index = start + 1; index <= end; index += 1) {
    const raw = events[index]
    const data = raw.data
    let event
    if (raw.type === 'user/message' && data.source?.kind === 'user') {
      event = addContent({ event_type: 'user_message' }, 'prompt', text(data.content))
    } else if (data.turn !== turn) {
      continue
    } else if (raw.type === 'assistant/message' && data.message?.source?.kind === 'model') {
      assistant = text(data.message.content)
      const reasoning = data.message.content.filter(item => item.type === 'reasoning')
        .map(item => item.text || item.reasoning || '').join('\n')
      if (!assistant && !reasoning) continue
      event = addContent({ event_type: 'assistant_message' }, 'assistant_message', assistant)
      if (reasoning) addContent(event, 'reasoning', reasoning)
    } else if (raw.type === 'tool/call') {
      const tool = {
        tool_name: redactString(data.name || 'unknown_tool', 256),
        tool_use_id: String(data.callId).slice(0, 512),
      }
      calls.set(String(data.callId), tool)
      event = addContent({ event_type: 'tool_start', ...tool }, 'tool_input', String(data.arguments || ''))
    } else if (raw.type === 'tool/result' && data.message?.source?.kind === 'tool') {
      const callId = String(data.message.source.callId)
      const tool = calls.get(callId)
      if (!tool) continue
      calls.delete(callId)
      const blocks = data.message.content.filter(item => item.type === 'tool-result')
      const failed = blocks.some(item => item.isError === true)
      const output = blocks.map(item => text(item.content)).filter(Boolean).join('\n')
      event = addContent(
        { event_type: failed ? 'tool_failure' : 'tool_success', ...tool },
        failed ? 'error' : 'tool_response',
        output || (failed ? String(data.error?.message || 'Tool failed') : ''),
      )
    } else if (raw.type === 'turn/end') {
      event = addContent({
        event_type: 'turn_end',
        status: data.reason?.kind === 'completed' ? 'ok' : 'error',
        reason: redactString(data.reason?.kind || 'unknown', 256),
      }, 'assistant_message', assistant)
    }
    if (event) {
      event.turn_id = String(turn)
      traces.push(traceEnvelope(config, sessionId, ['event', raw.seq ?? index], event, raw.time))
    }
  }
  return traces
}

function text(content) {
  return (Array.isArray(content) ? content : [])
    .filter(item => item.type === 'text' && typeof item.text === 'string')
    .map(item => item.text).join('\n').trim()
}
