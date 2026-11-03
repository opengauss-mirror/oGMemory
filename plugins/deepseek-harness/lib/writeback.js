import { PendingStore } from './pending-queue.js'

export function collectTurnMessages(events, turn, options) {
  if (!Array.isArray(events)) return []
  const endIndex = findLastIndex(events, event =>
    event?.type === 'turn/end' && event.data?.turn === turn)
  if (endIndex < 0) return []
  let startIndex = endIndex
  while (startIndex >= 0) {
    const event = events[startIndex]
    if (event?.type === 'turn/start' && event.data?.turn === turn) break
    startIndex -= 1
  }
  if (startIndex < 0) return []

  const calls = new Map()
  const messages = []
  for (const event of events.slice(startIndex + 1, endIndex)) {
    if (event?.type === 'tool/call' && event.data?.turn === turn) {
      calls.set(String(event.data.callId), {
        name: String(event.data.name ?? 'unknown'),
        arguments: String(event.data.arguments ?? ''),
      })
      continue
    }
    if (event?.type === 'user/message') {
      if (event.data?.source?.kind !== 'user') continue
      pushMessage(messages, 'user', visibleText(event.data.content), event.time, options.maxMessageChars)
      continue
    }
    if (event?.type === 'assistant/message' && event.data?.turn === turn) {
      if (event.data.message?.source?.kind !== 'model') continue
      pushMessage(
        messages,
        'assistant',
        visibleText(event.data.message?.content),
        event.time,
        options.maxMessageChars,
      )
      continue
    }
    if (options.captureToolResults && event?.type === 'tool/result' && event.data?.turn === turn) {
      if (event.data.message?.source?.kind !== 'tool') continue
      const callId = String(event.data.message?.source?.callId ?? '')
      const call = calls.get(callId)
      const output = toolResultText(event.data.message?.content)
      const text = [
        `tool=${call?.name ?? 'unknown'}`,
        call?.arguments ? `arguments=${call.arguments}` : '',
        output ? `result=${output}` : '',
      ].filter(Boolean).join('\n')
      pushMessage(messages, 'tool', text, event.time, options.maxToolResultChars)
    }
  }

  if (messages.length <= options.maxTurnMessages) return messages
  if (options.maxTurnMessages === 1) return [messages[0]]
  return [messages[0], ...messages.slice(-(options.maxTurnMessages - 1))]
}

export class OrderedWriteQueue {
  constructor({
    logger,
    send,
    maxAttempts,
    initialDelayMs,
    maxDelayMs,
    pendingDir = '',
    shutdownTimeoutMs = 5_000,
  }) {
    this.logger = logger
    this.send = send
    this.maxAttempts = maxAttempts
    this.initialDelayMs = initialDelayMs
    this.maxDelayMs = maxDelayMs
    this.shutdownTimeoutMs = shutdownTimeoutMs
    this.store = new PendingStore(pendingDir)
    this.sessions = new Map()
    this.blockedSessions = new Set()
    this.retryTimers = new Map()
    this.disposed = false
    this.startPromise = null
    this.persistence = Promise.resolve()
  }

  start() {
    if (!this.startPromise) {
      const entries = this.store.list()
      this.persistence = entries.catch(() => {})
      this.startPromise = this.replayPending(entries).catch(error => {
        this.logger.error(`ogmemory pending replay failed: ${errorMessage(error)}`)
      })
    }
    return this.startPromise
  }

  enqueue(sessionId, payload) {
    if (this.disposed) {
      this.logger.warn(`ogmemory writeback dropped after disposal session=${sessionId}`)
      return Promise.resolve()
    }
    this.start()
    const persisted = this.persistence
      .then(() => this.persistNew(sessionId, payload))
      .then(entry => ({ entry }), error => ({ error }))
    this.persistence = persisted
    const previous = this.sessions.get(sessionId) ?? this.startPromise
    const next = previous
      .then(async () => {
        const { entry, error } = await persisted
        if (error) throw error
        if (!this.blockedSessions.has(sessionId)) {
          let pending
          try {
            pending = await this.store.read(entry.path)
          } catch (error) {
            this.blockedSessions.add(sessionId)
            this.scheduleReplay(sessionId)
            throw error
          }
          if (pending) await this.deliver(sessionId, pending.payload, pending.path)
        }
      })
      .catch(error => {
        this.logger.error(`ogmemory writeback failed session=${sessionId}: ${errorMessage(error)}`)
      })
    this.sessions.set(sessionId, next)
    return next
  }

  flush(sessionId) {
    return this.sessions.get(sessionId) ?? Promise.resolve()
  }

  async retire(sessionId) {
    const pending = this.sessions.get(sessionId)
    if (pending === undefined) return
    await pending
    if (this.sessions.get(sessionId) === pending) this.sessions.delete(sessionId)
  }

  async dispose() {
    this.disposed = true
    for (const timer of this.retryTimers.values()) clearTimeout(timer)
    this.retryTimers.clear()
    const pending = Promise.allSettled([
      this.startPromise ?? Promise.resolve(),
      ...this.sessions.values(),
      this.persistence,
    ])
    const completed = await within(pending, this.shutdownTimeoutMs)
    if (!completed) {
      this.logger.warn(`ogmemory shutdown timed out after ${this.shutdownTimeoutMs}ms; writes remain durable`)
      return
    }
    this.sessions.clear()
  }

  async replayPending(pendingEntries) {
    const entries = await pendingEntries
    const groups = new Map()
    for (const entry of entries) {
      if (entry.error) {
        this.logger.error(`ogmemory invalid pending entry path=${entry.path}: ${errorMessage(entry.error)}`)
        continue
      }
      const group = groups.get(entry.sessionId) ?? []
      group.push(entry)
      groups.set(entry.sessionId, group)
    }
    await Promise.all([...groups.entries()].map(async ([sessionId, items]) => {
      for (const item of items) {
        if (this.blockedSessions.has(sessionId)) break
        await this.deliver(sessionId, item.payload, item.path)
      }
    }))
  }

  async persistNew(sessionId, payload) {
    let delayMs = this.initialDelayMs
    do {
      try {
        return await this.store.put(sessionId, payload)
      } catch (error) {
        if (this.disposed) break
        this.logger.warn(
          `ogmemory pending persist retry session=${sessionId} delayMs=${delayMs}: ${errorMessage(error)}`,
        )
        await delay(delayMs)
        delayMs = Math.min(delayMs * 2, this.maxDelayMs)
      }
    } while (!this.disposed)
    throw new Error(`ogmemory pending persist stopped during disposal session=${sessionId}`)
  }

  scheduleReplay(sessionId) {
    if (this.disposed || this.retryTimers.has(sessionId)) return
    const timer = setTimeout(() => {
      this.retryTimers.delete(sessionId)
      if (this.disposed) return
      const previous = this.sessions.get(sessionId) ?? Promise.resolve()
      const next = previous
        .then(() => this.replaySession(sessionId))
        .catch(error => {
          this.logger.error(`ogmemory pending replay failed session=${sessionId}: ${errorMessage(error)}`)
          if (this.blockedSessions.has(sessionId)) this.scheduleReplay(sessionId)
        })
      this.sessions.set(sessionId, next)
    }, this.maxDelayMs)
    timer.unref?.()
    this.retryTimers.set(sessionId, timer)
  }

  async replaySession(sessionId) {
    if (this.disposed) return
    const entries = await this.store.list()
    this.blockedSessions.delete(sessionId)
    for (const entry of entries) {
      if (entry.error || entry.sessionId !== sessionId) continue
      await this.deliver(sessionId, entry.payload, entry.path)
      if (this.blockedSessions.has(sessionId)) break
    }
  }

  async deliver(sessionId, payload, path) {
    try {
      await this.sendWithRetry(sessionId, payload)
      await this.store.remove(path)
    } catch (error) {
      if (error?.retryable === true) {
        this.blockedSessions.add(sessionId)
        this.scheduleReplay(sessionId)
        this.logger.warn(`ogmemory writeback queued for replay session=${sessionId}: ${errorMessage(error)}`)
        return
      }
      await this.store.remove(path)
      this.logger.error(`ogmemory non-retryable writeback dropped session=${sessionId}: ${errorMessage(error)}`)
    }
  }

  async sendWithRetry(sessionId, payload) {
    let delayMs = this.initialDelayMs
    for (let attempt = 1; attempt <= this.maxAttempts; attempt += 1) {
      try {
        const result = await this.send(payload)
        this.logger.debug(`ogmemory writeback complete session=${sessionId} attempt=${attempt}`)
        return result
      } catch (error) {
        const retryable = error?.retryable === true
        if (!retryable || attempt === this.maxAttempts) throw error
        this.logger.warn(
          `ogmemory writeback retry session=${sessionId} attempt=${attempt} delayMs=${delayMs}: ${errorMessage(error)}`,
        )
        await delay(delayMs)
        delayMs = Math.min(delayMs * 2, this.maxDelayMs)
      }
    }
  }
}

async function within(promise, timeoutMs) {
  let timer
  const timeout = new Promise(resolve => {
    timer = setTimeout(() => resolve(false), timeoutMs)
    timer.unref?.()
  })
  const completed = promise.then(() => true)
  try {
    return await Promise.race([completed, timeout])
  } finally {
    clearTimeout(timer)
  }
}

function visibleText(content) {
  if (!Array.isArray(content)) return ''
  return content
    .filter(block => block?.type === 'text' && typeof block.text === 'string')
    .map(block => block.text)
    .join('\n')
    .trim()
}

function toolResultText(content) {
  if (!Array.isArray(content)) return ''
  const blocks = []
  for (const block of content) {
    if (block?.type !== 'tool-result' || !Array.isArray(block.content)) continue
    const text = visibleText(block.content)
    if (text) blocks.push(text)
  }
  return blocks.join('\n')
}

function pushMessage(messages, role, content, time, maxChars) {
  const text = content.trim().slice(0, maxChars)
  if (!text) return
  messages.push({
    role,
    content: text,
    created_at: new Date(Number.isFinite(time) ? time : Date.now()).toISOString(),
  })
}

function findLastIndex(values, predicate) {
  for (let index = values.length - 1; index >= 0; index -= 1) {
    if (predicate(values[index])) return index
  }
  return -1
}

function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms))
}

function errorMessage(error) {
  return error instanceof Error ? error.message : String(error)
}
