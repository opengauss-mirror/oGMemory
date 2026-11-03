export class OgMemoryHttpError extends Error {
  constructor(message, { status, retryable = false } = {}) {
    super(message)
    this.name = 'OgMemoryHttpError'
    this.status = status
    this.retryable = retryable
  }
}

export class OgMemoryClient {
  constructor(ctx, config) {
    this.ctx = ctx
    this.config = config
  }

  async compose({ sessionId, prompt, tokenBudget }, signal) {
    return this.request('/api/v1/compose', {
      ...this.identity(),
      sessionId,
      prompt,
      tokenBudget,
    }, {
      signal,
      timeoutMs: this.config.composeTimeoutMs,
    })
  }

  async afterTurn({
    sessionId,
    pendingId,
    messages,
    prePromptMessageCount = 0,
    forceExtract = false,
    wait = false,
  }) {
    const result = await this.request('/api/v1/after_turn', {
      ...this.identity(),
      sessionId,
      client_request_id: pendingId,
      messages,
      prePromptMessageCount,
      forceExtract,
      wait,
    }, {
      timeoutMs: this.config.afterTurnTimeoutMs,
    })
    if (result?.ok === false) {
      throw new OgMemoryHttpError(
        `after_turn rejected: ${safeErrorText(result.error)}`,
        { retryable: result.retryable === true },
      )
    }
    return result
  }

  async dispose({ sessionId }) {
    const result = await this.request('/api/v1/dispose', {
      ...this.identity(),
      sessionId,
    }, {
      timeoutMs: this.config.disposeTimeoutMs,
    })
    if (result?.reason === 'thread_unavailable') {
      throw new OgMemoryHttpError('dispose background thread unavailable', { retryable: true })
    }
    return result
  }

  async trace({ envelope }) {
    const result = await this.request('/api/v1/trace/events', envelope, {
      timeoutMs: this.config.traceTimeoutMs,
    })
    if (result?.ok !== true) {
      throw new OgMemoryHttpError('trace persistence unavailable', {
        retryable: result?.retryable !== false,
      })
    }
    return result
  }

  queryMemory({ sessionId, query, types, topK, scoreThreshold }) {
    return this.request('/api/v1/memories/query', {
      ...this.identity(),
      sessionId,
      query,
      ...(types?.length ? { types } : {}),
      top_k: topK,
      ...(scoreThreshold === undefined ? {} : { score_threshold: scoreThreshold }),
    }, { timeoutMs: this.config.toolTimeoutMs })
  }

  readMemory({ sessionId, uri, level, maxChars }) {
    return this.request('/api/v1/memories/read', {
      ...this.identity(),
      sessionId,
      uri,
      level,
      max_chars: maxChars,
    }, { timeoutMs: this.config.toolTimeoutMs })
  }

  writeMemory({ sessionId, text, clientRequestId }) {
    return this.request('/api/v1/memories/write', {
      ...this.identity(),
      sessionId,
      text,
      client_request_id: clientRequestId,
      force_extract: true,
    }, { timeoutMs: this.config.toolTimeoutMs })
  }

  deleteMemory({ sessionId, uri, reason, clientRequestId }) {
    return this.request('/api/v1/memories/delete', {
      ...this.identity(),
      sessionId,
      uri,
      ...(reason ? { reason } : {}),
      client_request_id: clientRequestId,
    }, { timeoutMs: this.config.toolTimeoutMs })
  }

  identity() {
    return {
      accountId: this.config.accountId,
      userId: this.config.userId,
      agentId: this.config.agentId,
    }
  }

  async request(path, body, options) {
    const startedAt = Date.now()
    let outcome = 'ok'
    try {
      return await this.requestOnce(path, body, options)
    } catch (error) {
      outcome = 'error'
      throw error
    } finally {
      this.ctx.logger?.debug?.(
        `ogmemory http path=${path} session=${String(body.sessionId ?? '')} outcome=${outcome} durationMs=${Date.now() - startedAt}`,
      )
    }
  }

  async requestOnce(path, body, { signal, timeoutMs }) {
    const headers = {
      'Content-Type': 'application/json',
      'X-Account-ID': this.config.accountId,
      'X-User-ID': this.config.userId,
      'X-Agent-ID': this.config.agentId,
    }
    if (this.config.apiKeyEnv) {
      const resolved = await this.ctx.credentials.resolve(this.config.apiKeyEnv)
      if (resolved?.value) headers['X-API-Key'] = resolved.value
    }

    const timeoutSignal = AbortSignal.timeout(timeoutMs)
    const requestSignal = signal === undefined
      ? timeoutSignal
      : AbortSignal.any([signal, timeoutSignal])
    let response
    try {
      response = await fetch(`${this.config.baseUrl}${path}`, {
        method: 'POST',
        headers,
        body: JSON.stringify(body),
        signal: requestSignal,
      })
    } catch (error) {
      if (signal?.aborted) throw signal.reason ?? error
      throw new OgMemoryHttpError(`request failed: ${errorMessage(error)}`, { retryable: true })
    }

    if (!response.ok) {
      const detail = await boundedResponseText(response)
      throw new OgMemoryHttpError(
        `HTTP ${response.status} from ${path}${detail ? `: ${detail}` : ''}`,
        { status: response.status, retryable: response.status === 429 || response.status >= 500 },
      )
    }
    try {
      const result = await response.json()
      if (result === null || typeof result !== 'object' || Array.isArray(result)) {
        throw new Error('response is not an object')
      }
      return result
    } catch (error) {
      throw new OgMemoryHttpError(`invalid JSON from ${path}: ${errorMessage(error)}`)
    }
  }
}

async function boundedResponseText(response) {
  try {
    const text = await response.text()
    return text.replace(/\s+/g, ' ').trim().slice(0, 500)
  } catch {
    return ''
  }
}

function safeErrorText(value) {
  return typeof value === 'string' ? value.slice(0, 500) : 'unknown error'
}

function errorMessage(error) {
  return error instanceof Error ? error.message : String(error)
}
