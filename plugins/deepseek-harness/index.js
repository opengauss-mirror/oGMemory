/**
 * DeepSeek Harness lifecycle integration for oGMemory.
 *
 * Recall is resolved at the first pre-step of each turn and appended to that
 * same model request as a source-attributed snapshot message. Write-back
 * starts only after the durable turn/end event commits, then runs on a
 * per-session ordered queue.
 */

import { join } from 'node:path'
import { createUserMessage } from '@deepseek-ai/dsh-llm'

import { resolveConfig } from './lib/config.js'
import { OgMemoryClient } from './lib/client.js'
import { scopedPendingDir } from './lib/pending-queue.js'
import { extractDirectUserPrompt, renderComposeContext } from './lib/recall.js'
import { registerOgMemoryTools } from './lib/tools.js'
import { guardOgMemoryUri } from './lib/uri-guard.js'
import { collectTurnMessages, OrderedWriteQueue } from './lib/writeback.js'
import { collectTurnTraces, traceEnvelope } from './lib/trace.js'

export const name = 'ogmemory-lifecycle'

export const inject = ['agents', 'credentials', 'sessions', 'tools']

export function apply(ctx, inputConfig = {}) {
  const config = resolveConfig(inputConfig)
  const client = new OgMemoryClient(ctx, config)
  const activeSessions = new Set()
  const queueOptions = {
    logger: ctx.logger,
    maxAttempts: config.writebackMaxAttempts,
    initialDelayMs: config.writebackInitialDelayMs,
    maxDelayMs: config.writebackMaxDelayMs,
    pendingDir: scopedPendingDir(config.pendingDir, config),
    shutdownTimeoutMs: config.shutdownTimeoutMs,
  }
  const queue = config.autoCapture ? new OrderedWriteQueue({
    ...queueOptions,
    send: operation => operation.kind === 'dispose'
      ? client.dispose(operation)
      : client.afterTurn(operation),
  }) : null
  const traces = config.traceEnabled ? new OrderedWriteQueue({
    ...queueOptions,
    pendingDir: join(queueOptions.pendingDir, 'traces'),
    send: async operation => {
      for (const envelope of operation.envelopes ?? [operation.envelope]) {
        await client.trace({ sessionId: operation.sessionId, envelope })
      }
    },
  }) : null
  void queue?.start()
  void traces?.start()
  const traceBoundary = (sessionId, type) => traces?.enqueue(sessionId, {
    sessionId,
    envelope: traceEnvelope(config, sessionId, type, { event_type: type }),
  })
  registerOgMemoryTools(ctx, client, config)
  if (config.toolsEnabled) ctx.on('tools/pre-execute', guardOgMemoryUri)

  ctx.on('agent/session-start', ({ agent }) => {
    activeSessions.add(String(agent.session.id))
    traceBoundary(String(agent.session.id), 'session_start')
  })

  ctx.on('agent/pre-step', async ({ agent, step, signal }, next) => {
    const decision = await next()
    if (!config.autoRecall || !config.composeEnabled || decision.kind === 'reject' || step !== 1 || signal.aborted) {
      return decision
    }

    const prompt = extractDirectUserPrompt(decision.messages, config.maxPromptChars)
    if (prompt.length === 0) {
      return decision
    }

    try {
      const result = await client.compose({
        sessionId: String(agent.session.id),
        prompt,
        tokenBudget: config.tokenBudget,
      }, signal)
      if (signal.aborted) return decision
      const context = renderComposeContext(result, config.maxContextChars)
      ctx.logger.debug(
        `ogmemory recall session=${String(agent.session.id)} chars=${context.length}`,
      )
      if (context.length === 0) return decision
      return {
        kind: 'enter',
        messages: [
          ...decision.messages,
          createUserMessage({
            source: {
              kind: 'plugin',
              plugin: name,
              form: 'snapshot',
              sections: [{ name: 'ogmemory', text: context }],
            },
            content: [{ type: 'text', text: context }],
          }),
        ],
      }
    } catch (error) {
      if (!signal.aborted) {
        ctx.logger.warn(
          `ogmemory recall failed session=${String(agent.session.id)}: ${errorMessage(error)}`,
        )
      }
    }
    return decision
  }, { prepend: true })

  ctx.on('session/event', (session, event) => {
    if (event.type !== 'turn/end') return
    const events = session.snapshotEvents()
    if (traces) {
      traces.enqueue(String(session.id), {
        sessionId: String(session.id),
        envelopes: collectTurnTraces(events, event.data.turn, String(session.id), config),
      })
    }
    if (!config.autoCapture || !config.writebackEnabled) return
    const messages = collectTurnMessages(events, event.data.turn, {
      captureToolResults: config.captureToolResults,
      maxMessageChars: config.maxMessageChars,
      maxToolResultChars: config.maxToolResultChars,
      maxTurnMessages: config.maxTurnMessages,
    })
    if (messages.length === 0) return
    queue.enqueue(String(session.id), {
      kind: 'afterTurn',
      sessionId: String(session.id),
      messages,
      prePromptMessageCount: 0,
      forceExtract: config.forceExtract,
      wait: true,
    })
  })

  ctx.on('session/flush', session => Promise.all([
    queue?.flush(String(session.id)), traces?.flush(String(session.id)),
  ]))

  ctx.on('session/disposed', async session => {
    const sessionId = String(session.id)
    traceBoundary(sessionId, 'session_end')
    if (config.autoCapture && config.disposeEnabled) queue.enqueue(sessionId, { kind: 'dispose', sessionId })
    await Promise.all([queue?.retire(sessionId), traces?.retire(sessionId)])
    activeSessions.delete(sessionId)
  })

  ctx.effect(() => async () => {
    for (const sessionId of activeSessions) traceBoundary(sessionId, 'session_end')
    if (config.autoCapture && config.disposeEnabled) {
      for (const sessionId of activeSessions) {
        queue.enqueue(sessionId, { kind: 'dispose', sessionId })
      }
    }
    await Promise.all([queue?.dispose(), traces?.dispose()])
  }, 'ogmemory writeback queue')
}

function errorMessage(error) {
  return error instanceof Error ? error.message : String(error)
}

export { resolveConfig } from './lib/config.js'
export { OgMemoryClient, OgMemoryHttpError } from './lib/client.js'
export { extractDirectUserPrompt, renderComposeContext } from './lib/recall.js'
export { collectTurnMessages, OrderedWriteQueue } from './lib/writeback.js'
