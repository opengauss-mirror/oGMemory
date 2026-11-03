import { randomUUID } from 'node:crypto'

import { OgMemoryClient } from '../lib/client.js'
import { resolveConfig } from '../lib/config.js'

const config = resolveConfig({
  baseUrl: process.env.OGMEM_URL?.trim() || 'http://127.0.0.1:8090',
  apiKeyEnv: 'OGMEM_DSH_API_KEY',
  accountId: process.env.OGMEM_ACCOUNT_ID?.trim() || 'acct-demo',
  userId: process.env.OGMEM_USER_ID?.trim() || 'user-dsh',
  agentId: process.env.OGMEM_AGENT_ID?.trim() || 'deepseek-harness',
  allowInsecureHttp: process.env.OGMEM_ALLOW_INSECURE_HTTP === 'true',
})
const ctx = {
  credentials: {
    resolve: async reference => {
      const value = process.env[reference]?.trim()
      return value ? { value, source: 'environment' } : undefined
    },
  },
}
const client = new OgMemoryClient(ctx, config)
const sessionId = `dsh-acceptance-${randomUUID()}`
const marker = `dsh-wsl-${randomUUID()}`

const write = await client.afterTurn({
  sessionId,
  forceExtract: true,
  messages: [
    { role: 'user', content: `Remember acceptance marker ${marker}.` },
    { role: 'assistant', content: 'Marker recorded for the integration check.' },
  ],
})
const slots = [
  'identityContext',
  'episodicContext',
  'sessionContext',
  'taskContext',
  'retrievedEvidence',
]
const deadline = Date.now() + 60_000
let recall
let markerRecovered = false
do {
  recall = await client.compose({
    sessionId,
    prompt: 'What acceptance marker was just recorded?',
    tokenBudget: 8_000,
  })
  markerRecovered = slots.some(key => String(recall[key] ?? '').includes(marker))
  if (!markerRecovered) await new Promise(resolve => setTimeout(resolve, 2_000))
} while (!markerRecovered && Date.now() < deadline)

console.log(JSON.stringify({
  ok: write?.ok !== false,
  sessionId,
  writeStatus: write?.status ?? null,
  markerRecovered,
  composeSlots: Object.fromEntries(slots.map(key => [key, String(recall[key] ?? '').length])),
}, null, 2))
if (!markerRecovered) process.exitCode = 1
