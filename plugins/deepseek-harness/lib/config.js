const LOOPBACK_HOSTS = new Set(['127.0.0.1', 'localhost', '::1'])
const ENV_REF = /^[A-Za-z_][A-Za-z0-9_]*$/

export const DEFAULT_CONFIG = Object.freeze({
  baseUrl: 'http://127.0.0.1:8090',
  apiKeyEnv: 'OGMEM_DSH_API_KEY',
  accountId: 'acct-demo',
  userId: 'user-dsh',
  agentId: 'deepseek-harness',
  allowInsecureHttp: false,
  autoCapture: true,
  autoRecall: true,
  composeEnabled: true,
  writebackEnabled: true,
  disposeEnabled: true,
  toolsEnabled: true,
  writeToolsEnabled: false,
  deleteToolsEnabled: false,
  captureToolResults: false,
  traceEnabled: true,
  traceCaptureContent: true,
  traceTimeoutMs: 3_000,
  forceExtract: true,
  composeTimeoutMs: 15_000,
  afterTurnTimeoutMs: 300_000,
  disposeTimeoutMs: 3_000,
  shutdownTimeoutMs: 5_000,
  toolTimeoutMs: 30_000,
  pendingDir: '',
  tokenBudget: 32_000,
  maxPromptChars: 12_000,
  maxContextChars: 32_000,
  maxMessageChars: 12_000,
  maxToolResultChars: 4_000,
  maxTurnMessages: 128,
  writebackMaxAttempts: 3,
  writebackInitialDelayMs: 500,
  writebackMaxDelayMs: 5_000,
})

export function resolveConfig(input = {}) {
  if (!isRecord(input)) throw new TypeError('ogmemory: config must be an object')
  const config = { ...DEFAULT_CONFIG, ...input }
  for (const [key, envName] of [
    ['autoCapture', 'OGMEM_AUTO_CAPTURE'],
    ['autoRecall', 'OGMEM_AUTO_RECALL'],
  ]) {
    if (process.env[envName] !== undefined) {
      config[key] = ['1', 'true', 'yes'].includes(process.env[envName].trim().toLowerCase())
    }
  }
  config.baseUrl = normalizeBaseUrl(config.baseUrl, config.allowInsecureHttp)
  config.apiKeyEnv = optionalString(config.apiKeyEnv, 'apiKeyEnv')
  if (config.apiKeyEnv && !ENV_REF.test(config.apiKeyEnv)) {
    throw new TypeError('ogmemory: apiKeyEnv must be an environment-variable name')
  }
  for (const key of ['accountId', 'userId', 'agentId']) {
    config[key] = requiredString(config[key], key, 128)
  }
  for (const key of [
    'allowInsecureHttp',
    'autoCapture',
    'autoRecall',
    'composeEnabled',
    'writebackEnabled',
    'disposeEnabled',
    'toolsEnabled',
    'writeToolsEnabled',
    'deleteToolsEnabled',
    'captureToolResults',
    'traceEnabled',
    'traceCaptureContent',
    'forceExtract',
  ]) {
    if (typeof config[key] !== 'boolean') throw new TypeError(`ogmemory: ${key} must be boolean`)
  }
  config.pendingDir = optionalString(config.pendingDir, 'pendingDir')
  for (const key of [
    'composeTimeoutMs',
    'afterTurnTimeoutMs',
    'disposeTimeoutMs',
    'shutdownTimeoutMs',
    'toolTimeoutMs',
    'traceTimeoutMs',
    'tokenBudget',
    'maxPromptChars',
    'maxContextChars',
    'maxMessageChars',
    'maxToolResultChars',
    'maxTurnMessages',
    'writebackMaxAttempts',
    'writebackInitialDelayMs',
    'writebackMaxDelayMs',
  ]) {
    config[key] = positiveSafeInteger(config[key], key)
  }
  if (config.writebackInitialDelayMs > config.writebackMaxDelayMs) {
    throw new TypeError('ogmemory: writebackInitialDelayMs must not exceed writebackMaxDelayMs')
  }
  return Object.freeze(config)
}

function normalizeBaseUrl(value, allowInsecureHttp) {
  const text = requiredString(value, 'baseUrl', 2048).replace(/\/+$/, '')
  let url
  try {
    url = new URL(text)
  } catch {
    throw new TypeError('ogmemory: baseUrl must be an absolute HTTP(S) URL')
  }
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash) {
    throw new TypeError('ogmemory: baseUrl must be a credential-free HTTP(S) origin or path')
  }
  if (url.protocol === 'http:' && !LOOPBACK_HOSTS.has(url.hostname) && allowInsecureHttp !== true) {
    throw new TypeError(
      'ogmemory: non-loopback HTTP requires allowInsecureHttp=true; prefer HTTPS outside localhost',
    )
  }
  return text
}

function requiredString(value, key, maxLength) {
  if (typeof value !== 'string' || value.trim().length === 0 || value.length > maxLength) {
    throw new TypeError(`ogmemory: ${key} must be a non-empty string up to ${maxLength} characters`)
  }
  return value.trim()
}

function optionalString(value, key) {
  if (value === undefined || value === null || value === '') return ''
  if (typeof value !== 'string') throw new TypeError(`ogmemory: ${key} must be a string`)
  return value.trim()
}

function positiveSafeInteger(value, key) {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new TypeError(`ogmemory: ${key} must be a positive safe integer`)
  }
  return value
}

function isRecord(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}
