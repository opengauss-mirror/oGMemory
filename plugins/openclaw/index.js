import { createHash } from "node:crypto";
import { spawn } from "node:child_process";
import { writeFileSync, mkdirSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import {
  waitForHealth,
  waitForHealthOrExit,
  quickHealthCheck,
  prepareLocalPort,
  findGoBinary,
  findPythonBin,
} from "./process.js";
import { createSessionAgentResolver } from "./session-agent.js";
import { appendTakeoverCompactionBoundary } from "./compaction-boundary.js";
import {
  isOpenClawHeartbeatMessage,
  registerOpenClawTraceHooks,
} from "./trace.js";

export { appendTakeoverCompactionBoundary } from "./compaction-boundary.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = dirname(dirname(__dirname));

// ---------------------------------------------------------------------------
// Service defaults
// ---------------------------------------------------------------------------

function resolveBoolean(value, fallback) {
  if (typeof value === "boolean") return value;
  if (typeof value === "string") {
    const lowered = value.trim().toLowerCase();
    if (["1", "true", "yes", "on"].includes(lowered)) return true;
    if (["0", "false", "no", "off"].includes(lowered)) return false;
  }
  return fallback;
}

function resolveInteger(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function resolveJsonObject(value, fallback = {}) {
  if (value && typeof value === "object" && !Array.isArray(value)) return value;
  if (typeof value === "string" && value.trim()) {
    try {
      const parsed = JSON.parse(value);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) return parsed;
    } catch {}
  }
  return fallback;
}

export function resolveConfig(pluginConfig = {}) {
  const resolved = {
    mode: pluginConfig.mode || process.env.OGMEM_MODE || "remote",
    // AGFS
    agfsServerDir:
      pluginConfig.agfsServerDir ||
      process.env.AGFS_SERVER_DIR ||
      join(PROJECT_ROOT, "agfs"),
    agfsPort: pluginConfig.agfsPort || Number(process.env.AGFS_PORT) || 1833,
    agfsDataDir:
      pluginConfig.agfsDataDir ||
      process.env.AGFS_DATA_DIR ||
      join(PROJECT_ROOT, "..", "data"),
    // ContextEngine
    contextEnginePort:
      pluginConfig.contextEnginePort || Number(process.env.CONTEXTENGINE_PORT) || 8090,
    projectRoot: pluginConfig.projectRoot || process.env.CONTEXTENGINE_PROJECT_ROOT || PROJECT_ROOT,
    // HTTP mode (remote or locally-spawned)
    memoryApiBaseUrl:
      pluginConfig.memoryApiBaseUrl || process.env.OGMEM_API_URL || "http://127.0.0.1:8090",
    // Multi-tenant auth
    authApiKey: pluginConfig.authApiKey || process.env.OG_AUTH_API_KEY || "",
    authAccountId: pluginConfig.authAccountId || process.env.OG_AUTH_ACCOUNT_ID || "",
    // Timing
    startupTimeoutMs: pluginConfig.startupTimeoutMs || 60_000,
    traceTimeoutMs: resolveInteger(pluginConfig.traceTimeoutMs, 3000),
    traceCaptureContent: pluginConfig.traceCaptureContent !== false,
    commitTokenThreshold: pluginConfig.commitTokenThreshold ?? 0,
    autoCapture: resolveBoolean(pluginConfig.autoCapture ?? process.env.OGMEM_AUTO_CAPTURE, true),
    autoRecall: resolveBoolean(pluginConfig.autoRecall ?? process.env.OGMEM_AUTO_RECALL, true),
    prefetchEnabled: resolveBoolean(
      pluginConfig.prefetchEnabled ?? process.env.OGMEM_PREFETCH_ENABLED,
      false,
    ),
    prefetchTopK: resolveInteger(
      pluginConfig.prefetchTopK ?? process.env.OGMEM_PREFETCH_TOP_K,
      5,
    ),
    // Identity
    accountId: pluginConfig.accountId || process.env.OGMEM_ACCOUNT_ID || "",
    userId: pluginConfig.userId || process.env.OGMEM_USER_ID || "",
    agentId: pluginConfig.agentId || process.env.OGMEM_AGENT_ID || "",
    // LLM (passed as env vars to CE server)
    llmProvider: pluginConfig.llmProvider || process.env.CONTEXTENGINE_PROVIDER || "mock",
    llmApiKey:
      pluginConfig.llmApiKey || process.env.OGMEM_API_KEY || process.env.OPENAI_API_KEY || "",
    llmBaseUrl:
      pluginConfig.llmBaseUrl || process.env.OGMEM_BASE_URL || process.env.OPENAI_BASE_URL || "",
    llmModel:
      pluginConfig.llmModel || process.env.OGMEM_LLM_MODEL || process.env.OPENAI_LLM_MODEL || "gpt-4o-mini",
    llmReasoningEffort:
      pluginConfig.llmReasoningEffort || process.env.LLM_REASONING_EFFORT || "",
    llmIsReasoningModel:
      pluginConfig.llmIsReasoningModel ??
      (process.env.LLM_IS_REASONING_MODEL === undefined
        ? null
        : resolveBoolean(process.env.LLM_IS_REASONING_MODEL, null)),
    llmExtraBody: resolveJsonObject(
      pluginConfig.llmExtraBody ?? process.env.LLM_EXTRA_BODY,
    ),
    // Embedding (defaults to LLM settings if not specified)
    embeddingProvider: pluginConfig.embeddingProvider || process.env.EMBEDDING_PROVIDER || "",
    embeddingApiKey:
      pluginConfig.embeddingApiKey ||
      process.env.OGMEM_EMBEDDING_API_KEY ||
      process.env.OPENAI_EMBEDDING_API_KEY ||
      "",
    embeddingBaseUrl:
      pluginConfig.embeddingBaseUrl ||
      process.env.OGMEM_EMBEDDING_BASE_URL ||
      process.env.OPENAI_EMBEDDING_BASE_URL ||
      "",
    embeddingModel:
      pluginConfig.embeddingModel ||
      process.env.OGMEM_EMBEDDING_MODEL ||
      process.env.OPENAI_EMBEDDING_MODEL ||
      "text-embedding-ada-002",
    compactTakeoverEnabled: resolveBoolean(
      pluginConfig.compactTakeoverEnabled ?? process.env.OGMEM_COMPACT_TAKEOVER_ENABLED,
      true,
    ),
    summaryMaxChars: resolveInteger(
      pluginConfig.summaryMaxChars ?? process.env.OGMEM_SUMMARY_MAX_CHARS,
      4000,
    ),
    shortTermIndexMode: ["sync", "async", "off"].includes(pluginConfig.shortTermIndexMode)
      ? pluginConfig.shortTermIndexMode
      : ["sync", "async", "off"].includes(process.env.OGMEM_SHORT_TERM_INDEX_MODE)
        ? process.env.OGMEM_SHORT_TERM_INDEX_MODE
        : "sync",
  };
  resolved.modelConfigOverrides = Object.fromEntries(
    [
      "llmProvider",
      "llmApiKey",
      "llmBaseUrl",
      "llmModel",
      "llmReasoningEffort",
      "llmIsReasoningModel",
      "llmExtraBody",
      "embeddingProvider",
      "embeddingApiKey",
      "embeddingBaseUrl",
      "embeddingModel",
    ]
      .filter(
        (key) =>
          Object.prototype.hasOwnProperty.call(pluginConfig, key) &&
          (key === "llmIsReasoningModel" ? typeof pluginConfig[key] === "boolean" : Boolean(pluginConfig[key])),
      )
      .map((key) => [key, pluginConfig[key]]),
  );
  return resolved;
}

// ---------------------------------------------------------------------------
// Service spawners
// ---------------------------------------------------------------------------

function spawnAgfs(cfg, logger) {
  const goCheck = findGoBinary();
  if (!goCheck.ok) {
    throw new Error(
      "Go is not installed. Install Go 1.22+ to run AGFS, or use mode=remote.\n" +
        "  → https://go.dev/dl/",
    );
  }
  logger.info(`agfs: using ${goCheck.version}`);

  // Ensure data dir exists
  mkdirSync(cfg.agfsDataDir, { recursive: true });

  const child = spawn("go", ["run", "cmd/server/main.go", "-addr", `:${cfg.agfsPort}`], {
    cwd: cfg.agfsServerDir,
    env: { ...process.env },
    stdio: ["ignore", "pipe", "pipe"],
  });

  child.stdout?.on("data", (d) => logger.debug?.(`[agfs:stdout] ${String(d).trim()}`));
  child.stderr?.on("data", (d) => logger.debug?.(`[agfs:stderr] ${String(d).trim()}`));
  child.on("exit", (code, sig) => {
    if (!child.killed) logger.warn(`agfs: exited unexpectedly (code=${code}, signal=${sig})`);
  });

  return child;
}

export function spawnContextEngine(cfg, logger, deps = {}) {
  const pythonBin = (deps.findPython || findPythonBin)(cfg.projectRoot);
  logger.info(`ce: using python=${pythonBin}`);

  const spawnFn = deps.spawnFn || spawn;
  const child = spawnFn(pythonBin, ["server/app.py"], {
    cwd: cfg.projectRoot,
    env: {
      ...process.env,
      PYTHONUNBUFFERED: "1",
      AGFS_PORT: String(cfg.agfsPort),
      AGFS_BASE_URL: `http://127.0.0.1:${cfg.agfsPort}`,
      OGMEM_CONFIG: join(cfg.projectRoot, "config", "ogmem.yaml"),
      OGMEM_MODEL_CONFIG_OVERRIDES: JSON.stringify(cfg.modelConfigOverrides || {}),
      // LLM config
      CONTEXTENGINE_PROVIDER: cfg.llmProvider,
      OGMEM_API_KEY: cfg.llmApiKey,
      OGMEM_BASE_URL: cfg.llmBaseUrl,
      OGMEM_LLM_MODEL: cfg.llmModel,
      LLM_REASONING_EFFORT: cfg.llmReasoningEffort,
      LLM_IS_REASONING_MODEL:
        cfg.llmIsReasoningModel === null ? "" : String(cfg.llmIsReasoningModel),
      LLM_EXTRA_BODY: JSON.stringify(cfg.llmExtraBody || {}),
      OPENAI_API_KEY: cfg.llmApiKey,
      OPENAI_BASE_URL: cfg.llmBaseUrl,
      OPENAI_LLM_MODEL: cfg.llmModel,
      // Embedding config (fallback to LLM settings)
      EMBEDDING_PROVIDER: cfg.embeddingProvider || cfg.llmProvider,
      OGMEM_EMBEDDING_API_KEY: cfg.embeddingApiKey || cfg.llmApiKey,
      OGMEM_EMBEDDING_BASE_URL: cfg.embeddingBaseUrl || cfg.llmBaseUrl,
      OGMEM_EMBEDDING_MODEL: cfg.embeddingModel,
      OPENAI_EMBEDDING_API_KEY: cfg.embeddingApiKey || cfg.llmApiKey,
      OPENAI_EMBEDDING_BASE_URL: cfg.embeddingBaseUrl || cfg.llmBaseUrl,
      OPENAI_EMBEDDING_MODEL: cfg.embeddingModel,
      OGMEM_PREFETCH_ENABLED: String(Boolean(cfg.prefetchEnabled)),
      OGMEM_PREFETCH_TOP_K: String(cfg.prefetchTopK),
    },
    stdio: ["ignore", "pipe", "pipe"],
  });

  child.stdout?.on("data", (d) => logger.debug?.(`[ce:stdout] ${String(d).trim()}`));
  child.stderr?.on("data", (d) => logger.debug?.(`[ce:stderr] ${String(d).trim()}`));
  child.on("exit", (code, sig) => {
    if (!child.killed) logger.warn(`ce: exited unexpectedly (code=${code}, signal=${sig})`);
  });

  return child;
}

// ---------------------------------------------------------------------------
// Transport: HTTP
// ---------------------------------------------------------------------------

function buildAuthHeaders(authCfg = {}, params = {}, contentType = null) {
  const headers = {};
  if (contentType) headers["Content-Type"] = contentType;
  if (authCfg.authApiKey) headers["X-API-Key"] = authCfg.authApiKey;
  if (authCfg.authAccountId) headers["X-Account-ID"] = authCfg.authAccountId;

  const userId = typeof params?.userId === "string" ? params.userId.trim() : "";
  if (userId) headers["X-User-ID"] = userId;
  if (params?.agentId) headers["X-Agent-ID"] = params.agentId;

  return headers;
}

function degradedSourceArchived(result) {
  const sourceIds = result?.source_message_ids;
  const archivedIds = result?.archived_source_ids;
  if (!Array.isArray(sourceIds) || !sourceIds.length || !Array.isArray(archivedIds)) return false;
  const archived = new Set(archivedIds);
  return Boolean(result.request_status === "DEGRADED" && !(result.httpStatus >= 400)
    && result.semantic_status === "DEGRADED"
    && ["ARCHIVED", "RAW_ARCHIVED"].includes(result.source_status)
    && result.archive_id && new Set(sourceIds).size === archived.size
    && sourceIds.every((id) => archived.has(id)));
}

function degradedWriteSettled(result) {
  return degradedSourceArchived(result) && ["DONE", "NOT_REQUIRED"].includes(result.projection_status);
}

function writeRequestNeedsPoll(result) {
  return result.retry_action === "poll"
    || (degradedSourceArchived(result) && result.projection_status === "PENDING");
}

function writeReceiptSummary(result) {
  return Object.fromEntries([
    "request_id", "status", "request_status", "source_status", "semantic_status",
    "projection_status", "retry_action", "error_code", "error_type", "archive_id",
  ].filter((key) => result[key] != null).map((key) => [key, result[key]]));
}

function writeHttpObservation(method, path, response, result, started) {
  return {
    method, path, http_status: response.status, at: new Date().toISOString(),
    duration_ms: performance.now() - started, ...writeReceiptSummary(result),
  };
}

export async function pollWriteRequest(baseUrl, params, cfg, initial, deadline) {
  const requestId = initial.request_id || params.clientRequestId;
  const sessionId = params.sessionId || params.session_id;
  if (!requestId || !sessionId) return initial;
  const path = `/api/v1/sessions/${encodeURIComponent(sessionId)}`
    + `/write-requests?clientRequestId=${encodeURIComponent(requestId)}`;
  const observations = [...(initial.http_observations || [])];
  let result = initial;
  let interruption = { observation_status: "OBSERVATION_TIMEOUT" };
  while (writeRequestNeedsPoll(result) && performance.now() + 1000 < deadline) {
    await new Promise((resolve) => setTimeout(resolve, 1000));
    const started = performance.now();
    try {
      const response = await fetch(`${baseUrl}${path}`, {
        headers: buildAuthHeaders(cfg, params, "application/json"),
        signal: AbortSignal.timeout(Math.max(1, Math.floor(Math.min(30_000, deadline - performance.now())))),
      });
      const outcome = await response.json();
      observations.push(writeHttpObservation("GET", path, response, outcome, started));
      if (!response.ok && !["FAILED", "NEEDS_CONFIRMATION"].includes(outcome.request_status)) {
        interruption = { observation_status: "OBSERVATION_HTTP_ERROR", observation_http_status: response.status };
        break;
      }
      if (outcome.request_status === "DEGRADED" && outcome.request_id !== requestId) {
        interruption = { observation_status: "OBSERVATION_RECEIPT_MISMATCH" };
        break;
      }
      const state = outcome.request_status;
      const handled = state === "SUCCEEDED" || (state === "ACCEPTED" && outcome.retry_action === "none");
      result = state === "DEGRADED"
        ? { compacted: initial.compacted, result: initial.result, ...outcome, ok: false }
        : { ...initial, ...outcome, ok: handled && outcome.ok !== false };
    } catch (error) {
      observations.push({ method: "GET", path, request_id: requestId, at: new Date().toISOString(),
        duration_ms: performance.now() - started, http_status: null, error_type: error.name });
      interruption = {
        observation_status: error.name === "TimeoutError" ? "OBSERVATION_TIMEOUT" : "OBSERVATION_ERROR",
        observation_error_type: error.name,
      };
      break;
    }
  }
  const polling = writeRequestNeedsPoll(result);
  const { observation_status, observation_error_type, observation_http_status, ...receipt } = result;
  return {
    ...receipt,
    ok: !polling && result.ok,
    retryable: result.retry_action === "retry_same_request",
    http_observations: observations,
    ...(polling ? { retry_action: "poll", ...interruption } : {}),
  };
}

export async function callHttp(baseUrl, method, params, cfg = {}) {
  const timeoutMs = cfg.requestTimeoutMs ?? (method === "after_turn" ? 300_000 : 120_000);
  const deadline = performance.now() + timeoutMs;
  if (method === "after_turn" && !params.clientRequestId) {
    const source = JSON.stringify([params.accountId || cfg.authAccountId, params.userId,
      params.agentId, params.sessionId, params.semanticTurnId,
      params.semanticTurnSourceNamespace, params.runtimeContext?.currentMessageId,
      params.prePromptMessageCount || 0, params.messages]);
    params = { ...params, clientRequestId: "oc-" + createHash("sha256").update(source).digest("hex") };
  }
  try {
    const headers = buildAuthHeaders(cfg, params, "application/json");
    const resp = await fetch(`${baseUrl}/api/v1/${method}`, {
      method: "POST",
      headers,
      body: JSON.stringify(params),
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (!resp.ok) {
      const body = await resp.text();
      let parsed = {};
      let parsedBody = false;
      try {
        parsed = body ? JSON.parse(body) : {};
        parsedBody = true;
      } catch {
        parsed = {};
      }
      if (method === "after_turn") {
        parsed.http_observations = [writeHttpObservation("POST", `/api/v1/${method}`, resp, parsed, deadline - timeoutMs)];
      }
      if (method === "after_turn" && parsed.retry_action === "poll") {
        return pollWriteRequest(baseUrl, params, cfg, parsed, deadline);
      }
      const safeErrorBody = parsedBody && parsed && typeof parsed === "object"
        ? writeReceiptSummary(parsed)
        : null;
      console.error(
        `[og-memory] HTTP ${resp.status} from ${method}, body: ${safeErrorBody
          ? JSON.stringify(safeErrorBody)
          : `<non-json response, ${body.length} chars>`}`,
      );
      if (cfg.structuredFailures === true) {
        const retryableSessionConflict = Boolean(
          resp.status === 409
          && parsed?.error_code === "session_operation_in_progress"
          && parsed?.conflict_type === "session_replaced"
          && parsed?.retryable === true
        );
        return {
          ...parsed,
          ok: false,
          retryable: parsed?.retry_action ? parsed.retry_action === "retry_same_request" : (
            (
              retryableSessionConflict
              || [408, 425, 429].includes(resp.status)
              || resp.status >= 500
            )
            && parsed?.retryable !== false
          ),
          httpStatus: resp.status,
        };
      }
      return null;
    }
    const result = await resp.json();
    if (method === "after_turn") {
      result.http_observations = [writeHttpObservation("POST", `/api/v1/${method}`, resp, result, deadline - timeoutMs)];
      if (["DEGRADED", "FAILED", "NEEDS_CONFIRMATION"].includes(result.request_status)) result.ok = false;
      if (result.retry_action) result.retryable = result.retry_action === "retry_same_request";
    }
    return (method === "after_turn" || method === "compact") && writeRequestNeedsPoll(result)
      ? pollWriteRequest(baseUrl, params, cfg, result, deadline) : result;
  } catch (err) {
    console.error(`[og-memory] HTTP ${method} error: ${err.name}`);
    if (cfg.structuredFailures === true) {
      return {
        ok: false,
        retryable: true,
        status: "transport_error",
        error_type: err.name,
        request_id: params.clientRequestId,
        observation_status: err.name === "TimeoutError" ? "OBSERVATION_TIMEOUT" : "OBSERVATION_ERROR",
      };
    }
    return null;
  }
}

const COMPOSE_MAX_ATTEMPTS = 3;
const COMPOSE_RETRY_DELAY_MS = 250;
const COMPOSE_TIMEOUT_MS = 120_000;

async function composeWithRetry(call, params) {
  const deadline = performance.now() + COMPOSE_TIMEOUT_MS;
  let result;
  for (let attempt = 1; attempt <= COMPOSE_MAX_ATTEMPTS; attempt += 1) {
    const remainingMs = Math.floor(deadline - performance.now());
    if (remainingMs <= 0) break;
    result = await call("compose", params, remainingMs);
    if (result?.ok !== false && result?.status !== "processing" && Array.isArray(result?.messages)) {
      return result;
    }
    const processing = result?.status === "processing"
      && result?.retryable === true
      && result?.httpStatus !== 429;
    if (!processing || attempt === COMPOSE_MAX_ATTEMPTS) break;
    const delayMs = Math.min(COMPOSE_RETRY_DELAY_MS * attempt, deadline - performance.now());
    if (delayMs <= 0) break;
    await new Promise((resolve) => setTimeout(resolve, delayMs));
  }
  console.error(
    `[og-memory] compose unavailable: ${result?.error_code || result?.status || result?.httpStatus || "invalid_context"}`,
  );
  return null;
}

const DISPOSE_MAX_ATTEMPTS = 3;
const DISPOSE_RETRY_DELAY_MS = 250;

async function disposeWithRetry(call, params) {
  for (let attempt = 1; attempt <= DISPOSE_MAX_ATTEMPTS; attempt += 1) {
    const result = await call("dispose", params);
    if (result?.disposed === true && result.reason !== "thread_unavailable") {
      return result;
    }

    if (attempt < DISPOSE_MAX_ATTEMPTS) {
      await new Promise((resolve) => {
        setTimeout(resolve, DISPOSE_RETRY_DELAY_MS * attempt);
      });
    }
  }

  throw new Error(
    `[og-memory] dispose failed after ${DISPOSE_MAX_ATTEMPTS} attempts`,
  );
}

export async function loadRuntimeCompactionDelegate() {
  for (const specifier of ["openclaw/plugin-sdk", "openclaw/plugin-sdk/core"]) {
    try {
      const mod = await import(specifier);
      if (typeof mod.delegateCompactionToRuntime === "function") {
        return mod.delegateCompactionToRuntime;
      }
    } catch {
      // The plugin can still run its preflight JS path when the runtime SDK is absent.
    }
  }
  return null;
}

function buildCompactionParams(params, cfg) {
  return {
    ...params,
    accountId: params?.accountId || cfg.accountId,
    userId: params?.userId || cfg.userId,
    agentId: params?.agentId || cfg.agentId,
    summaryMaxChars: cfg.summaryMaxChars,
    shortTermIndexMode: cfg.shortTermIndexMode,
  };
}

function requireSuccessfulTransportResult(method, result) {
  if (!result) {
    throw new Error(`og-memory ${method} failed: empty transport response`);
  }
  if (result.error && !degradedWriteSettled(result)) {
    throw new Error(`og-memory ${method} failed: ${result.error}`);
  }
  return result;
}

export async function prepareCompactionWithTransport({ call, params }) {
  const result = await call("prepare_compaction", params);
  return requireSuccessfulTransportResult("prepare_compaction", result);
}

export async function delegateCompactionToRuntimeOrThrow({
  params,
  runtimeDelegate,
  loadDelegate = loadRuntimeCompactionDelegate,
}) {
  const delegate = runtimeDelegate ?? (await loadDelegate());
  if (typeof delegate === "function") {
    const result = await delegate(params);
    if (!result) {
      throw new Error("delegateCompactionToRuntime failed: empty runtime response");
    }
    return result;
  }
  throw new Error("delegateCompactionToRuntime unavailable");
}

export async function runCompactControlFlow({
  cfg,
  params,
  call,
  runtimeDelegate,
  loadDelegate = loadRuntimeCompactionDelegate,
  persistTakeoverCompactionBoundary = appendTakeoverCompactionBoundary,
}) {
  const enhancedParams = buildCompactionParams(params, cfg);
  const prepared = await prepareCompactionWithTransport({ call, params: enhancedParams });

  if (!cfg.compactTakeoverEnabled) {
    return delegateCompactionToRuntimeOrThrow({
      params,
      runtimeDelegate,
      loadDelegate,
    });
  }
  if (!prepared.prepareToken) {
    throw new Error("prepare_compaction failed: missing prepare token");
  }

  const result = await call("compact", {
    ...enhancedParams,
    prepareToken: prepared.prepareToken,
  });
  const successfulResult = requireSuccessfulTransportResult("compact", result);
  const degraded = degradedWriteSettled(successfulResult);
  const summary = successfulResult.result?.summary;
  if (successfulResult.compacted && degraded && (typeof summary !== "string" || !summary.trim())) {
    return delegateCompactionToRuntimeOrThrow({ params, runtimeDelegate, loadDelegate });
  }
  if ((successfulResult.ok || degraded) && successfulResult.compacted) {
    const boundary = await persistTakeoverCompactionBoundary({
      sessionFile: params?.sessionFile,
      summary: successfulResult.result?.summary,
      tokensBefore: successfulResult.result?.tokensBefore,
      details: successfulResult.result?.details,
      params,
      upstreamFirstKeptEntryId: successfulResult.result?.firstKeptEntryId,
    });
    if (boundary?.firstKeptEntryId) {
      successfulResult.result = {
        ...(successfulResult.result ?? {}),
        firstKeptEntryId: boundary.firstKeptEntryId,
      };
    }
  }
  return degraded && successfulResult.compacted ? { ...successfulResult, ok: true } : successfulResult;
}

export function buildLayeredComposeMessages(result, fallbackMessages = []) {
  const layeredMessages = [];
  if (result?.identityContext) {
    layeredMessages.push({
      role: "user",
      content: `[User Profile - background, identity, preferences, and stable traits]\n${result.identityContext}`,
    });
  }
  if (result?.episodicContext) {
    layeredMessages.push({
      role: "user",
      content: `[Episodic Memory - past conversations, events, and experiences]\n${result.episodicContext}`,
    });
  }
  layeredMessages.push(
    ...(result?.messages ?? fallbackMessages ?? []).filter((message) => !message?._ogmem),
  );
  if (result?.sessionContext) {
    layeredMessages.push({
      role: "user",
      content: `[Current Session - recent messages and context from this conversation]\n${result.sessionContext}`,
    });
  }
  if (result?.retrievedEvidence) {
    layeredMessages.push({
      role: "user",
      content: `[Retrieved Memories - relevant facts recalled from long-term memory]\n${result.retrievedEvidence}`,
    });
  }
  return layeredMessages;
}

// ---------------------------------------------------------------------------
// Plugin entry
// ---------------------------------------------------------------------------

export default {
  id: "og-memory-context-engine",
  name: "oG-Memory Context Engine",
  description: "oG-Memory integration: semantic search with hierarchical retrieval",
  kind: "context-engine",

  register(api) {
    const cfg = resolveConfig(api.pluginConfig);
    const isLocal = cfg.mode === "local";

    api.logger.info(
      `og-memory: mode=${isLocal ? "local" : "remote"}, url=${cfg.memoryApiBaseUrl}`,
    );

    // =======================================================================
    // Service lifecycle — start/stop AGFS + ContextEngine
    // =======================================================================

    let agfsProcess = null;
    let ceProcess = null;

    api.registerService({
      id: "contextengine-backend",
      start: async () => {
        if (!isLocal) {
          // Remote mode: just verify connectivity
          api.logger.info("og-memory: remote mode — verifying service connectivity...");
          const healthUrl = cfg.memoryApiBaseUrl
            ? `${cfg.memoryApiBaseUrl}/api/v1/health`
            : `http://127.0.0.1:${cfg.contextEnginePort}/api/v1/health`;
          const ceOk = await quickHealthCheck(
            healthUrl,
            5000,
          );
          if (!ceOk) {
            throw new Error(
              `Cannot reach ContextEngine at ${cfg.memoryApiBaseUrl || `http://127.0.0.1:${cfg.contextEnginePort}`}. ` +
              `Ensure the server is running, or switch to mode=local.`,
            );
          }
          api.logger.info("og-memory: remote services verified ✓");
          return;
        }

        // --- Local mode: spawn AGFS then ContextEngine ---

        // Check if AGFS server dir exists
        if (!existsSync(cfg.agfsServerDir)) {
          throw new Error(
            `AGFS server directory not found: ${cfg.agfsServerDir}\n` +
            `  Set agfsServerDir in plugin config or AGFS_SERVER_DIR env var.\n` +
            `  Or check that the agfs/ directory exists in the project.`,
          );
        }

        // 1. Start AGFS
        const agfsPort = await prepareLocalPort(
          cfg.agfsPort,
          (p) => `http://127.0.0.1:${p}/`,
          api.logger,
        );
        api.logger.info(`agfs: spawning on :${agfsPort} (dir: ${cfg.agfsServerDir})`);
        agfsProcess = spawnAgfs({ ...cfg, agfsPort }, api.logger);

        try {
          await waitForHealthOrExit(
            `http://127.0.0.1:${agfsPort}/`,
            cfg.startupTimeoutMs,
            500,
            agfsProcess,
            true, // AGFS returns 404 for / — accept any HTTP response
          );
          api.logger.info(`agfs: ready (port ${agfsPort}, pid ${agfsProcess.pid})`);
        } catch (err) {
          agfsProcess.kill("SIGTERM");
          agfsProcess = null;
          throw new Error(`AGFS failed to start: ${err.message}`);
        }

        // 2. Start ContextEngine
        const cePort = await prepareLocalPort(
          cfg.contextEnginePort,
          (p) => `http://127.0.0.1:${p}/api/v1/health`,
          api.logger,
        );
        const ceCfg = { ...cfg, agfsPort, contextEnginePort: cePort };
        api.logger.info(`ce: spawning on :${cePort} (root: ${cfg.projectRoot})`);
        ceProcess = spawnContextEngine(ceCfg, api.logger);

        try {
          await waitForHealthOrExit(
            `http://127.0.0.1:${cePort}/api/v1/health`,
            cfg.startupTimeoutMs,
            500,
            ceProcess,
          );
          api.logger.info(`ce: ready (port ${cePort}, pid ${ceProcess.pid})`);
        } catch (err) {
          ceProcess.kill("SIGTERM");
          ceProcess = null;
          agfsProcess.kill("SIGTERM");
          agfsProcess = null;
          throw new Error(`ContextEngine failed to start: ${err.message}`);
        }

        // Update port in config if auto-selected
        if (agfsPort !== cfg.agfsPort) cfg.agfsPort = agfsPort;
        if (cePort !== cfg.contextEnginePort) {
          cfg.contextEnginePort = cePort;
          cfg.memoryApiBaseUrl = `http://127.0.0.1:${cePort}`;
        }

        api.logger.info("og-memory: all services ready ✓");
      },

      stop: () => {
        if (ceProcess) {
          ceProcess.kill("SIGTERM");
          api.logger.info(`ce: stopped (pid ${ceProcess.pid})`);
          ceProcess = null;
        }
        if (agfsProcess) {
          agfsProcess.kill("SIGTERM");
          api.logger.info(`agfs: stopped (pid ${agfsProcess.pid})`);
          agfsProcess = null;
        }
        api.logger.info("og-memory: services stopped");
      },
    });

    // =======================================================================
    // Transport helpers
    // =======================================================================

    const ogmemUrl = cfg.memoryApiBaseUrl || (isLocal ? `http://127.0.0.1:${cfg.contextEnginePort}` : "");

    const authCfg = {
      authApiKey: cfg.authApiKey,
      authAccountId: cfg.authAccountId,
    };

    function call(method, params) {
      return callHttp(ogmemUrl, method, params, authCfg);
    }

    function callAsync(method, params, requestTimeoutMs) {
      return callHttp(ogmemUrl, method, params, {
        ...authCfg,
        structuredFailures: true,
        requestTimeoutMs,
      });
    }

    function extractContentText(content) {
      if (typeof content === "string") return content;
      if (Array.isArray(content)) {
        return content
          .map((b) => (typeof b === "string" ? b : b?.text || ""))
          .join(" ")
          .trim();
      }
      return String(content || "");
    }

    function extractUserIdFromTag(content) {
      if (typeof content !== "string") return null;
      const match = content.match(/^\[ogmem-user:\s*([^\]]+)\]/i);
      return match ? match[1].trim() : null;
    }

    function extractUserIdFromMessages(messages) {
      if (!Array.isArray(messages)) return null;
      // Search user messages first
      for (const msg of messages) {
        if (msg?.role !== "user") continue;
        const content = extractContentText(msg.content || "");
        const userId = extractUserIdFromTag(content);
        if (userId) return userId;
      }
      // Fallback: search all messages
      for (const msg of messages) {
        const content = extractContentText(msg?.content || "");
        const userId = extractUserIdFromTag(content);
        if (userId) return userId;
      }
      return null;
    }

    function isHeartbeatContextRun(params) {
      if (params?.isHeartbeat) return true;
      return isOpenClawHeartbeatMessage(params?.message);
    }

    function withoutHeartbeatTurns(messages, prePromptMessageCount) {
      const prefixCount = Number.isInteger(prePromptMessageCount)
        ? Math.max(0, Math.min(prePromptMessageCount, messages.length))
        : 0;
      const filtered = messages.slice(0, prefixCount);
      let skippingHeartbeatTurn = false;
      for (let index = prefixCount; index < messages.length; index += 1) {
        const message = messages[index];
        if (message?.role === "user") {
          skippingHeartbeatTurn = isOpenClawHeartbeatMessage(message);
        }
        if (!skippingHeartbeatTurn) filtered.push(message);
      }
      return { messages: filtered, prefixCount };
    }

    // =======================================================================
    // Context engine registration
    // =======================================================================

    const sessionIdentities = new Map();
    const sessionAgents = createSessionAgentResolver(cfg.agentId);
    const withSessionAgent = (params) => ({
      ...params,
      agentId: sessionAgents.resolve(params),
    });
    registerOpenClawTraceHooks(api, cfg, { sessionAgents });

    api.on("session_end", async (event, ctx) => {
      const sessionId = event?.sessionId || ctx?.sessionId || "";
      if (!sessionId) {
        sessionAgents.forget(event, ctx);
        return;
      }

      const savedIdentity = sessionIdentities.get(sessionId);
      const identity = {
        accountId: cfg.accountId,
        userId: cfg.userId,
        ...savedIdentity,
        agentId: sessionAgents.resolve(event, ctx),
      };
      sessionAgents.forget(event, ctx);
      console.log(
        `[og-memory] session_end dispose called, sessionId=${sessionId}, reason=${event?.reason || "unknown"}`,
      );
      try {
        const result = await disposeWithRetry(call, { sessionId, ...identity });
        return result;
      } catch (err) {
        console.error(
          `[og-memory] session_end dispose failed, sessionId=${sessionId}: ${err.message}`,
        );
      } finally {
        sessionIdentities.delete(sessionId);
      }
    });

    api.registerContextEngine("og-memory-context-engine", () => {
      const pendingWriteReceipts = new Map();
      const afterTurnChains = new Map();
      const submitAfterTurn = async (params) => {
        const sessionKey = params.sessionId || params.sessionKey;
        const previous = afterTurnChains.get(sessionKey) || Promise.resolve();
        const run = previous.then(async () => {
          const pending = pendingWriteReceipts.get(sessionKey);
          if (pending) {
            const receipt = await pollWriteRequest(
              ogmemUrl, pending.params, authCfg, pending.receipt,
              performance.now() + (cfg.requestTimeoutMs ?? 300_000),
            );
            if (writeRequestNeedsPoll(receipt)) {
              pending.receipt = receipt;
              return receipt;
            }
            pendingWriteReceipts.delete(sessionKey);
          }
          const result = await callAsync("after_turn", params);
          if (result && writeRequestNeedsPoll(result)) {
            pendingWriteReceipts.set(sessionKey, { params, receipt: result });
          }
          if (result && ["DEGRADED", "FAILED", "NEEDS_CONFIRMATION", "UNKNOWN"].includes(result.request_status)) {
            console.error(`[og-memory] after_turn receipt: ${JSON.stringify(writeReceiptSummary(result))}`);
          }
          return result;
        });
        const tail = run.catch(() => null);
        afterTurnChains.set(sessionKey, tail);
        void tail.then(() => {
          if (afterTurnChains.get(sessionKey) === tail) afterTurnChains.delete(sessionKey);
        });
        return run;
      };
      return ({
      info: {
        id: "og-memory-context-engine",
        name: "oG-Memory Context Engine",
        version: "1.0.0",
        ownsCompaction: cfg.compactTakeoverEnabled,
      },

      async ingest(params) {
        params = withSessionAgent(params);
        if (!cfg.autoCapture) return { ingested: false };
        if (isHeartbeatContextRun(params)) return { ingested: false };
        return call("ingest", params);
      },

      async assemble(params) {
        params = withSessionAgent(params);
        const messages = params?.messages || [];
        if (!cfg.autoRecall) {
          return { messages, estimatedTokens: 0, systemPromptAddition: "" };
        }

        const taggedUserId = extractUserIdFromMessages(messages);
        if (taggedUserId) {
          console.log(`[og-memory] assemble: extracted userId from tag: ${taggedUserId}`);
        }
        const enhancedParams = {
          ...params,
          userId: params?.userId || cfg.userId || taggedUserId,
          accountId: params?.accountId || cfg.accountId,
          agentId: params?.agentId || cfg.agentId,
        };

        if (cfg.prefetchEnabled) {
          try {
            await call("prefetch", {
              ...enhancedParams,
              prefetchTopK: cfg.prefetchTopK,
            });
          } catch (err) {
            console.error(`[og-memory] prefetch skipped: ${err.message}`);
          }
        }

        const result = await composeWithRetry(callAsync, enhancedParams);

        console.log(`[og-memory] compose result keys: ${Object.keys(result || {}).join(',')}`);
        console.log(`[og-memory] assemble: identity=${(result?.identityContext||'').length}b, episodic=${(result?.episodicContext||'').length}b, session=${(result?.sessionContext||'').length}b, ws=${(result?.retrievedEvidence||'').length}b, msgs=${(result?.messages||[]).length}`);

        // Handle null result from callHttp errors
        if (!result) {
          console.error(`[og-memory] compose returned null - check CE server logs`);
          return {
            messages: params.messages || [],
            estimatedTokens: 0,
            systemPromptAddition: "",
          };
        }

        // Full compose snapshot for tests/e2e/eval.py (one JSON per session key, every assemble)
        // ALWAYS write snapshot, even if empty, for debugging
        try {
          const wsDir = "/tmp/ogmem_ws";
          mkdirSync(wsDir, { recursive: true });
          const wsKey = params?.sessionKey || params?.sessionId || "unknown";
          const composeSnapshot = {
            estimatedTokens: result.estimatedTokens ?? 0,
            archiveCount: result.archiveCount ?? 0,
            archiveIncluded: result.archiveIncluded ?? false,
            stats: result.stats ?? {},
            budgetUsedBySlot: result.budgetUsedBySlot ?? {},
            openLoops: result.openLoops ?? [],
            uncertainties: result.uncertainties ?? [],
            identityContext: result.identityContext ?? "",
            episodicContext: result.episodicContext ?? "",
            sessionContext: result.sessionContext ?? "",
            taskContext: result.taskContext ?? "",
            retrievedEvidence: result.retrievedEvidence ?? "",
            composeMessagesCount: Array.isArray(result.messages) ? result.messages.length : 0,
            systemPromptAddition: result.systemPromptAddition ?? "",
            systemPromptSuffix: result.systemPromptSuffix ?? "",
            memoryUserMessage: result.memoryUserMessage ?? "",
          };
          const snapshotPath = `${wsDir}/${wsKey}.compose.json`;
          writeFileSync(snapshotPath, JSON.stringify(composeSnapshot, null, 2), "utf-8");
          console.log(`[og-memory] Wrote compose snapshot: ${snapshotPath}`);
        } catch (e) {
          console.error(`[og-memory] Failed to write snapshot: ${e.message}`);
        }

        // All context layers as "user" role messages for KV-cache efficiency.
        // systemPromptAddition is empty — profile in user msg avoids invalidating
        // the stable system prompt KV-cache when profile updates.
        // OpenClaw's convertMessagesToInputItems drops "system" role messages,
        // so context must use "user" role to reach the LLM.

        // Build layered messages for KV-cache efficiency:
        // Stable prefix (profile, episodic) -> conversation history -> dynamic suffix (session, WS)
        // This maximizes KV-cache reuse: stable prefix + conversation prefix cached across turns.
        const layeredMessages = buildLayeredComposeMessages(result, params.messages ?? []);

        console.log(`[og-memory] assemble return: ${layeredMessages.length} msgs, ws_injected=${!!result.retrievedEvidence}, first_3_roles=${layeredMessages.slice(0,3).map(m=>m.role).join(',')}`);
        console.log(`[og-memory] assemble params keys: ${Object.keys(params || {}).join(',')}, sessionKey=${params?.sessionKey}, sessionId=${params?.sessionId}`);

        // Write WS to temp file for eval.py (backward compat: working-set text only)
        // ALWAYS write WS file, even if empty, for debugging
        try {
          const wsDir = "/tmp/ogmem_ws";
          mkdirSync(wsDir, { recursive: true });
          const wsKey = params?.sessionKey || params?.sessionId || "unknown";
          const wsPath = `${wsDir}/${wsKey}.txt`;
          writeFileSync(wsPath, result.retrievedEvidence || "", "utf-8");
          if (result.retrievedEvidence) {
            console.log(`[og-memory] Wrote working set: ${wsPath} (${result.retrievedEvidence.length} chars)`);
          } else {
            console.log(`[og-memory] Wrote empty working set: ${wsPath}`);
          }
        } catch (e) {
          console.error(`[og-memory] Failed to write working set: ${e.message}`);
        }

        return {
          messages: layeredMessages,
          estimatedTokens: result.estimatedTokens ?? 0,
          systemPromptAddition: "",
        };
      },

      async compact(params) {
        params = withSessionAgent(params);
        return runCompactControlFlow({
          cfg,
          params,
          call,
        });
      },

      async afterTurn(params) {
        params = withSessionAgent(params);
        if (isHeartbeatContextRun(params)) return;
        const messages = params?.messages || [];
        const filtered = withoutHeartbeatTurns(
          messages,
          params?.prePromptMessageCount,
        );
        if (
          filtered.messages.length < messages.length
          && filtered.messages.length <= filtered.prefixCount
        ) return;
        const lifecycleParams = filtered.messages.length === messages.length
          ? params
          : {
              ...params,
              messages: filtered.messages,
              prePromptMessageCount: filtered.prefixCount,
            };
        // Session cleanup still needs the host identity when capture is disabled.
        const identity = {
          accountId: lifecycleParams?.accountId || cfg.accountId,
          userId: lifecycleParams?.userId || cfg.userId,
          agentId: lifecycleParams?.agentId || cfg.agentId,
        };
        if (lifecycleParams?.sessionId) {
          sessionAgents.retain(lifecycleParams.sessionId);
          sessionIdentities.set(lifecycleParams.sessionId, identity);
        }
        // Only capture work is skipped; local lifecycle bookkeeping remains active.
        if (!cfg.autoCapture) return;

        // EXTRACT userId from tagged message
        const taggedUserId = extractUserIdFromMessages(filtered.messages);
        if (taggedUserId) {
          console.log(`[og-memory] afterTurn: extracted userId from tag: ${taggedUserId}`);
        }

        console.log(`[og-memory] afterTurn called, sessionId=${params?.sessionId || "unknown"}, messages=${messages.length}, userId=${taggedUserId || params?.userId || "(none)"}`);

        // Always route through server's after_turn endpoint
        // It handles: session buffer → memory extraction → write AGFS → index → archive
        const enhancedParams = {
          ...lifecycleParams,
          ...identity,
          commitTokenThreshold: cfg.commitTokenThreshold,
        };

        return submitAfterTurn(enhancedParams);
      },

      async dispose() {
        return { disposed: true, flushed: false, reason: "run_cleanup" };
      },
    })});
  },
};
