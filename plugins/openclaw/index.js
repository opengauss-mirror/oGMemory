import { spawn } from "node:child_process";
import { createHash, createHmac, randomUUID } from "node:crypto";
import {
  chmodSync,
  closeSync,
  constants as fsConstants,
  existsSync,
  fchmodSync,
  lstatSync,
  mkdirSync,
  openSync,
  readFileSync,
  readdirSync,
  renameSync,
  statSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
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

const FIRST_VIEW_ACTIVE_TURN_MAX_AGE_MS = 30 * 60 * 1000;
const FIRST_VIEW_SESSION_CONTEXTS_SYMBOL = Symbol.for(
  "ogmemory.openclaw.first-view-session-contexts.v1",
);

function sharedFirstViewSessionContexts() {
  if (!(globalThis[FIRST_VIEW_SESSION_CONTEXTS_SYMBOL] instanceof Map)) {
    globalThis[FIRST_VIEW_SESSION_CONTEXTS_SYMBOL] = new Map();
  }
  return globalThis[FIRST_VIEW_SESSION_CONTEXTS_SYMBOL];
}

function userMessageSemanticTurnMode(value) {
  const normalized = typeof value === "string" ? value.trim().toLowerCase() : "";
  if (normalized && normalized !== "user_message") {
    throw new Error(
      "semanticTurnMode currently supports only 'user_message'",
    );
  }
  return "user_message";
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
    toolResultExternalizationEnabled: resolveBoolean(
      pluginConfig.toolResultExternalizationEnabled ??
        process.env.OGMEM_TOOL_RESULT_EXTERNALIZATION_ENABLED,
      false,
    ),
    toolResultFirstViewExternalizationEnabled: resolveBoolean(
      pluginConfig.toolResultFirstViewExternalizationEnabled ??
        process.env.OGMEM_TOOL_RESULT_FIRST_VIEW_ENABLED,
      true,
    ),
    toolResultFirstViewMinChars: resolveInteger(
      pluginConfig.toolResultFirstViewMinChars ??
        process.env.OGMEM_TOOL_RESULT_FIRST_VIEW_MIN_CHARS,
      20_000,
    ),
    toolResultFirstViewStateDir:
      pluginConfig.toolResultFirstViewStateDir
      || process.env.OGMEM_TOOL_RESULT_FIRST_VIEW_STATE_DIR
      || join(
        process.env.OPENCLAW_STATE_DIR
          || process.env.XDG_STATE_HOME
          || join(process.env.HOME || "/tmp", ".openclaw"),
        "ogmemory-first-view",
      ),
    semanticTurnMode: userMessageSemanticTurnMode(
      pluginConfig.semanticTurnMode ??
        process.env.OGMEM_TOOL_RESULT_SEMANTIC_TURN_MODE,
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

function secureFirstViewStateDirectory(directory) {
  mkdirSync(directory, { recursive: true, mode: 0o700 });
  const stat = lstatSync(directory);
  if (stat.isSymbolicLink() || !stat.isDirectory()) {
    throw new Error("first-view state path must be a real directory");
  }
  if (typeof process.getuid === "function" && stat.uid !== process.getuid()) {
    throw new Error("first-view state directory is not owned by this process user");
  }
  chmodSync(directory, 0o700);
  return directory;
}

function writePrivateFile(path, content) {
  const noFollow = fsConstants.O_NOFOLLOW;
  if (typeof noFollow !== "number") {
    throw new Error("secure no-follow file writes are unavailable");
  }
  const fd = openSync(
    path,
    fsConstants.O_WRONLY
      | fsConstants.O_CREAT
      | fsConstants.O_TRUNC
      | fsConstants.O_CLOEXEC
      | noFollow,
    0o600,
  );
  try {
    fchmodSync(fd, 0o600);
    writeFileSync(fd, content, "utf-8");
  } finally {
    closeSync(fd);
  }
}

const FIRST_VIEW_LOCK_WAIT = new Int32Array(new SharedArrayBuffer(4));

function firstViewStatePath(directory, sessionId) {
  const digest = createHash("sha256").update(String(sessionId)).digest("hex");
  return join(secureFirstViewStateDirectory(directory), `${digest}.json`);
}

function canonicalFirstViewStateMac(state, secret) {
  const unsigned = { ...state };
  delete unsigned.hmac_sha256;
  return createHmac("sha256", secret)
    .update(JSON.stringify(canonicalRuntimeValue(unsigned)))
    .digest("hex");
}

function validateFirstViewDiskState(state, sessionId, secret) {
  const expectedSessionHash = createHash("sha256")
    .update(String(sessionId))
    .digest("hex");
  if (
    typeof sessionId !== "string"
    || !sessionId
    || state?.schema_version !== 1
    || state?.session_id !== sessionId
    || state?.session_sha256 !== expectedSessionHash
    || !Array.isArray(state?.leases)
    || typeof state?.hmac_sha256 !== "string"
    || canonicalFirstViewStateMac(state, secret) !== state.hmac_sha256
  ) {
    throw new Error("first-view state attestation is invalid");
  }
  return state;
}

function withFirstViewStateLock(path, operation) {
  const lockPath = `${path}.lock`;
  let fd = null;
  for (let attempt = 0; attempt < 40; attempt += 1) {
    try {
      fd = openSync(
        lockPath,
        fsConstants.O_WRONLY
          | fsConstants.O_CREAT
          | fsConstants.O_EXCL
          | fsConstants.O_CLOEXEC
          | fsConstants.O_NOFOLLOW,
        0o600,
      );
      break;
    } catch (error) {
      if (error?.code !== "EEXIST") throw error;
      try {
        if (Date.now() - statSync(lockPath).mtimeMs > 30_000) unlinkSync(lockPath);
      } catch (statError) {
        if (statError?.code !== "ENOENT") throw statError;
      }
      Atomics.wait(FIRST_VIEW_LOCK_WAIT, 0, 0, 5);
    }
  }
  if (fd == null) throw new Error("first-view state lock timeout");
  try {
    return operation();
  } finally {
    closeSync(fd);
    try {
      unlinkSync(lockPath);
    } catch (error) {
      if (error?.code !== "ENOENT") throw error;
    }
  }
}

function readFirstViewDiskState(path, sessionId, secret) {
  let raw;
  try {
    raw = readFileSync(path, "utf8");
  } catch (error) {
    if (error?.code === "ENOENT") return null;
    throw error;
  }
  return validateFirstViewDiskState(JSON.parse(raw), sessionId, secret);
}

function readUnknownFirstViewDiskState(path, secret) {
  const state = JSON.parse(readFileSync(path, "utf8"));
  return validateFirstViewDiskState(state, state?.session_id, secret);
}

function writeFirstViewDiskState(path, state, secret) {
  const signed = {
    ...state,
    hmac_sha256: canonicalFirstViewStateMac(state, secret),
  };
  const temporary = `${path}.${process.pid}.${randomUUID()}.tmp`;
  try {
    writePrivateFile(
      temporary,
      `${JSON.stringify(signed)}\n`,
    );
    renameSync(temporary, path);
  } finally {
    try {
      unlinkSync(temporary);
    } catch (error) {
      if (error?.code !== "ENOENT") throw error;
    }
  }
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
      OGMEM_TOOL_RESULT_EXTERNALIZATION_ENABLED: String(
        Boolean(cfg.toolResultExternalizationEnabled),
      ),
      OGMEM_TOOL_RESULT_FIRST_VIEW_ENABLED: String(
        Boolean(cfg.toolResultFirstViewExternalizationEnabled),
      ),
      OGMEM_TOOL_RESULT_SEMANTIC_TURN_MODE: cfg.semanticTurnMode,
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

  return headers;
}

async function callHttp(baseUrl, method, params, cfg = {}) {
  try {
    const headers = buildAuthHeaders(cfg, params, "application/json");
    const resp = await fetch(`${baseUrl}/api/v1/${method}`, {
      method: "POST",
      headers,
      body: JSON.stringify(params),
      signal: AbortSignal.timeout(
        cfg.requestTimeoutMs ?? (
          method === "after_turn"
            ? 300_000
            : method === "govern_first_view_tool_result"
              ? 10_000
              : 120_000
        ),
      ),
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
      const safeErrorBody = parsedBody && parsed && typeof parsed === "object"
        ? { ...parsed }
        : null;
      if (safeErrorBody) {
        delete safeErrorBody.toolResultGovernanceFallbackToken;
      }
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
          retryable: (
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
    return await resp.json();
  } catch (err) {
    console.error(`[og-memory] HTTP ${method} error: ${err.message}`);
    if (cfg.structuredFailures === true) {
      return {
        ok: false,
        retryable: true,
        status: "transport_error",
        error: err.message,
      };
    }
    return null;
  }
}

async function deleteSessionHttp(baseUrl, sessionId, params, cfg = {}) {
  try {
    const headers = buildAuthHeaders(cfg, params, "application/json");
    const resp = await fetch(
      `${baseUrl}/api/v1/sessions/${encodeURIComponent(sessionId)}`,
      {
        method: "DELETE",
        headers,
        body: JSON.stringify(params),
        signal: AbortSignal.timeout(120_000),
      },
    );
    if (!resp.ok) {
      const body = await resp.text();
      console.error(
        `[og-memory] HTTP ${resp.status} from delete_session, body: ${body}`,
      );
      let parsed = {};
      try {
        parsed = body ? JSON.parse(body) : {};
      } catch {}
      return {
        ...parsed,
        ok: false,
        retryable: (
          ([408, 409, 425, 429].includes(resp.status) || resp.status >= 500)
          && parsed?.retryable !== false
        ),
        httpStatus: resp.status,
      };
    }
    return await resp.json();
  } catch (err) {
    console.error(`[og-memory] HTTP delete_session error: ${err.message}`);
    return {
      ok: false,
      retryable: true,
      status: "transport_error",
      error: err.message,
    };
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
const SESSION_DELETE_MAX_ATTEMPTS = 30;
const SESSION_DELETE_RETRY_DELAY_MS = 250;

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

async function deleteSessionWithRetry(callDelete, params) {
  for (let attempt = 1; attempt <= SESSION_DELETE_MAX_ATTEMPTS; attempt += 1) {
    const result = await callDelete(params);
    if (result?.ok === true || result?.cleaned === true) return result;
    if (result?.retryable === false) {
      throw new Error(
        `[og-memory] session delete permanently rejected: ${result?.error_code || result?.error || "unknown"}`,
      );
    }
    if (attempt < SESSION_DELETE_MAX_ATTEMPTS) {
      await new Promise((resolve) => {
        setTimeout(
          resolve,
          Math.min(2_000, SESSION_DELETE_RETRY_DELAY_MS * attempt),
        );
      });
    }
  }
  throw new Error(
    `[og-memory] session delete failed after ${SESSION_DELETE_MAX_ATTEMPTS} attempts`,
  );
}

function firstNonEmptyString(...values) {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) return value.trim();
  }
  return "";
}

function canonicalRuntimeValue(value) {
  if (Array.isArray(value)) return value.map(canonicalRuntimeValue);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.keys(value).sort().map((key) => [key, canonicalRuntimeValue(value[key])]),
    );
  }
  return value;
}

function runtimeMessageFingerprint(message) {
  return createHash("sha256")
    .update(JSON.stringify(canonicalRuntimeValue(message)))
    .digest("hex");
}

function runtimeMessageVersion(message) {
  if (!message || typeof message !== "object" || Array.isArray(message)) return null;
  const seen = new WeakSet();
  seen.add(message);
  const capture = (value) => {
    if (value == null || typeof value !== "object") {
      return { kind: "primitive", value };
    }
    if (seen.has(value)) return { kind: "reference", value };
    seen.add(value);
    return {
      kind: "object",
      value,
      entries: Object.keys(value).sort().map((key) => [key, capture(value[key])]),
    };
  };
  return Object.keys(message).sort().map((key) => [key, capture(message[key])]);
}

function sameRuntimeMessageVersion(previous, message) {
  if (!previous || previous.message !== message) return false;
  const entries = previous.entries;
  if (!Array.isArray(entries)) return false;
  const keys = Object.keys(message).sort();
  if (entries.length !== keys.length) return false;
  const same = (version, value) => {
    if (!version || typeof version !== "object") return false;
    if (version.kind === "primitive" || version.kind === "reference") {
      return version.value === value;
    }
    if (version.kind !== "object" || version.value !== value) return false;
    const nestedEntries = version.entries;
    if (!Array.isArray(nestedEntries)) return false;
    const nestedKeys = Object.keys(value).sort();
    if (nestedEntries.length !== nestedKeys.length) return false;
    return nestedEntries.every(([key, child], index) => (
      key === nestedKeys[index] && same(child, value[key])
    ));
  };
  return entries.every(([key, version], index) => (
    key === keys[index] && same(version, message[key])
  ));
}

function messageProjectionRecord(message, { ignoreUserContent = false } = {}) {
  if (!message || typeof message !== "object" || Array.isArray(message)) return message;
  const record = {};
  for (const key of Object.keys(message).sort()) {
    if (key === "_ogmem_source" || key === "__openclawTranscriptPromptText") continue;
    if (ignoreUserContent && message.role === "user" && key === "content") continue;
    record[key] = message[key];
  }
  return record;
}

function messageProjectionFingerprint(message, options = {}) {
  return createHash("sha256")
    .update(JSON.stringify(canonicalRuntimeValue(messageProjectionRecord(message, options))))
    .digest("hex");
}

function projectionSignatureFromFingerprints(fingerprints) {
  return createHash("sha256")
    .update(JSON.stringify(fingerprints))
    .digest("hex");
}

function assistantHasToolCall(message) {
  if (Array.isArray(message?.toolCalls) && message.toolCalls.length) return true;
  if (Array.isArray(message?.tool_calls) && message.tool_calls.length) return true;
  if (Array.isArray(message?.functionCall) && message.functionCall.length) return true;
  if (Array.isArray(message?.function_call) && message.function_call.length) return true;
  if (message?.toolCall && typeof message.toolCall === "object") return true;
  if (message?.tool_call && typeof message.tool_call === "object") return true;
  if (message?.functionCall && typeof message.functionCall === "object") return true;
  if (message?.function_call && typeof message.function_call === "object") return true;
  const stopReason = firstNonEmptyString(message?.stopReason, message?.stop_reason)
    .toLowerCase()
    .replace(/[\s-]+/gu, "_");
  if ([
    "function_call",
    "function_calls",
    "functioncall",
    "functioncalls",
    "tool_call",
    "tool_calls",
    "tooluse",
    "tool_use",
  ].includes(stopReason)) return true;
  const blocks = Array.isArray(message?.content)
    ? message.content
    : message?.content && typeof message.content === "object"
      ? [message.content]
      : [];
  return blocks.some((block) => {
    if (!block || typeof block !== "object") return false;
    const type = firstNonEmptyString(block.type)
      .toLowerCase()
      .replace(/[.\s-]+/gu, "_");
    return [
      "function_call",
      "function_calls",
      "functioncall",
      "functioncalls",
      "toolcall",
      "tool_call",
      "tool_calls",
      "tooluse",
      "tool_use",
    ].includes(type);
  });
}

function assistantHasTerminalContent(message) {
  const content = message?.content;
  if (typeof content === "string") return Boolean(content.trim());
  if (!Array.isArray(content)) return false;
  return content.some((block) => {
    if (typeof block === "string") return Boolean(block.trim());
    if (!block || typeof block !== "object") return false;
    const type = firstNonEmptyString(block.type)
      .toLowerCase()
      .replace(/[.\s-]+/gu, "_");
    if (["toolcall", "tool_call", "tool_calls", "tooluse", "tool_use"].includes(type)) {
      return false;
    }
    if (typeof block.text === "string") return Boolean(block.text.trim());
    return Boolean(type);
  });
}

function semanticCheckpointComplete(messages, prePromptMessageCount, runtimeContext = {}) {
  const delta = messages.slice(prePromptMessageCount);
  if (!delta.length) return false;
  const last = delta.at(-1);
  if (!last || last.role !== "assistant") return false;
  if (last.isError === true || last.error || last.errorMessage) return false;
  if (assistantHasToolCall(last)) return false;
  const stopReason = firstNonEmptyString(last.stopReason, last.stop_reason).toLowerCase();
  if (stopReason && /(error|abort|cancel|tool|fail)/u.test(stopReason)) return false;
  const currentTokenCount = Number(runtimeContext?.currentTokenCount);
  return assistantHasTerminalContent(last) || (
    Number.isFinite(currentTokenCount) && currentTokenCount > 0
  );
}

function semanticTurnSourceNamespace(sessionKey, semanticTurnId, generation = 0) {
  const digest = createHash("sha256")
    .update(`${sessionKey}\0${semanticTurnId}\0${generation}`)
    .digest("hex")
    .slice(0, 24);
  return `ogmem-openclaw-${digest}`;
}

function normalizedMessageCount(value, fallback = 0) {
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed >= 0 ? parsed : fallback;
}

export function updateSemanticTurnState(previous, params = {}, options = {}) {
  const messages = Array.isArray(params.messages) ? params.messages : [];
  const semanticTurnId = firstNonEmptyString(
    params?.runtimeContext?.currentMessageId == null
      ? ""
      : String(params.runtimeContext.currentMessageId),
  );
  if (!semanticTurnId) return null;

  const sessionKey = firstNonEmptyString(params.sessionId, params.sessionKey);
  if (!sessionKey) return null;

  const prePromptMessageCount = Math.min(
    messages.length,
    normalizedMessageCount(params.prePromptMessageCount, messages.length),
  );
  let start = prePromptMessageCount;
  let previousFingerprints = [];
  let previousVersions = [];
  let generation = 0;
  if (previous?.semanticTurnId === semanticTurnId) {
    start = normalizedMessageCount(previous.semanticTurnStartMessageCount, prePromptMessageCount);
    previousFingerprints = Array.isArray(previous.messageFingerprints)
      ? previous.messageFingerprints
      : [];
    previousVersions = Array.isArray(previous.messageVersions)
      ? previous.messageVersions
      : [];
    generation = normalizedMessageCount(previous.generation, 0);
  }

  const boundaryValid = start <= prePromptMessageCount && start <= messages.length;
  const semanticMessages = boundaryValid ? messages.slice(start) : [];
  const fingerprintMessage = options.fingerprintMessage || runtimeMessageFingerprint;
  const currentFingerprints = semanticMessages.map((message, index) => (
    sameRuntimeMessageVersion(previousVersions[index], message)
      ? previousFingerprints[index]
      : fingerprintMessage(message)
  ));
  const appendOnly = boundaryValid && previousFingerprints.every(
    (fingerprint, index) => currentFingerprints[index] === fingerprint,
  );
  if (!appendOnly) {
    // A rewritten/truncated transcript cannot safely reuse relative positions.
    // Restart at this checkpoint, preserving only the bounded new batch.
    start = prePromptMessageCount;
    generation += 1;
  }
  const finalMessages = messages.slice(start);
  const canReuseCurrentFingerprints = boundaryValid && start === (
    previous?.semanticTurnId === semanticTurnId
      ? normalizedMessageCount(previous.semanticTurnStartMessageCount, prePromptMessageCount)
      : prePromptMessageCount
  );
  const messageFingerprints = canReuseCurrentFingerprints
    ? currentFingerprints
    : finalMessages.map(fingerprintMessage);
  return {
    semanticTurnId,
    semanticTurnStartMessageCount: start,
    semanticTurnSourceNamespace: semanticTurnSourceNamespace(
      sessionKey,
      semanticTurnId,
      generation,
    ),
    semanticTurnAppendOnlyVerified: true,
    semanticTurnComplete: semanticCheckpointComplete(
      messages,
      prePromptMessageCount,
      params.runtimeContext,
    ),
    messageFingerprints,
    messageVersions: finalMessages.map((message) => ({
      message,
      entries: runtimeMessageVersion(message),
    })),
    expectedMessageCount: messages.length,
    generation,
  };
}

function semanticTurnTransportFields(state) {
  if (!state) return {};
  return {
    semanticTurnId: state.semanticTurnId,
    semanticTurnStartMessageCount: state.semanticTurnStartMessageCount,
    semanticTurnSourceNamespace: state.semanticTurnSourceNamespace,
    semanticTurnAppendOnlyVerified: state.semanticTurnAppendOnlyVerified === true,
    semanticTurnComplete: state.semanticTurnComplete === true,
  };
}

export function withSemanticTurnParams(params, state, enabled = false) {
  return enabled ? { ...params, ...semanticTurnTransportFields(state) } : params;
}

function normalizedRole(message) {
  return firstNonEmptyString(message?.role)
    .toLowerCase()
    .replace(/[.\s-]+/gu, "_");
}

export function withExternalizationSemanticTurnModeParams(
  params,
  enabled = false,
) {
  if (!enabled) return params;
  return {
    ...params,
    semanticTurnMode: "user_message",
  };
}

export function createSemanticTurnTracker({
  maxSessions = 1024,
  fingerprintMessage = runtimeMessageFingerprint,
} = {}) {
  if (!Number.isInteger(maxSessions) || maxSessions < 1) {
    throw new Error("semantic turn tracker maxSessions must be positive");
  }
  const semanticTurns = new Map();
  const pendingSemanticAssembles = new Map();
  const projectionFingerprintCache = new WeakMap();
  const projectionFingerprints = (messages) => messages.map((message) => {
    if (!message || typeof message !== "object" || Array.isArray(message)) {
      const exact = messageProjectionFingerprint(message);
      return { exact, compatible: exact };
    }
    const cached = projectionFingerprintCache.get(message);
    if (cached && sameRuntimeMessageVersion(cached.version, message)) return cached;
    const exact = messageProjectionFingerprint(message);
    const compatible = message.role === "user"
      ? messageProjectionFingerprint(message, { ignoreUserContent: true })
      : exact;
    const value = {
      exact,
      compatible,
      version: { message, entries: runtimeMessageVersion(message) },
    };
    projectionFingerprintCache.set(message, value);
    return value;
  });
  const resolveSessionKey = (params = {}) => firstNonEmptyString(
    params.sessionId,
    params.sessionKey,
  );
  const enforceCapacity = () => {
    while (semanticTurns.size > maxSessions) {
      const oldestKey = semanticTurns.keys().next().value;
      semanticTurns.delete(oldestKey);
      pendingSemanticAssembles.delete(oldestKey);
    }
  };
  return {
    update(params = {}) {
      const key = resolveSessionKey(params);
      if (!key) return null;
      const next = updateSemanticTurnState(semanticTurns.get(key), params, { fingerprintMessage });
      if (!next) {
        semanticTurns.delete(key);
        pendingSemanticAssembles.delete(key);
        return null;
      }
      semanticTurns.delete(key);
      const projection = projectionFingerprints(Array.isArray(params.messages) ? params.messages : []);
      next.exactProjectionSignature = projectionSignatureFromFingerprints(
        projection.map((item) => item.exact),
      );
      next.compatibleProjectionSignature = projectionSignatureFromFingerprints(
        projection.map((item) => item.compatible),
      );
      next.exactProjectionFingerprints = projection.map((item) => item.exact);
      next.compatibleProjectionFingerprints = projection.map((item) => item.compatible);
      semanticTurns.set(key, next);
      enforceCapacity();
      return next;
    },
    enqueue(params = {}, state) {
      if (!state) return;
      const key = resolveSessionKey(params);
      if (!key) return;
      const queue = pendingSemanticAssembles.get(key) || [];
      const logicalKey = [
        state.semanticTurnId,
        state.generation,
        state.semanticTurnStartMessageCount,
        state.semanticTurnSourceNamespace,
        state.expectedMessageCount,
        state.exactProjectionSignature,
      ].join("\0");
      const existing = queue.findIndex((candidate) => ([
        candidate?.semanticTurnId,
        candidate?.generation,
        candidate?.semanticTurnStartMessageCount,
        candidate?.semanticTurnSourceNamespace,
        candidate?.expectedMessageCount,
        candidate?.exactProjectionSignature,
      ].join("\0") === logicalKey));
      if (existing >= 0) queue[existing] = state;
      else queue.push(state);
      pendingSemanticAssembles.set(key, queue.slice(-8));
    },
    take(params = {}) {
      const key = resolveSessionKey(params);
      if (!key) return null;
      const queue = pendingSemanticAssembles.get(key) || [];
      const messages = Array.isArray(params.messages) ? params.messages : [];
      const messageCount = messages.length;
      const countMatches = queue
        .map((candidate, index) => ({ candidate, index }))
        .filter(({ candidate }) => candidate?.expectedMessageCount === messageCount);
      const projection = projectionFingerprints(messages);
      const exactSignature = projectionSignatureFromFingerprints(
        projection.map((item) => item.exact),
      );
      const compatibleSignature = projectionSignatureFromFingerprints(
        projection.map((item) => item.compatible),
      );
      const exactMatches = countMatches.filter(({ candidate }) => (
        candidate?.exactProjectionSignature === exactSignature
      ));
      const lastUserIndex = messages.findLastIndex((message) => message?.role === "user");
      const compatibleMatches = countMatches.filter(({ candidate }) => {
        if (candidate?.compatibleProjectionSignature !== compatibleSignature) return false;
        const candidateExact = candidate?.exactProjectionFingerprints;
        const candidateCompatible = candidate?.compatibleProjectionFingerprints;
        if (
          !Array.isArray(candidateExact)
          || !Array.isArray(candidateCompatible)
          || candidateExact.length !== projection.length
        ) return false;
        const differing = projection
          .map((item, index) => item.exact === candidateExact[index] ? -1 : index)
          .filter((index) => index >= 0);
        if (differing.length !== 1 || differing[0] !== lastUserIndex) return false;
        const current = messages[lastUserIndex];
        if (projection[lastUserIndex].compatible !== candidateCompatible[lastUserIndex]) {
          return false;
        }
        // OpenClaw removes its private transcript marker before assemble. The
        // remaining safe discriminator is the active user's stable runtime
        // identity; compatibleProjectionFingerprint already proves every
        // non-content field (including this identity) matches the candidate.
        return [
          current?.id,
          current?.messageId,
          current?.message_id,
          current?.entryId,
          current?.entry_id,
          current?.timestamp,
        ].some((value) => (
          (typeof value === "string" && value.trim())
          || (typeof value === "number" && Number.isFinite(value))
        ));
      });
      const matches = exactMatches.length ? exactMatches : compatibleMatches;
      const logicalKeys = new Set(matches.map(({ candidate }) => [
        candidate?.semanticTurnId,
        candidate?.generation,
        candidate?.semanticTurnStartMessageCount,
        candidate?.semanticTurnSourceNamespace,
      ].join("\0")));
      const selected = matches.length && logicalKeys.size === 1 ? matches[0] : null;
      const state = selected ? queue.splice(selected.index, 1)[0] : null;
      if (queue.length) pendingSemanticAssembles.set(key, queue);
      else pendingSemanticAssembles.delete(key);
      if (!state) return null;
      const start = normalizedMessageCount(
        state.semanticTurnStartMessageCount,
        Number.MAX_SAFE_INTEGER,
      );
      return start <= messageCount ? state : null;
    },
    clear(params = {}) {
      const key = resolveSessionKey(params);
      if (!key) return;
      semanticTurns.delete(key);
      pendingSemanticAssembles.delete(key);
    },
    clearAll() {
      semanticTurns.clear();
      pendingSemanticAssembles.clear();
    },
  };
}

export function attachOgmemSource(messages, params = {}, enabled = false) {
  if (!Array.isArray(messages)) return [];
  if (!enabled) return messages;
  const turnId = firstNonEmptyString(
    params?.turnId,
    params?.turn_id,
    params?.runtimeTurnId,
  );
  const semanticTurnId = firstNonEmptyString(params?.semanticTurnId);
  const semanticStart = normalizedMessageCount(
    params?.semanticTurnStartMessageCount,
    messages.length,
  );
  const semanticNamespace = firstNonEmptyString(params?.semanticTurnSourceNamespace);
  const canSynthesizeSemanticIds = Boolean(
    semanticTurnId
      && semanticNamespace
      && params?.semanticTurnAppendOnlyVerified === true
      && semanticStart <= messages.length,
  );
  return messages.map((message, index) => {
    if (!message || typeof message !== "object" || Array.isArray(message)) return message;
    const { _ogmem_source: _untrustedSource, ...cleanMessage } = message;
    const semanticIndex = index - semanticStart;
    if (canSynthesizeSemanticIds && semanticIndex >= 0) {
      return {
        ...cleanMessage,
        _ogmem_source: {
          source_message_id: `${semanticNamespace}-${semanticIndex}`,
          turn_id: semanticTurnId,
          semantic_turn_attested: true,
        },
      };
    }
    const sourceMessageId = firstNonEmptyString(
      message.id,
      message.messageId,
      message.message_id,
      message.entryId,
      message.entry_id,
    );
    const sourceTurnId = firstNonEmptyString(
      turnId,
    );
    if (!sourceMessageId && !sourceTurnId) return cleanMessage;
    return {
      ...cleanMessage,
      _ogmem_source: {
        ...(sourceMessageId ? { source_message_id: sourceMessageId } : {}),
        ...(sourceTurnId ? { turn_id: sourceTurnId } : {}),
      },
    };
  });
}

function withOgmemSourceParams(params = {}, enabled = false) {
  if (!enabled) return params;
  const result = { ...params };
  if (Array.isArray(params.messages)) {
    result.messages = attachOgmemSource(params.messages, params, true);
  }
  if (params.message && typeof params.message === "object") {
    result.message = attachOgmemSource([params.message], params, true)[0];
  }
  return result;
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
    ...withOgmemSourceParams(params, cfg.toolResultExternalizationEnabled),
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
  if (result.error) {
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
  if (successfulResult.ok && successfulResult.compacted) {
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
  return successfulResult;
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

export function stripPrivateOgmemSource(messages, enabled = false) {
  if (!Array.isArray(messages) || !enabled) return messages;
  return messages.map((message) => {
    if (!message || typeof message !== "object" || Array.isArray(message)) return message;
    if (!Object.hasOwn(message, "_ogmem_source")) return message;
    const { _ogmem_source: _privateSource, ...publicMessage } = message;
    return publicMessage;
  });
}

export function toolResultModelTextChars(result) {
  const content = Array.isArray(result?.content) ? result.content : [];
  return content.reduce((total, block) => (
    block?.type === "text" && typeof block.text === "string"
      ? total + block.text.length
      : total
  ), 0);
}

export function createFirstViewToolResultMiddleware({
  cfg,
  call,
  rankingPromptForSession = () => "",
  sessionContextForSession = () => ({}),
  soleActiveSessionContext = () => ({}),
  logger = console,
}) {
  return async (event, context = {}) => {
    if (
      !cfg.toolResultExternalizationEnabled
      || !cfg.toolResultFirstViewExternalizationEnabled
      || toolResultModelTextChars(event?.result) < cfg.toolResultFirstViewMinChars
    ) {
      return undefined;
    }
    // OpenClaw 2026.6.10 constructs the middleware runner with only
    // `{ runtime: "openclaw" }`; the trusted session identity is carried by
    // the host-produced event.threadId. Keep the richer context fields first
    // for forward compatibility, then use that real runtime shape.
    const contextSessionId = firstNonEmptyString(
      context.sessionId,
      context.sessionKey,
    );
    const eventSessionId = firstNonEmptyString(event?.threadId);
    if (
      contextSessionId
      && eventSessionId
      && contextSessionId !== eventSessionId
    ) {
      logger.warn?.(
        "og-memory: first-view externalization skipped because trusted "
          + "session identities conflict",
      );
      return undefined;
    }
    let sessionId = firstNonEmptyString(contextSessionId, eventSessionId);
    let trustedSession = sessionId
      ? sessionContextForSession(sessionId) || {}
      : soleActiveSessionContext() || {};
    if (!sessionId) sessionId = firstNonEmptyString(trustedSession.sessionId);
    if (trustedSession.ambiguous === true) {
      logger.warn?.(
        "og-memory: first-view externalization skipped because trusted "
          + "session owner identity is ambiguous",
      );
      return undefined;
    }
    if (trustedSession.stale === true) {
      logger.warn?.(
        "og-memory: first-view externalization skipped because trusted "
          + "session Turn identity is stale",
      );
      return undefined;
    }
    const runtimeAgentId = firstNonEmptyString(context.agentId);
    if (
      runtimeAgentId
      && trustedSession.agentId
      && runtimeAgentId !== trustedSession.agentId
    ) {
      logger.warn?.(
        "og-memory: first-view externalization skipped because trusted "
          + "agent identities conflict",
      );
      return undefined;
    }
    const turnId = firstNonEmptyString(
      event?.turnId,
      context.runId,
      trustedSession.adapterTurnNonce,
    );
    const toolCallId = firstNonEmptyString(event?.toolCallId);
    const toolName = firstNonEmptyString(event?.toolName);
    if (!sessionId || !turnId || !toolCallId || !toolName) {
      logger.warn?.(
        "og-memory: first-view externalization skipped because trusted "
          + "session/turn/tool identity is incomplete",
      );
      return undefined;
    }

    const response = await call("govern_first_view_tool_result", {
      sessionId,
      adapterId: "openclaw",
      turnId,
      toolCallId,
      toolName,
      content: event.result.content,
      rankingPrompt: firstNonEmptyString(
        trustedSession.rankingPrompt,
        rankingPromptForSession(sessionId),
      ),
      userId: firstNonEmptyString(trustedSession.userId, cfg.userId),
      accountId: firstNonEmptyString(trustedSession.accountId, cfg.accountId),
      agentId: firstNonEmptyString(
        runtimeAgentId,
        trustedSession.agentId,
        cfg.agentId,
      ),
    });
    if (
      !response
      || response.ok === false
      || Number(response.externalized || 0) < 1
      || !Array.isArray(response.content)
    ) {
      if (response?.ok === false || Number(response?.failed || 0) > 0) {
        logger.warn?.(
          "og-memory: first-view externalization failed open; using raw tool result",
        );
      }
      return undefined;
    }
    return {
      result: {
        ...event.result,
        content: response.content,
      },
    };
  };
}

const ARTIFACT_RECALL_TOOL_NAMES = new Set([
  "list_tool_result_artifacts",
  "read_tool_result_artifact",
  "search_tool_result_artifacts",
  "ogmemory__list_tool_result_artifacts",
  "ogmemory__read_tool_result_artifact",
  "ogmemory__search_tool_result_artifacts",
]);

export function bindTrustedArtifactRecallSession(toolName, params, sessionId) {
  if (!ARTIFACT_RECALL_TOOL_NAMES.has(String(toolName || ""))) return null;
  const trustedSessionId = firstNonEmptyString(sessionId);
  if (!trustedSessionId) return null;
  const source = params && typeof params === "object" && !Array.isArray(params)
    ? params
    : {};
  const { sessionId: _untrustedCamelSessionId, ...rest } = source;
  return {
    ...rest,
    session_id: trustedSessionId,
  };
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
    if (
      cfg.toolResultExternalizationEnabled
      && cfg.toolResultFirstViewExternalizationEnabled
    ) {
      api.logger.info(
        "og-memory: first-view Turn state="
          + `${cfg.authApiKey ? "hmac" : "memory-only"}, dir_sha256=`
          + createHash("sha256")
            .update(String(cfg.toolResultFirstViewStateDir || ""))
            .digest("hex")
            .slice(0, 16),
      );
    }

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

    function deleteSession(params) {
      return deleteSessionHttp(
        ogmemUrl,
        params.sessionId,
        params,
        authCfg,
      );
    }

    // OpenClaw 2026.6.10 can register the context engine and the agent
    // tool-result middleware through separate plugin registry passes. Keep
    // their small trusted identity ledger process-global so assemble and the
    // post-tool middleware see the same active user-message Turn.
    const firstViewSessionContexts = sharedFirstViewSessionContexts();
    const firstViewDiskStateEnabled = Boolean(
      cfg.authApiKey
      && cfg.toolResultFirstViewStateDir,
    );
    const firstViewBucketFromState = (state) => ({
      leases: new Map(
        Array.isArray(state?.leases)
          ? state.leases
            .filter((value) => (
              value
              && typeof value.leaseKey === "string"
              && value.leaseKey
            ))
            .map((value) => [value.leaseKey, value])
          : [],
      ),
    });
    const firstViewStateFromBucket = (sessionKey, bucket) => ({
      schema_version: 1,
      session_id: sessionKey,
      session_sha256: createHash("sha256")
        .update(String(sessionKey))
        .digest("hex"),
      leases: [...bucket.leases.entries()].map(([leaseKey, value]) => ({
        ...value,
        leaseKey,
      })),
    });
    const loadFirstViewDiskBucket = (sessionKey) => {
      if (!firstViewDiskStateEnabled || !sessionKey) return null;
      const path = firstViewStatePath(cfg.toolResultFirstViewStateDir, sessionKey);
      const state = readFirstViewDiskState(path, sessionKey, cfg.authApiKey);
      return state ? firstViewBucketFromState(state) : null;
    };
    const mutateFirstViewDiskBucket = (sessionKey, operation) => {
      if (!firstViewDiskStateEnabled || !sessionKey) return null;
      const path = firstViewStatePath(cfg.toolResultFirstViewStateDir, sessionKey);
      return withFirstViewStateLock(path, () => {
        const current = readFirstViewDiskState(path, sessionKey, cfg.authApiKey);
        const bucket = current
          ? firstViewBucketFromState(current)
          : { leases: new Map() };
        const result = operation(bucket);
        if (!bucket.leases.size) {
          try {
            unlinkSync(path);
          } catch (error) {
            if (error?.code !== "ENOENT") throw error;
          }
        } else {
          writeFirstViewDiskState(
            path,
            firstViewStateFromBucket(sessionKey, bucket),
            cfg.authApiKey,
          );
        }
        return { bucket, result };
      });
    };
    const firstViewOwnerIdentity = (params = {}) => ({
      accountId: firstNonEmptyString(params?.accountId, cfg.accountId),
      userId: firstNonEmptyString(params?.userId, cfg.userId),
      agentId: firstNonEmptyString(params?.agentId, cfg.agentId),
    });
    const firstViewOwnerKey = (identity) => createHash("sha256")
      .update([
        identity.accountId,
        identity.userId,
        identity.agentId,
      ].join("\0"))
      .digest("hex");
    const touchFirstViewSessionBucket = (sessionKey, bucket) => {
      firstViewSessionContexts.delete(sessionKey);
      firstViewSessionContexts.set(sessionKey, bucket);
      while (firstViewSessionContexts.size > 1024) {
        firstViewSessionContexts.delete(firstViewSessionContexts.keys().next().value);
      }
    };
    const rememberFirstViewSessionContext = (
      sessionKey,
      params = {},
      leaseKey = "",
    ) => {
      if (
        !sessionKey
        || !leaseKey
        || !cfg.toolResultExternalizationEnabled
        || !cfg.toolResultFirstViewExternalizationEnabled
      ) return null;
      const now = Date.now();
      const identity = firstViewOwnerIdentity(params);
      const ownerKey = firstViewOwnerKey(identity);
      const mutate = (bucket) => {
        const previous = bucket.leases.get(leaseKey);
        if (previous) {
          if (previous.ownerKey !== ownerKey) previous.ambiguous = true;
          if (
            !Number.isFinite(previous.lastActiveAt)
            || now - previous.lastActiveAt > FIRST_VIEW_ACTIVE_TURN_MAX_AGE_MS
          ) {
            // Without a host Turn id it is unsafe to replace an expired lease:
            // a delayed afterTurn for the old Turn could otherwise delete the
            // newly armed identity. Stay fail-open until that lifecycle closes.
            previous.stale = true;
          } else if (!previous.ambiguous) {
            previous.rankingPrompt = typeof params?.prompt === "string"
              ? params.prompt.slice(0, 32_768)
              : previous.rankingPrompt;
            previous.lastActiveAt = now;
          }
        } else {
          bucket.leases.set(leaseKey, {
            rankingPrompt: typeof params?.prompt === "string"
              ? params.prompt.slice(0, 32_768)
              : "",
            ...identity,
            ownerKey,
            adapterTurnNonce: firstNonEmptyString(
              params?.turnId,
              params?.runtimeContext?.currentMessageId == null
                ? ""
                : String(params.runtimeContext.currentMessageId),
              `ogmfv-host-${randomUUID()}`,
            ),
            lastActiveAt: now,
            stale: false,
            ambiguous: false,
          });
        }
        return { leaseKey, ownerKey };
      };
      try {
        if (firstViewDiskStateEnabled) {
          const persisted = mutateFirstViewDiskBucket(sessionKey, mutate);
          touchFirstViewSessionBucket(sessionKey, persisted.bucket);
          return persisted.result;
        }
        const bucket = firstViewSessionContexts.get(sessionKey) || { leases: new Map() };
        const result = mutate(bucket);
        touchFirstViewSessionBucket(sessionKey, bucket);
        return result;
      } catch (error) {
        api.logger.warn?.(
          `og-memory: first-view Turn state failed open during assemble: ${error.message}`,
        );
        return null;
      }
    };
    const projectFirstViewBucket = (bucket) => {
      if (!bucket || !(bucket.leases instanceof Map)) return {};
      const now = Date.now();
      for (const value of bucket.leases.values()) {
        if (
          !Number.isFinite(value?.lastActiveAt)
          || now - value.lastActiveAt > FIRST_VIEW_ACTIVE_TURN_MAX_AGE_MS
        ) {
          value.stale = true;
        }
      }
      if (bucket.leases.size !== 1) return { ambiguous: true };
      const value = bucket.leases.values().next().value;
      if (value.stale === true) return { stale: true };
      if (value.ambiguous === true) return { ambiguous: true };
      return value;
    };
    const firstViewSessionContext = (sessionKey) => {
      let bucket;
      try {
        bucket = firstViewDiskStateEnabled
          ? loadFirstViewDiskBucket(sessionKey)
          : firstViewSessionContexts.get(sessionKey);
      } catch (error) {
        api.logger.warn?.(
          `og-memory: first-view Turn state failed open during middleware: ${error.message}`,
        );
        return {};
      }
      if (bucket) touchFirstViewSessionBucket(sessionKey, bucket);
      const projected = projectFirstViewBucket(bucket);
      return projected && !projected.ambiguous && !projected.stale
        ? { ...projected, sessionId: sessionKey }
        : projected;
    };
    const soleActiveFirstViewSessionContext = () => {
      if (!firstViewDiskStateEnabled) return {};
      try {
        const directory = secureFirstViewStateDirectory(
          cfg.toolResultFirstViewStateDir,
        );
        const candidates = [];
        for (const name of readdirSync(directory)) {
          if (!/^[a-f0-9]{64}\.json$/u.test(name)) continue;
          const state = readUnknownFirstViewDiskState(
            join(directory, name),
            cfg.authApiKey,
          );
          const projected = projectFirstViewBucket(
            firstViewBucketFromState(state),
          );
          if (projected.ambiguous || projected.stale) continue;
          if (projected.adapterTurnNonce) {
            candidates.push({ ...projected, sessionId: state.session_id });
          }
        }
        if (candidates.length === 1) return candidates[0];
        return candidates.length > 1 ? { ambiguous: true } : {};
      } catch (error) {
        api.logger.warn?.(
          `og-memory: first-view active Turn discovery failed open: ${error.message}`,
        );
        return {};
      }
    };
    const clearFirstViewSessionContext = (sessionKey, leaseKey) => {
      if (!sessionKey || !leaseKey) return;
      try {
        if (firstViewDiskStateEnabled) {
          const persisted = mutateFirstViewDiskBucket(sessionKey, (bucket) => {
            bucket.leases.delete(leaseKey);
          });
          if (persisted.bucket.leases.size) {
            touchFirstViewSessionBucket(sessionKey, persisted.bucket);
          } else {
            firstViewSessionContexts.delete(sessionKey);
          }
          return;
        }
        const bucket = firstViewSessionContexts.get(sessionKey);
        if (!bucket || !(bucket.leases instanceof Map)) return;
        bucket.leases.delete(leaseKey);
        if (!bucket.leases.size) {
          firstViewSessionContexts.delete(sessionKey);
          return;
        }
        touchFirstViewSessionBucket(sessionKey, bucket);
      } catch (error) {
        firstViewSessionContexts.delete(sessionKey);
        api.logger.warn?.(
          `og-memory: first-view Turn state cleanup failed open: ${error.message}`,
        );
      }
    };

    if (
      cfg.toolResultExternalizationEnabled
      && cfg.toolResultFirstViewExternalizationEnabled
    ) {
      if (typeof api.registerAgentToolResultMiddleware !== "function") {
        api.logger.warn?.(
          "og-memory: OpenClaw host lacks agent tool-result middleware; "
            + "first-view externalization is unavailable",
        );
      } else {
        api.registerAgentToolResultMiddleware(
          createFirstViewToolResultMiddleware({
            cfg,
            call: callAsync,
            rankingPromptForSession: (sessionId) => (
              firstViewSessionContext(sessionId)?.rankingPrompt || ""
            ),
            sessionContextForSession: (sessionId) => (
              firstViewSessionContext(sessionId)
            ),
            soleActiveSessionContext: soleActiveFirstViewSessionContext,
            logger: api.logger,
          }),
          { runtimes: ["openclaw"] },
        );
      }
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

    if (cfg.toolResultExternalizationEnabled) {
      api.on("before_tool_call", (event, ctx) => {
        const toolName = event?.toolName || ctx?.toolName || "";
        if (!ARTIFACT_RECALL_TOOL_NAMES.has(String(toolName))) return;
        const trustedSessionId = firstNonEmptyString(ctx?.sessionId);
        if (!trustedSessionId) {
          return {
            block: true,
            blockReason: "oGMemory Artifact recall requires a trusted OpenClaw session ID",
          };
        }
        return {
          params: bindTrustedArtifactRecallSession(
            toolName,
            event?.params,
            trustedSessionId,
          ),
        };
      });
    }

    const sessionFlushers = new Map();
    api.on("session_end", async (event, ctx) => {
      const sessionId = event?.sessionId || ctx?.sessionId || "";
      if (!sessionId) {
        sessionAgents.forget(event, ctx);
        return;
      }
      const reason = event?.reason || "unknown";
      const deleteRequested = reason === "deleted";

      const savedIdentity = sessionIdentities.get(sessionId);
      const identity = {
        accountId: cfg.accountId,
        userId: cfg.userId,
        ...savedIdentity,
        agentId: sessionAgents.resolve(event, ctx),
      };
      sessionAgents.forget(event, ctx);
      console.log(
        `[og-memory] session_end dispose called, sessionId=${sessionId}, reason=${reason}`,
      );
      let disposeResult;
      try {
        const flush = sessionFlushers.get(sessionId);
        if (flush) {
          const flushResult = await flush();
          if (flushResult?.ok === false && flushResult?.retryable !== false) {
            throw new Error(
              `pending externalization checkpoints remain for session=${sessionId}`,
            );
          }
        }
        disposeResult = await disposeWithRetry(call, { sessionId, ...identity });
      } catch (err) {
        console.error(
          `[og-memory] session_end dispose failed, sessionId=${sessionId}: ${err.message}`,
        );
      }
      try {
        if (deleteRequested) {
          const deleteResult = await deleteSessionWithRetry(deleteSession, {
            sessionId,
            ...identity,
          });
          return { dispose: disposeResult, sessionDelete: deleteResult };
        }
        return disposeResult;
      } catch (err) {
        console.error(
          `[og-memory] session delete cleanup failed, sessionId=${sessionId}: ${err.message}`,
        );
      } finally {
        sessionIdentities.delete(sessionId);
        sessionFlushers.delete(sessionId);
      }
    });

    api.registerContextEngine("og-memory-context-engine", () => {
      const instanceSessionIds = new Set();
      const instanceSessionFlushers = new Map();
      const engineFirstViewLeaseId = randomUUID();
      const instanceFirstViewLeases = new Map();
      const semanticTurnTracker = createSemanticTurnTracker();
      const semanticTrackingEnabled = Boolean(
        cfg.toolResultExternalizationEnabled,
      );
      const pendingAfterTurns = new Map();
      const maxPendingAfterTurnsPerSession = 8;
      const maxAfterTurnGovernanceRetries = 1;
      const afterTurnFlushChains = new Map();

      const afterTurnSessionKey = (params = {}) => firstNonEmptyString(
        params.sessionId,
        params.sessionKey,
      );
      const afterTurnCheckpointKey = (params = {}) => firstNonEmptyString(
        params.semanticTurnSourceNamespace,
        params.semanticTurnId,
      );
      const lifecycleSucceeded = (result) => Boolean(result && result.ok !== false);
      const enqueuePendingAfterTurn = (params) => {
        const sessionKey = afterTurnSessionKey(params);
        if (!sessionKey) return false;
        let snapshot;
        try {
          // Retry the exact JSON Turn snapshot that was append-only attested.
          // OpenClaw can mutate nested content blocks in place after this hook.
          snapshot = JSON.parse(JSON.stringify(params));
        } catch (error) {
          console.error(
            `[og-memory] cannot queue non-JSON after_turn Turn snapshot for `
              + `session=${sessionKey}: ${error.message}`,
          );
          return false;
        }
        const queue = pendingAfterTurns.get(sessionKey) || [];
        const checkpointKey = afterTurnCheckpointKey(snapshot);
        const existing = checkpointKey
          ? queue.findIndex((item) => item.checkpointKey === checkpointKey)
          : -1;
        const previousAttempts = existing >= 0
          ? Number(queue[existing]?.attempts || 0)
          : 0;
        const item = {
          checkpointKey,
          params: snapshot,
          attempts: previousAttempts,
          fallbackToken: "",
        };
        if (existing >= 0) queue[existing] = item;
        else queue.push(item);
        while (queue.length > maxPendingAfterTurnsPerSession) {
          queue.shift();
          console.error(
            `[og-memory] after_turn retry queue overflow for session=${sessionKey}; `
              + "dropped oldest Turn snapshot",
          );
        }
        pendingAfterTurns.set(sessionKey, queue);
        return true;
      };
      const flushPendingAfterTurns = async (sessionKey) => {
        const queue = pendingAfterTurns.get(sessionKey) || [];
        let lastResult = null;
        while (queue.length) {
          const attempted = queue[0];
          attempted.attempts = Number(attempted.attempts || 0) + 1;
          lastResult = await callAsync("after_turn", attempted.params);
          attempted.fallbackToken = firstNonEmptyString(
            lastResult?.toolResultGovernanceFallbackToken,
            attempted.fallbackToken,
          );
          if (lifecycleSucceeded(lastResult)) {
            // A concurrent Turn snapshot can supersede the same semantic-turn
            // slot while this request is in flight. Never remove that newer
            // snapshot merely because the older request succeeded.
            if (queue[0] !== attempted) break;
            queue.shift();
            continue;
          }
          if (lastResult?.retryable === false) {
            const rejected = attempted;
            if (queue[0] !== attempted) break;
            queue.shift();
            console.error(
              `[og-memory] dropping permanently rejected after_turn Turn snapshot `
                + `for session=${sessionKey}, checkpoint=${rejected?.checkpointKey || "unknown"}`,
            );
            continue;
          }
          if (attempted.attempts > maxAfterTurnGovernanceRetries) {
            if (queue[0] !== attempted) break;
            const fallbackToken = firstNonEmptyString(
              lastResult?.toolResultGovernanceFallbackToken,
              attempted.fallbackToken,
            );
            if (fallbackToken) {
              const fallbackResult = await callAsync("after_turn", {
                ...attempted.params,
                toolResultGovernanceFallbackToken: fallbackToken,
              });
              if (queue[0] !== attempted) break;
              if (lifecycleSucceeded(fallbackResult)) {
                queue.shift();
                lastResult = fallbackResult;
                console.error(
                  `[og-memory] persisted original Turn snapshot after `
                    + `${maxAfterTurnGovernanceRetries} externalization retries `
                    + `for session=${sessionKey}, `
                    + `checkpoint=${attempted?.checkpointKey || "unknown"}; `
                    + "later checkpoints remain eligible for externalization",
                );
                continue;
              }
              queue.shift();
              lastResult = {
                ok: false,
                retryable: false,
                status: "tool_result_governance_fallback_failed",
                abandoned: true,
                abandonedCheckpoints: 1,
                attempts: attempted.attempts,
                retries: maxAfterTurnGovernanceRetries,
                lastStatus: fallbackResult?.status || "fallback_unavailable",
              };
              console.error(
                `[og-memory] original-Turn-snapshot fallback failed after `
                  + `${maxAfterTurnGovernanceRetries} externalization retries `
                  + `for session=${sessionKey}, `
                  + `checkpoint=${attempted?.checkpointKey || "unknown"}`,
              );
              continue;
            }
            queue.shift();
            const abandonedStatus = lastResult?.status || "store_unavailable";
            lastResult = {
              ok: false,
              retryable: false,
              status: "tool_result_governance_abandoned",
              abandoned: true,
              abandonedCheckpoints: 1,
              attempts: attempted.attempts,
              retries: maxAfterTurnGovernanceRetries,
              lastStatus: abandonedStatus,
            };
            console.error(
              `[og-memory] abandoning after_turn Turn snapshot after `
                + `${maxAfterTurnGovernanceRetries} retries `
                + `(${attempted.attempts} total attempts) for session=${sessionKey}, `
                + `checkpoint=${attempted?.checkpointKey || "unknown"}; `
                + "server did not authorize original-Turn-snapshot fallback",
            );
            continue;
          }
          break;
        }
        if (queue.length) pendingAfterTurns.set(sessionKey, queue);
        else pendingAfterTurns.delete(sessionKey);
        return lastResult;
      };
      const serializeAfterTurnFlush = (sessionKey, operation) => {
        const previous = afterTurnFlushChains.get(sessionKey) || Promise.resolve();
        const run = previous.then(operation, operation);
        const tail = run.catch(() => null);
        afterTurnFlushChains.set(sessionKey, tail);
        void tail.then(() => {
          if (afterTurnFlushChains.get(sessionKey) === tail) {
            afterTurnFlushChains.delete(sessionKey);
          }
        });
        return run;
      };
      const clearFirstViewLeasesForSession = (sessionKey) => {
        const leases = instanceFirstViewLeases.get(sessionKey);
        if (!leases) return;
        for (const leaseKey of leases.values()) {
          clearFirstViewSessionContext(sessionKey, leaseKey);
        }
        instanceFirstViewLeases.delete(sessionKey);
      };
      const flushAndClearSession = async (sessionKey) => {
        let pendingResult = null;
        if (
          cfg.toolResultExternalizationEnabled
          && sessionKey
          && pendingAfterTurns.has(sessionKey)
        ) {
          // afterTurn already made the initial request. Cleanup consumes the
          // remaining bounded retry budget in one call so dispose/session_end
          // cannot remain pending forever when the Artifact store stays down.
          for (
            let retry = 1;
            retry <= maxAfterTurnGovernanceRetries
              && pendingAfterTurns.has(sessionKey);
            retry += 1
          ) {
            pendingResult = await serializeAfterTurnFlush(
              sessionKey,
              () => flushPendingAfterTurns(sessionKey),
            );
            if (
              pendingAfterTurns.has(sessionKey)
              && pendingResult?.retryable !== true
            ) break;
          }
          if (pendingAfterTurns.has(sessionKey)) {
            const abandonedCheckpoints = (
              pendingAfterTurns.get(sessionKey) || []
            ).length;
            pendingAfterTurns.delete(sessionKey);
            console.error(
              `[og-memory] abandoning ${abandonedCheckpoints} after_turn `
                + `Turn snapshot(s) after cleanup exhausted its retry budget `
                + `for session=${sessionKey}; an authorized original-Turn-snapshot `
                + "fallback could not be completed",
            );
          }
        }
        semanticTurnTracker.clear({ sessionId: sessionKey });
        pendingAfterTurns.delete(sessionKey);
        clearFirstViewLeasesForSession(sessionKey);
        const ownedFlusher = instanceSessionFlushers.get(sessionKey);
        if (ownedFlusher && sessionFlushers.get(sessionKey) === ownedFlusher) {
          sessionFlushers.delete(sessionKey);
        }
        instanceSessionFlushers.delete(sessionKey);
        instanceSessionIds.delete(sessionKey);
        return { ok: true };
      };
      const registerInstanceSession = (sessionKey) => {
        if (!sessionKey) return;
        instanceSessionIds.add(sessionKey);
        let flusher = instanceSessionFlushers.get(sessionKey);
        if (!flusher) {
          flusher = () => flushAndClearSession(sessionKey);
          instanceSessionFlushers.set(sessionKey, flusher);
        }
        sessionFlushers.set(sessionKey, flusher);
      };
      const clearAllInstanceState = () => {
        semanticTurnTracker.clearAll();
        pendingAfterTurns.clear();
        for (const sessionKey of instanceFirstViewLeases.keys()) {
          clearFirstViewLeasesForSession(sessionKey);
        }
        for (const [sessionKey, flusher] of instanceSessionFlushers.entries()) {
          if (sessionFlushers.get(sessionKey) === flusher) {
            sessionFlushers.delete(sessionKey);
          }
        }
        instanceSessionFlushers.clear();
        instanceSessionIds.clear();
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
        return call(
          "ingest",
          withOgmemSourceParams(params, cfg.toolResultExternalizationEnabled),
        );
      },

      async assemble(params) {
        params = withSessionAgent(params);
        const messages = params?.messages || [];
        const sessionKey = afterTurnSessionKey(params);
        if (sessionKey) {
          registerInstanceSession(sessionKey);
        }
        if (
          cfg.toolResultExternalizationEnabled
          && sessionKey
          && pendingAfterTurns.has(sessionKey)
        ) {
          await serializeAfterTurnFlush(
            sessionKey,
            () => flushPendingAfterTurns(sessionKey),
          );
        }
        if (
          sessionKey
          && !pendingAfterTurns.has(sessionKey)
        ) {
          const ownerKey = firstViewOwnerKey(firstViewOwnerIdentity(params));
          const leases = instanceFirstViewLeases.get(sessionKey) || new Map();
          const leaseKey = leases.get(ownerKey)
            || `${engineFirstViewLeaseId}:${randomUUID()}`;
          const remembered = rememberFirstViewSessionContext(
            sessionKey,
            params,
            leaseKey,
          );
          if (remembered) {
            leases.set(ownerKey, leaseKey);
            instanceFirstViewLeases.set(sessionKey, leases);
          }
        }
        // First-view tool externalization needs session context even without recall.
        if (!cfg.autoRecall) {
          return { messages, estimatedTokens: 0, systemPromptAddition: "" };
        }
        const semanticTurn = semanticTrackingEnabled
          ? semanticTurnTracker.take(params)
          : null;
        const semanticParams = withSemanticTurnParams(
          params,
          semanticTurn,
          semanticTrackingEnabled,
        );
        const externalizationParams = withExternalizationSemanticTurnModeParams(
          semanticParams,
          cfg.toolResultExternalizationEnabled,
        );

        const enhancedParams = {
          ...withOgmemSourceParams(externalizationParams, cfg.toolResultExternalizationEnabled),
          // Identity must come from the trusted host/configuration boundary.
          // Conversation text is untrusted and must never select an owner.
          userId: params?.userId || cfg.userId,
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

        // All context layers as "user" role messages for KV-cache efficiency.
        // systemPromptAddition is empty — profile in user msg avoids invalidating
        // the stable system prompt KV-cache when profile updates.
        // OpenClaw's convertMessagesToInputItems drops "system" role messages,
        // so context must use "user" role to reach the LLM.

        // Build layered messages for KV-cache efficiency:
        // Stable prefix (profile, episodic) -> conversation history -> dynamic suffix (session, WS)
        // This maximizes KV-cache reuse: stable prefix + conversation prefix cached across turns.
        const publicResult = cfg.toolResultExternalizationEnabled
          ? {
              ...result,
              // _ogmem_source is an oGMemory transport envelope. Even if an
              // older/misconfigured server echoes it, it must not become part
              // of OpenClaw's model-facing transcript. Trusted source and
              // attestation fields remain intact.
              messages: stripPrivateOgmemSource(result.messages, true),
            }
          : result;
        const layeredMessages = buildLayeredComposeMessages(
          publicResult,
          params.messages ?? [],
        );

        console.log(`[og-memory] assemble return: ${layeredMessages.length} msgs, ws_injected=${!!result.retrievedEvidence}, first_3_roles=${layeredMessages.slice(0,3).map(m=>m.role).join(',')}`);
        console.log(`[og-memory] assemble params keys: ${Object.keys(params || {}).join(',')}, sessionKey=${params?.sessionKey}, sessionId=${params?.sessionId}`);

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
        if (
          cfg.toolResultExternalizationEnabled
          && cfg.toolResultFirstViewExternalizationEnabled
        ) {
          const completedSessionKey = afterTurnSessionKey(lifecycleParams);
          if (completedSessionKey) {
            const leases = instanceFirstViewLeases.get(completedSessionKey);
            if (leases) {
              for (const leaseKey of leases.values()) {
                clearFirstViewSessionContext(completedSessionKey, leaseKey);
              }
              instanceFirstViewLeases.delete(completedSessionKey);
            }
          }
        }
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
        const semanticTurn = semanticTrackingEnabled
          ? semanticTurnTracker.update(lifecycleParams)
          : null;
        const semanticParams = withSemanticTurnParams(
          lifecycleParams,
          semanticTurn,
          semanticTrackingEnabled,
        );
        const externalizationParams = withExternalizationSemanticTurnModeParams(
          semanticParams,
          cfg.toolResultExternalizationEnabled,
        );

        const currentSessionKey = (
          cfg.toolResultExternalizationEnabled
        )
          ? afterTurnSessionKey(lifecycleParams)
          : firstNonEmptyString(lifecycleParams?.sessionId);
        if (currentSessionKey) {
          registerInstanceSession(currentSessionKey);
        }

        console.log(`[og-memory] afterTurn called, sessionId=${lifecycleParams?.sessionId || "unknown"}, messages=${filtered.messages.length}, userId=${lifecycleParams?.userId || cfg.userId || "(none)"}`);

        // Always route through server's after_turn endpoint
        // It handles: session buffer → memory extraction → write AGFS → index → archive
        const enhancedParams = {
          ...withOgmemSourceParams(externalizationParams, cfg.toolResultExternalizationEnabled),
          ...identity,
          commitTokenThreshold: cfg.commitTokenThreshold,
        };

        if (cfg.toolResultExternalizationEnabled) {
          semanticTurnTracker.enqueue(lifecycleParams, semanticTurn);
          const sessionKey = afterTurnSessionKey(enhancedParams);
          if (sessionKey && enqueuePendingAfterTurn(enhancedParams)) {
            return serializeAfterTurnFlush(
              sessionKey,
              () => flushPendingAfterTurns(sessionKey),
            );
          }
          if (sessionKey) {
            return {
              ok: false,
              retryable: false,
              status: "invalid_after_turn_payload",
            };
          }
        }
        return call("after_turn", enhancedParams);
      },

      async dispose() {
        const sessionIds = [...instanceSessionIds];
        let deferredResult = null;
        let cleanupError = null;
        for (const sessionId of sessionIds) {
          console.log(`[og-memory] dispose called, sessionId=${sessionId}`);
          try {
            const cleanupResult = await flushAndClearSession(sessionId);
            if (cleanupResult?.ok === false && deferredResult == null) {
              console.error(
                `[og-memory] dispose deferred for session=${sessionId}; `
                  + "retryable after_turn checkpoints remain",
              );
              deferredResult = cleanupResult;
            }
          } catch (error) {
            cleanupError ||= error;
          }
        }
        if (instanceSessionIds.size === 0) clearAllInstanceState();
        if (cleanupError) throw cleanupError;
        if (deferredResult) return deferredResult;
        return { disposed: true, flushed: false, reason: "run_cleanup" };
      },
    })});
  },
};
