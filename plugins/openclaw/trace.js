import { randomUUID as nodeRandomUUID } from "node:crypto";
import { isIP } from "node:net";
import { createSessionAgentResolver } from "./session-agent.js";

const MAX_TRACE_CONTENT_CHARS = 10_000;
const MAX_TRACE_BODY_BYTES = 60_000;
const MAX_TRACE_DEPTH = 32;
const MAX_TRACE_NODES = 2_000;
const TRACE_MAX_IN_FLIGHT = 4;
const TRACE_CIRCUIT_BREAK_MS = 30_000;
const TRACE_SESSION_IDENTITY_TTL_MS = 2 * 60 * 60 * 1000;
const TRACE_MAX_SESSION_IDENTITY_ALIASES = 2_048;
const TRACE_MAX_PENDING_SESSION_START_ALIASES = 2_048;
const TRACE_HEARTBEAT_STATE_TTL_MS = 2 * 60 * 60 * 1000;
const TRACE_MAX_HEARTBEAT_RUNS = 2_048;
const TRACE_MAX_HEARTBEAT_SESSIONS = 2_048;
const HEARTBEAT_TRANSCRIPT_PROMPT = "[OpenClaw heartbeat poll]";
const SECRET_KEY = /(?:api[_-]?key|authorization|cookie|password|secret|token)/i;
const SECRET_NAME = /(?:[a-z0-9]+[_-])*(?:api[_-]?key|password|secret(?:[_-]access[_-]key)?|token)|authorization|cookie/;
const COOKIE_HEADER = /\b(cookie\s*:\s*)([^\r\n]+)/gi;
const QUOTED_SECRET = new RegExp(`\\b(${SECRET_NAME.source})(\\s*[:=]\\s*)(["'])(.*?)\\3`, "gi");
const INLINE_SECRET = new RegExp(`\\b(${SECRET_NAME.source})(\\s*[:=]\\s*)(?:(?:bearer|basic)\\s+)?([^\\s,;"']+)`, "gi");
const QUOTED_CLI_SECRET = /(--(?:api[-_]?key|password|secret|token)\s+)(["'])(.*?)\2/gi;
const CLI_SECRET = /(--(?:api[-_]?key|password|secret|token)\s+)([^\s,;"']+)/gi;
const CONNECTION_STRING = /\b[a-z][a-z0-9+.-]*:\/\/[^\s/@:]+:[^\s/@]+@[^\s]+/gi;
const BEARER_TOKEN = /(\bbearer\s+)([^\s,;"']+)/gi;
const OPENAI_API_KEY = /\bsk-(?:proj-)?[A-Za-z0-9_-]{16,}\b/g;
const GITHUB_TOKEN = /\b(?:(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{16,}|github_pat_[A-Za-z0-9_]{20,})\b/g;
const SLACK_TOKEN = /\b(?:xox[a-z]?|xapp)-[A-Za-z0-9-]{20,}\b/g;
const JWT = /\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b/g;
const EMAIL = /(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+(?![\w.-])/g;
const E164_PHONE = /(?<![\d+])\+\d{1,3}(?:[\s.-]?\(?\d{2,4}\)?){2,4}[\s.-]?\d{3,4}(?!\d)/g;
const PHONE = /(?<!\d)1[3-9]\d{9}(?!\d)/g;
const ID_CARD = /(?<![A-Za-z0-9])\d{17}[0-9Xx](?![A-Za-z0-9])/g;
const PASSPORT = /(?<![A-Za-z0-9])[EeGg]\d{8}(?![A-Za-z0-9])/g;
const BANK_CARD = /(?<!\d)\d{16,19}(?!\d)/g;
const IPV6 = /(?<![\w:])(?:[0-9A-Fa-f]{0,4}:){2,}(?:(?:\d{1,3}\.){3}\d{1,3}|[0-9A-Fa-f]{0,4})(?![\w:.])/g;
const IPV4 = /(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])/g;

function secretKind(name) {
  const normalized = String(name).toLowerCase().replaceAll("-", "_");
  if (normalized.includes("api_key") || normalized.includes("apikey")) return "api_key";
  if (normalized.includes("password")) return "password";
  if (normalized.includes("authorization")) return "authorization";
  if (normalized.includes("cookie")) return "cookie";
  if (normalized.includes("token")) return "token";
  return "secret";
}

function safeString(value) {
  try {
    return String(value);
  } catch {
    return "[Unserializable]";
  }
}

class RedactionState {
  constructor() {
    this.placeholders = new Map();
    this.counts = new Map();
    this.seen = new WeakSet();
    this.nodes = 0;
  }

  placeholder(kind, value) {
    const valueType = typeof value;
    const plaintext = value !== null && (valueType === "object" || valueType === "function")
      ? `[${valueType}]`
      : safeString(value);
    if (!this.placeholders.has(plaintext)) {
      const count = (this.counts.get(kind) || 0) + 1;
      this.counts.set(kind, count);
      this.placeholders.set(plaintext, `{{${kind}_${count}}}`);
    }
    return this.placeholders.get(plaintext);
  }

  redactString(text) {
    let value = text.replace(CONNECTION_STRING, (match) => this.placeholder("connection_string", match));
    value = value.replace(COOKIE_HEADER, (_match, prefix, cookieValue) => (
      `${prefix}${cookieValue.replace(/([^=;\s]+)(\s*=\s*)([^;\s]+)/g, (_pair, name, separator, plaintext) => (
        `${name}${separator}${this.placeholder("cookie", plaintext)}`
      ))}`
    ));
    value = value.replace(QUOTED_SECRET, (_match, name, separator, quote, plaintext) => (
      /\{\{[^{}]+\}\}/.test(plaintext)
        ? _match
        : `${name}${separator}${quote}${this.placeholder(secretKind(name), plaintext)}${quote}`
    ));
    value = value.replace(INLINE_SECRET, (match, name, separator, plaintext) => (
      /\{\{[^{}]+\}\}/.test(plaintext)
        ? match
        : `${name}${separator}${this.placeholder(/\bbearer\s/i.test(match) ? "token" : secretKind(name), plaintext)}`
    ));
    value = value.replace(BEARER_TOKEN, (_match, prefix, token) => (
      `${prefix}${this.placeholder("token", token)}`
    ));
    value = value.replace(OPENAI_API_KEY, (match) => this.placeholder("api_key", match));
    value = value.replace(GITHUB_TOKEN, (match) => this.placeholder("token", match));
    value = value.replace(SLACK_TOKEN, (match) => this.placeholder("token", match));
    value = value.replace(JWT, (match) => this.placeholder("token", match));
    value = value.replace(QUOTED_CLI_SECRET, (_match, prefix, quote, plaintext) => (
      `${prefix}${quote}${this.placeholder(secretKind(prefix), plaintext)}${quote}`
    ));
    value = value.replace(CLI_SECRET, (_match, prefix, plaintext) => (
      `${prefix}${this.placeholder(secretKind(prefix), plaintext)}`
    ));
    value = value.replace(EMAIL, (match) => this.placeholder("email", match));
    value = value.replace(E164_PHONE, (match) => this.placeholder("phone", match));
    value = value.replace(PHONE, (match) => this.placeholder("phone", match));
    value = value.replace(ID_CARD, (match) => this.placeholder("id_card", match));
    value = value.replace(PASSPORT, (match) => this.placeholder("passport", match));
    value = value.replace(BANK_CARD, (match) => this.placeholder("bank_card", match));
    value = value.replace(IPV6, (match) => (
      isIP(match) === 6 ? this.placeholder("ipv6", match) : match
    ));
    return value.replace(IPV4, (match) => (
      isIP(match) === 4 ? this.placeholder("ipv4", match) : match
    ));
  }

  redact(value, depth = 0) {
    this.nodes += 1;
    if (this.nodes > MAX_TRACE_NODES) return "[NodeLimit]";
    if (depth > MAX_TRACE_DEPTH) return "[DepthLimit]";
    if (["function", "symbol", "undefined"].includes(typeof value)) return undefined;
    if (typeof value === "bigint") return value.toString();
    if (typeof value === "string") return this.redactString(value);
    if (value && typeof value === "object") {
      if (this.seen.has(value)) return "[Circular]";
      this.seen.add(value);
      try {
        if (Array.isArray(value)) {
          const redacted = [];
          for (const item of value) {
            if (this.nodes >= MAX_TRACE_NODES) {
              redacted.push("[NodeLimit]");
              break;
            }
            redacted.push(this.redact(item, depth + 1));
          }
          return redacted;
        }
        const redacted = {};
        for (const key of Object.keys(value)) {
          if (this.nodes >= MAX_TRACE_NODES) {
            redacted.__truncated__ = "[NodeLimit]";
            break;
          }
          if (key === "toJSON") continue;
          const descriptor = Object.getOwnPropertyDescriptor(value, key);
          if (!descriptor) continue;
          if (!Object.prototype.hasOwnProperty.call(descriptor, "value")) {
            redacted[key] = "[Getter]";
            continue;
          }
          const item = descriptor.value;
          const redactedItem = SECRET_KEY.test(key)
            ? this.placeholder(secretKind(key), item)
            : this.redact(item, depth + 1);
          if (redactedItem !== undefined) redacted[key] = redactedItem;
        }
        return redacted;
      } finally {
        this.seen.delete(value);
      }
    }
    return value;
  }
}

export function redactTraceValue(value) {
  return new RedactionState().redact(value);
}

function contentSize(value) {
  if (typeof value === "string") return value.length;
  try {
    return JSON.stringify(value).length;
  } catch {
    return safeString(value).length;
  }
}

function bounded(value) {
  let serialized;
  try {
    serialized = typeof value === "string" ? value : JSON.stringify(value);
  } catch {
    serialized = safeString(value);
  }
  if (serialized.length <= MAX_TRACE_CONTENT_CHARS) return value;
  const suffix = `\n[${serialized.length - MAX_TRACE_CONTENT_CHARS} more chars omitted]`;
  return `${serialized.slice(0, MAX_TRACE_CONTENT_CHARS - suffix.length)}${suffix}`;
}

function textContent(content) {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .map((part) => (typeof part === "string" ? part : part?.text || ""))
    .filter(Boolean)
    .join("\n");
}

export function isOpenClawHeartbeatMessage(message) {
  if (message?.role !== "user") return false;
  const text = textContent(message.content).trim();
  return text === HEARTBEAT_TRANSCRIPT_PROMPT
    || text.endsWith(`\n${HEARTBEAT_TRANSCRIPT_PROMPT}`)
    || text.includes(`\n${HEARTBEAT_TRANSCRIPT_PROMPT}\n`);
}

function lastAssistantText(messages) {
  if (!Array.isArray(messages)) return "";
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index]?.role !== "assistant") continue;
    const text = textContent(messages[index].content);
    if (text.trim()) return text;
  }
  return "";
}

function addRunId(result, event, ctx) {
  const runId = event?.runId || ctx?.runId;
  if (runId) result.run_id = String(runId);
}

export function openClawHookToEvent(hookName, event = {}, ctx = {}) {
  const state = new RedactionState();
  const result = {};
  const captureContent = ctx.traceCaptureContent !== false;

  switch (hookName) {
    case "session_start":
      result.event_type = "session_start";
      if (event.resumedFrom) result.resumed_from = String(event.resumedFrom);
      break;
    case "before_agent_run":
      result.event_type = "user_message";
      addRunId(result, event, ctx);
      {
        const prompt = state.redact(event.prompt || "");
        result.prompt_size = contentSize(prompt);
        if (captureContent) result.prompt = bounded(prompt);
      }
      break;
    case "before_tool_call":
      result.event_type = "tool_start";
      addRunId(result, event, ctx);
      result.tool_name = String(event.toolName || "unknown_tool");
      if (event.toolCallId || ctx.toolCallId) result.tool_use_id = String(event.toolCallId || ctx.toolCallId);
      {
        const toolInput = state.redact(event.params || {});
        result.tool_input_size = contentSize(toolInput);
        if (captureContent) result.tool_input = bounded(toolInput);
      }
      break;
    case "after_tool_call": {
      const failed = event.error !== undefined && event.error !== null && event.error !== "";
      result.event_type = failed ? "tool_failure" : "tool_success";
      addRunId(result, event, ctx);
      result.tool_name = String(event.toolName || "unknown_tool");
      if (event.toolCallId || ctx.toolCallId) result.tool_use_id = String(event.toolCallId || ctx.toolCallId);
      const toolInput = state.redact(event.params || {});
      result.tool_input_size = contentSize(toolInput);
      if (captureContent) result.tool_input = bounded(toolInput);
      if (failed) {
        const error = state.redact(event.error);
        result.error_size = contentSize(error);
        if (captureContent) result.error = bounded(error);
      } else if (event.result !== undefined) {
        const toolResponse = state.redact(event.result);
        result.tool_response_size = contentSize(toolResponse);
        if (captureContent) result.tool_response = bounded(toolResponse);
      }
      if (Number.isFinite(event.durationMs)) result.duration_ms = event.durationMs;
      break;
    }
    case "agent_end": {
      result.event_type = "turn_end";
      addRunId(result, event, ctx);
      const assistantMessage = lastAssistantText(event.messages);
      const assistantMessageRedacted = state.redact(assistantMessage);
      result.assistant_message_size = contentSize(assistantMessageRedacted);
      if (captureContent) result.assistant_message = bounded(assistantMessageRedacted);
      if (captureContent && Array.isArray(event.messages)) {
        const start = event.messages.findLastIndex(message => message.role === "user");
        const modelMessages = event.messages.slice(start + 1).filter(message => message.role === "assistant").map(message => ({
          text: typeof message.content === "string" ? message.content : textContent(
            (message.content || []).filter(part => part.type === "text")),
          reasoning: (Array.isArray(message.content) ? message.content : [])
            .filter(part => ["thinking", "reasoning"].includes(part.type))
            .map(part => part.thinking || part.text || part.reasoning || "").join("\n"),
        }));
        if (modelMessages.length) result.assistant_messages = bounded(state.redact(modelMessages));
      }
      result.status = event.success ? "ok" : "error";
      if (event.error) {
        const error = state.redact(event.error);
        result.error_size = contentSize(error);
        if (captureContent) result.error = bounded(error);
      }
      if (Number.isFinite(event.durationMs)) result.duration_ms = event.durationMs;
      break;
    }
    case "before_compaction":
      result.event_type = "pre_compact";
      addRunId(result, event, ctx);
      if (Number.isFinite(event.messageCount)) result.message_count = event.messageCount;
      if (Number.isFinite(event.compactingCount)) result.compacting_count = event.compactingCount;
      if (Number.isFinite(event.tokenCount)) result.token_count = event.tokenCount;
      break;
    case "session_end":
      result.event_type = "session_end";
      if (Number.isFinite(event.messageCount)) result.message_count = event.messageCount;
      if (Number.isFinite(event.durationMs)) result.duration_ms = event.durationMs;
      if (event.reason && captureContent) result.reason = bounded(state.redact(String(event.reason)));
      break;
    default:
      result.event_type = "hook_event";
  }
  return result;
}

function traceHeaders(cfg, envelope) {
  const headers = { "Content-Type": "application/json" };
  if (cfg.authApiKey) headers["X-API-Key"] = cfg.authApiKey;
  if (cfg.authAccountId) headers["X-Account-ID"] = cfg.authAccountId;
  if (envelope.user_id) headers["X-User-ID"] = envelope.user_id;
  return headers;
}

function boundedTraceBody(envelope) {
  let body = JSON.stringify(envelope);
  if (Buffer.byteLength(body, "utf8") <= MAX_TRACE_BODY_BYTES) return body;
  const event = { ...envelope.event };
  const payload = { ...envelope, event };
  for (const field of ["assistant_messages", "assistant_message"]) {
    if (event[field] === undefined) continue;
    event.truncated = true;
    const text = typeof event[field] === "string" ? event[field] : JSON.stringify(event[field]);
    event[field] = text;
    const excess = Buffer.byteLength(JSON.stringify(payload), "utf8") - MAX_TRACE_BODY_BYTES;
    const encoded = Buffer.from(text, "utf8");
    event[field] = new TextDecoder().decode(
      encoded.subarray(0, Math.max(0, encoded.length - excess)), { stream: true },
    );
    body = JSON.stringify(payload);
    if (Buffer.byteLength(body, "utf8") <= MAX_TRACE_BODY_BYTES) break;
  }
  return body;
}

export async function postTraceEvent(cfg, envelope, deps = {}) {
  const fetchFn = deps.fetchFn || fetch;
  const timeoutSignalFn = deps.timeoutSignalFn || ((timeoutMs) => AbortSignal.timeout(timeoutMs));
  const logger = deps.logger || console;
  const url = `${String(cfg.memoryApiBaseUrl || "").replace(/\/$/, "")}/api/v1/trace/events`;
  let lastError;

  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      const body = boundedTraceBody(envelope);
      if (Buffer.byteLength(body, "utf8") > MAX_TRACE_BODY_BYTES) {
        logger.error?.("[og-memory] trace upload dropped: body exceeds 60000 bytes");
        return "drop";
      }
      const response = await fetchFn(url, {
        method: "POST",
        headers: traceHeaders(cfg, envelope),
        body,
        signal: timeoutSignalFn(cfg.traceTimeoutMs),
      });
      if (response.ok) return true;
      if (response.status >= 400 && response.status < 500 && ![408, 429].includes(response.status)) {
        logger.error?.(`[og-memory] trace upload dropped: HTTP ${response.status}`);
        return "drop";
      }
      lastError = new Error(`HTTP ${response.status ?? "error"}`);
    } catch (error) {
      lastError = error;
    }
  }

  logger.error?.(`[og-memory] trace upload failed after 2 attempts: ${lastError?.message || lastError}`);
  return false;
}

const TRACE_HOOK_NAMES = [
  "session_start",
  // Non-bundled OpenClaw plugins need
  // plugins.entries.og-memory-context-engine.hooks.allowConversationAccess=true
  // for conversation-bearing before_agent_run and agent_end payloads.
  "before_agent_run",
  "before_tool_call",
  "after_tool_call",
  "agent_end",
  "before_compaction",
  "session_end",
];

export function registerOpenClawTraceHooks(api, cfg, deps = {}) {
  const randomUUID = deps.randomUUID || nodeRandomUUID;
  const now = deps.now || (() => new Date());
  const nowMs = deps.nowMs || (() => Date.now());
  const sessionAgents = deps.sessionAgents || createSessionAgentResolver(cfg.agentId, nowMs);
  const sessionIdentities = new Map();
  const pendingSessionStarts = new Map();
  const heartbeatRuns = new Map();
  const unkeyedHeartbeatSessions = new Map();
  let inFlight = 0;
  let circuitOpenUntil = 0;
  let circuitLogged = false;

  function cleanupSessionIdentities() {
    const cutoff = nowMs() - TRACE_SESSION_IDENTITY_TTL_MS;
    for (const [alias, identity] of sessionIdentities) {
      if ((identity.lastSeenMs || 0) < cutoff) sessionIdentities.delete(alias);
    }
    if (sessionIdentities.size <= TRACE_MAX_SESSION_IDENTITY_ALIASES) return;
    const oldestAliases = [...sessionIdentities.entries()]
      .sort((left, right) => (left[1].lastSeenMs || 0) - (right[1].lastSeenMs || 0));
    for (const [alias] of oldestAliases.slice(
      0,
      sessionIdentities.size - TRACE_MAX_SESSION_IDENTITY_ALIASES,
    )) {
      sessionIdentities.delete(alias);
    }
  }

  function scheduleTraceUpload(envelope) {
    const logger = deps.logger || api.logger;
    if (nowMs() < circuitOpenUntil) {
      if (!circuitLogged) {
        logger?.error?.("[og-memory] trace upload skipped while circuit is open");
        circuitLogged = true;
      }
      return;
    }
    circuitLogged = false;
    if (inFlight >= TRACE_MAX_IN_FLIGHT) {
      logger?.error?.("[og-memory] trace upload skipped: in-flight limit reached");
      return;
    }
    inFlight += 1;
    void postTraceEvent(cfg, envelope, {
      ...deps,
      logger,
    }).then((ok) => {
      if (ok === false) {
        circuitOpenUntil = nowMs() + TRACE_CIRCUIT_BREAK_MS;
        circuitLogged = false;
      }
    }).finally(() => {
      inFlight -= 1;
    });
  }

  function trimOldestEntries(map, maxSize) {
    while (map.size > maxSize) map.delete(map.keys().next().value);
  }

  function cleanupHeartbeatState(currentMs) {
    const cutoff = currentMs - TRACE_HEARTBEAT_STATE_TTL_MS;
    for (const [runId, state] of heartbeatRuns) {
      if (state.lastSeenMs < cutoff) heartbeatRuns.delete(runId);
    }
    for (const [sessionId, lastSeenMs] of unkeyedHeartbeatSessions) {
      if (lastSeenMs < cutoff) unkeyedHeartbeatSessions.delete(sessionId);
    }
  }

  function rememberHeartbeatRun(runId, sessionId, currentMs) {
    const knownSessionId = heartbeatRuns.get(runId)?.sessionId;
    heartbeatRuns.delete(runId);
    heartbeatRuns.set(runId, { sessionId: sessionId || knownSessionId || "", lastSeenMs: currentMs });
    trimOldestEntries(heartbeatRuns, TRACE_MAX_HEARTBEAT_RUNS);
  }

  function rememberUnkeyedHeartbeat(sessionId, currentMs) {
    unkeyedHeartbeatSessions.delete(sessionId);
    unkeyedHeartbeatSessions.set(sessionId, currentMs);
    trimOldestEntries(unkeyedHeartbeatSessions, TRACE_MAX_HEARTBEAT_SESSIONS);
  }

  function takePendingSessionStart(aliases) {
    let pending;
    for (const alias of aliases) {
      pending = pendingSessionStarts.get(alias);
      if (pending) break;
    }
    if (!pending) return undefined;
    for (const [alias, envelope] of pendingSessionStarts) {
      if (envelope === pending) pendingSessionStarts.delete(alias);
    }
    return pending;
  }

  function cachePendingSessionStart(aliases, envelope) {
    while (pendingSessionStarts.size + aliases.length > TRACE_MAX_PENDING_SESSION_START_ALIASES) {
      const oldest = pendingSessionStarts.values().next().value;
      if (!oldest) break;
      for (const [alias, pending] of pendingSessionStarts) {
        if (pending === oldest) pendingSessionStarts.delete(alias);
      }
    }
    for (const alias of aliases) pendingSessionStarts.set(alias, envelope);
  }

  for (const hookName of TRACE_HOOK_NAMES) {
    api.on(hookName, async (event, ctx) => {
      const aliases = [...new Set([
        event?.sessionId,
        ctx?.sessionId,
        event?.sessionKey,
        ctx?.sessionKey,
      ].filter(Boolean).map(String))];
      const runId = String(event?.runId || ctx?.runId || "");
      const heartbeatSessionId = aliases[0] || "";
      const currentMs = nowMs();
      cleanupHeartbeatState(currentMs);
      const heartbeatPrompt = hookName === "before_agent_run"
        && isOpenClawHeartbeatMessage({ role: "user", content: event?.prompt });
      if (ctx?.trigger === "heartbeat" || heartbeatPrompt) {
        if (runId) {
          rememberHeartbeatRun(runId, heartbeatSessionId, currentMs);
        } else if (heartbeatSessionId) {
          rememberUnkeyedHeartbeat(heartbeatSessionId, currentMs);
        }
      }
      const isHeartbeat = runId
        ? heartbeatRuns.has(runId)
        : Boolean(heartbeatSessionId
          && unkeyedHeartbeatSessions.has(heartbeatSessionId));
      if (isHeartbeat) {
        if (hookName === "agent_end" || hookName === "session_end") {
          takePendingSessionStart(aliases);
          if (runId) {
            heartbeatRuns.delete(runId);
          } else if (heartbeatSessionId) {
            unkeyedHeartbeatSessions.delete(heartbeatSessionId);
            if (hookName === "session_end") {
              for (const [heartbeatRunId, state] of heartbeatRuns) {
                if (state.sessionId === heartbeatSessionId) heartbeatRuns.delete(heartbeatRunId);
              }
            }
          }
        } else if (runId) {
          rememberHeartbeatRun(runId, heartbeatSessionId, currentMs);
        } else if (heartbeatSessionId) {
          rememberUnkeyedHeartbeat(heartbeatSessionId, currentMs);
        }
        return;
      }
      let canonicalEvent;
      try {
        canonicalEvent = openClawHookToEvent(hookName, event, {
          ...ctx,
          traceCaptureContent: cfg.traceCaptureContent,
        });
      } catch (error) {
        api.logger?.error?.(`[og-memory] trace ${hookName} skipped: ${error?.message || error}`);
        return;
      }
      if (!canonicalEvent) return;
      cleanupSessionIdentities();
      const sessionId = aliases[0];
      if (!sessionId) {
        api.logger?.error?.(`[og-memory] trace ${canonicalEvent.event_type} skipped: missing session id`);
        return;
      }
      const matchedIdentities = new Set(
        aliases.map((alias) => sessionIdentities.get(alias)).filter(Boolean),
      );
      const cachedIdentity = matchedIdentities.values().next().value || {};
      for (const identity of matchedIdentities) {
        cachedIdentity.accountId ||= identity.accountId;
        cachedIdentity.userId ||= identity.userId;
      }
      const runtimeIdentity = {
        accountId: event?.accountId || ctx?.accountId,
        userId: event?.senderId || ctx?.senderId,
      };
      cachedIdentity.accountId = runtimeIdentity.accountId || cachedIdentity.accountId;
      cachedIdentity.userId = runtimeIdentity.userId || cachedIdentity.userId;
      cachedIdentity.lastSeenMs = nowMs();
      if ([cachedIdentity.accountId, cachedIdentity.userId].some(Boolean)) {
        for (const [alias, identity] of sessionIdentities) {
          if (matchedIdentities.has(identity)) sessionIdentities.set(alias, cachedIdentity);
        }
        for (const alias of aliases) sessionIdentities.set(alias, cachedIdentity);
      }
      const eventId = randomUUID();
      const envelope = {
        request_id: eventId,
        timestamp: now().toISOString(),
        account_id: cachedIdentity.accountId || cfg.accountId,
        user_id: cachedIdentity.userId || cfg.userId,
        agent_id: sessionAgents.resolve(event, ctx),
        session_id: String(sessionId),
        event: canonicalEvent,
      };
      if (hookName === "session_start") {
        cachePendingSessionStart(aliases, envelope);
        return;
      }
      const pendingSessionStart = takePendingSessionStart(aliases);
      if (pendingSessionStart) {
        scheduleTraceUpload(pendingSessionStart);
      }
      scheduleTraceUpload(envelope);
      if (hookName === "session_end") {
        if (!deps.sessionAgents) sessionAgents.forget(event, ctx);
        const currentAliases = new Set(aliases);
        for (const [alias, identity] of sessionIdentities) {
          if (identity === cachedIdentity || currentAliases.has(alias)) {
            sessionIdentities.delete(alias);
          }
        }
      }
    });
  }
}
