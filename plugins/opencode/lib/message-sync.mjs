import { redactString, safeErrorMessage } from "./redaction.mjs";

const MAX_MESSAGES_PER_SESSION = 500;
const MAX_MESSAGE_CHARS = 100000;
const MAX_SESSIONS = 128;
const MAX_BUFFER_CHARS = 8 * 1024 * 1024;
const MAX_CONCURRENT_FLUSHES = 8;
const WARNING_INTERVAL_MS = 60000;

export function mergeMessageContent(existing, incoming) {
  const next = typeof incoming === "string" ? incoming.trim() : "";
  if (!next) return existing || "";
  if (!existing || next === existing) return next;
  if (next.startsWith(existing) || next.includes(existing)) return next;
  if (existing.startsWith(next) || existing.includes(next)) return existing;
  return `${existing}\n${next}`.slice(0, MAX_MESSAGE_CHARS);
}

function sessionIdOf(value) {
  return value?.sessionID || value?.sessionId || "";
}

function generated(info) {
  const metadata = info?.metadata || {};
  return Boolean(
    info?.synthetic
    || info?.ignored
    || info?.compaction
    || (info?.role === "assistant" && info?.summary === true)
    || metadata.compaction
    || metadata.compaction_continue
    || metadata.compactionContinue
  );
}

function completed(info) {
  return Boolean(info?.time?.completed || ["stop", "end_turn"].includes(info?.finish));
}

export function createReasoningCapture({ trace, limit = MAX_MESSAGES_PER_SESSION, maxParts = 32, maxChars = 10000 }) {
  const messages = new Map();
  const stateFor = (sessionId, messageId) => {
    if (!sessionId || !messageId) return null;
    const key = JSON.stringify([sessionId, messageId]);
    if (!messages.has(key)) {
      if (messages.size >= limit) messages.delete(messages.keys().next().value);
      messages.set(key, { sessionId, generated: false, known: false, parts: new Map() });
    }
    return messages.get(key);
  };
  const emit = (state, id, text) => trace.emit("assistant_reasoning", state.sessionId, { objectId: id, content: text });
  return {
    observeMessage(info) {
      if (info?.role !== "assistant") return;
      const state = stateFor(sessionIdOf(info), info.id);
      if (!state) return;
      state.generated ||= generated(info);
      state.known = true;
      if (state.generated) state.parts.clear();
      else {
        for (const [id, text] of state.parts) emit(state, id, text);
        state.parts.clear();
      }
    },
    observePart(part) {
      if (part?.type !== "reasoning" || !part.time?.end || part.synthetic || part.ignored
        || !part.id || typeof part.text !== "string") return;
      const state = stateFor(sessionIdOf(part), part.messageID || part.messageId);
      if (!state || state.generated) return;
      if (state.known) {
        emit(state, part.id, part.text);
        return;
      }
      if (state.parts.size >= maxParts && !state.parts.has(part.id)) return;
      const used = [...state.parts.values()].reduce((total, text) => total + text.length, 0);
      const remaining = maxChars - used + (state.parts.get(part.id)?.length || 0);
      if (remaining <= 0) return;
      state.parts.set(part.id, redactString(part.text, remaining));
    },
    clear(sessionId) {
      for (const [key, state] of messages) {
        if (sessionId === undefined || state.sessionId === sessionId) messages.delete(key);
      }
    },
  };
}

export function createMessageSync({
  http,
  logger = console,
  limit = MAX_MESSAGES_PER_SESSION,
  maxSessions = MAX_SESSIONS,
  maxBufferChars = MAX_BUFFER_CHARS,
  flushConcurrency = MAX_CONCURRENT_FLUSHES,
  now = () => Date.now(),
}) {
  const sessions = new Map();
  const queues = new Map();
  let bufferedChars = 0;
  let lastDropWarningAt = -Infinity;
  let suppressedDropWarnings = 0;

  const warnDrop = (message) => {
    if (now() - lastDropWarningAt < WARNING_INTERVAL_MS) {
      suppressedDropWarnings += 1;
      return;
    }
    logger.warn?.(`${message}${suppressedDropWarnings ? ` (${suppressedDropWarnings} similar warning(s) suppressed)` : ""}`);
    lastDropWarningAt = now();
    suppressedDropWarnings = 0;
  };

  const deleteMessage = (state, messageId) => {
    const message = state.messages.get(messageId);
    if (!message) return false;
    bufferedChars = Math.max(0, bufferedChars - message.content.length);
    state.messages.delete(messageId);
    state.order = state.order.filter((id) => id !== messageId);
    return true;
  };

  const deleteSession = (sessionId) => {
    const state = sessions.get(sessionId);
    if (!state) return false;
    for (const message of state.messages.values()) {
      bufferedChars = Math.max(0, bufferedChars - message.content.length);
    }
    sessions.delete(sessionId);
    return true;
  };

  const stateFor = (sessionId) => {
    let state = sessions.get(sessionId);
    if (!state) {
      if (sessions.size >= maxSessions) {
        const evicted = [...sessions.keys()].find((candidate) => !queues.has(candidate));
        if (!evicted) {
          warnDrop(`oGMemory dropped an OpenCode message because all ${maxSessions} in-memory session slots are active`);
          return null;
        }
        deleteSession(evicted);
        warnDrop(`oGMemory dropped the oldest unsent OpenCode session at the ${maxSessions}-session in-memory limit`);
      }
      state = { order: [], messages: new Map() };
      sessions.set(sessionId, state);
    }
    return state;
  };

  const update = (sessionId, messageId, patch) => {
    if (!sessionId || !messageId) return;
    const state = stateFor(sessionId);
    if (!state) return;
    if (!state.messages.has(messageId)) state.order.push(messageId);
    const current = state.messages.get(messageId) || {
      id: messageId,
      role: "",
      content: "",
      generated: false,
      completed: false,
    };
    const next = {
      ...current,
      ...patch,
      content: patch.content === undefined
        ? current.content
        : mergeMessageContent(current.content, patch.content).slice(0, MAX_MESSAGE_CHARS),
    };
    state.messages.set(messageId, next);
    bufferedChars += next.content.length - current.content.length;
    while (state.order.length > limit) {
      const dropped = state.order.shift();
      const message = state.messages.get(dropped);
      bufferedChars = Math.max(0, bufferedChars - (message?.content.length || 0));
      state.messages.delete(dropped);
      warnDrop(`oGMemory dropped the oldest unsent OpenCode message at the ${limit}-message in-memory limit`);
    }
    while (bufferedChars > maxBufferChars) {
      const entry = [...sessions.entries()].find(([, candidate]) => candidate.order.length);
      if (!entry) break;
      const [candidateId, candidate] = entry;
      deleteMessage(candidate, candidate.order[0]);
      if (!candidate.order.length) sessions.delete(candidateId);
      warnDrop(`oGMemory dropped the oldest unsent OpenCode message at the ${maxBufferChars}-character global in-memory limit`);
    }
  };

  const snapshot = (sessionId) => {
    const state = sessions.get(sessionId);
    if (!state) return [];
    return state.order
      .map((id) => state.messages.get(id))
      .filter((message) => (
        message
        && !message.generated
        && ["user", "assistant"].includes(message.role)
        && (message.content || (message.role === "assistant" && message.completed))
      ))
      .map(({ id, role, content }) => ({ id, role, content }));
  };

  const enqueue = (sessionId, operation) => {
    const previous = queues.get(sessionId) || Promise.resolve();
    const next = previous.catch(() => {}).then(operation).finally(() => {
      if (queues.get(sessionId) === next) queues.delete(sessionId);
    });
    queues.set(sessionId, next);
    return next;
  };

  const flushNow = async (sessionId) => {
    const records = snapshot(sessionId);
    if (!records.length) return { success: true, count: 0, records: [], lastAssistant: "" };
    try {
      await http.afterTurn(sessionId, records.map(({ role, content }) => ({ role, content })));
      const state = sessions.get(sessionId);
      if (state) {
        const removed = new Set();
        for (const sent of records) {
          const current = state.messages.get(sent.id);
          if (current?.role === sent.role && current?.content === sent.content) {
            bufferedChars = Math.max(0, bufferedChars - current.content.length);
            state.messages.delete(sent.id);
            removed.add(sent.id);
          }
        }
        if (removed.size) state.order = state.order.filter((id) => !removed.has(id));
        if (!state.order.length) sessions.delete(sessionId);
      }
      return {
        success: true,
        count: records.length,
        records,
        lastAssistant: [...records].reverse().find((message) => message.role === "assistant")?.content || "",
      };
    } catch (error) {
      logger.warn?.(`oGMemory after_turn failed open; ${records.length} message(s) retained in memory: ${safeErrorMessage(error)}`);
      return {
        success: false,
        count: 0,
        records,
        lastAssistant: [...records].reverse().find((message) => message.role === "assistant")?.content || "",
      };
    }
  };

  const flush = (sessionId) => {
    if (!sessionId) return Promise.resolve({ success: false, count: 0, records: [], lastAssistant: "" });
    return enqueue(sessionId, () => flushNow(sessionId));
  };

  return {
    captureUser(sessionId, messageId, content) {
      update(sessionId, messageId, { role: "user", content });
    },
    observeMessage(info) {
      const sessionId = sessionIdOf(info);
      const isGenerated = generated(info);
      if (isGenerated) {
        const state = sessions.get(sessionId);
        if (state && deleteMessage(state, info?.id)) {
          if (!state.order.length) sessions.delete(sessionId);
        }
        return { generated: true };
      }
      update(sessionId, info?.id, {
        role: info?.role || "",
        generated: false,
        completed: completed(info),
      });
      return { generated: false };
    },
    observePart(part) {
      if (part?.type !== "text" || part?.synthetic || part?.ignored) return;
      update(sessionIdOf(part), part?.messageID || part?.messageId, { content: part?.text });
    },
    getMessage(sessionId, messageId) {
      return sessions.get(sessionId)?.messages.get(messageId) || null;
    },
    flush,
    async flushAll() {
      const sessionIds = [...sessions.keys()];
      const results = new Array(sessionIds.length);
      let nextIndex = 0;
      const worker = async () => {
        while (nextIndex < sessionIds.length) {
          const index = nextIndex;
          nextIndex += 1;
          results[index] = await flush(sessionIds[index]);
        }
      };
      await Promise.all(Array.from(
        { length: Math.min(flushConcurrency, sessionIds.length) },
        worker,
      ));
      return results;
    },
    clear(sessionId) {
      deleteSession(sessionId);
    },
  };
}
