import { createComposeController } from "./compose.mjs";
import { loadConfig } from "./config.mjs";
import { createHttpClient } from "./http-client.mjs";
import { createMessageSync, createReasoningCapture } from "./message-sync.mjs";
import { safeErrorMessage } from "./redaction.mjs";
import { createTraceAdapter } from "./trace-adapter.mjs";

const DISPOSE_TIMEOUT_MS = 3000;
const MAX_PROMPT_CHARS = 100000;

function sessionIdOf(event) {
  const properties = event?.properties || {};
  return properties.sessionID
    || properties.sessionId
    || properties.part?.sessionID
    || properties.part?.sessionId
    || properties.info?.sessionID
    || properties.info?.sessionId
    || properties.info?.id
    || "";
}

function textParts(output) {
  let content = "";
  for (const part of Array.isArray(output?.parts) ? output.parts : []) {
    if (part?.type !== "text" || part.synthetic || part.ignored || typeof part.text !== "string") continue;
    const separator = content ? "\n" : "";
    const remaining = MAX_PROMPT_CHARS - content.length - separator.length;
    if (remaining <= 0) break;
    content += separator + part.text.slice(0, remaining);
  }
  return content.trim();
}

function parentSessionIdOf(info) {
  return info?.parentID || info?.parentId || info?.parentSessionID || info?.parentSessionId || "";
}

function completed(info) {
  return Boolean(info?.time?.completed || ["stop", "end_turn"].includes(info?.finish));
}

export async function OgMemoryPlugin({ client } = {}, dependencies = {}) {
  const logger = dependencies.logger || console;
  let config;
  try {
    config = await (dependencies.loadConfig || loadConfig)(dependencies.configOptions);
  } catch (error) {
    logger.error?.(`oGMemory plugin disabled because configuration is invalid: ${safeErrorMessage(error)}`);
    return {};
  }
  if (!config.enabled) return {};

  const http = (dependencies.createHttpClient || createHttpClient)(config, dependencies.fetch);
  const compose = (dependencies.createComposeController || createComposeController)({ http, logger });
  const sync = config.autoCapture === false
    ? null
    : (dependencies.createMessageSync || createMessageSync)({ http, logger });
  const trace = (dependencies.createTraceAdapter || createTraceAdapter)({ config, http, logger });
  const reasoning = createReasoningCapture({ trace });
  let sessionErrorSequence = 0;

  const traceCommittedMessages = async (sessionId, result, terminalState = "ok") => {
    for (const message of result.records || []) {
      trace.emit(`${message.role}_message`, sessionId, { objectId: message.id, content: message.content });
    }
    await trace.flush(sessionId);
    const assistant = [...(result.records || [])].reverse().find((message) => message.role === "assistant");
    if (assistant) trace.emit("turn_end", sessionId, {
      objectId: assistant.id,
      terminalState,
      content: assistant.content,
    });
  };

  const flushSession = async (sessionId, terminalState = "ok") => {
    if (!sync) return { success: true, count: 0, records: [] };
    const result = await sync.flush(sessionId);
    await traceCommittedMessages(sessionId, result, terminalState);
    return result;
  };

  return {
    "chat.message": async (input, output) => {
      const sessionId = input?.sessionID || input?.sessionId;
      const messageId = output?.message?.id || input?.messageID || input?.messageId;
      const prompt = textParts(output);
      sync?.captureUser(sessionId, messageId, prompt);
      if (config.autoRecall !== false) await compose.recall(sessionId, messageId, prompt);
    },

    "experimental.chat.system.transform": (input, output) => {
      if (config.autoRecall !== false) compose.inject(input?.sessionID || input?.sessionId, output);
    },

    "tool.execute.before": (input, output) => {
      trace.emit("tool_start", input?.sessionID || input?.sessionId, {
        objectId: input?.callID || input?.callId,
        content: output?.args,
        metadata: { tool: input?.tool },
      });
    },

    "tool.execute.after": (input, output) => {
      trace.emit("tool_success", input?.sessionID || input?.sessionId, {
        objectId: input?.callID || input?.callId,
        terminalState: "success",
        content: output,
        metadata: { tool: input?.tool },
      });
    },

    "experimental.session.compacting": async (input) => {
      const sessionId = input?.sessionID || input?.sessionId;
      const result = await flushSession(sessionId);
      trace.emit("pre_compact", sessionId, {
        objectId: sessionId,
        metadata: { message_count: result.count },
      });
    },

    event: async ({ event } = {}) => {
      const sessionId = sessionIdOf(event);
      if (!sessionId) return;
      const properties = event.properties || {};

      if (event.type === "session.created") {
        const parentSessionId = parentSessionIdOf(properties.info);
        trace.emit("session_start", sessionId, {
          objectId: sessionId,
          metadata: parentSessionId ? { parent_session_id: parentSessionId } : {},
        });
        return;
      }

      if (event.type === "message.updated") {
        reasoning.observeMessage(properties.info);
        if (!sync) return;
        const info = properties.info;
        const observation = sync.observeMessage(info);
        if (!observation.generated && info?.role === "assistant" && completed(info)) await flushSession(sessionId);
        return;
      }

      if (event.type === "message.part.updated") {
        const part = properties.part;
        sync?.observePart(part);
        reasoning.observePart(part);
        if (part?.type === "tool" && part?.state?.status === "error") {
          trace.emit("tool_failure", sessionId, {
            objectId: part.callID || part.callId || part.id,
            terminalState: "error",
            content: part.state,
            metadata: { tool: part.tool },
          });
        }
        return;
      }

      if (event.type === "session.idle") {
        await flushSession(sessionId);
        compose.clear(sessionId);
        return;
      }

      if (event.type === "session.error") {
        trace.emit("session_error", sessionId, {
          objectId: properties.error?.id || `${sessionId}:error:${++sessionErrorSequence}`,
          terminalState: "error",
          content: properties.error,
        });
        await flushSession(sessionId, "error");
        return;
      }

      if (event.type === "session.deleted") {
        reasoning.clear(sessionId);
        const result = await flushSession(sessionId);
        trace.emit("session_end", sessionId, { objectId: sessionId, terminalState: "deleted" });
        if (result.success) sync?.clear(sessionId);
        compose.clear(sessionId);
      }
    },

    dispose: async () => {
      reasoning.clear();
      let timer;
      await Promise.race([
        (async () => {
          const results = sync ? await sync.flushAll() : [];
          for (const result of results) {
            if (!result.success) logger.warn?.("oGMemory dispose left unsent messages in memory");
          }
          await trace.flush();
        })(),
        new Promise((resolve) => { timer = setTimeout(resolve, DISPOSE_TIMEOUT_MS); }),
      ]).catch((error) => logger.warn?.(`oGMemory dispose failed open: ${safeErrorMessage(error)}`))
        .finally(() => clearTimeout(timer));
    },
  };
}

export default OgMemoryPlugin;
