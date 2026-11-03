export class HttpError extends Error {
  constructor(message, { status = 0, retryable = false } = {}) {
    super(message);
    this.name = "HttpError";
    this.status = status;
    this.retryable = retryable;
  }
}

function retryableStatus(status) {
  return status === 408 || status === 429 || status >= 500;
}

function retryableOversizedResponse(response) {
  return response?.ok === true || (response?.status >= 200 && response?.status < 300) || retryableStatus(response?.status);
}

async function readBoundedText(response, maxBytes) {
  const declared = Number(response.headers?.get?.("content-length"));
  if (Number.isFinite(declared) && declared > maxBytes) {
    await response.body?.cancel?.().catch(() => {});
    throw new HttpError(`oGMemory response exceeded ${maxBytes} bytes`, { status: response.status, retryable: retryableOversizedResponse(response) });
  }
  if (!response.body?.getReader) {
    const text = await response.text();
    if (Buffer.byteLength(text) > maxBytes) throw new HttpError(`oGMemory response exceeded ${maxBytes} bytes`, { status: response.status, retryable: retryableOversizedResponse(response) });
    return text;
  }
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      received += value.byteLength;
      if (received > maxBytes) throw new HttpError(`oGMemory response exceeded ${maxBytes} bytes`, { status: response.status, retryable: retryableOversizedResponse(response) });
      chunks.push(value);
    }
  } catch (error) {
    await reader.cancel().catch(() => {});
    throw error;
  }
  const merged = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) { merged.set(chunk, offset); offset += chunk.byteLength; }
  return new TextDecoder().decode(merged);
}

export function createHttpClient(config, fetchImpl = globalThis.fetch) {
  if (typeof fetchImpl !== "function") throw new Error("A Fetch API implementation is required");
  const identity = (sessionId) => ({
    accountId: config.accountId,
    userId: config.userId,
    agentId: config.agentId,
    sessionId,
  });

  async function request(path, body, timeoutMs, maxResponseBytes) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const headers = { "content-type": "application/json" };
      if (config.apiKey) headers["X-API-Key"] = config.apiKey;
      if (config.accountId) headers["X-Account-ID"] = config.accountId;
      if (config.userId) headers["X-User-ID"] = config.userId;
      const response = await fetchImpl(`${config.endpoint}${path}`, {
        method: "POST",
        headers,
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      const text = await readBoundedText(response, maxResponseBytes);
      let data = {};
      if (text) {
        try { data = JSON.parse(text); }
        catch {
          if (!response.ok) throw new HttpError(`HTTP ${response.status}`, {
            status: response.status,
            retryable: retryableStatus(response.status),
          });
          throw new HttpError("oGMemory returned invalid JSON", { status: response.status, retryable: true });
        }
      }
      if (!response.ok) throw new HttpError(`oGMemory request failed with HTTP ${response.status}`, {
        status: response.status,
        retryable: retryableStatus(response.status),
      });
      return data;
    } catch (error) {
      if (error instanceof HttpError) throw error;
      const timedOut = error?.name === "AbortError";
      throw new HttpError(timedOut ? `Request timed out after ${timeoutMs}ms` : error.message, { retryable: true });
    } finally {
      clearTimeout(timer);
    }
  }

  return {
    compose: (sessionId, prompt) => request("/api/v1/compose", { ...identity(sessionId), prompt, messages: [{ role: "user", content: prompt }] }, config.composeTimeoutMs, 1024 * 1024),
    afterTurn: async (sessionId, messages) => {
      const result = await request("/api/v1/after_turn", { ...identity(sessionId), messages }, config.writeTimeoutMs, 64 * 1024);
      if (result?.ok === false) throw new HttpError("oGMemory after_turn rejected payload", { retryable: result.retryable !== false });
      if (result?.ok !== true) throw new HttpError("oGMemory after_turn returned no success acknowledgement", { retryable: true });
      return result;
    },
    trace: async (envelope) => {
      const result = await request("/api/v1/trace/events", envelope, config.traceTimeoutMs, 64 * 1024);
      if (result?.ok === false) throw new HttpError("oGMemory trace rejected payload", { retryable: result.retryable !== false });
      if (result?.ok !== true) throw new HttpError("oGMemory trace returned no success acknowledgement", { retryable: true });
      return result;
    },
  };
}
