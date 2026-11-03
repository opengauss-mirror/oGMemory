import test from "node:test";
import assert from "node:assert/strict";
import { createHttpClient, HttpError } from "../lib/http-client.mjs";

const config = {
  endpoint: "http://127.0.0.1:8090",
  apiKey: "key",
  accountId: "acct",
  userId: "user",
  agentId: "opencode",
  composeTimeoutMs: 100,
  writeTimeoutMs: 100,
  traceTimeoutMs: 100,
};

test("compose sends identity, prompt, and authentication headers", async () => {
  const calls = [];
  const client = createHttpClient(config, async (url, options) => {
    calls.push({ url, options });
    return new Response(JSON.stringify({ messages: [] }), { status: 200 });
  });
  await client.compose("s1", "hello");
  const body = JSON.parse(calls[0].options.body);
  assert.equal(calls[0].url, "http://127.0.0.1:8090/api/v1/compose");
  assert.equal(calls[0].options.headers["X-API-Key"], "key");
  assert.equal(body.sessionId, "s1");
  assert.equal(body.prompt, "hello");
});

test("after_turn and Trace require an explicit success acknowledgement", async () => {
  const missingAck = createHttpClient(config, async () => new Response("{}", { status: 200 }));
  await assert.rejects(missingAck.afterTurn("s1", [{ role: "user", content: "x" }]), /no success acknowledgement/);
  await assert.rejects(missingAck.trace({ request_id: "r", session_id: "s1", event: {} }), /no success acknowledgement/);

  const accepted = createHttpClient(config, async () => new Response('{"ok":true}', { status: 200 }));
  assert.equal((await accepted.afterTurn("s1", [{ role: "user", content: "x" }])).ok, true);
});

test("request timeout aborts the underlying fetch", async () => {
  let aborted = false;
  const client = createHttpClient({ ...config, composeTimeoutMs: 20 }, async (_url, { signal }) => new Promise((_resolve, reject) => {
    signal.addEventListener("abort", () => {
      aborted = true;
      reject(Object.assign(new Error("aborted"), { name: "AbortError" }));
    });
  }));
  await assert.rejects(client.compose("s1", "hello"), /timed out/);
  assert.equal(aborted, true);
});

test("oversized response bodies are rejected before JSON parsing", async () => {
  const client = createHttpClient(config, async () => new Response("x".repeat(1024 * 1024 + 1), { status: 200 }));
  await assert.rejects(client.compose("s1", "hello"), (error) => error instanceof HttpError && /exceeded/.test(error.message));
});

test("HTTP status errors expose only bounded status semantics", async () => {
  const client = createHttpClient(config, async () => new Response('{"secret":"do-not-log"}', { status: 503 }));
  await assert.rejects(client.compose("s1", "hello"), (error) => {
    assert.equal(error.status, 503);
    assert.equal(error.retryable, true);
    assert.doesNotMatch(error.message, /do-not-log/);
    return true;
  });
});
