import test from "node:test";
import assert from "node:assert/strict";
import { createMessageSync, createReasoningCapture, mergeMessageContent } from "../lib/message-sync.mjs";

test("pending reasoning retains recent messages within a shared per-message content budget", () => {
  const captured = [];
  const reasoning = createReasoningCapture({ trace: { emit: (...args) => captured.push(args) },
    limit: 2, maxParts: 2, maxChars: 8 });
  for (const messageID of ["a1", "a2", "a3"]) {
    for (const id of ["r1", "r2"]) reasoning.observePart({
      sessionID: "s1", messageID, id: `${messageID}-${id}`, type: "reasoning", text: "12345", time: { end: 42 },
    });
  }
  for (const id of ["a2", "a3"]) reasoning.observeMessage({ sessionID: "s1", id, role: "assistant" });
  assert.deepEqual(captured.map(([, sessionId, { objectId, content }]) => [sessionId, objectId, content]), [
    ["s1", "a2-r1", "12345"], ["s1", "a2-r2", "123"],
    ["s1", "a3-r1", "12345"], ["s1", "a3-r2", "123"],
  ]);
});

test("message content updates replace prefixes instead of duplicating streamed text", () => {
  assert.equal(mergeMessageContent("hel", "hello"), "hello");
  assert.equal(mergeMessageContent("hello", "hel"), "hello");
  assert.equal(mergeMessageContent("first", "second"), "first\nsecond");
});

test("live OpenCode events flush user and assistant messages to after_turn", async () => {
  const calls = [];
  const sync = createMessageSync({ http: { afterTurn: async (...args) => calls.push(args) } });
  sync.captureUser("s1", "u1", "What is stored?");
  sync.observePart({ sessionID: "s1", messageID: "a1", type: "text", text: "It is stored." });
  sync.observeMessage({ sessionID: "s1", id: "a1", role: "assistant", finish: "stop" });

  const result = await sync.flush("s1");

  assert.equal(result.success, true);
  assert.equal(result.lastAssistant, "It is stored.");
  assert.deepEqual(calls, [["s1", [
    { role: "user", content: "What is stored?" },
    { role: "assistant", content: "It is stored." },
  ]]]);
});

test("failed writes stay in memory and are retried at the next boundary", async () => {
  let attempts = 0;
  const payloads = [];
  const sync = createMessageSync({
    http: {
      afterTurn: async (_sessionId, messages) => {
        attempts += 1;
        payloads.push(messages);
        if (attempts === 1) throw new Error("offline");
      },
    },
    logger: { warn() {} },
  });
  sync.captureUser("s1", "u1", "remember me");

  assert.equal((await sync.flush("s1")).success, false);
  assert.equal((await sync.flush("s1")).success, true);
  assert.equal(attempts, 2);
  assert.deepEqual(payloads[0], payloads[1]);
  assert.equal((await sync.flush("s1")).count, 0);
});

test("synthetic and compaction summary messages are not written", async () => {
  const calls = [];
  const sync = createMessageSync({ http: { afterTurn: async (...args) => calls.push(args) } });
  sync.observePart({ sessionID: "s1", messageID: "a1", type: "text", text: "generated summary" });
  sync.observeMessage({ sessionID: "s1", id: "a1", role: "assistant", summary: true });
  sync.captureUser("s1", "u1", "real user");

  await sync.flush("s1");

  assert.deepEqual(calls[0][1], [{ role: "user", content: "real user" }]);
});

test("generated messages are removed from the bounded in-memory buffer", async () => {
  const calls = [];
  const warnings = [];
  const sync = createMessageSync({
    http: { afterTurn: async (...args) => calls.push(args) },
    logger: { warn: (message) => warnings.push(message) },
    limit: 1,
  });
  sync.observePart({ sessionID: "s1", messageID: "summary", type: "text", text: "generated" });
  sync.observeMessage({ sessionID: "s1", id: "summary", role: "assistant", summary: true });
  sync.captureUser("s1", "u1", "real user");

  await sync.flush("s1");

  assert.deepEqual(calls[0][1], [{ role: "user", content: "real user" }]);
  assert.equal(warnings.length, 0);
});

test("per-session flushes remain isolated", async () => {
  const calls = [];
  const sync = createMessageSync({ http: { afterTurn: async (...args) => calls.push(args) } });
  sync.captureUser("s1", "u1", "one");
  sync.captureUser("s2", "u2", "two");

  await Promise.all([sync.flush("s1"), sync.flush("s2")]);

  assert.deepEqual(new Map(calls.map(([id, messages]) => [id, messages[0].content])), new Map([["s1", "one"], ["s2", "two"]]));
});

test("the in-memory buffer is bounded and drops the oldest unsent message", async () => {
  const warnings = [];
  const calls = [];
  const sync = createMessageSync({
    http: { afterTurn: async (...args) => calls.push(args) },
    logger: { warn: (message) => warnings.push(message) },
    limit: 2,
  });
  sync.captureUser("s1", "u1", "one");
  sync.captureUser("s1", "u2", "two");
  sync.captureUser("s1", "u3", "three");

  await sync.flush("s1");

  assert.deepEqual(calls[0][1].map((message) => message.content), ["two", "three"]);
  assert.equal(warnings.length, 1);
});

test("concurrent flush requests serialize and do not write the same message twice", async () => {
  let release;
  const gate = new Promise((resolve) => { release = resolve; });
  let calls = 0;
  const sync = createMessageSync({
    http: { afterTurn: async () => { calls += 1; await gate; } },
  });
  sync.captureUser("s1", "u1", "one");

  const first = sync.flush("s1");
  const second = sync.flush("s1");
  release();
  await Promise.all([first, second]);

  assert.equal(calls, 1);
});

test("global session retention and shutdown write concurrency are bounded", async () => {
  const warnings = [];
  let active = 0;
  let maxActive = 0;
  const calls = [];
  const sync = createMessageSync({
    http: {
      afterTurn: async (sessionId) => {
        calls.push(sessionId);
        active += 1;
        maxActive = Math.max(maxActive, active);
        await new Promise((resolve) => setImmediate(resolve));
        active -= 1;
      },
    },
    logger: { warn: (message) => warnings.push(message) },
    maxSessions: 3,
    flushConcurrency: 2,
  });
  for (let index = 0; index < 4; index += 1) {
    sync.captureUser(`s${index}`, `u${index}`, `message ${index}`);
  }

  await sync.flushAll();

  assert.deepEqual(calls.sort(), ["s1", "s2", "s3"]);
  assert.equal(maxActive, 2);
  assert.equal(warnings.length, 1);
});

test("global buffered content has a finite budget", async () => {
  const calls = [];
  const warnings = [];
  const sync = createMessageSync({
    http: { afterTurn: async (...args) => calls.push(args) },
    logger: { warn: (message) => warnings.push(message) },
    maxBufferChars: 8,
  });
  sync.captureUser("s1", "u1", "12345");
  sync.captureUser("s2", "u2", "67890");

  await sync.flushAll();

  assert.deepEqual(calls, [["s2", [{ role: "user", content: "67890" }]]]);
  assert.equal(warnings.length, 1);
});
