import test from "node:test";
import assert from "node:assert/strict";
import { redact, redactString } from "../lib/redaction.mjs";
import { createTraceAdapter, stableEventId } from "../lib/trace-adapter.mjs";

const config = {
  accountId: "acct",
  userId: "user",
  agentId: "opencode",
  traceCaptureContent: true,
};

test("Trace identifiers are stable for duplicate lifecycle delivery", () => {
  assert.equal(
    stableEventId("turn_end", "session", "message", "ok"),
    stableEventId("turn_end", "session", "message", "ok"),
  );
  assert.notEqual(
    stableEventId("turn_end", "session", "message", "ok"),
    stableEventId("turn_end", "session", "message", "error"),
  );
});

test("Trace content is redacted before HTTP delivery", async () => {
  const sent = [];
  const trace = createTraceAdapter({
    config,
    http: { trace: async (event) => sent.push(event) },
  });
  trace.emit("tool_start", "s1", {
    objectId: "call",
    content: { apiKey: "secret-value", command: "--token abcdefghijklmnop" },
    metadata: { tool: "bash" },
  });
  await trace.flush();
  const serialized = JSON.stringify(sent[0]);
  assert.doesNotMatch(serialized, /secret-value|abcdefghijklmnop/);
  assert.match(serialized, /REDACTED/);
});

test("parent session identity is preserved as bounded canonical metadata", async () => {
  const sent = [];
  const trace = createTraceAdapter({
    config,
    http: { trace: async (event) => sent.push(event) },
  });
  trace.emit("session_start", "child", { objectId: "child", metadata: { parent_session_id: "parent" } });
  await trace.flush();
  assert.equal(sent[0].event.parent_session_id, "parent");
});

test("Trace delivery failures are fail-open and observable", async () => {
  const warnings = [];
  const trace = createTraceAdapter({
    config,
    http: { trace: async () => { throw new Error("offline"); } },
    logger: { warn: (message) => warnings.push(message) },
  });
  assert.equal(trace.emit("session_start", "s1", { objectId: "s1" }), true);
  await trace.flush();
  assert.match(warnings[0], /failed open/);
});

test("session flush completes its submitted snapshot while later deliveries continue", async () => {
  const releases = new Map();
  const delivered = [];
  const trace = createTraceAdapter({
    config,
    http: { trace: (envelope) => new Promise((resolve) => {
      releases.set(envelope.event.reasoning, () => {
        delivered.push(envelope.event.reasoning);
        resolve();
      });
    }) },
  });
  trace.emit("assistant_reasoning", "busy", { objectId: "b1", content: "busy" });
  trace.emit("assistant_reasoning", "ready", { objectId: "a1", content: "submitted" });
  const flushed = trace.flush("ready");
  trace.emit("assistant_reasoning", "ready", { objectId: "a2", content: "later" });
  await new Promise((resolve) => setImmediate(resolve));
  let timer;
  try {
    releases.get("submitted")();
    await Promise.race([
      flushed,
      new Promise((_, reject) => { timer = setTimeout(() => reject(new Error("snapshot flush timed out")), 1000); }),
    ]);
    assert.deepEqual(delivered, ["submitted"]);
  } finally {
    clearTimeout(timer);
    releases.get("busy")();
    releases.get("later")();
    await trace.flush();
  }
  assert.deepEqual(delivered, ["submitted", "busy", "later"]);
});

test("Trace in-flight work is bounded and drops excess events fail-open", async () => {
  let release;
  const gate = new Promise((resolve) => { release = resolve; });
  const warnings = [];
  let calls = 0;
  const trace = createTraceAdapter({
    config,
    http: { trace: async () => { calls += 1; await gate; } },
    logger: { warn: (message) => warnings.push(message) },
    maxPending: 2,
  });

  assert.equal(trace.emit("session_start", "s1"), true);
  assert.equal(trace.emit("session_start", "s2"), true);
  assert.equal(trace.emit("session_start", "s3"), false);
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(calls, 2);
  assert.equal(warnings.length, 1);
  release();
  await trace.flush();
});

test("redaction handles secret keys, credentials, and cycles", () => {
  const value = { token: "abc", nested: { password: "xyz" } };
  value.self = value;
  const result = redact(value);
  assert.equal(result.token, "[REDACTED]");
  assert.equal(result.nested.password, "[REDACTED]");
  assert.equal(result.self, "[REPEATED_REFERENCE]");
  assert.doesNotMatch(redactString("https://user:pass@example.com"), /user:pass/);
});
