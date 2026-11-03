import test from "node:test";
import assert from "node:assert/strict";
import { OgMemoryPlugin } from "../lib/plugin.mjs";

const config = Object.freeze({
  enabled: true,
  endpoint: "http://127.0.0.1:8090",
  accountId: "acct",
  userId: "user",
  agentId: "opencode",
  apiKey: "",
  composeTimeoutMs: 5000,
  writeTimeoutMs: 15000,
  traceTimeoutMs: 3000,
  traceCaptureContent: true,
});

function createHarness(overrides = {}) {
  const calls = { compose: [], afterTurn: [], trace: [] };
  const http = {
    compose: async (...args) => {
      calls.compose.push(args);
      return { messages: [{ role: "system", content: "remembered context", _ogmem: true }] };
    },
    afterTurn: async (...args) => { calls.afterTurn.push(args); return { ok: true }; },
    trace: async (envelope) => { calls.trace.push(envelope); return { ok: true }; },
    ...overrides,
  };
  return {
    calls,
    dependencies: {
      loadConfig: async () => config,
      createHttpClient: () => http,
      logger: { warn() {}, error() {}, info() {} },
    },
  };
}

async function completeAssistant(hooks, sessionId = "s1", messageId = "a1", text = "assistant answer") {
  await hooks.event({ event: {
    type: "message.part.updated",
    properties: { part: { sessionID: sessionId, messageID: messageId, type: "text", text } },
  } });
  await hooks.event({ event: {
    type: "message.updated",
    properties: { info: { sessionID: sessionId, id: messageId, role: "assistant", finish: "stop" } },
  } });
}

for (const [parentFirst, autoCapture] of [[true, true], [true, false], [false, true], [false, false]]) {
  test(`reasoning follows its parent: parentFirst=${parentFirst}, capture=${autoCapture}`, async () => {
    const { calls, dependencies } = createHarness();
    dependencies.loadConfig = async () => ({ ...config, autoCapture });
    const hooks = await OgMemoryPlugin({}, dependencies);
    const parent = { event: { type: "message.updated", properties: {
      info: { id: "a1", sessionID: "s1", role: "assistant" },
    } } };
    if (parentFirst) await hooks.event(parent);
    await hooks.event({ event: { type: "message.part.updated", properties: { part: {
      id: "r1", sessionID: "s1", messageID: "a1", type: "reasoning", text: "Inspect the contract first.", time: { end: 42 },
    } } } });
    if (!parentFirst) await hooks.event(parent);
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(calls.trace.find(item => item.event.event_type === "assistant_reasoning").event.reasoning,
      "Inspect the contract first.");
    await completeAssistant(hooks);
    await hooks.dispose();
    const reasoning = calls.trace.findIndex(item => item.event.event_type === "assistant_reasoning");
    const end = calls.trace.findIndex(item => item.event.event_type === "turn_end");
    if (autoCapture) assert.ok(end > reasoning);
  });
}

test("completed turn progresses while another session has an active trace upload", async () => {
  let releaseBusy;
  const busy = new Promise((resolve) => { releaseBusy = resolve; });
  const delivered = [];
  const { dependencies } = createHarness({ trace: async (envelope) => {
    if (envelope.session_id === "busy") await busy;
    delivered.push(envelope);
    return { ok: true };
  } });
  const hooks = await OgMemoryPlugin({}, dependencies);
  hooks["tool.execute.before"]({ sessionID: "busy", callID: "c1", tool: "read_file" }, { args: {} });
  let timer;
  try {
    await Promise.race([
      completeAssistant(hooks, "ready"),
      new Promise((_, reject) => { timer = setTimeout(() => reject(new Error("turn completion timed out")), 1000); }),
    ]);
    await new Promise((resolve) => setImmediate(resolve));
    assert.deepEqual(delivered.map(({ event }) => event.event_type), ["assistant_message", "turn_end"]);
  } finally {
    clearTimeout(timer);
    releaseBusy();
    await hooks.dispose();
  }
});

test("disabled configuration creates no runtime dependencies or hooks", async () => {
  let constructed = false;
  const hooks = await OgMemoryPlugin({}, {
    loadConfig: async () => ({ ...config, enabled: false }),
    createHttpClient: () => { constructed = true; },
  });
  assert.deepEqual(hooks, {});
  assert.equal(constructed, false);
});

test("invalid configuration disables the plugin without leaking file contents", async () => {
  const errors = [];
  const hooks = await OgMemoryPlugin({}, {
    loadConfig: async () => { throw new Error("invalid configuration"); },
    logger: { error: (message) => errors.push(message) },
  });
  assert.deepEqual(hooks, {});
  assert.match(errors[0], /plugin disabled/);
});

test("one OpenCode turn recalls context, injects it, and writes both roles", async () => {
  const { calls, dependencies } = createHarness();
  const hooks = await OgMemoryPlugin({}, dependencies);
  await hooks["chat.message"](
    { sessionID: "s1", messageID: "u1" },
    { message: { id: "u1" }, parts: [{ type: "text", text: "user question" }] },
  );
  const output = { system: [] };
  hooks["experimental.chat.system.transform"]({ sessionID: "s1" }, output);
  await completeAssistant(hooks);
  await hooks.dispose();

  assert.deepEqual(calls.compose[0], ["s1", "user question"]);
  assert.match(output.system[0], /remembered context/);
  assert.deepEqual(calls.afterTurn[0], ["s1", [
    { role: "user", content: "user question" },
    { role: "assistant", content: "assistant answer" },
  ]]);
  assert.ok(calls.trace.some(({ event }) => event.event_type === "user_message"));
  assert.ok(calls.trace.some(({ event }) => event.event_type === "assistant_message"));
  assert.ok(calls.trace.some(({ event }) => event.event_type === "turn_end"));
});

test("automatic capture and recall are independent across every write boundary", async (t) => {
  for (const autoCapture of [true, false]) {
    for (const autoRecall of [true, false]) {
      await t.test(`capture=${autoCapture}, recall=${autoRecall}`, async () => {
        const { calls, dependencies } = createHarness();
        dependencies.loadConfig = async () => ({ ...config, autoCapture, autoRecall });
        if (!autoCapture) dependencies.createMessageSync = () => {
          assert.fail("capture disabled must not create a message buffer or retry queue");
        };
        const hooks = await OgMemoryPlugin({}, dependencies);
        await hooks.event({ event: { type: "session.created", properties: { info: { id: "s1" } } } });
        const output = { system: ["existing system prompt"] };
        await hooks["chat.message"](
          { sessionID: "s1", messageID: "u1" },
          { message: { id: "u1" }, parts: [{ type: "text", text: "user question" }] },
        );
        hooks["experimental.chat.system.transform"]({ sessionID: "s1" }, output);
        assert.equal(calls.compose.length, autoRecall ? 1 : 0);
        if (autoRecall) assert.match(output.system[1], /remembered context/);
        else assert.deepEqual(output.system, ["existing system prompt"]);
        await completeAssistant(hooks);
        assert.equal(calls.afterTurn.length, autoCapture ? 1 : 0);
        if (autoCapture) assert.deepEqual(calls.afterTurn[0][1], [
          { role: "user", content: "user question" },
          { role: "assistant", content: "assistant answer" },
        ]);

        // Exercise pending-message flushes at each remaining lifecycle boundary.
        for (const boundary of ["compact", "session.idle", "session.error", "session.deleted", "dispose"]) {
          await hooks["chat.message"](
            { sessionID: "s1", messageID: `u-${boundary}` },
            { parts: [{ type: "text", text: "pending user message" }] },
          );
          if (boundary === "compact") await hooks["experimental.session.compacting"]({ sessionID: "s1" });
          else if (boundary === "dispose") await hooks.dispose();
          else await hooks.event({ event: { type: boundary, properties: { sessionID: "s1" } } });
        }
        assert.equal(calls.afterTurn.length, autoCapture ? 6 : 0);
        assert.equal(calls.compose.length, autoRecall ? 6 : 0);
        assert.ok(calls.trace.some(({ event }) => event.event_type === "session_start"));
        assert.ok(calls.trace.some(({ event }) => event.event_type === "session_end"));
      });
    }
  }
});

test("capture disabled preserves tool traces including failure events", async () => {
  const { calls, dependencies } = createHarness();
  dependencies.loadConfig = async () => ({ ...config, autoCapture: false });
  const hooks = await OgMemoryPlugin({}, dependencies);
  hooks["tool.execute.before"]({ sessionID: "s1", callID: "c1", tool: "bash" }, { args: {} });
  hooks["tool.execute.after"]({ sessionID: "s1", callID: "c1", tool: "bash" }, { output: "ok" });
  await hooks.event({ event: {
    type: "message.part.updated",
    properties: { part: { sessionID: "s1", type: "tool", callID: "c2", tool: "bash", state: { status: "error" } } },
  } });
  await hooks.dispose();
  assert.deepEqual(calls.trace.map(({ event }) => event.event_type), ["tool_start", "tool_success", "tool_failure"]);
  assert.equal(calls.afterTurn.length, 0);
});

test("large multi-part prompts are bounded before compose and write buffering", async () => {
  const { calls, dependencies } = createHarness();
  const hooks = await OgMemoryPlugin({}, dependencies);
  await hooks["chat.message"](
    { sessionID: "s1", messageID: "u1" },
    {
      message: { id: "u1" },
      parts: [
        { type: "text", text: "a".repeat(75000) },
        { type: "text", text: "b".repeat(75000) },
      ],
    },
  );

  assert.equal(calls.compose[0][1].length, 100000);
});

test("a completed empty assistant is written and traced as a turn", async () => {
  const { calls, dependencies } = createHarness();
  const hooks = await OgMemoryPlugin({}, dependencies);
  await hooks["chat.message"](
    { sessionID: "s1", messageID: "u1" },
    { message: { id: "u1" }, parts: [{ type: "text", text: "tool-only prompt" }] },
  );
  await hooks.event({ event: {
    type: "message.updated",
    properties: { info: { sessionID: "s1", id: "a1", role: "assistant", finish: "stop" } },
  } });
  await hooks.dispose();

  assert.deepEqual(calls.afterTurn[0][1], [
    { role: "user", content: "tool-only prompt" },
    { role: "assistant", content: "" },
  ]);
  assert.ok(calls.trace.some(({ event }) => event.event_type === "assistant_message"));
  assert.ok(calls.trace.some(({ event }) => event.event_type === "turn_end"));
});

test("parent-child session identity is recorded in canonical Trace", async () => {
  const { calls, dependencies } = createHarness();
  const hooks = await OgMemoryPlugin({}, dependencies);

  await hooks.event({ event: {
    type: "session.created",
    properties: { info: { id: "child", parentID: "parent" } },
  } });
  await hooks.dispose();

  const start = calls.trace.find(({ event }) => event.event_type === "session_start");
  assert.equal(start.session_id, "child");
  assert.equal(start.event.parent_session_id, "parent");
});

test("tool Trace submission never blocks the OpenCode hook", async () => {
  let resolveTrace;
  const pendingTrace = new Promise((resolve) => { resolveTrace = resolve; });
  const { calls, dependencies } = createHarness({ trace: async (event) => { calls.trace.push(event); await pendingTrace; } });
  const hooks = await OgMemoryPlugin({}, dependencies);

  const result = hooks["tool.execute.before"](
    { sessionID: "s1", callID: "call-1", tool: "bash" },
    { args: { command: "pwd" } },
  );
  assert.equal(result, undefined);
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(calls.trace.length, 1);
  resolveTrace();
  await hooks.dispose();
});

test("a failed after_turn is retained and retried on session idle", async () => {
  let attempts = 0;
  const { calls, dependencies } = createHarness({
    afterTurn: async (...args) => {
      calls.afterTurn.push(args);
      attempts += 1;
      if (attempts === 1) throw new Error("offline");
      return { ok: true };
    },
  });
  const hooks = await OgMemoryPlugin({}, dependencies);
  await hooks["chat.message"](
    { sessionID: "s1", messageID: "u1" },
    { message: { id: "u1" }, parts: [{ type: "text", text: "retry me" }] },
  );
  await completeAssistant(hooks);
  await hooks.event({ event: { type: "session.idle", properties: { sessionID: "s1" } } });

  assert.equal(attempts, 2);
  assert.deepEqual(calls.afterTurn[0], calls.afterTurn[1]);
});

test("synthetic assistant summaries are excluded from write-back", async () => {
  const { calls, dependencies } = createHarness();
  const hooks = await OgMemoryPlugin({}, dependencies);
  await hooks.event({ event: {
    type: "message.part.updated",
    properties: { part: { sessionID: "s1", messageID: "summary", type: "text", text: "generated" } },
  } });
  await hooks.event({ event: {
    type: "message.updated",
    properties: { info: { sessionID: "s1", id: "summary", role: "assistant", summary: true, finish: "stop" } },
  } });
  assert.equal(calls.afterTurn.length, 0);
});

test("a generated summary does not flush an unfinished real turn", async () => {
  const { calls, dependencies } = createHarness();
  const hooks = await OgMemoryPlugin({}, dependencies);
  await hooks["chat.message"](
    { sessionID: "s1", messageID: "u1" },
    { message: { id: "u1" }, parts: [{ type: "text", text: "real prompt" }] },
  );
  await hooks.event({ event: {
    type: "message.part.updated",
    properties: { part: { sessionID: "s1", messageID: "summary", type: "text", text: "generated" } },
  } });
  await hooks.event({ event: {
    type: "message.updated",
    properties: { info: { sessionID: "s1", id: "summary", role: "assistant", summary: true, finish: "stop" } },
  } });

  assert.equal(calls.afterTurn.length, 0);
  await completeAssistant(hooks);
  assert.deepEqual(calls.afterTurn[0][1], [
    { role: "user", content: "real prompt" },
    { role: "assistant", content: "assistant answer" },
  ]);
});

test("repeated session errors without upstream IDs create distinct Trace events", async () => {
  const { calls, dependencies } = createHarness();
  const hooks = await OgMemoryPlugin({}, dependencies);
  const event = { type: "session.error", properties: { sessionID: "s1", error: { message: "failed" } } };

  await hooks.event({ event });
  await hooks.event({ event });
  await hooks.dispose();

  const errors = calls.trace.filter(({ event: traceEvent }) => traceEvent.event_type === "session_error");
  assert.equal(errors.length, 2);
  assert.notEqual(errors[0].request_id, errors[1].request_id);
});

test("session deletion flushes pending messages and records session_end", async () => {
  const { calls, dependencies } = createHarness();
  const hooks = await OgMemoryPlugin({}, dependencies);
  await hooks["chat.message"](
    { sessionID: "s1", messageID: "u1" },
    { message: { id: "u1" }, parts: [{ type: "text", text: "last message" }] },
  );
  await hooks.event({ event: { type: "session.deleted", properties: { sessionID: "s1" } } });
  await hooks.dispose();

  assert.equal(calls.afterTurn.length, 1);
  assert.ok(calls.trace.some(({ event }) => event.event_type === "session_end"));
});

test("session deletion retains a failed write for dispose retry", async () => {
  let attempts = 0;
  const { calls, dependencies } = createHarness({
    afterTurn: async (...args) => {
      calls.afterTurn.push(args);
      attempts += 1;
      if (attempts === 1) throw new Error("offline");
      return { ok: true };
    },
  });
  const hooks = await OgMemoryPlugin({}, dependencies);
  await hooks["chat.message"](
    { sessionID: "s1", messageID: "u1" },
    { message: { id: "u1" }, parts: [{ type: "text", text: "final message" }] },
  );

  await hooks.event({ event: { type: "session.deleted", properties: { sessionID: "s1" } } });
  await hooks.dispose();

  assert.equal(attempts, 2);
  assert.deepEqual(calls.afterTurn[0], calls.afterTurn[1]);
});
