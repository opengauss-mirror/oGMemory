import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  openClawHookToEvent,
  postTraceEvent,
  redactTraceValue,
  registerOpenClawTraceHooks,
} from "./trace.js";
import plugin, { resolveConfig } from "./index.js";

const fixtureUrl = new URL("../../tests/fixtures/trace/redaction_cases.json", import.meta.url);
const redactionCases = JSON.parse(await readFile(fixtureUrl, "utf8"));
const pluginDir = dirname(fileURLToPath(import.meta.url));
const flushTraceUploads = () => new Promise((resolve) => setImmediate(resolve));

test("retains model text and exposed thinking from the completed turn", () => {
  const event = openClawHookToEvent("agent_end", { success: true, messages: [
    { role: "user", content: "Check the limit" },
    { role: "assistant", content: [{ type: "thinking", thinking: "Inspect the API contract." }, { type: "text", text: "Reading docs." }] },
    { role: "tool", content: "The limit is 100." },
    { role: "assistant", content: "Use batches of 100." },
  ] });
  assert.deepEqual(event.assistant_messages, [
    { text: "Reading docs.", reasoning: "Inspect the API contract." },
    { text: "Use batches of 100.", reasoning: "" },
  ]);
});

test("local defaults resolve from the repository root after plugin migration", () => {
  const repoRoot = dirname(dirname(pluginDir));
  const config = resolveConfig({ mode: "local" });

  assert.equal(config.projectRoot, repoRoot);
  assert.equal(config.agfsServerDir, join(repoRoot, "agfs"));
  assert.equal(config.agfsDataDir, join(repoRoot, "..", "data"));
});

test("maps supported OpenClaw hooks to canonical event types", () => {
  const cases = [
    ["session_start", { sessionId: "s" }, "session_start"],
    ["before_agent_run", { prompt: "hello", messages: [] }, "user_message"],
    ["before_tool_call", { toolName: "exec", params: {} }, "tool_start"],
    ["after_tool_call", { toolName: "exec", params: {}, result: "ok" }, "tool_success"],
    ["after_tool_call", { toolName: "exec", params: {}, error: "bad" }, "tool_failure"],
    ["agent_end", { messages: [], success: true }, "turn_end"],
    ["before_compaction", { messageCount: 2 }, "pre_compact"],
    ["session_end", { sessionId: "s", messageCount: 2 }, "session_end"],
  ];

  for (const [hookName, event, expectedType] of cases) {
    const canonical = openClawHookToEvent(hookName, event, {});
    assert.equal(canonical.event_type, expectedType);
    assert.equal("hook_type" in canonical, false);
    assert.equal("source" in canonical, false);
    assert.equal("agent_id" in canonical, false);
  }
  assert.deepEqual(openClawHookToEvent("unknown", {}, {}), {
    event_type: "hook_event",
  });
});

test("extracts bounded fields from verified OpenClaw typed hooks", () => {
  assert.deepEqual(
    openClawHookToEvent(
      "after_tool_call",
      {
        toolName: "exec",
        params: { command: "echo ok" },
        result: { output: "ok" },
        runId: "run-event",
        toolCallId: "call-1",
        durationMs: 12,
      },
      { runId: "run-context", agentId: "must-not-leak" },
    ),
    {
      event_type: "tool_success",
      run_id: "run-event",
      tool_name: "exec",
      tool_use_id: "call-1",
      tool_input: { command: "echo ok" },
      tool_input_size: 21,
      tool_response: { output: "ok" },
      tool_response_size: 15,
      duration_ms: 12,
    },
  );

  const turnEnd = openClawHookToEvent(
    "agent_end",
    {
      runId: "run-2",
      messages: [
        { role: "assistant", content: "first" },
        { role: "user", content: "question" },
        { role: "assistant", content: [{ type: "text", text: "final a@example.com" }] },
      ],
      success: false,
      error: "failed",
      durationMs: 34,
    },
    {},
  );
  assert.equal(turnEnd.assistant_message, "final {{email_1}}");
  assert.equal(turnEnd.assistant_message_size, "final {{email_1}}".length);
  assert.equal(turnEnd.status, "error");
  assert.equal(turnEnd.error, "failed");
  assert.equal(turnEnd.duration_ms, 34);

  assert.deepEqual(
    openClawHookToEvent(
      "before_compaction",
      { messageCount: 8, compactingCount: 5, tokenCount: 1200 },
      { runId: "run-3" },
    ),
    {
      event_type: "pre_compact",
      run_id: "run-3",
      message_count: 8,
      compacting_count: 5,
      token_count: 1200,
    },
  );
  assert.deepEqual(
    openClawHookToEvent(
      "session_end",
      { sessionId: "s", messageCount: 8, durationMs: 99, reason: "idle" },
      {},
    ),
    {
      event_type: "session_end",
      message_count: 8,
      duration_ms: 99,
      reason: "idle",
    },
  );

  const bounded = openClawHookToEvent(
    "before_agent_run",
    { prompt: "x".repeat(20_000), messages: [] },
    {},
  );
  assert.equal(bounded.prompt_size, 20_000);
  assert.ok(bounded.prompt.length <= 10_000);
});

test("redacts cyclic OpenClaw payloads without throwing", () => {
  const params = { command: "echo ok" };
  params.self = params;

  const canonical = openClawHookToEvent(
    "before_tool_call",
    { toolName: "exec", params },
    {},
  );

  assert.equal(canonical.event_type, "tool_start");
  assert.deepEqual(canonical.tool_input, {
    command: "echo ok",
    self: "[Circular]",
  });
});

test("redacts shared OpenClaw object references without marking them circular", () => {
  const shared = { email: "a@example.com", n: 1 };

  assert.deepEqual(redactTraceValue({ first: shared, second: shared }), {
    first: { email: "{{email_1}}", n: 1 },
    second: { email: "{{email_1}}", n: 1 },
  });
});

test("redacts JSON-hostile OpenClaw payload values without throwing", () => {
  const canonical = openClawHookToEvent(
    "before_tool_call",
    { toolName: "exec", params: { count: 1n } },
    {},
  );

  assert.deepEqual(canonical.tool_input, { count: "1" });
});

test("redacts deep OpenClaw payloads with a depth limit", () => {
  const payload = {};
  let current = payload;
  for (let index = 0; index < 64; index += 1) {
    current.next = {};
    current = current.next;
  }

  const canonical = openClawHookToEvent(
    "before_tool_call",
    { toolName: "exec", params: payload },
    {},
  );

  assert.match(JSON.stringify(canonical.tool_input), /DepthLimit/);
});

test("redacts wide OpenClaw payloads with a node limit", () => {
  const params = Object.fromEntries(
    Array.from({ length: 3_000 }, (_, index) => [`k${index}`, index]),
  );

  const redacted = redactTraceValue(params);

  assert.equal(redacted.__truncated__, "[NodeLimit]");
});

test("can omit OpenClaw trace content while keeping size metadata", () => {
  const canonical = openClawHookToEvent(
    "after_tool_call",
    {
      toolName: "exec",
      params: { command: "echo private@example.com" },
      result: { output: "secret@example.com" },
    },
    { traceCaptureContent: false },
  );

  assert.equal(canonical.event_type, "tool_success");
  assert.equal(canonical.tool_input_size > 0, true);
  assert.equal(canonical.tool_response_size > 0, true);
  assert.equal("tool_input" in canonical, false);
  assert.equal("tool_response" in canonical, false);
});

test("redacted OpenClaw payloads cannot reintroduce secrets through toJSON", () => {
  let toJsonCalls = 0;
  const canonical = openClawHookToEvent(
    "before_tool_call",
    {
      toolName: "exec",
      params: {
        safe: "ok",
        toJSON() {
          toJsonCalls += 1;
          return { apiKey: "sk-secret-after-redaction" };
        },
      },
    },
    {},
  );

  const serialized = JSON.stringify(canonical);
  assert.equal(toJsonCalls, 0);
  assert.equal(serialized.includes("sk-secret-after-redaction"), false);
  assert.deepEqual(canonical.tool_input, { safe: "ok" });
});

test("redacts hostile stringification values without throwing", async () => {
  const hostile = {
    toString() {
      throw new Error("boom");
    },
  };
  const canonical = openClawHookToEvent(
    "before_tool_call",
    { toolName: "exec", params: { apiKey: hostile, value: hostile } },
    {},
  );

  assert.deepEqual(canonical.tool_input, {
    apiKey: "{{api_key_1}}",
    value: {},
  });

  const handlers = new Map();
  const errors = [];
  registerOpenClawTraceHooks(
    {
      on(name, handler) {
        handlers.set(name, handler);
      },
      logger: { error(message) { errors.push(message); } },
    },
    { memoryApiBaseUrl: "http://memory.test", traceTimeoutMs: 75 },
    {
      fetchFn: async () => ({ ok: true }),
      randomUUID: () => "uuid-hostile",
      now: () => new Date("2026-07-16T01:02:03.000Z"),
    },
  );
  await handlers.get("before_tool_call")(
    { toolName: "exec", params: { value: hostile } },
    { sessionId: "s" },
  );

  assert.deepEqual(errors, []);
});

test("drops JSON-hostile OpenClaw count fields from canonical events", () => {
  assert.deepEqual(
    openClawHookToEvent("session_end", { sessionId: "s", messageCount: 1n }, {}),
    { event_type: "session_end" },
  );
});

test("agent_end skips tool-call-only assistant messages when finding final text", () => {
  const event = openClawHookToEvent(
    "agent_end",
    {
      messages: [
        { role: "assistant", content: "actual final answer" },
        { role: "assistant", content: [{ type: "toolCall", name: "exec" }] },
        { role: "assistant", content: "" },
      ],
      success: true,
    },
    {},
  );
  assert.equal(event.assistant_message, "actual final answer");
});

for (const fixture of redactionCases) {
  test(`redacts shared fixture: ${fixture.name}`, () => {
    const redacted = redactTraceValue(fixture.text);
    const serialized = JSON.stringify(redacted);
    for (const plaintext of [fixture.plaintext].flat()) {
      assert.equal(serialized.includes(plaintext), false);
    }
    for (const placeholder of [fixture.placeholder].flat()) {
      assert.equal(serialized.includes(placeholder), true);
    }
  });
}

test("reuses a typed placeholder for repeated values in one event", () => {
  const event = openClawHookToEvent(
    "after_tool_call",
    {
      toolName: "lookup",
      params: { email: "repeat.fixture@example.com" },
      result: "repeat.fixture@example.com",
    },
    {},
  );
  const serialized = JSON.stringify(event);
  assert.equal(serialized.includes("repeat.fixture@example.com"), false);
  assert.equal(serialized.match(/\{\{email_1\}\}/g)?.length, 2);
});

test("redacts named credentials embedded in strings", () => {
  const redacted = redactTraceValue(
    "password=hunter2 authorization: Basic auth-value cookie=session-value --api-key 'quoted-value'",
  );
  assert.equal(redacted.includes("hunter2"), false);
  assert.equal(redacted.includes("auth-value"), false);
  assert.equal(redacted.includes("session-value"), false);
  assert.equal(redacted.includes("quoted-value"), false);
  assert.match(redacted, /\{\{password_1\}\}/);
  assert.match(redacted, /\{\{authorization_1\}\}/);
  assert.match(redacted, /\{\{cookie_1\}\}/);
  assert.match(redacted, /\{\{api_key_1\}\}/);
});

test("does not classify an invalid dotted number as IPv4", () => {
  assert.equal(redactTraceValue("version 999.999.999.999"), "version 999.999.999.999");
});

test("posts an unchanged trace envelope with identity headers and configured timeout", async () => {
  const calls = [];
  const timeouts = [];
  const envelope = {
    request_id: "event-1",
    timestamp: "2026-07-16T00:00:00.000Z",
    account_id: "account",
    user_id: "user",
    agent_id: "agent",
    session_id: "session",
    event: { event_type: "tool_start" },
  };
  const result = await postTraceEvent(
    {
      memoryApiBaseUrl: "http://memory.test/",
      authApiKey: "key",
      authAccountId: "auth-account",
      traceTimeoutMs: 1234,
    },
    envelope,
    {
      fetchFn: async (...args) => {
        calls.push(args);
        return { ok: true };
      },
      timeoutSignalFn: (timeoutMs) => {
        timeouts.push(timeoutMs);
        return { timeoutMs };
      },
    },
  );

  assert.equal(result, true);
  assert.equal(calls.length, 1);
  assert.equal(calls[0][0], "http://memory.test/api/v1/trace/events");
  assert.deepEqual(calls[0][1].headers, {
    "Content-Type": "application/json",
    "X-API-Key": "key",
    "X-Account-ID": "auth-account",
    "X-User-ID": "user",
  });
  assert.deepEqual(JSON.parse(calls[0][1].body), envelope);
  assert.deepEqual(timeouts, [1234]);
});

for (const intermediate of ["", "字".repeat(10_000), "😀".repeat(10_000), "\u0001".repeat(10_000)]) {
  test(`posts a bounded completed turn with ${JSON.stringify(intermediate.slice(0, 2))} intermediate text`, async () => {
    const answer = "答".repeat(10_000);
    const messages = [{ role: "user", content: "Give the detailed result." }];
    if (intermediate) messages.push({ role: "assistant", content: intermediate });
    messages.push({ role: "assistant", content: answer });
    const envelope = {
      request_id: "completed-turn", trace_id: "completed-turn", operation: "openclaw_hook",
      status: "ok", timestamp: "2026-10-11T00:00:00.000Z", account_id: "account",
      user_id: "user", agent_id: "agent", session_id: "session",
      event: openClawHookToEvent("agent_end", { messages, success: true }),
    };
    const calls = [];
    const result = await postTraceEvent(
      { memoryApiBaseUrl: "http://memory.test", traceTimeoutMs: 1234 }, envelope,
      { fetchFn: async (_url, request) => { calls.push(request); return { ok: true }; } },
    );
    assert.equal(result, true);
    assert.equal(calls.length, 1);
    assert.ok(Buffer.byteLength(calls[0].body, "utf8") <= 60_000);
    const stored = JSON.parse(calls[0].body);
    assert.equal(stored.request_id, envelope.request_id);
    assert.equal(stored.event.event_type, "turn_end");
    assert.equal(stored.event.assistant_message, answer);
    assert.equal(envelope.event.assistant_message, answer);
  });
}

test("postTraceEvent reports serialization failures without throwing", async () => {
  const errors = [];
  const result = await postTraceEvent(
    { memoryApiBaseUrl: "http://memory.test", traceTimeoutMs: 1234 },
    { request_id: "bad", event: { count: 1n } },
    {
      fetchFn: async () => {
        throw new Error("fetch must not run");
      },
      logger: { error(message) { errors.push(message); } },
    },
  );

  assert.equal(result, false);
  assert.equal(errors.length, 1);
  assert.match(errors[0], /serialize|BigInt|Do not know/i);
});

test("postTraceEvent drops oversized and permanent failures without retrying", async () => {
  const errors = [];
  let fetchCount = 0;
  const oversized = await postTraceEvent(
    { memoryApiBaseUrl: "http://memory.test", traceTimeoutMs: 1234 },
    {
      request_id: "oversized",
      event: { tool_input: "😀".repeat(20_000) },
    },
    {
      fetchFn: async () => {
        fetchCount += 1;
        return { ok: true };
      },
      logger: { error(message) { errors.push(message); } },
    },
  );
  assert.equal(oversized, "drop");
  assert.equal(fetchCount, 0);

  const permanent = await postTraceEvent(
    { memoryApiBaseUrl: "http://memory.test", traceTimeoutMs: 1234 },
    { request_id: "bad", event: { event_type: "hook_event" } },
    {
      fetchFn: async () => {
        fetchCount += 1;
        return { ok: false, status: 413 };
      },
      logger: { error(message) { errors.push(message); } },
    },
  );
  assert.equal(permanent, "drop");
  assert.equal(fetchCount, 1);
  assert.equal(errors.some((message) => String(message).includes("dropped")), true);
});

test("postTraceEvent retries transient HTTP responses", async () => {
  const statuses = [408, 429, 500, 503];
  const calls = [];
  for (const status of statuses) {
    let fetchCount = 0;
    const result = await postTraceEvent(
      { memoryApiBaseUrl: "http://memory.test", traceTimeoutMs: 1234 },
      { request_id: `retry-${status}`, event: { event_type: "hook_event" } },
      {
        fetchFn: async () => {
          fetchCount += 1;
          return fetchCount === 1 ? { ok: false, status } : { ok: true, status: 200 };
        },
        logger: { error(message) { calls.push(message); } },
      },
    );

    assert.equal(result, true);
    assert.equal(fetchCount, 2);
  }
  assert.equal(calls.length, 0);
});

test("registers typed hooks and swallows transport failures after two attempts", async () => {
  const handlers = new Map();
  const fetchCalls = [];
  const timeouts = [];
  let uuidCount = 0;
  const cfg = {
    memoryApiBaseUrl: "http://memory.test",
    authApiKey: "",
    authAccountId: "",
    accountId: "account",
    userId: "user",
    agentId: "configured-agent",
    traceTimeoutMs: 75,
  };
  registerOpenClawTraceHooks(
    {
      on(name, handler) {
        handlers.set(name, handler);
      },
      logger: { error() {} },
    },
    cfg,
    {
      fetchFn: async (...args) => {
        fetchCalls.push(args);
        throw new Error("offline");
      },
      timeoutSignalFn: (timeoutMs) => {
        timeouts.push(timeoutMs);
        return { timeoutMs };
      },
      randomUUID: () => `uuid-${++uuidCount}`,
      now: () => new Date("2026-07-16T01:02:03.000Z"),
    },
  );

  const expectedNames = [
    "session_start",
    "before_agent_run",
    "before_tool_call",
    "after_tool_call",
    "agent_end",
    "before_compaction",
    "session_end",
  ];
  assert.deepEqual([...handlers.keys()], expectedNames);

  const invocations = {
    session_start: [{ sessionId: "s" }, { sessionId: "s", agentId: "runtime-agent" }],
    before_agent_run: [{ prompt: "hello", messages: [] }, { sessionId: "s", agentId: "runtime-agent" }],
    before_tool_call: [{ toolName: "exec", params: {} }, { sessionId: "s", agentId: "runtime-agent" }],
    after_tool_call: [{ toolName: "exec", params: {}, result: "ok" }, { sessionId: "s", agentId: "runtime-agent" }],
    agent_end: [{ messages: [], success: true }, { sessionId: "s", agentId: "runtime-agent" }],
    before_compaction: [{ messageCount: 2 }, { sessionId: "s", agentId: "runtime-agent" }],
    session_end: [{ sessionId: "s", messageCount: 2 }, { sessionId: "s", agentId: "runtime-agent" }],
  };
  assert.equal(await handlers.get("session_start")(...invocations.session_start), undefined);
  assert.equal(await handlers.get("before_agent_run")(...invocations.before_agent_run), undefined);
  await flushTraceUploads();
  await flushTraceUploads();

  assert.equal(fetchCalls.length, 4);
  assert.deepEqual(timeouts, [75, 75, 75, 75]);
  assert.equal(uuidCount, 2);
  const bodiesByRequestId = new Map();
  for (const call of fetchCalls) {
    const body = JSON.parse(call[1].body);
    const bodies = bodiesByRequestId.get(body.request_id) || [];
    bodies.push(body);
    bodiesByRequestId.set(body.request_id, bodies);
  }
  for (const bodies of bodiesByRequestId.values()) {
    assert.equal(bodies.length, 2);
    const [firstBody, retryBody] = bodies;
    assert.deepEqual(retryBody, firstBody);
    assert.equal(firstBody.account_id, "account");
    assert.equal(firstBody.user_id, "user");
    assert.equal(firstBody.agent_id, "runtime-agent");
    assert.equal(firstBody.session_id, "s");
    assert.match(firstBody.event.event_type, /^(session_start|user_message|tool_start|tool_success|turn_end|pre_compact|session_end)$/);
  }
});

test("trace hooks skip heartbeat runs", async () => {
  const handlers = new Map();
  const fetchCalls = [];
  registerOpenClawTraceHooks(
    {
      on(name, handler) {
        handlers.set(name, handler);
      },
      logger: { error() {} },
    },
    {
      memoryApiBaseUrl: "http://memory.test",
      accountId: "account",
      userId: "user",
      agentId: "agent",
      traceTimeoutMs: 75,
    },
    {
      fetchFn: async (...args) => {
        fetchCalls.push(args);
        return { ok: true, status: 200 };
      },
    },
  );

  await handlers.get("session_start")(
    { sessionId: "fresh-heartbeat" },
    { sessionId: "fresh-heartbeat" },
  );
  await handlers.get("before_agent_run")(
    { prompt: "[OpenClaw heartbeat poll]", messages: [] },
    { sessionId: "fresh-heartbeat", runId: "fresh-heartbeat-run" },
  );
  await handlers.get("agent_end")(
    { messages: [], success: true },
    { sessionId: "fresh-heartbeat", runId: "fresh-heartbeat-run" },
  );
  await flushTraceUploads();
  assert.equal(fetchCalls.length, 0);

  const invocations = {
    session_start: { sessionId: "s" },
    before_agent_run: { prompt: "heartbeat", messages: [] },
    before_tool_call: { toolName: "exec", params: {} },
    after_tool_call: { toolName: "exec", params: {}, result: "ok" },
    agent_end: { messages: [], success: true },
    before_compaction: { messageCount: 2 },
    session_end: { sessionId: "s", messageCount: 2 },
  };
  for (const [hookName, event] of Object.entries(invocations)) {
    await handlers.get(hookName)(event, { sessionId: "s", trigger: "heartbeat" });
  }
  await flushTraceUploads();

  assert.equal(fetchCalls.length, 0);

  await handlers.get("before_agent_run")(
    {
      prompt: "[Queued user message]\n[OpenClaw heartbeat poll]\n\n[System] Resume interrupted turn.",
      messages: [],
    },
    { sessionId: "s", runId: "recovered-heartbeat" },
  );
  await handlers.get("before_tool_call")(
    { toolName: "exec", params: {} },
    { sessionId: "s", runId: "recovered-heartbeat" },
  );
  await handlers.get("agent_end")(
    { messages: [], success: false },
    { sessionId: "s", runId: "recovered-heartbeat" },
  );
  await flushTraceUploads();
  assert.equal(fetchCalls.length, 0);

  await handlers.get("before_agent_run")(
    { prompt: "normal user turn", messages: [] },
    { sessionId: "s", runId: "normal-run" },
  );
  await flushTraceUploads();
  assert.equal(fetchCalls.length, 1);
});

test("trace hooks isolate concurrent heartbeat and normal runs in one session", async () => {
  const handlers = new Map();
  const fetchCalls = [];
  registerOpenClawTraceHooks(
    {
      on(name, handler) {
        handlers.set(name, handler);
      },
      logger: { error() {} },
    },
    {
      memoryApiBaseUrl: "http://memory.test",
      accountId: "account",
      userId: "user",
      agentId: "agent",
      traceTimeoutMs: 75,
    },
    {
      fetchFn: async (...args) => {
        fetchCalls.push(args);
        return { ok: true, status: 200 };
      },
    },
  );

  await handlers.get("session_start")(
    { sessionId: "shared" },
    { sessionId: "shared" },
  );
  await handlers.get("before_agent_run")(
    { prompt: "[OpenClaw heartbeat poll]", messages: [] },
    { sessionId: "shared", runId: "heartbeat-run" },
  );
  await handlers.get("before_agent_run")(
    { prompt: "normal user turn", messages: [] },
    { sessionId: "shared", runId: "normal-run" },
  );
  await handlers.get("agent_end")(
    { messages: [], success: true },
    { sessionId: "shared", runId: "normal-run" },
  );
  await handlers.get("before_tool_call")(
    { toolName: "exec", params: {} },
    { sessionId: "shared", runId: "heartbeat-run" },
  );
  await handlers.get("agent_end")(
    { messages: [], success: true },
    { sessionId: "shared", runId: "heartbeat-run" },
  );
  await flushTraceUploads();

  assert.deepEqual(
    fetchCalls.map(([, init]) => JSON.parse(init.body).event.event_type),
    ["session_start", "user_message", "turn_end"],
  );
});

test("trace hooks isolate keyed and unkeyed events in concurrent runs", async () => {
  const handlers = new Map();
  const fetchCalls = [];
  registerOpenClawTraceHooks(
    {
      on(name, handler) {
        handlers.set(name, handler);
      },
      logger: { error() {} },
    },
    {
      memoryApiBaseUrl: "http://memory.test",
      accountId: "account",
      userId: "user",
      agentId: "agent",
      traceTimeoutMs: 75,
    },
    {
      fetchFn: async (...args) => {
        fetchCalls.push(args);
        return { ok: true, status: 200 };
      },
    },
  );

  await handlers.get("before_agent_run")(
    { prompt: "[OpenClaw heartbeat poll]", messages: [] },
    { sessionId: "shared", runId: "heartbeat-run" },
  );
  await handlers.get("before_tool_call")(
    { toolName: "normal-tool", params: {} },
    { sessionId: "shared" },
  );
  await handlers.get("agent_end")(
    { messages: [], success: true },
    { sessionId: "shared", runId: "heartbeat-run" },
  );
  await handlers.get("before_agent_run")(
    { prompt: "[OpenClaw heartbeat poll]", messages: [] },
    { sessionId: "reverse-shared" },
  );
  await handlers.get("before_tool_call")(
    { toolName: "normal-tool", params: {} },
    { sessionId: "reverse-shared", runId: "normal-run" },
  );
  await handlers.get("agent_end")(
    { messages: [], success: true },
    { sessionId: "reverse-shared" },
  );
  await flushTraceUploads();

  assert.deepEqual(
    fetchCalls.map(([, init]) => JSON.parse(init.body).event.event_type),
    ["tool_start", "tool_start"],
  );
});

test("trace hooks expire stale heartbeat run and session state", async () => {
  const handlers = new Map();
  const fetchCalls = [];
  let clockMs = 1_000;
  registerOpenClawTraceHooks(
    {
      on(name, handler) {
        handlers.set(name, handler);
      },
      logger: { error() {} },
    },
    {
      memoryApiBaseUrl: "http://memory.test",
      accountId: "account",
      userId: "user",
      agentId: "agent",
      traceTimeoutMs: 75,
    },
    {
      fetchFn: async (...args) => {
        fetchCalls.push(args);
        return { ok: true, status: 200 };
      },
      nowMs: () => clockMs,
    },
  );

  await handlers.get("before_agent_run")(
    { prompt: "[OpenClaw heartbeat poll]", messages: [] },
    { sessionId: "keyed-session", runId: "stale-run" },
  );
  await handlers.get("before_agent_run")(
    { prompt: "[OpenClaw heartbeat poll]", messages: [] },
    { sessionId: "unkeyed-session" },
  );

  clockMs += (2 * 60 * 60 * 1000) + 1;
  await handlers.get("before_tool_call")(
    { toolName: "keyed-normal", params: {} },
    { sessionId: "keyed-session", runId: "stale-run" },
  );
  await handlers.get("before_tool_call")(
    { toolName: "unkeyed-normal", params: {} },
    { sessionId: "unkeyed-session" },
  );
  await flushTraceUploads();

  assert.deepEqual(
    fetchCalls.map(([, init]) => JSON.parse(init.body).event.tool_name),
    ["keyed-normal", "unkeyed-normal"],
  );
});

test("trace hooks bound heartbeat run and session state", async () => {
  const handlers = new Map();
  const fetchCalls = [];
  registerOpenClawTraceHooks(
    {
      on(name, handler) {
        handlers.set(name, handler);
      },
      logger: { error() {} },
    },
    {
      memoryApiBaseUrl: "http://memory.test",
      accountId: "account",
      userId: "user",
      agentId: "agent",
      traceTimeoutMs: 75,
    },
    {
      fetchFn: async (...args) => {
        fetchCalls.push(args);
        return { ok: true, status: 200 };
      },
    },
  );

  for (let index = 0; index <= 2_048; index += 1) {
    await handlers.get("before_agent_run")(
      { prompt: "[OpenClaw heartbeat poll]", messages: [] },
      { sessionId: `keyed-${index}`, runId: `run-${index}` },
    );
    await handlers.get("before_agent_run")(
      { prompt: "[OpenClaw heartbeat poll]", messages: [] },
      { sessionId: `unkeyed-${index}` },
    );
  }

  await handlers.get("before_tool_call")(
    { toolName: "evicted-keyed", params: {} },
    { sessionId: "keyed-0", runId: "run-0" },
  );
  await handlers.get("before_tool_call")(
    { toolName: "evicted-unkeyed", params: {} },
    { sessionId: "unkeyed-0" },
  );
  await flushTraceUploads();

  assert.deepEqual(
    fetchCalls.map(([, init]) => JSON.parse(init.body).event.tool_name),
    ["evicted-keyed", "evicted-unkeyed"],
  );
});

test("trace hooks match pending session starts through every session alias", async () => {
  const handlers = new Map();
  const fetchCalls = [];
  registerOpenClawTraceHooks(
    {
      on(name, handler) {
        handlers.set(name, handler);
      },
      logger: { error() {} },
    },
    {
      memoryApiBaseUrl: "http://memory.test",
      accountId: "account",
      userId: "user",
      agentId: "agent",
      traceTimeoutMs: 75,
    },
    {
      fetchFn: async (...args) => {
        fetchCalls.push(args);
        return { ok: true, status: 200 };
      },
    },
  );

  await handlers.get("session_start")(
    { sessionKey: "stable-key" },
    { sessionKey: "stable-key" },
  );
  await handlers.get("before_agent_run")(
    {
      prompt: "normal user turn",
      messages: [],
      sessionId: "runtime-id",
      sessionKey: "stable-key",
    },
    {
      sessionId: "runtime-id",
      sessionKey: "stable-key",
      runId: "normal-run",
    },
  );
  await flushTraceUploads();

  assert.deepEqual(
    fetchCalls.map(([, init]) => JSON.parse(init.body).event.event_type),
    ["session_start", "user_message"],
  );
});

test("trace hooks discard a pending session start when a heartbeat run ends", async () => {
  const handlers = new Map();
  const fetchCalls = [];
  registerOpenClawTraceHooks(
    {
      on(name, handler) {
        handlers.set(name, handler);
      },
      logger: { error() {} },
    },
    {
      memoryApiBaseUrl: "http://memory.test",
      accountId: "account",
      userId: "user",
      agentId: "agent",
      traceTimeoutMs: 75,
    },
    {
      fetchFn: async (...args) => {
        fetchCalls.push(args);
        return { ok: true, status: 200 };
      },
    },
  );

  await handlers.get("session_start")(
    { sessionId: "shared" },
    { sessionId: "shared" },
  );
  await handlers.get("before_agent_run")(
    { prompt: "[OpenClaw heartbeat poll]", messages: [] },
    { sessionId: "shared", runId: "heartbeat-run" },
  );
  await handlers.get("agent_end")(
    { messages: [], success: true },
    { sessionId: "shared", runId: "heartbeat-run" },
  );
  await handlers.get("before_agent_run")(
    { prompt: "normal user turn", messages: [] },
    { sessionId: "shared", runId: "normal-run" },
  );
  await flushTraceUploads();

  assert.deepEqual(
    fetchCalls.map(([, init]) => JSON.parse(init.body).event.event_type),
    ["user_message"],
  );
});

test("trace hooks bound unmatched pending session starts", async () => {
  const handlers = new Map();
  const fetchCalls = [];
  registerOpenClawTraceHooks(
    {
      on(name, handler) {
        handlers.set(name, handler);
      },
      logger: { error() {} },
    },
    {
      memoryApiBaseUrl: "http://memory.test",
      accountId: "account",
      userId: "user",
      agentId: "agent",
      traceTimeoutMs: 75,
    },
    {
      fetchFn: async (...args) => {
        fetchCalls.push(args);
        return { ok: true, status: 200 };
      },
    },
  );

  for (let index = 0; index <= 2_048; index += 1) {
    await handlers.get("session_start")(
      { sessionId: `pending-${index}` },
      { sessionId: `pending-${index}` },
    );
  }
  await handlers.get("before_agent_run")(
    { prompt: "normal oldest turn", messages: [] },
    { sessionId: "pending-0", runId: "oldest-run" },
  );
  await handlers.get("before_agent_run")(
    { prompt: "normal newest turn", messages: [] },
    { sessionId: "pending-2048", runId: "newest-run" },
  );
  await flushTraceUploads();

  assert.deepEqual(
    fetchCalls.map(([, init]) => JSON.parse(init.body).event.event_type),
    ["user_message", "session_start", "user_message"],
  );
});

test("trace hooks schedule uploads without awaiting slow transport", async () => {
  const handlers = new Map();
  registerOpenClawTraceHooks(
    {
      on(name, handler) {
        handlers.set(name, handler);
      },
      logger: { error() {} },
    },
    {
      memoryApiBaseUrl: "http://memory.test",
      accountId: "account",
      userId: "user",
      agentId: "agent",
      traceTimeoutMs: 3000,
    },
    {
      fetchFn: async () => new Promise(() => {}),
      randomUUID: () => "uuid-pending",
      now: () => new Date("2026-07-16T01:02:03.000Z"),
    },
  );

  const result = await Promise.race([
    handlers.get("before_tool_call")(
      { toolName: "exec", params: {} },
      { sessionId: "s" },
    ).then(() => "returned"),
    new Promise((resolve) => setTimeout(() => resolve("blocked"), 10)),
  ]);

  assert.equal(result, "returned");
});

test("trace hooks bound background upload concurrency", async () => {
  const handlers = new Map();
  const fetchCalls = [];
  const errors = [];
  registerOpenClawTraceHooks(
    {
      on(name, handler) {
        handlers.set(name, handler);
      },
      logger: { error(message) { errors.push(message); } },
    },
    {
      memoryApiBaseUrl: "http://memory.test",
      accountId: "account",
      userId: "user",
      agentId: "agent",
      traceTimeoutMs: 3000,
    },
    {
      fetchFn: async (...args) => {
        fetchCalls.push(args);
        return new Promise(() => {});
      },
      randomUUID: (() => {
        let count = 0;
        return () => `uuid-pending-${++count}`;
      })(),
      now: () => new Date("2026-07-16T01:02:03.000Z"),
    },
  );

  for (let index = 0; index < 5; index += 1) {
    assert.equal(
      await handlers.get("before_tool_call")(
        { toolName: "exec", params: { index } },
        { sessionId: "s" },
      ),
      undefined,
    );
  }

  assert.equal(fetchCalls.length, 4);
  assert.equal(
    errors.some((message) => String(message).includes("in-flight limit")),
    true,
  );
});

test("trace hooks open a circuit after upload failures", async () => {
  const handlers = new Map();
  const errors = [];
  let fetchCount = 0;
  let clockMs = 1_000;
  registerOpenClawTraceHooks(
    {
      on(name, handler) {
        handlers.set(name, handler);
      },
      logger: { error(message) { errors.push(message); } },
    },
    {
      memoryApiBaseUrl: "http://memory.test",
      accountId: "account",
      userId: "user",
      agentId: "agent",
      traceTimeoutMs: 75,
    },
    {
      fetchFn: async () => {
        fetchCount += 1;
        throw new Error("offline");
      },
      timeoutSignalFn: () => ({}),
      randomUUID: (() => {
        let count = 0;
        return () => `uuid-failed-${++count}`;
      })(),
      now: () => new Date("2026-07-16T01:02:03.000Z"),
      nowMs: () => clockMs,
    },
  );

  await handlers.get("before_tool_call")(
    { toolName: "exec", params: {} },
    { sessionId: "s" },
  );
  await flushTraceUploads();
  await flushTraceUploads();
  assert.equal(fetchCount, 2);

  await handlers.get("before_tool_call")(
    { toolName: "exec", params: {} },
    { sessionId: "s" },
  );
  assert.equal(fetchCount, 2);
  assert.equal(
    errors.some((message) => String(message).includes("circuit is open")),
    true,
  );

  clockMs += 30_001;
  await handlers.get("before_tool_call")(
    { toolName: "exec", params: {} },
    { sessionId: "s" },
  );
  assert.equal(fetchCount, 4);
});

test("keeps runtime trace identities isolated by session and clears them at session end", async () => {
  const handlers = new Map();
  const bodies = [];
  registerOpenClawTraceHooks(
    {
      on(name, handler) {
        handlers.set(name, handler);
      },
      logger: { error() {} },
    },
    {
      memoryApiBaseUrl: "http://memory.test",
      authApiKey: "auth-key",
      authAccountId: "auth-account",
      accountId: "cfg-account",
      userId: "cfg-user",
      agentId: "cfg-agent",
      traceTimeoutMs: 75,
    },
    {
      fetchFn: async (_url, options) => {
        bodies.push({ body: JSON.parse(options.body), headers: options.headers });
        return { ok: true };
      },
      timeoutSignalFn: () => ({}),
      randomUUID: (() => {
        let count = 0;
        return () => `identity-${++count}`;
      })(),
      now: () => new Date("2026-07-16T01:02:03.000Z"),
    },
  );

  await handlers.get("before_agent_run")(
    { prompt: "a", messages: [], accountId: "runtime-account-a", senderId: "runtime-user-a" },
    { sessionKey: "session-a", agentId: "runtime-agent-a" },
  );
  await handlers.get("before_agent_run")(
    { prompt: "b", messages: [], accountId: "runtime-account-b", senderId: "runtime-user-b" },
    { sessionKey: "session-b", agentId: "runtime-agent-b" },
  );
  await handlers.get("before_tool_call")(
    { toolName: "exec", params: {} },
    { sessionKey: "session-a", toolName: "exec" },
  );
  await handlers.get("after_tool_call")(
    { toolName: "exec", params: {}, result: "ok" },
    { sessionKey: "session-b", toolName: "exec" },
  );

  assert.deepEqual(
    bodies.slice(0, 4).map(({ body }) => [body.session_id, body.account_id, body.user_id, body.agent_id]),
    [
      ["session-a", "runtime-account-a", "runtime-user-a", "runtime-agent-a"],
      ["session-b", "runtime-account-b", "runtime-user-b", "runtime-agent-b"],
      ["session-a", "runtime-account-a", "runtime-user-a", "runtime-agent-a"],
      ["session-b", "runtime-account-b", "runtime-user-b", "runtime-agent-b"],
    ],
  );
  assert.equal(bodies[2].headers["X-API-Key"], "auth-key");
  assert.equal(bodies[2].headers["X-Account-ID"], "auth-account");
  assert.equal(bodies[2].headers["X-User-ID"], "runtime-user-a");

  await handlers.get("before_tool_call")(
    { toolName: "exec", params: {} },
    { sessionKey: "session-a", toolName: "exec", agentId: "runtime-agent-a2" },
  );
  assert.equal(bodies.at(-1).body.agent_id, "runtime-agent-a2");

  await handlers.get("session_end")(
    { sessionId: "session-a", messageCount: 1 },
    { sessionId: "session-a" },
  );
  await handlers.get("before_tool_call")(
    { toolName: "exec", params: {} },
    { sessionKey: "session-a", toolName: "exec" },
  );
  assert.deepEqual(
    [bodies.at(-1).body.account_id, bodies.at(-1).body.user_id, bodies.at(-1).body.agent_id],
    ["cfg-account", "cfg-user", "cfg-agent"],
  );
});

test("session_end resolves and clears identity through every session alias", async () => {
  const handlers = new Map();
  const bodies = [];
  registerOpenClawTraceHooks(
    {
      on(name, handler) {
        handlers.set(name, handler);
      },
      logger: { error() {} },
    },
    {
      memoryApiBaseUrl: "http://memory.test",
      accountId: "cfg-account",
      userId: "cfg-user",
      agentId: "cfg-agent",
      traceTimeoutMs: 75,
    },
    {
      fetchFn: async (_url, options) => {
        bodies.push(JSON.parse(options.body));
        return { ok: true };
      },
      timeoutSignalFn: () => ({}),
      randomUUID: () => `alias-${bodies.length + 1}`,
      now: () => new Date("2026-07-16T01:02:03.000Z"),
    },
  );

  await handlers.get("before_agent_run")(
    {
      prompt: "hello",
      messages: [],
      accountId: "runtime-account",
      senderId: "runtime-user",
    },
    { sessionKey: "stable-key", agentId: "runtime-agent" },
  );
  await handlers.get("session_end")(
    { sessionId: "ended-id", sessionKey: "stable-key", messageCount: 1 },
    { sessionId: "ctx-ended-id", sessionKey: "stable-key" },
  );

  assert.deepEqual(
    [bodies[1].session_id, bodies[1].account_id, bodies[1].user_id, bodies[1].agent_id],
    ["ended-id", "runtime-account", "runtime-user", "runtime-agent"],
  );

  for (const sessionAlias of ["stable-key", "ended-id", "ctx-ended-id"]) {
    await handlers.get("before_tool_call")(
      { toolName: "exec", params: {} },
      { sessionKey: sessionAlias, toolName: "exec" },
    );
    assert.deepEqual(
      [bodies.at(-1).account_id, bodies.at(-1).user_id, bodies.at(-1).agent_id],
      ["cfg-account", "cfg-user", "cfg-agent"],
    );
  }
});

test("trace identity cache expires stale session aliases", async () => {
  const handlers = new Map();
  const bodies = [];
  let clockMs = 1_000;
  registerOpenClawTraceHooks(
    {
      on(name, handler) {
        handlers.set(name, handler);
      },
      logger: { error() {} },
    },
    {
      memoryApiBaseUrl: "http://memory.test",
      accountId: "cfg-account",
      userId: "cfg-user",
      agentId: "cfg-agent",
      traceTimeoutMs: 75,
    },
    {
      fetchFn: async (_url, options) => {
        bodies.push(JSON.parse(options.body));
        return { ok: true };
      },
      timeoutSignalFn: () => ({}),
      randomUUID: () => `ttl-${bodies.length + 1}`,
      now: () => new Date("2026-07-16T01:02:03.000Z"),
      nowMs: () => clockMs,
    },
  );

  await handlers.get("before_agent_run")(
    { prompt: "hello", messages: [], accountId: "runtime-account", senderId: "runtime-user" },
    { sessionId: "stale-session", agentId: "runtime-agent" },
  );
  await flushTraceUploads();
  assert.deepEqual(
    [bodies.at(-1).account_id, bodies.at(-1).user_id, bodies.at(-1).agent_id],
    ["runtime-account", "runtime-user", "runtime-agent"],
  );

  clockMs += (2 * 60 * 60 * 1000) + 1;
  await handlers.get("before_tool_call")(
    { toolName: "exec", params: {} },
    { sessionId: "stale-session" },
  );
  await flushTraceUploads();

  assert.deepEqual(
    [bodies.at(-1).account_id, bodies.at(-1).user_id, bodies.at(-1).agent_id],
    ["cfg-account", "cfg-user", "cfg-agent"],
  );
});

test("registered OpenClaw hooks honor traceCaptureContent opt-out", async () => {
  const handlers = new Map();
  const bodies = [];
  registerOpenClawTraceHooks(
    {
      on(name, handler) {
        handlers.set(name, handler);
      },
      logger: { error() {} },
    },
    {
      memoryApiBaseUrl: "http://memory.test",
      accountId: "cfg-account",
      userId: "cfg-user",
      agentId: "cfg-agent",
      traceTimeoutMs: 75,
      traceCaptureContent: false,
    },
    {
      fetchFn: async (_url, options) => {
        bodies.push(JSON.parse(options.body));
        return { ok: true };
      },
      timeoutSignalFn: () => ({}),
      randomUUID: () => `content-off-${bodies.length + 1}`,
      now: () => new Date("2026-07-16T01:02:03.000Z"),
    },
  );

  await handlers.get("before_agent_run")(
    { prompt: "private prompt", messages: [] },
    { sessionId: "sess-content-off" },
  );
  await handlers.get("after_tool_call")(
    {
      toolName: "exec",
      params: { command: "echo private@example.com" },
      result: { output: "secret@example.com" },
    },
    { sessionId: "sess-content-off" },
  );
  await handlers.get("agent_end")(
    {
      messages: [{ role: "assistant", content: "private answer" }],
      success: false,
      error: "private failure",
    },
    { sessionId: "sess-content-off" },
  );
  await handlers.get("session_end")(
    { sessionId: "sess-content-off", reason: "private reason", messageCount: 3 },
    { sessionId: "sess-content-off" },
  );
  await flushTraceUploads();
  await flushTraceUploads();

  assert.deepEqual(
    bodies.map((body) => body.event.event_type),
    ["user_message", "tool_success", "turn_end", "session_end"],
  );
  for (const { event } of bodies) {
    for (const contentField of [
      "prompt",
      "tool_input",
      "tool_response",
      "assistant_message",
      "error",
      "reason",
    ]) {
      assert.equal(contentField in event, false);
    }
  }
  assert.equal(bodies[0].event.prompt_size > 0, true);
  assert.equal(bodies[1].event.tool_input_size > 0, true);
  assert.equal(bodies[1].event.tool_response_size > 0, true);
  assert.equal(bodies[2].event.assistant_message_size > 0, true);
  assert.equal(bodies[2].event.error_size > 0, true);
  assert.equal(bodies[3].event.message_count, 3);
});

test("plugin configuration and registration enable trace hooks before the context engine", async () => {
  assert.equal(resolveConfig({}).traceTimeoutMs, 3000);
  assert.equal(resolveConfig({ traceTimeoutMs: 456 }).traceTimeoutMs, 456);
  assert.equal(resolveConfig({}).traceCaptureContent, true);
  assert.equal(resolveConfig({ traceCaptureContent: false }).traceCaptureContent, false);
  const manifest = JSON.parse(
    await readFile(new URL("./openclaw.plugin.json", import.meta.url), "utf8"),
  );
  assert.equal(manifest.configSchema.properties.traceTimeoutMs.type, "integer");
  assert.equal(manifest.configSchema.properties.traceTimeoutMs.default, 3000);
  assert.equal(manifest.configSchema.properties.traceCaptureContent.type, "boolean");
  assert.equal(manifest.configSchema.properties.traceCaptureContent.default, true);

  const registrations = [];
  plugin.register({
    pluginConfig: { mode: "remote" },
    logger: { info() {}, error() {} },
    on(name) {
      registrations.push(`hook:${name}`);
    },
    registerService() {},
    registerContextEngine() {
      registrations.push("context-engine");
    },
  });
  assert.equal(registrations.filter((value) => value.startsWith("hook:")).length, 8);
  assert.equal(registrations.filter((value) => value === "hook:session_end").length, 2);
  assert.ok(registrations.indexOf("hook:session_start") < registrations.indexOf("context-engine"));
});

test("published package contains the trace runtime", () => {
  const command = process.platform === "win32" ? process.env.ComSpec || "cmd.exe" : "npm";
  const args = process.platform === "win32"
    ? ["/d", "/s", "/c", "npm pack --dry-run --json"]
    : ["pack", "--dry-run", "--json"];
  const packed = spawnSync(command, args, {
    cwd: pluginDir,
    encoding: "utf8",
  });
  assert.equal(packed.status, 0, packed.error?.message || packed.stderr);
  const [{ files }] = JSON.parse(packed.stdout);
  assert.equal(files.some(({ path }) => path === "trace.js"), true);
});

test("README documents the host-level conversation hook permission", async () => {
  const readme = await readFile(new URL("./README.md", import.meta.url), "utf8");
  assert.match(readme, /"hooks"\s*:\s*\{\s*"allowConversationAccess"\s*:\s*true\s*\}/);
  assert.match(readme, /before_agent_run/);
  assert.match(readme, /agent_end/);
  assert.match(readme, /raw conversation/i);
  assert.match(readme, /polic(?:y|ies).*block/i);
  assert.match(readme, /\/api\/v1\/trace\/events/);
  assert.match(readme, /traceTimeoutMs/);
  assert.match(readme, /traceCaptureContent=false/);
  assert.match(readme, /default:\s*3000/i);
  assert.match(readme, /non-blocking/i);
  assert.match(readme, /\{\{email_1\}\}/);
  assert.match(readme, /\{\{connection_string_1\}\}/);
  for (const [hookName, eventType] of [
    ["session_start", "session_start"],
    ["before_agent_run", "user_message"],
    ["before_tool_call", "tool_start"],
    ["after_tool_call", "tool_success"],
    ["after_tool_call", "tool_failure"],
    ["agent_end", "turn_end"],
    ["before_compaction", "pre_compact"],
    ["session_end", "session_end"],
  ]) {
    assert.match(readme, new RegExp(hookName));
    assert.match(readme, new RegExp(eventType));
  }
});
