import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import plugin, {
  createFirstViewToolResultMiddleware,
  resolveConfig,
  toolResultModelTextChars,
} from "./index.js";

function textResult(text, extra = {}) {
  return {
    content: [{ type: "text", text }],
    details: { source: "fixture" },
    progress: "done",
    terminate: false,
    ...extra,
  };
}

function identity(overrides = {}) {
  return {
    event: {
      turnId: "turn-1",
      toolCallId: "call-1",
      toolName: "fixture",
      result: textResult("X".repeat(20)),
    },
    context: {
      runtime: "openclaw",
      sessionId: "session-1",
      runId: "run-1",
      agentId: "host-agent",
    },
    ...overrides,
  };
}

test("first-view defaults are enabled behind the main externalization switch", () => {
  const defaults = resolveConfig({});
  assert.equal(defaults.toolResultExternalizationEnabled, false);
  assert.equal(defaults.toolResultFirstViewExternalizationEnabled, true);
  assert.equal(defaults.toolResultFirstViewMinChars, 20_000);
});
test("qualifying first-view result is governed once and preserves result metadata", async () => {
  const requests = [];
  const cfg = resolveConfig({
    toolResultExternalizationEnabled: true,
    toolResultFirstViewExternalizationEnabled: true,
    toolResultFirstViewMinChars: 20,
    accountId: "acct",
    userId: "user",
    agentId: "host-agent",
  });
  const middleware = createFirstViewToolResultMiddleware({
    cfg,
    rankingPromptForSession: (sessionId) => `rank:${sessionId}`,
    call: async (method, body) => {
      requests.push({ method, body });
      return {
        ok: true,
        externalized: 1,
        content: [{ type: "text", text: "[ogmemory-tool-result v=2]\nstub" }],
      };
    },
  });
  const { event, context } = identity();

  const override = await middleware(event, context);

  assert.equal(requests.length, 1);
  assert.equal(requests[0].method, "govern_first_view_tool_result");
  assert.deepEqual(requests[0].body, {
    sessionId: "session-1",
    adapterId: "openclaw",
    turnId: "turn-1",
    toolCallId: "call-1",
    toolName: "fixture",
    content: event.result.content,
    rankingPrompt: "rank:session-1",
    userId: "user",
    accountId: "acct",
    agentId: "host-agent",
  });
  assert.equal(override.result.content[0].text, "[ogmemory-tool-result v=2]\nstub");
  assert.deepEqual(override.result.details, event.result.details);
  assert.equal(override.result.progress, "done");
  assert.equal(override.result.terminate, false);
  assert.deepEqual(event.result.content, [{ type: "text", text: "X".repeat(20) }]);
});

test("real OpenClaw middleware shape takes trusted session identity from event.threadId", async () => {
  const requests = [];
  const cfg = resolveConfig({
    toolResultExternalizationEnabled: true,
    toolResultFirstViewExternalizationEnabled: true,
    toolResultFirstViewMinChars: 1,
  });
  const middleware = createFirstViewToolResultMiddleware({
    cfg,
    sessionContextForSession: () => ({
      accountId: "dynamic-account",
      userId: "dynamic-user",
      agentId: "dynamic-agent",
      adapterTurnNonce: "host-attested-turn",
    }),
    call: async (method, body) => {
      requests.push({ method, body });
      return {
        ok: true,
        externalized: 1,
        content: [{ type: "text", text: "stub" }],
      };
    },
  });
  const { event } = identity();
  event.threadId = "runtime-thread-session";
  delete event.turnId;

  const override = await middleware(event, { runtime: "openclaw" });

  assert.equal(override.result.content[0].text, "stub");
  assert.equal(requests.length, 1);
  assert.equal(requests[0].body.sessionId, "runtime-thread-session");
  assert.equal(requests[0].body.turnId, "host-attested-turn");
  assert.equal(requests[0].body.accountId, "dynamic-account");
  assert.equal(requests[0].body.userId, "dynamic-user");
  assert.equal(requests[0].body.agentId, "dynamic-agent");
});

test("runtime agent identity cannot be mixed with a different cached owner", async () => {
  let calls = 0;
  const warnings = [];
  const cfg = resolveConfig({
    semanticTurnMode: "user_message",
    toolResultExternalizationEnabled: true,
    toolResultFirstViewExternalizationEnabled: true,
    toolResultFirstViewMinChars: 1,
  });
  const middleware = createFirstViewToolResultMiddleware({
    cfg,
    sessionContextForSession: () => ({
      accountId: "account-a",
      userId: "user-a",
      agentId: "agent-a",
      adapterTurnNonce: "host-attested-turn",
    }),
    call: async () => { calls += 1; },
    logger: { warn: (message) => warnings.push(message) },
  });
  const { event } = identity();
  event.threadId = "runtime-thread-session";
  delete event.turnId;

  assert.equal(await middleware(event, {
    runtime: "openclaw",
    agentId: "agent-b",
  }), undefined);
  assert.equal(calls, 0);
  assert.match(warnings[0], /agent identities conflict/u);
});

test("disabled, small, incomplete-identity, and failed governance paths fail open", async () => {
  for (const cfg of [
    resolveConfig({
      toolResultExternalizationEnabled: false,
      toolResultFirstViewExternalizationEnabled: true,
      toolResultFirstViewMinChars: 1,
    }),
    resolveConfig({
      toolResultExternalizationEnabled: true,
      toolResultFirstViewExternalizationEnabled: false,
      toolResultFirstViewMinChars: 1,
    }),
    resolveConfig({
      toolResultExternalizationEnabled: true,
      toolResultFirstViewExternalizationEnabled: true,
      toolResultFirstViewMinChars: 21,
    }),
  ]) {
    let calls = 0;
    const middleware = createFirstViewToolResultMiddleware({
      cfg,
      call: async () => { calls += 1; },
    });
    const { event, context } = identity();
    assert.equal(await middleware(event, context), undefined);
    assert.equal(calls, 0);
  }

  const warnings = [];
  let calls = 0;
  const cfg = resolveConfig({
    toolResultExternalizationEnabled: true,
    toolResultFirstViewExternalizationEnabled: true,
    toolResultFirstViewMinChars: 1,
  });
  const missingIdentity = createFirstViewToolResultMiddleware({
    cfg,
    call: async () => { calls += 1; },
    logger: { warn: (message) => warnings.push(message) },
  });
  const incomplete = identity();
  delete incomplete.event.toolCallId;
  assert.equal(await missingIdentity(incomplete.event, incomplete.context), undefined);
  assert.equal(calls, 0);
  assert.match(warnings[0], /identity is incomplete/u);

  const serverFailure = createFirstViewToolResultMiddleware({
    cfg,
    call: async () => ({ ok: false, failed: 1 }),
    logger: { warn: (message) => warnings.push(message) },
  });
  const valid = identity();
  assert.equal(await serverFailure(valid.event, valid.context), undefined);
  assert.match(warnings.at(-1), /failed open/u);
});

test("plugin registers OpenClaw middleware only when the suite is enabled", () => {
  const registrations = [];
  plugin.register({
    pluginConfig: {
      mode: "remote",
      memoryApiBaseUrl: "http://ogmem.invalid",
      toolResultExternalizationEnabled: true,
      toolResultFirstViewExternalizationEnabled: true,
    },
    logger: { info() {}, error() {}, warn() {}, debug() {} },
    on() {},
    registerService() {},
    registerContextEngine() {},
    registerAgentToolResultMiddleware(handler, options) {
      registrations.push({ handler, options });
    },
  });
  assert.equal(registrations.length, 1);
  assert.deepEqual(registrations[0].options, { runtimes: ["openclaw"] });

  const disabled = [];
  plugin.register({
    pluginConfig: {
      mode: "remote",
      memoryApiBaseUrl: "http://ogmem.invalid",
      toolResultExternalizationEnabled: false,
    },
    logger: { info() {}, error() {}, warn() {}, debug() {} },
    on() {},
    registerService() {},
    registerContextEngine() {},
    registerAgentToolResultMiddleware(handler, options) {
      disabled.push({ handler, options });
    },
  });
  assert.deepEqual(disabled, []);
});

for (const autoRecall of [true, false]) {
test(`first-view ranking and identity survive autoRecall=${autoRecall}`, async () => {
  let engineFactory;
  let middleware;
  plugin.register({
    pluginConfig: {
      mode: "remote",
      autoRecall,
      prefetchEnabled: true,
      memoryApiBaseUrl: "http://ogmem.invalid",
      toolResultExternalizationEnabled: true,
      toolResultFirstViewExternalizationEnabled: true,
      toolResultFirstViewMinChars: 1,
    },
    logger: { info() {}, error() {}, warn() {}, debug() {} },
    on() {},
    registerService() {},
    registerContextEngine(_id, create) {
      engineFactory = create;
    },
    registerAgentToolResultMiddleware(handler) {
      middleware = handler;
    },
  });
  const requests = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url, options) => {
    const method = String(url).split("/").at(-1);
    const body = JSON.parse(options.body);
    requests.push({ method, body });
    return {
      ok: true,
      status: 200,
      async json() {
        return method === "compose"
          ? { messages: body.messages, estimatedTokens: 0 }
          : {
              ok: true,
              externalized: 1,
              content: [{ type: "text", text: "stub" }],
            };
      },
      async text() { return ""; },
    };
  };
  try {
    const engine = engineFactory();
    await engine.assemble({
      sessionId: `ranking-session-${autoRecall}`,
      prompt: "Find the failed deployment request id",
      accountId: "turn-account",
      userId: "turn-user",
      agentId: "turn-agent",
      messages: [{ role: "user", content: "inspect the deployment" }],
    });
    const pair = identity();
    pair.event.threadId = `ranking-session-${autoRecall}`;
    delete pair.event.turnId;
    await middleware(pair.event, { runtime: "openclaw" });
    await engine.dispose();
  } finally {
    globalThis.fetch = originalFetch;
  }

  assert.equal(requests.some((r) => r.method === "compose"), autoRecall);
  assert.equal(requests.some((r) => r.method === "prefetch"), autoRecall);
  const governed = requests.find((request) => (
    request.method === "govern_first_view_tool_result"
  ));
  assert.equal(
    governed.body.rankingPrompt,
    "Find the failed deployment request id",
  );
  assert.equal(governed.body.accountId, "turn-account");
  assert.equal(governed.body.userId, "turn-user");
  assert.equal(governed.body.agentId, "turn-agent");
  assert.match(governed.body.turnId, /^ogmfv-host-[0-9a-f-]{36}$/u);
});
}

for (const autoCapture of [true, false]) {
for (const autoRecall of [true, false]) {
test(`first-view nonce rotates with capture=${autoCapture}, recall=${autoRecall}`, async () => {
  let engineFactory;
  let middleware;
  const warnings = [];
  const governed = [];
  const methods = [];
  plugin.register({
    pluginConfig: {
      mode: "remote",
      autoCapture,
      autoRecall,
      memoryApiBaseUrl: "http://ogmem.invalid",
      semanticTurnMode: "user_message",
      toolResultExternalizationEnabled: true,
      toolResultFirstViewExternalizationEnabled: true,
      toolResultFirstViewMinChars: 1,
    },
    logger: { info() {}, error() {}, warn(message) { warnings.push(message); }, debug() {} },
    on() {},
    registerService() {},
    registerContextEngine(_id, create) {
      engineFactory = create;
    },
    registerAgentToolResultMiddleware(handler) {
      middleware = handler;
    },
  });
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url, options) => {
    const method = String(url).split("/").at(-1);
    const body = JSON.parse(options.body);
    methods.push(method);
    if (method === "govern_first_view_tool_result") governed.push(body);
    return {
      ok: true,
      status: 200,
      async json() {
        if (method === "compose") {
          return { messages: body.messages, estimatedTokens: 0 };
        }
        if (method === "govern_first_view_tool_result") {
          return {
            ok: true,
            externalized: 1,
            content: [{ type: "text", text: "stub" }],
          };
        }
        return { ok: true };
      },
      async text() { return ""; },
    };
  };
  try {
    const engine = engineFactory();
    await engine.assemble({
      sessionId: `completed-session-${autoCapture}-${autoRecall}`,
      prompt: "inspect",
      messages: [],
    });
    const active = identity();
    active.event.threadId = `completed-session-${autoCapture}-${autoRecall}`;
    delete active.event.turnId;
    assert.equal(
      (await middleware(active.event, { runtime: "openclaw" })).result.content[0].text,
      "stub",
    );

    await engine.assemble({
      sessionId: `completed-session-${autoCapture}-${autoRecall}`,
      prompt: "inspect",
      messages: [],
    });
    const secondTool = identity();
    secondTool.event.threadId = `completed-session-${autoCapture}-${autoRecall}`;
    secondTool.event.toolCallId = "call-2";
    delete secondTool.event.turnId;
    assert.equal(
      (await middleware(secondTool.event, { runtime: "openclaw" })).result.content[0].text,
      "stub",
    );
    assert.equal(governed[0].turnId, governed[1].turnId);

    await engine.afterTurn({
      sessionId: `completed-session-${autoCapture}-${autoRecall}`,
      messages: [{ role: "assistant", content: "done", stopReason: "stop" }],
      prePromptMessageCount: 0,
    });
    const late = identity();
    late.event.threadId = `completed-session-${autoCapture}-${autoRecall}`;
    delete late.event.turnId;
    assert.equal(await middleware(late.event, { runtime: "openclaw" }), undefined);

    await engine.assemble({
      sessionId: `completed-session-${autoCapture}-${autoRecall}`,
      prompt: "next turn",
      messages: [
        { role: "user", content: "inspect" },
        { role: "assistant", content: "done", stopReason: "stop" },
        { role: "user", content: "next turn" },
      ],
    });
    const nextTurn = identity();
    nextTurn.event.threadId = `completed-session-${autoCapture}-${autoRecall}`;
    nextTurn.event.toolCallId = "call-3";
    delete nextTurn.event.turnId;
    assert.equal(
      (await middleware(nextTurn.event, { runtime: "openclaw" })).result.content[0].text,
      "stub",
    );
    assert.notEqual(governed[0].turnId, governed[2].turnId);
    if (!autoCapture) {
      assert.deepEqual(await engine.ingest({ sessionId: `completed-session-${autoCapture}-${autoRecall}`, messages: [] }), { ingested: false });
    }
    await engine.dispose();
    assert.equal(methods.includes("after_turn"), autoCapture);
    assert.equal(methods.includes("compose"), autoRecall);
    assert.equal(methods.includes("ingest"), false);
  } finally {
    globalThis.fetch = originalFetch;
  }
  assert.match(warnings.at(-1), /identity is incomplete/u);
});
}
}

test("same session with multiple owners fails open without cross-tenant mixing", async () => {
  let engineFactory;
  let middleware;
  const warnings = [];
  const governed = [];
  plugin.register({
    pluginConfig: {
      mode: "remote",
      memoryApiBaseUrl: "http://ogmem.invalid",
      semanticTurnMode: "user_message",
      toolResultExternalizationEnabled: true,
      toolResultFirstViewExternalizationEnabled: true,
      toolResultFirstViewMinChars: 1,
    },
    logger: { info() {}, error() {}, warn(message) { warnings.push(message); }, debug() {} },
    on() {},
    registerService() {},
    registerContextEngine(_id, create) { engineFactory = create; },
    registerAgentToolResultMiddleware(handler) { middleware = handler; },
  });
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url, options) => {
    const method = String(url).split("/").at(-1);
    const body = JSON.parse(options.body);
    if (method === "govern_first_view_tool_result") governed.push(body);
    return {
      ok: true,
      status: 200,
      async json() {
        return method === "compose"
          ? { messages: body.messages, estimatedTokens: 0 }
          : method === "govern_first_view_tool_result"
            ? {
                ok: true,
                externalized: 1,
                content: [{ type: "text", text: "stub" }],
              }
            : { ok: true };
      },
      async text() { return ""; },
    };
  };
  try {
    const engineA = engineFactory();
    const engineB = engineFactory();
    await engineA.assemble({
      sessionId: "shared-session",
      accountId: "account-a",
      userId: "user-a",
      agentId: "agent-a",
      prompt: "A",
      messages: [],
    });
    await engineB.assemble({
      sessionId: "shared-session",
      accountId: "account-b",
      userId: "user-b",
      agentId: "agent-b",
      prompt: "B",
      messages: [],
    });
    const ambiguous = identity();
    ambiguous.event.threadId = "shared-session";
    delete ambiguous.event.turnId;
    assert.equal(await middleware(ambiguous.event, { runtime: "openclaw" }), undefined);
    assert.equal(governed.length, 0);
    assert.match(warnings.at(-1), /owner identity is ambiguous/u);

    await engineA.afterTurn({
      sessionId: "shared-session",
      messages: [{ role: "assistant", content: "done", stopReason: "stop" }],
      prePromptMessageCount: 0,
    });
    const remaining = identity();
    remaining.event.threadId = "shared-session";
    delete remaining.event.turnId;
    assert.equal(
      (await middleware(remaining.event, { runtime: "openclaw" })).result.content[0].text,
      "stub",
    );
    assert.equal(governed[0].accountId, "account-b");
    assert.equal(governed[0].userId, "user-b");
    assert.equal(governed[0].agentId, "agent-b");
    await engineA.dispose();
    await engineB.dispose();
  } finally {
    globalThis.fetch = originalFetch;
  }
});
