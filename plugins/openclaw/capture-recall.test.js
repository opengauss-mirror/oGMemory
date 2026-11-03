import assert from "node:assert/strict";
import { createServer } from "node:http";
import { once } from "node:events";
import { readFileSync } from "node:fs";
import test from "node:test";
import plugin, { resolveConfig, runCompactControlFlow } from "./index.js";

function host(config) {
  let factory;
  const hooks = [];
  plugin.register({
    pluginConfig: { mode: "remote", ...config },
    logger: { info() {}, warn() {}, error() {} },
    on: (name, handler) => hooks.push({ name, handler }),
    registerService() {},
    registerContextEngine: (_id, value) => { factory = value; },
  });
  return { engine: factory(), hooks };
}

async function identityServer(t, onRequest = () => {}) {
  const calls = [];
  const server = createServer(async (req, res) => {
    let raw = "";
    for await (const chunk of req) raw += chunk;
    const call = { path: req.url, body: JSON.parse(raw || "{}") };
    calls.push(call);
    await onRequest(call);
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify({ ok: true, disposed: true, messages: [], estimatedTokens: 0, prepareToken: "test-token" }));
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  t.after(async () => {
    server.closeAllConnections();
    await new Promise((resolve) => server.close(resolve));
  });
  const instance = host({
    memoryApiBaseUrl: `http://127.0.0.1:${server.address().port}`,
    agentId: "configured-fallback",
    autoCapture: false,
    prefetchEnabled: false,
  });
  const end = instance.hooks.filter((hook) => hook.name === "session_end").at(-1).handler;
  return { ...instance, end, calls };
}

for (const eviction of ["expiry", "capacity"]) {
  test(`updated runtime identity survives ${eviction}`, async (t) => {
    let timestamp = Date.now();
    t.mock.method(Date, "now", () => timestamp);
    const { engine, end, calls } = await identityServer(t);
    await engine.afterTurn({ sessionId: "live", agentId: "old-agent", messages: [] });
    await engine.ingest({ sessionId: "live", agentId: "new-agent" });
    if (eviction === "expiry") timestamp += 2 * 60 * 60 * 1000 + 1;
    else {
      for (let index = 0; index < 2049; index += 1) {
        await engine.ingest({ sessionId: `other-${index}`, agentId: "other-agent" });
      }
    }
    await engine.assemble({ sessionId: "live", messages: [] });
    await end({ sessionId: "live", reason: "deleted" }, {});
    assert.deepEqual(calls.map(({ body }) => body.agentId), ["new-agent", "new-agent", "new-agent"]);
  });
  for (const keyOnly of [false, true]) {
    test(`trace restores saved identity after ${eviction} before CE, keyOnly=${keyOnly}`, { timeout: 5000 }, async (t) => {
      let timestamp = Date.now();
      t.mock.method(Date, "now", () => timestamp);
      let receiveTrace;
      const trace = new Promise((resolve) => { receiveTrace = resolve; });
      const { engine, hooks, calls } = await identityServer(t, ({ path, body }) => {
        if (path.endsWith("/trace/events")) receiveTrace(body);
      });
      const session = { sessionId: "live", sessionKey: "agent:main:legacy-key" };
      await engine.afterTurn({ ...session, agentId: "runtime-agent", messages: [] });
      if (eviction === "expiry") timestamp += 2 * 60 * 60 * 1000 + 1;
      else {
        for (let index = 0; index < 2049; index += 1) {
          await engine.ingest({ sessionId: `other-${index}`, agentId: "other-agent" });
        }
      }
      for (const hook of hooks.filter((hook) => hook.name === "before_tool_call")) {
        await hook.handler({ toolName: "read", params: { path: "README.md" } }, keyOnly ? { sessionKey: session.sessionKey } : session);
      }
      const envelope = await trace;
      await engine.assemble({ sessionId: "live", messages: [] });
      assert.deepEqual([envelope.agent_id, calls.find(({ path }) => path.endsWith("/compose")).body.agentId],
        ["runtime-agent", "runtime-agent"]);
    });
  }
  for (const operation of ["assemble", "compact", "afterTurn"]) {
    test(`live session retains identity after ${eviction} during ${operation}`, async (t) => {
      let timestamp = Date.now();
      t.mock.method(Date, "now", () => timestamp);
      const { engine, end, calls } = await identityServer(t);
      await engine.afterTurn({ sessionId: "live", sessionKey: "agent:main:live", messages: [] });
      if (eviction === "expiry") timestamp += 2 * 60 * 60 * 1000 + 1;
      else {
        for (let index = 0; index < 2049; index += 1) {
          await engine.ingest({ sessionId: `other-${index}`, agentId: "other-agent" });
        }
      }
      await engine[operation]({ sessionId: "live", messages: [] });
      await end({ sessionId: "live", reason: "deleted" }, {});
      assert.ok(calls.length >= 2);
      for (const { body } of calls) assert.equal(body.agentId, "main");
    });
  }
  for (const sessionKey of [undefined, "agent:main:legacy-key"]) {
    test(`session cleanup preserves its saved identity after ${eviction}, key=${sessionKey}`, async (t) => {
      let timestamp = Date.now();
      t.mock.method(Date, "now", () => timestamp);
      const { engine, hooks, calls } = await identityServer(t);
      await engine.afterTurn({ sessionId: "old", sessionKey, agentId: "runtime-agent", messages: [] });
      if (eviction === "expiry") timestamp += 2 * 60 * 60 * 1000 + 1;
      else {
        for (let index = 0; index < 2049; index += 1) {
          await engine.ingest({ sessionId: `other-${index}`, agentId: "other-agent" });
        }
      }
      const event = { sessionId: "old", sessionKey, reason: "deleted" };
      for (const hook of hooks.filter((hook) => hook.name === "session_end")) {
        await hook.handler(event, {});
      }
      assert.deepEqual(calls.filter(({ path }) => !path.endsWith("/trace/events"))
        .map(({ path, body }) => [path, body.agentId]), [
        ["/api/v1/dispose", "runtime-agent"],
        ["/api/v1/sessions/old", "runtime-agent"],
      ]);
    });
  }
}

test("session key supplies identity after an initial configured fallback", async (t) => {
  const { engine, calls } = await identityServer(t);
  await engine.afterTurn({ sessionId: "live", messages: [] });
  await engine.assemble({ sessionId: "live", sessionKey: "agent:main:live", messages: [] });
  assert.equal(calls.find(({ path }) => path.endsWith("/compose")).body.agentId, "main");
});

test("alias association preserves the latest explicit identity", async (t) => {
  const { engine, calls } = await identityServer(t);
  await engine.ingest({ sessionId: "live", agentId: "old-agent" });
  await engine.ingest({ sessionKey: "agent:main:live", agentId: "new-agent" });
  await engine.ingest({ sessionId: "live" });
  await engine.assemble({ sessionId: "live", sessionKey: "agent:main:live", messages: [] });
  assert.equal(calls.find(({ path }) => path.endsWith("/compose")).body.agentId, "new-agent");
});

test("alias association prefers explicit identity over a newer key-derived identity", async (t) => {
  const { engine, calls } = await identityServer(t);
  await engine.ingest({ sessionId: "live", agentId: "runtime-agent" });
  await engine.ingest({ sessionKey: "agent:main:live" });
  await engine.assemble({ sessionId: "live", sessionKey: "agent:main:live", messages: [] });
  assert.equal(calls.find(({ path }) => path.endsWith("/compose")).body.agentId, "runtime-agent");
});

test("completed session releases its retained identity before reuse", async (t) => {
  const { engine, end, calls } = await identityServer(t);
  await engine.afterTurn({ sessionId: "live", agentId: "runtime-agent", messages: [] });
  await end({ sessionId: "live", reason: "deleted" }, {});
  await engine.assemble({ sessionId: "live", sessionKey: "agent:main:reused", messages: [] });
  assert.equal(calls.find(({ path }) => path.endsWith("/compose")).body.agentId, "main");
});

test("late session key survives expiry and is released at session end", async (t) => {
  let timestamp = Date.now();
  t.mock.method(Date, "now", () => timestamp);
  const { engine, end, calls } = await identityServer(t);
  const sessionKey = "agent:main:late";
  await engine.afterTurn({ sessionId: "live", agentId: "runtime-agent", messages: [] });
  await engine.ingest({ sessionId: "live", sessionKey });
  timestamp += 2 * 60 * 60 * 1000 + 1;
  await engine.assemble({ sessionKey, messages: [] });
  await end({ sessionKey }, {});
  await engine.assemble({ sessionKey, messages: [] });
  assert.deepEqual(calls.filter(({ path }) => path.endsWith("/compose")).map(({ body }) => body.agentId),
    ["runtime-agent", "main"]);
});

test("live session uses the latest runtime identity before its saved identity", async (t) => {
  const { engine, end, calls } = await identityServer(t);
  await engine.afterTurn({ sessionId: "live", agentId: "old-agent", messages: [] });
  await engine.ingest({ sessionId: "live", agentId: "new-agent" });
  await engine.assemble({ sessionId: "live", messages: [] });
  assert.equal(calls.find(({ path }) => path.endsWith("/compose")).body.agentId, "new-agent");
  await end({ sessionId: "live", reason: "deleted" }, {});
  assert.deepEqual(calls.map(({ body }) => body.agentId), ["new-agent", "new-agent", "new-agent"]);
});

test("reset preserves the new session identity while old disposal is pending", { timeout: 5000 }, async (t) => {
  let releaseDispose;
  let disposalStarted;
  const pending = new Promise((resolve) => { releaseDispose = resolve; });
  const started = new Promise((resolve) => { disposalStarted = resolve; });
  const { engine, end, calls } = await identityServer(t, async ({ path }) => {
    if (path.endsWith("/dispose")) {
      disposalStarted();
      await pending;
    }
  });
  const sessionKey = "agent:runtime-agent:channel:conversation";
  await engine.afterTurn({ sessionId: "old", sessionKey, agentId: "runtime-agent", messages: [] });
  const cleanup = end({ sessionId: "old", sessionKey, reason: "reset" }, {});
  try {
    await started;
    await engine.assemble({ sessionId: "new", sessionKey, agentId: "runtime-agent", messages: [] });
  } finally {
    releaseDispose();
    await cleanup;
  }
  await engine.assemble({ sessionId: "new", messages: [] });
  assert.deepEqual(calls.filter(({ path }) => path.endsWith("/compose"))
    .map(({ body }) => body.agentId), ["runtime-agent", "runtime-agent"]);
});

test("active session aliases retain their runtime identity at cache capacity", { timeout: 5000 }, async (t) => {
  let receiveTrace;
  const trace = new Promise((resolve) => { receiveTrace = resolve; });
  const { engine, hooks } = await identityServer(t, ({ path, body }) => {
    if (path.endsWith("/trace/events")) receiveTrace(body);
  });
  const sessionKey = "agent:main:active";
  await engine.ingest({ sessionId: "active", sessionKey, agentId: "runtime-agent" });
  for (let index = 0; index < 2046; index += 1) {
    await engine.ingest({ sessionId: `idle-${index}`, agentId: "idle-agent" });
  }
  await engine.ingest({ sessionId: "active" });
  await engine.ingest({ sessionId: "new", agentId: "new-agent" });
  for (const hook of hooks.filter((hook) => hook.name === "before_tool_call")) {
    await hook.handler({ toolName: "read", params: { path: "README.md" } }, { sessionKey });
  }
  assert.equal((await trace).agent_id, "runtime-agent");
});

test("sessionKey-only end releases the runtime identity for the next session", async (t) => {
  const { engine, hooks, calls } = await identityServer(t);
  const sessionKey = "agent:main:reused";
  await engine.ingest({ sessionKey, agentId: "runtime-agent" });
  for (const hook of hooks.filter((hook) => hook.name === "session_end")) {
    await hook.handler({ sessionKey }, {});
  }
  await engine.assemble({ sessionKey, messages: [] });
  assert.equal(calls.find(({ path }) => path.endsWith("/compose")).body.agentId, "main");
});

test("capture/recall defaults, environment and explicit false precedence", () => {
  const names = ["OGMEM_AUTO_CAPTURE", "OGMEM_AUTO_RECALL"];
  const saved = names.map((name) => process.env[name]);
  try {
    for (const name of names) delete process.env[name];
    assert.equal(resolveConfig().autoCapture, true);
    assert.equal(resolveConfig().autoRecall, true);
    for (const name of names) process.env[name] = "false";
    assert.equal(resolveConfig().autoCapture, false);
    assert.equal(resolveConfig().autoRecall, false);
    assert.equal(resolveConfig({ autoCapture: true }).autoCapture, true);
    for (const name of names) process.env[name] = "true";
    assert.equal(resolveConfig({ autoCapture: false }).autoCapture, false);
    assert.equal(resolveConfig({ autoRecall: false }).autoRecall, false);
    const schema = JSON.parse(readFileSync(new URL("./openclaw.plugin.json", import.meta.url)));
    for (const key of ["autoCapture", "autoRecall"]) {
      assert.equal(schema.configSchema.properties[key].type, "boolean");
      // OpenClaw applies schema defaults before register(), which would mask env settings.
      assert.equal(Object.hasOwn(schema.configSchema.properties[key], "default"), false);
    }
  } finally {
    names.forEach((name, i) => {
      if (saved[i] === undefined) delete process.env[name];
      else process.env[name] = saved[i];
    });
  }
});

test("native plugin phase switches over real HTTP", async (t) => {
  const calls = [];
  const server = createServer(async (req, res) => {
    let text = "";
    for await (const chunk of req) text += chunk;
    const body = text ? JSON.parse(text) : {};
    calls.push({ path: req.url, method: req.method, headers: req.headers, body });
    const result = req.url.endsWith("/compose")
      ? { messages: body.messages, identityContext: "Remembered history", estimatedTokens: 20 }
      : { ok: true, ingested: true, disposed: true };
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify(result));
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  const memoryApiBaseUrl = `http://127.0.0.1:${server.address().port}`;
  const messages = [{ role: "user", content: "QA question" }];
  const params = { sessionId: "phase-test", messages, prompt: "QA question" };
  try {
    for (const autoCapture of [true, false]) {
      for (const autoRecall of [true, false]) {
        await t.test(`capture=${autoCapture}, recall=${autoRecall}`, async () => {
          calls.length = 0;
          const { engine } = host({ memoryApiBaseUrl, autoCapture, autoRecall, prefetchEnabled: true });
          await engine.ingest(params);
          await engine.afterTurn(params);
          const assembled = await engine.assemble(params);
          const paths = calls.map((call) => call.path);
          assert.equal(paths.includes("/api/v1/ingest"), autoCapture);
          assert.equal(paths.includes("/api/v1/after_turn"), autoCapture);
          assert.equal(paths.includes("/api/v1/compose"), autoRecall);
          assert.equal(paths.includes("/api/v1/prefetch"), autoRecall);
          if (autoRecall) assert.match(JSON.stringify(assembled.messages), /Remembered history/);
          else assert.deepEqual(assembled, { messages, estimatedTokens: 0, systemPromptAddition: "" });
          await engine.dispose();
        });
      }
    }
    for (const externalization of [false, true]) {
      for (const reason of ["reset", "deleted"]) {
        await t.test(`capture off keeps cleanup identity: externalization=${externalization}, reason=${reason}`, async () => {
          calls.length = 0;
          const defaults = { accountId: "default-account", userId: "default-user", agentId: "default-agent" };
          const identity = { accountId: "host-account", userId: "host-user", agentId: "host-agent" };
          const sessionId = `dynamic-${externalization}-${reason}`;
          const { engine, hooks } = host({ memoryApiBaseUrl, ...defaults, autoCapture: false,
            toolResultExternalizationEnabled: externalization,
            toolResultFirstViewExternalizationEnabled: externalization });
          const input = { ...params, ...identity, sessionId };
          assert.deepEqual(await engine.ingest(input), { ingested: false });
          await engine.afterTurn(input);
          await engine.dispose();
          assert.equal(calls.length, 0, "identity registration must not capture or queue writes");
          const end = hooks.filter((hook) => hook.name === "session_end").at(-1).handler;
          await end({ sessionId, reason }, {});
          assert.deepEqual(calls.map(({ path, method }) => ({ path, method })), [
            { path: "/api/v1/dispose", method: "POST" },
            ...(reason === "deleted" ? [{ path: `/api/v1/sessions/${sessionId}`, method: "DELETE" }] : []),
          ]);
          for (const { body, headers } of calls) {
            assert.deepEqual({ accountId: body.accountId, userId: body.userId, agentId: body.agentId }, identity);
            assert.equal(headers["x-user-id"], identity.userId);
            assert.equal(body.messages, undefined);
          }
          // A different session only overrides userId; defaults must still be used for other fields.
          calls.length = 0;
          const partialId = sessionId + "-partial";
          await engine.afterTurn({ ...params, sessionId: partialId, userId: "partial-user" });
          await end({ sessionId: partialId, reason: "reset" }, {});
          assert.equal(calls.length, 1);
          assert.deepEqual(calls[0].body, { sessionId: partialId, ...defaults, userId: "partial-user" });
        });
      }
    }
    await t.test("capture off creates no after-turn retry work; cleanup still disposes existing buffers", async () => {
      calls.length = 0;
      const { engine, hooks } = host({ memoryApiBaseUrl, autoCapture: false,
        toolResultExternalizationEnabled: true, toolResultFirstViewExternalizationEnabled: false });
      assert.deepEqual(await engine.ingest(params), { ingested: false });
      await engine.afterTurn(params);
      await engine.dispose();
      assert.equal(calls.length, 0);
      // Last session_end handler is memory cleanup, not the independent trace hook.
      await hooks.filter((hook) => hook.name === "session_end").at(-1).handler(
        { sessionId: "previously-buffered", reason: "reset" }, {});
      assert.deepEqual(calls.map((call) => call.path), ["/api/v1/dispose"]);
      assert.equal(calls[0].body.messages, undefined);
    });
  } finally {
    const closed = once(server, "close");
    server.close();
    server.closeAllConnections?.();
    await closed;
  }
});

test("explicit compact remains available with automatic capture disabled", async () => {
  const calls = [];
  const result = await runCompactControlFlow({
    cfg: resolveConfig({ autoCapture: false, compactTakeoverEnabled: true }),
    params: { sessionId: "existing-history" },
    call: async (method) => {
      calls.push(method);
      return method === "prepare_compaction"
        ? { ok: true, prepareToken: "prepared-history" }
        : { ok: true, compacted: true, result: { summary: "existing history" } };
    },
    persistTakeoverCompactionBoundary: async () => ({}),
  });
  assert.equal(result.compacted, true);
  assert.deepEqual(calls, ["prepare_compaction", "compact"]);
});

test("trace and context engine share the runtime agent over HTTP", async () => {
  const calls = [];
  const server = createServer(async (req, res) => {
    let raw = "";
    for await (const chunk of req) raw += chunk;
    const body = JSON.parse(raw || "{}");
    calls.push({ path: req.url, body });
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify(req.url.endsWith("/compose")
      ? { messages: [], estimatedTokens: 0 }
      : { ok: true, prepareToken: "prepared", compacted: false }));
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  const { engine, hooks } = host({
    memoryApiBaseUrl: `http://127.0.0.1:${server.address().port}`,
    agentId: "configured-fallback",
    compactTakeoverEnabled: true,
    prefetchEnabled: true,
  });
  const emit = async (name, event, context) => {
    for (const hook of hooks.filter((hook) => hook.name === name)) {
      await hook.handler(event, context);
    }
  };
  const waitForTrace = async (sessionId) => {
    for (let attempt = 0; attempt < 100; attempt += 1) {
      if (calls.some(({ path, body }) => path.endsWith("/trace/events")
        && body.session_id === sessionId)) return;
      await new Promise((resolve) => setTimeout(resolve, 10));
    }
    throw new Error(`Trace upload timed out for ${sessionId}`);
  };
  try {
    for (const agentId of ["main", "custom-agent"]) {
      calls.length = 0;
      const sessionId = `session-${agentId}`;
      const sessionKey = `agent:${agentId}:channel:conversation`;
      const params = { sessionId, sessionKey, messages: [{ role: "user", content: "hello" }] };
      await engine.assemble(params);
      await emit("before_agent_run", { prompt: "hello", messages: [] }, { sessionId, sessionKey, agentId });
      await waitForTrace(sessionId);
      await engine.ingest({ sessionId, messages: params.messages });
      await engine.afterTurn({ sessionId, messages: params.messages });
      await engine.compact({ sessionId });
      await emit("session_end", { sessionId, reason: "reset" }, {});
      const paths = calls.map(({ path }) => path);
      for (const endpoint of ["prefetch", "compose", "trace/events", "ingest", "after_turn", "prepare_compaction", "compact", "dispose"]) {
        assert.ok(paths.includes(`/api/v1/${endpoint}`), endpoint);
      }
      for (const { body } of calls) assert.equal(body.agent_id || body.agentId, agentId);
    }
    calls.length = 0;
    await emit("before_agent_run", { prompt: "hello", messages: [] }, { sessionId: "hook-first", agentId: "runtime-custom" });
    await waitForTrace("hook-first");
    await engine.assemble({ sessionId: "hook-first", sessionKey: "agent:main:legacy-key", messages: [] });
    assert.equal(calls.find(({ path }) => path.endsWith("/compose")).body.agentId, "runtime-custom");
    await emit("session_end", { sessionId: "hook-first", reason: "reset" }, {});
    await engine.assemble({ sessionId: "hook-first", messages: [] });
    assert.equal(calls.filter(({ path }) => path.endsWith("/compose")).at(-1).body.agentId, "configured-fallback");
  } finally {
    server.closeAllConnections();
    await new Promise((resolve) => server.close(resolve));
  }
});

for (const busyStatus of [null, 200, 503]) {
  test(`compose returns canonical context after ${busyStatus ?? "ready"} response`, async (t) => {
    const calls = [];
    const context = "Canonical source confirms the release milestone.";
    const busyAttempts = busyStatus === 503 ? 2 : Number(busyStatus !== null);
    const server = createServer(async (req, res) => {
      let raw = "";
      for await (const chunk of req) raw += chunk;
      const body = JSON.parse(raw || "{}");
      calls.push({ path: req.url, body });
      res.setHeader("Content-Type", "application/json");
      if (req.url.endsWith("/compose")) {
        const attempt = calls.filter((call) => call.path.endsWith("/compose")).length;
        if (attempt <= busyAttempts) {
          res.statusCode = busyStatus;
          res.end(JSON.stringify({ ok: false, retryable: true, status: "processing" }));
          return;
        }
        res.end(JSON.stringify({ messages: body.messages, retrievedEvidence: context, estimatedTokens: 42 }));
        return;
      }
      res.end(JSON.stringify({ ok: true, disposed: true, cleaned: true }));
    });
    server.listen(0, "127.0.0.1");
    await once(server, "listening");
    t.after(async () => {
      server.closeAllConnections();
      await new Promise((resolve) => server.close(resolve));
    });
    const { engine, hooks } = host({
      memoryApiBaseUrl: `http://127.0.0.1:${server.address().port}`,
      autoCapture: false, prefetchEnabled: false, accountId: "compose-account", userId: "compose-user",
    });
    const sessionId = `compose-recovery-${busyStatus ?? "ready"}`;
    const messages = [{ role: "user", content: "Recall the release milestone." }];
    const params = { sessionId, messages, prompt: messages[0].content };
    const assembled = await engine.assemble(params);
    const composeCalls = calls.filter((call) => call.path.endsWith("/compose"));
    assert.equal(composeCalls.length, busyAttempts + 1);
    assert.ok(composeCalls.every((call) => JSON.stringify(call.body) === JSON.stringify(composeCalls[0].body)));
    assert.deepEqual(assembled.messages[0], messages[0]);
    assert.ok(assembled.messages.some((message) => message.content.includes(context)));
    assert.equal(assembled.estimatedTokens, 42);
    const sessionEnd = hooks.filter((hook) => hook.name === "session_end").at(-1).handler;
    await sessionEnd({ sessionId, reason: "deleted" }, {});
    const reused = await engine.assemble(params);
    assert.ok(reused.messages.some((message) => message.content.includes(context)));
    assert.equal(calls.filter((call) => call.path.endsWith("/compose")).length, composeCalls.length + 1);
  });
}
