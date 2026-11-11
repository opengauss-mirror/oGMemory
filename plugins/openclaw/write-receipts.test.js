import assert from "node:assert/strict";
import { once } from "node:events";
import { createServer } from "node:http";
import test from "node:test";
import plugin, { callHttp, runCompactControlFlow } from "./index.js";

async function receiptServer(t, buffered = false, requestId = "server-receipt", replies = []) {
  const calls = [];
  const server = createServer(async (req, res) => {
    let body = "";
    for await (const chunk of req) body += chunk;
    calls.push({ method: req.method, path: req.url, headers: req.headers, body: JSON.parse(body || "{}") });
    const accepted = req.method === "POST";
    const [status, response] = replies[calls.length - 1] ?? [accepted ? 202 : 200, {
      ok: true, request_id: requestId, request_status: accepted ? "ACCEPTED" : "SUCCEEDED",
      source_status: buffered ? "BUFFERED" : "ARCHIVED",
      retry_action: accepted && !buffered ? "poll" : "none",
    }];
    if (status === null) {
      req.socket.destroy();
      return;
    }
    res.writeHead(status, { "Content-Type": "application/json" });
    res.end(JSON.stringify(response));
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  t.after(async () => {
    server.closeAllConnections();
    await new Promise((resolve) => server.close(resolve));
  });
  return { url: `http://127.0.0.1:${server.address().port}`, calls };
}

test("accepted write follows its server receipt through HTTP GET", async (t) => {
  const { url, calls } = await receiptServer(t);
  const result = await callHttp(url, "after_turn", {
    sessionId: "session one", userId: "alice", agentId: "worker",
    messages: [{ role: "user", content: "Remember the deployment command." }],
  }, { authApiKey: "fixture-key", authAccountId: "acme", requestTimeoutMs: 5000 });

  assert.equal(result.request_status, "SUCCEEDED");
  assert.equal(result.ok, true);
  assert.deepEqual(calls.map(({ method }) => method), ["POST", "GET"]);
  assert.equal(calls[1].path, "/api/v1/sessions/session%20one/write-requests?clientRequestId=server-receipt");
  assert.match(calls[0].body.clientRequestId, /^oc-[a-f0-9]{64}$/);
  assert.equal(calls[1].headers["x-agent-id"], "worker");
  assert.equal(calls[1].headers["x-api-key"], "fixture-key");
  assert.deepEqual(result.http_observations.map(({ method, http_status, request_id }) => (
    [method, http_status, request_id]
  )), [["POST", 202, "server-receipt"], ["GET", 200, "server-receipt"]]);
});

test("receipt polling preserves slash and percent characters", async (t) => {
  const requestId = "/turn/%2F";
  const { url, calls } = await receiptServer(t, false, requestId);
  const result = await callHttp(url, "after_turn", {
    sessionId: "session", clientRequestId: requestId,
    messages: [{ role: "user", content: "Remember blue." }],
  }, { requestTimeoutMs: 5000 });

  assert.equal(result.request_status, "SUCCEEDED");
  assert.equal(calls[0].body.clientRequestId, requestId);
  const query = new URL(calls[1].path, url);
  assert.equal(query.pathname, "/api/v1/sessions/session/write-requests");
  assert.equal(query.searchParams.get("clientRequestId"), requestId);
});

test("buffered write keeps its accepted receipt for normal session flush", async (t) => {
  const { url, calls } = await receiptServer(t, true);
  const result = await callHttp(url, "after_turn", {
    sessionId: "buffered", userId: "alice", agentId: "worker", clientRequestId: "stable-client-id",
    messages: [{ role: "user", content: "Remember blue." }],
  });

  assert.equal(result.source_status, "BUFFERED");
  assert.equal(result.request_status, "ACCEPTED");
  assert.equal(result.retry_action, "none");
  assert.deepEqual(calls.map(({ method }) => method), ["POST"]);
  assert.equal(result.http_observations[0].request_id, "server-receipt");
});

test("compaction completes after a source-only receipt finishes projection", async (t) => {
  const receipt = {
    ok: false, request_id: "compact-1", request_status: "DEGRADED", source_status: "ARCHIVED",
    semantic_status: "DEGRADED", archive_id: "archive-1", source_message_ids: ["msg-1"],
    archived_source_ids: ["msg-1"], recovery_mode: "source_only", retry_action: "manual_review",
  };
  const { url, calls } = await receiptServer(t, false, "compact-1", [
    [200, { ok: true, prepareToken: "prepared-1" }],
    [200, { ...receipt, projection_status: "PENDING", compacted: true,
      result: { summary: "Alice saved the deployment command.", tokensBefore: 12 } }],
    [200, { ...receipt, projection_status: "DONE" }],
  ]);
  const boundaries = [];
  const result = await runCompactControlFlow({
    cfg: { compactTakeoverEnabled: true },
    params: { sessionId: "compact-session", userId: "alice" },
    call: (method, params) => callHttp(url, method, params, { requestTimeoutMs: 5000 }),
    persistTakeoverCompactionBoundary: async (boundary) => { boundaries.push(boundary); },
  });

  assert.equal(result.ok, true);
  assert.equal(result.compacted, true);
  assert.equal(result.request_status, "DEGRADED");
  assert.equal(result.projection_status, "DONE");
  assert.equal(boundaries[0].summary, "Alice saved the deployment command.");
  assert.deepEqual(calls.map(({ method }) => method), ["POST", "POST", "GET"]);
  assert.equal(calls[1].body.prepareToken, "prepared-1");
  assert.equal(calls[2].path, "/api/v1/sessions/compact-session/write-requests?clientRequestId=compact-1");
});

function receiptEngine(url, toolResultExternalizationEnabled = false) {
  let factory;
  plugin.register({
    pluginConfig: { mode: "remote", memoryApiBaseUrl: url, autoCapture: true,
      prefetchEnabled: false, toolResultExternalizationEnabled },
    logger: { info() {}, warn() {}, error() {} },
    on() {},
    registerService() {},
    registerContextEngine(_id, create) { factory = create; },
  });
  return factory();
}

const pendingSourceReceipt = {
  ok: false, request_id: "original-request", request_status: "DEGRADED",
  source_status: "ARCHIVED", semantic_status: "DEGRADED", projection_status: "PENDING",
  archive_id: "archive-1", source_message_ids: ["msg-1"], archived_source_ids: ["msg-1"],
  retry_action: "manual_review",
};

for (const [name, reply, observationStatus] of [
  ["unrelated receipt", [200, { ...pendingSourceReceipt, request_id: "other-request",
    archive_id: "archive-2", projection_status: "DONE" }],
    "OBSERVATION_RECEIPT_MISMATCH"],
  ["HTTP 503", [503, {}], "OBSERVATION_HTTP_ERROR"],
  ["closed connection", [null], "OBSERVATION_ERROR"],
]) {
  test(`after_turn resumes receipt polling after ${name}`, async (t) => {
    const { url, calls } = await receiptServer(t, false, "next-request", [
      [200, pendingSourceReceipt],
      reply,
      [200, { ...pendingSourceReceipt, projection_status: "DONE" }],
      [200, { ok: true, request_id: "next-request", request_status: "SUCCEEDED", retry_action: "none" }],
    ]);
    const engine = receiptEngine(url, true);
    const snapshot = { sessionId: "resume-poll", userId: "alice", agentId: "worker",
      prePromptMessageCount: 0, messages: [{ role: "user", content: "Remember blue." }] };
    const pending = await engine.afterTurn({
      ...snapshot, runtimeContext: { currentMessageId: "inbound-1" },
    });
    const next = await engine.afterTurn({
      ...snapshot, runtimeContext: { currentMessageId: "inbound-2" },
    });
    const receiptPath = "/api/v1/sessions/resume-poll/write-requests?clientRequestId=original-request";

    assert.deepEqual(calls.map(({ method, path }) => [method, path]), [
      ["POST", "/api/v1/after_turn"],
      ["GET", receiptPath],
      ["GET", receiptPath],
      ["POST", "/api/v1/after_turn"],
    ], JSON.stringify({ pending, next }));
    assert.equal(pending.request_id, pendingSourceReceipt.request_id);
    assert.equal(pending.archive_id, pendingSourceReceipt.archive_id);
    assert.equal(pending.projection_status, "PENDING");
    assert.equal(pending.retry_action, "poll");
    assert.equal(pending.observation_status, observationStatus);
    assert.equal(next.request_id, "next-request");
    assert.equal(next.request_status, "SUCCEEDED");
  });
}

test("missing write receipt stays unconfirmed and the next queued turn completes", async (t) => {
  const { url, calls } = await receiptServer(t, false, "server-receipt", [
    [202, { ok: true, accepted: true, request_id: "server-receipt", request_status: "ACCEPTED",
      source_status: "CHECKPOINTED", retry_action: "poll" }],
    [200, { request_id: "server-receipt", request_status: "UNKNOWN", source_status: "UNKNOWN",
      semantic_status: "UNKNOWN", projection_status: "UNKNOWN", accepted: false, terminal: false,
      retry_action: "manual_review", http_status: 200 }],
    [200, { ok: true, request_id: "next-receipt", request_status: "SUCCEEDED",
      source_status: "ARCHIVED", retry_action: "none" }],
  ]);
  const errors = [];
  t.mock.method(console, "error", (line) => errors.push(line));
  const engine = receiptEngine(url, true);
  const snapshot = { sessionId: "receipt-missing", userId: "alice", agentId: "worker",
    prePromptMessageCount: 0, messages: [{ role: "user", content: "Remember blue." }] };
  const unresolved = await engine.afterTurn({ ...snapshot, runtimeContext: { currentMessageId: "inbound-1" } });
  const next = await engine.afterTurn({ ...snapshot, runtimeContext: { currentMessageId: "inbound-2" } });

  assert.equal(unresolved.request_id, "server-receipt");
  assert.equal(unresolved.request_status, "UNKNOWN");
  assert.equal(unresolved.retry_action, "manual_review");
  assert.equal(unresolved.ok, false);
  assert.equal(unresolved.retryable, false);
  assert.equal(next.request_status, "SUCCEEDED");
  assert.equal(next.ok, true);
  assert.deepEqual(calls.map(({ method, path }) => [method, path]), [
    ["POST", "/api/v1/after_turn"],
    ["GET", "/api/v1/sessions/receipt-missing/write-requests?clientRequestId=server-receipt"],
    ["POST", "/api/v1/after_turn"],
  ]);
  assert.notEqual(calls[0].body.clientRequestId, calls[2].body.clientRequestId);
  assert.match(errors.join("\n"), /"request_id":"server-receipt".*"request_status":"UNKNOWN".*"retry_action":"manual_review"/);
});

test("identical dialogue in distinct runtime turns has separate stable receipts", async (t) => {
  const { url, calls } = await receiptServer(t, true);
  const engine = receiptEngine(url);
  const snapshot = { sessionId: "repeated-dialogue", userId: "alice", agentId: "worker",
    prePromptMessageCount: 0, messages: [
      { role: "user", content: "Remember that my preferred color is blue." },
      { role: "assistant", content: "I will remember that." },
    ] };
  for (const currentMessageId of ["inbound-1", "inbound-2", "inbound-2"]) {
    await engine.afterTurn({ ...snapshot, runtimeContext: { currentMessageId } });
  }
  const requestIds = calls.map(({ body }) => body.clientRequestId);
  assert.equal(new Set(requestIds).size, 2);
  assert.equal(requestIds[1], requestIds[2]);
});

for (const field of ["semanticTurnId", "semanticTurnSourceNamespace"]) {
  test(`receipt identity preserves ${field} across repeated snapshots`, async (t) => {
    const { url, calls } = await receiptServer(t, true);
    const snapshot = { sessionId: "semantic-dialogue", userId: "alice", agentId: "worker",
      prePromptMessageCount: 0, messages: [{ role: "user", content: "Remember blue." }] };
    for (const identity of ["turn-1", "turn-2", "turn-2"]) {
      await callHttp(url, "after_turn", { ...snapshot, [field]: identity });
    }
    const requestIds = calls.map(({ body }) => body.clientRequestId);
    assert.equal(new Set(requestIds).size, 2);
    assert.equal(requestIds[1], requestIds[2]);
  });
}
