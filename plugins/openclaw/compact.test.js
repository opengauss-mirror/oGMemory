import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import plugin from "./index.js";
import {
  appendTakeoverCompactionBoundary,
  delegateCompactionToRuntimeOrThrow,
  buildLayeredComposeMessages,
  prepareCompactionWithTransport,
  resolveConfig,
  runCompactControlFlow,
  spawnContextEngine,
} from "./index.js";

function messageEntry(id, role, content = "", extra = {}) {
  return {
    id,
    type: "message",
    parentId: null,
    message: {
      role,
      content,
      ...extra,
    },
  };
}

function compactionEntry(id, firstKeptEntryId, summary = "summary") {
  return {
    id,
    type: "compaction",
    parentId: null,
    summary,
    firstKeptEntryId,
    tokensBefore: 100,
  };
}

function createFakeSessionManager(branch, appendIds = "compaction-test", customIds = []) {
  const appendCalls = [];
  const customAppendCalls = [];
  const entries = [...branch];
  const ids = Array.isArray(appendIds) ? [...appendIds] : [appendIds];
  const customEntryIds = Array.isArray(customIds) ? [...customIds] : [customIds];
  const manager = {
    appendCalls,
    customAppendCalls,
    getBranch: () => entries,
    getEntries: () => entries,
    getHeader: () => ({ type: "session", id: "session-test" }),
    getLeafEntry: () => entries[entries.length - 1] ?? null,
    getLeafId: () => entries[entries.length - 1]?.id ?? null,
    appendCustomEntry: (customType, data) => {
      const appendId =
        customEntryIds.shift() ?? `custom-boundary-${customAppendCalls.length + 1}`;
      customAppendCalls.push({ customType, data });
      entries.push({
        id: appendId,
        type: "custom",
        parentId: entries[entries.length - 1]?.id ?? null,
        customType,
        data,
      });
      return appendId;
    },
    appendCompaction: (summary, firstKeptEntryId, tokensBefore, details) => {
      const appendId = ids.shift() ?? `compaction-${appendCalls.length + 1}`;
      appendCalls.push({ summary, firstKeptEntryId, tokensBefore, details });
      entries.push({
        id: appendId,
        type: "compaction",
        parentId: entries[entries.length - 1]?.id ?? null,
        summary,
        firstKeptEntryId,
        tokensBefore,
        details,
      });
      return appendId;
    },
  };
  return manager;
}

const resolved = resolveConfig({
  compactTakeoverEnabled: false,
  summaryMaxChars: 1234,
  shortTermIndexMode: "off",
});

assert.equal(resolved.compactTakeoverEnabled, false);
assert.equal(resolved.summaryMaxChars, 1234);
assert.equal(resolved.shortTermIndexMode, "off");
assert.equal(resolved.toolResultExternalizationEnabled, false);
assert.equal(resolved.toolResultFirstViewExternalizationEnabled, true);
assert.equal(resolved.toolResultFirstViewMinChars, 20_000);
assert.equal(resolved.semanticTurnMode, "user_message");

const externalizationResolved = resolveConfig({
  toolResultExternalizationEnabled: true,
  semanticTurnMode: "user_message",
});
assert.equal(externalizationResolved.toolResultExternalizationEnabled, true);
assert.equal(externalizationResolved.semanticTurnMode, "user_message");

const prefetchResolved = resolveConfig({
  prefetchEnabled: true,
  prefetchTopK: 7,
});
assert.equal(prefetchResolved.prefetchEnabled, true);
assert.equal(prefetchResolved.prefetchTopK, 7);

const pluginManifest = JSON.parse(
  readFileSync(new URL("./openclaw.plugin.json", import.meta.url), "utf8"),
);
const toolResultSchema = pluginManifest.configSchema.properties;
assert.deepEqual(toolResultSchema.toolResultExternalizationEnabled, {
  type: "boolean",
  description: "Enable optional large tool-result lifecycle transport and retry support. Stable host source identity is attached when available; user_message mode can use the server Turn Ledger when OpenClaw has no message IDs. Keep aligned with the oGMemory server setting.",
  default: false,
});
assert.equal(toolResultSchema.toolResultFirstViewExternalizationEnabled.type, "boolean");
assert.equal(toolResultSchema.toolResultFirstViewExternalizationEnabled.default, true);
assert.equal(toolResultSchema.toolResultFirstViewMinChars.type, "integer");
assert.equal(toolResultSchema.toolResultFirstViewMinChars.minimum, 1);
assert.equal(toolResultSchema.toolResultFirstViewMinChars.default, 20_000);
assert.equal(toolResultSchema.toolResultFirstViewStateDir.type, "string");
assert.deepEqual(toolResultSchema.semanticTurnMode.enum, ["user_message"]);
assert.equal(toolResultSchema.semanticTurnMode.default, "user_message");
for (const key of ["llmProvider", "llmModel", "embeddingModel"]) {
  assert.equal(
    Object.hasOwn(pluginManifest.configSchema.properties[key], "default"),
    false,
    `${key} schema default would be mistaken for an explicit plugin override`,
  );
}

const modelEnvKeys = [
  "OGMEM_API_KEY",
  "OGMEM_BASE_URL",
  "OGMEM_LLM_MODEL",
  "OGMEM_EMBEDDING_API_KEY",
  "OGMEM_EMBEDDING_BASE_URL",
  "OGMEM_EMBEDDING_MODEL",
];
const previousModelEnv = Object.fromEntries(
  modelEnvKeys.map((key) => [key, process.env[key]]),
);
Object.assign(process.env, {
  OGMEM_API_KEY: "env-llm-key",
  OGMEM_BASE_URL: "https://env-llm.example/v1",
  OGMEM_LLM_MODEL: "env-llm-model",
  OGMEM_EMBEDDING_API_KEY: "env-embedding-key",
  OGMEM_EMBEDDING_BASE_URL: "https://env-embedding.example/v1",
  OGMEM_EMBEDDING_MODEL: "env-embedding-model",
});
try {
  const envResolved = resolveConfig({});
  assert.equal(envResolved.llmApiKey, "env-llm-key");
  assert.equal(envResolved.llmBaseUrl, "https://env-llm.example/v1");
  assert.equal(envResolved.llmModel, "env-llm-model");
  assert.equal(envResolved.embeddingApiKey, "env-embedding-key");
  assert.equal(envResolved.embeddingBaseUrl, "https://env-embedding.example/v1");
  assert.equal(envResolved.embeddingModel, "env-embedding-model");
} finally {
  for (const key of modelEnvKeys) {
    if (previousModelEnv[key] === undefined) delete process.env[key];
    else process.env[key] = previousModelEnv[key];
  }
}

const spawnedEnv = {};
assert.deepEqual(resolveConfig({ llmModel: "explicit-model" }).modelConfigOverrides, {
  llmModel: "explicit-model",
});
assert.deepEqual(resolveConfig({ llmModel: "" }).modelConfigOverrides, {});
const spawned = spawnContextEngine(
  {
    ...resolveConfig({
      projectRoot: "/tmp/project",
      agfsPort: 1833,
      contextEnginePort: 8090,
      prefetchEnabled: true,
      prefetchTopK: 7,
      llmProvider: "openai",
      llmApiKey: "llm-key",
      llmBaseUrl: "https://llm.example/v1",
      llmModel: "llm-model",
      llmReasoningEffort: "xhigh",
      llmIsReasoningModel: true,
      llmExtraBody: { thinking: { type: "enabled" } },
      embeddingProvider: "openai",
      embeddingApiKey: "embedding-key",
      embeddingBaseUrl: "https://embedding.example/v1",
      embeddingModel: "embedding-model",
    }),
  },
  { info: () => {}, debug: () => {}, warn: () => {} },
  {
    spawnFn: (_cmd, _args, options) => {
      Object.assign(spawnedEnv, options.env);
      return {
        stdout: { on: () => {} },
        stderr: { on: () => {} },
        on: () => {},
      };
    },
    findPython: () => "python",
  },
);
assert.ok(spawned);
assert.equal(spawnedEnv.OGMEM_PREFETCH_ENABLED, "true");
assert.equal(spawnedEnv.OGMEM_PREFETCH_TOP_K, "7");
assert.equal(spawnedEnv.OGMEM_TOOL_RESULT_EXTERNALIZATION_ENABLED, "false");
assert.equal(spawnedEnv.OGMEM_TOOL_RESULT_FIRST_VIEW_ENABLED, "true");
assert.equal(spawnedEnv.OGMEM_TOOL_RESULT_SEMANTIC_TURN_MODE, "user_message");
assert.equal(spawnedEnv.OGMEM_API_KEY, "llm-key");
assert.equal(spawnedEnv.OGMEM_BASE_URL, "https://llm.example/v1");
assert.equal(spawnedEnv.OGMEM_LLM_MODEL, "llm-model");
assert.equal(spawnedEnv.LLM_REASONING_EFFORT, "xhigh");
assert.equal(spawnedEnv.LLM_IS_REASONING_MODEL, "true");
assert.deepEqual(JSON.parse(spawnedEnv.LLM_EXTRA_BODY), { thinking: { type: "enabled" } });
assert.equal(spawnedEnv.OGMEM_EMBEDDING_API_KEY, "embedding-key");
assert.equal(spawnedEnv.OGMEM_EMBEDDING_BASE_URL, "https://embedding.example/v1");
assert.equal(spawnedEnv.OGMEM_EMBEDDING_MODEL, "embedding-model");
assert.deepEqual(JSON.parse(spawnedEnv.OGMEM_MODEL_CONFIG_OVERRIDES), {
  llmProvider: "openai",
  llmApiKey: "llm-key",
  llmBaseUrl: "https://llm.example/v1",
  llmModel: "llm-model",
  llmReasoningEffort: "xhigh",
  llmIsReasoningModel: true,
  llmExtraBody: { thinking: { type: "enabled" } },
  embeddingProvider: "openai",
  embeddingApiKey: "embedding-key",
  embeddingBaseUrl: "https://embedding.example/v1",
  embeddingModel: "embedding-model",
});

const originalFetch = globalThis.fetch;
const lifecycleRequests = [];
const sessionEndHandlers = [];
let disposeAttempts = 0;
let disposeFailuresRemaining = 1;
let disposeResult = { disposed: true };
globalThis.fetch = async (url, options) => {
  lifecycleRequests.push({ url, body: JSON.parse(options.body) });
  if (url.endsWith("/dispose")) {
    disposeAttempts += 1;
    if (disposeFailuresRemaining > 0) {
      disposeFailuresRemaining -= 1;
      return {
        ok: false,
        status: 503,
        text: async () => "temporarily unavailable",
      };
    }
    return {
      ok: true,
      json: async () => disposeResult,
    };
  }
  return { ok: true, json: async () => ({ ok: true }) };
};
try {
  let createContextEngine;
  plugin.register({
    pluginConfig: {
      mode: "remote",
      memoryApiBaseUrl: "https://memory.example",
      accountId: "acct-1",
      userId: "user-1",
      agentId: "agent-1",
    },
    logger: { info: () => {}, error: () => {} },
    on: (event, handler) => {
      if (event === "session_end") sessionEndHandlers.push(handler);
    },
    registerService: () => {},
    registerContextEngine: (_id, factory) => {
      createContextEngine = factory;
    },
  });
  const contextEngine = createContextEngine();
  const requestCountBeforeHeartbeat = lifecycleRequests.length;
  assert.deepEqual(
    await contextEngine.ingest({
      sessionId: "heartbeat-session",
      isHeartbeat: true,
      message: { role: "user", content: "[OpenClaw heartbeat poll]" },
    }),
    { ingested: false },
  );
  await contextEngine.afterTurn({
    sessionId: "heartbeat-session",
    isHeartbeat: true,
    messages: [
      { role: "user", content: "[OpenClaw heartbeat poll]" },
      { role: "assistant", content: "HEARTBEAT_OK" },
    ],
  });
  assert.equal(lifecycleRequests.length, requestCountBeforeHeartbeat);
  assert.deepEqual(
    await contextEngine.ingest({
      sessionId: "heartbeat-session",
      message: { role: "user", content: "[OpenClaw heartbeat poll]" },
    }),
    { ingested: false },
  );
  await contextEngine.afterTurn({
    sessionId: "heartbeat-session",
    prePromptMessageCount: 1,
    messages: [
      { role: "user", content: "old message" },
      {
        role: "user",
        content: "[Queued user message]\n[OpenClaw heartbeat poll]\n\n[System] Resume interrupted turn.",
      },
    ],
  });
  assert.equal(lifecycleRequests.length, requestCountBeforeHeartbeat);

  await contextEngine.afterTurn({
    sessionId: "normal-after-heartbeat",
    prePromptMessageCount: 1,
    messages: [
      { role: "system", content: "memory prefix" },
      { role: "user", content: "[OpenClaw heartbeat poll]" },
      { role: "assistant", content: "HEARTBEAT_OK" },
      { role: "user", content: "normal user turn" },
      { role: "assistant", content: "normal reply" },
    ],
  });
  assert.equal(lifecycleRequests.at(-1).url, "https://memory.example/api/v1/after_turn");
  assert.equal(lifecycleRequests.at(-1).body.prePromptMessageCount, 1);
  assert.deepEqual(lifecycleRequests.at(-1).body.messages, [
    { role: "system", content: "memory prefix" },
    { role: "user", content: "normal user turn" },
    { role: "assistant", content: "normal reply" },
  ]);

  const requestCountBeforeMixedBatch = lifecycleRequests.length;
  await contextEngine.afterTurn({
    sessionId: "normal-before-heartbeat-without-boundary",
    messages: [
      { role: "user", content: "normal user turn" },
      { role: "assistant", content: "normal reply" },
      { role: "user", content: "[OpenClaw heartbeat poll]" },
      { role: "assistant", content: "HEARTBEAT_OK" },
    ],
  });
  assert.equal(lifecycleRequests.length, requestCountBeforeMixedBatch + 1);
  assert.deepEqual(lifecycleRequests.at(-1).body.messages, [
    { role: "user", content: "normal user turn" },
    { role: "assistant", content: "normal reply" },
  ]);

  const requestCountBeforePureHeartbeat = lifecycleRequests.length;
  await contextEngine.afterTurn({
    sessionId: "heartbeat-without-boundary",
    messages: [
      { role: "user", content: "[OpenClaw heartbeat poll]" },
      { role: "assistant", content: "HEARTBEAT_OK" },
    ],
  });
  assert.equal(lifecycleRequests.length, requestCountBeforePureHeartbeat);

  await contextEngine.afterTurn({
    sessionId: "session-1",
    accountId: "turn-acct",
    userId: "turn-user",
    agentId: "turn-agent",
    messages: [],
  });
  const requestCountBeforeDispose = lifecycleRequests.length;
  assert.deepEqual(await contextEngine.dispose(), {
    disposed: true,
    flushed: false,
    reason: "run_cleanup",
  });
  assert.equal(lifecycleRequests.length, requestCountBeforeDispose);

  await sessionEndHandlers.at(-1)(
    { sessionId: "session-1", reason: "reset" },
    {},
  );
  assert.equal(disposeAttempts, 2);
  assert.deepEqual(lifecycleRequests.at(-1), {
    url: "https://memory.example/api/v1/dispose",
    body: {
      sessionId: "session-1",
      accountId: "turn-acct",
      userId: "turn-user",
      agentId: "turn-agent",
    },
  });

  await contextEngine.afterTurn({
    sessionId: "session-2",
    accountId: "retained-acct",
    userId: "retained-user",
    agentId: "retained-agent",
    messages: [],
  });
  disposeFailuresRemaining = 3;
  assert.equal(
    await sessionEndHandlers.at(-1)(
      { sessionId: "session-2", reason: "reset" },
      {},
    ),
    undefined,
  );
  assert.equal(disposeAttempts, 5);

  await sessionEndHandlers.at(-1)(
    { sessionId: "session-2", reason: "reset" },
    {},
  );
  assert.equal(disposeAttempts, 6);
  assert.deepEqual(lifecycleRequests.at(-1), {
    url: "https://memory.example/api/v1/dispose",
    body: {
      sessionId: "session-2",
      accountId: "acct-1",
      userId: "user-1",
      agentId: "agent-1",
    },
  });

  await contextEngine.afterTurn({
    sessionId: "session-3",
    accountId: "retry-acct",
    userId: "retry-user",
    agentId: "retry-agent",
    messages: [],
  });
  disposeResult = {
    disposed: true,
    flushed: false,
    reason: "thread_unavailable",
  };
  assert.equal(
    await sessionEndHandlers.at(-1)(
      { sessionId: "session-3", reason: "reset" },
      {},
    ),
    undefined,
  );
  assert.equal(disposeAttempts, 9);

  disposeResult = { disposed: true };
  await sessionEndHandlers.at(-1)(
    { sessionId: "session-3", reason: "reset" },
    {},
  );
  assert.equal(disposeAttempts, 10);
  assert.deepEqual(lifecycleRequests.at(-1), {
    url: "https://memory.example/api/v1/dispose",
    body: {
      sessionId: "session-3",
      accountId: "acct-1",
      userId: "user-1",
      agentId: "agent-1",
    },
  });
} finally {
  globalThis.fetch = originalFetch;
}

const layered = buildLayeredComposeMessages(
  {
    identityContext: "profile",
    episodicContext: "archive",
    sessionContext: "session",
    retrievedEvidence: "retrieved",
    messages: [
      { role: "user", content: "server profile", _ogmem: true },
      { role: "user", content: "question" },
      { role: "user", content: "server retrieved", _ogmem: true },
    ],
  },
  [],
);
assert.deepEqual(layered.map((m) => m.content), [
  "[User Profile - background, identity, preferences, and stable traits]\nprofile",
  "[Episodic Memory - past conversations, events, and experiences]\narchive",
  "question",
  "[Current Session - recent messages and context from this conversation]\nsession",
  "[Retrieved Memories - relevant facts recalled from long-term memory]\nretrieved",
]);

const prepareCalls = [];
const prepared = await prepareCompactionWithTransport({
  call: async (method, params) => {
    prepareCalls.push({ method, params });
    return { ok: true, prepared: true, prepareToken: "prepare-1" };
  },
  params: { sessionId: "sess-prepare" },
});

assert.deepEqual(prepared, { ok: true, prepared: true, prepareToken: "prepare-1" });
assert.deepEqual(prepareCalls, [
  {
    method: "prepare_compaction",
    params: { sessionId: "sess-prepare" },
  },
]);

const takeoverCalls = [];
const takeoverResult = await runCompactControlFlow({
  cfg: resolveConfig({
    compactTakeoverEnabled: true,
    summaryMaxChars: 88,
    shortTermIndexMode: "sync",
    accountId: "acct-1",
    userId: "user-1",
    agentId: "agent-1",
  }),
  params: { sessionId: "sess-takeover", tokenBudget: 4096 },
  call: async (method, params) => {
    takeoverCalls.push({ method, params });
    if (method === "prepare_compaction") return { ok: true, prepareToken: "prepare-2" };
    if (method === "compact") {
      return { ok: true, compacted: true, result: { summary: "takeover-summary" } };
    }
    throw new Error(`unexpected method: ${method}`);
  },
});

assert.deepEqual(takeoverResult, {
  ok: true,
  compacted: true,
  result: { summary: "takeover-summary" },
});
assert.deepEqual(takeoverCalls, [
  {
    method: "prepare_compaction",
    params: {
      sessionId: "sess-takeover",
      tokenBudget: 4096,
      accountId: "acct-1",
      userId: "user-1",
      agentId: "agent-1",
      summaryMaxChars: 88,
      shortTermIndexMode: "sync",
    },
  },
  {
    method: "compact",
    params: {
      sessionId: "sess-takeover",
      tokenBudget: 4096,
      accountId: "acct-1",
      userId: "user-1",
      agentId: "agent-1",
      summaryMaxChars: 88,
      shortTermIndexMode: "sync",
      prepareToken: "prepare-2",
    },
  },
]);

const boundaryCalls = [];
const persistedBoundary = await appendTakeoverCompactionBoundary({
  sessionFile: "/tmp/session.jsonl",
  summary: "takeover-summary",
  tokensBefore: 99,
  details: { ok: true },
  loadSessionManagerModule: async () => ({
    SessionManager: {
      open: () => ({
        getLeafId: () => "entry-9",
        appendCompaction: (summary, firstKeptEntryId, tokensBefore, details) => {
          boundaryCalls.push({ summary, firstKeptEntryId, tokensBefore, details });
          return "compaction-1";
        },
      }),
    },
  }),
});

assert.deepEqual(persistedBoundary, {
  firstKeptEntryId: "entry-9",
  compactionEntryId: "compaction-1",
  reused: false,
});
assert.deepEqual(boundaryCalls, [
  {
    summary: "takeover-summary",
    firstKeptEntryId: "entry-9",
    tokensBefore: 99,
    details: { ok: true },
  },
]);

const assistantAfterToolManager = createFakeSessionManager(
  [
    messageEntry("entry-user", "user", "read the large file"),
    messageEntry("entry-tool-call", "assistant", [
      { type: "text", text: "reading" },
      { type: "toolCall", id: "call-1", name: "read_file", arguments: {} },
    ]),
    messageEntry("entry-tool-result", "toolResult", [
      { type: "text", text: "TOOL_RESULT_BLOB_MARKER" },
    ]),
    messageEntry("entry-final-assistant", "assistant", "I checked it."),
  ],
  "compaction-assistant-tail",
);
const assistantAfterToolBoundary = await appendTakeoverCompactionBoundary({
  sessionFile: "/tmp/session-assistant-tail.jsonl",
  summary: "summary",
  tokensBefore: 1000,
  loadSessionManagerModule: async () => ({
    SessionManager: { open: () => assistantAfterToolManager },
  }),
});
assert.equal(assistantAfterToolBoundary.firstKeptEntryId, "entry-final-assistant");
assert.equal(
  assistantAfterToolManager.appendCalls[0].firstKeptEntryId,
  "entry-final-assistant",
);

const toolLeafManager = createFakeSessionManager(
  [
    messageEntry("entry-user-tool-leaf", "user", "read the large file"),
    messageEntry("entry-tool-call-leaf", "assistant", [
      { type: "text", text: "reading" },
      { type: "tool_use", id: "call-2", name: "read_file", input: {} },
    ]),
    messageEntry("entry-tool-result-leaf", "toolResult", [
      { type: "text", text: "TOOL_RESULT_BLOB_MARKER" },
    ]),
  ],
  "compaction-tool-final",
  "custom-tool-boundary",
);
const toolLeafBoundary = await appendTakeoverCompactionBoundary({
  sessionFile: "/tmp/session-tool-leaf.jsonl",
  summary: "summary",
  tokensBefore: 1000,
  loadSessionManagerModule: async () => ({
    SessionManager: { open: () => toolLeafManager },
  }),
});
assert.equal(toolLeafBoundary.firstKeptEntryId, "custom-tool-boundary");
assert.equal(toolLeafBoundary.compactionEntryId, "compaction-tool-final");
assert.equal(toolLeafManager.customAppendCalls.length, 1);
assert.equal(toolLeafManager.appendCalls.length, 1);
assert.equal(toolLeafManager.appendCalls[0].firstKeptEntryId, "custom-tool-boundary");

const fallbackHardCheckpointCalls = [];
const fallbackHardCheckpointManager = {
  getBranch: () => [
    messageEntry("entry-fallback-user", "user", "read the large file"),
    messageEntry("entry-fallback-tool-result", "toolResult", [
      { type: "text", text: "TOOL_RESULT_BLOB_MARKER" },
    ]),
  ],
  getLeafId: () => "entry-fallback-tool-result",
  getLeafEntry: () => messageEntry("entry-fallback-tool-result", "toolResult"),
  appendCompaction: (summary, firstKeptEntryId, tokensBefore, details) => {
    const id =
      fallbackHardCheckpointCalls.length === 0
        ? "compaction-fallback-anchor"
        : "compaction-fallback-final";
    fallbackHardCheckpointCalls.push({ summary, firstKeptEntryId, tokensBefore, details });
    return id;
  },
};
const fallbackHardCheckpointBoundary = await appendTakeoverCompactionBoundary({
  sessionFile: "/tmp/session-fallback-hard-checkpoint.jsonl",
  summary: "summary",
  tokensBefore: 1000,
  loadSessionManagerModule: async () => ({
    SessionManager: { open: () => fallbackHardCheckpointManager },
  }),
});
assert.equal(fallbackHardCheckpointBoundary.firstKeptEntryId, "compaction-fallback-anchor");
assert.equal(fallbackHardCheckpointBoundary.compactionEntryId, "compaction-fallback-final");
assert.equal(fallbackHardCheckpointCalls.length, 2);
assert.equal(fallbackHardCheckpointCalls[0].firstKeptEntryId, "entry-fallback-user");
assert.equal(
  fallbackHardCheckpointCalls[1].firstKeptEntryId,
  "compaction-fallback-anchor",
);

const unsafeExistingCompactionManager = createFakeSessionManager(
  [
    compactionEntry("entry-existing-summary", "entry-existing-summary", "old summary"),
    messageEntry("entry-existing-tool-call", "assistant", [
      { type: "tool_use", id: "call-existing", name: "read_file", input: {} },
    ]),
    messageEntry("entry-existing-tool-result", "toolResult", [
      { type: "text", text: "TOOL_RESULT_BLOB_MARKER" },
    ]),
    compactionEntry(
      "entry-existing-leaf-compaction",
      "entry-existing-summary",
      "unsafe leaf summary",
    ),
  ],
  "compaction-existing-hardened",
  "custom-existing-boundary",
);
const unsafeExistingCompactionBoundary = await appendTakeoverCompactionBoundary({
  sessionFile: "/tmp/session-existing-compaction.jsonl",
  summary: "fresh summary",
  tokensBefore: 1000,
  loadSessionManagerModule: async () => ({
    SessionManager: { open: () => unsafeExistingCompactionManager },
  }),
});
assert.equal(
  unsafeExistingCompactionBoundary.firstKeptEntryId,
  "custom-existing-boundary",
);
assert.equal(
  unsafeExistingCompactionBoundary.compactionEntryId,
  "compaction-existing-hardened",
);
assert.equal(unsafeExistingCompactionManager.customAppendCalls.length, 1);
assert.equal(unsafeExistingCompactionManager.appendCalls.length, 1);
assert.equal(
  unsafeExistingCompactionManager.appendCalls[0].firstKeptEntryId,
  "custom-existing-boundary",
);

const manualManager = createFakeSessionManager(
  [
    messageEntry("entry-manual-user", "user", "old request"),
    messageEntry("entry-manual-assistant", "assistant", "old response"),
  ],
  "compaction-manual-final",
  "custom-manual-boundary",
);
const manualBoundary = await appendTakeoverCompactionBoundary({
  sessionFile: "/tmp/session-manual.jsonl",
  summary: "manual summary",
  tokensBefore: 100,
  params: {
    force: true,
    compactionTarget: "threshold",
    runtimeContext: { trigger: "manual" },
  },
  loadSessionManagerModule: async () => ({
    SessionManager: { open: () => manualManager },
  }),
});
assert.equal(manualBoundary.firstKeptEntryId, "custom-manual-boundary");
assert.equal(manualBoundary.compactionEntryId, "compaction-manual-final");
assert.equal(manualManager.customAppendCalls.length, 1);
assert.equal(manualManager.appendCalls.length, 1);
assert.equal(manualManager.appendCalls[0].firstKeptEntryId, "custom-manual-boundary");

const upstreamBoundaryManager = createFakeSessionManager(
  [
    messageEntry("entry-upstream-user", "user", "safe user"),
    messageEntry("entry-upstream-assistant", "assistant", "safe assistant"),
  ],
  "compaction-upstream",
);
const upstreamBoundary = await appendTakeoverCompactionBoundary({
  sessionFile: "/tmp/session-upstream.jsonl",
  summary: "upstream summary",
  tokensBefore: 100,
  upstreamFirstKeptEntryId: "entry-upstream-user",
  loadSessionManagerModule: async () => ({
    SessionManager: { open: () => upstreamBoundaryManager },
  }),
});
assert.equal(upstreamBoundary.firstKeptEntryId, "entry-upstream-user");
assert.equal(
  upstreamBoundaryManager.appendCalls[0].firstKeptEntryId,
  "entry-upstream-user",
);

const upstreamToolTailManager = createFakeSessionManager(
  [
    messageEntry("entry-upstream-tool-call", "assistant", [
      { type: "toolCall", id: "call-upstream", name: "read_file", arguments: {} },
    ]),
    messageEntry("entry-upstream-tool-result", "toolResult", [
      { type: "text", text: "small result" },
    ]),
  ],
  "compaction-upstream-tool-tail",
);
const upstreamToolTailBoundary = await appendTakeoverCompactionBoundary({
  sessionFile: "/tmp/session-upstream-tool-tail.jsonl",
  summary: "upstream tool tail summary",
  tokensBefore: 100,
  upstreamFirstKeptEntryId: "entry-upstream-tool-call",
  loadSessionManagerModule: async () => ({
    SessionManager: { open: () => upstreamToolTailManager },
  }),
});
assert.equal(upstreamToolTailBoundary.firstKeptEntryId, "entry-upstream-tool-call");
assert.equal(
  upstreamToolTailManager.appendCalls[0].firstKeptEntryId,
  "entry-upstream-tool-call",
);

for (const [name, branch] of [
  [
    "object-tool-call",
    [
      messageEntry("entry-object-tool-call", "assistant", {
        type: "toolCall",
        id: "call-object",
        name: "read_file",
        arguments: {},
      }),
      messageEntry("entry-object-tool-result", "toolResult", [
        { type: "text", text: "TOOL_RESULT_BLOB_MARKER" },
      ]),
    ],
  ],
  [
    "function-call",
    [
      messageEntry("entry-function-call", "assistant", {
        type: "function_call",
        id: "call-function",
        name: "read_file",
        arguments: "{}",
      }),
      messageEntry("entry-function-tool-result", "tool", [
        { type: "text", text: "TOOL_RESULT_BLOB_MARKER" },
      ]),
    ],
  ],
  [
    "stopreason-tooluse",
    [
      messageEntry("entry-stopreason-tooluse", "assistant", "calling tool", {
        stopReason: "toolUse",
      }),
    ],
  ],
]) {
  const customBoundaryId = `custom-${name}-boundary`;
  const manager = createFakeSessionManager(
    branch,
    `compaction-${name}-final`,
    customBoundaryId,
  );
  const boundary = await appendTakeoverCompactionBoundary({
    sessionFile: `/tmp/session-${name}.jsonl`,
    summary: `${name} summary`,
    tokensBefore: 100,
    loadSessionManagerModule: async () => ({
      SessionManager: { open: () => manager },
    }),
  });
  assert.equal(boundary.firstKeptEntryId, customBoundaryId);
  assert.equal(manager.customAppendCalls.length, 1);
  assert.equal(manager.appendCalls.length, 1);
  assert.equal(manager.appendCalls[0].firstKeptEntryId, customBoundaryId);
}

const emptySummaryManager = createFakeSessionManager(
  [messageEntry("entry-empty-summary-user", "user", "pending request")],
  "compaction-empty-summary",
);
const emptySummaryBoundary = await appendTakeoverCompactionBoundary({
  sessionFile: "/tmp/session-empty-summary.jsonl",
  summary: "",
  tokensBefore: 100,
  loadSessionManagerModule: async () => ({
    SessionManager: { open: () => emptySummaryManager },
  }),
});
assert.equal(emptySummaryBoundary.firstKeptEntryId, "entry-empty-summary-user");
assert.equal(emptySummaryManager.appendCalls[0].summary, "");

const takeoverWithBoundaryCalls = [];
const takeoverWithBoundaryResult = await runCompactControlFlow({
  cfg: resolveConfig({
    compactTakeoverEnabled: true,
    summaryMaxChars: 88,
    shortTermIndexMode: "sync",
  }),
  params: {
    sessionId: "sess-takeover-boundary",
    sessionFile: "/tmp/session.jsonl",
    tokenBudget: 4096,
  },
  call: async (method, params) => {
    takeoverWithBoundaryCalls.push({ method, params });
    if (method === "prepare_compaction") return { ok: true, prepareToken: "prepare-9" };
    if (method === "compact") {
      return {
        ok: true,
        compacted: true,
        result: {
          summary: "boundary-summary",
          tokensBefore: 123,
          tokensAfter: 45,
          details: { safe: true },
        },
      };
    }
    throw new Error(`unexpected method: ${method}`);
  },
  persistTakeoverCompactionBoundary: async ({ sessionFile, summary, tokensBefore, details }) => {
    takeoverWithBoundaryCalls.push({
      method: "persist_boundary",
      params: { sessionFile, summary, tokensBefore, details },
    });
    return {
      firstKeptEntryId: "entry-42",
      compactionEntryId: "compaction-42",
      reused: false,
    };
  },
});

assert.equal(takeoverWithBoundaryResult.result.firstKeptEntryId, "entry-42");
assert.deepEqual(takeoverWithBoundaryCalls, [
  {
    method: "prepare_compaction",
    params: {
      sessionId: "sess-takeover-boundary",
      sessionFile: "/tmp/session.jsonl",
      tokenBudget: 4096,
      accountId: "",
      userId: "",
      agentId: "",
      summaryMaxChars: 88,
      shortTermIndexMode: "sync",
    },
  },
  {
    method: "compact",
    params: {
      sessionId: "sess-takeover-boundary",
      sessionFile: "/tmp/session.jsonl",
      tokenBudget: 4096,
      accountId: "",
      userId: "",
      agentId: "",
      summaryMaxChars: 88,
      shortTermIndexMode: "sync",
      prepareToken: "prepare-9",
    },
  },
  {
    method: "persist_boundary",
    params: {
      sessionFile: "/tmp/session.jsonl",
      summary: "boundary-summary",
      tokensBefore: 123,
      details: { safe: true },
    },
  },
]);

const delegatedCalls = [];
const delegated = await runCompactControlFlow({
  cfg: resolveConfig({
    compactTakeoverEnabled: false,
    summaryMaxChars: 99,
    shortTermIndexMode: "async",
    accountId: "acct-2",
  }),
  params: { sessionId: "sess-delegate", tokenBudget: 2048 },
  call: async (method, params) => {
    delegatedCalls.push({ method, params });
    return { ok: true, prepareToken: "prepare-3" };
  },
  runtimeDelegate: async (params) => ({
    ok: true,
    compacted: true,
    result: { summary: `delegated:${params.sessionId}` },
  }),
});

assert.deepEqual(delegated, {
  ok: true,
  compacted: true,
  result: { summary: "delegated:sess-delegate" },
});
assert.deepEqual(delegatedCalls, [
  {
    method: "prepare_compaction",
    params: {
      sessionId: "sess-delegate",
      tokenBudget: 2048,
      accountId: "acct-2",
      userId: "",
      agentId: "",
      summaryMaxChars: 99,
      shortTermIndexMode: "async",
    },
  },
]);

await assert.rejects(
  delegateCompactionToRuntimeOrThrow({
    params: { sessionId: "sess-no-delegate" },
    loadDelegate: async () => null,
  }),
  /delegateCompactionToRuntime unavailable/,
);
