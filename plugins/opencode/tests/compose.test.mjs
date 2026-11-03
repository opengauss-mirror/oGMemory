import test from "node:test";
import assert from "node:assert/strict";
import { createComposeController, formatComposeResult, MEMORY_MARKER } from "../lib/compose.mjs";

test("compose extracts only server-marked memory messages", () => {
  assert.equal(formatComposeResult({ messages: [
    { role: "user", content: "original" },
    { role: "user", content: "profile", _ogmem: true },
    { role: "user", content: "working", _ogmem: true },
  ] }), "profile\n\nworking");
});

test("compose combines the complete legacy server context", () => {
  const context = formatComposeResult({
    systemPromptAddition: "PROFILE",
    systemPromptSuffix: "ARCHIVE_AND_SESSION",
    memoryUserMessage: "RETRIEVED_EVIDENCE",
  });
  for (const value of ["PROFILE", "ARCHIVE_AND_SESSION", "RETRIEVED_EVIDENCE"]) {
    assert.equal(context.split(value).length - 1, 1);
  }
});

test("compose prefers marked messages over duplicate legacy fields", () => {
  assert.equal(formatComposeResult({
    messages: [{ content: "CURRENT", _ogmem: true }],
    systemPromptAddition: "LEGACY",
  }), "CURRENT");
});

test("system transform injects cached recall once", async () => {
  const controller = createComposeController({ http: { compose: async () => ({ retrievedEvidence: "remember me" }) } });
  await controller.recall("s1", "m1", "question");
  const output = { system: ["base"] };
  assert.equal(controller.inject("s1", output), true);
  assert.match(output.system[1], new RegExp(MEMORY_MARKER.replace(/[\[\]]/g, "\\$&")));
  assert.equal(controller.inject("s1", output), false);
});

test("parallel agent sessions keep recalled memory isolated", async () => {
  const controller = createComposeController({
    http: { compose: async (session) => ({ retrievedEvidence: `memory-for-${session}` }) },
  });
  await Promise.all([
    controller.recall("parent-session", "parent-message", "parent question"),
    controller.recall("child-session", "child-message", "child question"),
  ]);

  const parent = { system: [] };
  const child = { system: [] };
  assert.equal(controller.inject("parent-session", parent), true);
  assert.equal(controller.inject("child-session", child), true);
  assert.match(parent.system[0], /memory-for-parent-session/);
  assert.doesNotMatch(parent.system[0], /memory-for-child-session/);
  assert.match(child.system[0], /memory-for-child-session/);
  assert.doesNotMatch(child.system[0], /memory-for-parent-session/);
});

test("a failed recall never reuses previous-turn context", async () => {
  let fail = false;
  const controller = createComposeController({ http: { compose: async () => { if (fail) throw new Error("offline"); return { retrievedEvidence: "old memory" }; } }, logger: { warn() {} } });
  await controller.recall("s1", "m1", "first");
  fail = true;
  await controller.recall("s1", "m2", "second");
  const output = { system: [] };
  assert.equal(controller.inject("s1", output), false);
});

test("only the newest concurrent recall may populate cache and entries expire", async () => {
  let resolveOld;
  let clock = 0;
  const old = new Promise((resolve) => { resolveOld = resolve; });
  const controller = createComposeController({
    http: { compose: async (_session, prompt) => prompt === "old" ? old : { retrievedEvidence: "new memory" } },
    ttlMs: 10, now: () => clock,
  });
  const pending = controller.recall("s", "m1", "old");
  await controller.recall("s", "m2", "new");
  resolveOld({ retrievedEvidence: "old memory" });
  await pending;
  const fresh = { system: [] };
  assert.equal(controller.inject("s", fresh), true);
  assert.match(fresh.system[0], /new memory/);
  clock = 11;
  assert.equal(controller.inject("s", { system: [] }), false);
});

test("an older failed recall cannot delete newer cached context", async () => {
  let rejectOld;
  const old = new Promise((_, reject) => { rejectOld = reject; });
  const controller = createComposeController({
    http: { compose: async (_session, prompt) => prompt === "old" ? old : { retrievedEvidence: "new memory" } },
    logger: { warn() {} },
  });
  const pending = controller.recall("s", "m1", "old");
  await controller.recall("s", "m2", "new");
  rejectOld(new Error("late failure"));
  await pending;
  const output = { system: [] };
  assert.equal(controller.inject("s", output), true);
  assert.match(output.system[0], /new memory/);
});

test("compose warnings are redacted, rate-limited, and report recovery", async () => {
  let clock = 0;
  let fail = true;
  const warnings = [];
  const infos = [];
  const controller = createComposeController({
    http: { compose: async () => { if (fail) throw new Error("password=hunter2 person@example.com"); return { retrievedEvidence: "recovered" }; } },
    logger: { warn: (message) => warnings.push(message), info: (message) => infos.push(message) },
    logIntervalMs: 100, now: () => clock,
  });
  await controller.recall("s", "m1", "one");
  clock = 50;
  await controller.recall("s", "m2", "two");
  assert.equal(warnings.length, 1);
  assert.doesNotMatch(warnings[0], /hunter2|person@example.com/);
  fail = false;
  await controller.recall("s", "m3", "three");
  assert.match(infos[0], /recovered.*suppressing 1 warning/);
});

test("compose context is bounded before caching and injection", async () => {
  const controller = createComposeController({ http: { compose: async () => ({ retrievedEvidence: "x".repeat(200000) }) } });
  await controller.recall("s", "m", "question");
  const output = { system: [] };
  assert.equal(controller.inject("s", output), true);
  assert.ok(output.system[0].length < 65000);
});
