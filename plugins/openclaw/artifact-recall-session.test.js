import assert from "node:assert/strict";
import test from "node:test";

import plugin, { bindTrustedArtifactRecallSession } from "./index.js";


const RECALL_TOOL_NAMES = [
  "list_tool_result_artifacts",
  "read_tool_result_artifact",
  "search_tool_result_artifacts",
  "ogmemory__list_tool_result_artifacts",
  "ogmemory__read_tool_result_artifact",
  "ogmemory__search_tool_result_artifacts",
];


test("Artifact recall always replaces model-provided session identity", () => {
  for (const toolName of RECALL_TOOL_NAMES) {
    assert.deepEqual(
      bindTrustedArtifactRecallSession(
        toolName,
        {
          session_id: "agent:main:explicit:model-selected-session",
          sessionId: "another-model-selected-session",
          query: "Finished",
          ref: "ogmemory://tool-results/tr_example",
        },
        "trusted-runtime-session",
      ),
      {
        session_id: "trusted-runtime-session",
        query: "Finished",
        ref: "ogmemory://tool-results/tr_example",
      },
      toolName,
    );
  }
});


test("Artifact recall session binding ignores unrelated tools and missing trusted identity", () => {
  assert.equal(
    bindTrustedArtifactRecallSession("run_cargo_build", { session_id: "model" }, "trusted"),
    null,
  );
  assert.equal(
    bindTrustedArtifactRecallSession(
      "ogmemory__search_tool_result_artifacts",
      { session_id: "model" },
      "",
    ),
    null,
  );
});


function registerPlugin(toolResultExternalizationEnabled) {
  const hooks = new Map();
  plugin.register({
    pluginConfig: {
      mode: "remote",
      memoryApiBaseUrl: "http://ogmem.invalid",
      toolResultExternalizationEnabled,
      toolResultFirstViewExternalizationEnabled: false,
    },
    logger: { info() {}, warn() {}, error() {} },
    on(name, handler) {
      const handlers = hooks.get(name) || [];
      handlers.push(handler);
      hooks.set(name, handlers);
    },
    registerService() {},
    registerContextEngine() {},
  });
  return hooks;
}


test("enabled plugin injects the trusted OpenClaw session before Artifact recall", async () => {
  const hooks = registerPlugin(true);
  const results = [];
  for (const handler of hooks.get("before_tool_call") || []) {
    results.push(await handler(
      {
        toolName: "ogmemory__search_tool_result_artifacts",
        params: {
          session_id: "agent:main:explicit:wrong-session",
          query: "foo_399",
        },
      },
      {
        toolName: "ogmemory__search_tool_result_artifacts",
        sessionId: "trusted-session",
      },
    ));
  }
  const rewrite = results.find((result) => result?.params?.session_id === "trusted-session");
  assert.deepEqual(rewrite, {
    params: {
      session_id: "trusted-session",
      query: "foo_399",
    },
  });
});


test("enabled plugin blocks Artifact recall without a trusted OpenClaw session", async () => {
  const hooks = registerPlugin(true);
  const results = [];
  for (const handler of hooks.get("before_tool_call") || []) {
    results.push(await handler(
      {
        toolName: "ogmemory__read_tool_result_artifact",
        params: {
          session_id: "model-selected-session",
          ref: "ogmemory://tool-results/tr_example",
        },
      },
      { toolName: "ogmemory__read_tool_result_artifact" },
    ));
  }
  assert.ok(results.some((result) => (
    result?.block === true
    && result?.blockReason === "oGMemory Artifact recall requires a trusted OpenClaw session ID"
  )));
});


test("disabled plugin preserves the original Artifact tool path", () => {
  const disabledHooks = registerPlugin(false);
  const enabledHooks = registerPlugin(true);
  assert.equal(
    (enabledHooks.get("before_tool_call") || []).length,
    (disabledHooks.get("before_tool_call") || []).length + 1,
  );
});
