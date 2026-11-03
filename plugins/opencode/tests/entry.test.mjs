import test from "node:test";
import assert from "node:assert/strict";

test("public entry exports only the plugin factory aliases", async () => {
  const entry = await import("../index.mjs");
  assert.deepEqual(Object.keys(entry).sort(), ["OgMemoryPlugin", "default"]);
  assert.equal(entry.default, entry.OgMemoryPlugin);
});
