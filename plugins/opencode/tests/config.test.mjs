import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { loadConfig } from "../lib/config.mjs";

async function withConfig(value, callback, env = {}) {
  const home = await mkdtemp(join(tmpdir(), "ogmemory-config-"));
  try {
    const directory = join(home, ".config", "opencode");
    await mkdir(directory, { recursive: true });
    await writeFile(join(directory, "ogmemory-config.json"), JSON.stringify(value));
    return await callback({ home, env });
  } finally {
    await rm(home, { recursive: true, force: true });
  }
}

test("configuration defaults are sufficient for a local oGMemory service", async () => {
  const home = await mkdtemp(join(tmpdir(), "ogmemory-config-"));
  try {
    const config = await loadConfig({ home, env: {} });
    assert.equal(config.enabled, true);
    assert.equal(config.autoCapture, true);
    assert.equal(config.autoRecall, true);
    assert.equal(config.endpoint, "http://127.0.0.1:8090");
    assert.equal(config.agentId, "opencode");
  } finally {
    await rm(home, { recursive: true, force: true });
  }
});

test("environment identity and endpoint values override the file", async () => {
  await withConfig(
    { endpoint: "https://memory.example/base/", accountId: "file-account" },
    async (options) => {
      const config = await loadConfig(options);
      assert.equal(config.endpoint, "https://other.example/api");
      assert.equal(config.accountId, "env-account");
      assert.equal(config.apiKey, "env-key");
    },
    {
      OG_MEMORY_URL: "https://other.example/api/",
      OG_AUTH_ACCOUNT_ID: "env-account",
      OG_AUTH_API_KEY: "env-key",
    },
  );
});

test("capture and recall switches use environment fallbacks and explicit JSON precedence", async () => {
  for (const [field, variable] of [["autoCapture", "OGMEM_AUTO_CAPTURE"], ["autoRecall", "OGMEM_AUTO_RECALL"]]) {
    for (const value of ["false", "0", "off", "no", " FALSE "]) {
      await withConfig({}, async (options) => {
        assert.equal((await loadConfig(options))[field], false);
      }, { [variable]: value });
    }
    for (const explicit of [true, false]) {
      await withConfig({ [field]: explicit }, async (options) => {
        assert.equal((await loadConfig(options))[field], explicit);
      }, { [variable]: String(!explicit) });
    }
    for (const value of ["false", 0, null]) {
      await assert.rejects(
        withConfig({ [field]: value }, (options) => loadConfig(options)),
        /expected boolean/,
      );
    }
  }
});

test("endpoint policy rejects credentials, query, fragment, and remote plaintext", async () => {
  for (const endpoint of [
    "https://user:pass@example.com",
    "https://example.com?",
    "https://example.com#",
    "http://example.com",
  ]) {
    await assert.rejects(
      withConfig({ endpoint }, (options) => loadConfig(options)),
      /Invalid oGMemory configuration field endpoint/,
      endpoint,
    );
  }
});

test("unknown fields and invalid field types fail closed", async () => {
  await assert.rejects(
    withConfig({ traceSpoolLimit: 10 }, (options) => loadConfig(options)),
    /unexpected field/,
  );
  await assert.rejects(
    withConfig({ enabled: "yes" }, (options) => loadConfig(options)),
    /expected boolean/,
  );
});

test("numeric configuration boundaries reject invalid values", async () => {
  const ranges = {
    composeTimeoutMs: [100, 60000],
    writeTimeoutMs: [100, 120000],
    traceTimeoutMs: [100, 60000],
  };
  for (const [field, [min, max]] of Object.entries(ranges)) {
    for (const value of [min - 1, max + 1, 1.5, "not-a-number", null]) {
      await assert.rejects(
        withConfig({ [field]: value }, (options) => loadConfig(options)),
        /expected an integer/,
        `${field}=${value}`,
      );
    }
    await withConfig({ [field]: min }, async (options) => assert.equal((await loadConfig(options))[field], min));
    await withConfig({ [field]: max }, async (options) => assert.equal((await loadConfig(options))[field], max));
  }
});
