import test from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { cp, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { fileURLToPath, pathToFileURL } from "node:url";

const packageRoot = fileURLToPath(new URL("..", import.meta.url));
function runNpm(args, options) {
  if (process.platform !== "win32") return spawnSync("npm", args, options);
  const command = `npm.cmd ${args.map((value) => /[\s&|<>^]/.test(String(value)) ? `"${String(value).replaceAll('"', '""')}"` : String(value)).join(" ")}`;
  return spawnSync(process.env.ComSpec, ["/d", "/s", "/c", command], options);
}

test("packed npm artifact installs and exposes the plugin factory", async () => {
  const directory = await mkdtemp(join(tmpdir(), "ogmemory-pack-"));
  try {
    const packed = runNpm(["pack", "--json", "--pack-destination", directory], { cwd: packageRoot, encoding: "utf8", timeout: 30000 });
    assert.equal(packed.status, 0, packed.stderr || packed.stdout);
    const artifact = JSON.parse(packed.stdout)[0];
    assert.equal(artifact.files.some(({ path }) => path.startsWith("wrappers/")), false);
    assert.equal(artifact.files.some(({ path }) => path.startsWith("tests/")), false);
    assert.equal(artifact.files.some(({ path }) => path.startsWith("scripts/")), false);
    const filename = artifact.filename;
    const project = join(directory, "consumer");
    await mkdir(project);
    await writeFile(join(project, "package.json"), JSON.stringify({ type: "module", dependencies: { "@opengauss/ogmemory-opencode-plugin": `file:${join(directory, filename)}` } }));
    const installed = runNpm(["install", "--ignore-scripts", "--no-package-lock"], { cwd: project, encoding: "utf8", timeout: 30000 });
    assert.equal(installed.status, 0, installed.stderr || installed.stdout);
    await writeFile(join(project, "probe.mjs"), 'import plugin, { OgMemoryPlugin } from "@opengauss/ogmemory-opencode-plugin"; if (plugin !== OgMemoryPlugin || typeof plugin !== "function") process.exit(2);');
    const probe = spawnSync(process.execPath, ["probe.mjs"], { cwd: project, encoding: "utf8", timeout: 5000 });
    assert.equal(probe.status, 0, probe.stderr || probe.stdout);
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test("documented source wrapper layout imports the copied plugin", async () => {
  const directory = await mkdtemp(join(tmpdir(), "ogmemory-source-"));
  try {
    const plugins = join(directory, "plugins");
    const target = join(plugins, "ogmemory");
    await mkdir(target, { recursive: true });
    await cp(join(packageRoot, "lib"), join(target, "lib"), { recursive: true });
    await writeFile(join(target, "index.mjs"), await readFile(join(packageRoot, "index.mjs")));
    await writeFile(join(target, "package.json"), await readFile(join(packageRoot, "package.json")));
    await writeFile(join(plugins, "ogmemory.js"), await readFile(join(packageRoot, "wrappers", "ogmemory.js")));
    const imported = await import(`${pathToFileURL(join(plugins, "ogmemory.js")).href}?test=${Date.now()}`);
    assert.equal(typeof imported.default, "function");
    assert.equal(imported.default, imported.OgMemoryPlugin);
  } finally { await rm(directory, { recursive: true, force: true }); }
});
