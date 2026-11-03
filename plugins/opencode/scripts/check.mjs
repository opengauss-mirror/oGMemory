import { readdir } from "node:fs/promises";
import { spawn } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const files = [join(root, "index.mjs"), ...(await readdir(join(root, "lib"))).filter((name) => name.endsWith(".mjs")).map((name) => join(root, "lib", name))];
for (const file of files) {
  await new Promise((resolve, reject) => {
    const child = spawn(process.execPath, ["--check", file], { stdio: "inherit", windowsHide: true });
    child.once("exit", (code) => code === 0 ? resolve() : reject(new Error(`Syntax check failed: ${file}`)));
  });
}
