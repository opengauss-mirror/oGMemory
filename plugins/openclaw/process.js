/**
 * process.js — Process management utilities for oG-Memory plugin.
 *
 * Provides health checking, port probing, and process lifecycle helpers.
 */

import { execSync } from "node:child_process";
import { Socket } from "node:net";
import { platform } from "node:os";

const IS_WIN = platform() === "win32";

// ---------------------------------------------------------------------------
// Health checking
// ---------------------------------------------------------------------------

/**
 * Poll `baseUrl` until it returns `{ status: "ok" }` (or any 2xx for AGFS).
 * Rejects on timeout.
 */
export function waitForHealth(baseUrl, timeoutMs, intervalMs = 500, validator) {
  const deadline = Date.now() + timeoutMs;
  return new Promise((resolve, reject) => {
    const tick = () => {
      if (Date.now() > deadline) {
        reject(new Error(`Health check timeout after ${timeoutMs}ms: ${baseUrl}`));
        return;
      }
      fetch(baseUrl, { signal: AbortSignal.timeout(3000) })
        .then((r) => {
          // If a validator is provided, accept any HTTP response (even non-2xx)
          // This handles AGFS which returns 404 for / but is still healthy
          if (validator) { resolve(); return; }
          if (!r.ok) { setTimeout(tick, intervalMs); return r; }
          return r.json().catch(() => ({})).then((body) => {
            if (isHealthy(body)) { resolve(); return; }
            setTimeout(tick, intervalMs);
          });
        })
        .catch(() => setTimeout(tick, intervalMs));
    };
    tick();
  });
}

/**
 * Wait for health but also watch the child process — reject early if it exits.
 */
export function waitForHealthOrExit(baseUrl, timeoutMs, intervalMs, child, validator) {
  if (child.killed || child.exitCode !== null || child.signalCode !== null) {
    return Promise.reject(
      new Error(`Subprocess exited before health check (code=${child.exitCode}, signal=${child.signalCode})`),
    );
  }

  return new Promise((resolve, reject) => {
    let settled = false;
    const cleanup = () => { child.off("error", onError); child.off("exit", onExit); };
    const finish = (fn, arg) => { if (settled) return; settled = true; cleanup(); fn(arg); };

    const onError = (err) => finish(reject, err);
    const onExit = (code, signal) => finish(reject, new Error(`Subprocess exited (code=${code}, signal=${signal})`));

    child.once("error", onError);
    child.once("exit", onExit);

    waitForHealth(baseUrl, timeoutMs, intervalMs, validator)
      .then(() => finish(resolve))
      .catch((err) => finish(reject, err));
  });
}

/**
 * Fast health check — returns true/false within timeoutMs.
 */
export async function quickHealthCheck(baseUrl, timeoutMs = 2000) {
  try {
    const resp = await fetch(baseUrl, { signal: AbortSignal.timeout(timeoutMs) });
    if (!resp.ok) return false;
    const body = await resp.json().catch(() => ({}));
    return isHealthy(body);
  } catch {
    return false;
  }
}

function isHealthy(body) {
  if (!body || typeof body !== "object") return false;
  // ContextEngine health: { status: "ok", ... }
  if (body.status === "ok") return true;
  // AGFS doesn't have a standard health endpoint, accept any 2xx JSON response
  return false;
}

// ---------------------------------------------------------------------------
// Port management
// ---------------------------------------------------------------------------

/**
 * Quick TCP probe — check if a port is listening.
 */
export function quickTcpProbe(host, port, timeoutMs = 500) {
  return new Promise((resolve) => {
    const socket = new Socket();
    let done = false;
    const finish = (ok) => { if (done) return; done = true; socket.destroy(); resolve(ok); };
    socket.setTimeout(timeoutMs);
    socket.once("connect", () => finish(true));
    socket.once("timeout", () => finish(false));
    socket.once("error", () => finish(false));
    try { socket.connect(port, host); } catch { finish(false); }
  });
}

/**
 * Prepare a port for local service startup.
 * - If port has a healthy instance of our service → kill it, reuse port.
 * - If port is occupied by something else → try next free port.
 * - If port is free → return as-is.
 */
export async function prepareLocalPort(port, healthUrlFn, logger, maxRetries = 10) {
  const healthy = await quickHealthCheck(healthUrlFn(port), 2000);
  if (healthy) {
    logger.info(`killing stale service on port ${port}`);
    await killProcessOnPort(port, logger);
    return port;
  }

  const occupied = await quickTcpProbe("127.0.0.1", port, 500);
  if (!occupied) return port;

  logger.warn(`port ${port} occupied by another process, searching for free port...`);
  for (let candidate = port + 1; candidate <= port + maxRetries; candidate++) {
    if (candidate > 65535) break;
    const taken = await quickTcpProbe("127.0.0.1", candidate, 300);
    if (!taken) {
      logger.info(`using free port ${candidate} instead of ${port}`);
      return candidate;
    }
  }
  throw new Error(`Port ${port} occupied, no free port in range ${port + 1}-${port + maxRetries}`);
}

// ---------------------------------------------------------------------------
// Process cleanup
// ---------------------------------------------------------------------------

async function killProcessOnPort(port, logger) {
  if (IS_WIN) {
    try {
      const out = execSync(`netstat -ano | findstr "LISTENING" | findstr ":${port}"`, { encoding: "utf-8", shell: "cmd.exe" }).trim();
      const pids = new Set();
      for (const line of out.split(/\r?\n/)) { const m = line.trim().match(/\s(\d+)\s*$/); if (m) pids.add(Number(m[1])); }
      for (const pid of pids) { if (pid > 0) { try { execSync(`taskkill /PID ${pid} /F`, { shell: "cmd.exe" }); } catch {} } }
    } catch {}
  } else {
    try {
      const out = execSync(`lsof -ti tcp:${port} -s tcp:listen 2>/dev/null || true`, { encoding: "utf-8", shell: "/bin/sh" }).trim();
      if (out) {
        const pids = out.split(/\s+/).map(Number).filter((n) => n > 0);
        for (const pid of pids) { try { process.kill(pid, "SIGKILL"); } catch {} }
        if (pids.length) await new Promise((r) => setTimeout(r, 500));
      }
    } catch {}
  }
}

// ---------------------------------------------------------------------------
// Dependency checks
// ---------------------------------------------------------------------------

/**
 * Check if `go` is available on PATH.
 */
export function findGoBinary() {
  try {
    const out = execSync("go version", { encoding: "utf-8", shell: IS_WIN ? "cmd.exe" : "/bin/sh" }).trim();
    return { ok: true, version: out };
  } catch {
    return { ok: false, version: null };
  }
}

/**
 * Find Python binary — prefers venv, falls back to system python3.
 */
export function findPythonBin(projectRoot) {
  const venvPython = IS_WIN
    ? `${projectRoot}/.venv/Scripts/python.exe`
    : `${projectRoot}/.venv/bin/python`;
  try {
    execSync(`"${venvPython}" --version`, { encoding: "utf-8", stdio: "pipe" });
    return venvPython;
  } catch {}
  return IS_WIN ? "python" : "python3";
}
