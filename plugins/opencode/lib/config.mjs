import { readFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";

const DEFAULTS = Object.freeze({
  enabled: true,
  autoCapture: true,
  autoRecall: true,
  endpoint: "http://127.0.0.1:8090",
  accountId: "",
  userId: "",
  agentId: "opencode",
  apiKey: "",
  composeTimeoutMs: 5000,
  writeTimeoutMs: 15000,
  traceTimeoutMs: 3000,
  traceCaptureContent: true,
});

const envString = (env, key, fallback) => env[key]?.trim() || fallback;
const envBoolean = (env, key, fallback) => env[key] == null ? fallback : !["0", "false", "off", "no"].includes(env[key].trim().toLowerCase());
const configError = (reason, message) => Object.assign(new Error(message), { configReason: reason });
const boundedInt = (value, fallback, min, max) => {
  const number = Number(value);
  return Number.isInteger(number) && number >= min && number <= max ? number : fallback;
};

export function defaultConfigPath(home = homedir()) {
  return join(home, ".config", "opencode", "ogmemory-config.json");
}

export async function loadConfig({ env = process.env, home = homedir(), configPath } = {}) {
  let file = {};
  const path = configPath || defaultConfigPath(home);
  try {
    file = JSON.parse(await readFile(path, "utf8"));
    if (!file || Array.isArray(file) || typeof file !== "object") throw configError("PARSE_ERROR", "Invalid oGMemory configuration file");
  } catch (error) {
    if (error?.code !== "ENOENT") throw error?.configReason ? error : configError("PARSE_ERROR", "Invalid oGMemory configuration file");
  }

  for (const key of ["endpoint", "apiKey", "accountId", "userId", "agentId"]) {
    if (file[key] !== undefined && typeof file[key] !== "string") throw configError("INVALID_FIELD_TYPE", `Invalid oGMemory configuration field ${key}: expected string`);
  }
  const allowedKeys = new Set([...Object.keys(DEFAULTS), "runtimeDir"]);
  const unknownKeys = Object.keys(file).filter((key) => !allowedKeys.has(key));
  if (unknownKeys.length) throw configError("UNKNOWN_FIELD", `Invalid oGMemory configuration: unexpected field(s): ${unknownKeys.join(", ")}`);
  const numericFields = {
    composeTimeoutMs: [100, 60000],
    writeTimeoutMs: [100, 120000],
    traceTimeoutMs: [100, 60000],
  };
  for (const [key, [min, max]] of Object.entries(numericFields)) {
    if (file[key] !== undefined && (!Number.isInteger(Number(file[key])) || Number(file[key]) < min || Number(file[key]) > max)) {
      throw configError("INVALID_FIELD_RANGE", `Invalid oGMemory configuration field ${key}: expected an integer from ${min} to ${max}`);
    }
  }
  for (const key of ["enabled", "autoCapture", "autoRecall", "traceCaptureContent"]) {
    if (file[key] !== undefined && typeof file[key] !== "boolean") throw configError("INVALID_FIELD_TYPE", `Invalid oGMemory configuration field ${key}: expected boolean`);
  }
  const endpointInput = envString(env, "OG_MEMORY_URL", file.endpoint || DEFAULTS.endpoint).trim();
  let parsedEndpoint;
  try { parsedEndpoint = new URL(endpointInput); } catch { throw configError("INVALID_ENDPOINT_URL", "Invalid oGMemory configuration field endpoint: expected an HTTP(S) URL"); }
  if (!["http:", "https:"].includes(parsedEndpoint.protocol)) throw configError("INVALID_ENDPOINT_URL", "Invalid oGMemory configuration field endpoint: expected an HTTP(S) URL");
  if (parsedEndpoint.username || parsedEndpoint.password) throw configError("INVALID_ENDPOINT_POLICY", "Invalid oGMemory configuration field endpoint: URL credentials are not supported");
  if (endpointInput.includes("?")) throw configError("INVALID_ENDPOINT_POLICY", "Invalid oGMemory configuration field endpoint: query strings are not supported");
  if (endpointInput.includes("#")) throw configError("INVALID_ENDPOINT_POLICY", "Invalid oGMemory configuration field endpoint: URL fragments are not supported");
  const hostname = parsedEndpoint.hostname.replace(/^\[|\]$/g, "").toLowerCase();
  const loopback = hostname === "localhost" || hostname === "::1" || /^127(?:\.\d{1,3}){3}$/.test(hostname);
  if (parsedEndpoint.protocol === "http:" && !loopback) throw configError("INVALID_ENDPOINT_POLICY", "Invalid oGMemory configuration field endpoint: HTTPS is required for non-loopback hosts");
  const endpoint = parsedEndpoint.href.replace(/\/+$/, "");
  const config = {
    ...DEFAULTS,
    ...file,
    autoCapture: file.autoCapture ?? envBoolean(env, "OGMEM_AUTO_CAPTURE", DEFAULTS.autoCapture),
    autoRecall: file.autoRecall ?? envBoolean(env, "OGMEM_AUTO_RECALL", DEFAULTS.autoRecall),
    endpoint,
    apiKey: envString(env, "OG_AUTH_API_KEY", file.apiKey || DEFAULTS.apiKey),
    accountId: envString(env, "OG_AUTH_ACCOUNT_ID", file.accountId || DEFAULTS.accountId),
    userId: envString(env, "OG_AUTH_USER_ID", file.userId || DEFAULTS.userId),
    agentId: envString(env, "OG_AUTH_AGENT_ID", file.agentId || DEFAULTS.agentId),
    traceCaptureContent: envBoolean(env, "OGMEM_TRACE_CAPTURE_CONTENT", file.traceCaptureContent ?? DEFAULTS.traceCaptureContent),
  };
  for (const [key, [min, max]] of Object.entries(numericFields)) {
    config[key] = boundedInt(config[key], DEFAULTS[key], min, max);
  }
  return Object.freeze(config);
}

export { DEFAULTS };
