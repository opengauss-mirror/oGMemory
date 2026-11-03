// Trace rules are bundled from skills/trace_redaction_rules.json.
import { isIP } from "node:net";
import { readFileSync } from "node:fs";

const MAX_NODES = 2000;
const MAX_DEPTH = 16;
const MAX_STRING = 10000;
const REDACTION_OVERLAP = 4096;
function isSecretKey(key) {
  const segments = String(key).replace(/([a-z0-9])([A-Z])/g, "$1_$2").toLowerCase().split(/[^a-z0-9]+/).filter(Boolean);
  const joined = segments.join("");
  if (["authorization", "cookie"].some((suffix) => joined.endsWith(suffix))) return true;
  if (segments.some((segment, index) => ["authorization", "cookie"].includes(segment) && ["header", "value", "credential"].includes(segments[index + 1]))) return true;
  if (["credential", "credentials", "privatekey", "apikey", "password", "passwordhash", "passwd", "pwd", "secret", "token"].some((suffix) => joined.endsWith(suffix))) return true;
  if (segments.some((segment) => ["credential", "credentials", "password", "passwd", "pwd", "secret", "token"].includes(segment))) return true;
  return segments.some((segment, index) => segment === "api" && segments[index + 1] === "key");
}
function hasSensitiveJsonKey(text) {
  for (const match of text.matchAll(/("(?:\\.|[^"\\])*")\s*:/g)) {
    let key;
    try { key = JSON.parse(match[1]); }
    catch { key = match[1]; }
    if (isSecretKey(key)) return true;
  }
  return false;
}

function redactEmbeddedJson(text, depth, budget) {
  for (const match of text.matchAll(/'([^'\\\r\n]*)'\s*:/g)) {
    if (isSecretKey(match[1])) return "[REDACTED]";
  }
  const starts = /[\[{"]/g;
  let result = "";
  let cursor = 0;
  for (let match; (match = starts.exec(text));) {
    budget.remaining -= text.length;
    if (budget.remaining < 0) return "[REDACTED_SCAN_LIMIT]";
    const start = match.index;
    let parsedFragment = false;
    let nesting = 0;
    let quoted = false;
    let escaped = false;
    for (let end = start; end < text.length; end++) {
      const char = text[end];
      if (quoted) {
        if (escaped) escaped = false;
        else if (char === "\\") escaped = true;
        else if (char === '"') quoted = false;
        if (quoted || nesting !== 0) continue;
      } else {
        if (char === '"') { quoted = true; continue; }
        if (char === "{" || char === "[") nesting++;
        else if (char === "}" || char === "]") nesting--;
        if (nesting !== 0) continue;
      }
      try {
        const parsed = JSON.parse(text.slice(start, end + 1));
        if (typeof parsed === "string" && isSecretKey(parsed) && /^\s*:/.test(text.slice(end + 1))) {
          return "[REDACTED]";
        }
        result += text.slice(cursor, start) + JSON.stringify(redact(parsed, { depth: depth + 1, budget }));
        cursor = end + 1;
        starts.lastIndex = cursor;
        parsedFragment = true;
      } catch { /* Continue scanning text for the next JSON fragment. */ }
      break;
    }
    if (!parsedFragment && hasSensitiveJsonKey(text)) return "[REDACTED]";
  }
  return result + text.slice(cursor);
}

const RULES = JSON.parse(readFileSync(new URL("./trace_redaction_rules.json", import.meta.url), "utf8"))
  .map(({ pattern, flags, replacement, validator }) => [
    new RegExp(pattern, flags),
    validator === "ipv6" ? value => isIP(value) === 6 ? replacement : value : replacement,
  ]);

export function redactString(value, maxLength = MAX_STRING, jsonDepth = 0, budget = { remaining: 262144 }) {
  if (jsonDepth > MAX_DEPTH) return "[REDACTED_NESTED_JSON]";
  let output;
  try {
    const input = String(value);
    const oversized = input.length > maxLength + REDACTION_OVERLAP;
    const trimmed = (oversized ? input.slice(0, maxLength + REDACTION_OVERLAP) : input).trimStart();
    if (oversized && /[\[{"]/.test(trimmed)) return "[TRUNCATED_JSON]";
    if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
      try {
        const parsed = JSON.parse(input);
        const serialized = JSON.stringify(redact(parsed, { depth: jsonDepth, budget }));
        return serialized.slice(0, maxLength);
      }
      catch { /* Fall back to text redaction for non-JSON content. */ }
    }
    if (trimmed.startsWith('"')) {
      try {
        const decoded = JSON.parse(input);
        const nested = typeof decoded === "string" ? decoded.trimStart() : "";
        if (["{", "[", '"'].includes(nested[0])) {
          return JSON.stringify(redactString(decoded, maxLength, jsonDepth + 1, budget)).slice(0, maxLength);
        }
      }
      catch { /* Fall back to text redaction for non-JSON content. */ }
    }
    output = (input.length > maxLength ? input.slice(0, maxLength + REDACTION_OVERLAP) : input)
      .replace(/-----BEGIN [^-\r\n]*PRIVATE KEY-----[\s\S]*?(?:-----END [^-\r\n]*PRIVATE KEY-----|$)/g, "[REDACTED_PRIVATE_KEY]");
  }
  catch { return "[UNSERIALIZABLE]"; }
  const yamlMultiline = /^[ \t]*(?:-[ \t]+)?(?:"([^"\r\n]+)"|'([^'\r\n]+)'|([A-Za-z_][A-Za-z0-9_.-]*))[ \t]*:[ \t]*(?:[|>][1-9+-]{0,2})?[ \t]*(?:#[^\r\n]*)?\r?$/gm;
  for (const match of output.matchAll(yamlMultiline)) {
    if (isSecretKey(match[1] ?? match[2] ?? match[3])) return "[REDACTED_SECRET]";
  }
  output = redactEmbeddedJson(output, jsonDepth, budget);
  output = output.replace(/("(?:\\.|[^"\\])*")\s*:\s*("(?:\\.|[^"\\])*")/g, (field, key, encoded) => {
    let name;
    try { name = JSON.parse(key); }
    catch { name = key; }
    if (/^\[REDACTED(?:_[A-Z]+)?\](?:_\d+)?$/.test(name)) return field;
    if (isSecretKey(name)) return `${key}:"[REDACTED]"`;
    let original;
    try { original = JSON.parse(encoded); }
    catch { return field; }
    const cleaned = redactString(original, maxLength, jsonDepth + 1, budget);
    return cleaned === original ? field : `${key}:${JSON.stringify(cleaned)}`;
  });
  const assignments = /(?:--([A-Za-z_][A-Za-z0-9_.-]*)\s+|\b([A-Za-z_][A-Za-z0-9_.-]*)[ \t]*[:=][ \t]*)(?:(?:Bearer|Basic)\s+)?(?:"(?:\\[\s\S]|[^"\\])*(?:"|\\?$)|'(?:''|\\[\s\S]|[^'\\])*(?:'|\\?$)|\\[\s\S]?|[^\s,;'"\\])+/gi;
  let cursor = 0;
  let cleaned = "";
  for (let match; (match = assignments.exec(output));) {
    const key = match[1] ?? match[2];
    if (!isSecretKey(key)) {
      assignments.lastIndex = match.index + key.length + (match[1] ? 2 : 0);
      continue;
    }
    const replacement = match[1] ? `--${key} [REDACTED]` : `${key}=[REDACTED]`;
    cleaned += output.slice(cursor, match.index) + replacement;
    cursor = assignments.lastIndex;
  }
  output = cleaned + output.slice(cursor);
  for (const [pattern, replacement] of RULES) output = output.replace(pattern, replacement);
  output = output
    .replace(/\b[A-Z0-9._%+-]+@[A-Z0-9.-]*$/i, "[REDACTED_TRUNCATED]")
    .replace(/\b(?:sk-(?:proj-)?|gh[pousr]_|github_pat_|glpat-|xox[baprs]-)[A-Za-z0-9_-]*$/i, "[REDACTED_TRUNCATED]");
  return output.slice(0, maxLength);
}

export function redact(value, options = {}) {
  const budget = options.budget || { remaining: 262144 };
  const state = options.state || { nodes: 0, seen: new WeakSet() };
  const depth = options.depth || 0;
  state.nodes += 1;
  if (state.nodes > (options.maxNodes || MAX_NODES)) return "[NODE_LIMIT]";
  if (depth > (options.maxDepth || MAX_DEPTH)) return "[DEPTH_LIMIT]";
  if (typeof value === "string") return redactString(value, MAX_STRING, depth + 1, budget);
  if (typeof value === "number") {
    const text = String(value);
    const cleaned = redactString(text, MAX_STRING, depth + 1, budget);
    return cleaned === text ? value : cleaned;
  }
  if (typeof value === "bigint") return value.toString();
  if (["function", "symbol", "undefined"].includes(typeof value)) return undefined;
  if (Array.isArray(value)) {
    if (state.seen.has(value)) return "[REPEATED_REFERENCE]";
    state.seen.add(value);
    const result = [];
    for (const item of value) {
      if (state.nodes >= (options.maxNodes || MAX_NODES)) { result.push("[NODE_LIMIT]"); break; }
      result.push(redact(item, { ...options, state, budget, depth: depth + 1 }));
    }
    return result;
  }
  if (value && typeof value === "object") {
    if (state.seen.has(value)) return "[REPEATED_REFERENCE]";
    state.seen.add(value);
    const result = {};
    for (const key of Object.keys(value)) {
      if (state.nodes >= (options.maxNodes || MAX_NODES)) { result.__truncated__ = "[NODE_LIMIT]"; break; }
      const descriptor = Object.getOwnPropertyDescriptor(value, key);
      if (!descriptor) continue;
      const baseKey = redactString(key, MAX_STRING, depth + 1, budget);
      let safeKey = baseKey;
      for (let suffix = 2; Object.hasOwn(result, safeKey); suffix++) safeKey = `${baseKey}_${suffix}`;
      const item = descriptor.value;
      const redacted = !("value" in descriptor) ? "[GETTER]" :
        isSecretKey(key) ? "[REDACTED]" : redact(item, { ...options, state, budget, depth: depth + 1 });
      if (redacted !== undefined) Object.defineProperty(result, safeKey, {
        value: redacted, enumerable: true, configurable: true, writable: true,
      });
    }
    return result;
  }
  return value;
}
