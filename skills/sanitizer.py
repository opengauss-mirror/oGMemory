"""Deterministic trust boundary for trace-derived skill generation."""

from __future__ import annotations

import ast
import hashlib
import ipaddress
import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class UnsafeTraceError(ValueError):
    """Raised when a trace contains instruction-like or unverifiable input."""


_REDACTED = "[REDACTED_SECRET]"
_URI_USERINFO = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)([^/@\s]+)@")
_SECRET_KEY = re.compile(
    r"(?:api[_-]?key|authorization|cookie|credential(?:s)?|password|passwd|pwd|private[_-]?key|"
    r"secret(?:[_-]access[_-]key)?|token)",
    re.IGNORECASE,
)
_QUOTED_JSON_FIELD = re.compile(r'("(?:\\.|[^"\\])*")\s*:\s*("(?:\\.|[^"\\])*")')
_SECRET_VALUE = (
    r'''(?:"(?:\\[\s\S]|[^"\\])*(?:"|\\?$)|'''
    r"'(?:''|\\[\s\S]|[^'\\])*(?:'|\\?$)|"
    r'''\\[\s\S]?|[^\s,;'"\\])+'''
)
_YAML_MULTILINE_KEY = re.compile(
    r'''^[ \t]*(?:-[ \t]+)?(?:"([^"\r\n]+)"|'([^'\r\n]+)'|([A-Za-z_][A-Za-z0-9_.-]*))'''
    r"[ \t]*:[ \t]*(?:[|>][1-9+-]{0,2})?[ \t]*(?:#[^\r\n]*)?\r?$",
    re.MULTILINE,
)
_TRACE_ASSIGNMENT = re.compile(
    r"(?:--([A-Za-z_][A-Za-z0-9_.-]*)\s+|\b([A-Za-z_][A-Za-z0-9_.-]*)[ \t]*[:=][ \t]*)"
    r"(?:(?:Bearer|Basic)\s+)?" + _SECRET_VALUE,
    re.IGNORECASE,
)
_SECRET_PATTERNS = (
    # PEM payloads must be removed as a unit, not merely have their header hidden.
    re.compile(
        r"-----BEGIN(?: [A-Z0-9]+)* PRIVATE KEY-----.*?"
        r"(?:-----END(?: [A-Z0-9]+)* PRIVATE KEY-----|\Z)",
        re.DOTALL,
    ),
    # Provider credentials are often pasted without a label.
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\b(?:ghp_|github_pat_)[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    # JWTs are credentials even when no surrounding field identifies them as such.
    re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\b"),
    re.compile(r"(?i)\b(?:set-cookie|cookie)[ \t]*[:=][ \t]*[^\r\n]+"),
    re.compile(r"(?i)\bAuthorization\s*:\s*(?:Bearer|Basic)\s+[^\s,'\";]+"),
    re.compile(
        r"(?i)\b(?:[a-z0-9]+[_-])*(?:api[_-]?key|credential(?:s)?|password|passwd|pwd|private[_-]?key|"
        r"secret(?:[_-]access[_-]key)?|token|authorization|cookie)"
        r"\s*[:=]\s*(?:(?:bearer|basic)\s+)?" + _SECRET_VALUE
    ),
    re.compile(
        rf"(?i)--{_SECRET_KEY.pattern}"
        r"(?:=|\s+)" + _SECRET_VALUE
    ),
)
_TRACE_RULES = tuple(
    (
        re.compile(rule["pattern"], re.ASCII | (re.I if "i" in rule["flags"] else 0)),
        rule["replacement"].replace("$1", r"\g<1>"),
        rule.get("validator"),
    )
    for rule in json.loads(Path(__file__).with_name("trace_redaction_rules.json").read_text(encoding="utf-8"))
)
_ABSOLUTE_PATHS = (
    re.compile(r"(?<![\w])(?:[A-Za-z]:\\|[A-Za-z]:/)(?:[^\s<>:'\"|?*]+[\\/])*[^\s<>:'\"|?*]*"),
    # UNC paths, including administrative shares. Require both a server and
    # share so ordinary escaped text is not treated as a path.
    re.compile(r"(?<![\w\\])\\\\[^\s\\/'\"<>]+\\[^\s\\/'\"<>]+(?:\\[^\s'\"<>]+)*"),
    # Home-relative paths are machine/user specific even without a username.
    re.compile(r"(?<![\w])~[\\/][^\s'\"<>]+"),
    # General POSIX absolute paths. The lookbehind and (?!/) deliberately
    # exclude the // in URL schemes and protocol-relative URLs.
    re.compile(r"(?<![\w:/])/(?!/)(?:[^\s/'\"<>]+/)*[^\s/'\"<>]+"),
)
_INJECTION = re.compile(
    r"(?i)(ignore|disregard|override|forget)\s+"
    r"(?:(?:all|the|your|these|those|above|previous|prior|system|developer)\s+){0,3}instructions|"
    r"do\s+not\s+follow\s+"
    r"(?:(?:the|these|those|above|previous|prior|system|developer)\s+){0,3}instructions|"
    r"reveal\s+(the\s+)?(system prompt|credentials|secrets)|"
    r"you\s+are\s+now\s+(?:a|an)|<\s*(?:system|developer)\s*>"
)
_SESSION_VALUE = re.compile(r"\b(?:session|trace|request)[-_ ]?id\s*[:=]\s*[A-Za-z0-9._:-]+", re.I)
_FAILED_OUTCOME = re.compile(
    r"(?i)(?:\b(?:i|we)\s+(?:could(?:n't| not)|can(?:'t| not)|was unable|were unable|failed)\b|"
    r"\b(?:task|operation|request|work)\s+(?:is|was|remains)?\s*(?:failed|incomplete|unfinished)\b|"
    r"\b(?:did not|didn't|have not|haven't)\s+(?:finish|complete|succeed)\b|"
    r"\b(?:timed out|unhandled exception|fatal error)\b)"
)
_MESSAGE_FIELDS = {
    "user_message": ("prompt", "message", "content"),
    "assistant_message": ("assistant_message", "message", "content"),
    "turn_end": ("assistant_message", "message", "content"),
}
_IDENTIFIER_BODY = (
    r"[A-Za-z](?:[A-Za-z0-9_-]*[A-Za-z0-9])?"
    r"(?:\.[A-Za-z](?:[A-Za-z0-9_-]*[A-Za-z0-9])?)*"
)
_IDENTIFIER = (
    r"(?=[A-Za-z0-9_.-]{1,64}(?![A-Za-z0-9_.-]))"
    + _IDENTIFIER_BODY
)
_STRONG_TOOL_IDENTIFIER = (
    r"(?=[A-Za-z0-9_.-]{1,64}(?![A-Za-z0-9_.-]))"
    r"(?=[A-Za-z0-9_.-]*[_.][A-Za-z0-9_-])"
    + _IDENTIFIER_BODY
)
_STRUCTURED_TOOL_PATTERNS = (
    re.compile(rf"(?i)[`'\"]?(?:tool|tool_name)[`'\"]?\s*[:=]\s*[`'\"]?({_IDENTIFIER})"),
    re.compile(rf"(?i)\b(?:run|call|invoke|execute)\s+(?:the\s+)?tool\s+[`'\"]?({_IDENTIFIER})"),
    re.compile(rf"(?i)\b(?:run|call|invoke|execute)\s+[`'\"]?({_STRONG_TOOL_IDENTIFIER})"),
    re.compile(rf"(?i)\buse\s+(?:the\s+)?tool\s+[`'\"]?({_IDENTIFIER})"),
    re.compile(rf"(?i)[`'\"]({_IDENTIFIER})[`'\"]\s+tool\b"),
    # A bare "Use" is only strong enough for snake-case or namespaced names.
    # Kebab-case words commonly occur as ordinary modifiers ("read-only mode").
    re.compile(rf"(?i)\buse\s+[`'\"]?({_STRONG_TOOL_IDENTIFIER})"),
    re.compile(
        r"(?i)\b(?:use|run|call|invoke|execute)\s+(?:the\s+)?"
        r"(bash|shell|python(?:3(?:\.\d+)?)?|powershell|pwsh|cmd)\b"
    ),
)
_EXECUTION_GATE_TOOL_LIST = re.compile(
    r"(?i)Execution gate:\s*begin by invoking an observed execution tool\s*\(([^)\r\n]+)\)"
)
_INTERNAL_SKILL_MARKER = re.compile(
    r"(?i)(?:ogmemory|ogmemory://|ctx://|"
    r"(?<![A-Za-z0-9_])(?:skill_candidate|candidate_id|session_id|trace_id|review_score|"
    r"publication_status|artifact_version|review_policy_version|generation[_-]?ir|"
    r"tool[_-]?(?:use|call)[_-]?id|call[_-]?id)(?![A-Za-z0-9_])|"
    r"(?<![A-Za-z0-9_])(?:session|trace|request|candidate)[ _-]+(?:id|identifier)"
    r"(?![A-Za-z0-9_]))"
)
_FENCED_BLOCK = re.compile(
    r"(?ms)^[ ]{0,3}(?P<fence>```|~~~)(?P<info>[^\r\n]*)\r?\n"
    r"(?P<body>.*?)[ ]{0,3}(?P=fence)[ \t]*$"
)
_INLINE_CODE = re.compile(r"`[^`\r\n]*`")
_UNSAFE_EXECUTABLE_CODE = re.compile(
    r"(?ix)(?:"
    r"\b(?:eval|exec|__import__)\s*\(|"
    r"\bpython(?:3(?:\.\d+)?)?\s+-(?:c|m)\b|"
    r"\b(?:invoke-expression|invoke-webrequest|start-process)\b|\biex\s*[ (]|"
    r"\b(?:bash|sh|zsh|powershell|pwsh)\s+-[A-Za-z]*c\b|\bcmd(?:\.exe)?\s+/c\b|"
    r"\b(?:import|from)\s+(?:subprocess|socket|requests|httpx|aiohttp|urllib3?|"
    r"ftplib|paramiko|smtplib|imaplib|poplib|telnetlib)\b|"
    r"\b(?:subprocess|socket|requests|httpx|aiohttp|urllib3?|ftplib|paramiko|"
    r"smtplib|imaplib|poplib|telnetlib)\s*\.|"
    r"\bos\.(?:system|popen|remove|unlink|rmdir)\s*\(|"
    r"\bshutil\.rmtree\s*\(|"
    r"\.(?:unlink|rmdir)\s*\(|"
    r"(?:^|[^A-Za-z0-9_-])(?:rm|rmdir|remove-item|curl|wget)\s+"
    r")"
)
_UNSAFE_PROCEDURE_COMMAND = re.compile(
    r"(?ix)(?:"
    r"\b(?:eval|exec|__import__)\s*\(|"
    r"\bpython(?:3(?:\.\d+)?)?(?:\.exe)?\s+-(?:c|m)\b|"
    r"\bpy(?:\.exe)?(?:\s+-3(?:\.\d+)?)?\s+-(?:c|m)\b|"
    r"\b(?:invoke-expression|invoke-webrequest|start-process)\b|\biex\s*[ (]|"
    r"\b(?:powershell|pwsh)(?:\.exe)?\b[^\r\n]{0,200}\s+-(?:c[a-z]*|e[a-z]*)\b|"
    r"\b(?:bash|sh|zsh)\s+-[A-Za-z]*c\b|"
    r"\bcmd(?:\.exe)?\s+(?:/c\b|\[LOCAL_PATH\])|"
    r"\bos\.(?:system|popen|remove|unlink|rmdir)\s*\(|\bshutil\.rmtree\s*\(|"
    r"(?:^|[^A-Za-z0-9_-])(?:rm|rmdir|del|remove-item)\s+|"
    r"\b(?:curl|wget)\s+(?:https?://|ftp://|[-$'\"`])"
    r")"
)
_PYTHON_TEMPLATE_LANGUAGES = frozenset({"py", "python", "python3"})
_SAFE_PYTHON_IMPORTS = frozenset({
    "argparse", "base64", "binascii", "collections", "csv", "datetime", "decimal",
    "fractions", "functools", "hashlib", "heapq", "io", "itertools", "json", "math",
    "numpy", "openpyxl", "pandas", "pathlib", "re", "statistics", "sys", "typing", "xml",
})
_UNSAFE_PYTHON_CALLS = frozenset({
    "__import__", "delattr", "eval", "exec", "getattr", "globals", "locals",
    "setattr", "vars",
})
_UNSAFE_PYTHON_ATTRIBUTES = frozenset({
    "eval", "popen", "query", "read_html", "read_pickle", "rmdir", "rmtree",
    "run", "startfile", "system", "to_pickle", "unlink",
})
_PATH_CONSTRUCTOR_CALLS = frozenset({"Path", "PurePath", "open"})
_PATH_BEARING_METHODS = frozenset({
    "joinpath", "load_workbook", "open", "parse", "read_csv", "read_excel", "read_json",
    "read_parquet", "save", "savetxt", "to_csv", "to_excel", "to_json", "to_parquet",
})
_PATH_RECEIVER_METHODS = frozenset({
    "exists", "is_dir", "is_file", "open", "read_bytes", "read_text",
    "write_bytes", "write_text",
})
_PATH_ARGUMENT_KEYWORDS = frozenset({
    "file", "filename", "fname", "filepath_or_buffer", "io", "path", "path_or_buf",
})
_PATH_BEARING_MODULE_CALLS = frozenset({
    "numpy.load", "numpy.save", "numpy.savetxt", "openpyxl.load_workbook",
    "pandas.read_csv", "pandas.read_excel", "pandas.read_json", "pandas.read_parquet",
    "xml.etree.ElementTree.parse",
})
_SAFE_MODULE_API_REGISTRY = {
    "argparse": frozenset({"ArgumentParser"}),
    "base64": frozenset({"b64decode", "b64encode", "standard_b64decode", "standard_b64encode"}),
    "binascii": frozenset({"a2b_base64", "b2a_base64", "crc32", "hexlify", "unhexlify"}),
    "collections": frozenset({"Counter", "OrderedDict", "defaultdict", "deque"}),
    "csv": frozenset({"DictReader", "DictWriter", "reader", "writer"}),
    "datetime": frozenset({"date", "datetime", "time", "timedelta"}),
    "decimal": frozenset({"Decimal"}),
    "fractions": frozenset({"Fraction"}),
    "functools": frozenset({"reduce"}),
    "hashlib": frozenset({"blake2b", "blake2s", "md5", "sha1", "sha224", "sha256", "sha384", "sha512"}),
    "heapq": frozenset({"heapify", "heappop", "heappush", "heapreplace", "nlargest", "nsmallest"}),
    "io": frozenset({"BytesIO", "StringIO"}),
    "itertools": frozenset({
        "accumulate", "chain", "combinations", "combinations_with_replacement", "compress",
        "dropwhile", "filterfalse", "groupby", "islice", "pairwise", "permutations", "product",
        "starmap", "takewhile", "tee", "zip_longest",
    }),
    "json": frozenset({"dump", "dumps", "load", "loads"}),
    "numpy": frozenset({"array", "asarray", "load", "ones", "save", "savetxt", "zeros"}),
    "openpyxl": frozenset({"Workbook", "load_workbook"}),
    "openpyxl.styles": frozenset({
        "Alignment", "Border", "Font", "NamedStyle", "PatternFill", "Protection", "Side",
    }),
    "openpyxl.utils": frozenset({
        "column_index_from_string", "get_column_letter", "range_boundaries",
    }),
    "pandas": frozenset({
        "DataFrame", "Series", "concat", "merge", "read_csv", "read_excel", "read_json",
        "read_parquet",
    }),
    "pathlib": frozenset({"Path", "PurePath"}),
    "re": frozenset({"compile", "findall", "fullmatch", "match", "search", "sub"}),
    "xml.etree.ElementTree": frozenset({
        "Element", "SubElement", "fromstring", "parse", "tostring",
    }),
}
_SAFE_MODULE_CALLS = frozenset(
    f"{module}.{name}"
    for module, names in _SAFE_MODULE_API_REGISTRY.items()
    for name in names
)
_SAFE_MODULE_PREFIXES = ("math.", "statistics.")
_SAFE_OBJECT_METHODS = frozenset({
    "add", "add_argument", "append", "astype", "cell", "clear", "compile", "copy",
    "decode", "drop", "drop_duplicates", "dropna", "encode", "endswith", "exists",
    "extend", "fillna", "find", "findall", "format", "get", "group", "groupby",
    "getroot", "hexdigest", "insert", "is_dir", "is_file", "items", "iter_rows", "join", "keys",
    "lower", "lstrip", "match", "parse_args", "pop", "read", "read_bytes", "read_text",
    "remove", "replace", "resolve", "save", "search", "set", "setdefault", "sort", "sort_values",
    "split", "startswith", "strip", "update", "upper", "values", "write", "write_bytes",
    "to_csv", "to_excel", "to_json", "to_parquet", "with_suffix", "write_text",
    "writeheader", "writerow", "writerows",
})
_SAFE_PYTHON_BUILTIN_CALLS = frozenset({
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int",
    "isinstance", "len", "list", "map", "max", "min", "next", "print", "range",
    "open", "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip",
})


@dataclass(frozen=True)
class SanitizedTrace:
    events: list[dict[str, Any]]
    tools: frozenset[str]
    failed_tools: frozenset[str] = frozenset()
    failed: bool = False


def sanitize_text(value: str, *, reject_instructions: bool = False) -> str:
    """Remove secrets and machine/session-specific values from text."""
    if reject_instructions and _INJECTION.search(value):
        raise UnsafeTraceError("trace contains prompt-injection-like instructions")
    result = _URI_USERINFO.sub(r"\1[REDACTED_SECRET]@", value)
    for pattern in _SECRET_PATTERNS:
        result = pattern.sub(_REDACTED, result)
    for pattern in _ABSOLUTE_PATHS:
        result = pattern.sub("[LOCAL_PATH]", result)
    result = _SESSION_VALUE.sub("session_id=[SESSION]", result)
    return result


def sanitize_trace_text(value: str) -> str:
    """Redact structured JSON and quoted credentials embedded in trace text."""
    if len(value) > 65_536:
        return _REDACTED
    scan_budget = 262_144

    def redact_ipv6(match):
        try:
            ipaddress.IPv6Address(match[0])
        except ValueError:
            return match[0]
        return "[REDACTED_IP]"

    def redact_text(text):
        if any(_SECRET_KEY.search(next(group for group in match.groups() if group is not None))
               for match in _YAML_MULTILINE_KEY.finditer(text)):
            return _REDACTED
        result = sanitize_text(text)
        parts = []
        cursor = position = 0
        while match := _TRACE_ASSIGNMENT.search(result, position):
            key = match[1] or match[2]
            if _SECRET_KEY.search(key):
                parts.extend((result[cursor:match.start()], _REDACTED))
                cursor = position = match.end()
            else:
                position = match.start() + len(key) + (2 if match[1] else 0)
        parts.append(result[cursor:])
        result = "".join(parts)
        for pattern, replacement, validator in _TRACE_RULES:
            result = pattern.sub(redact_ipv6 if validator == "ipv6" else replacement, result)
        return result

    def has_sensitive_json_key(text):
        for match in re.finditer(r'("(?:\\.|[^"\\])*")\s*:', text):
            try:
                key = json.loads(match[1])
            except ValueError:
                key = match[1]
            if _SECRET_KEY.search(key):
                return True
        return False

    def redact_embedded(text, depth=0):
        nonlocal scan_budget
        for match in re.finditer(r"'([^'\\\r\n]*)'\s*:", text):
            if _SECRET_KEY.search(match[1]):
                return _REDACTED
        decoder = json.JSONDecoder()
        parts = []
        cursor = 0
        for match in re.finditer(r'[\[{"]', text):
            if match.start() < cursor:
                continue
            scan_budget -= len(text)
            if scan_budget < 0:
                return _REDACTED
            try:
                parsed, end = decoder.raw_decode(text, match.start())
            except ValueError:
                if has_sensitive_json_key(text):
                    return _REDACTED
                continue
            except RecursionError:
                return _REDACTED
            if isinstance(parsed, str) and _SECRET_KEY.search(parsed) and text[end:].lstrip().startswith(":"):
                return _REDACTED
            parts.extend((text[cursor:match.start()], json.dumps(redact(parsed, depth + 1), ensure_ascii=False)))
            cursor = end
        parts.append(text[cursor:])
        return "".join(parts)

    def redact(value, depth=0):
        if depth >= 32:
            return _REDACTED
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                base_key = redact_text(key)
                safe_key = base_key
                suffix = 2
                while safe_key in result:
                    safe_key = f"{base_key}_{suffix}"
                    suffix += 1
                result[safe_key] = _REDACTED if _SECRET_KEY.search(key) else redact(item, depth + 1)
            return result
        if isinstance(value, list):
            return [redact(item, depth + 1) for item in value]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            text = str(value)
            cleaned = redact_text(text)
            return value if cleaned == text else cleaned
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except RecursionError:
                return _REDACTED
            except ValueError:
                return redact_text(_QUOTED_JSON_FIELD.sub(
                    lambda match: redact_field(match, depth + 1), redact_embedded(value, depth + 1),
                ))
            if not isinstance(parsed, (dict, list, str)):
                return redact_text(value)
            return json.dumps(redact(parsed, depth + 1), ensure_ascii=False)
        return value

    def redact_field(match, depth=0):
        key = match[1]
        try:
            name = json.loads(key)
        except ValueError:
            name = key
        if re.fullmatch(r"\[REDACTED(?:_[A-Z]+)?\](?:_\d+)?", name):
            return match[0]
        if _SECRET_KEY.search(name):
            return f'{key}: "{_REDACTED}"'
        try:
            original = json.loads(match[2])
        except ValueError:
            return match[0]
        cleaned = redact(original, depth + 1)
        return match[0] if cleaned == original else f'{key}: {json.dumps(cleaned, ensure_ascii=False)}'

    try:
        parsed = json.loads(value)
    except RecursionError:
        return _REDACTED
    except ValueError:
        value = redact_embedded(value)
    else:
        value = json.dumps(redact(parsed), ensure_ascii=False)
    return redact_text(_QUOTED_JSON_FIELD.sub(redact_field, value))


def sanitize_trace(events: list[dict[str, Any]], *, allow_failed: bool = False) -> SanitizedTrace:
    """Sanitize a JSON trace and verify messages and tool outcomes are complete."""
    if not isinstance(events, list) or not events:
        raise UnsafeTraceError("trace is empty")
    validator = StreamingTraceSanitizer(allow_failed=allow_failed, retain_events=True)
    for raw in events:
        validator.feed(raw)
    return validator.finish()


class StreamingTraceSanitizer:
    """Validate and sanitize ordered trace pages without retaining raw history."""

    POLICY_VERSION = 1
    MAX_ACTIVE_TOOL_CALLS = 4096
    MAX_CHECKPOINT_BYTES = MAX_ACTIVE_TOOL_CALLS * 256

    def __init__(
        self, *, allow_failed: bool = False, retain_events: bool = False,
        checkpoint: dict[str, Any] | None = None,
    ) -> None:
        self.allow_failed = allow_failed
        self.retain_events = retain_events
        self.cleaned: list[dict[str, Any]] = []
        self.started: dict[str, int] = {}
        self.completed: dict[str, int] = {}
        self.failed_calls: dict[str, int] = {}
        self.unresolved_failures: dict[str, int] = {}
        self.completed_tools: set[str] = set()
        self.failed_tools: set[str] = set()
        self.failed_outcome = False
        self.has_user = False
        self.has_assistant = False
        self.events_seen = 0
        self.active_tool_calls = 0
        if checkpoint:
            self._restore(checkpoint)

    @staticmethod
    def _counter(value: Any) -> dict[str, int]:
        if not isinstance(value, dict):
            return {}
        return {
            str(key): int(count) for key, count in value.items()
            if isinstance(count, int) and count >= 0
        }

    def _restore(self, checkpoint: dict[str, Any]) -> None:
        if checkpoint.get("policy_version") != self.POLICY_VERSION:
            raise UnsafeTraceError("trace sanitizer checkpoint policy mismatch")
        self.started = self._counter(checkpoint.get("started"))
        self.completed = self._counter(checkpoint.get("completed"))
        self.failed_calls = self._counter(checkpoint.get("failed_calls"))
        self.unresolved_failures = self._counter(checkpoint.get("unresolved_failures"))
        self.completed_tools = {
            str(value) for value in checkpoint.get("completed_tools", [])
            if isinstance(value, str)
        }
        self.failed_tools = {
            str(value) for value in checkpoint.get("failed_tools", [])
            if isinstance(value, str)
        }
        self.failed_outcome = bool(checkpoint.get("failed_outcome"))
        self.has_user = bool(checkpoint.get("has_user"))
        self.has_assistant = bool(checkpoint.get("has_assistant"))
        self.events_seen = max(0, int(checkpoint.get("events_seen", 0)))
        self.active_tool_calls = max(
            0,
            sum(self.started.values())
            - sum(self.completed.values())
            - sum(self.failed_calls.values()),
        )
        if self.active_tool_calls > self.MAX_ACTIVE_TOOL_CALLS:
            raise UnsafeTraceError("trace has too many simultaneous unresolved tool calls")

    @staticmethod
    def _call_key(tool_name: str, call_id: str) -> str:
        identity = (
            "id:" + hashlib.sha256(call_id.encode("utf-8")).hexdigest()[:24]
            if call_id else "unidentified"
        )
        return f"{tool_name}|{identity}"

    def feed(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Consume one raw event and return its sanitized representation."""
        if not isinstance(raw, dict):
            raise UnsafeTraceError("trace contains a malformed event")
        event_type = raw.get("event_type")
        if not isinstance(event_type, str) or not event_type.strip():
            raise UnsafeTraceError("trace event has no event_type")
        event_type = event_type.strip()
        if event_type in _MESSAGE_FIELDS:
            message = _message_text(raw, _MESSAGE_FIELDS[event_type])
            reasoning = _message_text(raw, ("reasoning",)) if event_type == "assistant_message" else ""
            if not message and not reasoning:
                raise UnsafeTraceError(f"{event_type} has no meaningful content")
            if event_type == "user_message":
                self.has_user = True
            elif message:
                if _FAILED_OUTCOME.search(message) and not self.allow_failed:
                    raise UnsafeTraceError("trace contains an explicit failed or incomplete outcome")
                self.failed_outcome = self.failed_outcome or bool(_FAILED_OUTCOME.search(message))
                self.has_assistant = True
        if event_type == "tool_failure" and not self.allow_failed:
            raise UnsafeTraceError("trace contains a failed tool call")
        tool_name_value = raw.get("tool_name")
        tool_name = tool_name_value.strip() if isinstance(tool_name_value, str) else ""
        if event_type in {"tool_start", "tool_success", "tool_failure"} and not tool_name:
            raise UnsafeTraceError(f"{event_type} has no tool_name")
        if event_type in {"tool_start", "tool_success", "tool_failure"} and re.fullmatch(_IDENTIFIER, tool_name) is None:
            raise UnsafeTraceError(f"{event_type} has an invalid tool_name")
        call_id = next(
            (
                raw.get(field).strip()
                for field in ("tool_use_id", "tool_call_id", "call_id")
                if isinstance(raw.get(field), str) and raw.get(field).strip()
            ),
            "",
        )
        call_key = self._call_key(tool_name, call_id)
        if event_type == "tool_start":
            if self.active_tool_calls >= self.MAX_ACTIVE_TOOL_CALLS:
                raise UnsafeTraceError("trace has too many simultaneous unresolved tool calls")
            self.started[call_key] = self.started.get(call_key, 0) + 1
            self.active_tool_calls += 1
        if event_type == "tool_success":
            self.completed[call_key] = self.completed.get(call_key, 0) + 1
            self.completed_tools.add(tool_name)
            if (
                self.completed[call_key] + self.failed_calls.get(call_key, 0)
                > self.started.get(call_key, 0)
            ):
                raise UnsafeTraceError(f"tool success has no observed start: {tool_name}")
            self.active_tool_calls -= 1
            if self.unresolved_failures.get(call_key, 0):
                self.unresolved_failures[call_key] -= 1
        if event_type == "tool_failure":
            self.failed_calls[call_key] = self.failed_calls.get(call_key, 0) + 1
            self.unresolved_failures[call_key] = self.unresolved_failures.get(call_key, 0) + 1
            self.failed_tools.add(tool_name)
            if (
                self.failed_calls[call_key] + self.completed.get(call_key, 0)
                > self.started.get(call_key, 0)
            ):
                raise UnsafeTraceError(f"tool failure has no observed start: {tool_name}")
            self.active_tool_calls -= 1
        if event_type in {"tool_success", "tool_failure"}:
            observed_outcomes = (
                self.completed.get(call_key, 0) + self.failed_calls.get(call_key, 0)
            )
            if (
                observed_outcomes == self.started.get(call_key, 0)
                and not self.unresolved_failures.get(call_key, 0)
            ):
                self.started.pop(call_key, None)
                self.completed.pop(call_key, None)
                self.failed_calls.pop(call_key, None)
                self.unresolved_failures.pop(call_key, None)
        cleaned = _sanitize_value(raw)
        self.events_seen += 1
        if self.retain_events:
            self.cleaned.append(cleaned)
        return cleaned

    def checkpoint(self) -> dict[str, Any]:
        """Return bounded validation state without raw correlation identifiers."""
        checkpoint = {
            "policy_version": self.POLICY_VERSION,
            "started": dict(sorted(self.started.items())),
            "completed": dict(sorted(self.completed.items())),
            "failed_calls": dict(sorted(self.failed_calls.items())),
            "unresolved_failures": dict(sorted(self.unresolved_failures.items())),
            "completed_tools": sorted(self.completed_tools),
            "failed_tools": sorted(self.failed_tools),
            "failed_outcome": self.failed_outcome,
            "has_user": self.has_user,
            "has_assistant": self.has_assistant,
            "events_seen": self.events_seen,
        }
        checkpoint_bytes = len(
            json.dumps(checkpoint, ensure_ascii=False, sort_keys=True).encode("utf-8")
        )
        if checkpoint_bytes > self.MAX_CHECKPOINT_BYTES:
            raise UnsafeTraceError(
                "trace sanitizer checkpoint exceeds persistent byte budget "
                f"({checkpoint_bytes} > {self.MAX_CHECKPOINT_BYTES})"
            )
        return checkpoint

    def finish(self) -> SanitizedTrace:
        """Validate end-of-trace invariants and return the accumulated summary."""
        if not self.events_seen:
            raise UnsafeTraceError("trace is empty")
        if not self.has_user or not self.has_assistant:
            raise UnsafeTraceError("trace must contain user and assistant messages")
        for call_key, count in self.started.items():
            if self.completed.get(call_key, 0) + self.failed_calls.get(call_key, 0) < count:
                tool_name = call_key.partition("|")[0]
                noun = "outcome" if self.allow_failed else "success"
                raise UnsafeTraceError(f"tool call has no observed {noun}: {tool_name}")
        failed = self.failed_outcome or any(self.unresolved_failures.values())
        if failed and not self.allow_failed:
            raise UnsafeTraceError("trace contains a failed outcome")
        return SanitizedTrace(
            list(self.cleaned), frozenset(self.completed_tools),
            frozenset(self.failed_tools), failed,
        )


def validate_skill_markdown(content: str, observed_tools: set[str] | frozenset[str]) -> str:
    """Apply the final publication boundary and reject invented agent tools."""
    sanitized = sanitize_text(content, reject_instructions=True)
    reject_internal_skill_metadata(sanitized)
    if len(sanitized.splitlines()) > 500:
        raise UnsafeTraceError("SKILL.md exceeds 500 lines")
    if len(sanitized.split()) > 5000:
        raise UnsafeTraceError("SKILL.md exceeds 5000 words")
    allowed = {tool.casefold() for tool in observed_tools}
    for token in _referenced_tools(sanitized):
        if token.casefold() not in allowed:
            raise UnsafeTraceError(f"skill references an unobserved tool: {token}")
    for match in _FENCED_BLOCK.finditer(sanitized):
        info = match.group("info").strip()
        language = info.split(maxsplit=1)[0] if info else None
        code = match.group("body")
        if language is None and code.strip():
            raise UnsafeTraceError("skill contains an unlabeled executable block")
        validate_implementation_code(code, language=language)
        for token in re.findall(r"\b([A-Za-z][A-Za-z0-9_.-]{2,})\s*\(", code):
            normalized = token.casefold().replace(".", "_").replace("-", "_")
            if (
                normalized.startswith(("ogm_", "mcp_"))
                or "destruct" in normalized
            ) and normalized not in allowed:
                raise UnsafeTraceError(f"skill references an unobserved tool: {token}")
    for match in re.finditer(
        r"(?m)^Invocation pattern(?: \([^\n)]*\))?:\s*(.+)$", sanitized,
    ):
        invocation = match.group(1)
        runtime_names = {
            _normalized_identifier(name)
            for group in re.findall(
                r"<([^<>\r\n]+)>|\$\{([A-Za-z_]\w*)\}|\{([A-Za-z_]\w*)\}",
                invocation,
            )
            for name in group if name
            if _normalized_identifier(name) not in {
                "temporary_script", "temporary_script_created_from_template",
            }
        }
        validate_python_invocation(invocation, allowed_runtime_names=runtime_names)
    prose = _FENCED_BLOCK.sub("", sanitized)
    if _UNSAFE_PROCEDURE_COMMAND.search(prose):
        raise UnsafeTraceError("skill contains an unsafe command instruction")
    for line in sanitized.splitlines():
        if " Tool: `" in line:
            validate_implementation_code(line)
    in_procedure = False
    for line in sanitized.splitlines():
        if line == "## Procedure":
            in_procedure = True
            continue
        if line.startswith("## "):
            in_procedure = False
        elif in_procedure and line.strip():
            if _UNSAFE_PROCEDURE_COMMAND.search(line):
                raise UnsafeTraceError("skill procedure contains an unsafe command")
    return sanitized


def validate_implementation_code(
    body: str,
    *,
    language: str | None = None,
    allowed_runtime_names: frozenset[str] | set[str] | None = None,
) -> str:
    """Reject executable primitives that are unsafe in portable generated skills."""
    if _UNSAFE_EXECUTABLE_CODE.search(body):
        raise UnsafeTraceError("skill implementation template contains unsafe executable code")
    if language:
        normalized_language = language.casefold()
        if normalized_language not in _PYTHON_TEMPLATE_LANGUAGES:
            raise UnsafeTraceError("skill implementation template language is not supported")
        _validate_python_template(body, allowed_runtime_names=allowed_runtime_names)
    return body


def validate_python_invocation(
    invocation: str, *, allowed_runtime_names: frozenset[str] | set[str],
) -> str:
    """Accept one parameterized Python command, never a shell command list."""
    if not invocation.strip() or any(character in invocation for character in "\r\n"):
        raise UnsafeTraceError("skill implementation invocation is invalid")
    placeholders: list[str] = []
    placeholder_names: dict[str, str] = {}

    def protect(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2) or match.group(3)
        placeholders.append(_normalized_identifier(name))
        marker = f"__OGM_PARAMETER_{len(placeholders)}__"
        placeholder_names[marker] = placeholders[-1]
        return marker

    protected = re.sub(
        r"<([^<>\r\n]+)>|\$\{([A-Za-z_]\w*)\}|\{([A-Za-z_]\w*)\}",
        protect,
        invocation,
    )
    if re.search(r"\|\||[&;|`<>]|\$\(|\$\{|\{[^{}]*\}", protected):
        raise UnsafeTraceError("skill implementation invocation contains shell syntax")
    try:
        tokens = shlex.split(protected, posix=True)
    except ValueError as exc:
        raise UnsafeTraceError("skill implementation invocation is invalid") from exc
    if not tokens:
        raise UnsafeTraceError("skill implementation invocation must use Python")
    if re.fullmatch(r"python(?:3(?:\.\d+)*)?(?:\.exe)?", tokens[0], re.IGNORECASE):
        script_index = 1
    elif tokens[0].casefold() in {"py", "py.exe"}:
        script_index = 2 if len(tokens) > 1 and re.fullmatch(r"-3(?:\.\d+)?", tokens[1]) else 1
    else:
        raise UnsafeTraceError("skill implementation invocation must use Python")
    if len(tokens) <= script_index or tokens[script_index] in {"-c", "-m"} or not (
        tokens[script_index].endswith(".py")
        or tokens[script_index].startswith("__OGM_PARAMETER_")
    ):
        raise UnsafeTraceError("skill implementation invocation must name one script template")
    script_parameter = placeholder_names.get(tokens[script_index])
    if script_parameter is not None and script_parameter not in {
        "temporary_script", "temporary_script_created_from_template",
    }:
        raise UnsafeTraceError(
            "skill implementation invocation cannot execute a runtime input as code"
        )
    if any(
        name not in {"temporary_script", "temporary_script_created_from_template"}
        and name not in allowed_runtime_names
        for name in placeholders
    ):
        raise UnsafeTraceError("skill implementation invocation binds an undeclared input")
    arguments = tokens[script_index + 1:]
    index = 0
    while index < len(arguments):
        token = arguments[index]
        if token.startswith("--"):
            option, separator, inline_value = token.partition("=")
            option_name = _normalized_identifier(option.removeprefix("--"))
            if separator:
                if placeholder_names.get(inline_value) != option_name:
                    raise UnsafeTraceError(
                        "skill implementation invocation contains a fixed option value"
                    )
                index += 1
                continue
            if index + 1 < len(arguments) and not arguments[index + 1].startswith("--"):
                value = arguments[index + 1]
                if placeholder_names.get(value) != option_name:
                    raise UnsafeTraceError(
                        "skill implementation invocation contains a fixed option value"
                    )
                index += 2
                continue
            index += 1
            continue
        if token not in placeholder_names:
            raise UnsafeTraceError(
                "skill implementation invocation contains a fixed positional value"
            )
        index += 1
    missing_inputs = set(allowed_runtime_names) - set(placeholders)
    if missing_inputs:
        raise UnsafeTraceError("skill implementation invocation does not bind input")
    return invocation


def _validate_python_template(
    body: str, *, allowed_runtime_names: frozenset[str] | set[str] | None = None,
) -> None:
    """Structurally validate the supported executable-template language."""
    try:
        tree = ast.parse(body)
        compile(tree, "<generated-skill>", "exec")
    except SyntaxError as exc:
        raise UnsafeTraceError("skill implementation template is invalid Python") from exc

    imported_names: dict[str, str] = {}
    module_aliases: dict[str, str] = {}
    local_functions = {
        node.name for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assignments: dict[str, ast.AST] = {}
    function_nodes = {
        node.name: node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    call_graph = {
        name: {
            descendant.func.id
            for descendant in ast.walk(function)
            if isinstance(descendant, ast.Call)
            and isinstance(descendant.func, ast.Name)
            and descendant.func.id in function_nodes
        }
        for name, function in function_nodes.items()
    }
    if _graph_has_cycle(call_graph):
        raise UnsafeTraceError("skill implementation template contains recursive calls")
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _UNSAFE_PYTHON_CALLS:
            raise UnsafeTraceError("skill implementation template references dynamic execution")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise UnsafeTraceError("skill implementation template uses a private attribute")
        if isinstance(node, ast.While):
            raise UnsafeTraceError("skill implementation template contains an unbounded loop")
        if isinstance(node, ast.BinOp):
            size = _static_int_value(node)
            if size is not None and abs(size) > 1_000_000:
                raise UnsafeTraceError("skill implementation template contains an oversized expression")
            if isinstance(node.op, ast.Mult):
                multiplier = _static_int_value(node.right) or _static_int_value(node.left)
                sequence = node.left if _static_int_value(node.right) is not None else node.right
                if (
                    multiplier is not None
                    and multiplier > 1_000_000
                    and isinstance(sequence, (ast.List, ast.Tuple, ast.Set, ast.Constant))
                ):
                    raise UnsafeTraceError("skill implementation template contains an oversized allocation")
        if isinstance(node, ast.Import):
            modules = [alias.name.split(".", 1)[0] for alias in node.names]
            if any(module not in _SAFE_PYTHON_IMPORTS for module in modules):
                raise UnsafeTraceError("skill implementation template imports an unsafe module")
            for alias, module in zip(node.names, modules, strict=True):
                module_aliases[alias.asname or module] = alias.name if alias.asname else module
        elif isinstance(node, ast.ImportFrom):
            full_module = node.module or ""
            module = full_module.split(".", 1)[0]
            if module not in _SAFE_PYTHON_IMPORTS:
                raise UnsafeTraceError("skill implementation template imports an unsafe module")
            for alias in node.names:
                if alias.name.casefold() in _UNSAFE_PYTHON_ATTRIBUTES:
                    raise UnsafeTraceError("skill implementation template imports an unsafe operation")
                imported_names[alias.asname or alias.name] = f"{full_module}.{alias.name}"
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    if target.id in assignments:
                        assignments[target.id] = ast.Constant(
                            value="[UNTRUSTED_REASSIGNED_VALUE]",
                        )
                    else:
                        assignments[target.id] = node.value
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            assignments[node.target.id] = ast.Constant(value="[UNTRUSTED_AUGMENTED_VALUE]")
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if isinstance(item.optional_vars, ast.Name):
                    if item.optional_vars.id in assignments:
                        assignments[item.optional_vars.id] = ast.Constant(
                            value="[UNTRUSTED_REASSIGNED_VALUE]",
                        )
                    else:
                        assignments[item.optional_vars.id] = item.context_expr
        elif isinstance(node, ast.ExceptHandler) and node.name:
            assignments[node.name] = ast.Constant(value="[UNTRUSTED_EXCEPTION_VALUE]")

    direct_store_nodes = {
        id(target)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name)
    }
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Store)
            and id(node) not in direct_store_nodes
        ):
            assignments[node.id] = ast.Constant(value="[UNTRUSTED_BOUND_VALUE]")
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            assignments[node.name] = ast.Constant(value="[UNTRUSTED_BOUND_VALUE]")
        elif isinstance(node, ast.MatchMapping) and node.rest:
            assignments[node.rest] = ast.Constant(value="[UNTRUSTED_BOUND_VALUE]")

    allowed_name_calls = (
        set(_SAFE_PYTHON_BUILTIN_CALLS)
        | set(imported_names)
        | set(local_functions)
    )
    if set(assignments) & allowed_name_calls:
        raise UnsafeTraceError("skill implementation template rebinds a callable name")
    function_argument_safety: dict[str, list[bool]] = {}
    function_argument_names = {
        _normalized_identifier(argument.arg)
        for function in function_nodes.values()
        for argument in function.args.args
    }
    for name, function in function_nodes.items():
        call_sites = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == name
        ]
        for position, argument in enumerate(function.args.args):
            values = []
            for call in call_sites:
                if position < len(call.args):
                    values.append(call.args[position])
                    continue
                keyword = next(
                    (item.value for item in call.keywords if item.arg == argument.arg), None,
                )
                if keyword is None:
                    values = []
                    break
                values.append(keyword)
            safe = bool(values) and all(
                _is_task_bound_path_expression(
                    value, assignments,
                    allowed_runtime_names=allowed_runtime_names,
                )
                for value in values
            )
            function_argument_safety.setdefault(
                _normalized_identifier(argument.arg), [],
            ).append(safe)
    trusted_function_arguments = {
        name for name, safety in function_argument_safety.items()
        if safety and all(safety)
    }
    path_runtime_names = (
        None
        if allowed_runtime_names is None
        else set(allowed_runtime_names) | trusted_function_arguments
    )
    path_constructor_names = {
        name for name, origin in imported_names.items()
        if origin in {"pathlib.Path", "pathlib.PurePath"}
    }
    untrusted_function_arguments = function_argument_names - trusted_function_arguments
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if not isinstance(node.func, (ast.Name, ast.Attribute)):
                raise UnsafeTraceError("skill implementation template uses an indirect call")
            if isinstance(node.func, ast.Name) and node.func.id not in allowed_name_calls:
                raise UnsafeTraceError("skill implementation template calls an unresolved function")
            if isinstance(node.func, ast.Name) and node.func.id in _UNSAFE_PYTHON_CALLS:
                raise UnsafeTraceError("skill implementation template uses dynamic execution")
            imported_origin = (
                imported_names.get(node.func.id) if isinstance(node.func, ast.Name) else None
            )
            canonical_call = _canonical_module_call(
                node.func, module_aliases=module_aliases, imported_names=imported_names,
            )
            if canonical_call and not (
                canonical_call in _SAFE_MODULE_CALLS
                or canonical_call.startswith(_SAFE_MODULE_PREFIXES)
            ):
                raise UnsafeTraceError("skill implementation template calls an unsupported module API")
            if (
                isinstance(node.func, ast.Attribute)
                and canonical_call is None
                and node.func.attr not in _SAFE_OBJECT_METHODS
            ):
                raise UnsafeTraceError("skill implementation template calls an unsupported object API")
            if imported_origin and imported_origin.rsplit(".", 1)[-1].casefold() in _UNSAFE_PYTHON_ATTRIBUTES:
                raise UnsafeTraceError("skill implementation template uses an unsafe operation")
            path_call = (
                isinstance(node.func, ast.Name)
                and node.func.id in _PATH_CONSTRUCTOR_CALLS
                or imported_origin in {"pathlib.Path", "pathlib.PurePath"}
                or isinstance(node.func, ast.Attribute)
                and node.func.attr in {"Path", "PurePath"}
            )
            path_bearing_call = path_call or canonical_call in _PATH_BEARING_MODULE_CALLS or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr.casefold() in _PATH_BEARING_METHODS
            ) or (
                imported_origin is not None
                and imported_origin.rsplit(".", 1)[-1].casefold() in _PATH_BEARING_METHODS
            ) or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "write"
                and _is_xml_tree_expression(
                    node.func.value, assignments,
                    module_aliases=module_aliases, imported_names=imported_names,
                )
            )
            path_receiver_call = (
                isinstance(node.func, ast.Attribute)
                and node.func.attr.casefold() in _PATH_RECEIVER_METHODS
                and _is_path_object_expression(
                    node.func.value, assignments,
                    path_constructor_names=path_constructor_names,
                )
            )
            if path_receiver_call and (
                _expression_references_names(
                    node.func.value, untrusted_function_arguments,
                )
                or not _is_task_bound_path_expression(
                    node.func.value, assignments, allowed_runtime_names=path_runtime_names,
                    path_constructor_names=path_constructor_names,
                )
            ):
                raise UnsafeTraceError("skill implementation template contains a fixed path")
            if (
                path_bearing_call
                and node.args
                and (
                    _expression_references_names(
                        node.args[0], untrusted_function_arguments,
                    )
                    or not _is_task_bound_path_expression(
                        node.args[0], assignments, allowed_runtime_names=path_runtime_names,
                    )
                )
            ):
                raise UnsafeTraceError("skill implementation template contains a fixed path")
            if path_bearing_call and any(
                keyword.arg in _PATH_ARGUMENT_KEYWORDS
                and (
                    _expression_references_names(
                        keyword.value, untrusted_function_arguments,
                    )
                    or not _is_task_bound_path_expression(
                        keyword.value, assignments, allowed_runtime_names=path_runtime_names,
                    )
                )
                for keyword in node.keywords
            ):
                raise UnsafeTraceError("skill implementation template contains a fixed path")
            if path_bearing_call and any(keyword.arg is None for keyword in node.keywords):
                raise UnsafeTraceError("skill implementation template expands unverified path options")
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr.casefold() in _UNSAFE_PYTHON_ATTRIBUTES
            ):
                raise UnsafeTraceError("skill implementation template uses an unsafe operation")
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in {"replace", "rename", "symlink_to", "hardlink_to"}
                and _is_path_object_expression(
                    node.func.value, assignments,
                    path_constructor_names=path_constructor_names,
                )
            ):
                raise UnsafeTraceError("skill implementation template mutates a path outside the task contract")
            numpy_load = (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "load"
                and isinstance(node.func.value, ast.Name)
                and module_aliases.get(node.func.value.id) == "numpy"
                or imported_origin == "numpy.load"
            )
            allow_pickle_values = [
                keyword.value for keyword in node.keywords if keyword.arg == "allow_pickle"
            ]
            if numpy_load and any(keyword.arg is None for keyword in node.keywords):
                raise UnsafeTraceError("skill implementation template expands unsafe load options")
            if numpy_load and len(node.args) >= 3:
                allow_pickle_values.append(node.args[2])
            if numpy_load and any(
                not (
                    isinstance(value, ast.Constant)
                    and type(value.value) is bool
                    and value.value is False
                )
                for value in allow_pickle_values
            ):
                raise UnsafeTraceError("skill implementation template enables unsafe deserialization")
            if isinstance(node.func, ast.Name) and node.func.id == "range" and any(
                (size := _static_int_value(argument)) is not None and abs(size) > 1_000_000
                for argument in node.args
            ):
                raise UnsafeTraceError("skill implementation template contains an oversized range")


def _canonical_module_call(
    function: ast.AST, *, module_aliases: dict[str, str], imported_names: dict[str, str],
) -> str | None:
    parts: list[str] = []
    current = function
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    root = module_aliases.get(current.id) or imported_names.get(current.id)
    if not root:
        return None
    return ".".join([root, *reversed(parts)])


def _graph_has_cycle(graph: dict[str, set[str]]) -> bool:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        if any(visit(neighbor) for neighbor in graph.get(node, set())):
            return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(node) for node in graph)


def _static_int_value(value: ast.AST) -> int | None:
    if isinstance(value, ast.Constant) and type(value.value) is int:
        return value.value
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "int"
        and len(value.args) == 1
        and not value.keywords
        and isinstance(value.args[0], ast.Constant)
        and isinstance(value.args[0].value, str)
        and re.fullmatch(r"[+-]?\d+", value.args[0].value.strip())
    ):
        digits = value.args[0].value.strip()
        return 1_000_001 if len(digits.lstrip("+-")) > 18 else int(digits)
    if not isinstance(value, ast.BinOp):
        return None
    left = _static_int_value(value.left)
    right = _static_int_value(value.right)
    if left is None or right is None:
        return None
    try:
        if isinstance(value.op, ast.Add):
            return left + right
        if isinstance(value.op, ast.Sub):
            return left - right
        if isinstance(value.op, ast.Mult):
            return left * right
        if isinstance(value.op, ast.FloorDiv):
            return left // right
        if isinstance(value.op, ast.Mod):
            return left % right
        if isinstance(value.op, ast.LShift):
            return 1_000_001 if right < 0 or right > 20 else left << right
        if isinstance(value.op, ast.RShift):
            return left >> right
        if isinstance(value.op, ast.Pow):
            if abs(right) > 20:
                return 1_000_001
            return left ** right
    except (ArithmeticError, OverflowError):
        return 1_000_001
    return None


def _is_task_bound_path_expression(
    value: ast.AST,
    assignments: dict[str, ast.AST],
    seen: frozenset[str] = frozenset(),
    *,
    allowed_runtime_names: frozenset[str] | set[str] | None = None,
    path_constructor_names: frozenset[str] | set[str] = frozenset(),
) -> bool:
    """Accept only path expressions derived from runtime names, never fixed strings."""
    if isinstance(value, ast.Name):
        if value.id not in assignments:
            return (
                allowed_runtime_names is None
                or _normalized_identifier(value.id) in allowed_runtime_names
            )
        if value.id in seen:
            return False
        if (
            _is_sys_argv_subscript(assignments[value.id])
            and (
                allowed_runtime_names is None
                or _normalized_identifier(value.id) in allowed_runtime_names
            )
        ):
            return True
        return _is_task_bound_path_expression(
            assignments[value.id], assignments, seen | {value.id},
            allowed_runtime_names=allowed_runtime_names,
            path_constructor_names=path_constructor_names,
        )
    if isinstance(value, ast.Attribute):
        if (
            isinstance(value.value, ast.Name)
            and value.value.id in assignments
            and _is_parse_args_call(assignments[value.value.id])
        ):
            return (
                allowed_runtime_names is None
                or _normalized_identifier(value.attr) in allowed_runtime_names
            )
        return _is_task_bound_path_expression(
            value.value, assignments, seen, allowed_runtime_names=allowed_runtime_names,
            path_constructor_names=path_constructor_names,
        )
    if isinstance(value, ast.Call):
        if _is_parse_args_call(value):
            return True
        if _call_name(value.func) in {
            "Path", "PurePath", "open",
        } | set(path_constructor_names) and value.args:
            return _is_task_bound_path_expression(
                value.args[0], assignments, seen, allowed_runtime_names=allowed_runtime_names,
                path_constructor_names=path_constructor_names,
            )
        if isinstance(value.func, ast.Attribute) and value.func.attr in {"resolve", "with_suffix"}:
            if value.func.attr == "with_suffix" and (
                len(value.args) != 1 or not _is_safe_path_suffix(value.args[0])
            ):
                return False
            return _is_task_bound_path_expression(
                value.func.value, assignments, seen,
                allowed_runtime_names=allowed_runtime_names,
                path_constructor_names=path_constructor_names,
            )
    if isinstance(value, ast.BinOp) and isinstance(value.op, (ast.Add, ast.Div)):
        if isinstance(value.op, ast.Add) and _is_safe_path_suffix(value.right):
            return _is_task_bound_path_expression(
                value.left, assignments, seen, allowed_runtime_names=allowed_runtime_names,
                path_constructor_names=path_constructor_names,
            )
        return (
            _is_task_bound_path_expression(
                value.left, assignments, seen, allowed_runtime_names=allowed_runtime_names,
                path_constructor_names=path_constructor_names,
            )
            and _is_task_bound_path_expression(
                value.right, assignments, seen, allowed_runtime_names=allowed_runtime_names,
                path_constructor_names=path_constructor_names,
            )
        )
    return False


def _normalized_identifier(value: str) -> str:
    return re.sub(r"\W+", "_", value).strip("_").casefold()


def _expression_references_names(value: ast.AST, names: set[str]) -> bool:
    return any(
        isinstance(node, ast.Name) and _normalized_identifier(node.id) in names
        for node in ast.walk(value)
    )


def _call_name(function: ast.AST) -> str | None:
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return None


def _is_parse_args_call(value: ast.AST) -> bool:
    return isinstance(value, ast.Call) and _call_name(value.func) == "parse_args"


def _is_sys_argv_subscript(value: ast.AST) -> bool:
    return (
        isinstance(value, ast.Subscript)
        and isinstance(value.value, ast.Attribute)
        and isinstance(value.value.value, ast.Name)
        and value.value.value.id == "sys"
        and value.value.attr == "argv"
    )


def _is_path_object_expression(
    value: ast.AST,
    assignments: dict[str, ast.AST],
    seen: frozenset[str] = frozenset(),
    *,
    path_constructor_names: set[str] | frozenset[str] = frozenset(),
) -> bool:
    if isinstance(value, ast.Call):
        if _call_name(value.func) in {"Path", "PurePath"} | set(path_constructor_names):
            return True
        return (
            isinstance(value.func, ast.Attribute)
            and value.func.attr in {"resolve", "with_suffix"}
            and _is_path_object_expression(
                value.func.value, assignments, seen,
                path_constructor_names=path_constructor_names,
            )
        )
    if isinstance(value, ast.Name) and value.id in assignments:
        if value.id in seen:
            return False
        return _is_path_object_expression(
            assignments[value.id], assignments, seen | {value.id},
            path_constructor_names=path_constructor_names,
        )
    if isinstance(value, ast.Attribute):
        return _is_path_object_expression(
            value.value, assignments, seen,
            path_constructor_names=path_constructor_names,
        )
    if isinstance(value, ast.Subscript):
        return _is_path_object_expression(
            value.value, assignments, seen,
            path_constructor_names=path_constructor_names,
        )
    if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Div):
        return _is_path_object_expression(
            value.left, assignments, seen,
            path_constructor_names=path_constructor_names,
        )
    return False


def _is_safe_path_suffix(value: ast.AST) -> bool:
    return (
        isinstance(value, ast.Constant)
        and isinstance(value.value, str)
        and value.value.startswith(".")
        and len(value.value) <= 32
        and not any(separator in value.value for separator in ("/", "\\"))
    )


def _is_xml_tree_expression(
    value: ast.AST,
    assignments: dict[str, ast.AST],
    seen: frozenset[str] = frozenset(),
    *,
    module_aliases: dict[str, str],
    imported_names: dict[str, str],
) -> bool:
    if isinstance(value, ast.Name) and value.id in assignments:
        if value.id in seen:
            return False
        return _is_xml_tree_expression(
            assignments[value.id], assignments, seen | {value.id},
            module_aliases=module_aliases, imported_names=imported_names,
        )
    return (
        isinstance(value, ast.Call)
        and _canonical_module_call(
            value.func, module_aliases=module_aliases, imported_names=imported_names,
        ) == "xml.etree.ElementTree.parse"
    )


def reject_internal_skill_metadata(content: str) -> str:
    """Reject product-governance details that cannot appear in a portable skill."""
    if _INTERNAL_SKILL_MARKER.search(content):
        raise UnsafeTraceError("skill contains internal memory-system metadata")
    return content


def referenced_tools(content: str) -> frozenset[str]:
    """Return deterministic tool references recognized by the publication boundary."""
    return frozenset(_referenced_tools(content))


def prompt_payload(events: list[dict[str, Any]]) -> str:
    """Serialize untrusted evidence as JSON data, never as prompt instructions."""
    return json.dumps(without_correlation_ids(events), ensure_ascii=False, sort_keys=True)


def without_correlation_ids(value: Any) -> Any:
    """Keep transport correlation IDs local to deterministic trace processing."""
    if isinstance(value, list):
        return [without_correlation_ids(item) for item in value]
    if isinstance(value, dict):
        return {
            key: without_correlation_ids(item)
            for key, item in value.items()
            if str(key).casefold() not in {"tool_use_id", "tool_call_id", "call_id"}
        }
    return value


def _message_text(event: dict[str, Any], fields: tuple[str, ...]) -> str:
    for field in fields:
        value = event.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _referenced_tools(content: str) -> set[str]:
    references: set[str] = set()
    explicit_references: set[str] = set()
    prose = _FENCED_BLOCK.sub("", content)
    for match in _EXECUTION_GATE_TOOL_LIST.finditer(prose):
        references.update(
            token.strip()
            for token in match.group(1).split(",")
            if re.fullmatch(_IDENTIFIER, token.strip())
        )
    for index, pattern in enumerate(_STRUCTURED_TOOL_PATTERNS):
        found = {match.group(1) for match in pattern.finditer(prose)}
        references.update(found)
        if index in {0, 1, 3, 4}:
            explicit_references.update(found)
    references.difference_update({"after", "before", "for", "on", "to", "when", "with"})

    # Call syntax is executable. A bare code span is not: it may name a field,
    # option, or data value and requires explicit tool wording to be governed.
    inline_calls = {
        match.group(1)
        for span in _INLINE_CODE.findall(prose)
        for match in re.finditer(rf"\b({_IDENTIFIER})\s*\(", span)
    }
    references.difference_update(inline_calls - explicit_references)
    call_prose = _INLINE_CODE.sub("", prose)
    for match in re.finditer(rf"\b({_IDENTIFIER})\s*\(", call_prose):
        token = match.group(1)
        if "_" in token or "-" in token or "." in token:
            references.add(token)
    return references


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, str):
        return sanitize_text(value, reject_instructions=True)
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            secret_key = _SECRET_KEY.search(str(key)) is not None
            clean_key = _sanitize_key(key)
            if clean_key in cleaned:
                suffix = 2
                candidate = f"{clean_key}_{suffix}"
                while candidate in cleaned:
                    suffix += 1
                    candidate = f"{clean_key}_{suffix}"
                clean_key = candidate
            cleaned[clean_key] = _REDACTED if secret_key else _sanitize_value(item)
        return cleaned
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return sanitize_text(str(value), reject_instructions=True)


def _sanitize_key(key: Any) -> str:
    value = str(key)
    if _SECRET_KEY.search(value):
        return _REDACTED
    return sanitize_text(value, reject_instructions=True)
