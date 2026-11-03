"""Deterministic, bounded reducers for Read and Bash tool output."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

_CODE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".go",
    ".h",
    ".hpp",
    ".java",
    ".js",
    ".jsx",
    ".kt",
    ".m",
    ".php",
    ".py",
    ".pyi",
    ".pyx",
    ".pxd",
    ".rb",
    ".rs",
    ".scala",
    ".sh",
    ".sql",
    ".swift",
    ".ts",
    ".tsx",
}
_CODE_DECLARATION_RE = re.compile(
    r"^\s*(?:async\s+def|def|class|function|func|interface|struct|enum|"
    r"type|module|namespace|export\s+(?:default\s+)?(?:class|function|const|let|var)|"
    r"public\s+(?:class|interface|static|void)|private\s+(?:class|static|void))\b"
)
_CODE_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+\S+\s+import\b|import\s+\S+|package\s+\S+|"
    r"using\s+\S+|#include\s+[<\"]|require\(|use\s+\S+)"
)
_TRACEBACK_RE = re.compile(
    r"Traceback \(most recent call last\)|"
    r"During handling of the above exception|"
    r"The above exception was the direct cause|"
    r"ExceptionGroup|\b(?:[A-Za-z_][\w.]*)(?:Error|Exception):"
)
_TRACEBACK_HEADER_RE = re.compile(r"Traceback \(most recent call last\)|ExceptionGroup")
_CAUSE_TRANSITION_RE = re.compile(
    r"During handling of the above exception|The above exception was the direct cause"
)
_PYTHON_FRAME_RE = re.compile(r"^\s*File\s+['\"]")
_TEST_RE = re.compile(
    r"=+\s+(?:FAILURES|ERRORS)\s+=+|"
    r"^\s*FAIL(?:\s|$)|"
    r"^\s*npm ERR!(?:\s|$)|"
    r"^\s*[●✕✖]\s+\S|"
    r"\b(?:FAILED|ERROR)\b|"
    r"Traceback \(most recent call last\)|"
    r"AssertionError|"
    r"short test summary info|"
    r"\b(?:expected|actual)\s*[:=]|"
    r"\b\d+\s+(?:failed|passed|error)s?\b",
    re.IGNORECASE,
)
_TEST_LOG_DETECTION_RE = re.compile(
    r"=+\s+(?:FAILURES|ERRORS)\s+=+|"
    r"^\s*FAIL(?:\s|$)|"
    r"^\s*npm ERR!(?:\s|$)|"
    r"^\s*[●✕✖]\s+\S|"
    r"^\s*FAILED\s+\S|"
    r"short test summary info|"
    r"^\s*\d+\s+(?:failed|passed|errors?)\b|"
    r"(?:Test Suites|Tests):.*\b\d+\s+(?:failed|passed|errors?)\b",
    re.IGNORECASE | re.MULTILINE,
)
_TEST_BLOCK_HEADER_RE = re.compile(
    r"^\s*(?:"
    r"_{3,}\s+\S(?:.*\S)?\s+_{3,}|"
    r"FAIL(?:\s+\S.*)?|"
    r"[●✕✖]\s+\S.*"
    r")\s*$",
    re.IGNORECASE,
)
_PYTEST_SECTION_HEADER_RE = re.compile(r"^\s*=+\s+(?:FAILURES|ERRORS)\s+=+\s*$", re.IGNORECASE)
_TEST_SUMMARY_RE = re.compile(
    r"short test summary info|\b\d+\s+(?:failed|passed|errors?)\b", re.IGNORECASE
)
_PYTEST_DIAGNOSTIC_RE = re.compile(
    r"Traceback \(most recent call last\)|"
    r"AssertionError|"
    r"\b(?:expected|actual)\s*[:=]|"
    r"^\s*(?:FAILED|ERROR)\b",
    re.IGNORECASE,
)
_PYTEST_TERMINAL_DIAGNOSTIC_RE = re.compile(r"^\s*(?:FAILED|ERROR)\b", re.IGNORECASE)
_DIFF_RE = re.compile(r"^diff --git |^@@\s+-\d+(?:,\d+)?\s+\+\d+(?:,\d+)?\s+@@", re.MULTILINE)
_PRIORITY_LOG_RE = re.compile(
    r"\b(?:error|exception|fatal|failed|failure|warning|warn|diagnostic|summary)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ReductionOutcome:
    """One normalized, auditable reduction result before delivery notices."""

    text: str
    detected: str
    strategy: str
    original_chars: int
    original_lines: int
    filtered_chars: int
    filtered_lines: int
    omitted_lines: int
    preserved_blocks: int
    fallback: str
    retained_line_ranges: tuple[tuple[int, int], ...]
    omitted_chars: int = 0
    output_max_chars: int | None = None

    @property
    def was_shortened(self) -> bool:
        """Whether the rendered payload omits source material."""

        return (
            self.omitted_lines > 0
            or self.omitted_chars > 0
            or self.filtered_chars < self.original_chars
        )

    def reduction_metadata(self) -> dict[str, Any]:
        return {
            "omitted_lines": self.omitted_lines,
            "preserved_blocks": self.preserved_blocks,
            "fallback": self.fallback,
            "omitted_chars": self.omitted_chars,
            "char_cap_applied": self.output_max_chars is not None,
            "output_max_chars": self.output_max_chars,
            "retained_line_ranges": [
                {"start": start, "end": end} for start, end in self.retained_line_ranges
            ],
        }


def shorten_head_tail(text: str, *, max_lines: int, label: str) -> str:
    """Return the historical head/tail reduction used as a conservative fallback."""

    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    if max_lines <= 1:
        return lines[0] if lines else ""
    # The omission marker itself is a rendered line. Reserve one line for it
    # so legacy fallbacks obey the same configured delivery budget as adaptive
    # reducers instead of leaking one extra line at every shortening boundary.
    source_budget = max_lines - 1
    head_count = max(1, source_budget // 2)
    tail_count = max(0, source_budget - head_count)
    omitted = len(lines) - head_count - tail_count
    rendered = [*lines[:head_count], f"... [{label}: {omitted} lines omitted] ..."]
    if tail_count:
        rendered.extend(lines[-tail_count:])
    return "\n".join(rendered)


def detect_content(text: str, *, relative_path: str = "") -> str:
    """Classify only high-confidence content shapes in a deterministic order."""

    # A source Read may contain diagnostic text without being a log.
    if Path(relative_path).suffix.lower() in _CODE_SUFFIXES:
        return "code"
    if _TRACEBACK_RE.search(text):
        return "python_error"
    if _DIFF_RE.search(text):
        return "diff"
    if _TEST_LOG_DETECTION_RE.search(text):
        return "test_log"

    lines = text.splitlines()
    suffix = Path(relative_path).suffix.lower()
    declaration_count = sum(bool(_CODE_DECLARATION_RE.match(line)) for line in lines)
    if suffix in _CODE_SUFFIXES or declaration_count >= 2:
        return "code"

    non_empty = [line for line in lines if line.strip()]
    if len(non_empty) >= 4:
        repeated = len(non_empty) - len(set(non_empty))
        timestamp_like = sum(
            bool(re.match(r"^\d{4}-\d{2}-\d{2}[ T]|^\[[A-Z]+\]", line)) for line in non_empty
        )
        if repeated or timestamp_like >= max(2, len(non_empty) // 4):
            return "log"
    return "generic"


def reduce_read_output(text: str, *, relative_path: str, config: Any) -> ReductionOutcome:
    """Reduce a Read payload while preserving a legacy fallback for uncertain inputs."""

    # Preserve authorized source Reads verbatim for editing.
    # Range-limited Reads already bypass filtering; Bash/log output does not.
    if Path(relative_path).suffix.lower() in _CODE_SUFFIXES:
        return passthrough_reduction(text, detected="code", strategy="read_source_passthrough")
    max_lines = _positive_config(config, "tool_output_filter_read_max_lines", 60)
    adaptive_enabled = bool(getattr(config, "tool_output_filter_adaptive_enabled", False))
    if not adaptive_enabled:
        outcome = _legacy_head_tail(
            text,
            max_lines=max_lines,
            label="file content folded",
            detected="generic",
            strategy="read_head_tail",
        )
    else:
        detected = detect_content(text, relative_path=relative_path)
        if detected == "code":
            outcome = _reduce_code(text, max_lines=max_lines)
        elif detected == "test_log":
            outcome = _reduce_test_log(
                text,
                max_lines=_positive_config(config, "tool_output_filter_test_max_lines", 120),
                strategy="read_test_log_adaptive",
            )
        elif detected == "diff":
            outcome = _reduce_diff(
                text,
                max_lines=_positive_config(config, "tool_output_filter_diff_max_lines", 100),
                strategy="read_diff_adaptive",
            )
        elif detected == "python_error":
            outcome = _reduce_python_error(
                text,
                max_lines=_positive_config(config, "tool_output_filter_error_max_lines", 100),
                strategy="read_python_error_adaptive",
            )
        elif detected == "log":
            outcome = _reduce_log(
                text,
                max_lines=_positive_config(config, "tool_output_filter_generic_max_lines", 80),
            )
        else:
            outcome = _legacy_head_tail(
                text,
                max_lines=max_lines,
                label="file content folded",
                detected="generic",
                strategy="read_generic_head_tail",
            )
    return _cap_configured_output(outcome, config)


def legacy_read_reduction(text: str, config: Any) -> ReductionOutcome:
    """Return the pre-adaptive Read behavior for an explicit safe fallback."""

    return _cap_configured_output(
        _legacy_head_tail(
            text,
            max_lines=_positive_config(config, "tool_output_filter_read_max_lines", 60),
            label="file content folded",
            detected="generic",
            strategy="read_head_tail",
        ),
        config,
    )


def passthrough_reduction(text: str, *, detected: str, strategy: str) -> ReductionOutcome:
    """Represent an intentionally unshortened payload with normalized metadata."""

    lines = text.splitlines()
    return _outcome(
        original=text,
        text=text,
        detected=detected,
        strategy=strategy,
        retained_ranges=((1, len(lines)),) if lines else (),
        fallback="passthrough",
    )


def cap_reduction_output(outcome: ReductionOutcome, *, max_chars: int) -> ReductionOutcome:
    """Bound rendered text without changing its source-selection metadata.

    The reducers normally budget source lines.  A single generated/log/error line can
    nevertheless be much larger than the line budget, so this final rendering guard
    bounds the actual payload while recording the extra character-level omission.
    """

    if max_chars <= 0 or len(outcome.text) <= max_chars:
        return outcome

    rendered, omitted_chars = _cap_rendered_text(
        outcome.text,
        max_chars=max_chars,
        preserve_terminal_prefix=outcome.detected == "python_error",
    )
    return replace(
        outcome,
        text=rendered,
        filtered_chars=len(rendered),
        filtered_lines=len(rendered.splitlines()),
        omitted_chars=outcome.omitted_chars + omitted_chars,
        output_max_chars=max_chars,
    )


def reduce_bash_output(
    command: str,
    output: str,
    category: str,
    config: Any,
) -> ReductionOutcome:
    """Reduce a classified Bash result without changing the classifier contract."""

    del command
    if not bool(getattr(config, "tool_output_filter_adaptive_enabled", False)):
        return legacy_bash_reduction(output, category, config)

    if category == "test_runner":
        outcome = _reduce_test_log(
            output,
            max_lines=_positive_config(config, "tool_output_filter_test_max_lines", 120),
            strategy="bash_test_runner_adaptive",
        )
    elif category == "diff":
        outcome = _reduce_diff(
            output,
            max_lines=_positive_config(config, "tool_output_filter_diff_max_lines", 100),
            strategy="bash_diff_adaptive",
        )
    elif category == "python_error":
        outcome = _reduce_python_error(
            output,
            max_lines=_positive_config(config, "tool_output_filter_error_max_lines", 100),
            strategy="bash_python_error_adaptive",
        )
    else:
        outcome = _legacy_head_tail(
            output,
            max_lines=_positive_config(config, "tool_output_filter_generic_max_lines", 80),
            label="output folded",
            detected="generic",
            strategy="bash_head_tail",
        )
    return _cap_configured_output(outcome, config)


def legacy_bash_reduction(output: str, category: str, config: Any) -> ReductionOutcome:
    """Represent the pre-existing Bash reducers as normalized outcomes."""

    if category == "test_runner":
        kept: list[tuple[int, str]] = []
        patterns = (
            r"=+\s+FAILURES\s+=+",
            r"=+\s+ERRORS\s+=+",
            r"FAILED\s+",
            r"ERROR\s+",
            r"Traceback \(most recent call last\)",
            r"AssertionError",
            r"\b\d+\s+failed\b",
            r"\b\d+\s+passed\b",
            r"short test summary info",
        )
        for index, line in enumerate(output.splitlines(), start=1):
            if any(re.search(pattern, line) for pattern in patterns):
                kept.append((index, line))
        if kept:
            selected = kept[: _positive_config(config, "tool_output_filter_test_max_lines", 120)]
            text = "\n".join(line for _, line in selected)
            retained_ranges = tuple(
                _normalize_ranges(
                    [(index, index) for index, _ in selected],
                    len(output.splitlines()),
                )
            )
            return _cap_configured_output(
                _outcome(
                    original=output,
                    text=text,
                    detected="test_log",
                    strategy="bash_test_runner",
                    retained_ranges=retained_ranges,
                    fallback="legacy",
                ),
                config,
            )
        return _cap_configured_output(
            _legacy_head_tail(
                output,
                max_lines=_positive_config(config, "tool_output_filter_generic_max_lines", 80),
                label="test output folded",
                detected="test_log",
                strategy="bash_test_runner_head_tail",
            ),
            config,
        )
    if category == "diff":
        return _cap_configured_output(
            _legacy_head_tail(
                output,
                max_lines=_positive_config(config, "tool_output_filter_diff_max_lines", 100),
                label="diff folded",
                detected="diff",
                strategy="bash_diff_head_tail",
            ),
            config,
        )
    if category == "python_error":
        lines = output.splitlines()
        traceback_start = 0
        for index, line in enumerate(lines):
            if "Traceback (most recent call last)" in line:
                traceback_start = index
        text = "\n".join(
            lines[traceback_start:][
                -_positive_config(config, "tool_output_filter_error_max_lines", 100) :
            ]
        )
        start = max(traceback_start + 1, len(lines) - len(text.splitlines()) + 1)
        return _cap_configured_output(
            _outcome(
                original=output,
                text=text,
                detected="python_error",
                strategy="bash_python_error_tail",
                retained_ranges=((start, len(lines)),) if text else (),
                fallback="legacy",
            ),
            config,
        )
    return _cap_configured_output(
        _legacy_head_tail(
            output,
            max_lines=_positive_config(config, "tool_output_filter_generic_max_lines", 80),
            label="output folded",
            detected="generic",
            strategy="bash_head_tail",
        ),
        config,
    )


def _reduce_code(text: str, *, max_lines: int) -> ReductionOutcome:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return _outcome(
            original=text,
            text=text,
            detected="code",
            strategy="read_code_adaptive",
            retained_ranges=((1, len(lines)),) if lines else (),
        )

    ranges: list[tuple[int, int]] = [(1, min(8, len(lines)))]
    imports = [index + 1 for index, line in enumerate(lines) if _CODE_IMPORT_RE.match(line)]
    declarations = [
        index + 1 for index, line in enumerate(lines) if _CODE_DECLARATION_RE.match(line)
    ]
    for index in _evenly_spaced(imports, max(1, min(6, max_lines // 10))):
        ranges.append((index, min(index + 1, len(lines))))
    for index in _evenly_spaced(declarations, max(1, min(12, max_lines // 4))):
        ranges.append((index, min(index + 2, len(lines))))
    ranges.append((max(1, len(lines) - 5), len(lines)))
    return _ranges_outcome(
        original=text,
        ranges=ranges,
        max_lines=max_lines,
        detected="code",
        strategy="read_code_adaptive",
        label="code content folded",
    )


def _reduce_test_log(text: str, *, max_lines: int, strategy: str) -> ReductionOutcome:
    lines = text.splitlines()
    ranges: list[tuple[int, int]] = []
    current_test_header: int | None = None
    for index, line in enumerate(lines, start=1):
        if _TEST_BLOCK_HEADER_RE.match(line):
            # Pytest and Jest failure sections name the test before its
            # assertion and traceback. Retain that exact heading as part of
            # its block rather than returning an orphaned diagnostic under a
            # tight budget.
            current_test_header = index
            ranges.append((index, index))
            continue
        if _PYTEST_SECTION_HEADER_RE.match(line) or _TEST_SUMMARY_RE.search(line):
            current_test_header = None
        if not _TEST_RE.search(line):
            continue
        after = 6 if "Traceback" in line or "FAILURES" in line or "ERRORS" in line else 3
        # Begin each diagnostic at its test heading when available. Under a
        # very small budget, fair allocation then keeps a meaningful block
        # anchor rather than an arbitrary preceding noise line.
        start = (
            current_test_header
            if current_test_header is not None and _PYTEST_DIAGNOSTIC_RE.search(line)
            else index
        )
        ranges.append((start, min(len(lines), index + after)))
        if _PYTEST_TERMINAL_DIAGNOSTIC_RE.match(line):
            current_test_header = None
    if not ranges:
        return _legacy_head_tail(
            text,
            max_lines=max_lines,
            label="test output folded",
            detected="test_log",
            strategy=f"{strategy}_fallback",
        )
    # Keep the final line as a separate priority block.  A broad tail window
    # can merge into the last failure block and let a tight fair budget discard
    # the summary that it was meant to retain.
    if lines:
        ranges.append((len(lines), len(lines)))
    return _ranges_outcome(
        original=text,
        ranges=ranges,
        max_lines=max_lines,
        detected="test_log",
        strategy=strategy,
        label="test output folded",
    )


def _reduce_diff(text: str, *, max_lines: int, strategy: str) -> ReductionOutcome:
    lines = text.splitlines()
    ranges: list[tuple[int, int]] = []
    has_diff_header = False
    has_diff_structure = False
    for index, line in enumerate(lines, start=1):
        if line.startswith("commit ") or line.startswith("Author:") or line.startswith("Date:"):
            ranges.append((index, min(len(lines), index + 1)))
        elif line.startswith("diff --git "):
            has_diff_header = True
            ranges.append((index, min(len(lines), index + 5)))
        elif line.startswith("@@ "):
            has_diff_structure = True
            ranges.append((index, min(len(lines), index + 4)))
        elif line.startswith(
            (
                "rename from ",
                "rename to ",
                "Binary files ",
                "GIT binary patch",
                "new file mode ",
                "deleted file mode ",
                "similarity index ",
            )
        ):
            has_diff_structure = True
            ranges.append((index, min(len(lines), index + 1)))
        elif line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
            ranges.append((index, index))
    # A bare `diff --git` header is not a safe enough structural parse: retain
    # the command evidence through the conservative fallback instead of
    # fabricating a file/hunk-aware result for malformed output.
    if not ranges or (has_diff_header and not has_diff_structure):
        return _legacy_head_tail(
            text,
            max_lines=max_lines,
            label="diff folded",
            detected="diff",
            strategy=f"{strategy}_fallback",
        )
    return _ranges_outcome(
        original=text,
        ranges=ranges,
        max_lines=max_lines,
        detected="diff",
        strategy=strategy,
        label="diff folded",
    )


def _reduce_python_error(text: str, *, max_lines: int, strategy: str) -> ReductionOutcome:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return _outcome(
            original=text,
            text=text,
            detected="python_error",
            strategy=strategy,
            retained_ranges=((1, len(lines)),) if lines else (),
        )

    headers = [
        index for index, line in enumerate(lines, start=1) if _TRACEBACK_HEADER_RE.search(line)
    ]
    transitions = [
        index for index, line in enumerate(lines, start=1) if _CAUSE_TRANSITION_RE.search(line)
    ]
    frames = [index for index, line in enumerate(lines, start=1) if _PYTHON_FRAME_RE.match(line)]
    transition_set = set(transitions)
    header_set = set(headers)
    exceptions = [
        index
        for index, line in enumerate(lines, start=1)
        if _TRACEBACK_RE.search(line) and index not in (header_set | transition_set)
    ]
    terminal = next(
        (index for index in range(len(lines), 0, -1) if lines[index - 1].strip()),
        0,
    )
    if not headers and not transitions and not exceptions:
        return _legacy_head_tail(
            text,
            max_lines=max_lines,
            label="error output folded",
            detected="python_error",
            strategy=f"{strategy}_fallback",
        )

    # A rendered range needs source lines plus an omission marker.  Selecting
    # fewer, deliberately separated blocks avoids the old behavior where a
    # long contiguous recursive stack merged into one range and a line budget
    # clipped its tail (including the final exception).
    max_blocks = max(1, (max_lines + 1) // 3)
    selected: list[tuple[int, int]] = []
    if terminal:
        selected.append((terminal, terminal))

    def add_block(start: int, end: int) -> None:
        if len(selected) >= max_blocks or start < 1 or end < start:
            return
        if any(
            not (end < existing_start - 1 or start > existing_end + 1)
            for existing_start, existing_end in selected
        ):
            return
        selected.append((start, end))

    # Keep the final traceback header, the first header when distinct, and
    # explicit causal transitions before spending the remaining budget on
    # sampled frames.  The terminal exception was reserved above.
    for index in _unique_positions((headers[-1] if headers else 0, headers[0] if headers else 0)):
        add_block(index, index)
    for index in _evenly_spaced(transitions, min(len(transitions), max_blocks - len(selected))):
        preceding_exception = next(
            (candidate for candidate in reversed(exceptions) if candidate < index),
            0,
        )
        # A chained-cause transition is only meaningful with the immediately
        # preceding exception. Keep them as one bounded block so spacing does
        # not drop the root cause merely because the two lines are adjacent.
        start = preceding_exception if index - preceding_exception <= 2 else index
        add_block(start, index)
    for index in _evenly_spaced(exceptions, min(len(exceptions), max_blocks - len(selected))):
        add_block(index, index)

    frame_limit = min(len(frames), max_blocks - len(selected))
    for index in _evenly_spaced(frames, frame_limit):
        # The following source line is the useful diagnostic context for a
        # frame.  Do not let that window become adjacent to a protected final
        # exception, or normalization would merge it and lose the tail again.
        add_block(index, min(len(lines), index + 1))

    if not selected:
        return _legacy_head_tail(
            text,
            max_lines=max_lines,
            label="error output folded",
            detected="python_error",
            strategy=f"{strategy}_fallback",
        )
    return _ranges_outcome(
        original=text,
        ranges=selected,
        max_lines=max_lines,
        detected="python_error",
        strategy=strategy,
        label="error output folded",
    )


def _reduce_log(text: str, *, max_lines: int) -> ReductionOutcome:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return _outcome(
            original=text,
            text=text,
            detected="log",
            strategy="read_log_deduplicated",
            retained_ranges=((1, len(lines)),) if lines else (),
        )

    ranges: list[tuple[int, int]] = [(1, min(4, len(lines))), (max(1, len(lines) - 3), len(lines))]
    seen: set[str] = set()
    for index, line in enumerate(lines, start=1):
        normalized = line.strip()
        if not normalized:
            continue
        if _PRIORITY_LOG_RE.search(line):
            ranges.append((index, min(len(lines), index + 1)))
            continue
        if normalized not in seen:
            seen.add(normalized)
            if index <= 16 or index > len(lines) - 16:
                ranges.append((index, index))
    return _ranges_outcome(
        original=text,
        ranges=ranges,
        max_lines=max_lines,
        detected="log",
        strategy="read_log_deduplicated",
        label="log output folded",
    )


def _legacy_head_tail(
    original: str,
    *,
    max_lines: int,
    label: str,
    detected: str,
    strategy: str,
) -> ReductionOutcome:
    lines = original.splitlines()
    text = shorten_head_tail(original, max_lines=max_lines, label=label)
    ranges: tuple[tuple[int, int], ...]
    if len(lines) <= max_lines:
        ranges = ((1, len(lines)),) if lines else ()
    elif max_lines <= 1:
        ranges = ((1, 1),)
    else:
        source_budget = max_lines - 1
        head = max(1, source_budget // 2)
        tail = max(0, source_budget - head)
        ranges = ((1, head),)
        if tail:
            ranges += ((len(lines) - tail + 1, len(lines)),)
    return _outcome(
        original=original,
        text=text,
        detected=detected,
        strategy=strategy,
        retained_ranges=ranges,
        fallback="legacy",
    )


def _ranges_outcome(
    *,
    original: str,
    ranges: Iterable[tuple[int, int]],
    max_lines: int,
    detected: str,
    strategy: str,
    label: str,
) -> ReductionOutcome:
    lines = original.splitlines()
    normalized = _normalize_ranges(ranges, len(lines))
    fitted = _fit_ranges(normalized, max_lines)
    rendered = _render_ranges(lines, fitted, label)
    return _outcome(
        original=original,
        text=rendered,
        detected=detected,
        strategy=strategy,
        retained_ranges=tuple(fitted),
    )


def _outcome(
    *,
    original: str,
    text: str,
    detected: str,
    strategy: str,
    retained_ranges: tuple[tuple[int, int], ...] | Iterable[tuple[int, int]],
    fallback: str = "none",
) -> ReductionOutcome:
    original_lines = len(original.splitlines())
    filtered_lines = len(text.splitlines())
    ranges = tuple(retained_ranges)
    retained_count = sum(end - start + 1 for start, end in ranges)
    return ReductionOutcome(
        text=text,
        detected=detected,
        strategy=strategy,
        original_chars=len(original),
        original_lines=original_lines,
        filtered_chars=len(text),
        filtered_lines=filtered_lines,
        omitted_lines=max(0, original_lines - retained_count),
        preserved_blocks=len(ranges),
        fallback=fallback,
        retained_line_ranges=ranges,
    )


def _cap_configured_output(outcome: ReductionOutcome, config: Any) -> ReductionOutcome:
    return cap_reduction_output(
        outcome,
        max_chars=_positive_config(config, "tool_output_filter_output_max_chars", 50000),
    )


def _cap_rendered_text(
    text: str,
    *,
    max_chars: int,
    preserve_terminal_prefix: bool,
) -> tuple[str, int]:
    """Render a bounded, auditable head/tail view of an already-selected payload."""

    if max_chars < 2:
        return text[:max_chars], max(0, len(text) - max_chars)

    # A Python terminal exception can itself be a generated one-line payload.
    # Preserve both its type/message prefix and its trailing evidence instead
    # of letting a generic tail-only fragment discard the exception name.
    root_anchor_start: int | None = None
    error_anchor_start: int | None = None
    if preserve_terminal_prefix:
        root_anchor_start, error_anchor_start = _chained_error_anchor_starts(text)
    if root_anchor_start is not None and error_anchor_start is not None:
        rendered = _cap_with_chained_error_anchors(
            text,
            max_chars=max_chars,
            root_anchor_start=root_anchor_start,
            terminal_anchor_start=error_anchor_start,
        )
        if rendered is not None:
            return rendered
    if error_anchor_start is not None:
        rendered = _cap_with_terminal_prefix(
            text,
            max_chars=max_chars,
            anchor_start=error_anchor_start,
        )
        if rendered is not None:
            return rendered

    marker = _character_omission_marker(len(text))
    for _ in range(4):
        retained = max_chars - len(marker)
        if retained < 2:
            return text[:max_chars], max(0, len(text) - max_chars)
        omitted = len(text) - retained
        next_marker = _character_omission_marker(omitted)
        if next_marker == marker:
            break
        marker = next_marker
    head = retained // 2
    tail = retained - head
    return text[:head] + marker + text[-tail:], omitted


def _cap_with_terminal_prefix(
    text: str,
    *,
    max_chars: int,
    anchor_start: int,
) -> tuple[str, int] | None:
    marker = _character_omission_marker(len(text))
    for _ in range(4):
        retained = max_chars - 2 * len(marker)
        if retained < 3:
            return None
        head = max(1, retained // 3)
        terminal_prefix = max(1, retained // 3)
        tail = retained - head - terminal_prefix
        anchor_tail_length = len(text) - anchor_start
        if anchor_start <= head or anchor_tail_length <= terminal_prefix + tail:
            return None
        omitted = len(text) - head - terminal_prefix - tail
        next_marker = _character_omission_marker(omitted)
        if next_marker == marker:
            return (
                (
                    text[:head]
                    + marker
                    + text[anchor_start : anchor_start + terminal_prefix]
                    + marker
                    + text[-tail:]
                ),
                omitted,
            )
        marker = next_marker
    return None


def _cap_with_chained_error_anchors(
    text: str,
    *,
    max_chars: int,
    root_anchor_start: int,
    terminal_anchor_start: int,
) -> tuple[str, int] | None:
    """Keep a chained root and terminal exception under the character guard."""

    if not 0 <= root_anchor_start < terminal_anchor_start < len(text):
        return None

    # A selected root error may sit behind a long quoted frame and an omission
    # marker. Reserve space for all three possible gaps so the cap remains
    # strict while the source fragments retain both exception anchors.
    marker_capacity = len(_character_omission_marker(len(text)))
    source_budget = max_chars - 3 * marker_capacity
    if source_budget < 3:
        return None
    root_budget = max(1, source_budget // 3)
    terminal_prefix_budget = max(1, (source_budget - root_budget) // 2)
    terminal_tail_budget = source_budget - root_budget - terminal_prefix_budget

    chunks = [
        (root_anchor_start, min(terminal_anchor_start, root_anchor_start + root_budget)),
        (
            terminal_anchor_start,
            min(len(text), terminal_anchor_start + terminal_prefix_budget),
        ),
        (
            max(
                terminal_anchor_start + terminal_prefix_budget,
                len(text) - terminal_tail_budget,
            ),
            len(text),
        ),
    ]
    merged: list[tuple[int, int]] = []
    for start, end in chunks:
        if start >= end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    if len(merged) < 2:
        return None

    omitted_chars = len(text) - sum(end - start for start, end in merged)
    marker = _character_omission_marker(omitted_chars)
    rendered: list[str] = []
    previous_end = 0
    for start, end in merged:
        if start > previous_end:
            rendered.append(marker)
        rendered.append(text[start:end])
        previous_end = end
    if previous_end < len(text):
        rendered.append(marker)
    result = "".join(rendered)
    if len(result) > max_chars:
        return None
    return result, omitted_chars


def _character_omission_marker(omitted_chars: int) -> str:
    # Keep this marker inline: character safety must not make a reducer exceed
    # its independently configured line budget.
    return f" ... [output safety cap: {omitted_chars} chars omitted] ... "


def _last_error_anchor_start(text: str) -> int | None:
    offset = 0
    anchor: int | None = None
    for line in text.splitlines(keepends=True):
        if _TRACEBACK_RE.search(line):
            anchor = offset
        offset += len(line)
    return anchor


def _chained_error_anchor_starts(text: str) -> tuple[int | None, int | None]:
    """Return the first explicit chained cause and the terminal error anchor."""

    exception_starts: list[int] = []
    transition_starts: list[int] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        if _CAUSE_TRANSITION_RE.search(line):
            transition_starts.append(offset)
        elif _TRACEBACK_RE.search(line) and not _TRACEBACK_HEADER_RE.search(line):
            exception_starts.append(offset)
        offset += len(line)

    terminal = exception_starts[-1] if exception_starts else _last_error_anchor_start(text)
    if terminal is None:
        return None, terminal
    root = (
        next(
            (start for start in exception_starts if start < transition_starts[0]),
            None,
        )
        if transition_starts
        else None
    )
    # Range selection may retain the root and the next traceback header but
    # omit the transition line between them. In that rendered form, the first
    # of two distinct exception anchors is still the deterministic root to
    # protect alongside the terminal error.
    if root is None and len(exception_starts) >= 2:
        root = exception_starts[0]
    return root, terminal


def _positive_config(config: Any, name: str, fallback: int) -> int:
    value = getattr(config, name, fallback)
    if isinstance(value, bool):
        return fallback
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed > 0 else fallback


def _evenly_spaced(values: list[int], limit: int) -> list[int]:
    if len(values) <= limit:
        return values
    if limit <= 1:
        return [values[len(values) // 2]]
    return [values[round(index * (len(values) - 1) / (limit - 1))] for index in range(limit)]


def _unique_positions(values: tuple[int, ...]) -> list[int]:
    positions: list[int] = []
    for value in values:
        if value > 0 and value not in positions:
            positions.append(value)
    return positions


def _normalize_ranges(
    ranges: Iterable[tuple[int, int]],
    line_count: int,
) -> list[tuple[int, int]]:
    normalized: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        start = max(1, start)
        end = min(line_count, end)
        if start > end:
            continue
        if normalized and start <= normalized[-1][1] + 1:
            normalized[-1] = (normalized[-1][0], max(normalized[-1][1], end))
        else:
            normalized.append((start, end))
    return normalized


def _fit_ranges(ranges: list[tuple[int, int]], max_lines: int) -> list[tuple[int, int]]:
    if not ranges or max_lines < 1:
        return []

    # Every retained block needs at least one source line and every gap needs an
    # omission marker.  Sampling at most half the line budget therefore keeps a
    # middle block visible instead of spending the whole budget on one range.
    max_ranges = max(1, (max_lines + 1) // 2)
    if len(ranges) > max_ranges:
        ranges = [ranges[index] for index in _evenly_spaced(list(range(len(ranges))), max_ranges)]
    marker_count = max(0, len(ranges) - 1)
    source_budget = max(1, max_lines - marker_count)
    lengths = [end - start + 1 for start, end in ranges]
    if sum(lengths) <= source_budget:
        return ranges

    allocations = [0] * len(ranges)
    remaining = source_budget
    for index in _evenly_spaced(list(range(len(ranges))), min(len(ranges), remaining)):
        allocations[index] = 1
        remaining -= 1
    while remaining:
        progressed = False
        for index, length in enumerate(lengths):
            if remaining and allocations[index] < length:
                allocations[index] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return [
        (start, start + allocation - 1)
        for (start, _), allocation in zip(ranges, allocations)
        if allocation
    ]


def _render_ranges(lines: list[str], ranges: list[tuple[int, int]], label: str) -> str:
    rendered: list[str] = []
    previous_end = 0
    for start, end in ranges:
        if previous_end and start > previous_end + 1:
            rendered.append(f"... [{label}: {start - previous_end - 1} lines omitted] ...")
        rendered.extend(lines[start - 1 : end])
        previous_end = end
    return "\n".join(rendered)
