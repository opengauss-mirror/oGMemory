from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from core.errors import NodeNotFoundError
from core.models import RequestContext
from service.tool_output import ToolOutputFilterService, ToolOutputRepository
from service.tool_output_filtering import (
    detect_content,
    legacy_bash_reduction,
    legacy_read_reduction,
    passthrough_reduction,
    reduce_bash_output,
    reduce_read_output,
)


class FakeContextFS:
    """Unit double for the direct ContextFS repository boundary."""

    def __init__(self) -> None:
        self.nodes: dict[str, Any] = {}

    def exists(self, uri, ctx):
        del ctx
        return uri in self.nodes

    def write_node(self, node, ctx):
        del ctx
        self.nodes[node.uri] = node

    def read_node(self, uri, ctx):
        del ctx
        if uri not in self.nodes:
            raise NodeNotFoundError(uri)
        return self.nodes[uri]

    def list_children(self, uri, ctx):
        del ctx
        prefix = uri.rstrip("/") + "/"
        return [
            child_uri
            for child_uri in self.nodes
            if child_uri.startswith(prefix) and "/" not in child_uri[len(prefix) :]
        ]

    def delete_node(self, uri, ctx):
        del ctx
        if uri not in self.nodes:
            raise NodeNotFoundError(uri)
        del self.nodes[uri]


def _ctx(*, account: str = "acct", agent: str = "agent") -> RequestContext:
    return RequestContext(
        account_id=account,
        user_id="user",
        agent_id=agent,
        session_id="session",
        trace_id="trace",
    )


def _service(fs: FakeContextFS | None = None, **flags):
    values = {
        "tool_output_filter_enabled": True,
        "tool_output_filter_read_enabled": True,
        "tool_output_filter_bash_enabled": True,
        "tool_output_filter_adaptive_enabled": False,
        "tool_output_filter_read_min_chars": 8000,
        "tool_output_filter_read_max_lines": 60,
        "tool_output_filter_bash_min_chars": 8000,
        "tool_output_filter_bash_min_lines": 100,
        "tool_output_filter_test_max_lines": 120,
        "tool_output_filter_diff_max_lines": 100,
        "tool_output_filter_error_max_lines": 100,
        "tool_output_filter_generic_max_lines": 80,
        "tool_output_filter_output_max_chars": 50000,
        "tool_output_recovery_max_chars": 50000,
    }
    values.update(flags)
    backend = fs or FakeContextFS()
    return ToolOutputFilterService(
        ToolOutputRepository(backend), SimpleNamespace(**values)
    ), backend


def test_read_filter_persists_exact_native_content_before_returning_filtered_output():
    service, fs = _service(
        tool_output_filter_read_min_chars=10,
        tool_output_filter_read_max_lines=12,
    )
    original = "\n".join(f"line {index}" for index in range(120))

    result = service.filter_read(
        {"relative_path": "long.txt", "content": original},
        _ctx(),
    )

    assert result["ok"] is True
    assert result["was_shortened"] is True
    assert result["memory_uri"].endswith(result["memory_uri"].rsplit("/", 1)[1])
    assert len(result["memory_uri"].rsplit("_", 1)[1]) == 32
    filtered_text = result["filtered_output"]
    assert "file content folded" in filtered_text
    assert "<<<FILTER TRIGGERED>>>" not in filtered_text
    assert "ogm_read_memory(uri=" not in filtered_text
    assert "recover_original=true" not in filtered_text
    assert result["memory_uri"] not in filtered_text
    node = next(iter(fs.nodes.values()))
    assert node.content == original
    assert node.metadata["workspace_relative_path"] == "long.txt"
    assert node.metadata["source"] == "post_tool_use"
    assert node.metadata["was_shortened"] is True
    assert "workspace" not in node.overview


def test_read_filter_persistence_failure_never_returns_filtered_output(monkeypatch):
    service, fs = _service(
        tool_output_filter_read_min_chars=10,
        tool_output_filter_read_max_lines=12,
    )
    original = "\n".join(f"line {index}" for index in range(120))

    def fail_store(*args, **kwargs):
        del args, kwargs
        raise OSError("store failed")

    monkeypatch.setattr(service._repository, "store_original", fail_store)

    with pytest.raises(OSError, match="store failed"):
        service.filter_read(
            {"relative_path": "long.txt", "content": original},
            _ctx(),
        )
    assert not fs.nodes


def test_read_changed_but_not_shorter_delivery_is_skipped_without_recovery_state(
    monkeypatch,
):
    service, fs = _service(
        tool_output_filter_read_min_chars=1,
        tool_output_filter_read_max_lines=2,
    )
    original = "a\nb\nc"

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("a non-shorter Read result must not hash or store")

    monkeypatch.setattr("service.tool_output._hash_text_parts", forbidden)
    monkeypatch.setattr(service._repository, "store_original", forbidden)

    result = service.filter_read(
        {"relative_path": "short.txt", "content": original},
        _ctx(),
    )

    assert result["ok"] is True
    assert result["skipped"] is True
    assert result["reason"] == "not_shorter"
    assert result["was_shortened"] is False
    assert result["original_chars"] == len(original)
    assert result["filtered_chars"] == len(original)
    assert result["original_lines"] == 3
    assert result["filtered_lines"] == 3
    assert result["reduction"]["omitted_lines"] > 0
    assert result["diagnostics"]["state"] == "skipped"
    assert not fs.nodes


@pytest.mark.parametrize("tool_name", ["Read", "Bash"])
def test_oversized_multibyte_output_fails_open_before_reduction_hash_or_store(
    monkeypatch,
    tool_name,
):
    service, fs = _service(
        tool_output_filter_read_min_chars=1,
        tool_output_filter_bash_min_chars=1,
        tool_output_filter_bash_min_lines=1,
    )
    original = "界" * ((1024 * 1024 // 3) + 1)

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("oversized output must fail before expensive work")

    monkeypatch.setattr("service.tool_output._hash_text_parts", forbidden)
    monkeypatch.setattr(service._repository, "store_original", forbidden)
    if tool_name == "Read":
        monkeypatch.setattr("service.tool_output.reduce_read_output", forbidden)
        monkeypatch.setattr("service.tool_output.legacy_read_reduction", forbidden)
        result = service.filter_read(
            {"relative_path": "large.txt", "content": original},
            _ctx(),
        )
    else:
        monkeypatch.setattr("service.tool_output.reduce_bash_output", forbidden)
        monkeypatch.setattr("service.tool_output.legacy_bash_reduction", forbidden)
        result = service.filter_bash(
            {"command": "python -m pytest", "output": original},
            _ctx(),
        )

    assert len(original) < 1024 * 1024
    assert result["ok"] is False
    assert result["reason"] == "original_too_large"
    assert result["diagnostics"]["state"] == "skipped"
    assert result["diagnostics"]["reason"] == "original_too_large"
    assert "memory_uri" not in result
    assert "filtered_output" not in result
    assert not fs.nodes


def test_read_filter_rejects_unsafe_logical_locator():
    service, _ = _service()
    result = service.filter_read(
        {"relative_path": "../outside.txt", "content": "secret\n" * 100},
        _ctx(),
    )

    assert result["ok"] is False
    assert result["reason"] == "unsafe_locator"
    assert result["diagnostics"]["state"] == "skipped"
    assert result["diagnostics"]["reason"] == "unsafe_locator"
    assert "outside.txt" not in json.dumps(result["diagnostics"])


def test_filter_diagnostics_are_safe_for_disabled_unsupported_unchanged_and_fallback(tmp_path):
    disabled, _ = _service(tool_output_filter_enabled=False)
    disabled_result = disabled.filter_bash(
        {"command": "python -m pytest", "output": "private output"},
        _ctx(),
    )
    assert disabled_result["diagnostics"]["state"] == "disabled"

    service, _ = _service(
        tool_output_filter_read_min_chars=1,
        tool_output_filter_bash_min_chars=1,
        tool_output_filter_bash_min_lines=1,
        tool_output_filter_generic_max_lines=4,
    )
    unsupported = service.filter_bash(
        {"command": "echo", "output": "private unsupported output"},
        _ctx(),
    )
    assert unsupported["diagnostics"]["state"] == "unsupported"
    assert unsupported["diagnostics"]["reason"] == "not_whitelisted"

    unchanged = service.filter_read(
        {"relative_path": "unchanged.txt", "content": "secret unchanged output"},
        _ctx(),
    )
    assert unchanged["ok"] is True
    assert unchanged["skipped"] is True
    assert unchanged["diagnostics"]["state"] == "skipped"
    assert unchanged["diagnostics"]["was_shortened"] is False

    private_output = "private fallback output\n" * 80
    fallback = service.filter_bash(
        {"command": "python -m pytest", "output": private_output},
        _ctx(),
    )
    diagnostics = fallback["diagnostics"]
    assert fallback["was_shortened"] is True
    assert diagnostics["state"] == "fallback"
    assert diagnostics["fallback"] == "legacy"
    assert diagnostics["input_chars"] == len(private_output)
    assert diagnostics["output_chars"] == fallback["filtered_chars"]
    assert diagnostics["reduction"]["omitted_lines"] > 0
    serialized = json.dumps(diagnostics, sort_keys=True)
    assert "private-workspace" not in serialized
    assert "private unsupported output" not in serialized
    assert "private fallback output" not in serialized


@pytest.mark.parametrize(
    ("command", "output", "pre_category", "expected"),
    [
        ("python -m pytest", "1 failed", "", "test_runner"),
        ("git diff", "diff --git a/a.py b/a.py\n@@ -1 +1 @@", "", "diff"),
        ("git show HEAD", "diff --git a/a.py b/a.py\n@@ -1 +1 @@", "", "diff"),
        ("git log -p -1", "diff --git a/a.py b/a.py\n@@ -1 +1 @@", "", "diff"),
        (
            "python script.py",
            "Traceback (most recent call last):\nValueError: bad",
            "",
            "python_error",
        ),
        ("custom", "@@ -1 +1 @@", "diff_maybe", "diff"),
        ("custom", "ValueError: bad", "python_maybe", "python_error"),
        ("ls -la", "total 0", "", ""),
    ],
)
def test_bash_classification_matrix(command, output, pre_category, expected):
    assert (
        ToolOutputFilterService.classify_bash_output(
            command,
            output,
            pre_category,
        )
        == expected
    )


def test_supported_bash_below_threshold_is_skipped_without_recovery_state():
    service, fs = _service(
        tool_output_filter_bash_min_chars=1000,
        tool_output_filter_bash_min_lines=1000,
    )

    result = service.filter_bash(
        {"command": "python -m pytest", "output": "1 passed", "exit_code": 0},
        _ctx(),
    )

    assert result["ok"] is True
    assert result["skipped"] is True
    assert result["reason"] == "below_min_size"
    assert result["strategy"] == "bash_passthrough_below_min_size"
    assert result["was_shortened"] is False
    assert "filtered_output" not in result
    assert "memory_uri" not in result
    assert result["diagnostics"]["state"] == "skipped"
    assert not fs.nodes


def test_bash_digest_is_request_addressed():
    service, fs = _service(
        tool_output_filter_bash_min_chars=1,
        tool_output_filter_bash_min_lines=1,
        tool_output_filter_test_max_lines=4,
    )
    ctx = _ctx()
    output = "\n".join(["= FAILURES =", *[f"detail {index}" for index in range(80)], "1 failed"])

    first = service.filter_bash(
        {"command": "pytest", "output": output, "exit_code": 1},
        ctx,
    )
    second = service.filter_bash(
        {"command": "pytest", "output": output, "exit_code": 2},
        ctx,
    )

    assert first["memory_uri"] != second["memory_uri"]
    assert len(fs.nodes) == 2
    assert all(node.metadata["source"] == "post_tool_use" for node in fs.nodes.values())


def test_bash_unchanged_reduction_is_skipped_without_recovery_state(monkeypatch):
    service, fs = _service(
        tool_output_filter_bash_min_chars=1,
        tool_output_filter_bash_min_lines=1,
    )
    output = "Traceback (most recent call last):\nValueError: private"
    monkeypatch.setattr(
        "service.tool_output.reduce_bash_output",
        lambda command, text, category, config: passthrough_reduction(
            text,
            detected="python_error",
            strategy="unchanged_fixture",
        ),
    )

    result = service.filter_bash(
        {"command": "python worker.py", "output": output, "exit_code": 1},
        _ctx(),
    )

    assert result["ok"] is True
    assert result["skipped"] is True
    assert result["reason"] == "unchanged"
    assert result["was_shortened"] is False
    assert "filtered_output" not in result
    assert "memory_uri" not in result
    assert result["diagnostics"]["state"] == "skipped"
    assert not fs.nodes


def test_bash_changed_but_not_shorter_delivery_is_skipped_without_recovery_state(
    monkeypatch,
):
    service, fs = _service(
        tool_output_filter_bash_min_chars=1,
        tool_output_filter_bash_min_lines=1,
    )
    output = "Traceback (most recent call last):\nValueError: private"
    monkeypatch.setattr(
        "service.tool_output.reduce_bash_output",
        lambda command, text, category, config: passthrough_reduction(
            text + " changed",
            detected="python_error",
            strategy="changed_but_longer_fixture",
        ),
    )

    result = service.filter_bash(
        {"command": "python worker.py", "output": output},
        _ctx(),
    )

    assert result["ok"] is True
    assert result["skipped"] is True
    assert result["reason"] == "not_shorter"
    assert result["was_shortened"] is False
    assert result["filtered_chars"] == len(output)
    assert "filtered_output" not in result
    assert "memory_uri" not in result
    assert result["diagnostics"]["state"] == "skipped"
    assert not fs.nodes


def test_master_and_child_flags_gate_only_new_filter_writes():
    master_off, master_fs = _service(
        tool_output_filter_enabled=False,
        tool_output_filter_read_min_chars=1,
    )
    child_off, child_fs = _service(
        tool_output_filter_read_enabled=False,
        tool_output_filter_read_min_chars=1,
    )

    params = {"relative_path": "long.txt", "content": "x\n" * 100}
    master_result = master_off.filter_read(params, _ctx())
    child_result = child_off.filter_read(params, _ctx())
    assert master_result["ok"] is False
    assert master_result["reason"] == "feature_disabled"
    assert master_result["diagnostics"]["state"] == "disabled"
    assert child_result["ok"] is False
    assert child_result["reason"] == "feature_disabled"
    assert child_result["diagnostics"]["state"] == "disabled"
    assert not master_fs.nodes
    assert not child_fs.nodes


def test_read_adaptive_filter_preserves_middle_code_evidence():
    service, fs = _service(
        tool_output_filter_adaptive_enabled=True,
        tool_output_filter_read_min_chars=1,
        tool_output_filter_read_max_lines=32,
    )
    original = "\n".join(
        [
            "from package import dependency",
            "",
            "def head_symbol():",
            "    return dependency",
            *[f"filler_{index} = {index}" for index in range(80)],
            "def middle_symbol():",
            "    raise RuntimeError('MIDDLE_EVIDENCE')",
            *[f"middle_{index} = {index}" for index in range(80)],
            "class TailSymbol:",
            "    important_dependency = dependency",
            "    def run(self):",
            "        return 'TAIL_EVIDENCE'",
        ]
    )

    result = service.filter_read(
        {"relative_path": "code-excerpt.txt", "content": original},
        _ctx(),
    )

    filtered = result["filtered_output"]
    assert result["detected"] == "code"
    assert result["strategy"] == "read_code_adaptive"
    assert "from package import dependency" in filtered
    assert "def middle_symbol():" in filtered
    assert "class TailSymbol:" in filtered
    assert result["reduction"]["retained_line_ranges"]
    node = next(iter(fs.nodes.values()))
    assert node.metadata["reduction"]["fallback"] == "none"


@pytest.mark.parametrize("range_params", [{"offset": 0}, {"limit": 20}, {"offset": 1, "limit": 4}])
def test_read_filter_explicit_range_is_non_destructive(range_params):
    service, fs = _service(tool_output_filter_read_min_chars=1)

    result = service.filter_read(
        {"relative_path": "large.txt", "content": "line\n" * 200, **range_params},
        _ctx(),
    )

    assert result["ok"] is False
    assert result["reason"] == "explicit_read_range"
    assert result["diagnostics"]["state"] == "skipped"
    assert not fs.nodes


def test_read_filter_consumes_authorized_content_without_filesystem_read(monkeypatch):
    service, fs = _service(tool_output_filter_read_min_chars=1)
    original = "line\n" * 200

    def fail_if_read(*_args, **_kwargs):
        raise AssertionError("filter service must not read the file")

    monkeypatch.setattr(Path, "read_text", fail_if_read)
    result = service.filter_read(
        {"relative_path": "large.txt", "content": original},
        _ctx(),
    )

    assert result["ok"] is True
    assert result["was_shortened"] is True
    assert next(iter(fs.nodes.values())).content == original


def test_read_filter_rejects_binary_content_without_storing_original():
    service, fs = _service(tool_output_filter_read_min_chars=1)

    result = service.filter_read(
        {"relative_path": "artifact.bin", "content": "header\x00payload" * 100},
        _ctx(),
    )

    assert result["ok"] is False
    assert result["reason"] == "unsupported_content"
    assert not fs.nodes


@pytest.mark.parametrize("filename", ["image.png", "notebook.ipynb", "report.pdf"])
def test_read_filter_rejects_unsupported_content_suffixes_without_storage(filename):
    service, fs = _service(tool_output_filter_read_min_chars=1)

    result = service.filter_read(
        {"relative_path": filename, "content": "textual fixture\n" * 100},
        _ctx(),
    )

    assert result["ok"] is False
    assert result["reason"] == "unsupported_content"
    assert not fs.nodes


def test_content_detection_keeps_repeated_error_logs_out_of_test_strategy():
    error_log = "\n".join(
        [
            "2026-08-09 10:00:00 ERROR connection refused",
            "2026-08-09 10:00:01 INFO retrying",
            "2026-08-09 10:00:02 ERROR connection refused",
            "2026-08-09 10:00:03 INFO retrying",
        ]
    )

    assert detect_content(error_log, relative_path="service.log") == "log"
    assert detect_content("short test summary info\n1 failed in 0.01s") == "test_log"


@pytest.mark.parametrize("suffix", ["py", "pyi", "pyx", "js", "ts", "java"])
@pytest.mark.parametrize("adaptive", [True, False])
def test_source_read_is_not_classified_as_an_embedded_error_or_test_log(suffix, adaptive):
    source = (
        "class UserChangeForm:\n"
        "    password = ReadOnlyPasswordHashField()\n"
        "    def clean(self):\n"
        "        try:\n            return self.password\n"
        "        except ValueError:\n            return None\n"
        "    example = '''Traceback (most recent call last):\n"
        "ValueError: example\nshort test summary info\n1 failed'''\n"
    )
    source = "# padding\n" * 500 + source + "# tail\n" * 500
    assert detect_content(source, relative_path=f"forms.{suffix}") == "code"
    service, fs = _service(
        tool_output_filter_adaptive_enabled=adaptive,
        tool_output_filter_read_min_chars=1,
        tool_output_filter_read_max_lines=2,
        tool_output_filter_output_max_chars=100,
    )
    result = service.filter_read({"relative_path": f"forms.{suffix}", "content": source}, _ctx())
    assert result["detected"] == "code"
    assert result["was_shortened"] is False
    assert result["strategy"] == "read_source_passthrough"
    assert result["original_chars"] == result["filtered_chars"] == len(source)
    outcome = reduce_read_output(source, relative_path=f"forms.{suffix}", config=service._cfg)
    assert outcome.text == source
    assert not fs.nodes
    # Without a source filename the same diagnostic remains an actual error.
    assert detect_content("Traceback (most recent call last):\nValueError: bad", relative_path="run.log") == "python_error"


@pytest.mark.parametrize("max_lines", [1, 2, 12])
def test_legacy_head_tail_reserves_the_marker_within_its_line_budget(max_lines):
    original = "\n".join(f"line {index}" for index in range(40))
    outcome = legacy_read_reduction(
        original,
        SimpleNamespace(
            tool_output_filter_read_max_lines=max_lines,
            tool_output_filter_output_max_chars=50_000,
        ),
    )

    assert outcome.filtered_lines <= max_lines
    assert outcome.omitted_lines > 0
    if max_lines > 1:
        assert "file content folded" in outcome.text


def test_bash_adaptive_strategies_preserve_failure_diff_and_error_anchors():
    service, _ = _service(
        tool_output_filter_adaptive_enabled=True,
        tool_output_filter_bash_min_chars=1,
        tool_output_filter_bash_min_lines=1,
        tool_output_filter_test_max_lines=48,
        tool_output_filter_diff_max_lines=48,
        tool_output_filter_error_max_lines=48,
    )
    test_output = "\n".join(
        [
            *[f"collecting noise {index}" for index in range(40)],
            "________________________ test_alpha ________________________",
            "assert actual == expected",
            "E       AssertionError: expected 3, got 2",
            "Traceback (most recent call last):",
            "  File 'tests/test_alpha.py', line 12, in test_alpha",
            "FAILED tests/test_alpha.py::test_alpha - AssertionError",
            "short test summary info",
            "1 failed, 99 passed in 2.0s",
            *[f"tail noise {index}" for index in range(40)],
        ]
    )
    diff_output = "\n".join(
        [
            "commit abc123",
            "diff --git a/first.py b/first.py",
            "--- a/first.py",
            "+++ b/first.py",
            "@@ -1,2 +1,2 @@",
            "-old_first",
            "+new_first",
            *[" context" for _ in range(50)],
            "diff --git a/second.py b/second.py",
            "@@ -8 +8 @@",
            "-old_second",
            "+new_second",
        ]
    )
    error_output = "\n".join(
        [
            "Traceback (most recent call last):",
            "  File 'root.py', line 2, in root",
            "ValueError: ROOT_CAUSE",
            "During handling of the above exception, another exception occurred:",
            *[f"  File 'noise.py', line {index}, in recurse" for index in range(80)],
            "Traceback (most recent call last):",
            "  File 'worker.py', line 9, in run",
            "RuntimeError: FINAL_FAILURE",
        ]
    )

    test_result = service.filter_bash({"command": "pytest -q", "output": test_output}, _ctx())
    diff_result = service.filter_bash({"command": "git diff", "output": diff_output}, _ctx())
    error_result = service.filter_bash(
        {"command": "python worker.py", "output": error_output},
        _ctx(),
    )

    assert test_result["strategy"] == "bash_test_runner_adaptive"
    assert "AssertionError: expected 3, got 2" in test_result["filtered_output"]
    assert "short test summary info" in test_result["filtered_output"]
    assert diff_result["strategy"] == "bash_diff_adaptive"
    assert "diff --git a/first.py b/first.py" in diff_result["filtered_output"]
    assert "diff --git a/second.py b/second.py" in diff_result["filtered_output"]
    assert "+new_second" in diff_result["filtered_output"]
    assert error_result["strategy"] == "bash_python_error_adaptive"
    assert "ROOT_CAUSE" in error_result["filtered_output"]
    assert "FINAL_FAILURE" in error_result["filtered_output"]
    for result in (test_result, diff_result, error_result):
        assert result["memory_uri"] not in result["filtered_output"]
        assert "<<<FILTER TRIGGERED>>>" not in result["filtered_output"]
        assert "ogm_read_memory(uri=" not in result["filtered_output"]
        assert "recover_original=true" not in result["filtered_output"]
    assert all(
        result["reduction"]["fallback"] == "none"
        for result in (test_result, diff_result, error_result)
    )


def test_legacy_bash_reduction_reports_exact_source_ranges():
    output = "\n".join(
        [
            "collection noise",
            "FAILED tests/test_alpha.py::test_alpha",
            "more noise",
            "short test summary info",
            "1 failed in 0.01s",
        ]
    )
    config = SimpleNamespace(
        tool_output_filter_test_max_lines=2,
        tool_output_filter_generic_max_lines=80,
    )

    outcome = legacy_bash_reduction(output, "test_runner", config)

    assert outcome.text.splitlines() == [
        "FAILED tests/test_alpha.py::test_alpha",
        "short test summary info",
    ]
    assert outcome.reduction_metadata()["retained_line_ranges"] == [
        {"start": 2, "end": 2},
        {"start": 4, "end": 4},
    ]
    assert outcome.omitted_lines == 3
    assert outcome.was_shortened is True


def test_adaptive_test_reduction_fairly_retains_failures_and_summary():
    output_lines: list[str] = []
    for index in range(5):
        output_lines.extend(
            [
                f"FAILED tests/test_{index}.py::test_{index}",
                *[f"noise {index}-{line}" for line in range(6)],
            ]
        )
    output_lines.append("5 failed, 1 passed in 0.01s")
    output = "\n".join(output_lines)
    config = SimpleNamespace(
        tool_output_filter_adaptive_enabled=True,
        tool_output_filter_test_max_lines=11,
        tool_output_filter_diff_max_lines=100,
        tool_output_filter_error_max_lines=100,
        tool_output_filter_generic_max_lines=80,
    )

    outcome = reduce_bash_output("pytest -q", output, "test_runner", config)

    assert outcome.filtered_lines <= 11
    for index in range(5):
        assert f"FAILED tests/test_{index}.py::test_{index}" in outcome.text
    assert "5 failed, 1 passed in 0.01s" in outcome.text
    assert outcome.was_shortened is True


def test_adaptive_test_reduction_keeps_pytest_heading_with_its_diagnostic():
    heading = "________________________ test_alpha ________________________"
    output = "\n".join(
        [
            *[f"collection noise {index}" for index in range(12)],
            heading,
            "assert actual == expected",
            "E       AssertionError: expected 3, got 2",
            "FAILED tests/test_alpha.py::test_alpha - AssertionError",
            *[f"between-block noise {index}" for index in range(8)],
            "short test summary info",
            "1 failed in 0.01s",
        ]
    )
    config = SimpleNamespace(
        tool_output_filter_adaptive_enabled=True,
        tool_output_filter_test_max_lines=12,
        tool_output_filter_diff_max_lines=100,
        tool_output_filter_error_max_lines=100,
        tool_output_filter_generic_max_lines=80,
        tool_output_filter_output_max_chars=50000,
    )

    outcome = reduce_bash_output("pytest -q", output, "test_runner", config)
    repeated = reduce_bash_output("pytest -q", output, "test_runner", config)

    assert outcome.filtered_lines <= 12
    assert heading in outcome.text
    assert "AssertionError: expected 3, got 2" in outcome.text
    assert "short test summary info" in outcome.text
    assert outcome.reduction_metadata()["retained_line_ranges"][0]["start"] == 13
    assert outcome == repeated


def test_adaptive_test_reduction_keeps_jest_and_npm_failure_evidence():
    jest_output = "\n".join(
        [
            *[f"collection noise {index}" for index in range(12)],
            "FAIL src/widget.test.js",
            "  ● renders the widget",
            "    Expected: visible",
            "    Received: hidden",
            "Test Suites: 1 failed, 1 passed, 2 total",
            "Tests:       1 failed, 1 passed, 2 total",
        ]
    )
    npm_output = "\n".join(
        [
            "npm ERR! code ELIFECYCLE",
            "npm ERR! errno 1",
            "npm ERR! app@1.0.0 test: `jest --runInBand`",
            "npm ERR! Test failed with exit code 1",
            "npm ERR! A complete log is available in the npm cache",
        ]
    )
    config = SimpleNamespace(
        tool_output_filter_adaptive_enabled=True,
        tool_output_filter_test_max_lines=12,
        tool_output_filter_diff_max_lines=100,
        tool_output_filter_error_max_lines=100,
        tool_output_filter_generic_max_lines=80,
    )

    jest = reduce_bash_output("npm test", jest_output, "test_runner", config)
    npm = reduce_bash_output("npm test", npm_output, "test_runner", config)

    assert "FAIL src/widget.test.js" in jest.text
    assert "● renders the widget" in jest.text
    assert "Expected: visible" in jest.text
    assert "1 failed" in jest.text
    assert "npm ERR! code ELIFECYCLE" in npm.text
    assert "npm ERR! Test failed with exit code 1" in npm.text
    assert jest.was_shortened is True
    assert npm.strategy == "bash_test_runner_adaptive"


def test_adaptive_diff_falls_back_for_a_bare_malformed_diff_header():
    output = "\n".join(
        [
            "fatal: malformed patch after this header",
            "diff --git a/broken.py b/broken.py",
            *[f"diagnostic tail {index}" for index in range(40)],
        ]
    )
    config = SimpleNamespace(
        tool_output_filter_adaptive_enabled=True,
        tool_output_filter_diff_max_lines=8,
        tool_output_filter_test_max_lines=120,
        tool_output_filter_error_max_lines=100,
        tool_output_filter_generic_max_lines=80,
    )

    outcome = reduce_bash_output("git diff", output, "diff", config)

    assert outcome.strategy == "bash_diff_adaptive_fallback"
    assert outcome.fallback == "legacy"
    assert "fatal: malformed patch" in outcome.text
    assert "diff --git a/broken.py b/broken.py" in outcome.text


def test_adaptive_python_error_samples_quoted_frames_and_keeps_final_exception():
    output = "\n".join(
        [
            "Traceback (most recent call last):",
            *[
                line
                for index in range(80)
                for line in (
                    f'  File "recursive.py", line {index}, in recurse',
                    "    recurse()",
                )
            ],
            "RuntimeError: FINAL_RECURSION_FAILURE",
        ]
    )
    config = SimpleNamespace(
        tool_output_filter_adaptive_enabled=True,
        tool_output_filter_error_max_lines=18,
        tool_output_filter_test_max_lines=120,
        tool_output_filter_diff_max_lines=100,
        tool_output_filter_generic_max_lines=80,
        tool_output_filter_output_max_chars=50000,
    )

    outcome = reduce_bash_output("python recursive.py", output, "python_error", config)

    assert outcome.strategy == "bash_python_error_adaptive"
    assert 'File "recursive.py"' in outcome.text
    assert "RuntimeError: FINAL_RECURSION_FAILURE" in outcome.text
    assert outcome.filtered_lines <= 18
    assert outcome.reduction_metadata()["retained_line_ranges"][-1]["end"] == len(
        output.splitlines()
    )


def test_character_cap_keeps_chained_root_and_final_error_anchors():
    root_error = "ValueError: ROOT_CHAINED_CAUSE"
    output = "\n".join(
        [
            "Traceback (most recent call last):",
            '  File "/workspace/root.py", line 4, in root',
            "    raise ValueError('ROOT_CHAINED_CAUSE')",
            root_error,
            "The above exception was the direct cause of the following exception:",
            "Traceback (most recent call last):",
            *[
                line
                for index in range(32)
                for line in (
                    (
                        '  File "/workspace/'
                        + "very_long_segment/" * 8
                        + f'nested_{index}.py", line {index}, in recurse'
                    ),
                    "    recurse()",
                )
            ],
            "RuntimeError: " + "x" * 5000 + " FINAL_CHAINED_FAILURE",
        ]
    )
    config = SimpleNamespace(
        tool_output_filter_adaptive_enabled=True,
        tool_output_filter_error_max_lines=24,
        tool_output_filter_test_max_lines=120,
        tool_output_filter_diff_max_lines=100,
        tool_output_filter_generic_max_lines=80,
        tool_output_filter_output_max_chars=256,
    )

    outcome = reduce_bash_output("python chained.py", output, "python_error", config)

    assert len(outcome.text) <= 256
    assert root_error in outcome.text
    assert "RuntimeError:" in outcome.text
    assert "FINAL_CHAINED_FAILURE" in outcome.text
    assert outcome.filtered_lines <= 24
    assert outcome.reduction_metadata()["char_cap_applied"] is True


def test_output_character_cap_bounds_long_single_line_and_records_omission():
    service, _ = _service(
        tool_output_filter_adaptive_enabled=True,
        tool_output_filter_bash_min_chars=1,
        tool_output_filter_bash_min_lines=1,
        tool_output_filter_error_max_lines=20,
        tool_output_filter_output_max_chars=512,
    )
    output = "\n".join(
        [
            "Traceback (most recent call last):",
            '  File "worker.py", line 1, in run',
            "RuntimeError: " + "x" * 8000 + " FINAL_LONG_LINE_FAILURE",
        ]
    )

    result = service.filter_bash(
        {"command": "python worker.py", "output": output},
        _ctx(),
    )

    assert result["ok"] is True
    assert len(result["filtered_output"]) <= 512
    assert "RuntimeError:" in result["filtered_output"]
    assert "FINAL_LONG_LINE_FAILURE" in result["filtered_output"]
    assert result["reduction"]["char_cap_applied"] is True
    assert result["reduction"]["omitted_chars"] > 0
    assert result["reduction"]["output_max_chars"] == 512


def test_read_character_cap_bounds_long_single_line_before_tool_output_replacement():
    service, fs = _service(
        tool_output_filter_adaptive_enabled=True,
        tool_output_filter_read_min_chars=1,
        tool_output_filter_output_max_chars=512,
    )
    original = "prefix " + "x" * 8000 + " READ_LONG_LINE_TAIL"

    result = service.filter_read(
        {"relative_path": "generated.txt", "content": original},
        _ctx(),
    )

    delivered = result["filtered_output"]
    assert len(delivered) <= 512
    assert "READ_LONG_LINE_TAIL" in delivered
    assert result["reduction"]["char_cap_applied"] is True
    assert result["reduction"]["omitted_chars"] > 0
    assert next(iter(fs.nodes.values())).content == original
