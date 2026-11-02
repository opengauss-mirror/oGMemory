from __future__ import annotations

import hashlib
import ast
from collections import Counter

import pytest

from extraction.code_processor import (
    CodeFileChunker,
    detect_language,
    read_code_source,
    validate_language_hint,
)


def test_detect_language_from_extension():
    assert detect_language("service.py") == "python"
    assert detect_language("index.ts") == "typescript"
    assert detect_language("main.go") == "go"
    assert detect_language("setup.sh") == "shell"


def test_large_class_preserves_method_boundaries_and_decorators():
    source = "class Container:\n" + "    # padding\n" * 40 + (
        "    @staticmethod\n"
        "    def target():\n"
        '        """Target documentation."""\n'
        "        return important_value\n\n"
        "    async def other(self):\n"
        "        return 2\n"
    )
    chunks = CodeFileChunker(max_chars=400).chunk(source, "sample.py", "python")
    target = next(c for c in chunks if c.symbol == "Container.target")
    assert "@staticmethod" in target.content and "return important_value" in target.content
    assert "other" not in target.content
    assert target.signature == "target()"
    assert target.metadata["implementation_start_line"] == target.end_line
    assert any(c.symbol == "Container.other" and c.symbol_kind == "async_function" for c in chunks)
    assert all(len(c.content) <= 400 for c in chunks)
    covered = {i for c in chunks for i in range(c.start_line, c.end_line + 1)}
    assert {i for i, line in enumerate(source.splitlines(), 1) if line.strip()} <= covered
    for c in chunks:
        assert c.content.strip() == "\n".join(source.splitlines()[c.start_line - 1:c.end_line]).strip()


def test_module_assignment_names_are_searchable_without_changing_source():
    source = "a1, a2 = make_pair()\na21: object = make_child(a1)\n"
    chunks = CodeFileChunker().chunk(source, "sample.py", "python")
    assert chunks[0].signature == "a1, a2, a21"
    assert chunks[0].content == source.rstrip()


def test_interleaved_module_signatures_use_linear_top_level_traversal(monkeypatch):
    source = "".join(
        f"a{i}, b{i} = pair()\na{i}: int = 1\n\ndef f{i}(): return 0\n\n"
        for i in range(40)
    )
    tree = ast.parse(source)

    class CountedNodes(list):
        visits = 0

        def __iter__(self):
            for node in super().__iter__():
                self.visits += 1
                yield node

    tree.body = CountedNodes(tree.body)
    monkeypatch.setattr(ast, "parse", lambda *a, **kw: tree)
    chunks = CodeFileChunker().chunk(source, "sample.py", "python")
    modules = [c for c in chunks if c.symbol_kind == "module"]
    assert [c.signature for c in modules] == [f"a{i}, b{i}" for i in range(40)]
    assert all(c.content == "\n".join(source.splitlines()[c.start_line - 1:c.end_line]) for c in chunks)
    assert tree.body.visits <= 8 * len(tree.body)


@pytest.mark.parametrize("gap", ["", "    # retained comment"])
def test_split_class_extracts_graph_once_and_keeps_relations_on_one_chunk(monkeypatch, gap):
    from extraction import code_processor

    source = "class Large(Base):\n" + "    # header\n" * 60 + f"\n{gap}\n".join(
        f"    def method_{i}(self):\n        return call_{i}()\n" for i in range(80)
    )
    original = code_processor._build_python_graph_metadata
    calls = Counter()

    def graph(node, imports):
        calls[type(node)] += 1
        return original(node, imports)

    monkeypatch.setattr(code_processor, "_build_python_graph_metadata", graph)
    chunks = CodeFileChunker(max_chars=400).chunk(source, "large.py", "python")
    assert calls[ast.ClassDef] == 1
    assert calls[ast.FunctionDef] == 80
    class_chunks = [c for c in chunks if c.symbol_kind == "type"]
    assert sum(len(c.metadata["graph"]["contains"]) for c in class_chunks) == 80
    assert sum(len(c.metadata["graph"]["calls"]) for c in class_chunks) == 80
    for i in range(80):
        method = next(c for c in chunks if c.symbol == f"Large.method_{i}")
        assert method.metadata["graph"]["symbol"] == method.symbol
        assert method.metadata["graph"]["calls"] == [f"call_{i}"]
    covered = {i for c in chunks for i in range(c.start_line, c.end_line + 1)}
    assert {i for i, line in enumerate(source.splitlines(), 1) if line.strip()} <= covered


def test_explicit_language_hint_overrides_extension_and_is_validated():
    assert detect_language("Dockerfile", "shell") == "shell"
    assert detect_language("template.txt", "python") == "python"
    assert validate_language_hint(" TypeScript ") == "typescript"

    with pytest.raises(ValueError, match="unsupported code language hint"):
        detect_language("template.txt", "unknown")


def test_python_ast_chunks_symbols_graph_and_full_hash():
    source = (
        "import os\n\n"
        "def helper(value):\n"
        "    return os.path.join(value, 'x')\n\n"
        "class Runner(BaseRunner):\n"
        "    def run(self):\n"
        "        return helper('a')\n"
    )

    chunks = CodeFileChunker().chunk(source, "service.py", "python")
    by_symbol = {chunk.symbol: chunk for chunk in chunks}

    assert {"module_1", "helper", "Runner"} <= set(by_symbol)
    assert by_symbol["helper"].signature == "helper(value)"
    assert "os.path.join" in by_symbol["helper"].metadata["graph"]["calls"]
    assert "BaseRunner" in by_symbol["Runner"].metadata["graph"]["extends"]
    assert "run" in by_symbol["Runner"].metadata["graph"]["contains"]
    assert all(len(chunk.chunk_hash) == 64 for chunk in chunks)


def test_large_python_module_range_uses_overlapping_line_windows():
    entries = [
        f"    'lang_{index:03d}': {{'name': 'Language {index:03d}', 'bidi': False}},"
        for index in range(240)
    ]
    source = (
        "from django.utils.translation import gettext_noop as _\n\n"
        "LANG_INFO = {\n" + "\n".join(entries) + "\n}\n"
    )
    source_lines = source.splitlines()
    chunker = CodeFileChunker()

    chunks = chunker.chunk(source, "django/conf/locale/__init__.py", "python")
    module_chunks = [chunk for chunk in chunks if chunk.metadata["graph"]["symbol"] == "module_2"]
    repeated_chunks = [
        chunk
        for chunk in chunker.chunk(source, "django/conf/locale/__init__.py", "python")
        if chunk.metadata["graph"]["symbol"] == "module_2"
    ]

    assert len(source) > 12_000
    assert len(module_chunks) > 1
    assert [chunk.symbol for chunk in module_chunks] == [
        f"module_2_{index}" for index in range(1, len(module_chunks) + 1)
    ]
    assert [
        (chunk.symbol, chunk.start_line, chunk.end_line, chunk.chunk_hash)
        for chunk in module_chunks
    ] == [
        (chunk.symbol, chunk.start_line, chunk.end_line, chunk.chunk_hash)
        for chunk in repeated_chunks
    ]
    assert module_chunks[0].start_line == 3
    assert module_chunks[-1].end_line == len(source_lines)
    assert all(len(chunk.content) <= chunker.max_chars for chunk in module_chunks)

    covered_lines: set[int] = set()
    for chunk in module_chunks:
        expected_content = "\n".join(source_lines[chunk.start_line - 1 : chunk.end_line]).rstrip()
        covered_lines.update(range(chunk.start_line, chunk.end_line + 1))
        assert chunk.content == expected_content
        assert chunk.chunk_hash == hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()
        assert chunk.symbol_kind == "module"
        assert chunk.language == "python"
        assert chunk.metadata["graph"]["symbol"] == "module_2"
        assert chunk.metadata["graph"]["node_type"] == "Module"
        assert chunk.metadata["graph"]["imports"] == ["django.utils.translation.gettext_noop"]

    assert covered_lines == set(range(3, len(source_lines) + 1))
    for previous, current in zip(module_chunks, module_chunks[1:]):
        assert previous.start_line < current.start_line <= previous.end_line
        assert previous.end_line < current.end_line
        overlap_size = sum(
            len(line) + 1 for line in source_lines[current.start_line - 1 : previous.end_line]
        )
        assert overlap_size >= chunker.overlap_chars


def test_minified_single_line_respects_hard_chunk_limit():
    source = "const payload = '" + "x" * 1024 + "';"
    chunker = CodeFileChunker(target_chars=128, overlap_chars=32, max_chars=64)

    chunks = chunker.chunk(source, "generated.js", "javascript")

    assert len(chunks) > 1
    assert all(chunk.content for chunk in chunks)
    assert all(len(chunk.content) <= chunker.max_chars for chunk in chunks)
    assert "".join(chunk.content for chunk in chunks) == source
    assert {(chunk.start_line, chunk.end_line) for chunk in chunks} == {(1, 1)}
    assert [chunk.symbol for chunk in chunks] == [
        f"chunk_{index}" for index in range(1, len(chunks) + 1)
    ]


def test_python_ast_oversized_symbol_preserves_slice_order_and_line_provenance():
    oversized_line = "def generated(): return " + repr("y" * 1024)
    source = "header = 1\n\n" + oversized_line
    chunker = CodeFileChunker(target_chars=128, overlap_chars=32, max_chars=64)

    chunks = chunker.chunk(source, "generated.py", "python")
    symbol_chunks = [chunk for chunk in chunks if chunk.symbol.startswith("generated_")]

    assert len(symbol_chunks) > 10
    assert all(chunk.content for chunk in symbol_chunks)
    assert all(len(chunk.content) <= chunker.max_chars for chunk in chunks)
    assert "".join(chunk.content for chunk in symbol_chunks) == oversized_line
    assert {(chunk.start_line, chunk.end_line) for chunk in symbol_chunks} == {(3, 3)}
    assert [chunk.symbol for chunk in symbol_chunks] == [
        f"generated_{index}" for index in range(1, len(symbol_chunks) + 1)
    ]
    assert all(chunk.signature == "generated()" for chunk in symbol_chunks)
    assert all(chunk.metadata["graph"]["symbol"] == "generated" for chunk in symbol_chunks)


def test_line_window_fallback_is_deterministic_and_overlapping():
    source = "\n".join(f"line {index}" for index in range(1, 9))
    chunker = CodeFileChunker(target_chars=28, overlap_chars=7, max_chars=128)

    chunks = chunker.chunk(source, "service.ts", "typescript")

    assert [(chunk.start_line, chunk.end_line) for chunk in chunks] == [
        (1, 4),
        (4, 7),
        (7, 8),
    ]
    assert [chunk.chunk_hash for chunk in chunks] == [
        chunk.chunk_hash for chunk in chunker.chunk(source, "service.ts", "typescript")
    ]


def test_python_syntax_error_uses_line_window_fallback():
    chunks = CodeFileChunker().chunk("def broken(:\n    pass\n", "broken.py", "python")

    assert len(chunks) == 1
    assert chunks[0].symbol_kind == "code"
    assert chunks[0].metadata["graph"]["calls"] == []


def test_read_code_source_replaces_invalid_utf8_and_rejects_binary(tmp_path):
    text = tmp_path / "text.py"
    text.write_bytes(b"print('ok')\n\xff")
    binary = tmp_path / "binary.py"
    binary.write_bytes(b"print('prefix')\n" + b"x" * 9000 + b"\x00late-nul")

    assert "\ufffd" in read_code_source(text)
    try:
        read_code_source(binary)
    except ValueError as exc:
        assert "binary" in str(exc)
    else:
        raise AssertionError("binary source must be rejected")


@pytest.mark.parametrize("async_prefix", ["", "async "])
def test_split_function_preserves_symbol_span_and_partial_line_metadata(async_prefix):
    source = (f"@decorator\n{async_prefix}def compute(value):\n    \"\"\"Documentation.\"\"\"\n" +
              "    value += 1\n" * 35 + "    return '" + "x" * 500 + "'\n")
    chunks = CodeFileChunker(max_chars=150, target_chars=120).chunk(source, "sample.py", "python")
    assert len(chunks) > 3
    assert any(c.metadata.get("partial_source_line") for c in chunks)
    for chunk in chunks:
        assert chunk.metadata["symbol_start_line"] == 1
        assert chunk.metadata["symbol_end_line"] == len(source.splitlines())
        assert chunk.metadata["implementation_start_line"] == 4
        if not chunk.metadata.get("partial_source_line"):
            assert chunk.content == "\n".join(source.splitlines()[chunk.start_line - 1:chunk.end_line]).rstrip()
