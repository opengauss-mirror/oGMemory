"""Pure code-language detection and AST-first source chunking."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_CODE_FILE_SIZE = 10 * 1024 * 1024

LANGUAGE_BY_EXTENSION: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".kt": "kotlin",
    ".swift": "swift",
    ".scala": "scala",
    ".sql": "sql",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
}
SUPPORTED_CODE_EXTENSIONS = frozenset(LANGUAGE_BY_EXTENSION)
SUPPORTED_CODE_LANGUAGES = frozenset(LANGUAGE_BY_EXTENSION.values())


def validate_language_hint(language_hint: str | None) -> str | None:
    """Validate an explicit chunker language without accepting generic fallbacks."""

    if language_hint is None:
        return None
    if not isinstance(language_hint, str):
        raise ValueError("language hint must be a string")
    normalized = language_hint.strip().lower()
    if normalized not in SUPPORTED_CODE_LANGUAGES:
        raise ValueError("unsupported code language hint")
    return normalized


def detect_language(file_path: str | Path, language_hint: str | None = None) -> str:
    """Detect a source language from an explicit hint or file extension."""

    return validate_language_hint(language_hint) or LANGUAGE_BY_EXTENSION.get(
        Path(file_path).suffix.lower(), "code"
    )


def read_code_source(file_path: str | Path) -> str:
    """Read a bounded, non-binary code file with UTF-8 replacement semantics."""

    path = Path(file_path)
    if not path.is_file():
        raise ValueError("code source is not a regular file")
    with path.open("rb") as handle:
        data = handle.read(MAX_CODE_FILE_SIZE + 1)
    if len(data) > MAX_CODE_FILE_SIZE:
        raise ValueError("code source exceeds maximum file size")
    if b"\x00" in data:
        raise ValueError("binary code source is not supported")
    return data.decode("utf-8", errors="replace")


@dataclass
class CodeChunk:
    """A deterministic code chunk derived from a source file."""

    content: str
    start_line: int
    end_line: int
    symbol: str
    symbol_kind: str
    language: str
    chunk_hash: str
    signature: str = ""
    metadata: dict[str, Any] | None = None


class CodeFileChunker:
    """AST-first Python chunker with deterministic line-window fallback."""

    def __init__(
        self,
        target_chars: int = 2500,
        overlap_chars: int = 300,
        max_chars: int = 4000,
    ) -> None:
        if target_chars < 1 or overlap_chars < 0 or max_chars < 1:
            raise ValueError("chunk sizes must be positive")
        self.target_chars = target_chars
        self.overlap_chars = overlap_chars
        self.max_chars = max_chars

    def chunk(self, source: str, file_path: str | Path, language: str) -> list[CodeChunk]:
        if language == "python":
            chunks = self._chunk_python_ast(source, language)
            if chunks:
                return chunks
        return self._chunk_line_windows(
            source,
            language,
            symbol_kind="code",
        )

    def _chunk_python_ast(self, source: str, language: str) -> list[CodeChunk]:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        lines = source.splitlines()
        file_imports = _extract_python_imports(tree)
        chunks: list[CodeChunk] = []
        module_index = 1
        statements = iter(tree.body)
        statement = next(statements, None)
        for start, end in _python_module_ranges(tree.body):
            content = _slice_lines(lines, start, end)
            if not content.strip():
                continue
            symbol = f"module_{module_index}"
            graph = {
                **_empty_graph_metadata(),
                "symbol": symbol,
                "node_type": "Module",
                "imports": file_imports,
            }
            # Index assignment names while preserving source and module identity.
            defined = []
            while statement is not None and statement.lineno <= end:
                if start <= statement.lineno and isinstance(statement, (ast.Assign, ast.AnnAssign)):
                    targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                    defined.extend(n.id for target in targets for n in ast.walk(target) if isinstance(n, ast.Name))
                statement = next(statements, None)
            module_signature = ", ".join(dict.fromkeys(defined))
            if len(content) > self.max_chars:
                sub_chunks = self._chunk_line_windows(
                    content,
                    language,
                    target_chars=min(self.max_chars, self.target_chars),
                    overlap_chars=self.overlap_chars,
                    symbol_prefix=symbol,
                    base_start_line=start,
                    symbol_kind="module",
                )
                for sub_chunk in sub_chunks:
                    sub_chunk.metadata = {**sub_chunk.metadata, "graph": graph}
                    sub_chunk.signature = module_signature
                chunks.extend(sub_chunks)
                module_index += 1
                continue
            chunks.append(
                CodeChunk(
                    content=content,
                    start_line=start,
                    end_line=end,
                    symbol=symbol,
                    symbol_kind="module",
                    language=language,
                    chunk_hash=_hash_text(content),
                    signature=module_signature,
                    metadata={"graph": graph},
                )
            )
            module_index += 1

        class_graphs: dict[ast.AST, dict[str, Any]] = {}
        for node, start, end, symbol in _python_symbol_spans(tree.body, lines, self.max_chars):
            if start < 1 or end < start:
                continue
            content = _slice_lines(lines, start, end)
            if not content.strip():
                continue
            if isinstance(node, ast.AsyncFunctionDef):
                symbol_kind = "async_function"
            elif isinstance(node, ast.FunctionDef):
                symbol_kind = "function"
            else:
                symbol_kind = "type"
            signature = _build_python_signature(node)
            if node in class_graphs:
                # Whole-class relations belong to its first fragment only.
                graph = {**class_graphs[node], "calls": [], "contains": []}
            else:
                graph = _build_python_graph_metadata(node, file_imports)
                if isinstance(node, ast.ClassDef):
                    class_graphs[node] = graph
            graph = {**graph, "symbol": symbol}
            chunk_metadata = {"graph": graph}
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunk_metadata.update(symbol_start_line=start, symbol_end_line=end)
                body = node.body
                if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                    body = body[1:]
                if body:
                    chunk_metadata["implementation_start_line"] = min(
                        [body[0].lineno] + [d.lineno for d in getattr(body[0], "decorator_list", ())]
                    )
            if len(content) > self.max_chars:
                sub_chunks = self._chunk_line_windows(
                    content,
                    language,
                    target_chars=min(self.max_chars, self.target_chars),
                    overlap_chars=200,
                    symbol_prefix=symbol,
                    base_start_line=start,
                    symbol_kind=symbol_kind,
                )
                for index, sub_chunk in enumerate(sub_chunks):
                    sub_chunk.signature = signature
                    sub_chunk.metadata = {**sub_chunk.metadata, **chunk_metadata}
                    if isinstance(node, ast.ClassDef) and index:
                        sub_chunk.metadata["graph"] = {**graph, "calls": [], "contains": []}
                chunks.extend(sub_chunks)
                continue
            chunks.append(
                CodeChunk(
                    content=content,
                    start_line=start,
                    end_line=end,
                    symbol=symbol,
                    symbol_kind=symbol_kind,
                    language=language,
                    chunk_hash=_hash_text(content),
                    signature=signature,
                    metadata=chunk_metadata,
                )
            )
        # Python sorting is stable, so fragments from one logical line retain
        # their generation order when their source spans are identical.
        return sorted(chunks, key=lambda chunk: (chunk.start_line, chunk.end_line))

    def _chunk_line_windows(
        self,
        source: str,
        language: str,
        *,
        target_chars: int | None = None,
        overlap_chars: int | None = None,
        symbol_prefix: str = "chunk",
        base_start_line: int = 1,
        symbol_kind: str | None = None,
    ) -> list[CodeChunk]:
        lines = source.splitlines()
        if not lines:
            return []
        target = target_chars or self.target_chars
        window_limit = min(target, self.max_chars)
        overlap = self.overlap_chars if overlap_chars is None else overlap_chars
        chunks: list[CodeChunk] = []
        start_index = 0
        chunk_index = 1
        while start_index < len(lines):
            oversized_line = lines[start_index]
            if len(oversized_line) > self.max_chars:
                source_line = base_start_line + start_index
                for offset in range(0, len(oversized_line), self.max_chars):
                    content = oversized_line[offset : offset + self.max_chars]
                    chunks.append(
                        CodeChunk(
                            content=content,
                            start_line=source_line,
                            end_line=source_line,
                            symbol=f"{symbol_prefix}_{chunk_index}",
                            symbol_kind=symbol_kind or "code",
                            language=language,
                            chunk_hash=_hash_text(content),
                            metadata={"graph": _empty_graph_metadata(), "partial_source_line": True},
                        )
                    )
                    chunk_index += 1
                start_index += 1
                continue
            current_chars = 0
            end_index = start_index
            while end_index < len(lines):
                line_length = len(lines[end_index]) + 1
                if current_chars + line_length > window_limit and end_index > start_index:
                    break
                current_chars += line_length
                end_index += 1
                if current_chars >= self.max_chars:
                    break
            content = "\n".join(lines[start_index:end_index]).rstrip()
            if content.strip():
                start_line = base_start_line + start_index
                end_line = base_start_line + end_index - 1
                chunks.append(
                    CodeChunk(
                        content=content,
                        start_line=start_line,
                        end_line=end_line,
                        symbol=f"{symbol_prefix}_{chunk_index}",
                        symbol_kind=symbol_kind or "code",
                        language=language,
                        chunk_hash=_hash_text(content),
                        metadata={"graph": _empty_graph_metadata()},
                    )
                )
                chunk_index += 1
            if end_index >= len(lines):
                break
            overlap_lines = _overlap_line_count(lines, end_index, overlap)
            start_index = max(start_index + 1, end_index - overlap_lines)
        return chunks


def _is_python_symbol_node(node: ast.AST) -> bool:
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))


def _python_symbol_spans(nodes, lines, max_chars, prefix=""):
    """Split oversized classes at member boundaries, preserving all source lines."""
    for node in nodes:
        if not _is_python_symbol_node(node):
            continue
        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        end = node.end_lineno
        symbol = prefix + node.name
        members = [child for child in node.body if _is_python_symbol_node(child)]
        if not isinstance(node, ast.ClassDef) or not members or len(_slice_lines(lines, start, end)) <= max_chars:
            yield node, start, end, symbol
            continue
        cursor = start
        part = 0
        for child in members:
            child_start = min([child.lineno] + [d.lineno for d in child.decorator_list])
            if cursor < child_start:
                part += 1
                yield node, cursor, child_start - 1, f"{symbol}.__class_body_{part}"
            yield from _python_symbol_spans([child], lines, max_chars, symbol + ".")
            cursor = child.end_lineno + 1
        if cursor <= end:
            yield node, cursor, end, f"{symbol}.__class_body_{part + 1}"


def _python_module_ranges(nodes: list[ast.stmt]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    current_start: int | None = None
    current_end: int | None = None
    for node in nodes:
        if _is_python_symbol_node(node):
            if current_start is not None and current_end is not None:
                ranges.append((current_start, current_end))
            current_start = None
            current_end = None
            continue
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", start)
        if not isinstance(start, int) or not isinstance(end, int) or end < start:
            continue
        if current_start is None:
            current_start = start
            current_end = end
        elif current_end is not None and start <= current_end + 1:
            current_end = max(current_end, end)
        else:
            if current_end is None:
                current_end = end
            ranges.append((current_start, current_end))
            current_start = start
            current_end = end
    if current_start is not None and current_end is not None:
        ranges.append((current_start, current_end))
    return ranges


def _build_python_signature(node: ast.AST) -> str:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        arguments = [argument.arg for argument in node.args.args]
        return f"{node.name}({', '.join(arguments)})"
    if isinstance(node, ast.ClassDef):
        return f"class {node.name}"
    return ""


def _extract_python_imports(tree: ast.AST) -> list[str]:
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = "." * int(node.level or 0) + (node.module or "")
            for alias in node.names:
                if alias.name == "*":
                    imports.append(f"{module}.*" if module else "*")
                else:
                    imports.append(f"{module}.{alias.name}" if module else alias.name)
    return _dedupe_keep_order(imports)


def _build_python_graph_metadata(node: ast.AST, file_imports: list[str]) -> dict[str, Any]:
    extends: list[str] = []
    contains: list[str] = []
    if isinstance(node, ast.ClassDef):
        extends = [value for value in (_expr_to_dotted(base) for base in node.bases) if value]
        contains = [
            getattr(child, "name", "")
            for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
    return {
        "symbol": getattr(node, "name", ""),
        "node_type": type(node).__name__,
        "calls": _extract_python_calls(node),
        "imports": file_imports,
        "extends": _dedupe_keep_order(extends),
        "contains": _dedupe_keep_order([name for name in contains if name]),
        "relations": [],
    }


def _extract_python_calls(node: ast.AST) -> list[str]:
    calls: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            name = _expr_to_dotted(child.func)
            if name:
                calls.append(name)
    return _dedupe_keep_order(calls)


def _expr_to_dotted(expr: ast.AST) -> str:
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        base = _expr_to_dotted(expr.value)
        return f"{base}.{expr.attr}" if base else expr.attr
    if isinstance(expr, ast.Call):
        return _expr_to_dotted(expr.func)
    if isinstance(expr, ast.Subscript):
        return _expr_to_dotted(expr.value)
    return ""


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _overlap_line_count(lines: list[str], end_index: int, overlap_chars: int) -> int:
    chars = 0
    count = 0
    cursor = end_index - 1
    while cursor >= 0 and chars < overlap_chars:
        chars += len(lines[cursor]) + 1
        count += 1
        cursor -= 1
    return count


def _slice_lines(lines: list[str], start_line: int, end_line: int) -> str:
    return "\n".join(lines[max(0, start_line - 1) : min(len(lines), end_line)])


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _empty_graph_metadata() -> dict[str, list[Any]]:
    return {
        "calls": [],
        "imports": [],
        "extends": [],
        "contains": [],
        "relations": [],
    }
