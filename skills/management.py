# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Validation, normalization, and serialization for managed Skill packages."""

from __future__ import annotations

import base64
import codecs
import hashlib
import io
import mimetypes
import re
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

import yaml

from skills.artifact import normalize_skill_name
from skills.sanitizer import sanitize_text

MAX_SKILL_BYTES = 256 * 1024
MAX_DESCRIPTION_LENGTH = 1024
MAX_PACKAGE_FILES = 256
MAX_PACKAGE_FILE_BYTES = 5 * 1024 * 1024
MAX_PACKAGE_BYTES = 20 * 1024 * 1024
MAX_SKILL_ARCHIVE_BYTES = 20 * 1024 * 1024
MAX_PACKAGE_PATH_BYTES = 1024
MAX_MEDIA_TYPE_LENGTH = 255
MAX_FRONTMATTER_DEPTH = 32
MAX_FRONTMATTER_NODES = 10_000
_MANAGED_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_VALIDATION_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class PackageFile:
    path: str
    data: bytes
    media_type: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


@dataclass(frozen=True)
class ManagedSkill:
    name: str
    description: str
    content: str
    files: tuple[PackageFile, ...] = ()

    @property
    def auxiliary_files(self) -> int:
        return len(self.files)

    @property
    def total_bytes(self) -> int:
        return len(self.content.encode("utf-8")) + sum(len(item.data) for item in self.files)

    def serialized_files(self) -> list[dict[str, Any]]:
        return [
            {
                "path": item.path,
                "data": base64.b64encode(item.data).decode("ascii"),
                "encoding": "base64",
                "media_type": item.media_type,
                "size": len(item.data),
                "sha256": item.sha256,
            }
            for item in self.files
        ]


def decode_skill_archive_base64(value: Any) -> bytes:
    """Decode one base64-encoded Skill ZIP payload."""

    if not isinstance(value, str) or not value:
        raise ValueError("archive_base64 must be a non-empty string")
    try:
        data = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("archive_base64 must contain valid base64") from exc
    if len(data) > MAX_SKILL_ARCHIVE_BYTES:
        raise ValueError(f"skill archive exceeds {MAX_SKILL_ARCHIVE_BYTES} bytes")
    return data


def load_managed_skill_path(value: Any) -> str | dict[str, Any] | bytes:
    """Load a Skill from a server-local SKILL.md, ZIP, or directory."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("path must be a non-empty string")
    source = Path(value).expanduser().resolve(strict=True)
    if source.is_file():
        size = source.stat().st_size
        if source.suffix.casefold() == ".zip":
            if size > MAX_SKILL_ARCHIVE_BYTES:
                raise ValueError(f"skill archive exceeds {MAX_SKILL_ARCHIVE_BYTES} bytes")
            return source.read_bytes()
        if source.name.casefold() == "skill.md":
            if size > MAX_SKILL_BYTES:
                raise ValueError(f"SKILL.md exceeds {MAX_SKILL_BYTES} bytes")
            try:
                return source.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("SKILL.md must be UTF-8 text") from exc
        raise ValueError("path must reference SKILL.md, a ZIP file, or a Skill directory")
    if not source.is_dir():
        raise ValueError("path must reference SKILL.md, a ZIP file, or a Skill directory")

    entries = sorted(item for item in source.rglob("*") if item.is_file())
    if len(entries) > MAX_PACKAGE_FILES:
        raise ValueError(f"skill package exceeds {MAX_PACKAGE_FILES} files")
    records: list[dict[str, str]] = []
    total = 0
    for item in entries:
        if item.is_symlink():
            raise ValueError(f"symbolic link is not supported: {item}")
        path = _safe_relative_path(item.relative_to(source).as_posix())
        size = item.stat().st_size
        if size > MAX_PACKAGE_FILE_BYTES and path.casefold() != "skill.md":
            raise ValueError(f"package file exceeds {MAX_PACKAGE_FILE_BYTES} bytes: {path}")
        total += size
        if total > MAX_PACKAGE_BYTES:
            raise ValueError(f"skill package exceeds {MAX_PACKAGE_BYTES} bytes")
        records.append({
            "path": path,
            "content_base64": base64.b64encode(item.read_bytes()).decode("ascii"),
            "media_type": _normalize_media_type(None, path),
        })
    return {"files": records}


def validate_managed_skill_name(name: Any) -> str:
    """Validate managed package names accepted by oGMemory."""

    if not isinstance(name, str) or not _MANAGED_NAME.fullmatch(name):
        raise ValueError(f"invalid skill name: {name!r}")
    return name


def parse_managed_skill(data: Any) -> ManagedSkill:
    """Accept structured, MCP-tool, SKILL.md, or complete package input."""

    if isinstance(data, bytes):
        return parse_skill_archive(data)
    if isinstance(data, str):
        return _parse_markdown(data)
    if not isinstance(data, dict):
        raise ValueError("skill data must be a SKILL.md string or object")
    if "files" in data:
        return _parse_package(data["files"])
    if "inputSchema" in data:
        return _parse_markdown(_render_mcp_tool(data))
    return _parse_structured(data)


def parse_skill_archive(data: bytes) -> ManagedSkill:
    """Parse a ZIP archive containing one complete Skill package."""

    records, _ = _read_skill_archive(data)
    return _parse_package(records)


def validate_skill_format(
    data: Any,
    *,
    strict: bool = False,
    skill_dir_name: str | None = None,
    source_path: str | None = None,
) -> dict[str, Any]:
    """Validate a Skill using the shared Agent Skills format rules."""

    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    try:
        if isinstance(data, bytes):
            data, archive_dir_name = _skill_markdown_from_archive(data)
            skill_dir_name = skill_dir_name or archive_dir_name
        parsed = _parse_validation_input(data)
    except Exception as exc:
        return {
            "valid": False,
            "strict": strict,
            "errors": [_validation_issue("yaml_format", str(exc), "data")],
            "warnings": [],
            "source_path": source_path or "",
        }

    name = parsed.get("name")
    description = parsed.get("description")
    content = parsed.get("content") or ""
    if not isinstance(name, str) or not name.strip():
        errors.append(_validation_issue("name_required", "name is required", "name"))
    if not isinstance(description, str) or not description.strip():
        errors.append(_validation_issue(
            "description_required", "description is required", "description"
        ))

    def add_mode_issue(rule: str, message: str, field: str) -> None:
        issue = _validation_issue(rule, message, field)
        (errors if strict else warnings).append(issue)

    if isinstance(name, str) and name.strip():
        normalized_name = name.strip()
        normalized_dir_name = (skill_dir_name or "").strip()
        if normalized_dir_name and normalized_name != normalized_dir_name:
            add_mode_issue(
                "name_matches_directory",
                f"name '{normalized_name}' does not match directory name '{normalized_dir_name}'",
                "name",
            )
        if len(normalized_name) > 64:
            add_mode_issue("name_max_length", "name must not exceed 64 characters", "name")
        if not _VALIDATION_NAME.fullmatch(normalized_name):
            add_mode_issue(
                "name_allowed_characters",
                "name may only contain letters, numbers, underscores, and hyphens",
                "name",
            )
    if isinstance(description, str) and len(description) > 1024:
        add_mode_issue(
            "description_max_length",
            "description must not exceed 1024 characters",
            "description",
        )

    body_lines = len(content.splitlines())
    if strict and body_lines > 500:
        warnings.append(_validation_issue(
            "body_max_lines", "SKILL.md body exceeds 500 lines", "content"
        ))
    return {
        "valid": not errors,
        "strict": strict,
        "name": name or "",
        "description": description or "",
        "tags": parsed.get("tags") or [],
        "allowed_tools": parsed.get("allowed_tools") or [],
        "body_lines": body_lines,
        "source_path": source_path or "",
        "skill_dir_name": skill_dir_name or "",
        "errors": errors,
        "warnings": warnings,
    }


def _read_skill_archive(data: bytes) -> tuple[list[dict[str, str]], str]:
    if len(data) > MAX_SKILL_ARCHIVE_BYTES:
        raise ValueError(
            f"skill archive exceeds {MAX_SKILL_ARCHIVE_BYTES} bytes"
        )
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = [item for item in archive.infolist() if not item.is_dir()]
            if not entries:
                raise ValueError("skill archive is empty")
            if len(entries) > MAX_PACKAGE_FILES:
                raise ValueError(f"skill package exceeds {MAX_PACKAGE_FILES} files")

            total = 0
            records: list[dict[str, str]] = []
            normalized_paths: set[str] = set()
            for item in entries:
                path = _safe_relative_path(item.filename)
                path_key = path.casefold()
                if path_key in normalized_paths:
                    raise ValueError(f"duplicate package path: {path}")
                normalized_paths.add(path_key)
                if item.flag_bits & 0x1:
                    raise ValueError(f"encrypted ZIP entry is not supported: {path}")
                if stat.S_ISLNK(item.external_attr >> 16):
                    raise ValueError(f"symbolic link ZIP entry is not supported: {path}")
                total += item.file_size
                if total > MAX_PACKAGE_BYTES:
                    raise ValueError(f"skill package exceeds {MAX_PACKAGE_BYTES} bytes")
                records.append({
                    "path": path,
                    "content_base64": base64.b64encode(archive.read(item)).decode("ascii"),
                })
    except (zipfile.BadZipFile, zipfile.LargeZipFile, NotImplementedError, RuntimeError) as exc:
        raise ValueError("invalid or unsupported skill ZIP archive") from exc
    directory_name = ""
    if not any(record["path"].casefold() == "skill.md" for record in records):
        roots = {PurePosixPath(record["path"]).parts[0] for record in records}
        if len(roots) == 1:
            candidate = next(iter(roots))
            expected = f"{candidate}/SKILL.md".casefold()
            if any(record["path"].casefold() == expected for record in records):
                directory_name = candidate
    return records, directory_name


def _skill_markdown_from_archive(data: bytes) -> tuple[str, str]:
    records, directory_name = _read_skill_archive(data)
    expected = f"{directory_name}/SKILL.md" if directory_name else "SKILL.md"
    record = next(
        (item for item in records if item["path"].casefold() == expected.casefold()),
        None,
    )
    if record is None:
        raise ValueError("skill package must contain SKILL.md at its root")
    try:
        content = base64.b64decode(record["content_base64"], validate=True).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("SKILL.md must be UTF-8 text") from exc
    return content, directory_name


def _validation_issue(rule: str, message: str, field: str = "") -> dict[str, str]:
    issue = {"rule": rule, "message": message}
    if field:
        issue["field"] = field
    return issue


def _parse_validation_input(data: Any) -> dict[str, Any]:
    if isinstance(data, dict) and "files" in data:
        skill = parse_managed_skill(data)
        parsed = {
            "name": skill.name,
            "description": skill.description,
            "content": skill.content,
        }
    elif isinstance(data, dict):
        parsed = dict(data)
        parsed["content"] = parsed.get("content") or ""
    elif isinstance(data, str):
        content = data.replace("\r\n", "\n").replace("\r", "\n").removeprefix("\ufeff")
        lines = content.splitlines()
        if not lines or lines[0].strip() != "---":
            raise ValueError("SKILL.md must have YAML frontmatter")
        try:
            end = next(index for index, line in enumerate(lines[1:], 1) if line.strip() == "---")
        except StopIteration as exc:
            raise ValueError("SKILL.md must have closing YAML frontmatter delimiter") from exc
        try:
            metadata = yaml.safe_load("\n".join(lines[1:end]))
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML frontmatter: {exc}") from exc
        if not isinstance(metadata, dict):
            raise ValueError("Invalid YAML frontmatter")
        parsed = dict(metadata)
        parsed["content"] = "\n".join(lines[end + 1:]).strip()
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

    allowed_tools = parsed.get("allowed_tools") or parsed.get("allowed-tools")
    if allowed_tools is not None:
        parsed["allowed_tools"] = (
            allowed_tools if isinstance(allowed_tools, list) else [allowed_tools]
        )
    parsed.pop("allowed-tools", None)
    tags = parsed.get("tags")
    if tags is not None and not isinstance(tags, list):
        parsed["tags"] = [tags]
    return parsed


def deserialize_package_file(record: dict[str, Any]) -> PackageFile:
    path = _safe_relative_path(record.get("path"))
    if record.get("encoding") != "base64" or not isinstance(record.get("data"), str):
        raise ValueError(f"invalid stored package file: {path}")
    try:
        data = base64.b64decode(record["data"], validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid base64 package file: {path}") from exc
    if len(data) != record.get("size") or hashlib.sha256(data).hexdigest() != record.get("sha256"):
        raise ValueError(f"corrupt stored package file: {path}")
    return PackageFile(path, data, _normalize_media_type(record.get("media_type"), path))


def package_file_manifest(skill: ManagedSkill, *, source: str) -> list[dict[str, Any]]:
    rows = [
        {
            "name": "SKILL.md",
            "path": "SKILL.md",
            "uri": package_file_uri(skill.name, "SKILL.md"),
            "is_dir": False,
            "kind": "skill",
            "source": source,
        }
    ]
    rows.extend(
        {
            "name": PurePosixPath(item.path).name,
            "path": item.path,
            "uri": package_file_uri(skill.name, item.path),
            "is_dir": False,
            "kind": _file_kind(item.path),
            "source": source,
            "size": len(item.data),
            "sha256": item.sha256,
        }
        for item in skill.files
    )
    return rows


def package_file_uri(skill_name: str, path: str) -> str:
    """Return a canonical public URI for one managed package file."""

    name = validate_managed_skill_name(skill_name)
    safe_path = _safe_relative_path(path)
    return f"ogmemory://skills/imported/{name}/{quote(safe_path, safe='/')}"


def _parse_package(raw_files: Any) -> ManagedSkill:
    records = _normalize_file_records(raw_files)
    if not records:
        raise ValueError("skill package is empty")
    if len(records) > MAX_PACKAGE_FILES:
        raise ValueError(f"skill package exceeds {MAX_PACKAGE_FILES} files")
    normalized: dict[str, tuple[bytes, str]] = {}
    normalized_paths: set[str] = set()
    raw_total = 0
    for record in records:
        path = _safe_relative_path(record.get("path"))
        path_key = path.casefold()
        if path_key in normalized_paths:
            raise ValueError(f"duplicate package path: {path}")
        normalized_paths.add(path_key)
        data, media_type = _decode_input_file(record, path)
        if len(data) > MAX_PACKAGE_FILE_BYTES and path.casefold() != "skill.md":
            raise ValueError(f"package file exceeds {MAX_PACKAGE_FILE_BYTES} bytes: {path}")
        raw_total += len(data)
        if raw_total > MAX_PACKAGE_BYTES:
            raise ValueError(f"skill package exceeds {MAX_PACKAGE_BYTES} bytes")
        normalized[path] = (data, media_type)
    normalized = _strip_wrapper_directory(normalized)
    skill_entry = next((path for path in normalized if path.casefold() == "skill.md"), None)
    if skill_entry is None:
        raise ValueError("skill package must contain SKILL.md at its root")
    try:
        markdown = normalized.pop(skill_entry)[0].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("SKILL.md must be UTF-8 text") from exc
    parsed = _parse_markdown(markdown)
    auxiliary: list[PackageFile] = []
    total = len(parsed.content.encode("utf-8"))
    for path, (file_data, media_type) in sorted(normalized.items()):
        if len(file_data) > MAX_PACKAGE_FILE_BYTES:
            raise ValueError(f"package file exceeds {MAX_PACKAGE_FILE_BYTES} bytes: {path}")
        sanitized = _sanitize_file(file_data)
        total += len(sanitized)
        if total > MAX_PACKAGE_BYTES:
            raise ValueError(f"skill package exceeds {MAX_PACKAGE_BYTES} bytes")
        auxiliary.append(PackageFile(path, sanitized, media_type))
    if len(auxiliary) + 1 > MAX_PACKAGE_FILES:
        raise ValueError(f"skill package exceeds {MAX_PACKAGE_FILES} files")
    return ManagedSkill(parsed.name, parsed.description, parsed.content, tuple(auxiliary))


def _parse_structured(data: dict[str, Any]) -> ManagedSkill:
    name = validate_managed_skill_name(data.get("name"))
    description = sanitize_text(_validate_description(data.get("description")))
    body = data.get("content")
    if not isinstance(body, str):
        raise ValueError("skill content must be a string")
    if len(body.encode("utf-8")) > MAX_SKILL_BYTES:
        raise ValueError(f"SKILL.md exceeds {MAX_SKILL_BYTES} bytes")
    if body.lstrip().startswith("---"):
        parsed = _parse_markdown(body)
        if parsed.name != name:
            raise ValueError("skill content name must match the requested name")
        if parsed.description != description:
            raise ValueError("skill content description must match the requested description")
        return parsed
    metadata: dict[str, Any] = {"name": name, "description": description}
    for source, target in (
        ("allowed_tools", "allowed-tools"),
        ("allowed-tools", "allowed-tools"),
        ("tags", "tags"),
    ):
        if source in data:
            metadata[target] = data[source]
    frontmatter = yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False).strip()
    return _parse_markdown(f"---\n{frontmatter}\n---\n\n{body.strip()}\n")


def _parse_markdown(content: str) -> ManagedSkill:
    if len(content.encode("utf-8")) > MAX_SKILL_BYTES:
        raise ValueError(f"SKILL.md exceeds {MAX_SKILL_BYTES} bytes")
    content = content.replace("\r\n", "\n").replace("\r", "\n").removeprefix("\ufeff").strip()
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("SKILL.md must start with YAML frontmatter")
    try:
        end = next(
            index for index, line in enumerate(lines[1:], start=1)
            if line.strip() == "---"
        )
    except StopIteration as exc:
        raise ValueError("SKILL.md frontmatter is not closed") from exc
    try:
        metadata = yaml.safe_load("\n".join(lines[1:end]))
    except (yaml.YAMLError, RecursionError) as exc:
        raise ValueError("SKILL.md frontmatter is invalid YAML") from exc
    if not isinstance(metadata, dict):
        raise ValueError("SKILL.md frontmatter must be an object")
    _validate_yaml_shape(metadata)
    metadata = _sanitize_yaml_value(metadata)
    name = validate_managed_skill_name(metadata.get("name"))
    description = _validate_description(metadata.get("description"))
    body = "\n".join(lines[end + 1 :]).strip()
    if not body:
        raise ValueError("SKILL.md body must not be empty")
    frontmatter = yaml.safe_dump(
        metadata, allow_unicode=True, sort_keys=False
    ).strip()
    sanitized = f"---\n{frontmatter}\n---\n\n{sanitize_text(body)}\n"
    return ManagedSkill(name, description, sanitized)


def _sanitize_yaml_value(value: Any) -> Any:
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, list):
        return [_sanitize_yaml_value(item) for item in value]
    if isinstance(value, dict):
        return {
            sanitize_text(key) if isinstance(key, str) else key: _sanitize_yaml_value(item)
            for key, item in value.items()
        }
    return value


def _validate_yaml_shape(value: Any) -> None:
    stack = [(value, 0, frozenset())]
    nodes = 0
    while stack:
        item, depth, ancestors = stack.pop()
        nodes += 1
        if depth > MAX_FRONTMATTER_DEPTH or nodes > MAX_FRONTMATTER_NODES:
            raise ValueError("SKILL.md frontmatter is too complex")
        if not isinstance(item, dict | list):
            continue
        identity = id(item)
        if identity in ancestors:
            raise ValueError("SKILL.md frontmatter contains a recursive YAML alias")
        child_ancestors = ancestors | {identity}
        children = item.values() if isinstance(item, dict) else item
        stack.extend((child, depth + 1, child_ancestors) for child in children)


def _normalize_file_records(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        return [
            dict(value, path=path) if isinstance(value, dict) else {"path": path, "content": value}
            for path, value in raw.items()
        ]
    if isinstance(raw, list) and all(isinstance(item, dict) for item in raw):
        return list(raw)
    raise ValueError("package files must be an object or list")


def _decode_input_file(record: dict[str, Any], path: str) -> tuple[bytes, str]:
    if "content_base64" in record:
        try:
            data = base64.b64decode(record["content_base64"], validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"invalid base64 package file: {path}") from exc
    elif isinstance(record.get("content"), str):
        data = record["content"].encode("utf-8")
    else:
        raise ValueError(f"package file has no content: {path}")
    media_type = _normalize_media_type(record.get("media_type"), path)
    return data, media_type


def _safe_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("package file path is required")
    value = value.replace("\\", "/")
    if len(value.encode("utf-8")) > MAX_PACKAGE_PATH_BYTES:
        raise ValueError(f"package file path exceeds {MAX_PACKAGE_PATH_BYTES} bytes")
    parts = value.split("/")
    if (
        value.startswith("/")
        or re.match(r"^[A-Za-z]:", value)
        or any(
            part in {"", ".", ".."} or any(ord(char) < 32 for char in part)
            for part in parts
        )
    ):
        raise ValueError(f"unsafe package path: {value!r}")
    return PurePosixPath(*parts).as_posix()


def _normalize_media_type(value: Any, path: str) -> str:
    if value is None or value == "":
        return mimetypes.guess_type(path)[0] or "application/octet-stream"
    if not isinstance(value, str):
        raise ValueError(f"package file media type must be a string: {path}")
    if len(value) > MAX_MEDIA_TYPE_LENGTH:
        raise ValueError(
            f"package file media type exceeds {MAX_MEDIA_TYPE_LENGTH} characters: {path}"
        )
    if any(ord(char) < 32 for char in value):
        raise ValueError(f"invalid package file media type: {path}")
    return value


def _strip_wrapper_directory(files: dict[str, tuple[bytes, str]]) -> dict[str, tuple[bytes, str]]:
    if any(path.casefold() == "skill.md" for path in files):
        return files
    roots = {PurePosixPath(path).parts[0] for path in files}
    if len(roots) != 1:
        return files
    return {
        "/".join(PurePosixPath(path).parts[1:]): value
        for path, value in files.items()
        if len(PurePosixPath(path).parts) > 1
    }


def _sanitize_file(data: bytes) -> bytes:
    for bom, encoding in (
        (codecs.BOM_UTF32_LE, "utf-32-le"),
        (codecs.BOM_UTF32_BE, "utf-32-be"),
        (codecs.BOM_UTF16_LE, "utf-16-le"),
        (codecs.BOM_UTF16_BE, "utf-16-be"),
    ):
        if data.startswith(bom):
            try:
                text = data[len(bom):].decode(encoding)
            except UnicodeDecodeError:
                return data
            return bom + sanitize_text(text).encode(encoding)
    if b"\x00" in data:
        return data
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return data
    return sanitize_text(text).encode("utf-8")


def _validate_description(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("skill description must be a non-empty string")
    description = value.strip()
    if len(description) > MAX_DESCRIPTION_LENGTH:
        raise ValueError(f"skill description exceeds {MAX_DESCRIPTION_LENGTH} characters")
    return description


def _file_kind(path: str) -> str:
    root = PurePosixPath(path).parts[0].casefold()
    return root[:-1] if root in {"scripts", "references", "assets", "agents"} else "file"


def _render_mcp_tool(data: dict[str, Any]) -> str:
    raw_name = data.get("name")
    if not isinstance(raw_name, str):
        raise ValueError("MCP tool name must be a string")
    name = normalize_skill_name(raw_name)
    if not name:
        raise ValueError("MCP tool name is required")
    description = _validate_description(data.get("description"))
    schema = data.get("inputSchema")
    if not isinstance(schema, dict):
        raise ValueError("MCP inputSchema must be an object")
    properties, required = schema.get("properties") or {}, schema.get("required") or []
    if not isinstance(properties, dict) or not isinstance(required, list):
        raise ValueError("MCP inputSchema properties and required fields are invalid")
    params = []
    for param_name, spec in properties.items():
        if not isinstance(spec, dict):
            raise ValueError("MCP inputSchema property must be an object")
        params.append(
            f"- **{param_name}** ({spec.get('type') or 'any'}, {'required' if param_name in required else 'optional'}): {str(spec.get('description') or 'No description.').strip()}"
        )
    parameter_text = "\n".join(params) if params else "This skill has no parameters."
    metadata = yaml.safe_dump(
        {"name": name, "description": description}, allow_unicode=True, sort_keys=False
    ).strip()
    return f"---\n{metadata}\n---\n\n# {name}\n\n{description}\n\n## Parameters\n\n{parameter_text}\n\n## Usage\n\nUse the MCP tool `{data.get('name')}` when this capability is needed.\n"
