# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Inspect published Agent Skills and explicitly sync generated ones."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from skills.artifact import validate_skill_name

_MANIFEST_NAME = ".ogmemory-managed-skills.json"


def run(args: Any) -> int:
    try:
        client = _SkillClient.from_args(args)
        if args.skill_action == "list":
            payload = client.list_skills(query=args.query, limit=args.limit)
            _print_json(payload)
        elif args.skill_action == "show":
            payload = client.read_skill(args.name, source=args.source)
            print(payload.get("content", "") if args.raw else json.dumps(payload, ensure_ascii=False, indent=2))
        elif args.skill_action == "candidates":
            _print_json(client.list_candidates(state=args.state))
        elif args.skill_action == "sync":
            target = (
                args.target_dir
                or os.environ.get("OGMEM_SKILL_SYNC_TARGET")
                or _configured_sync_target(args.config)
            )
            if not target:
                raise ValueError(
                    "skill sync requires --target-dir or OGMEM_SKILL_SYNC_TARGET; no files were written"
                )
            result = sync_skills(
                client,
                Path(target).expanduser(),
                dry_run=bool(args.dry_run),
                prune=bool(args.prune),
            )
            _print_json(result)
        else:
            raise ValueError(f"unknown skill action: {args.skill_action}")
        return 0
    except (ValueError, OSError, httpx.HTTPError) as exc:
        print(f"ogmem skill: {exc}", file=__import__("sys").stderr)
        return 1


class _SkillClient:
    def __init__(self, base_url: str, headers: dict[str, str]):
        self.base_url = base_url.rstrip("/")
        self.headers = headers

    @classmethod
    def from_args(cls, args: Any) -> _SkillClient:
        # OGMEM_BASE_URL belongs to the LLM provider configuration.  Reusing it
        # here can accidentally send API credentials to an unrelated endpoint.
        base_url = (
            args.base_url
            or os.environ.get("OGMEM_HTTP_BASE_URL")
            or os.environ.get("OGMEM_SERVER_URL")
            or "http://127.0.0.1:8090"
        )
        headers = _request_headers(args)
        return cls(base_url, headers)

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        response = httpx.get(
            f"{self.base_url}{path}", headers=self.headers, params=params, timeout=30.0
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("oGMemory returned a non-object response")
        return payload

    def list_skills(
        self,
        *,
        query: str | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        params = {"limit": limit}
        if query:
            params["q"] = query
        if cursor:
            params["cursor"] = cursor
        return self._get("/api/v1/skills", params)

    def read_skill(self, name: str, source: str | None = None) -> dict[str, Any]:
        _validate_catalog_skill_name(name, source)
        params = {"source": source} if source else None
        return self._get(f"/api/v1/skills/{quote(name, safe='')}", params)

    def list_candidates(self, *, state: str | None = None) -> dict[str, Any]:
        return self._get("/api/v1/skill-candidates", {"state": state} if state else None)


def sync_skills(
    client: Any,
    target_dir: Path,
    *,
    dry_run: bool,
    prune: bool,
) -> dict[str, Any]:
    rows = _fetch_all_skills(client)

    manifest_path = target_dir / _MANIFEST_NAME
    manifest = _load_manifest(manifest_path)
    managed_entries = manifest.get("skills", [])
    for name in managed_entries:
        if not isinstance(name, str):
            raise ValueError(f"invalid managed manifest: {manifest_path}")
        _validate_skill_name(name)
        _skill_file(target_dir, name)
    managed_before = set(managed_entries)
    desired: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("skills response contains an invalid entry")
        source = row.get("source") or "generated"
        if source == "imported":
            continue
        if source != "generated":
            raise ValueError("skills response contains an invalid source")
        name = str(row.get("name") or row.get("id") or "")
        _validate_skill_name(name)
        payload = client.read_skill(name, source="generated")
        if payload.get("truncated") is True:
            raise ValueError(f"skill {name!r} was truncated by the server; no files were written")
        content = payload.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"skill {name!r} has no SKILL.md content")
        desired[name] = content

    collisions = []
    for name in desired:
        destination = _skill_file(target_dir, name)
        if destination.exists() and name not in managed_before:
            collisions.append(name)
    if collisions:
        raise ValueError(
            "refusing to overwrite unmanaged skills: " + ", ".join(sorted(collisions))
        )

    removed = sorted(managed_before - set(desired)) if prune else []
    managed_after = set(desired) if prune else managed_before | set(desired)
    result = {"target_dir": str(target_dir), "written": sorted(desired), "removed": removed, "dry_run": dry_run}
    if dry_run:
        return result

    affected = {manifest_path}
    affected.update(_skill_file(target_dir, name) for name in desired)
    affected.update(_skill_file(target_dir, name) for name in removed)
    snapshots = {path: path.read_bytes() if path.exists() else None for path in affected}
    directories_before = {
        path: path.exists()
        for path in {target_dir.resolve(strict=False), *(item.parent for item in affected)}
    }
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        for name, content in desired.items():
            skill_file = _skill_file(target_dir, name)
            skill_file.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_text(skill_file, content)
        for name in removed:
            skill_file = _skill_file(target_dir, name)
            if skill_file.exists():
                skill_file.unlink()
            _remove_empty_dir(skill_file.parent)
        _atomic_write_text(
            manifest_path,
            json.dumps(
                {"version": 1, "skills": sorted(managed_after)},
                ensure_ascii=False,
                indent=2,
            ) + "\n",
        )
    except BaseException:
        _restore_snapshots(snapshots, directories_before)
        raise
    return result


def _fetch_all_skills(client: Any) -> list[Any]:
    rows: list[Any] = []
    cursor: str | None = None
    seen_cursors: set[str] = set()
    while True:
        listing = client.list_skills(limit=100, cursor=cursor)
        page = listing.get("skills", [])
        if not isinstance(page, list):
            raise ValueError("skills response is malformed")
        rows.extend(page)
        truncated = listing.get("truncated", False)
        next_cursor = listing.get("next_cursor")
        if not isinstance(truncated, bool):
            raise ValueError("skills response has an invalid truncated flag")
        if not truncated:
            if next_cursor not in (None, ""):
                raise ValueError("skills response has a cursor but is not truncated")
            return rows
        if not isinstance(next_cursor, str) or not next_cursor:
            raise ValueError("truncated skills response has no next cursor")
        if next_cursor == cursor or next_cursor in seen_cursors:
            raise ValueError("skills pagination cursor did not advance")
        seen_cursors.add(next_cursor)
        cursor = next_cursor


def _skill_file(target_dir: Path, name: str) -> Path:
    _validate_skill_name(name)
    root = target_dir.resolve(strict=False)
    destination = (root / name / "SKILL.md").resolve(strict=False)
    if destination.parent.parent != root:
        raise ValueError(f"skill path escapes target directory: {name!r}")
    return destination


def _restore_snapshots(
    snapshots: dict[Path, bytes | None],
    directories_before: dict[Path, bool],
) -> None:
    errors: list[OSError] = []
    # Restore files first, then remove directories created by this invocation.
    for path, content in snapshots.items():
        try:
            if content is None:
                if path.exists():
                    path.unlink()
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                _atomic_write_bytes(path, content)
        except OSError as exc:
            errors.append(exc)
    for path, existed in sorted(
        directories_before.items(), key=lambda item: len(item[0].parts), reverse=True
    ):
        if not existed:
            _remove_empty_dir(path)
    if errors:
        raise RuntimeError("skill sync failed and rollback was incomplete") from errors[0]


def _remove_empty_dir(path: Path) -> None:
    try:
        path.rmdir()
    except OSError:
        pass


def _request_headers(args: Any) -> dict[str, str]:
    headers: dict[str, str] = {}
    api_key = args.api_key or os.environ.get("OGMEM_MCP_API_KEY") or os.environ.get("OG_AUTH_API_KEY")
    values = {
        "X-API-Key": api_key,
        "X-Account-ID": args.account_id or os.environ.get("OGMEM_ACCOUNT_ID"),
        "X-User-ID": args.user_id or os.environ.get("OGMEM_USER_ID"),
        "X-Agent-ID": args.agent_id or os.environ.get("OGMEM_AGENT_ID"),
    }
    for key, value in values.items():
        if value:
            headers[key] = str(value)
    return headers


def _configured_sync_target(config_path: str | None) -> str | None:
    from providers.unified_config import OgMemConfig

    return OgMemConfig.load(config_path).skill_sync_target


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "skills": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"invalid managed manifest: {path}") from exc
    if not isinstance(payload, dict) or payload.get("version") != 1 or not isinstance(payload.get("skills"), list):
        raise ValueError(f"invalid managed manifest: {path}")
    return payload


def _atomic_write_text(path: Path, content: str) -> None:
    _atomic_write_bytes(path, content.encode("utf-8"))


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _validate_skill_name(name: str) -> None:
    validate_skill_name(name)


def _validate_catalog_skill_name(name: str, source: str | None) -> None:
    if source not in {None, "generated", "imported"}:
        raise ValueError("source must be generated or imported")
    if source != "imported":
        try:
            validate_skill_name(name)
            return
        except ValueError:
            if source == "generated":
                raise
    from skills.management import validate_managed_skill_name

    validate_managed_skill_name(name)


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))
