"""Shared storage helpers for V1 control-plane state."""

from __future__ import annotations

import copy
import json
import os
import sqlite3
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Reserved control-plane row that serializes the account-wide root
# slug/fingerprint bijection. It is never a schedulable workspace registration.
CODE_WORKSPACE_SYNC_BINDINGS_ROOT_ID = "__workspace_sync_bindings__"
_WORKSPACE_SYNC_MUTEX_INITIALIZATION_GRACE_SECONDS = 5.0


class IdentityMutationBusyError(RuntimeError):
    pass


class ControlPlaneStore:
    """JSON-backed store for accounts, users, agents and audit logs.

    Supports either a local filesystem root (tests) or an AGFS client.
    """

    _workspace_sync_lock = threading.RLock()

    def __init__(self, mount_prefix: str, client=None, local_root: str | None = None):
        self._client = client
        if client is not None:
            from fs.agfs_adapter.client import enable_registration_fence

            enable_registration_fence(client)
        self._mount_prefix = mount_prefix.rstrip("/")
        self._local_root = Path(local_root) if local_root else None

    @contextmanager
    def identity_mutation(self, account_id: str | None = None):
        path = self.users_path(account_id) if account_id is not None else self.global_accounts_path()
        if self._local_root is not None:
            lock_path = path + ".lock"
            self._ensure_parent(lock_path)
            with open(lock_path, "a+b") as stream:
                if os.name == "nt":
                    import msvcrt

                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
                else:
                    import fcntl

                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                yield
            return

        # AGFS deletion cannot check the lock token atomically. Clear an
        # abandoned identity lock only after stopping every identity writer.
        lock = self._acquire_control_plane_mutex(path, attempts=100, reclaim_stale=False)
        try:
            yield
        finally:
            self._release_control_plane_mutex(lock)

    def global_accounts_path(self) -> str:
        return self._join("_system", "accounts.json")

    @contextmanager
    def code_workspace_registration_lock(self):
        """Keep the scan and publication in one registration session."""
        path = self._join("_system", "code_workspace_registration.lock")
        if self._local_root is not None:
            self._ensure_parent(path)
            connection = sqlite3.connect(path + ".sqlite")
            try:
                try:
                    connection.execute("BEGIN IMMEDIATE")
                except sqlite3.OperationalError as exc:
                    if exc.sqlite_errorcode == sqlite3.SQLITE_BUSY:
                        raise RuntimeError("workspace registration is busy") from None
                    raise
                self._guard_legacy_registration_lock(path)
                yield
            finally:
                connection.close()
            return

        from fs.agfs_adapter.client import registration_session

        with registration_session(self._client, path):
            self._ensure_parent(path)
            self._guard_legacy_registration_lock(path)
            yield

    def _guard_legacy_registration_lock(self, path):
        # Only writers that exclusively create this lock file are fenced by the directory.
        # Writers predating the lock must be stopped before enabling registration.
        if self._local_root is not None:
            try:
                Path(path).mkdir(exist_ok=True)
            except FileExistsError:
                raise RuntimeError("workspace registration is busy") from None
        else:
            try:
                self._client.mkdir(path)
            except Exception as exc:
                if "exists" not in str(exc).lower():
                    raise
            if self._client.stat(path).get("isDir") is not True:
                raise RuntimeError("workspace registration is busy")

    def users_path(self, account_id: str) -> str:
        return self._join("accounts", account_id, "_system", "users.json")

    def agents_path(self, account_id: str) -> str:
        return self._join("accounts", account_id, "_system", "agents.json")

    def audit_logs_path(self, account_id: str) -> str:
        return self._join("accounts", account_id, "_system", "audit_logs.json")

    def code_workspace_sync_path(self, account_id: str, root_id: str) -> str:
        """Return the protected persistence path for one workspace registry entry."""

        self._validate_workspace_sync_identity(account_id, root_id)
        return self._join(
            "accounts",
            account_id,
            "_system",
            "code_workspace_sync",
            f"{root_id}.json",
        )

    def read_code_workspace_sync_state(self, account_id: str, root_id: str) -> dict[str, Any] | None:
        """Read one protected workspace state without exposing it to retrieval."""

        path = self.code_workspace_sync_path(account_id, root_id)
        try:
            state = json.loads(self._read_bytes(path))
            if not isinstance(state, dict):
                raise ValueError("expected registry object")
        except FileNotFoundError:
            return None
        except Exception:
            raise RuntimeError("workspace registry record is unreadable") from None
        return state

    def write_code_workspace_sync_state(
        self,
        account_id: str,
        root_id: str,
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Replace a workspace state through the single mutation primitive."""

        replacement = copy.deepcopy(dict(state))
        return self.mutate_code_workspace_sync_state(
            account_id,
            root_id,
            lambda _current: replacement,
        )

    def list_code_workspace_sync_states(self, account_id: str) -> list[tuple[str, dict[str, Any]]]:
        """List one account's protected workspace states for reconciliation."""

        self._validate_workspace_sync_identity(account_id, "registry")
        directory = self._join(
            "accounts", account_id, "_system", "code_workspace_sync"
        )

        result = []
        for entry in self._list_workspace_registry_directory(directory):
            name = self._workspace_sync_entry_name(entry)
            if not name or not name.endswith(".json"):
                continue
            root_id = name[:-5]
            if root_id == CODE_WORKSPACE_SYNC_BINDINGS_ROOT_ID:
                continue
            try:
                state = self.read_code_workspace_sync_state(account_id, root_id)
            except ValueError:
                continue
            if state is not None:
                result.append((root_id, state))
        return sorted(result, key=lambda item: item[0])

    def list_code_workspace_sync_registrations(
        self,
    ) -> list[tuple[str, str, dict[str, Any]]]:
        """List protected workspace registrations for server-owned recovery.

        This is intentionally a control-plane-only operation.  Callers receive
        raw state so the sync service can reconstruct its internal account
        context after a process restart; public projections still go through
        the sync service's path-free renderer.
        """

        accounts_directory = self._join("accounts")
        entries = self._list_workspace_registry_directory(accounts_directory, directories_only=True)
        account_ids = [self._workspace_sync_entry_name(entry) for entry in entries]

        registrations: list[tuple[str, str, dict[str, Any]]] = []
        for account_id in sorted(set(account_ids)):
            try:
                self._validate_workspace_sync_identity(account_id, "registry")
            except ValueError:
                continue
            for root_id, state in self.list_code_workspace_sync_states(account_id):
                registrations.append((account_id, root_id, state))
        return registrations

    def _list_workspace_registry_directory(self, path: str, *, directories_only: bool = False) -> list[Any]:
        if self._local_root is not None:
            try:
                with os.scandir(path) as entries:
                    return [entry.name for entry in entries if not directories_only or entry.is_dir()]
            except FileNotFoundError:
                return []
            except OSError:
                raise RuntimeError("workspace registry directory is unreadable") from None
        try:
            from fs.agfs_adapter.client import list_workspace_registry

            return list_workspace_registry(self._client, path)
        except FileNotFoundError:
            return []
        except Exception as exc:
            # pyagfs can expose just the backend's error text, including its path.
            message = str(exc).lower()
            reason = message.partition(":")[0].strip()
            if (reason == "readdir" and message.endswith(": not found")) or reason in {
                "not found", "404 not found", "no such directory",
                "no such file", "no such file or directory",
            }:
                return []
            raise RuntimeError("workspace registry directory is unreadable") from None

    def mutate_code_workspace_sync_state(
        self,
        account_id: str,
        root_id: str,
        mutator: Callable[[dict[str, Any] | None], Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Atomically apply a short state transition for one workspace.

        The remote AGFS mutex only protects the control-plane read-modify-write
        window.  Long-running filesystem work is coordinated by the durable
        lease embedded in the state itself.
        """

        if not callable(mutator):
            raise ValueError("workspace sync mutator must be callable")
        with self._workspace_sync_lock:
            lock = self._acquire_control_plane_mutex(self.code_workspace_sync_path(account_id, root_id))
            try:
                current = self.read_code_workspace_sync_state(account_id, root_id)
                updated = mutator(copy.deepcopy(current))
                if not isinstance(updated, Mapping):
                    raise ValueError("workspace sync mutator must return a mapping")
                result = copy.deepcopy(dict(updated))
                self._publish_workspace_state(self.code_workspace_sync_path(account_id, root_id), result)
                return result
            finally:
                self._release_control_plane_mutex(lock)

    def _publish_workspace_state(self, path: str, state: dict[str, Any]) -> None:
        # Registry scans only enumerate *.json. Publish a complete sibling file in one rename.
        self._ensure_parent(path)
        payload = json.dumps(state, ensure_ascii=False, indent=2).encode("utf-8")
        if self._local_root is not None:
            fd, staged = tempfile.mkstemp(dir=Path(path).parent, suffix=".tmp")
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(payload)
                os.replace(staged, path)
            finally:
                Path(staged).unlink(missing_ok=True)
            return

        from fs.agfs_adapter.client import publish_workspace_file

        staged = f"{path}.{uuid.uuid4().hex}.tmp"
        try:
            self._client.write(staged, payload)
            publish_workspace_file(self._client, staged, path, payload)
        except Exception:
            try:
                self._client.rm(staged)
            except Exception:
                pass  # A failed cleanup leaves an unlisted staging file.
            raise

    def read_json(self, path: str, default: Any) -> Any:
        try:
            raw = self._read_bytes(path)
        except FileNotFoundError:
            return default
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return default

    def write_json(self, path: str, data: Any) -> None:
        self._ensure_parent(path)
        self._write_bytes(path, json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"))

    def _join(self, *parts: str) -> str:
        if self._local_root is not None:
            return str(self._local_root.joinpath(*parts))
        base = self._mount_prefix if self._mount_prefix else ""
        return "/".join([base.rstrip("/"), *parts]).replace("//", "/")

    def _read_bytes(self, path: str) -> bytes:
        if self._local_root is not None:
            return Path(path).read_bytes()
        try:
            raw = self._client.read(path)
        except Exception as exc:
            msg = str(exc).lower()
            if "not found" in msg or "no such file" in msg:
                raise FileNotFoundError(path) from exc
            raise
        if not isinstance(raw, bytes):
            raise TypeError("control-plane storage returned non-bytes data")
        return raw

    def _write_bytes(self, path: str, data: bytes) -> None:
        if self._local_root is not None:
            Path(path).write_bytes(data)
            return
        self._client.write(path, data)

    def _ensure_parent(self, path: str) -> None:
        parent = str(Path(path).parent) if self._local_root is not None else path.rsplit("/", 1)[0]
        if not parent:
            return
        if self._local_root is not None:
            Path(parent).mkdir(parents=True, exist_ok=True)
            return

        current = ""
        for part in [p for p in parent.split("/") if p]:
            current += "/" + part
            try:
                self._client.mkdir(current)
            except Exception as exc:
                if "exists" not in str(exc).lower():
                    raise

    @staticmethod
    def _validate_workspace_sync_identity(account_id: str, root_id: str) -> None:
        for label, value in (("account_id", account_id), ("root_id", root_id)):
            if (
                not isinstance(value, str)
                or not value
                or len(value) > 128
                or "/" in value
                or "\\" in value
                or value in {".", ".."}
            ):
                raise ValueError(f"workspace sync {label} is invalid")

    @staticmethod
    def _workspace_sync_entry_name(entry: Any) -> str:
        if isinstance(entry, str):
            return Path(entry).name
        if isinstance(entry, Mapping):
            value = entry.get("name") or entry.get("path")
            return Path(value).name if isinstance(value, str) else ""
        value = getattr(entry, "name", "")
        return Path(value).name if isinstance(value, str) else ""

    def _acquire_control_plane_mutex(
        self, state_path: str, *, attempts: int = 8, reclaim_stale: bool = True,
    ) -> tuple[str, str] | None:
        """Acquire a short AGFS mutex using atomic creation."""

        if self._local_root is not None or self._client is None:
            return None
        create = getattr(self._client, "create", None)
        remove = getattr(self._client, "rm", None) or getattr(self._client, "remove", None)
        stat = getattr(self._client, "stat", None)
        if not callable(create) or not callable(remove) or not callable(stat):
            raise RuntimeError("AGFS does not support atomic workspace sync state locking")

        self._ensure_parent(state_path)
        lock_path = state_path + ".lock"
        token = uuid.uuid4().hex
        expires_at = time.time() + 60.0
        payload = json.dumps(
            {
                "kind": "code_workspace_sync_mutex",
                "token": token,
                "expires_at": expires_at,
            },
            separators=(",", ":"),
        ).encode("utf-8")
        for _attempt in range(attempts):
            try:
                create(lock_path)
            except Exception as exc:
                if "exists" not in str(exc).lower():
                    raise
                if not reclaim_stale:
                    time.sleep(0.02)
                    continue
                marker = self._read_workspace_sync_mutex(lock_path)
                if marker is not None and marker.get("token") == token:
                    return lock_path, token
                if self._workspace_sync_mutex_reclaimable(lock_path, marker):
                    try:
                        remove(lock_path)
                    except Exception as remove_exc:
                        if "not found" not in str(remove_exc).lower():
                            raise
                time.sleep(0.02)
                continue
            try:
                self._client.write(lock_path, payload)
            except Exception:
                try:
                    remove(lock_path)
                except Exception:
                    pass
                raise
            marker = self._read_workspace_sync_mutex(lock_path)
            if marker is not None and marker.get("token") == token:
                return lock_path, token
            # The atomic create succeeded but a later read cannot prove that
            # this process still owns the lock. Do not mutate state without a
            # fencing marker; another process may have reclaimed a stale file.
            time.sleep(0.02)
        if not reclaim_stale:
            raise IdentityMutationBusyError("Tenant identity storage is busy")
        raise RuntimeError("workspace sync state transition is busy")

    def _release_control_plane_mutex(self, lock: tuple[str, str] | None) -> None:
        if lock is None or self._client is None:
            return
        remove = getattr(self._client, "rm", None) or getattr(self._client, "remove", None)
        if not callable(remove):
            return
        lock_path, token = lock
        try:
            marker = self._read_workspace_sync_mutex(lock_path)
            if marker is not None and marker.get("token") == token:
                remove(lock_path)
        except Exception:
            # A valid expired mutex can be reclaimed by a later transition.
            return

    @staticmethod
    def _workspace_sync_mutex_expired(marker: Mapping[str, Any]) -> bool:
        if marker.get("kind") != "code_workspace_sync_mutex":
            return False
        value = marker.get("expires_at")
        if not isinstance(value, (int, float, str, bytes, bytearray)):
            return False
        try:
            return float(value) <= time.time()
        except (TypeError, ValueError):
            return False

    def _workspace_sync_mutex_reclaimable(
        self,
        lock_path: str,
        marker: Mapping[str, Any] | None,
    ) -> bool:
        if marker is not None:
            return self._workspace_sync_mutex_expired(marker)
        return self._workspace_sync_mutex_initialization_expired(lock_path)

    def _workspace_sync_mutex_initialization_expired(self, lock_path: str) -> bool:
        """Recover a process that died after atomic create but before write.

        pyagfs exposes ``stat().modTime`` for this exact recovery case. A
        short grace period leaves a live creator time to publish its marker;
        without a timestamp-capable backend we fail closed rather than remove
        an active lock whose owner cannot be identified.
        """

        if self._client is None:
            return False
        stat = getattr(self._client, "stat", None)
        if not callable(stat):
            return False
        try:
            details = stat(lock_path)
        except Exception:
            return False
        if not isinstance(details, Mapping):
            return False
        modified_at = self._workspace_sync_mutex_modified_at(details)
        return (
            modified_at is not None
            and modified_at + _WORKSPACE_SYNC_MUTEX_INITIALIZATION_GRACE_SECONDS <= time.time()
        )

    @staticmethod
    def _workspace_sync_mutex_modified_at(details: Mapping[str, Any]) -> float | None:
        value = next(
            (
                details.get(key)
                for key in ("modTime", "mod_time", "mtime", "updated_at", "updatedAt")
                if details.get(key) is not None
            ),
            None,
        )
        if isinstance(value, datetime):
            return value.timestamp()
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric = float(value)
            if numeric > 10_000_000_000_000:
                return numeric / 1_000_000_000
            if numeric > 10_000_000_000:
                return numeric / 1_000
            return numeric
        if not isinstance(value, str):
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.timestamp()

    def _read_workspace_sync_mutex(self, lock_path: str) -> dict[str, Any] | None:
        if self._client is None:
            return None
        try:
            raw = self._client.read(lock_path)
        except Exception:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if not isinstance(raw, str):
            return None
        try:
            marker = json.loads(raw)
        except (TypeError, ValueError):
            return None
        return dict(marker) if isinstance(marker, Mapping) else None
