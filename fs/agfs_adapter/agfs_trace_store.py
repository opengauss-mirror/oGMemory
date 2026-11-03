"""AGFS persistence for lifecycle traces and managed-skill work state.

This adapter targets the single-oGMemory-instance deployment model.  Account
metadata and individual trace/work records are stored separately so request
writes remain bounded.  An in-process lock serializes updates, and recoverable
file replacement protects mutable records from interrupted writes.
"""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from datetime import UTC, datetime, timedelta

from pyagfs.exceptions import AGFSClientError

from skills.artifact import validate_skill_publication


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime | None = None) -> str:
    return (value or _now()).isoformat()


def _parse_time(value: str | None) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=UTC)
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _event_preview(content: dict, limit: int) -> str:
    event_type = str(content.get("event_type") or "")
    if event_type == "user_message":
        preview = str(content.get("prompt") or "")
    elif event_type in {"tool_start", "tool_success", "tool_failure"}:
        preview = f"[{content.get('hook_type') or event_type}] {content.get('tool_name') or 'unknown_tool'}"
    elif event_type in {"assistant_message", "turn_end"}:
        preview = str(content.get("assistant_message") or "")
    else:
        preview = f"[{content.get('hook_type') or event_type or 'hook_event'}]"
    return preview.replace("\n", " ")[:limit]


def _turn_id(
    account_id: str, owner_space: str, agent_id: str, session_id: str, stop_event_index: int
) -> str:
    identity = "\0".join((account_id, owner_space, agent_id, session_id, str(stop_event_index)))
    return f"turn_{hashlib.sha256(identity.encode()).hexdigest()[:32]}"


class AGFSTraceEventStore:
    """Single-instance AGFS implementation of the lifecycle trace repository."""

    def __init__(self, *, client, mount_prefix: str = "/local", max_turn_events: int = 128) -> None:
        if max_turn_events < 2 or max_turn_events > 1024:
            raise ValueError("max_turn_events must be between 2 and 1024")
        self._client = client
        self._mount_prefix = mount_prefix.rstrip("/")
        self._max_turn_events = max_turn_events
        self._lock = threading.RLock()
        self._recovered_accounts: set[str] = set()
        self._root = f"{self._mount_prefix}/.ogmemory/trace-store/accounts"
        self._work_accounts_root = f"{self._mount_prefix}/.ogmemory/trace-store/work-accounts"
        self._work_accounts_index = (
            f"{self._mount_prefix}/.ogmemory/trace-store/work-accounts-index.json"
        )
        self._mkdir_p(self._root)
        self._mkdir_p(self._work_accounts_root)

    def _mkdir_p(self, path: str) -> None:
        current = ""
        for index, part in enumerate(item for item in path.split("/") if item):
            current += "/" + part
            if index == 0 and self._mount_prefix:
                continue
            try:
                self._client.mkdir(current)
            except AGFSClientError as exc:
                if "exists" not in str(exc).lower():
                    raise

    @staticmethod
    def _account_key(account_id: str) -> str:
        return hashlib.sha256(account_id.encode()).hexdigest()

    def _account_dir(self, account_id: str) -> str:
        return f"{self._root}/{self._account_key(account_id)}"

    def _state_path(self, account_id: str) -> str:
        return f"{self._account_dir(account_id)}/state.json"

    def _work_account_path(self, account_id: str) -> str:
        return f"{self._work_accounts_root}/{self._account_key(account_id)}.json"

    def _collection_dir(
        self,
        account_id: str,
        collection: str,
        session_id: str | None = None,
    ) -> str:
        directory = f"{self._account_dir(account_id)}/{collection}"
        if session_id is not None and collection in {
            "events",
            "turns",
            "candidate_listing",
            "skill_session_entries",
            "example_session_entries",
        }:
            directory += "/" + hashlib.sha256(session_id.encode("utf-8")).hexdigest()
        return directory

    @staticmethod
    def _record_key(collection: str, record: dict) -> str:
        fields = {
            "events": "event_id",
            "event_ids": "event_id",
            "turns": "uri",
            "candidates": "candidate_id",
            "candidate_claims": "candidate_id",
            "candidate_work": "candidate_id",
            "candidate_listing": "candidate_id",
            "skill_sessions": "skill_uri",
            "example_sessions": "ledger_key",
            "skill_session_entries": "entry_key",
            "example_session_entries": "entry_key",
        }
        if collection in fields:
            return str(record[fields[collection]])
        payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _record_path(
        self,
        account_id: str,
        collection: str,
        key: str,
        session_id: str | None = None,
    ) -> str:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return f"{self._collection_dir(account_id, collection, session_id)}/{digest}.json"

    @staticmethod
    def _record_session(collection: str, record: dict) -> str | None:
        if collection in {"events", "turns"}:
            return str(record["session_id"])
        if collection == "candidate_listing":
            return str(record["agent_id"])
        if collection == "skill_session_entries":
            return str(record["skill_uri"])
        if collection == "example_session_entries":
            return str(record["ledger_key"])
        return None

    @staticmethod
    def _trace_session(uri: str) -> str | None:
        marker = "/traces/"
        if marker not in uri:
            return None
        session_id, separator, _ = uri.split(marker, 1)[1].rpartition("/turns/")
        return session_id if separator and session_id else None

    @staticmethod
    def _event_id_record(event: dict) -> dict:
        return {
            "event_id": event["event_id"],
            "session_id": event["session_id"],
            "uri": event["uri"],
            "event_index": event["event_index"],
        }

    @staticmethod
    def _is_missing(exc: Exception) -> bool:
        message = str(exc).lower()
        return any(marker in message for marker in ("not found", "no such file", "no such directory"))

    def _read_file(self, path: str) -> bytes:
        try:
            return self._client.read(path)
        except AGFSClientError as exc:
            if not self._is_missing(exc):
                raise
        try:
            return self._client.read(path + ".bak")
        except AGFSClientError as exc:
            if self._is_missing(exc):
                raise FileNotFoundError(path) from exc
            raise

    def _remove_if_present(self, path: str) -> None:
        try:
            self._client.rm(path)
        except AGFSClientError as exc:
            if not self._is_missing(exc):
                raise

    def _write_work_account(self, account_id: str, *, indexes_dirty: bool = False) -> None:
        payload = json.dumps(
            {
                "account_id": account_id,
                "updated_at": _iso(),
                "indexes_dirty": indexes_dirty,
            },
            separators=(",", ":"),
        ).encode("utf-8")
        self._replace_file(self._work_account_path(account_id), payload)

    def _remove_work_account(self, account_id: str) -> None:
        path = self._work_account_path(account_id)
        self._remove_if_present(path)
        self._remove_if_present(path + ".bak")

    def _read_work_accounts(self) -> list[tuple[str, bool]]:
        accounts = []
        for path in self._record_paths(self._work_accounts_root):
            try:
                record = json.loads(self._read_file(path).decode("utf-8"))
                account_id = str(record["account_id"])
                if path != self._work_account_path(account_id):
                    continue
                accounts.append(
                    (
                        record.get("updated_at", ""),
                        account_id,
                        bool(record.get("indexes_dirty", False)),
                    )
                )
            except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
                continue
        return [(account_id, indexes_dirty) for _, account_id, indexes_dirty in sorted(accounts)]

    def _replace_file(self, path: str, payload: bytes) -> None:
        """Replace one file while retaining a readable previous generation."""

        temporary = f"{path}.tmp-{uuid.uuid4().hex}"
        backup = path + ".bak"
        self._client.write(temporary, payload)
        current_exists = True
        try:
            self._client.read(path, offset=0, size=1)
        except AGFSClientError as exc:
            if self._is_missing(exc):
                current_exists = False
            else:
                raise
        try:
            if current_exists:
                self._remove_if_present(backup)
                self._client.mv(path, backup)
            self._client.mv(temporary, path)
        except Exception:
            self._remove_if_present(temporary)
            if current_exists:
                try:
                    self._client.mv(backup, path)
                except AGFSClientError:
                    pass
            raise
        self._remove_if_present(backup)

    @staticmethod
    def _empty_state(account_id: str) -> dict:
        return {
            "schema_version": 5,
            "session_partitioned": True,
            "event_ids_indexed": True,
            "candidate_claims_indexed": True,
            "candidate_work_indexed": True,
            "session_ledgers": True,
            "account_id": account_id,
            "next_event_index": 1,
            "events": {},
            "turns": {},
            "candidates": {},
            "example_evidence": [],
            "updated_at": _iso(),
        }

    def _record_paths(self, directory: str) -> list[str]:
        try:
            entries = self._client.ls(directory)
        except AGFSClientError as exc:
            if self._is_missing(exc):
                return []
            raise
        paths = set()
        for entry in entries:
            name = str(entry.get("name", ""))
            if entry.get("isDir", False):
                continue
            if name.endswith(".json.bak"):
                name = name[:-4]
            elif not name.endswith(".json"):
                continue
            paths.add(f"{directory}/{name}")
        return sorted(paths)

    def _read_collection(
        self,
        account_id: str,
        collection: str,
        session_id: str | None = None,
    ) -> tuple[dict | list, dict]:
        directory = self._collection_dir(account_id, collection, session_id)
        paths = self._record_paths(directory)
        if session_id is None and collection in {"events", "turns"}:
            try:
                entries = self._client.ls(directory)
            except AGFSClientError as exc:
                if not self._is_missing(exc):
                    raise
                entries = []
            for entry in entries:
                if entry.get("isDir", False):
                    paths.extend(self._record_paths(f"{directory}/{entry['name']}"))
        records = [] if collection == "example_evidence" else {}
        hashes = {}
        for path in sorted(paths):
            try:
                raw = self._read_file(path)
                record = json.loads(raw.decode("utf-8"))
                key = self._record_key(collection, record)
            except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
                continue
            hashes[key] = hashlib.sha256(raw).hexdigest()
            if isinstance(records, list):
                records.append(record)
            else:
                records[key] = record
        return records, hashes

    def _read_record(
        self,
        account_id: str,
        collection: str,
        key: str,
        session_id: str | None = None,
    ) -> dict | None:
        try:
            raw = self._read_file(self._record_path(account_id, collection, key, session_id))
        except FileNotFoundError:
            return None
        record = json.loads(raw.decode("utf-8"))
        return record if self._record_key(collection, record) == key else None

    def _write_record(self, account_id: str, collection: str, record: dict) -> None:
        key = self._record_key(collection, record)
        session_id = self._record_session(collection, record)
        self._mkdir_p(self._collection_dir(account_id, collection, session_id))
        payload = json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self._replace_file(
            self._record_path(account_id, collection, key, session_id),
            payload,
        )

    def _migrate_session_shards(self, account_id: str, state: dict) -> None:
        for collection in ("events", "turns"):
            directory = self._collection_dir(account_id, collection)
            for path in self._record_paths(directory):
                try:
                    record = json.loads(self._read_file(path).decode("utf-8"))
                    self._write_record(account_id, collection, record)
                except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
                    continue
                self._remove_if_present(path)
                self._remove_if_present(path + ".bak")
        state["session_partitioned"] = True
        state["_loaded_collections"] = set()
        state["_persisted_hashes"] = {}
        self._write_state(state)

    def _migrate_event_ids(self, account_id: str, state: dict) -> None:
        events, _ = self._read_collection(account_id, "events")
        for event in events.values():
            self._write_record(
                account_id,
                "event_ids",
                self._event_id_record(event),
            )
        state["event_ids_indexed"] = True
        state["_loaded_collections"] = set()
        state["_persisted_hashes"] = {}
        self._write_state(state)

    @staticmethod
    def _claim_turn(turn: dict) -> dict:
        return {
            "turn_id": turn["turn_id"],
            "uri": turn["uri"],
            "owner_space": turn["owner_space"],
            "agent_id": turn["agent_id"],
            "session_id": turn["session_id"],
            "metadata": turn.get("metadata") or {},
            "ended_at": turn.get("ended_at", ""),
            "status": turn.get("status"),
        }

    @staticmethod
    def _candidate_claimable(candidate: dict) -> bool:
        assessment = candidate.get("assessment") or {}
        return candidate.get("state") in {"pending", "processing"} and (
            candidate.get("capability_key") is None and not assessment.get("staged_publication")
        )

    @staticmethod
    def _candidate_active(candidate: dict) -> bool:
        return candidate.get("state") in {"pending", "processing", "assessed"}

    @staticmethod
    def _candidate_listed(candidate: dict) -> bool:
        return candidate.get("state") in {"assessed", "held"}

    def _sync_candidate_listing(self, account_id: str, candidate: dict) -> None:
        path = self._record_path(
            account_id,
            "candidate_listing",
            candidate["candidate_id"],
            candidate["agent_id"],
        )
        if self._candidate_listed(candidate):
            self._write_record(
                account_id,
                "candidate_listing",
                {
                    "candidate_id": candidate["candidate_id"],
                    "agent_id": candidate["agent_id"],
                },
            )
        else:
            self._remove_if_present(path)

    def _write_candidate_state(self, account_id: str, candidate: dict) -> None:
        if self._candidate_listed(candidate):
            self._sync_candidate_listing(account_id, candidate)
        self._write_record(account_id, "candidates", candidate)
        if not self._candidate_listed(candidate):
            self._sync_candidate_listing(account_id, candidate)

    def _migrate_candidate_claims(self, account_id: str, state: dict) -> None:
        candidates, _ = self._read_collection(account_id, "candidates")
        for candidate in candidates.values():
            if not self._candidate_claimable(candidate):
                continue
            turn = self._read_record(
                account_id,
                "turns",
                candidate["evidence_uri"],
                candidate["session_id"],
            )
            if not turn or turn.get("status") != "finalized":
                continue
            candidate["claim_turn"] = self._claim_turn(turn)
            self._write_record(account_id, "candidates", candidate)
            self._write_record(
                account_id,
                "candidate_claims",
                {"candidate_id": candidate["candidate_id"]},
            )
        state["candidate_claims_indexed"] = True
        state["_loaded_collections"] = set()
        state["_persisted_hashes"] = {}
        self._write_state(state)

    def _migrate_candidate_work(self, account_id: str, state: dict) -> None:
        candidates, _ = self._read_collection(account_id, "candidates")
        for candidate in candidates.values():
            if self._candidate_active(candidate):
                self._write_work_account(account_id)
                self._write_record(
                    account_id,
                    "candidate_work",
                    {"candidate_id": candidate["candidate_id"]},
                )
        state["candidate_work_indexed"] = True
        state["_loaded_collections"] = set()
        state["_persisted_hashes"] = {}
        self._write_state(state)

    def _migrate_candidate_listing(self, account_id: str, state: dict) -> None:
        candidates, _ = self._read_collection(account_id, "candidates")
        for candidate in candidates.values():
            if self._candidate_listed(candidate):
                self._sync_candidate_listing(account_id, candidate)
        directory = self._collection_dir(account_id, "candidate_listing")
        for path in self._record_paths(directory):
            self._remove_if_present(path)
            self._remove_if_present(path + ".bak")
        state["schema_version"] = 4
        state["_loaded_collections"] = set()
        state["_persisted_hashes"] = {}
        self._write_state(state)

    @staticmethod
    def _example_ledger_key(skill_uri: str, example_id: str, example_hash: str) -> str:
        return "\0".join((skill_uri, example_id, example_hash))

    def _migrate_session_ledgers(self, account_id: str, state: dict) -> None:
        candidates, _ = self._read_collection(account_id, "candidates")
        for candidate in candidates.values():
            skill_uri = candidate.get("skill_uri")
            if candidate.get("state") == "consumed" and skill_uri:
                self._add_session_to_ledger(
                    account_id,
                    "skill_session_entries",
                    skill_uri,
                    candidate["session_id"],
                    skill_uri=skill_uri,
                )
        evidence, _ = self._read_collection(account_id, "example_evidence")
        for item in evidence:
            key = self._example_ledger_key(
                item["skill_uri"], item["example_id"], item["example_hash"]
            )
            self._add_session_to_ledger(
                account_id,
                "example_session_entries",
                key,
                item["session_id"],
                ledger_key=key,
                skill_uri=item["skill_uri"],
                example_id=item["example_id"],
                example_hash=item["example_hash"],
            )
        state["session_ledgers"] = True
        state["schema_version"] = 5
        state["_loaded_collections"] = set()
        state["_persisted_hashes"] = {}
        self._write_state(state)

    def _migrate_session_ledger_shards(self, account_id: str, state: dict) -> None:
        for collection, entry_collection, key_field in (
            ("skill_sessions", "skill_session_entries", "skill_uri"),
            ("example_sessions", "example_session_entries", "ledger_key"),
        ):
            records, _ = self._read_collection(account_id, collection)
            for record in records.values():
                key = record[key_field]
                identity = {field: record[field] for field in record if field != "sessions"}
                for session_id in record.get("sessions", ()):
                    self._add_session_to_ledger(
                        account_id,
                        entry_collection,
                        key,
                        session_id,
                        **identity,
                    )
        state["schema_version"] = 5
        state["_loaded_collections"] = set()
        state["_persisted_hashes"] = {}
        self._write_state(state)

    def _ensure_session_ledgers(self, account_id: str) -> None:
        state = self._read_state(account_id, collections=())
        if int(state.get("schema_version", 1)) >= 5:
            return
        if not state.get("session_ledgers", False):
            self._migrate_session_ledgers(account_id, state)
        else:
            self._migrate_session_ledger_shards(account_id, state)

    @staticmethod
    def _session_ledger_entry_key(key: str, session_id: str) -> str:
        return json.dumps([key, session_id], ensure_ascii=False, separators=(",", ":"))

    def _add_session_to_ledger(
        self,
        account_id: str,
        collection: str,
        key: str,
        session_id: str,
        **identity,
    ) -> None:
        self._write_record(
            account_id,
            collection,
            identity
            | {
                "entry_key": self._session_ledger_entry_key(key, session_id),
                "session_id": session_id,
            },
        )

    def _session_ledger_paths(
        self,
        account_id: str,
        collection: str,
        key: str,
        session_ids: set[str],
    ) -> set[str]:
        paths = set(self._record_paths(self._collection_dir(account_id, collection, key)))
        paths.update(
            self._record_path(
                account_id,
                collection,
                self._session_ledger_entry_key(key, session_id),
                key,
            )
            for session_id in session_ids
        )
        return paths

    def _read_active_candidates(
        self,
        account_id: str,
        *,
        repair_indexes: bool = False,
    ) -> list[dict]:
        state = self._read_state(account_id, collections=())
        if not state.get("candidate_work_indexed", False):
            self._migrate_candidate_work(account_id, state)
        if repair_indexes:
            all_candidates, _ = self._read_collection(account_id, "candidates")
            for candidate in all_candidates.values():
                if self._candidate_active(candidate):
                    self._write_record(
                        account_id,
                        "candidate_work",
                        {"candidate_id": candidate["candidate_id"]},
                    )
                if self._candidate_claimable(candidate):
                    self._write_record(
                        account_id,
                        "candidate_claims",
                        {"candidate_id": candidate["candidate_id"]},
                    )
        records, _ = self._read_collection(account_id, "candidate_work")
        candidates = []
        stale = []
        recover_processing = account_id not in self._recovered_accounts
        for record in records.values():
            candidate_id = record["candidate_id"]
            candidate = self._read_record(account_id, "candidates", candidate_id)
            if not candidate or not self._candidate_active(candidate):
                stale.append(candidate_id)
                continue
            if recover_processing and self._recover_processing(
                {"candidates": {candidate_id: candidate}}
            ):
                self._write_candidate_state(account_id, candidate)
            candidates.append(candidate)
        for candidate_id in stale:
            self._remove_if_present(self._record_path(account_id, "candidate_work", candidate_id))
        if not candidates:
            self._remove_work_account(account_id)
        elif repair_indexes:
            self._write_work_account(account_id)
        self._recovered_accounts.add(account_id)
        return candidates

    def _read_listed_candidates(self, account_id: str, agent_id: str) -> list[dict]:
        state = self._read_state(account_id, collections=())
        if int(state.get("schema_version", 1)) < 4:
            self._migrate_candidate_listing(account_id, state)
        records, _ = self._read_collection(
            account_id,
            "candidate_listing",
            agent_id,
        )
        candidates = []
        for record in records.values():
            candidate = self._read_record(
                account_id,
                "candidates",
                record["candidate_id"],
            )
            if candidate and self._candidate_listed(candidate):
                candidates.append(candidate)
            else:
                self._remove_if_present(
                    self._record_path(
                        account_id,
                        "candidate_listing",
                        record["candidate_id"],
                        agent_id,
                    )
                )
        return candidates

    def _read_state(
        self,
        account_id: str,
        *,
        collections: tuple[str, ...] = ("candidates", "example_evidence"),
        session_id: str | None = None,
    ) -> dict:
        legacy = False
        try:
            raw = self._read_file(self._state_path(account_id))
        except FileNotFoundError:
            state = self._empty_state(account_id)
        else:
            state = json.loads(raw.decode("utf-8"))
            legacy = int(state.get("schema_version", 1)) < 2
        if state.get("account_id") != account_id:
            raise RuntimeError("AGFS trace state account mismatch")
        if legacy:
            collections = ("events", "turns", "candidates", "example_evidence")
        else:
            if not state.get("session_partitioned", False):
                self._migrate_session_shards(account_id, state)
            if not state.get("event_ids_indexed", False):
                self._migrate_event_ids(account_id, state)
        persisted = {}
        loaded = set(collections)
        for collection in collections:
            if legacy:
                value = state.get(collection, [] if collection == "example_evidence" else {})
                persisted[collection] = {}
            else:
                value, persisted[collection] = self._read_collection(
                    account_id,
                    collection,
                    session_id,
                )
            state[collection] = value
        for collection in {"events", "turns", "candidates", "example_evidence"} - loaded:
            state[collection] = [] if collection == "example_evidence" else {}
        state["_loaded_collections"] = loaded
        state["_persisted_hashes"] = persisted
        if "candidates" in loaded and account_id not in self._recovered_accounts:
            if self._recover_processing(state):
                self._write_state(state)
            self._recovered_accounts.add(account_id)
        if legacy:
            state["session_partitioned"] = True
            for event in state["events"].values():
                self._write_record(
                    account_id,
                    "event_ids",
                    self._event_id_record(event),
                )
            state["event_ids_indexed"] = True
            state["_loaded_collections"] = {
                "events",
                "turns",
                "candidates",
                "example_evidence",
            }
            self._write_state(state)
        return state

    @staticmethod
    def _recover_processing(state: dict) -> bool:
        recovered = False
        for candidate in state.get("candidates", {}).values():
            if candidate.get("state") != "processing":
                continue
            assessment = candidate.get("assessment") or {}
            if assessment.get("staged_publication"):
                candidate["lease_until"] = _iso(datetime.min.replace(tzinfo=UTC))
            else:
                candidate["state"] = "assessed" if candidate.get("capability_key") else "pending"
                candidate["lease_until"] = None
            candidate["lease_owner"] = None
            recovered = True
        return recovered

    def _write_state(self, state: dict) -> None:
        state["updated_at"] = _iso()
        account_id = state["account_id"]
        self._mkdir_p(self._account_dir(account_id))
        loaded = set(state.get("_loaded_collections", ()))
        persisted = state.get("_persisted_hashes", {})
        for collection in loaded:
            self._mkdir_p(self._collection_dir(account_id, collection))
            values = state[collection]
            records = values if isinstance(values, list) else values.values()
            current_keys = set()
            for record in records:
                key = self._record_key(collection, record)
                session_id = self._record_session(collection, record)
                self._mkdir_p(self._collection_dir(account_id, collection, session_id))
                current_keys.add(key)
                payload = json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                digest = hashlib.sha256(payload).hexdigest()
                if persisted.get(collection, {}).get(key) != digest:
                    self._replace_file(
                        self._record_path(account_id, collection, key, session_id),
                        payload,
                    )
            for key in set(persisted.get(collection, {})) - current_keys:
                if collection not in {"events", "turns"}:
                    self._remove_if_present(self._record_path(account_id, collection, key))
        metadata = {
            key: value
            for key, value in state.items()
            if key
            not in {
                "events",
                "turns",
                "candidates",
                "example_evidence",
                "_loaded_collections",
                "_persisted_hashes",
            }
        }
        metadata["schema_version"] = max(2, int(state.get("schema_version", 1)))
        payload = json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._replace_file(self._state_path(account_id), payload)

    def _read_candidate_for_uri(self, account_id: str, uri: str) -> dict | None:
        turn_id = uri.rsplit("/", 1)[-1]
        candidate_id = f"ctx://{account_id}/skills/candidates/{turn_id}"
        candidate = self._read_record(account_id, "candidates", candidate_id)
        return candidate if candidate and candidate.get("evidence_uri") == uri else None

    @staticmethod
    def _lease_active(candidate: dict, worker_id: str | None = None) -> bool:
        if candidate.get("state") != "processing":
            return False
        if worker_id is not None and candidate.get("lease_owner") != worker_id:
            return False
        return _parse_time(candidate.get("lease_until")) > _now()

    @staticmethod
    def _due(candidate: dict) -> bool:
        return _parse_time(candidate.get("next_attempt_at")) <= _now()

    def insert(
        self,
        *,
        event_id: str,
        account_id: str,
        owner_space: str,
        agent_id: str,
        session_id: str,
        hook_type: str,
        content: dict,
        event_time: datetime | None = None,
    ) -> dict:
        if not event_id or len(event_id) > 64:
            raise ValueError("valid event_id is required")
        if not account_id:
            raise ValueError("account_id is required")
        if not session_id or session_id == "unknown" or len(session_id) > 128:
            raise ValueError("valid session_id is required")
        if len(hook_type) > 64 or not isinstance(content, dict):
            raise ValueError("invalid trace event")
        with self._lock:
            state = self._read_state(account_id, collections=())
            event_id_record = self._read_record(account_id, "event_ids", event_id)
            inserted = event_id_record is None
            if inserted:
                event_index = int(state["next_event_index"])
                state["next_event_index"] = event_index + 1
                uri = f"ctx://{account_id}/traces/{session_id}/events/{event_id}"
            else:
                event_index = int(event_id_record["event_index"])
                uri = event_id_record["uri"]
                stored_session_id = event_id_record["session_id"]
                existing = self._read_record(
                    account_id,
                    "events",
                    event_id,
                    stored_session_id,
                )
                if existing is not None:
                    return {
                        "ok": True,
                        "event_id": event_id,
                        "uri": existing["uri"],
                        "event_index": existing["event_index"],
                        "inserted": False,
                    }
                if stored_session_id != session_id:
                    raise RuntimeError("trace event index references a missing event")
            event = {
                "event_id": event_id,
                "uri": uri,
                "account_id": account_id,
                "owner_space": owner_space,
                "agent_id": agent_id,
                "session_id": session_id,
                "hook_type": hook_type,
                "event_index": event_index,
                "event_time": _iso(event_time),
                "content": content,
                "abstract": _event_preview(content, 200),
                "overview": _event_preview(content, 500),
            }
            if inserted:
                self._write_state(state)
                self._write_record(
                    account_id,
                    "event_ids",
                    self._event_id_record(event),
                )
            self._write_record(account_id, "events", event)
            return {
                "ok": True,
                "event_id": event_id,
                "uri": uri,
                "event_index": event_index,
                "inserted": inserted,
            }

    def finalize_session(
        self,
        *,
        account_id: str,
        owner_space: str,
        agent_id: str,
        session_id: str,
        stop_event_index: int,
    ) -> dict:
        if not session_id or session_id == "unknown":
            raise ValueError("valid session_id is required")
        if stop_event_index <= 0:
            raise ValueError("stop_event_index must be positive")
        with self._lock:
            state = self._read_state(
                account_id,
                collections=("events", "turns"),
                session_id=session_id,
            )
            session_events = sorted(
                (
                    item
                    for item in state["events"].values()
                    if item["session_id"] == session_id
                    and item["owner_space"] == owner_space
                    and item["agent_id"] == agent_id
                ),
                key=lambda item: item["event_index"],
            )
            stop = next(
                (
                    item
                    for item in session_events
                    if item["event_index"] == stop_event_index
                    and item["hook_type"] in {"turn_end", "Stop"}
                ),
                None,
            )
            if stop is None:
                raise ValueError("turn-end event not found")
            batch_id = stop["content"].get("write_batch_id", "")
            session_events = [
                item for item in session_events
                if item["content"].get("write_batch_id", "") == batch_id
            ]
            previous = [
                item["event_index"]
                for item in session_events
                if item["event_index"] < stop_event_index
                and item["hook_type"] in {"turn_end", "Stop"}
            ]
            prev_stop = max(previous) if previous and not batch_id else None
            all_turn_events = [
                item
                for item in session_events
                if int(prev_stop or 0) < item["event_index"] <= stop_event_index
            ]
            if not all_turn_events:
                raise RuntimeError("turn-end-bounded trace turn is empty")
            if len(all_turn_events) > self._max_turn_events and self._max_turn_events > 1:
                retained = [all_turn_events[0], *all_turn_events[-(self._max_turn_events - 1) :]]
            else:
                retained = all_turn_events[-self._max_turn_events :]
            turn_id = _turn_id(account_id, owner_space, agent_id, session_id, stop_event_index)
            uri = f"ctx://{account_id}/traces/{session_id}/turns/{turn_id}"
            contents = [item["content"] for item in retained]
            hook_counts: dict[str, int] = {}
            tools = set()
            tool_event_count = 0
            for item in retained:
                hook = item["hook_type"]
                hook_counts[hook] = hook_counts.get(hook, 0) + 1
                content = item["content"]
                if content.get("event_type") in {"tool_start", "tool_success", "tool_failure"}:
                    tool_event_count += 1
                    if content.get("tool_name"):
                        tools.add(str(content["tool_name"]))
            metadata = {
                "event_count": len(all_turn_events),
                "stored_event_count": len(retained),
                "truncated": len(retained) < len(all_turn_events),
                "stored_tool_event_count": tool_event_count,
                "stored_tools_used": sorted(tools),
                "stored_hook_event_counts": hook_counts,
            }
            state["turns"][uri] = {
                "turn_id": turn_id,
                "uri": uri,
                "account_id": account_id,
                "owner_space": owner_space,
                "agent_id": agent_id,
                "session_id": session_id,
                "started_at": all_turn_events[0]["event_time"],
                "ended_at": all_turn_events[-1]["event_time"],
                "event_count": len(all_turn_events),
                "content": contents,
                "metadata": metadata,
                "status": "finalized",
                "stop_event_index": stop_event_index,
                "prev_stop_event_index": prev_stop,
            }
            self._write_state(state)
            return {
                "ok": True,
                "uri": uri,
                "turn_id": turn_id,
                "event_count": len(all_turn_events),
                "stored_event_count": len(retained),
                "stop_event_index": stop_event_index,
                "prev_stop_event_index": prev_stop,
                "status": "finalized",
            }

    def enqueue_skill_candidate_from_trace(
        self, *, account_id: str, agent_id: str, session_id: str, turn_id: str, trace_uri: str
    ) -> dict:
        candidate_id = f"ctx://{account_id}/skills/candidates/{turn_id}"
        with self._lock:
            turn = self._read_state(
                account_id,
                collections=("turns",),
                session_id=session_id,
            )["turns"].get(trace_uri)
            if not turn:
                raise ValueError("finalized trace not found")
            candidate = self._read_record(account_id, "candidates", candidate_id)
            if candidate is None or self._candidate_active(candidate):
                self._write_work_account(account_id, indexes_dirty=True)
            if candidate is None:
                candidate = {
                    "candidate_id": candidate_id,
                    "account_id": account_id,
                    "agent_id": agent_id,
                    "session_id": session_id,
                    "evidence_uri": trace_uri,
                    "claim_turn": self._claim_turn(turn),
                    "state": "pending",
                    "capability_key": None,
                    "assessment": {},
                    "attempts": 0,
                    "compile_attempts": 0,
                    "next_attempt_at": _iso(),
                    "lease_owner": None,
                    "lease_until": None,
                    "skill_uri": None,
                    "error": None,
                    "created_at": _iso(),
                    "updated_at": _iso(),
                }
                self._write_record(account_id, "candidates", candidate)
            if self._candidate_claimable(candidate):
                if not candidate.get("claim_turn"):
                    candidate["claim_turn"] = self._claim_turn(turn)
                    self._write_record(account_id, "candidates", candidate)
                self._write_record(
                    account_id,
                    "candidate_claims",
                    {"candidate_id": candidate_id},
                )
            if self._candidate_active(candidate):
                self._write_record(
                    account_id,
                    "candidate_work",
                    {"candidate_id": candidate_id},
                )
                self._write_work_account(account_id)
        return {"ok": True, "candidate_id": candidate_id}

    def claim_skill_candidates(
        self, *, account_id: str, worker_id: str, limit: int, lease_seconds: int
    ) -> list[dict]:
        if not worker_id or len(worker_id) > 128 or min(limit, lease_seconds) < 1:
            raise ValueError("invalid claim parameters")
        with self._lock:
            state = self._read_state(account_id, collections=())
            if not state.get("candidate_claims_indexed", False):
                self._migrate_candidate_claims(account_id, state)
            claim_records, _ = self._read_collection(account_id, "candidate_claims")
            eligible = []
            stale_claims = []
            recover_processing = account_id not in self._recovered_accounts
            for claim_record in claim_records.values():
                candidate_id = claim_record["candidate_id"]
                candidate = self._read_record(account_id, "candidates", candidate_id)
                if not candidate or not self._candidate_claimable(candidate):
                    stale_claims.append(candidate_id)
                    continue
                if recover_processing and self._recover_processing(
                    {"candidates": {candidate_id: candidate}}
                ):
                    self._write_candidate_state(account_id, candidate)
                reclaimable = candidate.get("state") == "processing" and not self._lease_active(
                    candidate
                )
                if self._due(candidate) and (candidate.get("state") == "pending" or reclaimable):
                    turn = candidate.get("claim_turn")
                    if turn and turn.get("status") == "finalized":
                        eligible.append((turn.get("ended_at", ""), candidate, turn))
            for candidate_id in stale_claims:
                self._remove_if_present(
                    self._record_path(account_id, "candidate_claims", candidate_id)
                )
            self._recovered_accounts.add(account_id)
            eligible.sort(key=lambda item: (item[0], item[1]["candidate_id"]))
            claimed = []
            for _, candidate, turn in eligible[:limit]:
                candidate.update(
                    {
                        "state": "processing",
                        "attempts": int(candidate.get("attempts", 0)) + 1,
                        "lease_owner": worker_id,
                        "lease_until": _iso(_now() + timedelta(seconds=lease_seconds)),
                        "error": None,
                        "updated_at": _iso(),
                    }
                )
                self._write_candidate_state(account_id, candidate)
                claimed.append(
                    {
                        "turn_id": turn["turn_id"],
                        "uri": turn["uri"],
                        "account_id": account_id,
                        "owner_space": turn["owner_space"],
                        "agent_id": turn["agent_id"],
                        "session_id": turn["session_id"],
                        "metadata": turn.get("metadata") or {},
                        "skill_attempts": candidate["attempts"],
                        "assessment": candidate.get("assessment") or {},
                    }
                )
            return claimed

    def read_skill_trace_event_page(
        self, *, account_id: str, evidence_uri: str, after_ordinal: int, limit: int
    ) -> dict:
        if after_ordinal < 0 or limit < 1:
            raise ValueError("invalid trace page")
        with self._lock:
            state = self._read_state(
                account_id,
                collections=("events", "turns"),
                session_id=self._trace_session(evidence_uri),
            )
            turn = state["turns"].get(evidence_uri)
            events = (turn or {}).get("content", [])
            if turn and turn.get("metadata", {}).get("truncated"):
                batch = events[-1].get("write_batch_id", "") if events else ""
                original = sorted((item for item in state["events"].values()
                    if item["session_id"] == turn["session_id"]
                    and item["owner_space"] == turn["owner_space"]
                    and item["agent_id"] == turn["agent_id"]
                    and int(turn.get("prev_stop_event_index") or 0) < item["event_index"] <= turn["stop_event_index"]
                    and item["content"].get("write_batch_id", "") == batch), key=lambda item: item["event_index"])
                if len(original) != turn["event_count"]:
                    raise RuntimeError("finalized trace source events are incomplete")
                events = [item["content"] for item in original]
            visible = events[after_ordinal : after_ordinal + limit]
            rows = [
                {"ordinal": after_ordinal + index, "event": event}
                for index, event in enumerate(visible, start=1)
            ]
            next_ordinal = rows[-1]["ordinal"] if rows else after_ordinal
            return {
                "events": rows,
                "next_ordinal": next_ordinal,
                "done": next_ordinal >= len(events),
            }

    def save_skill_compaction_checkpoint(
        self, *, account_id: str, uri: str, worker_id: str, checkpoint: dict, lease_seconds: int
    ) -> bool:
        with self._lock:
            candidate = self._read_candidate_for_uri(account_id, uri)
            if not candidate or not self._lease_active(candidate, worker_id):
                return False
            candidate.setdefault("assessment", {})["compaction"] = checkpoint
            candidate["lease_until"] = _iso(_now() + timedelta(seconds=lease_seconds))
            self._write_record(account_id, "candidates", candidate)
            return True

    def renew_skill_leases(
        self, *, account_id: str, uris: list[str], worker_id: str, lease_seconds: int
    ) -> int:
        if not uris:
            return 0
        if not worker_id or len(worker_id) > 128 or lease_seconds < 1:
            raise ValueError("valid worker_id and positive lease_seconds are required")
        with self._lock:
            changed = 0
            for uri in uris:
                candidate = self._read_candidate_for_uri(account_id, uri)
                if candidate and self._lease_active(candidate, worker_id):
                    candidate["lease_until"] = _iso(_now() + timedelta(seconds=lease_seconds))
                    self._write_record(account_id, "candidates", candidate)
                    changed += 1
            return changed

    def _finish_candidate(
        self,
        *,
        account_id: str,
        uri: str,
        worker_id: str,
        state_name: str,
        capability_key: str | None = None,
        assessment: dict | None = None,
        error: str | None = None,
    ) -> bool:
        with self._lock:
            candidate = self._read_candidate_for_uri(account_id, uri)
            if not candidate or not self._lease_active(candidate, worker_id):
                return False
            merged = dict(candidate.get("assessment") or {})
            merged.pop("staged_publication", None)
            merged.update(assessment or {})
            candidate.update(
                {
                    "state": state_name,
                    "capability_key": capability_key,
                    "assessment": merged,
                    "lease_owner": None,
                    "lease_until": None,
                    "error": (error or "")[:2000] or None,
                    "updated_at": _iso(),
                }
            )
            self._write_candidate_state(account_id, candidate)
            self._remove_if_present(
                self._record_path(account_id, "candidate_claims", candidate["candidate_id"])
            )
            if not self._candidate_active(candidate):
                self._remove_if_present(
                    self._record_path(account_id, "candidate_work", candidate["candidate_id"])
                )
                self._sync_work_account(account_id)
            return True

    def mark_skill_assessed(
        self, *, account_id: str, uri: str, worker_id: str, capability_key: str, assessment: dict
    ) -> bool:
        return self._finish_candidate(
            account_id=account_id,
            uri=uri,
            worker_id=worker_id,
            state_name="assessed",
            capability_key=capability_key,
            assessment=assessment,
        )

    def reject_skill_candidate(
        self, *, account_id: str, uri: str, worker_id: str, state: str, error: str
    ) -> bool:
        if state not in {"skipped", "held"}:
            raise ValueError("rejected candidate state must be skipped or held")
        return self._finish_candidate(
            account_id=account_id,
            uri=uri,
            worker_id=worker_id,
            state_name=state,
            error=error,
        )

    def list_ready_skill_capabilities(
        self, *, account_id: str, min_evidence_turns: int, min_distinct_sessions: int, limit: int
    ) -> list[dict]:
        if min_evidence_turns < 1 or min_distinct_sessions < 1:
            raise ValueError("evidence thresholds must be positive")
        if limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        with self._lock:
            candidates = self._read_active_candidates(account_id)
            groups: dict[tuple[str, str], list[dict]] = {}
            for candidate in candidates:
                if not candidate.get("capability_key") or not self._due(candidate):
                    continue
                staged = (candidate.get("assessment") or {}).get("staged_publication")
                if candidate.get("state") == "assessed" or (
                    candidate.get("state") == "processing"
                    and not self._lease_active(candidate)
                    and staged
                ):
                    groups.setdefault(
                        (candidate["agent_id"], candidate["capability_key"]), []
                    ).append(candidate)
            ready = []
            for (agent_id, capability), candidates in groups.items():
                sessions = {item["session_id"] for item in candidates}
                staged = any(
                    (item.get("assessment") or {}).get("staged_publication") for item in candidates
                )
                pending_support = any(
                    (item.get("assessment") or {}).get("eligibility_state") == "pending_support"
                    for item in candidates
                )
                if staged or (
                    len(candidates) >= min_evidence_turns
                    and len(sessions) >= min_distinct_sessions
                    and (len(candidates) >= 2 or not pending_support)
                ):
                    ready.append(
                        (
                            not staged,
                            min(item.get("created_at", "") for item in candidates),
                            agent_id,
                            capability,
                            {
                                "agent_id": agent_id,
                                "capability_key": capability,
                                "evidence_count": len(candidates),
                                "distinct_session_count": len(sessions),
                            },
                        )
                    )
            ready.sort(key=lambda item: item[:4])
            return [item[4] for item in ready[:limit]]

    def claim_skill_evidence(
        self,
        *,
        account_id: str,
        agent_id: str,
        capability_key: str,
        worker_id: str,
        min_evidence_turns: int,
        min_distinct_sessions: int,
        max_evidence_turns: int,
        lease_seconds: int,
    ) -> list[dict]:
        if not worker_id or len(worker_id) > 128:
            raise ValueError("valid worker_id is required")
        if not capability_key:
            raise ValueError("capability_key is required")
        if min_evidence_turns < 1 or min_distinct_sessions < 1:
            raise ValueError("evidence thresholds must be positive")
        if max_evidence_turns < max(min_evidence_turns, min_distinct_sessions):
            raise ValueError("max_evidence_turns must cover both evidence thresholds")
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be positive")
        with self._lock:
            candidates = self._read_active_candidates(account_id)
            if any(
                self._lease_active(item)
                and item.get("agent_id") == agent_id
                and item.get("capability_key")
                for item in candidates
            ):
                return []
            eligible = [
                item
                for item in candidates
                if item.get("agent_id") == agent_id
                and item.get("capability_key") == capability_key
                and self._due(item)
                and (
                    item.get("state") == "assessed"
                    or (item.get("state") == "processing" and not self._lease_active(item))
                )
            ]
            pending_support = any(
                (item.get("assessment") or {}).get("eligibility_state") == "pending_support"
                for item in eligible
            )
            required = max(min_evidence_turns, 2 if pending_support else min_evidence_turns)
            staged = any(
                (item.get("assessment") or {}).get("staged_publication") for item in eligible
            )
            if not staged and (
                len(eligible) < required
                or len({item["session_id"] for item in eligible}) < min_distinct_sessions
            ):
                return []
            eligible.sort(
                key=lambda item: (
                    (item.get("claim_turn") or {}).get("ended_at", ""),
                    item["evidence_uri"],
                )
            )
            session_ranks: dict[str, int] = {}
            ranked = []
            for candidate in eligible:
                session_id = candidate["session_id"]
                session_ranks[session_id] = session_ranks.get(session_id, 0) + 1
                ranked.append((candidate, session_ranks[session_id]))
            ranked.sort(
                key=lambda item: (
                    not bool((item[0].get("assessment") or {}).get("staged_publication")),
                    item[1] != 1,
                    (item[0].get("claim_turn") or {}).get("ended_at", ""),
                    item[0]["evidence_uri"],
                )
            )
            selected = [item[0] for item in ranked[:max_evidence_turns]]
            result = []
            for candidate in selected:
                candidate.update(
                    {
                        "state": "processing",
                        "compile_attempts": int(candidate.get("compile_attempts", 0)) + 1,
                        "lease_owner": worker_id,
                        "lease_until": _iso(_now() + timedelta(seconds=lease_seconds)),
                        "error": None,
                        "updated_at": _iso(),
                    }
                )
                assessment = candidate.get("assessment") or {}
                content = assessment.get("source_events")
                if not isinstance(content, list):
                    turn = self._read_record(
                        account_id,
                        "turns",
                        candidate["evidence_uri"],
                        candidate["session_id"],
                    )
                    content = (turn or {}).get("content", [])
                result.append(
                    {
                        "uri": candidate["evidence_uri"],
                        "session_id": candidate["session_id"],
                        "content": content,
                        "assessment": assessment,
                        "staged_publication": assessment.get("staged_publication"),
                        "skill_compile_attempts": candidate["compile_attempts"],
                    }
                )
                self._write_candidate_state(account_id, candidate)
            return result

    def stage_skill_publication(
        self, *, account_id: str, uris: list[str], worker_id: str, publication: dict
    ) -> int:
        if not uris:
            return 0
        if not worker_id or len(worker_id) > 128:
            raise ValueError("valid worker_id is required")
        publication = validate_skill_publication(publication)
        with self._lock:
            changed = 0
            for uri in uris:
                candidate = self._read_candidate_for_uri(account_id, uri)
                if candidate and self._lease_active(candidate, worker_id):
                    candidate["capability_key"] = candidate.get(
                        "capability_key"
                    ) or publication.get("capability_key")
                    candidate.setdefault("assessment", {})["staged_publication"] = publication
                    self._write_record(account_id, "candidates", candidate)
                    changed += 1
            return changed

    def mark_skill_terminal(
        self,
        *,
        account_id: str,
        uris: list[str],
        worker_id: str,
        state: str,
        skill_uri: str | None = None,
        error: str | None = None,
    ) -> int:
        if state not in {"consumed", "held", "skipped"}:
            raise ValueError("invalid terminal skill state")
        with self._lock:
            self._ensure_session_ledgers(account_id)
            changed = 0
            for uri in uris:
                candidate = self._read_candidate_for_uri(account_id, uri)
                if not candidate or not self._lease_active(candidate, worker_id):
                    continue
                if state == "consumed":
                    if skill_uri:
                        self._add_session_to_ledger(
                            account_id,
                            "skill_session_entries",
                            skill_uri,
                            candidate["session_id"],
                            skill_uri=skill_uri,
                        )
                    publication = (candidate.get("assessment") or {}).get(
                        "staged_publication"
                    ) or {}
                    for item in publication.get("example_evidence_keys", []):
                        if skill_uri and uri in item.get("evidence_uris", []):
                            key = self._example_ledger_key(
                                skill_uri,
                                item["example_id"],
                                item["example_hash"],
                            )
                            self._add_session_to_ledger(
                                account_id,
                                "example_session_entries",
                                key,
                                candidate["session_id"],
                                ledger_key=key,
                                skill_uri=skill_uri,
                                example_id=item["example_id"],
                                example_hash=item["example_hash"],
                            )
                candidate.update(
                    {
                        "state": state,
                        "skill_uri": skill_uri,
                        "error": (error or "")[:2000] or None,
                        "lease_owner": None,
                        "lease_until": None,
                        "updated_at": _iso(),
                    }
                )
                self._write_candidate_state(account_id, candidate)
                self._remove_if_present(
                    self._record_path(
                        account_id,
                        "candidate_work",
                        candidate["candidate_id"],
                    )
                )
                self._sync_work_account(account_id)
                changed += 1
            return changed

    def _retry(
        self,
        *,
        account_id: str,
        uris: list[str],
        worker_id: str,
        max_attempts: int,
        backoff_base_seconds: int,
        backoff_cap_seconds: int,
        error: str,
        candidate_attempts: int | None,
    ) -> str:
        with self._lock:
            owned = [
                item
                for uri in uris
                if (item := self._read_candidate_for_uri(account_id, uri))
                and self._lease_active(item, worker_id)
            ]
            if not owned:
                return "lost"
            staged = any((item.get("assessment") or {}).get("staged_publication") for item in owned)
            if staged:
                attempts = (
                    max(
                        int((item.get("assessment") or {}).get("staged_recovery_attempts", 0))
                        for item in owned
                    )
                    + 1
                )
                for item in owned:
                    item.setdefault("assessment", {})["staged_recovery_attempts"] = attempts
            elif candidate_attempts is not None:
                attempts = candidate_attempts
            else:
                attempts = max(int(item.get("compile_attempts", 0)) for item in owned)
            terminal = attempts >= max_attempts
            next_state = (
                "held"
                if terminal
                else (
                    "processing"
                    if staged
                    else ("pending" if candidate_attempts is not None else "assessed")
                )
            )
            delay = min(backoff_cap_seconds, backoff_base_seconds * (2 ** max(0, attempts - 1)))
            for item in owned:
                item.update(
                    {
                        "state": next_state,
                        "next_attempt_at": _iso(_now() + timedelta(seconds=delay)),
                        "lease_owner": None,
                        "lease_until": _iso(_now()) if staged and not terminal else None,
                        "error": error[:2000],
                        "updated_at": _iso(),
                    }
                )
                self._write_candidate_state(account_id, item)
            if terminal:
                for item in owned:
                    self._remove_if_present(
                        self._record_path(account_id, "candidate_work", item["candidate_id"])
                    )
                self._sync_work_account(account_id)
            return next_state

    def retry_skill_candidate(self, *, attempts: int, **kwargs) -> str:
        uri = kwargs.pop("uri")
        return self._retry(uris=[uri], candidate_attempts=attempts, **kwargs)

    def retry_skill_evidence(self, **kwargs) -> str:
        return self._retry(candidate_attempts=None, **kwargs)

    def release_cancelled_skill_candidate(
        self, *, account_id: str, uri: str, worker_id: str
    ) -> bool:
        with self._lock:
            candidate = self._read_candidate_for_uri(account_id, uri)
            if not candidate or not self._lease_active(candidate, worker_id):
                return False
            staged = bool((candidate.get("assessment") or {}).get("staged_publication"))
            candidate.update(
                {
                    "state": "processing" if staged else "pending",
                    "attempts": max(0, int(candidate.get("attempts", 0)) - 1),
                    "next_attempt_at": _iso(),
                    "lease_owner": None,
                    "lease_until": _iso() if staged else None,
                    "error": None,
                }
            )
            self._write_candidate_state(account_id, candidate)
            return True

    def release_cancelled_skill_evidence(
        self, *, account_id: str, uris: list[str], worker_id: str
    ) -> int:
        return self._release_evidence(account_id, uris, worker_id)

    def release_unstaged_skill_evidence(
        self, *, account_id: str, uris: list[str], worker_id: str
    ) -> int:
        return self._release_evidence(account_id, uris, worker_id)

    def _release_evidence(self, account_id: str, uris: list[str], worker_id: str) -> int:
        with self._lock:
            changed = 0
            for uri in uris:
                candidate = self._read_candidate_for_uri(account_id, uri)
                if (
                    candidate
                    and candidate.get("state") == "processing"
                    and candidate.get("lease_owner") == worker_id
                ):
                    candidate.update(
                        {
                            "state": "assessed",
                            "lease_owner": None,
                            "lease_until": None,
                            "next_attempt_at": _iso(),
                            "error": None,
                        }
                    )
                    candidate["compile_attempts"] = max(
                        0,
                        int(candidate.get("compile_attempts", 0)) - 1,
                    )
                    self._write_candidate_state(account_id, candidate)
                    changed += 1
            return changed

    def reset_stale_skill_publication(
        self,
        *,
        account_id: str,
        uris: list[str],
        worker_id: str,
        requeue_candidate: bool,
        error: str,
        max_attempts: int,
        backoff_base_seconds: int,
        backoff_cap_seconds: int,
    ) -> str:
        with self._lock:
            owned = [
                item
                for uri in uris
                if (item := self._read_candidate_for_uri(account_id, uri))
                and self._lease_active(item, worker_id)
            ]
            if len(owned) != len(uris):
                return "lost"
            attempts = max(
                int(item.get("attempts" if requeue_candidate else "compile_attempts", 0))
                for item in owned
            )
            next_state = (
                "held"
                if attempts >= max_attempts
                else ("pending" if requeue_candidate else "assessed")
            )
            delay = min(backoff_cap_seconds, backoff_base_seconds * (2 ** max(0, attempts - 1)))
            for item in owned:
                item.setdefault("assessment", {}).pop("staged_publication", None)
                item.update(
                    {
                        "state": next_state,
                        "next_attempt_at": _iso(_now() + timedelta(seconds=delay)),
                        "lease_owner": None,
                        "lease_until": None,
                        "error": error[:2000],
                    }
                )
                if requeue_candidate:
                    item["capability_key"] = None
                self._write_candidate_state(account_id, item)
            if next_state == "held":
                for item in owned:
                    self._remove_if_present(
                        self._record_path(account_id, "candidate_work", item["candidate_id"])
                    )
                self._sync_work_account(account_id)
            return next_state

    def count_skill_distinct_sessions(
        self, *, account_id: str, skill_uri: str, evidence_uris: list[str], worker_id: str
    ) -> int:
        with self._lock:
            self._ensure_session_ledgers(account_id)
            sessions = {
                candidate["session_id"]
                for uri in evidence_uris
                if (candidate := self._read_candidate_for_uri(account_id, uri))
                and self._lease_active(candidate, worker_id)
            }
            return len(
                self._session_ledger_paths(
                    account_id,
                    "skill_session_entries",
                    skill_uri,
                    sessions,
                )
            )

    def count_skill_example_distinct_sessions(
        self, *, account_id: str, skill_uri: str, examples: list[dict], worker_id: str
    ) -> dict[str, int]:
        with self._lock:
            self._ensure_session_ledgers(account_id)
            result = {}
            for example in examples:
                key = f"{example['example_id']}:{example['example_hash']}"
                ledger_key = self._example_ledger_key(
                    skill_uri,
                    example["example_id"],
                    example["example_hash"],
                )
                sessions = {
                    candidate["session_id"]
                    for uri in example["evidence_uris"]
                    if (candidate := self._read_candidate_for_uri(account_id, uri))
                    and self._lease_active(candidate, worker_id)
                }
                result[key] = len(
                    self._session_ledger_paths(
                        account_id,
                        "example_session_entries",
                        ledger_key,
                        sessions,
                    )
                )
            return result

    def list_skill_candidates(
        self, *, account_id: str, agent_id: str, state: str | None = None, limit: int = 50
    ) -> list[dict]:
        state_map = {None: {"assessed", "held"}, "accumulating": {"assessed"}, "held": {"held"}}
        if state not in state_map:
            raise ValueError("state must be accumulating or held")
        if limit < 1 or limit > 500:
            raise ValueError("limit must be between 1 and 500")
        with self._lock:
            candidates = [
                item
                for item in self._read_listed_candidates(account_id, agent_id)
                if item.get("agent_id") == agent_id and item.get("state") in state_map[state]
            ]
            groups: dict[tuple[str, str], list[dict]] = {}
            for item in candidates:
                groups.setdefault(
                    (item.get("capability_key") or "unclassified", item["state"]), []
                ).append(item)
            rows = sorted(
                groups.items(),
                key=lambda pair: max(x.get("updated_at", "") for x in pair[1]),
                reverse=True,
            )
            return [
                {
                    "capability_key": key[0],
                    "state": "accumulating" if key[1] == "assessed" else key[1],
                    "evidence_count": len(items),
                    "distinct_session_count": len({item["session_id"] for item in items}),
                    "summary": max(
                        ((item.get("assessment") or {}).get("summary", "") for item in items),
                        default="",
                    ),
                    "reason": max(
                        (
                            item.get("error") or (item.get("assessment") or {}).get("reason", "")
                            for item in items
                        ),
                        default="",
                    ),
                }
                for key, items in rows[:limit]
            ]

    def _all_account_ids(self) -> list[str]:
        try:
            entries = self._client.ls(self._root)
        except AGFSClientError:
            return []
        accounts = []
        for entry in entries:
            if not entry.get("isDir", False):
                continue
            try:
                raw = self._read_file(f"{self._root}/{entry['name']}/state.json")
                state = json.loads(raw.decode("utf-8"))
                accounts.append((state.get("updated_at", ""), str(state["account_id"])))
            except (
                AGFSClientError,
                FileNotFoundError,
                KeyError,
                ValueError,
                json.JSONDecodeError,
            ):
                continue
        return [account_id for _, account_id in sorted(accounts)]

    def _sync_work_account(self, account_id: str) -> None:
        if self._record_paths(self._collection_dir(account_id, "candidate_work")):
            return
        self._remove_work_account(account_id)

    def _ensure_work_accounts_index(self) -> None:
        try:
            marker = json.loads(self._read_file(self._work_accounts_index).decode("utf-8"))
            if marker.get("version") == 1:
                return
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            pass
        for account_id in self._all_account_ids():
            if self._read_active_candidates(account_id):
                self._write_work_account(account_id)
        self._replace_file(self._work_accounts_index, b'{"version":1}')

    def list_pending_skill_accounts(
        self, *, account_ids: list[str] | None = None, limit: int = 1000
    ) -> list[str]:
        if limit < 1 or limit > 10000:
            raise ValueError("limit must be between 1 and 10000")
        with self._lock:
            if account_ids is None:
                self._ensure_work_accounts_index()
                source = self._read_work_accounts()
            else:
                source = [(account_id, False) for account_id in account_ids]
            values = dict(source)
            pending = []
            for account_id, repair_indexes in values.items():
                candidates = self._read_active_candidates(
                    account_id,
                    repair_indexes=repair_indexes,
                )
                if not candidates:
                    continue
                if any(
                    self._due(item)
                    and (
                        item.get("state") in {"pending", "assessed"}
                        or (item.get("state") == "processing" and not self._lease_active(item))
                    )
                    for item in candidates
                ):
                    self._write_state(self._read_state(account_id, collections=()))
                    self._write_work_account(account_id)
                    pending.append(account_id)
                    if len(pending) >= limit:
                        break
        return pending

    def count_skill_work_accounts(self) -> int:
        with self._lock:
            self._ensure_work_accounts_index()
            return sum(
                1
                for account_id, repair_indexes in self._read_work_accounts()
                if any(
                    item.get("state") in {"pending", "assessed", "processing"}
                    for item in self._read_active_candidates(
                        account_id,
                        repair_indexes=repair_indexes,
                    )
                )
            )
