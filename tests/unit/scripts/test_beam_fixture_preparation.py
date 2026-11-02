"""Portable fixture setup and snapshot qualification; no providers or Docker."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import beam_snapshot as snapshot
from scripts import prepare_beam_agent as agent
from scripts.beam_eval_safety import digest, json_digest, schema_fingerprint_text, write_json


def config():
    return {"tools": {"deny": ["*"]}, "agents": {"list": [{"id": "main"}], "defaults": {
        "workspace": "/home/node/.openclaw/workspace", "skipBootstrap": True,
        "model": {"primary": "openai/preserved-model"}}}, "gateway": {"port": 21841},
        "models": {"providers": {"test": {"apiKey": "preserved-private-test-value"}}},
        "plugins": {"slots": {"contextEngine": "og-memory-context-engine"}, "entries": {
            "og-memory-context-engine": {"enabled": True, "config": {"accountId": "a", "userId": "u",
                "memoryApiBaseUrl": "http://127.0.0.1:19841", "autoCapture": False}}}}}


def test_auth_guard_only_inserts_missing_keys_and_preserves_credentials():
    original = config()
    saved = deepcopy(original)
    result = agent.guarded_config(original)
    values = result["plugins"]["entries"]["og-memory-context-engine"]["config"]
    assert values.pop("authApiKey") == values.pop("authAccountId") == ""
    assert result == saved and original == saved
    original["plugins"]["entries"]["og-memory-context-engine"]["config"].update(
        authApiKey="existing-secret", authAccountId="existing-account")
    assert agent.guarded_config(original) == original


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    state = tmp_path / "agent"
    state.mkdir()
    path = state / "openclaw.json"
    path.write_text(json.dumps(config()))
    info = {"Id": "agent-id", "State": {"Running": False}, "Mounts": [
        {"Destination": "/home/node/.openclaw", "Type": "bind", "Source": str(state), "RW": True}]}
    monkeypatch.setattr(agent, "require_eval_container", lambda *args: info)
    args = SimpleNamespace(agent_container="agent", run_id="run", account="a", user="u",
        memory_url="http://127.0.0.1:19841", base_url="http://127.0.0.1:21841", output_dir=tmp_path / "receipt")
    return args, path, info


def test_stopped_agent_has_private_backup_and_exact_settings_preservation(fixture):
    args, path, _ = fixture
    before = path.read_bytes()
    report = agent.prepare(args)
    assert (args.output_dir / "original.private.json").read_bytes() == before
    assert report["inserted_empty_keys"] == ["authApiKey", "authAccountId"]
    assert report["prepared_config_sha256"] == json_digest(agent.guarded_config(json.loads(before)))
    assert not (args.output_dir / "original.private.json").stat().st_mode & 0o077
    assert "preserved-private-test-value" not in json.dumps(report)
    with pytest.raises(RuntimeError, match="overwrite"):
        agent.prepare(args)


@pytest.mark.parametrize("mode", ["running", "wrong_tools", "wrong_port", "wrong_account",
                                 "auto_capture", "readonly", "symlink", "placeholders", "output_mounted"])
def test_agent_setup_rejects_unsafe_fixtures_without_changing_config(fixture, mode):
    args, path, info = fixture
    value = json.loads(path.read_text())
    if mode == "running": info["State"]["Running"] = True
    elif mode == "wrong_tools": value["tools"]["deny"] = []
    elif mode == "wrong_port": value["gateway"]["port"] = 12345
    elif mode == "wrong_account": args.account = "other"
    elif mode == "auto_capture": value["plugins"]["entries"]["og-memory-context-engine"]["config"]["autoCapture"] = True
    elif mode == "readonly": info["Mounts"][0]["RW"] = False
    elif mode == "output_mounted": args.output_dir = path.parent / "answers"
    elif mode == "placeholders": value["models"]["providers"]["test"]["apiKey"] = "__LLM_API_KEY__"
    if mode == "symlink":
        target = path.with_name("other.json")
        path.rename(target)
        path.symlink_to(target)
    else:
        path.write_text(json.dumps(value))
    before = path.read_bytes()
    with pytest.raises(RuntimeError):
        agent.prepare(args)
    assert path.read_bytes() == before and not args.output_dir.exists()


def schemas():
    before = "    CONSTRAINT state_check CHECK ((status = ANY ((ARRAY['PENDING'::character varying, 'can''t'::character varying])::text[])));\n\n"
    after = "    CONSTRAINT state_check CHECK ((status = ANY (ARRAY[('PENDING'::character varying)::text, ('can''t'::character varying)::text])));\n\n"
    return before, after


def test_exact_literal_cast_rewrite_with_trailing_newlines_is_proven():
    before, after = schemas()
    proof = snapshot.prove_cast_normalization(before, after)
    assert len(proof["changes"]) == 1 and proof["all_other_schema_bytes_identical"]
    assert proof["normalized_schema_sha256"] == snapshot.schema_digest(after)


@pytest.mark.parametrize("corruption", ["literal", "order", "type", "typmod", "newline", "other_ddl", "function"])
def test_other_schema_differences_are_never_ignored(corruption):
    before, after = schemas()
    if corruption == "literal": after = after.replace("PENDING", "OTHER")
    elif corruption == "order": after = after.replace("PENDING", "TEMP").replace("can''t", "PENDING").replace("TEMP", "can''t")
    elif corruption == "type": before = before.replace("character varying", "integer")
    elif corruption == "typmod": before = before.replace("character varying", "character varying(10)")
    elif corruption == "newline": after = after[:-1]
    elif corruption == "other_ddl": after += "CREATE TABLE extra (id int);"
    else: before, after = (value.replace("    CONSTRAINT state_check CHECK", "SELECT") for value in (before, after))
    with pytest.raises(RuntimeError): snapshot.prove_cast_normalization(before, after)


class FakeDB:
    def __init__(self, mismatch=False):
        self.calls = []
        self.mismatch = mismatch
        self.inventory = {"tables": {"public.context_nodes": {"count": 3, "rows_md5": "allrows"}},
                          "sequences": {"seq": "3|t"}, "schema_sha256": "normalized"}

    def freeze(self, directory, metadata):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "snapshot.dump").write_bytes(b"immutable dump")
        result = {**metadata, "inventory": deepcopy(self.inventory), "dump_sha256": digest(directory / "snapshot.dump")}
        write_json(directory / "snapshot.json", result)
        self.calls.append(("freeze", directory))
        return result

    def restore(self, directory):
        self.calls.append(("restore", Path(directory)))
        if self.mismatch and len([call for call in self.calls if call[0] == "restore"]) == 1:
            raise RuntimeError(snapshot.MISMATCH)
        return {"inventory_sha256": json_digest(self.inventory), "dump_sha256": digest(Path(directory) / "snapshot.dump")}


@pytest.mark.parametrize("mismatch", [False, True])
def test_qualified_snapshot_preserves_original_and_requires_two_exact_restores(tmp_path, monkeypatch, mismatch):
    db = FakeDB(mismatch)
    proof = snapshot.prove_cast_normalization(*schemas())
    monkeypatch.setattr(snapshot, "audit_restored_schema", lambda *_: (proof, db.inventory))
    root = tmp_path / "freeze"
    snapshot.freeze_qualified_snapshot(db, root, {"run_id": "run"})
    assert snapshot.resolve_snapshot(root) == root / "qualified"
    assert [call[0] for call in db.calls] == ["freeze", "restore", "freeze", "restore", "restore"]
    assert (root / "snapshot.dump").read_bytes() == b"immutable dump"
    report = json.loads((root / "qualification.json").read_text())
    assert len(report["native_restore_cycles"]) == 2
    (root / "qualified/snapshot.dump").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="evidence changed"): snapshot.resolve_snapshot(root)


def test_failed_qualification_cannot_be_used_as_a_legacy_snapshot(tmp_path, monkeypatch):
    db = FakeDB(True)
    def reject(*args): raise RuntimeError("corpus mismatch")
    monkeypatch.setattr(snapshot, "audit_restored_schema", reject)
    root = tmp_path / "freeze"
    with pytest.raises(RuntimeError, match="corpus mismatch"):
        snapshot.freeze_qualified_snapshot(db, root, {})
    assert (root / "snapshot.dump").exists()
    with pytest.raises(RuntimeError, match="incomplete"): snapshot.resolve_snapshot(root)


def test_legacy_snapshot_is_supported_and_schema_header_rule_unchanged(tmp_path):
    assert snapshot.resolve_snapshot(tmp_path) == tmp_path
    assert schema_fingerprint_text("-- Dumped from v16\n\\restrict random\nDDL;\n\\unrestrict random\n") == "DDL;"
