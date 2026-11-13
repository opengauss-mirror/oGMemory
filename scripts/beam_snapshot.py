"""Qualify full SQL snapshots without weakening the native restore hash guard.

PostgreSQL may distribute a literal varchar-array-to-text[] cast during the
first restore. Retain the original archive, prove ONLY this deparser rewrite
with identical rows/sequences, then bind a new snapshot and verify TWO exact
native restores. All other schema/data changes fail closed.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import subprocess

from scripts.beam_eval_safety import (
    digest, json_digest, require, schema_fingerprint_text, write_json,
)

MISMATCH = "Restored DB content/schema/sequence hash mismatch"
LITERAL = r"'(?:[^'\n]|'')*'::character varying"
ARRAY_CAST = re.compile(r"\(\(ARRAY\[(?P<items>" + LITERAL + r"(?:, " + LITERAL + r")*)\]\)::text\[\]\)")
LITERAL_PATTERN = re.compile(LITERAL)


def schema_digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def prove_cast_normalization(original, current):
    old_lines, new_lines = original.split("\n"), current.split("\n")
    require(len(old_lines) == len(new_lines), "Schema line count changed")
    changes = []
    for index, (before, after) in enumerate(zip(old_lines, new_lines)):
        if before == after:
            continue
        # Never normalize arbitrary SQL expressions, function bodies or quoted
        # strings. Only CHECK definitions and index predicates can qualify.
        require((before.lstrip().startswith("CONSTRAINT ") and " CHECK " in before)
                or re.match(r"CREATE (?:UNIQUE )?INDEX ", before),
                "Unexpected schema definition changed")
        def distribute(match):
            items = LITERAL_PATTERN.findall(match.group("items"))
            require(items and ", ".join(items) == match.group("items"), "Incomplete SQL literal match")
            return "(ARRAY[" + ", ".join("(" + item + ")::text" for item in items) + "])"
        converted, count = ARRAY_CAST.subn(distribute, before)
        require(count > 0 and converted == after,
                "Schema changed beyond literal varchar-array cast distribution")
        changes.append({"line": index + 1, "original": before, "normalized": after})
    require(changes, "Schema fingerprint differs without a proven cast normalization")
    return {"kind": "literal_varchar_array_cast_distribution", "changes": changes,
            "original_schema_sha256": schema_digest(original),
            "normalized_schema_sha256": schema_digest(current),
            "all_other_schema_bytes_identical": True}


def audit_restored_schema(db, directory, manifest):
    actual = db.fingerprint()
    expected = manifest["inventory"]
    require(actual["tables"] == expected["tables"] and actual["sequences"] == expected["sequences"],
            "Restored corpus rows or sequences changed; no schema qualification allowed")
    with (directory / "snapshot.dump").open("rb") as source:
        original = subprocess.check_output(["docker", "exec", "-i", db.container,
            "pg_restore", "--schema-only", "--file=-"], stdin=source).decode()
    current = subprocess.check_output(["docker", "exec", db.container, "pg_dump", "-U", db.user,
        "-d", db.database, "--schema-only"], text=True)
    original, current = schema_fingerprint_text(original), schema_fingerprint_text(current)
    require(schema_digest(original) == expected["schema_sha256"], "Original archive schema binding changed")
    require(schema_digest(current) == actual["schema_sha256"], "Restored schema changed during audit")
    proof = prove_cast_normalization(original, current)
    require(db.fingerprint() == actual, "Database changed during schema audit")
    return proof, actual


def freeze_qualified_snapshot(db, directory, metadata):
    directory = Path(directory)
    original = db.freeze(directory, metadata)
    original_sha = digest(directory / "snapshot.json")
    original_dump_sha = digest(directory / "snapshot.dump")
    write_json(directory / "qualification-pending.json", {
        "original_manifest_sha256": original_sha, "original_dump_sha256": original_dump_sha})
    normalization = None
    try:
        db.restore(directory)
        inventory = original["inventory"]
    except RuntimeError as exc:
        if str(exc) != MISMATCH:
            raise
        normalization, inventory = audit_restored_schema(db, directory, original)
        write_json(directory / "schema-normalization.json", normalization)
    qualified = directory / "qualified"
    bound_metadata = deepcopy(original)
    bound_metadata["snapshot_qualification"] = {
        "parent_manifest_sha256": original_sha,
        "parent_dump_sha256": original_dump_sha,
        "normalization_sha256": digest(directory / "schema-normalization.json") if normalization else None}
    verified = db.freeze(qualified, bound_metadata)
    require(verified["inventory"] == inventory, "Corpus changed before qualified snapshot creation")
    cycles = [db.restore(qualified), db.restore(qualified)]
    require(digest(directory / "snapshot.json") == original_sha
            and digest(directory / "snapshot.dump") == original_dump_sha, "Original snapshot changed")
    write_json(directory / "qualification.json", {
        "status": "verified", "original_manifest_sha256": original_sha,
        "original_dump_sha256": original_dump_sha,
        "qualified_manifest_sha256": digest(qualified / "snapshot.json"),
        "qualified_dump_sha256": verified["dump_sha256"],
        "normalization_sha256": bound_metadata["snapshot_qualification"]["normalization_sha256"],
        "native_restore_cycles": cycles, "all_original_rows_and_sequences_preserved": True})
    return verified


def resolve_snapshot(directory):
    """Select verified snapshots; keep legacy historical snapshots supported."""
    directory = Path(directory)
    receipt = directory / "qualification.json"
    if not receipt.exists():
        require(not (directory / "qualified").exists() and not (directory / "qualification-pending.json").exists(),
                "Snapshot qualification is incomplete")
        return directory
    report = json.loads(receipt.read_text())
    qualified = directory / "qualified"
    require(report["status"] == "verified" and len(report["native_restore_cycles"]) == 2,
            "Snapshot must pass two native restore checks")
    for path, key in ((directory / "snapshot.json", "original_manifest_sha256"),
                      (directory / "snapshot.dump", "original_dump_sha256"),
                      (qualified / "snapshot.json", "qualified_manifest_sha256"),
                      (qualified / "snapshot.dump", "qualified_dump_sha256")):
        require(digest(path) == report[key], "Qualified snapshot evidence changed")
    if report["normalization_sha256"] is not None:
        require(digest(directory / "schema-normalization.json") == report["normalization_sha256"],
                "Schema normalization proof changed")
    original = json.loads((directory / "snapshot.json").read_text())
    final = json.loads((qualified / "snapshot.json").read_text())
    require(report.get("all_original_rows_and_sequences_preserved") is True
            and original["inventory"]["tables"] == final["inventory"]["tables"]
            and original["inventory"]["sequences"] == final["inventory"]["sequences"],
            "Qualified snapshot changed original corpus rows or sequences")
    require(all(final.get(key) == value for key, value in original.items()
                if key not in ("inventory", "dump_sha256")), "Qualified snapshot runtime/dataset binding changed")
    expected_cycle = {"inventory_sha256": json_digest(final["inventory"]),
                      "dump_sha256": final["dump_sha256"]}
    require(all(cycle == expected_cycle for cycle in report["native_restore_cycles"]),
            "Native restore receipts do not bind the qualified snapshot")
    require(final["snapshot_qualification"] == {
        "parent_manifest_sha256": report["original_manifest_sha256"],
        "parent_dump_sha256": report["original_dump_sha256"],
        "normalization_sha256": report["normalization_sha256"]}, "Qualified snapshot provenance differs")
    return qualified
