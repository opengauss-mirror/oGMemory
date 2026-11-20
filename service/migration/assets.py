"""Content-addressed node attachments, restricted to an operator-selected directory."""

from __future__ import annotations

import copy
import hashlib
import re
import shutil
from pathlib import Path

from service.migration.package import PackageError, check_member_size

MAX_ASSET_BYTES = 10 * 1024**3
ASSET_FIELDS = {"source_path", "blob_path", "media_path", "image_path", "video_path", "audio_path"}
BLOB_URI = re.compile(r"ogmem-blob://([0-9a-f]{64})(\.[A-Za-z0-9]{1,12})?\Z")


def _file_digest(path, out=None):
    from core.operation_deadline import check_deadline
    digest = hashlib.sha256()
    with path.open('rb') as inp:
        while True:
            check_deadline()
            chunk = inp.read(1024 * 1024)
            check_deadline()
            if not chunk:
                break
            digest.update(chunk)
            if out is not None:
                out.write(chunk)
                check_deadline()
    return digest.hexdigest()


def bindings(record):
    for field in ASSET_FIELDS:
        value = record["metadata"].get(field)
        if value:
            match = BLOB_URI.fullmatch(value) if isinstance(value, str) else None
            if not match:
                raise PackageError(f"attachment {field} must have a portable blob reference")
            yield field, match[1], match[2] or ""


class Assets:
    def __init__(self, root, staging):
        self.root = Path(root).resolve() if root else None
        self.staging = Path(staging)
        self.files = {}
        self.existing = {}
        self.size = 0

    def capture(self, record, *, copy_bytes=True):
        result = copy.deepcopy(record)
        for field in ASSET_FIELDS:
            value = result["metadata"].get(field)
            if not value:
                continue
            if self.root is None:
                raise PackageError("node attachments require migration_asset_root on the source")
            if not isinstance(value, str):
                raise PackageError(f"attachment {field} must be a filesystem path")
            candidate = Path(value)
            path = (candidate if candidate.is_absolute() else self.root / candidate).resolve()
            if not path.is_relative_to(self.root) or not path.is_file():
                raise PackageError(f"attachment {field} is missing or outside migration_asset_root")
            suffix = path.suffix if re.fullmatch(r"\.[A-Za-z0-9]{1,12}", path.suffix) else ""
            check_member_size(path.stat().st_size)
            if path.stat().st_size > MAX_ASSET_BYTES:
                raise PackageError("attachments exceed package byte limit")
            digest = _file_digest(path)
            if not copy_bytes:
                self.existing[(digest, suffix)] = path
            if copy_bytes and digest not in self.files:
                self.size += path.stat().st_size
                if self.size > MAX_ASSET_BYTES:
                    raise PackageError("attachments exceed package byte limit")
                target = self.staging / digest
                with target.open("wb") as out:
                    if _file_digest(path, out) != digest:
                        raise PackageError("attachment changed during export")
                self.files[digest] = target
            result["metadata"][field] = f"ogmem-blob://{digest}{suffix}"
        return result

    def target(self, digest, suffix):
        if self.root is None:
            raise PackageError("attachments require a durable migration_asset_root on the target")
        path = self.root / (digest + suffix)
        if path.is_symlink() or not path.resolve().is_relative_to(self.root):
            raise PackageError("unsafe target attachment path")
        return path

    def validate(self, package, inv, *, check_target=True, check_declared=True):
        used = set()
        for record in inv.records("nodes"):
            for _, digest, suffix in bindings(record):
                package.blob(digest)
                used.add(f"blobs/sha256/{digest}")
                if not check_target:
                    continue
                path = self.target(digest, suffix)
                if path.exists():
                    with path.open("rb") as inp:
                        if hashlib.file_digest(inp, "sha256").hexdigest() != digest:
                            raise PackageError("target attachment conflicts with package")
        declared = {name for name in package.manifest["files"] if name.startswith("blobs/")}
        if check_declared and used != declared:
            raise PackageError("package has unreferenced or missing attachments")

    def validate_merged(self, package, records):
        for record in records:
            for _, digest, suffix in bindings(record):
                existing = self.existing.get((digest, suffix))
                if existing is not None:
                    if not existing.is_file() or _file_digest(existing) != digest:
                        raise PackageError('target attachment changed during merge')
                else:
                    package.blob(digest)
                    target = self.target(digest, suffix)
                    if target.exists() and _file_digest(target) != digest:
                        raise PackageError('target attachment conflicts with package')

    def restore(self, package, record):
        result = copy.deepcopy(record)
        for field, digest, suffix in bindings(record):
            target = self.existing.get((digest, suffix)) or self.target(digest, suffix)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                with target.open("xb") as out:
                    try:
                        with package.blob(digest).open("rb") as inp:
                            shutil.copyfileobj(inp, out, length=1024 * 1024)
                    except Exception:
                        target.unlink(missing_ok=True)
                        raise
            result["metadata"][field] = str(target)
        return result
