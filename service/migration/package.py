"""Stream and validate the portable archive before any target storage writes.

This module owns the container format only. Storage adapters must additionally
validate record schemas, supported business types and references before restore.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import tarfile
import tempfile
import zlib
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

FORMAT_VERSION = 1
DATASETS = ("nodes", "relations", "archives")
_CHUNK = 1024 * 1024
_BLOB_NAME = re.compile(r"blobs/sha256/[0-9a-f]{64}\Z")


class PackageError(ValueError):
    """The archive is invalid or exceeds the configured resource limits."""


@dataclass(frozen=True)
class PackageLimits:
    max_bytes: int = 10 * 1024**3
    max_files: int = 100_000
    max_record_bytes: int = 16 * 1024**2
    max_manifest_bytes: int = 4 * 1024**2

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in vars(self).values()):
            raise ValueError("package limits must be positive integers")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise PackageError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise PackageError(f"non-finite JSON constant: {value}")


def _decode(raw: bytes) -> dict:
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise PackageError("invalid JSON object") from exc
    if not isinstance(value, dict):
        raise PackageError("record must be a JSON object")
    return value


def _encode(value: dict) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise PackageError("record is not JSON serializable") from exc


def _records(path: Path, limit: int) -> Iterator[dict]:
    with path.open("rb") as stream:
        while line := stream.readline(limit + 1):
            if len(line) > limit:
                raise PackageError("record exceeds byte limit")
            yield _decode(line)


def _data_name(name: str) -> bool:
    return name in {f"entities/{kind}.jsonl" for kind in DATASETS}


def _allowed_name(name: str) -> bool:
    return name == "manifest.json" or _data_name(name) or bool(_BLOB_NAME.fullmatch(name))


def _file_summary(path: Path) -> dict:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(_CHUNK):
            digest.update(chunk)
            size += len(chunk)
    return {"size": size, "sha256": digest.hexdigest()}


def write_package(
    destination: Path,
    datasets: Mapping[str, Iterable[dict]],
    *,
    source_backend: str,
    source_version: str,
    source_environment: dict | None = None,
    blobs: Iterable[Path] = (),
    limits: PackageLimits = PackageLimits(),
) -> dict:
    """Publish one complete archive, never overwriting an existing destination.

    Iterables are consumed once. Attachment filenames in the package are content
    hashes, not source paths. The caller owns business DTO normalization.
    """
    destination = Path(destination)
    if set(datasets) != set(DATASETS):
        raise PackageError(f"datasets must be exactly {DATASETS}")
    if source_backend not in {"agfs", "sql"} or not isinstance(source_version, str):
        raise PackageError("invalid source description")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    with tempfile.TemporaryDirectory(prefix=".ogmem-export-", dir=destination.parent) as tmp:
        root = Path(tmp)
        files = {}
        total = 0
        (root / "entities").mkdir()
        for kind in DATASETS:
            name = f"entities/{kind}.jsonl"
            count = 0
            with (root / name).open("wb") as stream:
                for record in datasets[kind]:
                    if not isinstance(record, dict):
                        raise PackageError("record must be a JSON object")
                    raw = _encode(record) + b"\n"
                    if len(raw) > limits.max_record_bytes:
                        raise PackageError("record exceeds byte limit")
                    total += len(raw)
                    if total > limits.max_bytes:
                        raise PackageError("package exceeds byte limit")
                    stream.write(raw)
                    count += 1
            files[name] = {**_file_summary(root / name), "records": count}
        (root / "blobs/sha256").mkdir(parents=True)
        for source in blobs:
            staging = root / "attachment.tmp"
            digest = hashlib.sha256()
            size = 0
            with Path(source).open("rb") as inp, staging.open("wb") as out:
                while chunk := inp.read(_CHUNK):
                    size += len(chunk)
                    if total + size > limits.max_bytes:
                        raise PackageError("package exceeds byte limit")
                    digest.update(chunk)
                    out.write(chunk)
            name = f"blobs/sha256/{digest.hexdigest()}"
            if name in files:
                staging.unlink()
                continue
            total += size
            staging.rename(root / name)
            files[name] = {"size": size, "sha256": digest.hexdigest()}
            if len(files) + 1 > limits.max_files:
                raise PackageError("package exceeds file limit")
        manifest = {
            "format_version": FORMAT_VERSION,
            "source_backend": source_backend,
            "source_version": source_version,
            "files": files,
        }
        if source_environment is not None:
            from service.migration.description import validate_description

            validate_description(source_environment, source_backend)
            manifest["source_environment"] = source_environment
        raw_manifest = _encode(manifest)
        if len(raw_manifest) > limits.max_manifest_bytes:
            raise PackageError("manifest exceeds byte limit")
        if total + len(raw_manifest) > limits.max_bytes or len(files) + 1 > limits.max_files:
            raise PackageError("package exceeds resource limits")
        (root / "manifest.json").write_bytes(raw_manifest)
        staged = root / "package.tar.gz"
        with tarfile.open(staged, "w:gz", format=tarfile.USTAR_FORMAT) as archive:
            for name in ["manifest.json", *sorted(files)]:
                archive.add(root / name, arcname=name, recursive=False)
        # Same filesystem, atomic no-clobber publish; no partial final package.
        os.link(staged, destination)
    return manifest


@dataclass(frozen=True)
class ValidatedPackage:
    """Only valid inside open_package's context; files are temporary and private."""

    root: Path
    manifest: dict
    limits: PackageLimits

    def records(self, kind: str) -> Iterator[dict]:
        if kind not in DATASETS:
            raise PackageError(f"unknown dataset: {kind}")
        return _records(self.root / f"entities/{kind}.jsonl", self.limits.max_record_bytes)

    def blob(self, digest: str) -> Path:
        name = f"blobs/sha256/{digest}"
        if not _BLOB_NAME.fullmatch(name) or name not in self.manifest["files"]:
            raise PackageError("attachment is not declared in manifest")
        return self.root / name


def _validate(root: Path, observed: dict, limits: PackageLimits) -> dict:
    if "manifest.json" not in observed:
        raise PackageError("missing manifest")
    manifest = _decode((root / "manifest.json").read_bytes())
    if (
        type(manifest.get("format_version")) is not int
        or manifest["format_version"] != FORMAT_VERSION
    ):
        raise PackageError("unsupported format version")
    if manifest.get("source_backend") not in {"agfs", "sql"}:
        raise PackageError("unsupported source backend")
    if not isinstance(manifest.get("source_version"), str):
        raise PackageError("invalid source version")
    if "source_environment" in manifest:
        from service.migration.description import validate_description

        validate_description(manifest["source_environment"], manifest["source_backend"])
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(observed) - {"manifest.json"}:
        raise PackageError("manifest file set differs from archive")
    if not {f"entities/{kind}.jsonl" for kind in DATASETS} <= set(files):
        raise PackageError("missing required dataset")
    for name, expected in files.items():
        if not isinstance(expected, dict):
            raise PackageError("invalid file descriptor")
        if type(expected.get("size")) is not int or expected["size"] != observed[name]["size"]:
            raise PackageError(f"file size mismatch: {name}")
        if expected.get("sha256") != observed[name]["sha256"]:
            raise PackageError(f"checksum mismatch: {name}")
        if _data_name(name):
            count = sum(1 for _ in _records(root / name, limits.max_record_bytes))
            if type(expected.get("records")) is not int or expected["records"] != count:
                raise PackageError(f"record count mismatch: {name}")
        elif name.rsplit("/", 1)[1] != observed[name]["sha256"]:
            raise PackageError("attachment name does not match content")
    return manifest


class _UstarInfo(tarfile.TarInfo):
    @classmethod
    def fromtarfile(cls, archive):
        # Some Python versions use an internal _frombuf() and bypass a
        # frombuf() override. Gate the fixed header at the actual read entry.
        buf = archive.fileobj.read(tarfile.BLOCKSIZE)
        try:
            member = cls.frombuf(buf, archive.encoding, archive.errors)
        except tarfile.EOFHeaderError:
            raise  # Only a complete zero block may terminate member parsing.
        except tarfile.HeaderError as exc:
            raise PackageError("invalid or truncated tar header") from exc
        member.offset = archive.fileobj.tell() - tarfile.BLOCKSIZE
        return member._proc_member(archive)

    @classmethod
    def frombuf(cls, buf, encoding, errors):
        # Inspect the fixed 512-byte header before TarInfo._proc_member can
        # consume PAX/GNU extensions or sparse metadata into memory.
        if len(buf) == tarfile.BLOCKSIZE and buf.strip(b"\0"):
            if buf[257:265] != tarfile.POSIX_MAGIC or buf[156:157] not in (
                tarfile.REGTYPE,
                tarfile.AREGTYPE,
            ):
                raise PackageError(
                    "unsupported archive member; only USTAR regular files are accepted"
                )
        return super().frombuf(buf, encoding, errors)


class _BoundedTarStream:
    """Decode one gzip member with bounded input and output before tar parsing."""

    def __init__(self, stream, limit):
        self.stream, self.remaining = stream, limit
        # Allow DEFLATE framing overhead even for incompressible valid input.
        self.input_remaining = limit + limit // 1000 + 1024
        self.decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
        self.pending = b""
        self.finished = False

    def read(self, size=-1):
        size = min(_CHUNK, self.remaining + 1, size if size >= 0 else _CHUNK)
        output = bytearray()
        while len(output) < size and not self.finished:
            if self.decoder.eof:
                if self.decoder.unused_data or self.stream.read(1):
                    raise PackageError("trailing data or multiple gzip members")
                self.finished = True
                break
            data = self.pending
            if not data:
                data = self.stream.read(min(64 * 1024, self.input_remaining + 1))
                self.input_remaining -= len(data)
                if self.input_remaining < 0:
                    raise PackageError("compressed archive exceeds resource limits")
                if not data:
                    raise PackageError("invalid or truncated gzip stream")
            output.extend(self.decoder.decompress(data, size - len(output)))
            self.pending = self.decoder.unconsumed_tail
        self.remaining -= len(output)
        if self.remaining < 0:
            raise PackageError("decompressed archive exceeds resource limits")
        return bytes(output)

    def finish(self, tar_stream):
        # Drain through tarfile's stream to include bytes it already prefetched.
        # The parser consumed the first zero block; require the second and
        # accept only complete zero padding blocks until the verified gzip EOF.
        padding = 0
        while chunk := tar_stream.read(_CHUNK):
            if chunk.strip(b"\0"):
                raise PackageError("non-zero data after tar end marker")
            padding += len(chunk)
        if padding < tarfile.BLOCKSIZE or padding % tarfile.BLOCKSIZE:
            raise PackageError("invalid tar end padding")


@contextmanager
def open_package(
    source: Path,
    *,
    limits: PackageLimits = PackageLimits(),
    staging_parent: Path | None = None,
) -> Iterator[ValidatedPackage]:
    """Stage and verify the entire package before exposing any records.

    No tar extract API is used. Only known regular members are copied to generated
    paths; malformed archives, duplicate names and undeclared files are rejected.
    """
    with tempfile.TemporaryDirectory(prefix="ogmem-import-", dir=staging_parent) as tmp:
        root = Path(tmp)
        observed = {}
        total = 0
        try:
            # Account for bounded per-file headers/padding and tar end records
            # separately from the existing payload-byte limit.
            stream_limit = limits.max_bytes + limits.max_files * 1024 + tarfile.RECORDSIZE
            with (
                Path(source).open("rb") as compressed,
                tarfile.open(
                    fileobj=(decompressed := _BoundedTarStream(compressed, stream_limit)),
                    mode="r|",
                    tarinfo=_UstarInfo,
                ) as archive,
            ):
                for member in archive:
                    name = member.name
                    if not member.isfile() or not _allowed_name(name) or member.sparse is not None:
                        raise PackageError("unsupported archive member")
                    if name in observed:
                        raise PackageError("duplicate archive member")
                    total += member.size
                    if (
                        member.size < 0
                        or total > limits.max_bytes
                        or len(observed) >= limits.max_files
                    ):
                        raise PackageError("package exceeds resource limits")
                    if name == "manifest.json" and member.size > limits.max_manifest_bytes:
                        raise PackageError("manifest exceeds byte limit")
                    target = root / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    digest = hashlib.sha256()
                    size = 0
                    with archive.extractfile(member) as inp, target.open("xb") as out:
                        while chunk := inp.read(_CHUNK):
                            size += len(chunk)
                            digest.update(chunk)
                            out.write(chunk)
                    if size != member.size:
                        raise PackageError("truncated member")
                    observed[name] = {"size": size, "sha256": digest.hexdigest()}
                decompressed.finish(archive.fileobj)
            manifest = _validate(root, observed, limits)
        except (tarfile.TarError, EOFError, gzip.BadGzipFile, zlib.error) as exc:
            raise PackageError("invalid or truncated archive") from exc
        yield ValidatedPackage(root, manifest, limits)
