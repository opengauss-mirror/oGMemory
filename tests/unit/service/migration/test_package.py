"""Container-level tests; business adapters own schema/reference validation."""

import hashlib
import io
import json
import tarfile

import pytest

from service.migration.package import (
    DATASETS,
    PackageError,
    PackageLimits,
    open_package,
    write_package,
)


def datasets(nodes=()):
    return {"nodes": iter(nodes), "relations": iter(()), "archives": iter(())}


def write(path, data=None, **kwargs):
    return write_package(
        path,
        datasets() if data is None else data,
        source_backend="agfs",
        source_version="test-dev",
        **kwargs,
    )


def repack(path, transform):
    with tarfile.open(path, "r:gz") as archive:
        entries = [(member, archive.extractfile(member).read()) for member in archive]
    with tarfile.open(path, "w:gz") as archive:
        for member, raw in transform(entries):
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))


def test_round_trip_preserves_unicode_states_relations_and_attachment(tmp_path):
    node = {
        "uri": "ctx://acme/users/a/memories/events/test",
        "content": "迁移\n记忆",
        "status": "ARCHIVED",
        "version": 7,
        "metadata": {"custom": [1, False, None]},
    }
    edge = {"from_uri": node["uri"], "to_uri": node["uri"], "relation_type": "related_to"}
    attachment = tmp_path / "source.bin"
    attachment.write_bytes(b"\x00\xff\x01" * 100)
    digest = hashlib.sha256(attachment.read_bytes()).hexdigest()
    path = tmp_path / "memory.ogmem.tar.gz"
    manifest = write(
        path,
        {
            "nodes": iter([node]),
            "relations": iter([edge]),
            "archives": iter([{"messages": [{"content": "历史"}]}]),
        },
        blobs=[attachment, attachment],
    )
    assert len(manifest["files"]) == 4
    with open_package(path) as package:
        root = package.root
        assert list(package.records("nodes")) == [node]
        assert list(package.records("relations")) == [edge]
        assert package.blob(digest).read_bytes() == attachment.read_bytes()
        assert package.manifest == manifest
    assert not root.exists()


def test_empty_datasets_are_present_and_readable(tmp_path):
    path = tmp_path / "empty.tar.gz"
    write(path)
    with open_package(path) as package:
        assert all(list(package.records(kind)) == [] for kind in DATASETS)


def test_failed_export_does_not_publish_or_leave_staging(tmp_path):
    def broken():
        yield {"content": "valid"}
        raise RuntimeError("source read failed")

    with pytest.raises(RuntimeError, match="source read failed"):
        write(tmp_path / "failed.tar.gz", datasets(broken()))
    assert list(tmp_path.iterdir()) == []


def test_destination_never_overwritten(tmp_path):
    path = tmp_path / "existing.tar.gz"
    path.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        write(path)
    assert path.read_bytes() == b"original"


@pytest.mark.parametrize(
    "mutation,error",
    [
        (lambda m: m.update(format_version=2), "unsupported format"),
        (lambda m: m.update(format_version=True), "unsupported format"),
        (lambda m: m.update(source_backend="other"), "unsupported source"),
        (lambda m: m["files"]["entities/nodes.jsonl"].update(records=99), "record count"),
        (lambda m: m["files"]["entities/nodes.jsonl"].update(sha256="0" * 64), "checksum"),
        (lambda m: m["files"].pop("entities/archives.jsonl"), "file set"),
    ],
)
def test_manifest_checked_before_records_are_exposed(tmp_path, mutation, error):
    path = tmp_path / "bad.tar.gz"
    write(path)

    def change(entries):
        for member, raw in entries:
            if member.name == "manifest.json":
                manifest = json.loads(raw)
                mutation(manifest)
                raw = json.dumps(manifest).encode()
            yield member, raw

    repack(path, change)
    with pytest.raises(PackageError, match=error):
        with open_package(path):
            pytest.fail("invalid package exposed to importer")


@pytest.mark.parametrize(
    "name,kind",
    [
        ("../escaped", tarfile.REGTYPE),
        ("/absolute", tarfile.REGTYPE),
        ("entities/nodes.jsonl", tarfile.SYMTYPE),
        ("entities/nodes.jsonl", tarfile.LNKTYPE),
        ("entities/nodes.jsonl", tarfile.DIRTYPE),
        ("unexpected.json", tarfile.REGTYPE),
    ],
)
def test_rejects_unsafe_members(tmp_path, name, kind):
    path = tmp_path / "unsafe.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo(name)
        member.type = kind
        member.linkname = "../outside"
        archive.addfile(member, io.BytesIO())
    with pytest.raises(PackageError, match="unsupported archive member"):
        with open_package(path):
            pytest.fail("unsafe archive accepted")
    assert not (tmp_path / "escaped").exists()


def test_duplicate_members_rejected(tmp_path):
    path = tmp_path / "duplicate.tar.gz"
    write(path)
    repack(path, lambda entries: [*entries, entries[0]])
    with pytest.raises(PackageError, match="duplicate archive"):
        with open_package(path):
            pytest.fail("duplicate accepted")


@pytest.mark.parametrize("raw", [b"[]\n", b'{"x":1,"x":2}\n', b'{"x":NaN}\n', b"\xff\n"])
def test_invalid_records_rejected_even_with_matching_hash(tmp_path, raw):
    path = tmp_path / "record.tar.gz"
    write(path)

    def change(entries):
        for member, data in entries:
            if member.name == "manifest.json":
                manifest = json.loads(data)
                manifest["files"]["entities/nodes.jsonl"] = {
                    "size": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "records": 1,
                }
                data = json.dumps(manifest).encode()
            elif member.name == "entities/nodes.jsonl":
                data = raw
            yield member, data

    repack(path, change)
    with pytest.raises(PackageError):
        with open_package(path):
            pytest.fail("invalid JSON accepted")


@pytest.mark.parametrize(
    "limits",
    [PackageLimits(max_bytes=1), PackageLimits(max_files=1), PackageLimits(max_manifest_bytes=1)],
)
def test_resource_limits_apply_to_both_directions(tmp_path, limits):
    path = tmp_path / "valid.tar.gz"
    write(path)
    with pytest.raises(PackageError):
        with open_package(path, limits=limits):
            pytest.fail("limit bypassed")
    failed = tmp_path / "failed.tar.gz"
    with pytest.raises(PackageError):
        write(failed, limits=limits)
    assert not failed.exists()


def test_record_limit_and_staging_cleanup(tmp_path):
    path = tmp_path / "large-record.tar.gz"
    write(path, datasets([{"content": "x" * 100}]))
    staging = tmp_path / "staging"
    staging.mkdir()
    with pytest.raises(PackageError, match="record exceeds"):
        with open_package(path, limits=PackageLimits(max_record_bytes=10), staging_parent=staging):
            pytest.fail("oversized record accepted")
    assert list(staging.iterdir()) == []


def test_partial_dataset_export_is_rejected(tmp_path):
    with pytest.raises(PackageError, match="datasets must"):
        write(tmp_path / "partial.tar.gz", {"nodes": []})


def test_blob_lookup_cannot_escape_staging(tmp_path):
    path = tmp_path / "valid.tar.gz"
    write(path)
    with open_package(path) as package:
        with pytest.raises(PackageError, match="attachment"):
            package.blob("../../manifest.json")


def test_corrupt_record_payload_is_rejected_before_restore(tmp_path):
    path = tmp_path / "corrupt.tar.gz"
    write(path, datasets([{"content": "before"}]))

    def corrupt(entries):
        for member, raw in entries:
            if member.name == "entities/nodes.jsonl":
                raw = raw.replace(b"before", b"broken")
            yield member, raw

    repack(path, corrupt)
    with pytest.raises(PackageError, match="checksum mismatch"):
        with open_package(path):
            pytest.fail("corrupt data exposed to restore")


def test_truncated_compressed_archive_is_rejected(tmp_path):
    path = tmp_path / "truncated.tar.gz"
    write(path, datasets([{"content": "example"}]))
    raw = path.read_bytes()
    path.write_bytes(raw[: len(raw) // 2])
    with pytest.raises(PackageError):
        with open_package(path):
            pytest.fail("truncated archive accepted")


@pytest.mark.parametrize(
    "kind", [tarfile.XHDTYPE, tarfile.XGLTYPE, tarfile.GNUTYPE_LONGNAME, tarfile.GNUTYPE_LONGLINK]
)
def test_extensions_rejected_before_parser_reads_their_payload(tmp_path, monkeypatch, kind):
    import gzip

    header = tarfile.TarInfo("extension")
    header.type = kind
    header.size = 64 * 1024
    path = tmp_path / "extension.tar.gz"
    with gzip.open(path, "wb") as stream:
        stream.write(header.tobuf(format=tarfile.USTAR_FORMAT))
        stream.write(b"A" * header.size)
        stream.write(b"\0" * 1024)

    def forbidden(*args, **kwargs):
        pytest.fail("extension parser consumed an unsupported header")

    monkeypatch.setattr(tarfile.TarInfo, "_proc_pax", forbidden)
    monkeypatch.setattr(tarfile.TarInfo, "_proc_gnulong", forbidden)
    reads = []
    from service.migration.package import _BoundedTarStream

    original_read = _BoundedTarStream.read

    def bounded_read(self, size=-1):
        assert 0 < size <= tarfile.RECORDSIZE
        chunk = original_read(self, size)
        reads.append(len(chunk))
        return chunk

    monkeypatch.setattr(_BoundedTarStream, "read", bounded_read)
    with pytest.raises(PackageError, match="only USTAR"):
        with open_package(path, limits=PackageLimits(max_bytes=1024, max_files=4)):
            pytest.fail("unsupported extension accepted")
    assert sum(reads) <= tarfile.RECORDSIZE


def test_hidden_pax_metadata_cannot_bypass_payload_budget(tmp_path):
    path = tmp_path / "pax.tar.gz"
    write(path)

    def attach(entries):
        for i, (member, raw) in enumerate(entries):
            if i == 0:
                member.pax_headers = {"comment": "A" * 65536}
            yield member, raw

    repack(path, attach)
    assert path.stat().st_size < 1024
    with pytest.raises(PackageError, match="only USTAR"):
        with open_package(path, limits=PackageLimits(max_bytes=1024, max_files=4)):
            pytest.fail("hidden metadata accepted")


def test_invalid_deflate_stream_is_reported_as_package_error(tmp_path):
    path = tmp_path / "corrupt.tar.gz"
    path.write_bytes(b"\x1f\x8b\x08\x00" + b"\0" * 6 + b"\x07" + b"\0" * 8)
    with pytest.raises(PackageError, match="invalid or truncated"):
        with open_package(path):
            pytest.fail("invalid compressed stream accepted")


@pytest.mark.parametrize("mutation", ["comment", "trailer", "crc", "concatenated", "padding"])
def test_gzip_envelope_is_validated_before_records_are_exposed(tmp_path, mutation):
    import gzip

    path = tmp_path / "envelope.tar.gz"
    write(path)
    raw = path.read_bytes()
    if mutation == "comment":
        end = raw.index(b"\0", 10) + 1 if raw[3] & 8 else 10
        raw = raw[:3] + bytes([raw[3] | 16]) + raw[4:end] + b"A" * 65536 + b"\0" + raw[end:]
    elif mutation == "trailer":
        raw = raw[:-8]
    elif mutation == "crc":
        raw = raw[:-8] + bytes([raw[-8] ^ 1]) + raw[-7:]
    elif mutation == "concatenated":
        raw += gzip.compress(b"hidden member")
    else:
        raw = gzip.compress(gzip.decompress(raw) + b"\0" * 65536)
    path.write_bytes(raw)
    with pytest.raises(PackageError):
        with open_package(path, limits=PackageLimits(max_bytes=1024, max_files=4)):
            pytest.fail("invalid gzip envelope exposed records")


@pytest.mark.parametrize("tail", ["buffered", "appended", "second_tar", "duplicate_member"])
def test_tar_end_cannot_hide_payload_in_the_same_gzip_member(tmp_path, tail):
    import gzip

    path = tmp_path / "tail.tar.gz"
    write(path)
    raw = gzip.decompress(path.read_bytes())
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        members = list(archive)
    end = max(m.offset_data + ((m.size + 511) // 512) * 512 for m in members)
    if tail == "buffered":
        assert end + 1024 < tarfile.RECORDSIZE
        raw = raw[: end + 1024] + b"hidden" + raw[end + 1030 :]
    elif tail == "appended":
        raw += b"hidden"
    elif tail == "second_tar":
        raw += raw
    else:
        first = members[0]
        raw += raw[: first.offset_data + ((first.size + 511) // 512) * 512]
    path.write_bytes(gzip.compress(raw))
    with pytest.raises(PackageError, match="non-zero data after tar end marker"):
        with open_package(path):
            pytest.fail("hidden tar payload accepted")


@pytest.mark.parametrize("padding", [0, 512, 513, 1024, 1536])
def test_tar_end_requires_two_zero_blocks_and_aligned_padding(tmp_path, padding):
    import gzip

    path = tmp_path / "padding.tar.gz"
    expected = write(path)
    raw = gzip.decompress(path.read_bytes())
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        end = max(m.offset_data + ((m.size + 511) // 512) * 512 for m in archive)
    path.write_bytes(gzip.compress(raw[:end] + b"\0" * padding))
    if padding >= 1024 and padding % 512 == 0:
        with open_package(path) as loaded:
            assert loaded.manifest == expected
    else:
        with pytest.raises(PackageError):
            with open_package(path):
                pytest.fail("incomplete tar ending accepted")


def test_invalid_header_cannot_be_treated_as_tar_end(tmp_path):
    import gzip

    path = tmp_path / "header.tar.gz"
    write(path)
    raw = gzip.decompress(path.read_bytes())
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        end = max(m.offset_data + ((m.size + 511) // 512) * 512 for m in archive)
    damaged = bytearray(raw)
    header = tarfile.TarInfo("manifest.json").tobuf(format=tarfile.USTAR_FORMAT)
    damaged[end : end + 512] = header
    damaged[end + 148] = ord("x")  # Invalid checksum; still followed by zero padding.
    path.write_bytes(gzip.compress(damaged))
    with pytest.raises(PackageError, match="invalid or truncated tar header"):
        with open_package(path):
            pytest.fail("invalid header accepted as EOF")


def test_export_normalizes_high_owner_ids(tmp_path, monkeypatch):
    original = tarfile.TarFile.gettarinfo

    def high_owner(self, *args, **kwargs):
        info = original(self, *args, **kwargs)
        info.uid = info.gid = 2097152
        info.uname = info.gname = 'host-specific-owner'
        return info

    monkeypatch.setattr(tarfile.TarFile, 'gettarinfo', high_owner)
    path = tmp_path / 'portable.tar.gz'
    write(path)
    with tarfile.open(path) as archive:
        for member in archive:
            assert (member.uid, member.gid, member.uname, member.gname) == (0, 0, '', '')
    with open_package(path):
        pass


def test_ustar_size_boundary_and_early_blob_rejection(tmp_path):
    from service.migration.package import USTAR_MAX_MEMBER_BYTES, check_member_size

    check_member_size(USTAR_MAX_MEMBER_BYTES)
    with pytest.raises(PackageError, match='USTAR'):
        check_member_size(USTAR_MAX_MEMBER_BYTES + 1)
    blob = tmp_path / 'large.bin'
    with blob.open('wb') as stream:
        stream.truncate(9 * 1024**3)
    destination = tmp_path / 'rejected.tar.gz'
    with pytest.raises(PackageError, match='USTAR'):
        write(destination, blobs=[blob])
    assert not destination.exists()


def test_dataset_member_limit_is_cumulative(tmp_path):
    path = tmp_path / 'rejected.tar.gz'
    with pytest.raises(PackageError, match='USTAR'):
        write(path, datasets([{'text': 'x' * 60}] * 2),
              limits=PackageLimits(max_member_bytes=100))
    assert not path.exists()
