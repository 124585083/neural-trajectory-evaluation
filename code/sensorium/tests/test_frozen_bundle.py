"""Synthetic relocation checks; no model fitting or scientific scoring."""
import pytest

from sensorium_runtime import bundle
from trajectory_model_eval.unified_contract import file_sha256


def source_record(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "frozen.bin").write_bytes(b"unchanged frozen parameters")
    return source, {"total_bytes": 27, "files": {"frozen.bin": {
        "sha256": file_sha256(source / "frozen.bin"), "bytes": 27}}}


def test_copy_preserves_source_and_verifies_relocated_root(tmp_path, monkeypatch):
    source, record = source_record(tmp_path)
    destination, seen = tmp_path / "bundle", []
    monkeypatch.setattr(bundle, "preflight", lambda path: seen.append(path))
    bundle.copy_bundle(source, destination, record)
    assert (source / "frozen.bin").read_bytes() == (destination / "frozen.bin").read_bytes()
    assert seen == [destination]
    assert (destination / "bundle_manifest.json").exists()


def test_copy_rejects_changed_source_before_creating_destination(tmp_path):
    source, record = source_record(tmp_path)
    (source / "frozen.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        bundle.copy_bundle(source, tmp_path / "bundle", record)
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize("destination", ["source", "source/sub", "."])
def test_copy_rejects_overlapping_source(tmp_path, destination):
    source, record = source_record(tmp_path)
    with pytest.raises(ValueError, match="separate"):
        bundle.copy_bundle(source, tmp_path / destination, record)


def test_copy_never_overwrites_existing_destination(tmp_path):
    source, record = source_record(tmp_path)
    destination = tmp_path / "bundle"
    destination.mkdir()
    (destination / "existing.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        bundle.copy_bundle(source, destination, record)
    assert (destination / "existing.txt").read_text() == "keep"


@pytest.mark.parametrize("relative", ["../outside.bin", "sub/../../outside.bin"])
def test_copy_rejects_path_traversal_before_writes(tmp_path, relative):
    source, record = source_record(tmp_path)
    record["files"][relative] = record["files"].pop("frozen.bin")
    with pytest.raises(ValueError, match="within source and destination"):
        bundle.copy_bundle(source, tmp_path / "bundle", record)
    assert not (tmp_path / "bundle").exists()


def test_copy_rejects_absolute_record_path(tmp_path):
    source, record = source_record(tmp_path)
    record["files"][str(source / "frozen.bin")] = record["files"].pop("frozen.bin")
    with pytest.raises(ValueError, match="within source and destination"):
        bundle.copy_bundle(source, tmp_path / "bundle", record)
    assert not (tmp_path / "bundle").exists()
