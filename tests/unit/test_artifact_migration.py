"""Migration receipt and safe local deletion tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from trackio.artifact_migration import iter_local_blobs, migrate_local_artifacts
from trackio.artifact_storage import ArtifactObjectStat, ArtifactVerificationError


class _MemoryStore:
    def __init__(self):
        self.objects = {}

    def has(self, project, digest):
        return (project, digest) in self.objects

    def put_file(self, project, digest, source):
        payload = Path(source).read_bytes()
        actual = hashlib.sha256(payload).hexdigest()
        if actual != digest:
            raise ArtifactVerificationError("digest mismatch")
        self.objects[(project, digest)] = payload
        return ArtifactObjectStat(project, digest, len(payload), f"memory://{project}/{digest}")

    def verify(self, project, digest, size_bytes):
        payload = self.objects[(project, digest)]
        if len(payload) != size_bytes or hashlib.sha256(payload).hexdigest() != digest:
            raise ArtifactVerificationError("verification failed")
        return ArtifactObjectStat(project, digest, len(payload), f"memory://{project}/{digest}")


def _write_blob(root: Path, project: str, payload: bytes) -> Path:
    digest = hashlib.sha256(payload).hexdigest()
    path = root / "artifacts" / project / "blobs" / "sha256" / digest[:2] / digest
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    return path


def test_migration_dry_run_and_copy_are_idempotent(tmp_path):
    source = _write_blob(tmp_path, "project", b"weights")
    assert len(iter_local_blobs(tmp_path)) == 1
    store = _MemoryStore()
    first_receipt = tmp_path / "first.json"
    result = migrate_local_artifacts(tmp_path, store, first_receipt)
    assert result["verification"] == "complete"
    assert result["copied_count"] == 1
    assert source.exists()

    second_receipt = tmp_path / "second.json"
    second = migrate_local_artifacts(tmp_path, store, second_receipt)
    assert second["already_present_count"] == 1
    assert second["copied_count"] == 0


def test_migration_requires_matching_verified_receipt_before_delete(tmp_path):
    source = _write_blob(tmp_path, "project", b"weights")
    store = _MemoryStore()
    receipt = tmp_path / "verified.json"
    migrate_local_artifacts(tmp_path, store, receipt)

    with pytest.raises(ValueError, match="Could not read migration receipt"):
        migrate_local_artifacts(
            tmp_path,
            store,
            tmp_path / "delete.json",
            delete_local=True,
            verified_receipt=tmp_path / "missing.json",
        )

    result = migrate_local_artifacts(
        tmp_path,
        store,
        tmp_path / "delete.json",
        delete_local=True,
        verified_receipt=receipt,
    )
    assert result["deleted_local_count"] == 1
    assert not source.exists()
