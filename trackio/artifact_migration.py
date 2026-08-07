"""Copy and verify Trackio's local CAS into an S3-compatible artifact store."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from trackio import cas, utils
from trackio.artifact_storage import ArtifactStore, ArtifactStoreError


@dataclass(frozen=True)
class LocalBlob:
    project: str
    digest: str
    path: Path
    size_bytes: int


def _artifacts_root(source: str | Path) -> Path:
    path = Path(source).resolve()
    return path / "artifacts" if path.name != "artifacts" else path


def iter_local_blobs(source: str | Path, projects: Iterable[str] = ()) -> list[LocalBlob]:
    root = _artifacts_root(source)
    if not root.is_dir():
        raise FileNotFoundError(f"Trackio artifact directory does not exist: {root}")
    selected = {utils.canonical_project_name(project) for project in projects}
    blobs: list[LocalBlob] = []
    for project_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        project = utils.canonical_project_name(project_dir.name)
        if selected and project not in selected:
            continue
        sha_root = project_dir / "blobs" / "sha256"
        if not sha_root.is_dir():
            continue
        for path in sorted(sha_root.rglob("*")):
            if not path.is_file() or cas.is_partial_blob(path.name):
                continue
            digest = path.name
            if not cas.SHA256_DIGEST_RE.fullmatch(digest):
                continue
            blobs.append(LocalBlob(project, digest, path, path.stat().st_size))
    return blobs


def _source_digest(blobs: Iterable[LocalBlob]) -> str:
    payload = "\n".join(
        f"{blob.project}\0{blob.digest}\0{blob.size_bytes}"
        for blob in sorted(blobs, key=lambda item: (item.project, item.digest))
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _store_identity(store: ArtifactStore) -> dict[str, str]:
    """Return a redacted identity that binds a receipt to one target store."""

    identity = {"backend": type(store).__name__}
    for name in ("endpoint_url", "bucket", "prefix"):
        value = getattr(store, name, None)
        if value is not None:
            identity[name] = str(value)
    return identity


def _write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _load_verified_receipt(
    path: str | Path, source_digest: str, target_identity: dict[str, str]
) -> dict[str, Any]:
    try:
        receipt = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read migration receipt: {path}") from error
    if receipt.get("source_digest_set_sha256") != source_digest:
        raise ValueError("Migration receipt does not match the current local digest set")
    if receipt.get("target") != target_identity:
        raise ValueError("Migration receipt does not match the current target store")
    if receipt.get("verification") != "complete":
        raise ValueError("Migration receipt is not a complete verification receipt")
    return receipt


def migrate_local_artifacts(
    source: str | Path,
    store: ArtifactStore,
    receipt_path: str | Path,
    *,
    projects: Iterable[str] = (),
    dry_run: bool = False,
    verify_only: bool = False,
    delete_local: bool = False,
    verified_receipt: str | Path | None = None,
) -> dict[str, Any]:
    blobs = iter_local_blobs(source, projects)
    source_digest = _source_digest(blobs)
    if delete_local:
        if not verified_receipt:
            raise ValueError("--delete-local requires --verified-receipt")
        _load_verified_receipt(verified_receipt, source_digest, _store_identity(store))
        if dry_run or verify_only:
            raise ValueError("--delete-local cannot be combined with --dry-run or --verify-only")

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    verified = 0
    copied = 0
    already_present = 0
    for blob in blobs:
        record = {
            "project": blob.project,
            "digest": blob.digest,
            "source": str(blob.path),
            "size_bytes": blob.size_bytes,
        }
        try:
            if dry_run:
                record["status"] = "planned"
            elif verify_only:
                store.verify(blob.project, blob.digest, blob.size_bytes)
                record["status"] = "verified"
                verified += 1
            else:
                existed = store.has(blob.project, blob.digest)
                store.put_file(blob.project, blob.digest, blob.path)
                store.verify(blob.project, blob.digest, blob.size_bytes)
                record["status"] = "already_present" if existed else "copied"
                copied += 0 if existed else 1
                already_present += 1 if existed else 0
                verified += 1
        except (ArtifactStoreError, OSError, ValueError) as error:
            record["status"] = "failed"
            record["error"] = str(error)
            failures.append(record)
        records.append(record)

    if failures:
        verification = "failed"
    elif dry_run:
        verification = "not_run"
    else:
        verification = "complete"

    deleted = 0
    if delete_local:
        if failures or verification != "complete":
            raise ValueError("Refusing local deletion because migration verification failed")
        for blob in blobs:
            blob.path.unlink()
            deleted += 1

    receipt = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "source_root": str(_artifacts_root(source)),
        "source_digest_set_sha256": source_digest,
        "target": _store_identity(store),
        "projects": sorted({blob.project for blob in blobs}),
        "object_count": len(blobs),
        "source_bytes": sum(blob.size_bytes for blob in blobs),
        "copied_count": copied,
        "already_present_count": already_present,
        "verified_count": verified,
        "deleted_local_count": deleted,
        "verification": verification,
        "dry_run": dry_run,
        "verify_only": verify_only,
        "delete_local": delete_local,
        "failures": failures,
        "objects": records,
    }
    _write_receipt(Path(receipt_path), receipt)
    return receipt


__all__ = ["LocalBlob", "iter_local_blobs", "migrate_local_artifacts"]
