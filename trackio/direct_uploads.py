"""Restart-safe presigned multipart uploads for S3-compatible artifact stores."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from trackio import cas, utils
from trackio.artifact_storage import (
    S3ArtifactStore,
    get_artifact_store,
)
from trackio.resumable_uploads import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SESSION_TTL,
    IDEMPOTENCY_KEY_RE,
    UploadSessionError,
)


def _uploads_root(project: str) -> Path:
    return utils.project_artifacts_dir(utils.canonical_project_name(project)) / "direct-uploads"


def _session_dir(project: str, upload_id: str) -> Path:
    if len(upload_id) < 16 or not all(char.isalnum() or char == "-" for char in upload_id):
        raise UploadSessionError("Invalid direct upload id.")
    return _uploads_root(project) / upload_id


def _metadata_path(project: str, upload_id: str) -> Path:
    return _session_dir(project, upload_id) / "session.json"


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=".session.", suffix=".tmp", delete=False
    ) as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _read_session(project: str, upload_id: str) -> dict[str, Any]:
    path = _metadata_path(project, upload_id)
    if not path.is_file():
        raise FileNotFoundError("Direct upload session does not exist.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise UploadSessionError("Direct upload session metadata is invalid.") from error
    if payload.get("project") != utils.canonical_project_name(project):
        raise UploadSessionError("Direct upload session project does not match.")
    return payload


def _upload_id(project: str, idempotency_key: str) -> str:
    if not IDEMPOTENCY_KEY_RE.fullmatch(idempotency_key):
        raise UploadSessionError("Idempotency key must be 16-128 URL-safe alphanumeric characters.")
    return hashlib.sha256(
        f"direct\0{utils.canonical_project_name(project)}\0{idempotency_key}".encode()
    ).hexdigest()


def _part_count(size_bytes: int, chunk_size: int) -> int:
    return max(1, (size_bytes + chunk_size - 1) // chunk_size)


def _store() -> S3ArtifactStore:
    store = get_artifact_store()
    if not isinstance(store, S3ArtifactStore):
        raise UploadSessionError("Direct uploads require TRACKIO_ARTIFACT_STORAGE_BACKEND=s3")
    return store


def _public_session(session: Mapping[str, Any], store: S3ArtifactStore) -> dict[str, Any]:
    parts = ()
    if session.get("state") != "completed" and session.get("provider_upload_id"):
        parts = tuple(
            store.presign_part(
                str(session["project"]),
                str(session["digest"]),
                str(session["provider_upload_id"]),
                index,
            )
            for index in range(1, int(session["chunk_count"]) + 1)
        )
    return {
        "upload_mode": "direct",
        "upload_id": session["upload_id"],
        "digest": session["digest"],
        "size_bytes": session["size_bytes"],
        "chunk_size_bytes": session["chunk_size_bytes"],
        "chunk_count": session["chunk_count"],
        "acknowledged_parts": session.get("parts", []),
        "parts": [
            {"part_number": part.number, "index": part.number - 1, "url": part.url, "headers": part.headers}
            for part in parts
        ],
        "state": session["state"],
        "already_present": bool(session.get("already_present", False)),
        "expires_at": session["expires_at"],
        "expires_in": store.presign_seconds if parts else 0,
    }


def create_or_resume_session(
    *, project: str, digest: str, size_bytes: int, idempotency_key: str
) -> dict[str, Any]:
    store = _store()
    project = utils.canonical_project_name(project)
    digest = str(cas.validate_digest(digest))
    if not isinstance(size_bytes, int) or size_bytes < 0:
        raise UploadSessionError("Upload size must be a non-negative integer.")
    upload_id = _upload_id(project, idempotency_key)
    metadata = _metadata_path(project, upload_id)
    if metadata.is_file():
        return _public_session(_read_session(project, upload_id), store)

    if store.has(project, digest):
        existing = store.stat(project, digest)
        if existing.size_bytes != size_bytes:
            raise UploadSessionError(
                "An existing artifact object has the same digest but a different size."
            )
        # A stale or externally modified object must not be treated as a
        # successful content-addressed upload merely because its length matches.
        store.verify(project, digest, size_bytes)
        now = datetime.now(UTC)
        session = {
            "schema_version": 1,
            "upload_id": upload_id,
            "project": project,
            "digest": digest,
            "size_bytes": size_bytes,
            "chunk_size_bytes": DEFAULT_CHUNK_SIZE,
            "chunk_count": _part_count(size_bytes, DEFAULT_CHUNK_SIZE),
            "parts": [],
            "state": "completed",
            "already_present": True,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "expires_at": (now + DEFAULT_SESSION_TTL).isoformat(),
        }
        _write_json_atomic(metadata, session)
        return _public_session(session, store)

    chunk_size = DEFAULT_CHUNK_SIZE
    multipart = store.begin_multipart(project, digest, size_bytes, _part_count(size_bytes, chunk_size))
    now = datetime.now(UTC)
    session = {
        "schema_version": 1,
        "upload_id": upload_id,
        "provider_upload_id": multipart.upload_id,
        "project": project,
        "digest": digest,
        "size_bytes": size_bytes,
        "chunk_size_bytes": chunk_size,
        "chunk_count": len(multipart.parts),
        "parts": [],
        "state": "uploading",
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
        "expires_at": (now + DEFAULT_SESSION_TTL).isoformat(),
    }
    _write_json_atomic(metadata, session)
    return _public_session(session, store)


def get_session(project: str, upload_id: str) -> dict[str, Any]:
    return _public_session(_read_session(project, upload_id), _store())


def acknowledge_part(project: str, upload_id: str, part_number: int, etag: str) -> dict[str, Any]:
    if part_number < 1 or not etag:
        raise UploadSessionError("A direct upload part requires a part number and ETag.")
    session = _read_session(project, upload_id)
    if session["state"] == "completed":
        return {"part_number": part_number, "etag": etag, "already_present": True}
    if part_number > int(session["chunk_count"]):
        raise UploadSessionError("Direct upload part is outside the session range.")
    parts = {int(part["part_number"]): str(part["etag"]) for part in session.get("parts", [])}
    existing = parts.get(part_number)
    if existing is not None and existing != etag:
        raise UploadSessionError("Direct upload part is already bound to a different ETag.")
    parts[part_number] = etag
    session["parts"] = [
        {"part_number": number, "etag": value} for number, value in sorted(parts.items())
    ]
    session["updated_at"] = datetime.now(UTC).isoformat()
    _write_json_atomic(_metadata_path(project, upload_id), session)
    return {"part_number": part_number, "etag": etag, "already_present": existing is not None}


def complete_session(project: str, upload_id: str, parts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    store = _store()
    session = _read_session(project, upload_id)
    if session["state"] == "completed":
        return {"digest": session["digest"], "size_bytes": session["size_bytes"], "already_present": True}
    expected = int(session["chunk_count"])
    normalized = [
        {"PartNumber": int(part.get("PartNumber", part.get("part_number", 0))), "ETag": str(part.get("ETag", part.get("etag", "")))}
        for part in parts
    ]
    if len({part["PartNumber"] for part in normalized}) != expected or {part["PartNumber"] for part in normalized} != set(range(1, expected + 1)):
        raise UploadSessionError("Direct upload completion is missing one or more parts.")
    try:
        store.complete_multipart(
            project,
            str(session["digest"]),
            str(session["provider_upload_id"]),
            normalized,
        )
        store.verify(project, str(session["digest"]), int(session["size_bytes"]))
    except Exception:
        try:
            store.delete(project, str(session["digest"]))
        except Exception:
            pass
        raise
    session["state"] = "completed"
    session["parts"] = [
        {"part_number": part["PartNumber"], "etag": part["ETag"]} for part in normalized
    ]
    session["completed_at"] = datetime.now(UTC).isoformat()
    session["updated_at"] = session["completed_at"]
    _write_json_atomic(_metadata_path(project, upload_id), session)
    return {"digest": session["digest"], "size_bytes": session["size_bytes"], "already_present": False}


def abort_session(project: str, upload_id: str) -> bool:
    store = _store()
    session = _read_session(project, upload_id)
    if session["state"] == "completed":
        raise UploadSessionError("Completed direct uploads cannot be aborted.")
    store.abort_multipart(project, str(session["digest"]), str(session["provider_upload_id"]))
    import shutil  # noqa: PLC0415

    shutil.rmtree(_session_dir(project, upload_id))
    return True


def expire_sessions(
    *, project: str, older_than: datetime, dry_run: bool = True
) -> dict[str, Any]:
    """Report or abort direct multipart sessions older than a cutoff."""

    if older_than.tzinfo is None:
        raise UploadSessionError("Expiry cutoff must be timezone-aware.")
    root = _uploads_root(project)
    expired: list[dict[str, Any]] = []
    if root.is_dir():
        for metadata in root.glob("*/session.json"):
            try:
                session = json.loads(metadata.read_text(encoding="utf-8"))
                updated = datetime.fromisoformat(session["updated_at"])
            except (OSError, KeyError, ValueError, json.JSONDecodeError):
                continue
            if session.get("state") == "completed" or updated >= older_than:
                continue
            entry = {
                "upload_id": session.get("upload_id"),
                "updated_at": session.get("updated_at"),
            }
            expired.append(entry)
            if not dry_run:
                try:
                    _store().abort_multipart(
                        project,
                        str(session["digest"]),
                        str(session["provider_upload_id"]),
                    )
                finally:
                    import shutil  # noqa: PLC0415

                    shutil.rmtree(metadata.parent)
    return {
        "project": utils.canonical_project_name(project),
        "dry_run": dry_run,
        "sessions": expired,
        "session_count": len(expired),
    }


__all__ = [
    "acknowledge_part",
    "abort_session",
    "complete_session",
    "create_or_resume_session",
    "expire_sessions",
    "get_session",
]
