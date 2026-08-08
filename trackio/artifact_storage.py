"""Provider-neutral storage for Trackio-owned artifact blobs.

The Trackio API owns artifact manifests and authorization.  This module owns
only the bytes behind those manifests.  The local implementation preserves the
historical content-addressed filesystem, while the S3 implementation supports
any S3-compatible endpoint (including RustFS) and presigned multipart upload
and download.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Protocol

from trackio import cas, utils

DEFAULT_S3_REGION = "us-east-1"
DEFAULT_S3_PREFIX = "trackio-artifacts"
DEFAULT_PRESIGN_SECONDS = 900
MAX_S3_MULTIPART_PARTS = 10_000
HASH_CHUNK_SIZE = cas.HASH_CHUNK_SIZE


class ArtifactStoreError(RuntimeError):
    """Base error for provider operations."""


class ArtifactNotFoundError(ArtifactStoreError):
    """The requested provider object does not exist."""


class ArtifactVerificationError(ArtifactStoreError):
    """An object exists but does not match its manifest identity."""


@dataclass(frozen=True)
class ArtifactObjectStat:
    project: str
    digest: str
    size_bytes: int
    key: str
    etag: str | None = None


@dataclass(frozen=True)
class MultipartPart:
    number: int
    url: str
    headers: dict[str, str]


@dataclass(frozen=True)
class MultipartUpload:
    upload_id: str
    key: str
    parts: tuple[MultipartPart, ...]
    expires_in: int


class ArtifactStore(Protocol):
    def has(self, project: str, digest: str) -> bool: ...

    def put_file(self, project: str, digest: str, source: Path) -> ArtifactObjectStat: ...

    def open(self, project: str, digest: str) -> BinaryIO: ...

    def stat(self, project: str, digest: str) -> ArtifactObjectStat: ...

    def delete(self, project: str, digest: str) -> None: ...

    def bytes_for_project(self, project: str) -> int: ...

    def begin_multipart(
        self, project: str, digest: str, size_bytes: int, part_count: int
    ) -> MultipartUpload: ...

    def complete_multipart(
        self,
        project: str,
        digest: str,
        upload_id: str,
        parts: Sequence[Mapping[str, Any]],
    ) -> ArtifactObjectStat: ...

    def abort_multipart(self, project: str, digest: str, upload_id: str) -> None: ...

    def presign_get(self, project: str, digest: str) -> str | None: ...

    def verify(
        self, project: str, digest: str, size_bytes: int
    ) -> ArtifactObjectStat: ...


def _canonical(project: str) -> str:
    return utils.canonical_project_name(project)


def _validated_digest(digest: str) -> str:
    return str(cas.validate_digest(digest))


def _hash_stream(stream: Any) -> tuple[str, int]:
    sha = hashlib.sha256()
    size = 0
    try:
        while chunk := stream.read(HASH_CHUNK_SIZE):
            sha.update(chunk)
            size += len(chunk)
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()
    return sha.hexdigest(), size


def _is_not_found(error: BaseException) -> bool:
    response = getattr(error, "response", None)
    code = response.get("Error", {}).get("Code") if isinstance(response, dict) else None
    if code in {"404", "NoSuchKey", "NoSuchBucket", "NotFound"}:
        return True
    return any(marker in str(error) for marker in ("404", "NoSuchKey", "NotFound"))


class LocalArtifactStore:
    """The historical Trackio local content-addressed store."""

    def _path(self, project: str, digest: str) -> Path:
        return cas.blob_path(_canonical(project), _validated_digest(digest))

    def has(self, project: str, digest: str) -> bool:
        return self._path(project, digest).is_file()

    def put_file(self, project: str, digest: str, source: Path) -> ArtifactObjectStat:
        digest = _validated_digest(digest)
        actual, size = cas.hash_file(source)
        if actual != digest:
            raise ArtifactVerificationError(
                f"Local source digest mismatch: claimed {digest}, computed {actual}"
            )
        target = self._path(project, digest)
        cas.stage_blob_from_file(source, digest, target)
        return ArtifactObjectStat(_canonical(project), digest, size, str(target))

    def open(self, project: str, digest: str) -> BinaryIO:
        path = self._path(project, digest)
        if not path.is_file():
            raise ArtifactNotFoundError(f"Artifact blob {digest} is not present")
        return path.open("rb")

    def stat(self, project: str, digest: str) -> ArtifactObjectStat:
        path = self._path(project, digest)
        if not path.is_file():
            raise ArtifactNotFoundError(f"Artifact blob {digest} is not present")
        return ArtifactObjectStat(_canonical(project), _validated_digest(digest), path.stat().st_size, str(path))

    def delete(self, project: str, digest: str) -> None:
        self._path(project, digest).unlink(missing_ok=True)

    def bytes_for_project(self, project: str) -> int:
        root = utils.project_artifacts_dir(_canonical(project)) / "blobs" / "sha256"
        return sum(path.stat().st_size for path in root.rglob("*") if path.is_file()) if root.is_dir() else 0

    def begin_multipart(self, project: str, digest: str, size_bytes: int, part_count: int) -> MultipartUpload:
        del project, digest, size_bytes, part_count
        raise ArtifactStoreError("Direct multipart uploads require an S3-compatible backend")

    def complete_multipart(self, project: str, digest: str, upload_id: str, parts: Sequence[Mapping[str, Any]]) -> ArtifactObjectStat:
        del project, digest, upload_id, parts
        raise ArtifactStoreError("Direct multipart uploads require an S3-compatible backend")

    def abort_multipart(self, project: str, digest: str, upload_id: str) -> None:
        del project, digest, upload_id

    def presign_get(self, project: str, digest: str) -> str | None:
        del project, digest
        return None

    def verify(self, project: str, digest: str, size_bytes: int) -> ArtifactObjectStat:
        stat = self.stat(project, digest)
        if stat.size_bytes != size_bytes:
            raise ArtifactVerificationError(
                f"Artifact size mismatch: claimed {size_bytes}, found {stat.size_bytes}"
            )
        actual, actual_size = _hash_stream(self.open(project, digest))
        if actual != _validated_digest(digest) or actual_size != size_bytes:
            raise ArtifactVerificationError(
                f"Artifact verification failed for {digest}: computed {actual}/{actual_size}"
            )
        return stat


class S3ArtifactStore:
    """S3-compatible artifact store with optional dependency injection for tests."""

    def __init__(
        self,
        *,
        endpoint_url: str,
        bucket: str,
        presign_endpoint_url: str | None = None,
        prefix: str = DEFAULT_S3_PREFIX,
        region: str = DEFAULT_S3_REGION,
        access_key: str | None = None,
        secret_key: str | None = None,
        presign_seconds: int = DEFAULT_PRESIGN_SECONDS,
        client: Any | None = None,
    ) -> None:
        if not endpoint_url or not bucket:
            raise ArtifactStoreError("S3 endpoint and bucket are required")
        self.endpoint_url = endpoint_url.rstrip("/")
        self.presign_endpoint_url = (presign_endpoint_url or endpoint_url).rstrip("/")
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.region = region
        self.access_key = access_key
        self.secret_key = secret_key
        self.presign_seconds = max(60, int(presign_seconds))
        self._client = client
        self._presign_client: Any | None = None

    def _new_client(self, endpoint_url: str) -> Any:
        try:
            import boto3  # noqa: PLC0415
            from botocore.config import Config  # noqa: PLC0415
        except ImportError as error:
            raise ArtifactStoreError(
                "The S3 artifact backend requires the 'boto3' dependency"
            ) from error
        return boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=self.region,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        )

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = self._new_client(self.endpoint_url)
        return self._client

    @property
    def presign_client(self) -> Any:
        if self.presign_endpoint_url == self.endpoint_url:
            return self.client
        if self._presign_client is None:
            self._presign_client = self._new_client(self.presign_endpoint_url)
        return self._presign_client

    def key(self, project: str, digest: str) -> str:
        project = _canonical(project)
        digest = _validated_digest(digest)
        suffix = f"{project}/blobs/sha256/{digest[:2]}/{digest}"
        return f"{self.prefix}/{suffix}" if self.prefix else suffix

    def _stat_from_head(self, project: str, digest: str, head: Mapping[str, Any]) -> ArtifactObjectStat:
        return ArtifactObjectStat(
            _canonical(project),
            _validated_digest(digest),
            int(head.get("ContentLength", 0)),
            self.key(project, digest),
            str(head.get("ETag", "")).strip('"') or None,
        )

    def stat(self, project: str, digest: str) -> ArtifactObjectStat:
        try:
            return self._stat_from_head(
                project, digest, self.client.head_object(Bucket=self.bucket, Key=self.key(project, digest))
            )
        except Exception as error:
            if _is_not_found(error):
                raise ArtifactNotFoundError(f"Artifact blob {digest} is not present") from error
            raise ArtifactStoreError(f"S3 HEAD failed for artifact {digest}") from error

    def has(self, project: str, digest: str) -> bool:
        try:
            self.stat(project, digest)
            return True
        except ArtifactNotFoundError:
            return False

    def put_file(self, project: str, digest: str, source: Path) -> ArtifactObjectStat:
        digest = _validated_digest(digest)
        actual, size = cas.hash_file(source)
        if actual != digest:
            raise ArtifactVerificationError(
                f"S3 source digest mismatch: claimed {digest}, computed {actual}"
            )
        key = self.key(project, digest)
        try:
            existing = self.stat(project, digest)
            if existing.size_bytes == size:
                return self.verify(project, digest, size)
        except ArtifactNotFoundError:
            pass
        try:
            self.client.upload_file(
                str(source),
                self.bucket,
                key,
                ExtraArgs={"Metadata": {"sha256": digest}},
            )
            return self.verify(project, digest, size)
        except ArtifactStoreError:
            raise
        except Exception as error:
            raise ArtifactStoreError(f"S3 upload failed for artifact {digest}") from error

    def open(self, project: str, digest: str) -> BinaryIO:
        try:
            body = self.client.get_object(Bucket=self.bucket, Key=self.key(project, digest))["Body"]
        except Exception as error:
            if _is_not_found(error):
                raise ArtifactNotFoundError(f"Artifact blob {digest} is not present") from error
            raise ArtifactStoreError(f"S3 GET failed for artifact {digest}") from error
        return body

    def verify(self, project: str, digest: str, size_bytes: int) -> ArtifactObjectStat:
        stat = self.stat(project, digest)
        if stat.size_bytes != size_bytes:
            raise ArtifactVerificationError(
                f"Artifact size mismatch: claimed {size_bytes}, found {stat.size_bytes}"
            )
        try:
            actual, actual_size = _hash_stream(self.open(project, digest))
        except ArtifactStoreError:
            raise
        except Exception as error:
            raise ArtifactStoreError(f"S3 verification read failed for artifact {digest}") from error
        if actual != _validated_digest(digest) or actual_size != size_bytes:
            raise ArtifactVerificationError(
                f"Artifact verification failed for {digest}: computed {actual}/{actual_size}"
            )
        return stat

    def delete(self, project: str, digest: str) -> None:
        try:
            self.client.delete_object(Bucket=self.bucket, Key=self.key(project, digest))
        except Exception as error:
            if _is_not_found(error):
                return
            raise ArtifactStoreError(f"S3 delete failed for artifact {digest}") from error

    def bytes_for_project(self, project: str) -> int:
        prefix = self.key(project, "0" * 64).rsplit("/", 2)[0] + "/"
        total = 0
        try:
            paginator = self.client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                total += sum(int(item.get("Size", 0)) for item in page.get("Contents", []))
        except Exception as error:
            raise ArtifactStoreError(f"S3 listing failed for project {_canonical(project)}") from error
        return total

    def begin_multipart(self, project: str, digest: str, size_bytes: int, part_count: int) -> MultipartUpload:
        if size_bytes < 0 or part_count < 1 or part_count > MAX_S3_MULTIPART_PARTS:
            raise ArtifactStoreError("Invalid multipart size or part count")
        digest = _validated_digest(digest)
        key = self.key(project, digest)
        try:
            response = self.client.create_multipart_upload(
                Bucket=self.bucket,
                Key=key,
                Metadata={"sha256": digest, "size-bytes": str(size_bytes)},
            )
            upload_id = str(response["UploadId"])
            parts = tuple(self.presign_part(project, digest, upload_id, index) for index in range(1, part_count + 1))
            return MultipartUpload(upload_id, key, parts, self.presign_seconds)
        except Exception as error:
            raise ArtifactStoreError("S3 multipart upload initialization failed") from error

    def presign_part(
        self, project: str, digest: str, upload_id: str, part_number: int
    ) -> MultipartPart:
        if part_number < 1:
            raise ArtifactStoreError("Multipart part number must be positive")
        try:
            return MultipartPart(
                number=part_number,
                url=self.presign_client.generate_presigned_url(
                    "upload_part",
                    Params={
                        "Bucket": self.bucket,
                        "Key": self.key(project, digest),
                        "UploadId": upload_id,
                        "PartNumber": part_number,
                    },
                    ExpiresIn=self.presign_seconds,
                    HttpMethod="PUT",
                ),
                headers={},
            )
        except Exception as error:
            raise ArtifactStoreError("S3 multipart part URL generation failed") from error

    def complete_multipart(
        self,
        project: str,
        digest: str,
        upload_id: str,
        parts: Sequence[Mapping[str, Any]],
    ) -> ArtifactObjectStat:
        digest = _validated_digest(digest)
        normalized = []
        for part in parts:
            number = int(part.get("PartNumber", part.get("number", 0)))
            etag = str(part.get("ETag", part.get("etag", "")))
            if number < 1 or not etag:
                raise ArtifactStoreError("Multipart completion requires part numbers and ETags")
            normalized.append({"PartNumber": number, "ETag": etag})
        normalized.sort(key=lambda part: part["PartNumber"])
        try:
            self.client.complete_multipart_upload(
                Bucket=self.bucket,
                Key=self.key(project, digest),
                UploadId=upload_id,
                MultipartUpload={"Parts": normalized},
            )
            return self.stat(project, digest)
        except Exception as error:
            raise ArtifactStoreError("S3 multipart completion failed") from error

    def abort_multipart(self, project: str, digest: str, upload_id: str) -> None:
        try:
            self.client.abort_multipart_upload(
                Bucket=self.bucket,
                Key=self.key(project, digest),
                UploadId=upload_id,
            )
        except Exception as error:
            if _is_not_found(error):
                return
            raise ArtifactStoreError("S3 multipart abort failed") from error

    def presign_get(self, project: str, digest: str) -> str | None:
        try:
            return self.presign_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": self.key(project, digest)},
                ExpiresIn=self.presign_seconds,
                HttpMethod="GET",
            )
        except Exception as error:
            raise ArtifactStoreError("S3 download URL generation failed") from error

    def iter_project(self, project: str) -> Iterator[ArtifactObjectStat]:
        prefix = self.key(project, "0" * 64).rsplit("/", 2)[0] + "/"
        try:
            paginator = self.client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                for item in page.get("Contents", []):
                    key = str(item.get("Key", ""))
                    digest = key.rsplit("/", 1)[-1]
                    if cas.SHA256_DIGEST_RE.fullmatch(digest):
                        yield ArtifactObjectStat(
                            _canonical(project), digest, int(item.get("Size", 0)), key
                        )
        except Exception as error:
            raise ArtifactStoreError(f"S3 listing failed for project {_canonical(project)}") from error


def selected_artifact_backend() -> str:
    backend = os.environ.get("TRACKIO_ARTIFACT_STORAGE_BACKEND", "local").strip().lower()
    if backend not in {"local", "s3"}:
        raise ArtifactStoreError("TRACKIO_ARTIFACT_STORAGE_BACKEND must be 'local' or 's3'")
    return backend


def get_artifact_store() -> ArtifactStore:
    if selected_artifact_backend() == "local":
        return LocalArtifactStore()
    return S3ArtifactStore(
        endpoint_url=os.environ.get("TRACKIO_ARTIFACT_S3_ENDPOINT", ""),
        presign_endpoint_url=os.environ.get("TRACKIO_ARTIFACT_S3_PRESIGN_ENDPOINT") or None,
        bucket=os.environ.get("TRACKIO_ARTIFACT_S3_BUCKET", ""),
        prefix=os.environ.get("TRACKIO_ARTIFACT_S3_PREFIX", DEFAULT_S3_PREFIX),
        region=os.environ.get("TRACKIO_ARTIFACT_S3_REGION", DEFAULT_S3_REGION),
        access_key=os.environ.get("TRACKIO_ARTIFACT_S3_ACCESS_KEY"),
        secret_key=os.environ.get("TRACKIO_ARTIFACT_S3_SECRET_KEY"),
        presign_seconds=int(
            os.environ.get("TRACKIO_ARTIFACT_S3_PRESIGN_SECONDS", str(DEFAULT_PRESIGN_SECONDS))
        ),
    )


__all__ = [
    "ArtifactNotFoundError",
    "ArtifactObjectStat",
    "ArtifactStore",
    "ArtifactStoreError",
    "ArtifactVerificationError",
    "LocalArtifactStore",
    "MultipartPart",
    "MultipartUpload",
    "S3ArtifactStore",
    "get_artifact_store",
    "selected_artifact_backend",
]
