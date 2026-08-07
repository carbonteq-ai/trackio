"""Provider-neutral artifact storage tests."""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from trackio.artifact_storage import (
    ArtifactNotFoundError,
    ArtifactVerificationError,
    LocalArtifactStore,
    S3ArtifactStore,
)
from trackio.asgi_app import create_trackio_starlette_app


class _FakePaginator:
    def __init__(self, objects):
        self._objects = objects

    def paginate(self, **kwargs):
        prefix = kwargs["Prefix"]
        yield {"Contents": [{"Key": k, "Size": len(v)} for k, v in self._objects.items() if k.startswith(prefix)]}


class _FakeS3:
    def __init__(self):
        self.objects = {}
        self.uploads = {}
        self.counter = 0

    def _key(self, bucket, key):
        return (bucket, key)

    def head_object(self, *, Bucket, Key):
        value = self.objects.get(self._key(Bucket, Key))
        if value is None:
            raise RuntimeError("404 Not Found")
        return {"ContentLength": len(value), "ETag": '"etag"'}

    def upload_file(self, source, bucket, key, ExtraArgs=None):
        self.objects[self._key(bucket, key)] = Path(source).read_bytes()

    def get_object(self, *, Bucket, Key):
        value = self.objects.get(self._key(Bucket, Key))
        if value is None:
            raise RuntimeError("404 Not Found")
        return {"Body": BytesIO(value)}

    def delete_object(self, *, Bucket, Key):
        self.objects.pop(self._key(Bucket, Key), None)

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return _FakePaginator({key: value for (bucket, key), value in self.objects.items() if bucket == "bucket"})

    def create_multipart_upload(self, *, Bucket, Key, Metadata):
        self.counter += 1
        upload_id = f"upload-{self.counter}"
        self.uploads[upload_id] = {"bucket": Bucket, "key": Key, "parts": {}}
        return {"UploadId": upload_id}

    def generate_presigned_url(self, operation, Params, ExpiresIn, HttpMethod):
        return f"https://s3.invalid/{operation}/{Params['UploadId'] if 'UploadId' in Params else Params['Key']}"

    def complete_multipart_upload(self, *, Bucket, Key, UploadId, MultipartUpload):
        upload = self.uploads[UploadId]
        assert upload["bucket"] == Bucket
        assert upload["key"] == Key
        # The fake stores bytes written by the test directly in ``parts``.
        payload = b"".join(upload["parts"][part["PartNumber"]] for part in MultipartUpload["Parts"])
        self.objects[(Bucket, Key)] = payload

    def abort_multipart_upload(self, **kwargs):
        self.uploads.pop(kwargs["UploadId"], None)


def test_local_store_verifies_and_round_trips(temp_dir):
    payload = b"model-weights"
    source = Path(temp_dir) / "weights.bin"
    source.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    store = LocalArtifactStore()

    result = store.put_file("project", digest, source)
    assert result.size_bytes == len(payload)
    assert store.has("project", digest)
    assert store.open("project", digest).read() == payload
    assert store.verify("project", digest, len(payload)).digest == digest


def test_local_store_rejects_wrong_digest(temp_dir):
    source = Path(temp_dir) / "weights.bin"
    source.write_bytes(b"payload")
    with pytest.raises(ArtifactVerificationError):
        LocalArtifactStore().put_file("project", "0" * 64, source)


def test_s3_store_uses_stable_keys_and_presigned_multipart_urls(temp_dir):
    payload = b"model-weights"
    source = Path(temp_dir) / "weights.bin"
    source.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    fake = _FakeS3()
    store = S3ArtifactStore(
        endpoint_url="http://rustfs.invalid:9000",
        bucket="bucket",
        prefix="prod",
        client=fake,
    )

    key = store.key("project", digest)
    assert key == f"prod/project/blobs/sha256/{digest[:2]}/{digest}"
    upload = store.begin_multipart("project", digest, len(payload), 2)
    assert upload.upload_id == "upload-1"
    assert [part.number for part in upload.parts] == [1, 2]
    assert all(part.url.startswith("https://s3.invalid/upload_part/") for part in upload.parts)

    result = store.put_file("project", digest, source)
    assert result.size_bytes == len(payload)
    assert store.verify("project", digest, len(payload)).digest == digest
    assert store.presign_get("project", digest).startswith("https://s3.invalid/get_object/")
    assert list(store.iter_project("project"))[0].digest == digest


def test_s3_store_missing_object_is_not_present():
    store = S3ArtifactStore(
        endpoint_url="http://rustfs.invalid:9000",
        bucket="bucket",
        client=_FakeS3(),
    )
    assert store.has("project", "0" * 64) is False
    with pytest.raises(ArtifactNotFoundError):
        store.open("project", "0" * 64)


def test_direct_upload_session_completes_and_verifies_object(monkeypatch, temp_dir):
    import trackio.direct_uploads as direct_uploads

    payload = b"direct-model-weights"
    digest = hashlib.sha256(payload).hexdigest()
    fake = _FakeS3()
    store = S3ArtifactStore(
        endpoint_url="http://rustfs.invalid:9000",
        bucket="bucket",
        prefix="test",
        client=fake,
    )
    monkeypatch.setenv("TRACKIO_ARTIFACT_STORAGE_BACKEND", "s3")
    monkeypatch.setattr(direct_uploads, "get_artifact_store", lambda: store)
    client = TestClient(create_trackio_starlette_app([], {}))

    session = client.post(
        "/api/artifact-upload/direct/project",
        json={
            "digest": digest,
            "size_bytes": len(payload),
            "idempotency_key": "direct-client-key-0001",
        },
    )
    assert session.status_code == 200, session.text
    body = session.json()
    assert body["upload_mode"] == "direct"
    assert len(body["parts"]) == 1

    provider_upload_id = next(iter(fake.uploads))
    fake.uploads[provider_upload_id]["parts"][1] = payload
    acknowledged = client.post(
        f"/api/artifact-upload/direct/project/{body['upload_id']}/parts/1",
        json={"etag": "etag-1"},
    )
    assert acknowledged.status_code == 200, acknowledged.text
    resumed = client.get(
        f"/api/artifact-upload/direct/project/{body['upload_id']}"
    )
    assert resumed.status_code == 200
    assert resumed.json()["acknowledged_parts"] == [{"part_number": 1, "etag": "etag-1"}]
    completed = client.post(
        f"/api/artifact-upload/direct/project/{body['upload_id']}",
        json={"parts": [{"PartNumber": 1, "ETag": "etag-1"}]},
    )
    assert completed.status_code == 200, completed.text
    assert completed.json()["digest"] == digest
    assert store.open("project", digest).read() == payload
