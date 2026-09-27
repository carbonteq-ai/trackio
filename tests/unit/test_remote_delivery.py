"""Remote delivery stays bounded, keeps failed entries and survives slow uploads."""

from __future__ import annotations

import hashlib
import time
from concurrent.futures import Future

import httpx
import orjson
import pytest

from trackio import remote_client
from trackio import run as run_module
from trackio.remote_client import _TrackioHTTPClient
from trackio.run import Run, _bounded_batches


def _entry(index: int, size: int) -> dict:
    return {
        "project": "proj",
        "run": "r",
        "run_id": "r",
        "metrics": {"traces/verifiers": {"payload": "x" * size, "i": index}},
        "step": index,
        "log_id": f"log-{index}",
    }


def test_bounded_batches_split_by_bytes_and_keep_order():
    entries = [_entry(index, 3_000) for index in range(10)]

    batches = _bounded_batches(entries, max_bytes=10_000)

    assert [entry for batch in batches for entry in batch] == entries
    assert all(len(orjson.dumps(batch)) <= 10_000 for batch in batches)
    assert len(batches) == 4
    oversized = [_entry(0, 50_000)]
    assert _bounded_batches(oversized, max_bytes=10_000) == [oversized]


def test_failed_log_request_keeps_every_unsent_entry(monkeypatch):
    class Client:
        def __init__(self):
            self.sent: list[list[dict]] = []

        def predict(self, *, api_name, logs, **kwargs):
            if self.sent:
                raise httpx.ReadTimeout("proxy gave up")
            self.sent.append(logs)

    client = Client()
    run = Run(
        url="https://trackio.invalid",
        project="proj",
        client=client,
        name="bounded-run",
        server_base_url="https://trackio.invalid",
    )
    monkeypatch.setattr(run, "_ensure_sender_alive", lambda: None)
    monkeypatch.setattr(
        run_module,
        "_bounded_batches",
        lambda items: _bounded_batches(items, max_bytes=10_000),
    )
    kept: list[dict] = []
    monkeypatch.setattr(run, "_persist_logs_locally", kept.extend)
    warnings: list[str] = []
    monkeypatch.setattr(run_module, "_emit_nonfatal_warning", warnings.append)
    entries = [_entry(index, 3_000) for index in range(10)]
    run._queued_logs.extend(entries)
    run._stop_flag.set()

    run._batch_sender()

    # The first bounded request lands; the rest is kept for retry, not dropped.
    assert len(client.sent) == 1
    assert client.sent[0] + kept == entries
    assert len(warnings) == 1
    assert "ReadTimeout" in warnings[0] and "kept and retried" in warnings[0]


def _direct_session(parts_url: str, expires_in: int = 900) -> dict:
    return {
        "upload_id": "upload-0000000001",
        "chunk_count": 2,
        "chunk_size_bytes": 4,
        "parts": [
            {"index": 0, "part_number": 1, "url": f"{parts_url}/1", "headers": {}},
            {"index": 1, "part_number": 2, "url": f"{parts_url}/2", "headers": {}},
        ],
        "acknowledged_parts": [],
        "state": "uploading",
        "expires_in": expires_in,
    }


@pytest.mark.parametrize("cause", ["refused", "expired"])
def test_direct_upload_signs_parts_again_when_urls_expire(tmp_path, monkeypatch, cause):
    source = tmp_path / "blob.bin"
    source.write_bytes(b"abcdefgh")
    digest = "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
    opened: list[str] = []
    puts: list[str] = []

    def response(method, url, status=200, json=None, headers=None):
        return httpx.Response(
            status, request=httpx.Request(method, url), json=json, headers=headers
        )

    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **kwargs: response(
            "GET", url, json={"resumable": True, "direct_multipart": True}
        ),
    )

    def post(url, **kwargs):
        if str(url).endswith("/api/artifact-upload/direct/project"):
            generation = f"https://storage.invalid/g{len(opened)}"
            opened.append(generation)
            return response("POST", url, json=_direct_session(generation))
        if "/parts/" in str(url):
            return response("POST", url, json={})
        return response(
            "POST", url, json={"digest": digest, "size_bytes": source.stat().st_size}
        )

    clock = {"now": 1_000.0}

    def put(url, **kwargs):
        puts.append(str(url))
        if cause == "refused" and str(url) == "https://storage.invalid/g0/2":
            return response("PUT", url, status=403)
        clock["now"] += 500.0  # each part takes longer than half the URL lifetime
        return response("PUT", url, headers={"etag": f"etag-{len(puts)}"})

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(httpx, "put", put)
    monkeypatch.setattr(remote_client.time, "sleep", lambda _: None)
    if cause == "expired":
        monkeypatch.setattr(remote_client.time, "monotonic", lambda: clock["now"])

    client = _TrackioHTTPClient("https://trackio.invalid")
    assert client.upload_artifact_blob("project", digest, source) is True

    assert len(opened) == 2
    assert puts[0] == "https://storage.invalid/g0/1"
    assert puts[-1] == "https://storage.invalid/g1/2"


def _artifact_run(monkeypatch) -> Run:
    run = Run(
        url="https://trackio.invalid",
        project="proj",
        client=object(),
        name="artifact-run",
        server_base_url="https://trackio.invalid",
    )
    monkeypatch.setattr(run, "_ensure_sender_alive", lambda: None)
    monkeypatch.setattr(run_module, "ARTIFACT_PROGRESS_POLL_SECONDS", 0.01)
    return run


def test_artifact_drain_waits_while_uploads_make_progress(monkeypatch):
    run = _artifact_run(monkeypatch)
    future: Future = Future()
    run._artifact_futures["slow"] = future
    deadline = time.monotonic() + 0.3

    def progress(*_args, **_kwargs):
        if time.monotonic() >= deadline:
            future.set_result("committed")
        return time.monotonic()

    monkeypatch.setattr(run_module, "last_upload_progress", progress)

    assert run.flush_artifacts(timeout=0.1) == ("committed",)


def test_artifact_drain_times_out_when_uploads_stall(monkeypatch):
    run = _artifact_run(monkeypatch)
    run._artifact_futures["stuck"] = Future()
    monkeypatch.setattr(run_module, "last_upload_progress", lambda: 0.0)

    with pytest.raises(TimeoutError, match="no upload progress"):
        run.flush_artifacts(timeout=0.1)
