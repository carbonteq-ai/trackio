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


@pytest.fixture(autouse=True)
def _restore_upload_progress(monkeypatch):
    # Retries record upload progress in module state; keep it test-local.
    monkeypatch.setattr(
        remote_client, "_last_upload_progress", remote_client._last_upload_progress
    )


def _artifact_run(client) -> Run:
    return Run(
        url="https://trackio.invalid",
        project="proj",
        client=client,
        name="artifact-run",
        server_base_url="https://trackio.invalid",
    )


def test_artifact_commit_keeps_retrying_a_slow_server_until_it_commits(monkeypatch):
    # The commit is idempotent (keyed by manifest digest), so a server that is
    # slow for longer than a few short retries must not fail the run.
    class Client:
        def __init__(self):
            self.calls = 0

        def predict(self, *, api_name, **kwargs):
            self.calls += 1
            if self.calls <= 6:
                raise httpx.ReadTimeout("server busy")
            return {"manifest": kwargs["manifest"], "version": 0}

    client = Client()
    run = _artifact_run(client)
    sleeps: list[float] = []
    monkeypatch.setattr(run_module.time, "sleep", sleeps.append)
    warnings: list[str] = []
    monkeypatch.setattr(run_module, "_emit_nonfatal_warning", warnings.append)
    before = remote_client.last_upload_progress()

    record = run._artifact_log_with_retry(
        manifest=[{"path": "a", "digest": "sha256:" + "0" * 64}]
    )

    assert record["version"] == 0
    assert client.calls == 7
    assert sleeps == [0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
    assert warnings and "/artifact_log" in warnings[0]
    assert remote_client.last_upload_progress() >= before


def test_artifact_commit_stops_at_its_deadline_and_on_rejections(monkeypatch):
    class Timeouts:
        def predict(self, *, api_name, **kwargs):
            raise httpx.ReadTimeout("server busy")

    run = _artifact_run(Timeouts())
    clock = [0.0]
    monkeypatch.setattr(run_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        run_module.time,
        "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    monkeypatch.setattr(run_module, "_emit_nonfatal_warning", lambda message: None)
    with pytest.raises(httpx.ReadTimeout):
        run._artifact_log_with_retry(manifest=[])
    assert clock[0] < run_module.ARTIFACT_LOG_RETRY_DEADLINE_SECONDS

    class Rejects:
        calls = 0

        def predict(self, *, api_name, **kwargs):
            Rejects.calls += 1
            raise ValueError("Artifact manifest must be a non-empty list of entries.")

    with pytest.raises(ValueError):
        _artifact_run(Rejects())._artifact_log_with_retry(manifest=[])
    assert Rejects.calls == 1


def test_artifact_commit_requests_wait_longer_than_ordinary_calls():
    ordinary = remote_client._request_timeout_for_api(60, "/log")
    commit = remote_client._request_timeout_for_api(60, "/artifact_log")
    assert ordinary == 60
    assert commit.read == remote_client.ARTIFACT_LOG_TIMEOUT >= 300
    assert remote_client._request_timeout_for_api(600, "artifact_log") == 600
