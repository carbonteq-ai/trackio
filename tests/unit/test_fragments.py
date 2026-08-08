import time
from concurrent.futures import ThreadPoolExecutor

import trackio
from trackio import Run, fragments, utils
from trackio.remote_client import RemoteClient as Client
from trackio.sqlite_storage import SQLiteStorage


def make_metric_entries(project="proj", run="run1", run_id="rid1", n=3):
    return [
        {
            "project": project,
            "run": run,
            "run_id": run_id,
            "metrics": {"loss": 1.0 / (i + 1)},
            "step": i,
            "timestamp": f"2026-06-10T00:00:0{i}+00:00",
            "config": {"lr": 0.1} if i == 0 else None,
            "log_id": f"log-{i}",
        }
        for i in range(n)
    ]


def test_metric_fragment_roundtrip_and_idempotent_import(temp_dir):
    records = [fragments.metric_record(e) for e in make_metric_entries()]
    data = fragments.FragmentWriter.serialize_records(records)
    parsed = fragments.parse_fragment_bytes(data)
    assert len(parsed) == 3

    assert fragments.import_records(parsed) == 3
    logs = SQLiteStorage.get_logs("proj", "run1")
    assert len(logs) == 3
    assert logs[0]["loss"] == 1.0
    assert [log["step"] for log in logs] == [0, 1, 2]
    config = SQLiteStorage.get_run_config("proj", "run1")
    assert config["lr"] == 0.1

    fragments.import_records(parsed)
    assert len(SQLiteStorage.get_logs("proj", "run1")) == 3


def test_parse_tolerates_corrupt_and_unknown_lines():
    records = [fragments.metric_record(e) for e in make_metric_entries(n=2)]
    data = fragments.FragmentWriter.serialize_records(records)
    data += b'{"kind": "unknown-kind"}\n'
    data += b'{"kind": "metric", "project": "p", "truncated...'
    parsed = fragments.parse_fragment_bytes(data)
    assert len(parsed) == 2
    assert all(r["kind"] == "metric" for r in parsed)


def test_system_and_alert_fragment_roundtrip(temp_dir):
    system_entries = [
        {
            "project": "proj",
            "run": "run1",
            "run_id": "rid1",
            "metrics": {"gpu_util": 0.5},
            "timestamp": "2026-06-10T00:00:00+00:00",
            "log_id": "sys-0",
        }
    ]
    alert_entries = [
        {
            "project": "proj",
            "run": "run1",
            "run_id": "rid1",
            "title": "loss spike",
            "text": "loss exploded",
            "level": "ERROR",
            "step": 5,
            "timestamp": "2026-06-10T00:00:01+00:00",
            "alert_id": "alert-0",
        }
    ]
    records = [fragments.system_metric_record(e) for e in system_entries] + [
        fragments.alert_record(e) for e in alert_entries
    ]
    parsed = fragments.parse_fragment_bytes(
        fragments.FragmentWriter.serialize_records(records)
    )
    assert fragments.import_records(parsed) == 2

    system_logs = SQLiteStorage.get_system_logs("proj", "run1")
    assert len(system_logs) == 1
    assert system_logs[0]["gpu_util"] == 0.5

    alerts = SQLiteStorage.get_alerts("proj")
    assert len(alerts) == 1
    assert alerts[0]["title"] == "loss spike"
    assert alerts[0]["level"] == "ERROR"

    fragments.import_records(parsed)
    assert len(SQLiteStorage.get_alerts("proj")) == 1


def test_write_local_and_import_inbox_dir(temp_dir):
    writer = fragments.FragmentWriter()
    records = [fragments.metric_record(e) for e in make_metric_entries()]
    fragment_path = writer.write_local(records)
    assert fragment_path is not None and fragment_path.exists()
    assert fragment_path.suffix == ".jsonl"
    assert list(fragments.local_inbox_dir().rglob("*.tmp")) == []

    assert fragments.import_inbox_dir() == 3
    assert not fragment_path.exists()
    assert len(SQLiteStorage.get_logs("proj", "run1")) == 3
    assert fragments.import_inbox_dir() == 0


def test_import_inbox_dir_claims_fragments_for_concurrent_workers(temp_dir):
    writer = fragments.FragmentWriter()
    for index in range(8):
        entries = make_metric_entries(project="parallel", run=f"run-{index}")
        writer.write_local([fragments.metric_record(entry) for entry in entries])

    def import_one() -> int:
        total = 0
        while True:
            count = fragments.import_inbox_dir(max_files=1)
            if not count:
                return total
            total += count

    with ThreadPoolExecutor(max_workers=4) as executor:
        imported = sum(executor.map(lambda _: import_one(), range(4)))

    assert imported == 24
    assert not list(fragments.local_inbox_dir().rglob("*.jsonl"))
    assert not list(fragments.local_inbox_dir().rglob("*.processing"))


def test_claimed_batch_prioritizes_scalar_records_over_trace_records(temp_dir, monkeypatch):
    writer = fragments.FragmentWriter()
    writer.write_local(
        [
            fragments.metric_record(
                {
                    "project": "priority",
                    "run": "run",
                    "run_id": "rid",
                    "metrics": {"traces/verifiers": {"payload": "large"}},
                    "step": 0,
                }
            )
        ]
    )
    writer.write_local(
        [
            fragments.metric_record(
                {
                    "project": "priority",
                    "run": "run",
                    "run_id": "rid",
                    "metrics": {"train/rl/reward_mean": 0.25},
                    "step": 1,
                }
            )
        ]
    )

    imported_batches = []

    def capture(records):
        imported_batches.append(records)
        return len(records)

    monkeypatch.setattr(fragments, "import_records", capture)
    claimed = fragments.claim_inbox_batch(max_files=2)
    assert len(claimed) == 2
    assert fragments.import_claimed_fragments(claimed) == 2
    assert imported_batches[0][0]["metrics"] == {"train/rl/reward_mean": 0.25}
    assert imported_batches[1][0]["metrics"] == {
        "traces/verifiers": {"payload": "large"}
    }
    assert not list(fragments.local_inbox_dir().rglob("*.processing"))


def test_local_run_jsonl_mode_writes_fragments(temp_dir, monkeypatch):
    monkeypatch.setenv("TRACKIO_STORAGE_MODE", "jsonl")
    run = Run(url=None, project="proj", client=None, name="run1", space_id=None)
    run.log({"x": 1})
    run.log({"x": 2})
    run.finish()

    assert SQLiteStorage.get_logs("proj", "run1") == []
    fragment_files = list(fragments.local_inbox_dir().rglob("*.jsonl"))
    assert fragment_files

    imported = fragments.import_inbox_dir()
    assert imported == 2
    logs = SQLiteStorage.get_logs("proj", "run1")
    assert len(logs) == 2
    assert logs[0]["x"] == 1
    assert logs[1]["step"] == 1
    config = SQLiteStorage.get_run_config("proj", "run1")
    assert config is not None


def test_network_filesystem_jsonl_end_to_end(temp_dir, monkeypatch):
    monkeypatch.setattr(utils, "_filesystem_type_for_path", lambda path: "lustre")
    project = "test-lustre-project"
    run_name = "lustre-run"

    trackio.init(project=project, name=run_name)
    trackio.log(metrics={"loss": 0.1})
    trackio.log(metrics={"loss": 0.05, "acc": 0.9})
    trackio.finish()

    assert SQLiteStorage.get_logs(project=project, run=run_name) == []
    assert list(fragments.local_inbox_dir().rglob("*.jsonl"))

    app, url, _, _ = trackio.show(block_thread=False, open_browser=False)
    try:
        client = Client(url, verbose=False)
        summary = None
        deadline = time.time() + 30
        while time.time() < deadline:
            summary = client.predict(
                project=project, run=run_name, api_name="/get_run_summary"
            )
            if summary and summary.get("num_logs") == 2:
                break
            time.sleep(1)
        assert summary["num_logs"] == 2

        logs = SQLiteStorage.get_logs(project=project, run=run_name)
        assert [entry["loss"] for entry in logs] == [0.1, 0.05]
        assert logs[1]["acc"] == 0.9
    finally:
        app.close()
