from trackio import server as trackio_server
from trackio.trace_facts import TraceFactUpdate, projection_id


def _update_payload() -> dict:
    payload = {
        "namespace": "posttrain.train.reward",
        "calculator_version": "test.v1",
        "dimensions": {},
        "measures": {"algorithm_reward": 0.5},
        "reward_components": [],
        "provenance": {},
        "state": "complete",
    }
    return {
        "trace_type": "verifiers",
        "external_id": "trace-1",
        **payload,
        "projection_id": projection_id(payload),
    }


def test_native_trace_batches_use_the_durable_async_doris_inbox(monkeypatch):
    captured = []
    monkeypatch.setattr(trackio_server, "assert_can_write_metrics", lambda *args: None)
    monkeypatch.setattr(trackio_server, "_use_async_doris_writes", lambda: True)
    monkeypatch.setattr(
        trackio_server,
        "_enqueue_metric_fragment",
        lambda **payload: captured.append(payload),
    )

    trackio_server.bulk_log(
        None,
        [
            {
                "project": "proj",
                "run": "run",
                "run_id": "run-id",
                "metrics": {"traces/verifiers": {"_type": "trackio.verifiers_trace"}},
                "step": 1,
                "log_id": "log-1",
            }
        ],
        None,
    )

    assert captured[0]["metrics_list"] == [
        {"traces/verifiers": {"_type": "trackio.verifiers_trace"}}
    ]


def test_explicit_synchronous_trace_delivery_remains_available(monkeypatch):
    calls = []
    monkeypatch.setattr(trackio_server, "assert_can_write_metrics", lambda *args: None)
    monkeypatch.setattr(trackio_server, "_use_async_doris_writes", lambda: True)
    monkeypatch.setattr(
        trackio_server.Storage,
        "bulk_log",
        lambda **payload: calls.append(payload),
    )

    trackio_server.bulk_log(
        None,
        [
            {
                "project": "proj",
                "run": "run",
                "run_id": "run-id",
                "metrics": {"traces/verifiers": {"_type": "trackio.verifiers_trace"}},
                "step": 1,
                "log_id": "log-1",
            }
        ],
        None,
        synchronous=True,
    )

    assert calls[0]["project"] == "proj"


def test_trace_fact_enqueue_validates_then_writes_one_durable_fragment(monkeypatch):
    captured = []
    monkeypatch.setattr(trackio_server, "assert_can_write_metrics", lambda *args: None)
    monkeypatch.setattr(
        trackio_server,
        "_enqueue_trace_fact_fragments",
        lambda entries: captured.extend(entries),
    )

    response = trackio_server.enqueue_trace_facts(
        None,
        [
            {
                "project": "proj",
                "run": "run",
                "run_id": "run-id",
                "update": _update_payload(),
            }
        ],
        None,
    )

    assert response == {"accepted": 1}
    assert TraceFactUpdate.from_payload(captured[0]["update"]).external_id == "trace-1"
