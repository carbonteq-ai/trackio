from trackio.server import _contains_trace_metrics


def test_native_trace_batches_are_not_eligible_for_async_doris_acknowledgement():
    assert _contains_trace_metrics(
        [{"traces/verifiers": {"_type": "trackio.verifiers_trace"}}]
    )
    assert _contains_trace_metrics(
        [{"metric": 1.0}, {"traces/agent": {"_type": "trackio.trace"}}]
    )
    assert not _contains_trace_metrics([{"train/loss": 0.25}])
