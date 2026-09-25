import pytest

from trackio import (
    Run,
    TraceFactUpdate,
    TracePayloadMeasure,
    TracePayloadQuery,
    VerifiersTrace,
)
from trackio import server as trackio_server
from trackio.sqlite_storage import SQLiteStorage
from trackio.trace_facts import projection_id


def _record(trace_id: str, *, setup: float, model: float, harness, step_start: float):
    return {
        "id": trace_id,
        "version": 3,
        "agent": {"model": "org/model"},
        "nodes": [{"message": {"role": "user", "content": "go"}}],
        "calls": [],
        "rewards": {"task": 1.0},
        "timing": {
            "start": step_start,
            "setup": {"start": step_start, "end": step_start + setup},
            "agent": {
                "start": step_start + setup,
                "end": step_start + setup + model + 1,
                "model": {"duration": model},
                "harness": {"duration": harness},
            },
        },
    }


def _facts(run: Run, trace_id: str, step: int) -> None:
    dimensions = {"rollout_step": step}
    run.upsert_trace_facts(
        TraceFactUpdate(
            trace_type="verifiers",
            external_id=trace_id,
            namespace="verifiers.trace",
            calculator_version="test.v1",
            projection_id=projection_id(
                {
                    "namespace": "verifiers.trace",
                    "calculator_version": "test.v1",
                    "dimensions": dimensions,
                    "measures": {},
                    "reward_components": [],
                    "provenance": {},
                    "state": "complete",
                }
            ),
            dimensions=dimensions,
            replace_reward_components=True,
        )
    )


_QUERY_MEASURES = (
    TracePayloadMeasure("inference_s", "$.timing.agent.model.duration"),
    TracePayloadMeasure("harness_s", "$.timing.agent.harness.duration"),
    TracePayloadMeasure("setup_s", "$.timing.setup.end", minus="$.timing.setup.start"),
    TracePayloadMeasure(
        "mean_setup_s",
        "$.timing.setup.end",
        minus="$.timing.setup.start",
        operation="mean",
    ),
)


@pytest.fixture
def timed_run(temp_dir):
    run = Run(url=None, project="proj", client=None, name="timed", space_id=None)
    rows = [
        ("a", 1, 13.0, 50.0, 2.0),
        ("b", 1, 0.5, 40.0, 1.0),
        ("c", 2, 0.4, 30.0, "not-a-number"),
    ]
    for trace_id, step, setup, model, harness in rows:
        run.log(
            {
                "rollout": VerifiersTrace(
                    _record(
                        trace_id,
                        setup=setup,
                        model=model,
                        harness=harness,
                        step_start=1000.0 * step,
                    )
                )
            }
        )
    run._flush_queues_inline()
    for trace_id, step, *_ in rows:
        _facts(run, trace_id, step)
    return run


def test_payload_measures_aggregate_per_group_with_coverage(timed_run):
    result = timed_run.aggregate_trace_payload(
        TracePayloadQuery(measures=_QUERY_MEASURES, group_by=("rollout_step",))
    )

    by_step = {bucket.dimensions["rollout_step"]: bucket for bucket in result.buckets}
    assert set(by_step) == {1, 2}
    step1, step2 = by_step[1], by_step[2]
    assert step1.trace_count == 2
    assert step1.values["inference_s"] == pytest.approx(90.0)
    assert step1.values["harness_s"] == pytest.approx(3.0)
    assert step1.values["setup_s"] == pytest.approx(13.5)
    assert step1.values["mean_setup_s"] == pytest.approx(6.75)
    assert step1.coverage == {
        "inference_s": 2,
        "harness_s": 2,
        "setup_s": 2,
        "mean_setup_s": 2,
    }
    # A non-numeric value is missing, never zero.
    assert step2.values["harness_s"] is None
    assert step2.coverage["harness_s"] == 0
    assert step2.values["inference_s"] == pytest.approx(30.0)


def test_payload_measures_filter_by_fact_dimensions(timed_run):
    result = SQLiteStorage.aggregate_trace_payload(
        "proj",
        "timed",
        TracePayloadQuery(measures=_QUERY_MEASURES, dimensions={"rollout_step": 2}),
        run_id=timed_run.id,
    )

    [bucket] = result.buckets
    assert bucket.trace_count == 1
    assert bucket.values["setup_s"] == pytest.approx(0.4)


def test_payload_measures_ungrouped_cover_the_whole_run(timed_run):
    [bucket] = timed_run.aggregate_trace_payload(
        TracePayloadQuery(measures=_QUERY_MEASURES)
    ).buckets

    assert bucket.trace_count == 3
    assert bucket.values["inference_s"] == pytest.approx(120.0)
    assert bucket.coverage["harness_s"] == 2


def test_server_endpoint_serializes_payload_aggregates(timed_run):
    response = trackio_server.get_trace_payload_aggregates(
        "proj",
        "timed",
        [
            {
                "key": "setup_s",
                "path": "$.timing.setup.end",
                "minus": "$.timing.setup.start",
            }
        ],
        group_by=["rollout_step"],
        run_id=timed_run.id,
    )

    assert [bucket["dimensions"]["rollout_step"] for bucket in response["buckets"]] == [
        1,
        2,
    ]
    assert response["buckets"][0]["values"]["setup_s"] == pytest.approx(13.5)


@pytest.mark.parametrize(
    "path",
    ["$", "timing.setup", "$.a[0]", "$.a.*", "$.a;DROP", "$." + ".".join("k" * 9)],
)
def test_payload_paths_are_bounded_object_key_chains(path):
    with pytest.raises(ValueError, match="unsupported payload path"):
        TracePayloadMeasure("value", path)


def test_payload_queries_reject_reward_component_dimensions_and_duplicates():
    measure = TracePayloadMeasure("value", "$.a")
    with pytest.raises(ValueError, match="unsupported dimension"):
        TracePayloadQuery(measures=(measure,), group_by=("reward_component_name",))
    with pytest.raises(ValueError, match="unique"):
        TracePayloadQuery(measures=(measure, measure))
    with pytest.raises(ValueError, match="between 1 and"):
        TracePayloadQuery(measures=())
    with pytest.raises(ValueError, match="lower_snake_case"):
        TracePayloadMeasure("Bad Key", "$.a")
