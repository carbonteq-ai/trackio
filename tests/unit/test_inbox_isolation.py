from contextlib import contextmanager
from types import SimpleNamespace

import orjson
import pymysql
import pytest

import trackio
from trackio import fragments
from trackio.doris_pool import DorisConnectionPoolTimeout
from trackio.doris_storage import DorisStorage, DorisValueTooLargeError
from trackio.sqlite_storage import SQLiteStorage
from trackio.storage import Storage
from trackio.trace_facts import TraceFactUpdate, projection_id

STRICT_MODE_ERROR = pymysql.err.OperationalError(
    1105,
    "errCode = 2, detailMessage = Insert has filtered data in strict mode. "
    "first_error_msg: column_name[payload], the length of input string is too "
    "long than vec schema. limit length: 10485760; actual length: 15272937",
)


@pytest.fixture
def clock(monkeypatch):
    now = [1_800_000_000.0]
    monkeypatch.setattr(fragments, "time", SimpleNamespace(time=lambda: now[0]))
    return now


def _scalar_fragment(writer, run, value, step=0):
    return writer.write_local(
        [
            fragments.metric_record(
                {
                    "project": "proj",
                    "run": run,
                    "run_id": f"{run}-id",
                    "metrics": {"eval/reward_mean": value},
                    "step": step,
                    "log_id": f"{run}-{step}",
                }
            )
        ]
    )


def _trace_fragment(writer, run, trace_id, step=0):
    native = trackio.VerifiersTrace(
        {"id": trace_id, "version": 2, "nodes": []}
    )._to_dict("proj", run, step)
    return writer.write_local(
        [
            fragments.metric_record(
                {
                    "project": "proj",
                    "run": run,
                    "run_id": f"{run}-id",
                    "metrics": {
                        "traces/verifiers": native,
                        "eval/reward_mean": 0.75,
                        "eval/completed": 1,
                    },
                    "step": step,
                    "log_id": f"{run}-trace-{step}",
                }
            )
        ]
    )


def _fact_fragment(writer, run, trace_id):
    payload = {
        "namespace": "posttrain.train.reward",
        "calculator_version": "test.v1",
        "dimensions": {},
        "measures": {"algorithm_reward": 0.5},
        "reward_components": [],
        "provenance": {},
        "state": "complete",
    }
    update = TraceFactUpdate(
        trace_type="verifiers",
        external_id=trace_id,
        projection_id=projection_id(payload),
        **payload,
    )
    return writer.write_local(
        [
            fragments.trace_fact_record(
                {
                    "project": "proj",
                    "run": run,
                    "run_id": f"{run}-id",
                    "update": update.payload(),
                }
            )
        ]
    )


def _dead_letter_files():
    root = fragments.dead_letter_dir()
    return sorted(root.rglob("*.jsonl")) if root.exists() else []


def _pending_files():
    return sorted(fragments.local_inbox_dir().rglob("*.jsonl"))


def test_poison_fragment_is_dead_lettered_and_healthy_batch_mate_imports(
    temp_dir, monkeypatch, clock
):
    writer = fragments.FragmentWriter(writer_id="server-writer")
    poison = _trace_fragment(writer, "poison-run", "huge-trace")
    healthy = _trace_fragment(writer, "eval-run", "eval-trace")
    original_bulk_log = Storage.bulk_log

    def reject_poison(**kwargs):
        if kwargs["run"] == "poison-run":
            raise STRICT_MODE_ERROR
        return original_bulk_log(**kwargs)

    monkeypatch.setattr(Storage, "bulk_log", staticmethod(reject_poison))

    claimed = fragments.claim_inbox_batch(max_files=16)
    assert len(claimed) == 2
    assert fragments.import_claimed_fragments(claimed) == 1

    assert not healthy.exists()
    assert SQLiteStorage.get_logs("proj", "eval-run", run_id="eval-run-id")[0][
        "eval/reward_mean"
    ] == pytest.approx(0.75)
    assert len(SQLiteStorage.get_traces("proj", run_id="eval-run-id")) == 1

    assert not poison.exists()
    assert _pending_files() == []
    dead = _dead_letter_files()
    assert [path.relative_to(fragments.dead_letter_dir()) for path in dead] == [
        poison.relative_to(fragments.local_inbox_dir())
    ]
    assert fragments.dead_letter_dir().parent == fragments.local_inbox_dir().parent
    assert fragments.parse_fragment_bytes(dead[0].read_bytes())[0]["run"] == (
        "poison-run"
    )
    error = orjson.loads(
        dead[0]
        .with_name(dead[0].name + fragments.DEAD_LETTER_ERROR_SUFFIX)
        .read_bytes()
    )
    assert error["classification"] == fragments.PERMANENT
    assert error["error_class"] == "pymysql.err.OperationalError"
    assert "strict mode" in error["message"]
    assert error["attempts"] == 1
    assert error["records"] == 1
    assert error["kinds"] == ["metric"]
    assert not list(fragments.local_inbox_dir().rglob("*.retry.json"))


def test_orphan_trace_fact_retries_with_backoff_then_dead_letters(
    temp_dir, monkeypatch, clock
):
    monkeypatch.setenv("TRACKIO_INBOX_RETRY_MAX_AGE", "3600")
    writer = fragments.FragmentWriter(writer_id="orphan-writer")
    orphan = _fact_fragment(writer, "train-run", "never-arrives")
    scalar = _scalar_fragment(writer, "train-run", 0.5)
    first_failure = clock[0]

    def missing_parent(**kwargs):
        raise KeyError("trace 'never-arrives' does not exist")

    monkeypatch.setattr(Storage, "upsert_trace_facts_batch", missing_parent)

    claimed = fragments.claim_inbox_batch(max_files=16)
    assert len(claimed) == 2
    assert fragments.import_claimed_fragments(claimed) == 1
    assert not scalar.exists()
    assert len(SQLiteStorage.get_logs("proj", "train-run", run_id="train-run-id")) == 1
    assert orphan.exists()
    state = fragments.read_retry_state(orphan)
    assert state["attempts"] == 1
    assert state["classification"] == fragments.RETRYABLE
    assert state["first_failure_at"] == first_failure

    assert fragments.claim_inbox_batch(max_files=16) == []

    clock[0] += 6
    retry = fragments.claim_inbox_batch(max_files=16)
    assert [fragment.pending_path for fragment in retry] == [orphan]
    fragments.import_claimed_fragments(retry)
    assert orphan.exists()
    state = fragments.read_retry_state(orphan)
    assert state["attempts"] == 2
    assert state["first_failure_at"] == first_failure
    assert state["next_attempt_at"] == pytest.approx(clock[0] + 10)
    assert _dead_letter_files() == []

    clock[0] = first_failure + 3600
    fragments.import_claimed_fragments(fragments.claim_inbox_batch(max_files=16))
    assert not orphan.exists()
    assert _pending_files() == []
    dead = _dead_letter_files()
    assert len(dead) == 1
    error = orjson.loads(
        dead[0]
        .with_name(dead[0].name + fragments.DEAD_LETTER_ERROR_SUFFIX)
        .read_bytes()
    )
    assert error["attempts"] == 3
    assert error["error_class"] == "builtins.KeyError"
    assert error["classification"] == fragments.RETRYABLE


def test_trace_fact_imports_once_its_parent_arrives(temp_dir, monkeypatch, clock):
    writer = fragments.FragmentWriter(writer_id="late-parent")
    fact = _fact_fragment(writer, "run", "late-trace")
    fragments.import_claimed_fragments(fragments.claim_inbox_batch(max_files=16))
    assert fragments.read_retry_state(fact)["attempts"] == 1

    _trace_fragment(writer, "run", "late-trace")
    clock[0] += 6
    claimed = fragments.claim_inbox_batch(max_files=16)
    assert len(claimed) == 2
    assert fragments.import_claimed_fragments(claimed) == 2
    assert _pending_files() == []
    assert not list(fragments.local_inbox_dir().rglob("*.retry.json"))
    assert _dead_letter_files() == []


@pytest.mark.parametrize(
    "error",
    [
        pymysql.err.OperationalError(2013, "Lost connection to MySQL server"),
        DorisConnectionPoolTimeout("no Doris connection available"),
    ],
)
def test_transient_error_keeps_whole_batch_pending_without_dead_letter(
    temp_dir, monkeypatch, clock, error
):
    monkeypatch.setenv("TRACKIO_INBOX_RETRY_MAX_AGE", "0")
    writer = fragments.FragmentWriter(writer_id="outage")
    first = _scalar_fragment(writer, "run", 0.1, step=0)
    second = _scalar_fragment(writer, "run", 0.2, step=1)
    calls = []

    def outage(**kwargs):
        calls.append(kwargs)
        raise error

    monkeypatch.setattr(Storage, "bulk_log", staticmethod(outage))
    for attempt in range(1, 4):
        claimed = fragments.claim_inbox_batch(max_files=16)
        assert len(claimed) == 2
        assert fragments.import_claimed_fragments(claimed) == 0
        assert len(calls) == attempt
        for path in (first, second):
            assert path.exists()
            state = fragments.read_retry_state(path)
            assert state["attempts"] == attempt
            assert state["classification"] == fragments.TRANSIENT
        clock[0] += 3600
    assert _dead_letter_files() == []
    assert not list(fragments.local_inbox_dir().rglob("*.processing"))


def test_restart_recovery_preserves_retry_attempts(temp_dir, monkeypatch, clock):
    writer = fragments.FragmentWriter(writer_id="restart")
    orphan = _fact_fragment(writer, "run", "missing")
    monkeypatch.setattr(
        Storage,
        "upsert_trace_facts_batch",
        lambda **kwargs: (_ for _ in ()).throw(
            KeyError("trace 'missing' does not exist")
        ),
    )
    fragments.import_claimed_fragments(fragments.claim_inbox_batch(max_files=1))
    first_failure = fragments.read_retry_state(orphan)["first_failure_at"]

    clock[0] += 6
    claimed = fragments.claim_inbox_batch(max_files=1)
    assert claimed and not orphan.exists()
    stale_state = fragments.retry_state_path(orphan.with_name("99999999.jsonl"))
    stale_state.write_bytes(b'{"attempts": 7}')

    fragments.recover_processing_fragments()
    assert orphan.exists()
    assert not stale_state.exists()
    assert fragments.read_retry_state(orphan)["attempts"] == 1

    clock[0] += 6
    fragments.import_claimed_fragments(fragments.claim_inbox_batch(max_files=1))
    state = fragments.read_retry_state(orphan)
    assert state["attempts"] == 2
    assert state["first_failure_at"] == first_failure


def test_stale_claim_is_not_imported_after_its_file_is_moved(
    temp_dir, monkeypatch, clock
):
    writer = fragments.FragmentWriter(writer_id="stale")
    _scalar_fragment(writer, "run", 0.3)
    claimed = fragments.claim_inbox_batch(max_files=1)
    claimed[0].path.rename(claimed[0].path.with_name("moved-aside"))
    calls = []
    monkeypatch.setattr(
        Storage, "bulk_log", staticmethod(lambda **kwargs: calls.append(kwargs))
    )
    assert fragments.import_claimed_fragments(claimed) == 0
    assert calls == []
    assert _dead_letter_files() == []


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (STRICT_MODE_ERROR, fragments.PERMANENT),
        (DorisValueTooLargeError("too large"), fragments.PERMANENT),
        (ValueError("invalid trace fact"), fragments.PERMANENT),
        (pymysql.err.DataError(1406, "Data too long"), fragments.PERMANENT),
        (KeyError("trace 'x' does not exist"), fragments.RETRYABLE),
        (pymysql.err.OperationalError(1105, "unexpected"), fragments.RETRYABLE),
        (RuntimeError("unexpected"), fragments.RETRYABLE),
        (pymysql.err.OperationalError(2003, "Can't connect"), fragments.TRANSIENT),
        (pymysql.err.InterfaceError(0, ""), fragments.TRANSIENT),
        (
            pymysql.err.OperationalError(1105, "errCode = 2, detailMessage = timeout"),
            fragments.TRANSIENT,
        ),
        (DorisConnectionPoolTimeout("pool"), fragments.TRANSIENT),
    ],
)
def test_import_error_classification(error, expected):
    assert fragments.classify_import_error(error) == expected


class _RecordingCursor:
    def __init__(self):
        self.statements = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, query, params=None):
        self.statements.append(query)

    def executemany(self, query, rows):
        self.statements.append(query)


class _Connection:
    def __init__(self, cursor):
        self._cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def cursor(self):
        return self._cursor


def test_doris_bulk_log_refuses_oversized_trace_before_any_insert(monkeypatch):
    cursor = _RecordingCursor()

    @contextmanager
    def connection(**kwargs):
        yield _Connection(cursor)

    monkeypatch.setattr(DorisStorage, "_connection", staticmethod(connection))
    monkeypatch.setenv("TRACKIO_DORIS_MAX_STRING_BYTES", "4096")
    native = trackio.VerifiersTrace(
        {
            "id": "huge",
            "version": 2,
            "nodes": [{"message": {"role": "tool", "content": "x" * 8192}}],
        },
        messages=[],
    )._to_dict("proj", "run", 0)

    with pytest.raises(DorisValueTooLargeError, match="traces.payload"):
        DorisStorage.bulk_log(
            project="proj",
            run="run",
            run_id="run-id",
            metrics_list=[{"traces/verifiers": native}],
            steps=[0],
            log_ids=["log"],
        )
    assert not any("INSERT" in statement for statement in cursor.statements)

    monkeypatch.setenv("TRACKIO_DORIS_MAX_STRING_BYTES", "0")
    DorisStorage.bulk_log(
        project="proj",
        run="run",
        run_id="run-id",
        metrics_list=[{"traces/verifiers": native}],
        steps=[0],
        log_ids=["log"],
    )
    assert any("INSERT INTO traces" in statement for statement in cursor.statements)
