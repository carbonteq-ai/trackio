from __future__ import annotations

from contextlib import contextmanager

from trackio.doris_storage import DorisStorage


class _Cursor:
    def __init__(self) -> None:
        self.executed: list[tuple[str, list[object]]] = []
        self._rows: list[dict[str, object]] = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, query, params):
        self.executed.append((query, list(params)))
        if "SELECT run_id FROM configs" in query:
            self._rows = [{"run_id": "run-1"}]
        elif "FROM system_metrics" in query:
            self._rows = [
                {
                    "timestamp": "2026-08-24T00:00:00+00:00",
                    "metric_0": "42.5",
                    "metric_1": None,
                }
            ]
        else:
            self._rows = [
                {
                    "timestamp": "2026-08-24T00:00:00+00:00",
                    "step": 3,
                    "metric_0": "0.5",
                    "metric_1": '{"source_step":3}',
                },
                {
                    "timestamp": "2026-08-24T00:00:01+00:00",
                    "step": 4,
                    "metric_0": None,
                    "metric_1": None,
                },
            ]

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)


class _Connection:
    def __init__(self, cursor: _Cursor) -> None:
        self._cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def cursor(self):
        return self._cursor


def test_doris_metric_history_projects_json_and_bounds_query(monkeypatch) -> None:
    cursor = _Cursor()

    @contextmanager
    def connection(*, initialize=True, include_database=True):
        del initialize, include_database
        yield _Connection(cursor)

    monkeypatch.setattr(DorisStorage, "_connection", staticmethod(connection))

    rows = DorisStorage.get_logs(
        "project",
        run_id="run-1",
        keys=("train/loss", "train/loss/attributes"),
        start_step=3,
        end_step=7,
        drop_empty=True,
        limit=2,
        offset=4,
    )

    query, params = cursor.executed[-1]
    assert "SELECT timestamp, step, metrics" not in query
    assert "JSON_EXTRACT(metrics, %s) AS metric_0" in query
    assert "step >= %s" in query
    assert "step <= %s" in query
    assert "JSON_EXTRACT(metrics, %s) IS NOT NULL" in query
    assert "LIMIT %s OFFSET %s" in query
    assert params == [
        '$."train/loss"',
        '$."train/loss/attributes"',
        "project",
        "run-1",
        3,
        7,
        '$."train/loss"',
        '$."train/loss/attributes"',
        2,
        4,
    ]
    assert rows == [
        {
            "train/loss": 0.5,
            "train/loss/attributes": {"source_step": 3},
            "timestamp": "2026-08-24T00:00:00+00:00",
            "step": 3,
        },
        {"timestamp": "2026-08-24T00:00:01+00:00", "step": 4},
    ]


def test_doris_system_history_projects_requested_json_keys(monkeypatch) -> None:
    cursor = _Cursor()

    @contextmanager
    def connection(*, initialize=True, include_database=True):
        del initialize, include_database
        yield _Connection(cursor)

    monkeypatch.setattr(DorisStorage, "_connection", staticmethod(connection))

    rows = DorisStorage.get_system_logs(
        "project",
        run_id="run-1",
        keys=("gpu/mean_utilization", "cpu/utilization"),
        limit=1000,
    )

    query, params = cursor.executed[-1]
    assert "SELECT timestamp, metrics" not in query
    assert "JSON_EXTRACT(metrics, %s) AS metric_0" in query
    assert "LIMIT %s" in query
    assert params == [
        '$."gpu/mean_utilization"',
        '$."cpu/utilization"',
        "project",
        "run-1",
        1000,
    ]
    assert rows == [
        {
            "gpu/mean_utilization": 42.5,
            "timestamp": "2026-08-24T00:00:00+00:00",
        }
    ]
