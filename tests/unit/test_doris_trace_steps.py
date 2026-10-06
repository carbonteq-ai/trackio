import sqlite3
from contextlib import contextmanager

from trackio.doris_storage import DorisStorage


class _Cursor:
    def __init__(self, connection, statements):
        self._connection = connection
        self._statements = statements
        self._cursor = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, query, params=()):
        self._statements.append(" ".join(query.split()))
        self._cursor = self._connection.execute(query.replace("%s", "?"), tuple(params))

    def fetchone(self):
        row = self._cursor.fetchone()
        return dict(row) if row is not None else None

    def fetchall(self):
        return [dict(row) for row in self._cursor.fetchall()]


def test_doris_trace_steps_count_rows_without_reading_trace_bodies(monkeypatch):
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute(
        "CREATE TABLE traces (project_id, run_id, step, trace_type, messages, payload)"
    )
    rows = [
        ("p", "r", 1, "verifiers"),
        ("p", "r", 1, "verifiers"),
        ("p", "r", 5, "verifiers"),
        ("p", "r", 5, "trackio"),
        ("p", "other", 5, "verifiers"),
    ]
    connection.executemany(
        "INSERT INTO traces VALUES (?, ?, ?, ?, '[]', '{}')",
        rows,
    )
    statements: list[str] = []

    class Connection:
        def cursor(self):
            return _Cursor(connection, statements)

    @contextmanager
    def fake_connection(*args, **kwargs):
        yield Connection()

    monkeypatch.setattr(DorisStorage, "_connection", staticmethod(fake_connection))

    assert DorisStorage.get_trace_steps("p", run_id="r") == {
        "total": 4,
        "steps": [{"step": 1, "count": 2}, {"step": 5, "count": 2}],
    }
    assert DorisStorage.get_trace_steps("p", run_id="r", trace_type="verifiers") == {
        "total": 3,
        "steps": [{"step": 1, "count": 2}, {"step": 5, "count": 1}],
    }
    assert DorisStorage.get_trace_steps("p") == {"total": 0, "steps": []}
    assert all("payload" not in statement for statement in statements)
    assert all("messages" not in statement for statement in statements)
