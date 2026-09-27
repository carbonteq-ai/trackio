import json
from contextlib import contextmanager

import pytest

import trackio
from trackio import project_sql
from trackio.doris_storage import DorisStorage
from trackio.project_sql import ProjectSqlError, prepare
from trackio.sqlite_storage import SQLiteStorage


@pytest.mark.parametrize(
    ("sql", "message"),
    [
        ("", "empty"),
        ("select 1; select 2", "exactly one statement"),
        ("insert into metric_rows values (1)", "read-only SELECT"),
        ("delete from metric_rows", "read-only SELECT"),
        ("select * from metrics", "unknown table 'metrics'"),
        ("select * from trackio.metrics", "unqualified"),
        ("select * from s3('s3://bucket/x')", "table functions"),
        (
            "with metric_rows as (select 1) select * from metric_rows",
            "cannot reuse the table name",
        ),
        ("select * from", "cannot parse"),
    ],
)
def test_only_one_query_over_logical_tables_is_accepted(sql, message):
    with pytest.raises(ProjectSqlError, match=message):
        prepare(sql, engine="doris", project="p", database="trackio")


def test_doris_statements_read_only_the_projects_rows():
    prepared = prepare(
        "with r as (select run_id, max(step) as last_step from metric_rows group by run_id) "
        "select r.run_id, r.last_step, c.config from r join run_configs c on c.run_id = r.run_id",
        engine="doris",
        project="it's-mine",
        database="trackio",
    )
    assert (
        "metric_rows AS (SELECT run_id, run_name, step, `timestamp`, metrics FROM trackio.metrics"
        in prepared
    )
    assert "WHERE project_id = 'it''s-mine'" in prepared
    assert (
        "run_configs AS (SELECT run_id, run_name, `config`, created_at FROM trackio.configs"
        in prepared
    )
    assert max(
        prepared.index("run_configs AS"), prepared.index("metric_rows AS")
    ) < prepared.index("r AS (")


def test_doris_path_sets_and_restores_the_query_timeout(monkeypatch):
    executed = []

    class Cursor:
        description = (("run_id",), ("n",))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, query, params=()):
            executed.append((query, tuple(params or ())))

        def fetchone(self):
            return {"value": 900}

        def fetchmany(self, size):
            return [{"run_id": "r1", "n": 3}, {"run_id": "r2", "n": 1}][:size]

    class Connection:
        def cursor(self):
            return Cursor()

    @contextmanager
    def connection(*args, **kwargs):
        yield Connection()

    monkeypatch.setattr(DorisStorage, "_connection", staticmethod(connection))
    monkeypatch.setattr(
        DorisStorage, "_settings", classmethod(lambda cls: {"database": "trackio"})
    )
    result = DorisStorage.project_sql(
        "proj",
        "select run_id, count(*) as n from metric_rows group by run_id",
        max_rows=1,
        timeout_seconds=2.5,
    )
    assert result == {
        "engine": "doris",
        "columns": ["run_id", "n"],
        "rows": [["r1", 3]],
        "truncated": True,
    }
    assert executed[1] == ("SET query_timeout = %s", (3,))
    assert "FROM trackio.metrics WHERE project_id = 'proj'" in executed[2][0]
    assert executed[3] == ("SET query_timeout = %s", (900,))


@pytest.fixture
def local_project(tmp_path, monkeypatch):
    for module in ("trackio", "trackio.sqlite_storage", "trackio.utils"):
        monkeypatch.setattr(f"{module}.TRACKIO_DIR", tmp_path, raising=False)
    monkeypatch.setattr("trackio.bucket_storage.TRACKIO_DIR", tmp_path, raising=False)
    monkeypatch.setattr(
        "trackio.utils.ARTIFACTS_DIR", tmp_path / "artifacts", raising=False
    )
    for run_id, rewards in (("run-a", (0.1, 0.4, 0.3)), ("run-b", (0.5, 0.6, 0.9))):
        trackio.init(
            project="sql-project",
            name=f"{run_id}-name",
            config={"run_id": run_id, "job_kind": "train.grpo"},
        )
        for step, reward in enumerate(rewards, start=1):
            trackio.log(
                {"train/rl/reward_mean": reward, "train/rl/entropy": reward / 2},
                step=step,
            )
        trackio.finish()
    return "sql-project"


def test_sqlite_storage_runs_doris_sql_with_the_provided_functions(local_project):
    result = SQLiteStorage.project_sql(
        local_project,
        """
        select json_extract_string(c.config, '$.run_id') as run_id,
               count(*) as updates,
               max_by(json_extract_double(m.metrics, '$."train/rl/reward_mean"'), m.step) as last_reward,
               min_by(json_extract_double(m.metrics, '$."train/rl/reward_mean"'), m.step) as first_reward,
               percentile(json_extract_double(m.metrics, '$."train/rl/reward_mean"'), 0.5) as median_reward,
               stddev_samp(json_extract_double(m.metrics, '$."train/rl/entropy"')) as entropy_spread
        from metric_rows m join run_configs c on c.run_id = m.run_id
        where json_extract_double(m.metrics, '$."train/rl/reward_mean"') is not null
        group by json_extract_string(c.config, '$.run_id')
        order by run_id
        """,
    )
    assert result["engine"] == "sqlite" and result["truncated"] is False
    rows = {row[0]: row for row in result["rows"]}
    assert rows["run-a"][1:5] == [3, 0.3, 0.1, 0.3]
    assert rows["run-b"][1:5] == [3, 0.9, 0.5, 0.6]
    assert rows["run-a"][5] == pytest.approx(0.0763762615)


def test_sqlite_storage_refuses_what_it_cannot_run(local_project):
    with pytest.raises(
        ProjectSqlError, match="bitmap_count is not available on SQLite storage"
    ):
        SQLiteStorage.project_sql(
            local_project, "select bitmap_count(step) from metric_rows"
        )
    with pytest.raises(ProjectSqlError, match="longer than"):
        SQLiteStorage.project_sql(
            local_project,
            "select count(*) from metric_rows a, metric_rows b, metric_rows c, metric_rows d, metric_rows e, "
            "metric_rows f, metric_rows g, metric_rows h, metric_rows i, metric_rows j",
            timeout_seconds=0.05,
        )
    limited = SQLiteStorage.project_sql(
        local_project, "select step from metric_rows", max_rows=2
    )
    assert limited["truncated"] is True and len(limited["rows"]) == 2


def test_json_paths_follow_doris_quoting():
    document = json.dumps(
        {"train/rl/entropy": 0.5, "a": {"b": [1, {"c": "x"}]}, 'q"k': 2}
    )
    assert project_sql.json_value(document, '$."train/rl/entropy"') == 0.5
    assert project_sql.json_value(document, "$.a.b[1].c") == "x"
    assert project_sql.json_value(document, '$."q\\"k"') is None
    assert project_sql.json_value(document, "$.missing") is None
    assert project_sql.json_value("not json", "$.a") is None
    assert project_sql.json_value(document.encode(), '$."train/rl/entropy"') == 0.5


def test_client_reads_local_projects(local_project):
    result = trackio.Api().project_sql(
        local_project, "select count(*) as rows_logged from metric_rows"
    )
    assert result["columns"] == ["rows_logged"] and result["rows"][0][0] >= 6
    assert trackio.Api().capabilities()["project_sql"] is True
