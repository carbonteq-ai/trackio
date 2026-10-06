"""Run purge with artifact versions that several runs log identically.

Identical artifact content is one shared version with an output link per
run.  Purging one of those runs must keep the version for the survivors,
validate everything before deleting, and let an interrupted purge finish.
"""

import hashlib
import sqlite3
from contextlib import contextmanager

import pytest

from trackio import database as trackio_database
from trackio.doris_storage import DorisStorage
from trackio.purge import (
    purge_runs,
    purge_unreferenced_artifacts,
    run_purge_summary,
    unreferenced_artifact_purge_summary,
)
from trackio.sqlite_storage import SQLiteStorage


def _log(project, run_name, run_id):
    db_path = SQLiteStorage.init_db(project)
    with trackio_database.connect(str(db_path)) as conn:
        conn.execute(
            "INSERT INTO metrics (timestamp, run_id, run_name, step, metrics) "
            "VALUES (?, ?, ?, ?, ?)",
            ("2026-10-05T00:00:00+00:00", run_id, run_name, 0, "{}"),
        )


def _commit(project, digest, size, *, run_id, name="collection", metadata=None):
    return SQLiteStorage.commit_artifact_version(
        project=project,
        name=name,
        type="training-collection",
        description=None,
        manifest=[{"path": "rows.jsonl", "digest": digest, "size": size}],
        metadata=metadata,
        aliases=None,
        run_name=f"train.{run_id}",
        run_id=run_id,
    )


@pytest.fixture
def shared(temp_dir, stage_blob):
    project = "shared-purge"
    payload = b"identical collection rows"
    digest, blob = stage_blob(project, payload)
    _log(project, "train.first", "first")
    _log(project, "train.second", "second")
    first = _commit(project, digest, len(payload), run_id="first")
    second = _commit(project, digest, len(payload), run_id="second")
    assert first["version_id"] == second["version_id"]
    return project, first["version_id"], blob


def test_shared_output_version_is_retained_not_blocking(shared):
    project, version_id, blob = shared

    preview = run_purge_summary(SQLiteStorage, project, ("first",))

    assert preview["blockers"] == []
    [artifact] = preview["artifacts"]
    assert artifact["delete"] is False
    assert artifact["retained_by_run_ids"] == ["second"]
    receipt = purge_runs(
        SQLiteStorage, project, ("first",), plan_digest=preview["digest"]
    )
    assert receipt["deleted_provider_run_ids"] == ["first"]
    assert receipt["deleted_artifact_version_ids"] == []
    assert [record["id"] for record in SQLiteStorage.get_run_records(project)] == [
        "second"
    ]
    assert [
        output["version_id"]
        for output in SQLiteStorage.get_run_artifacts(
            project, "train.second", "second"
        )["output"]
    ] == [version_id]
    assert blob.exists()

    last = run_purge_summary(SQLiteStorage, project, ("second",))
    assert last["artifacts"][0]["delete"] is True
    purge_runs(SQLiteStorage, project, ("second",), plan_digest=last["digest"])
    assert SQLiteStorage.get_run_records(project) == []
    assert not blob.exists()


def test_consumer_of_a_retained_version_keeps_its_surviving_producer(shared):
    project, version_id, _blob = shared
    _log(project, "eval.reader", "reader")
    SQLiteStorage.insert_run_artifact_link(
        project, "eval.reader", "reader", version_id, "input"
    )

    preview = run_purge_summary(SQLiteStorage, project, ("first",))
    assert preview["blockers"] == []
    assert preview["artifacts"][0]["consumer_run_ids"] == ["reader"]

    sole = run_purge_summary(SQLiteStorage, project, ("first", "second"))
    assert sole["blockers"] == [
        f"artifact version {version_id} is consumed by unselected run 'reader'"
    ]


def test_storage_validates_artifact_links_before_deleting_run_rows(shared):
    project, version_id, blob = shared

    with pytest.raises(ValueError, match="still linked to unselected run"):
        SQLiteStorage.purge_runs(project, ("first",), (version_id,))

    assert {record["id"] for record in SQLiteStorage.get_run_records(project)} == {
        "first",
        "second",
    }
    assert blob.exists()


def test_absent_run_is_reported_only_when_allowed(shared):
    project, _version_id, _blob = shared
    preview = run_purge_summary(SQLiteStorage, project, ("first",))
    purge_runs(SQLiteStorage, project, ("first",), plan_digest=preview["digest"])

    blocked = run_purge_summary(SQLiteStorage, project, ("first",))
    assert blocked["blockers"] == [
        "run 'first' does not exist in Trackio project 'shared-purge'"
    ]
    resumed = run_purge_summary(SQLiteStorage, project, ("first",), allow_absent=True)
    assert resumed["blockers"] == []
    assert resumed["absent_run_ids"] == ["first"]
    receipt = purge_runs(
        SQLiteStorage,
        project,
        ("first",),
        plan_digest=resumed["digest"],
        allow_absent=True,
    )
    assert receipt["deleted_provider_run_ids"] == []
    assert receipt["already_absent_provider_run_ids"] == ["first"]


def test_unreferenced_versions_are_purged_by_exact_metadata(temp_dir, stage_blob):
    project = "leaked"
    _log(project, "train.alive", "alive")
    leaked_digest, leaked_blob = stage_blob(project, b"leaked rows")
    kept_digest, kept_blob = stage_blob(project, b"kept rows")
    leaked = _commit(
        project,
        leaked_digest,
        11,
        run_id="gone",
        name="leaked",
        metadata={"run_id": "smoke-r4", "project_id": "lab"},
    )
    _commit(
        project,
        kept_digest,
        9,
        run_id="alive",
        name="linked",
        metadata={"run_id": "smoke-r4", "project_id": "lab"},
    )
    _commit(
        project,
        kept_digest,
        9,
        run_id="gone",
        name="other",
        metadata={"run_id": "smoke-r5"},
    )
    db_path = SQLiteStorage.get_project_db_path(project)
    with sqlite3.connect(db_path) as conn:
        # What an interrupted purge left behind: links gone, producer cleared.
        conn.execute("DELETE FROM run_artifact_links WHERE run_id = 'gone'")
        conn.execute(
            "UPDATE artifact_versions SET producer_run_id = NULL WHERE producer_run_id = 'gone'"
        )

    selector = {"run_id": "smoke-r4", "project_id": "lab"}
    preview = unreferenced_artifact_purge_summary(SQLiteStorage, project, selector)
    assert [artifact["version_id"] for artifact in preview["artifacts"]] == [
        leaked["version_id"]
    ]
    with pytest.raises(ValueError, match="stale"):
        purge_unreferenced_artifacts(
            SQLiteStorage, project, selector, plan_digest="sha256:" + "0" * 64
        )
    receipt = purge_unreferenced_artifacts(
        SQLiteStorage, project, selector, plan_digest=preview["digest"]
    )

    assert receipt["deleted_artifact_version_ids"] == [leaked["version_id"]]
    assert not leaked_blob.exists()
    assert kept_blob.exists()
    assert {artifact["name"] for artifact in SQLiteStorage.get_artifacts(project)} == {
        "linked",
        "other",
    }
    assert (
        unreferenced_artifact_purge_summary(SQLiteStorage, project, selector)[
            "artifacts"
        ]
        == []
    )
    with pytest.raises(ValueError, match="non-empty"):
        unreferenced_artifact_purge_summary(SQLiteStorage, project, {})


def test_storage_refuses_to_purge_a_linked_version_as_unreferenced(shared):
    project, version_id, blob = shared
    with pytest.raises(ValueError, match="still linked"):
        SQLiteStorage.purge_artifact_versions(project, (version_id,))
    assert blob.exists()


_DORIS_TABLES = {
    "metrics": "project_id, run_id, run_name, timestamp",
    "configs": "project_id, run_id, run_name",
    "system_metrics": "project_id, run_id",
    "traces": "project_id, run_id",
    "trace_reward_components": "project_id, run_id",
    "trace_environment_metrics": "project_id, run_id",
    "alerts": "project_id, run_id",
    "run_notes": "project_id, run_id",
    "artifacts": "project_id, artifact_id, name, artifact_type, description",
    "artifact_versions": (
        "project_id, version_id, artifact_id, version_number, manifest, "
        "manifest_digest, metadata, size_bytes, producer_run_id, producer_run_name"
    ),
    "artifact_aliases": "project_id, artifact_id, alias, version_id",
    "run_artifact_links": "project_id, run_id, run_name, version_id, direction, created_at",
}


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


@pytest.fixture
def doris(monkeypatch):
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    for table, columns in _DORIS_TABLES.items():
        connection.execute(f"CREATE TABLE {table} ({columns})")
    statements: list[str] = []

    class Connection:
        def cursor(self):
            return _Cursor(connection, statements)

    @contextmanager
    def fake_connection(*args, **kwargs):
        yield Connection()

    deleted: list[str] = []

    class Store:
        def delete(self, project, digest):
            deleted.append(digest)

    monkeypatch.setattr(DorisStorage, "_connection", staticmethod(fake_connection))
    monkeypatch.setattr("trackio.doris_storage.get_artifact_store", lambda: Store())
    digest = hashlib.sha256(b"rows").hexdigest()
    manifest = f'[{{"path": "rows.jsonl", "digest": "{digest}", "size": 4}}]'
    for run_id in ("first", "second"):
        connection.execute(
            "INSERT INTO metrics VALUES ('p', ?, ?, '2026-10-05T00:00:00')",
            (run_id, f"train.{run_id}"),
        )
        connection.execute(
            "INSERT INTO run_artifact_links VALUES ('p', ?, ?, 7, 'output', '2026-10-05')",
            (run_id, f"train.{run_id}"),
        )
    connection.execute(
        "INSERT INTO artifacts VALUES "
        "('p', 70, 'collection', 'training-collection', NULL)"
    )
    connection.execute(
        "INSERT INTO artifact_versions VALUES "
        "('p', 7, 70, 0, ?, ?, ?, 4, 'first', 'train.first')",
        (manifest, digest, '{"run_id": "smoke"}'),
    )
    yield connection, statements, deleted, digest
    connection.close()


def test_doris_shared_version_is_retained_and_run_rows_go_last(doris):
    connection, statements, deleted, _digest = doris

    preview = run_purge_summary(DorisStorage, "p", ("first",))
    assert preview["blockers"] == []
    assert preview["artifacts"][0]["retained_by_run_ids"] == ["second"]
    purge_runs(DorisStorage, "p", ("first",), plan_digest=preview["digest"])

    assert [record["id"] for record in DorisStorage.get_run_records("p")] == ["second"]
    assert (
        connection.execute("SELECT COUNT(*) FROM artifact_versions").fetchone()[0] == 1
    )
    assert deleted == []
    deletes = [
        statement for statement in statements if statement.startswith("DELETE FROM")
    ]
    assert deletes[-2:] == [
        "DELETE FROM run_artifact_links WHERE project_id = ? AND run_id = ?".replace(
            "?", "%s"
        ),
        "DELETE FROM metrics WHERE project_id = ? AND run_id = ?".replace("?", "%s"),
    ]


def test_doris_rejects_a_linked_version_before_any_delete(doris):
    _connection, statements, deleted, _digest = doris

    with pytest.raises(ValueError, match="still linked to unselected run"):
        DorisStorage.purge_runs("p", ("first",), (7,))

    assert not any(
        statement.startswith(("DELETE", "UPDATE")) for statement in statements
    )
    assert {record["id"] for record in DorisStorage.get_run_records("p")} == {
        "first",
        "second",
    }
    assert deleted == []


def test_doris_purges_unreferenced_versions_by_metadata(doris):
    connection, _statements, deleted, digest = doris
    connection.execute("DELETE FROM run_artifact_links")
    connection.execute("DELETE FROM metrics")

    preview = unreferenced_artifact_purge_summary(
        DorisStorage, "p", {"run_id": "smoke"}
    )
    assert [artifact["version_id"] for artifact in preview["artifacts"]] == [7]
    purge_unreferenced_artifacts(
        DorisStorage, "p", {"run_id": "smoke"}, plan_digest=preview["digest"]
    )

    assert (
        connection.execute("SELECT COUNT(*) FROM artifact_versions").fetchone()[0] == 0
    )
    assert connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 0
    assert deleted == [digest]
