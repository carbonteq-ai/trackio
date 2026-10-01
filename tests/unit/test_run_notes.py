import sqlite3
from contextlib import contextmanager
from urllib.parse import parse_qs, urlparse

import pytest

import trackio
from trackio import run_notes
from trackio.api import Api
from trackio.doris_storage import DorisStorage
from trackio.run_notes import RunNoteConflictError, RunNoteNotFoundError
from trackio.sqlite_storage import SQLiteStorage


def _logged_run(project: str, name: str) -> str:
    run = trackio.init(project=project, name=name)
    trackio.log({"loss": 1.0})
    trackio.finish()
    return run.id


def _add(storage, project, **overrides):
    fields = {
        "scope": "run",
        "kind": "observation",
        "body_md": "Loss plateaued after **step 40**.",
        "source": "cli",
    }
    fields.update(overrides)
    return storage.add_run_note(project, **fields)


def test_add_note_round_trips_through_storage(temp_dir):
    run_id = _logged_run("notes-roundtrip", "train")

    note = _add(
        SQLiteStorage,
        "notes-roundtrip",
        run_name="train",
        title="  Plateau  ",
        metadata={"job": "job-1"},
    )

    assert note["revision"] == 1
    assert note["parent_revision"] is None
    assert note["scope"] == "run"
    assert note["run_id"] == run_id
    assert note["run_name"] == "train"
    assert note["title"] == "Plateau"
    assert note["source"] == "cli"
    assert note["deleted"] is False
    assert note["metadata"] == {"job": "job-1"}
    assert note["created_at"] == note["revised_at"]
    assert SQLiteStorage.get_run_notes("notes-roundtrip") == [note]
    assert SQLiteStorage.get_run_notes("notes-roundtrip", run_id=run_id) == [note]
    assert SQLiteStorage.get_run_notes("notes-roundtrip", run_id="other") == []
    assert SQLiteStorage.get_run_notes("notes-roundtrip", kind="decision") == []
    assert SQLiteStorage.get_run_note_history("notes-roundtrip", note["note_id"]) == [
        note
    ]


def test_revise_with_expected_revision_appends_and_keeps_history(temp_dir):
    _logged_run("notes-revise", "train")
    first = _add(SQLiteStorage, "notes-revise", run_name="train", title="Plateau")

    second = SQLiteStorage.revise_run_note(
        "notes-revise",
        first["note_id"],
        expected_revision=1,
        body_md="Loss recovered after the LR drop.",
        source="observatory",
        kind="decision",
    )

    assert second["revision"] == 2
    assert second["parent_revision"] == 1
    assert second["created_at"] == first["created_at"]
    assert second["revised_at"] >= first["revised_at"]
    assert second["kind"] == "decision"
    assert second["title"] == "Plateau"
    assert second["run_id"] == first["run_id"]
    history = SQLiteStorage.get_run_note_history("notes-revise", first["note_id"])
    assert [entry["revision"] for entry in history] == [1, 2]
    assert history[0]["body_md"] == first["body_md"]
    assert history[1]["body_md"] == "Loss recovered after the LR drop."
    assert SQLiteStorage.get_run_notes("notes-revise") == [second]

    cleared = SQLiteStorage.revise_run_note(
        "notes-revise",
        first["note_id"],
        expected_revision=2,
        body_md="Untitled now.",
        source="cli",
        title="",
    )
    assert cleared["title"] is None
    assert cleared["kind"] == "decision"


def test_stale_expected_revision_is_a_conflict_naming_the_current_revision(
    temp_dir,
):
    _logged_run("notes-stale", "train")
    note = _add(SQLiteStorage, "notes-stale", run_name="train")
    SQLiteStorage.revise_run_note(
        "notes-stale",
        note["note_id"],
        expected_revision=1,
        body_md="second",
        source="mcp",
    )

    with pytest.raises(RunNoteConflictError, match="at revision 2") as caught:
        SQLiteStorage.revise_run_note(
            "notes-stale",
            note["note_id"],
            expected_revision=1,
            body_md="lost update",
            source="cli",
        )

    assert caught.value.current_revision == 2
    assert caught.value.note_id == note["note_id"]
    assert caught.value.status_code == 409
    assert isinstance(caught.value, RuntimeError)
    history = SQLiteStorage.get_run_note_history("notes-stale", note["note_id"])
    assert [entry["body_md"] for entry in history] == [note["body_md"], "second"]

    with pytest.raises(RunNoteNotFoundError):
        SQLiteStorage.revise_run_note(
            "notes-stale",
            "missing-note",
            expected_revision=1,
            body_md="x",
            source="cli",
        )


def test_delete_adds_a_hidden_tombstone_and_keeps_history(temp_dir):
    _logged_run("notes-delete", "train")
    kept = _add(SQLiteStorage, "notes-delete", run_name="train", body_md="kept")
    note = _add(SQLiteStorage, "notes-delete", run_name="train", title="Gone")

    with pytest.raises(RunNoteConflictError) as stale:
        SQLiteStorage.delete_run_note(
            "notes-delete", note["note_id"], expected_revision=2, source="cli"
        )
    assert stale.value.current_revision == 1

    tombstone = SQLiteStorage.delete_run_note(
        "notes-delete", note["note_id"], expected_revision=1, source="observatory"
    )

    assert tombstone["revision"] == 2
    assert tombstone["deleted"] is True
    assert tombstone["body_md"] == note["body_md"]
    assert tombstone["title"] == "Gone"
    assert tombstone["source"] == "observatory"
    assert SQLiteStorage.get_run_notes("notes-delete") == [kept]
    listed = SQLiteStorage.get_run_notes("notes-delete", include_deleted=True)
    assert {entry["note_id"]: entry["deleted"] for entry in listed} == {
        kept["note_id"]: False,
        note["note_id"]: True,
    }
    history = SQLiteStorage.get_run_note_history("notes-delete", note["note_id"])
    assert [(entry["revision"], entry["deleted"]) for entry in history] == [
        (1, False),
        (2, True),
    ]

    with pytest.raises(RunNoteConflictError, match="deleted"):
        SQLiteStorage.revise_run_note(
            "notes-delete",
            note["note_id"],
            expected_revision=2,
            body_md="resurrect",
            source="cli",
        )
    with pytest.raises(RunNoteConflictError, match="deleted"):
        SQLiteStorage.delete_run_note(
            "notes-delete", note["note_id"], expected_revision=2, source="cli"
        )


def test_add_with_same_note_id_is_idempotent_for_identical_content(temp_dir):
    _logged_run("notes-idempotent", "train")
    first = _add(
        SQLiteStorage, "notes-idempotent", run_name="train", note_id="job-7/summary"
    )

    retried = _add(
        SQLiteStorage,
        "notes-idempotent",
        run_name="train",
        note_id="job-7/summary",
        source="mcp",
    )

    assert retried == first
    assert (
        len(SQLiteStorage.get_run_note_history("notes-idempotent", "job-7/summary"))
        == 1
    )


def test_add_with_same_note_id_and_different_content_conflicts(temp_dir):
    _logged_run("notes-add-conflict", "train")
    _add(SQLiteStorage, "notes-add-conflict", run_name="train", note_id="n-1")
    SQLiteStorage.revise_run_note(
        "notes-add-conflict", "n-1", expected_revision=1, body_md="v2", source="cli"
    )

    with pytest.raises(RunNoteConflictError, match="revision 2") as caught:
        _add(
            SQLiteStorage,
            "notes-add-conflict",
            run_name="train",
            note_id="n-1",
            body_md="different body",
        )

    assert caught.value.current_revision == 2
    assert len(SQLiteStorage.get_run_note_history("notes-add-conflict", "n-1")) == 2


def test_project_scope_notes_have_no_run_and_filter_by_scope(temp_dir):
    _logged_run("notes-project", "train")
    run_note = _add(SQLiteStorage, "notes-project", run_name="train")
    project_note = _add(
        SQLiteStorage,
        "notes-project",
        scope="project",
        kind="plan",
        body_md="# Plan\n\nScreen three LRs.",
    )

    assert project_note["scope"] == "project"
    assert project_note["run_id"] is None
    assert project_note["run_name"] is None
    assert SQLiteStorage.get_run_notes("notes-project", scope="project") == [
        project_note
    ]
    assert SQLiteStorage.get_run_notes("notes-project", scope="run") == [run_note]
    assert SQLiteStorage.get_run_notes("notes-project", kind="plan") == [project_note]

    with pytest.raises(ValueError, match="cannot name a run"):
        _add(SQLiteStorage, "notes-project", scope="project", run_name="train")
    with pytest.raises(ValueError, match="requires run_id or run_name"):
        _add(SQLiteStorage, "notes-project")
    with pytest.raises(ValueError, match="does not exist"):
        _add(SQLiteStorage, "notes-project", run_name="missing")
    with pytest.raises(ValueError, match="scope"):
        _add(SQLiteStorage, "notes-project", scope="job", run_name="train")
    with pytest.raises(ValueError, match="source"):
        _add(SQLiteStorage, "notes-project", run_name="train", source="  ")


def test_delete_run_removes_only_that_runs_notes(temp_dir):
    _logged_run("notes-delete-run", "keep")
    _logged_run("notes-delete-run", "drop")
    kept = _add(SQLiteStorage, "notes-delete-run", run_name="keep")
    dropped = _add(SQLiteStorage, "notes-delete-run", run_name="drop")
    project_note = _add(SQLiteStorage, "notes-delete-run", scope="project")

    assert SQLiteStorage.delete_run("notes-delete-run", "drop")

    remaining = SQLiteStorage.get_run_notes("notes-delete-run", include_deleted=True)
    assert {note["note_id"] for note in remaining} == {
        kept["note_id"],
        project_note["note_id"],
    }
    assert (
        SQLiteStorage.get_run_note_history("notes-delete-run", dropped["note_id"]) == []
    )


def test_purge_runs_removes_the_purged_runs_notes(temp_dir):
    keep_id = _logged_run("notes-purge", "keep")
    drop_id = _logged_run("notes-purge", "drop")
    kept = _add(SQLiteStorage, "notes-purge", run_id=keep_id)
    _add(SQLiteStorage, "notes-purge", run_id=drop_id)

    SQLiteStorage.purge_runs("notes-purge", (drop_id,), ())

    assert SQLiteStorage.get_run_notes("notes-purge", include_deleted=True) == [kept]


def test_rename_run_updates_note_run_name_across_revisions(temp_dir):
    run_id = _logged_run("notes-rename", "before")
    note = _add(SQLiteStorage, "notes-rename", run_name="before")
    SQLiteStorage.revise_run_note(
        "notes-rename", note["note_id"], expected_revision=1, body_md="v2", source="cli"
    )

    SQLiteStorage.rename_run("notes-rename", "before", "after", run_id=run_id)

    history = SQLiteStorage.get_run_note_history("notes-rename", note["note_id"])
    assert {entry["run_name"] for entry in history} == {"after"}
    assert {entry["run_id"] for entry in history} == {run_id}


def test_move_run_carries_note_history_to_the_new_project(temp_dir):
    run_id = _logged_run("notes-move-source", "train")
    note = _add(SQLiteStorage, "notes-move-source", run_name="train")
    SQLiteStorage.revise_run_note(
        "notes-move-source",
        note["note_id"],
        expected_revision=1,
        body_md="v2",
        source="cli",
    )
    project_note = _add(SQLiteStorage, "notes-move-source", scope="project")

    assert SQLiteStorage.move_run("notes-move-source", "train", "notes-move-target")

    moved = SQLiteStorage.get_run_note_history("notes-move-target", note["note_id"])
    assert [entry["revision"] for entry in moved] == [1, 2]
    assert {entry["run_id"] for entry in moved} == {run_id}
    assert {entry["run_name"] for entry in moved} == {"train"}
    assert SQLiteStorage.get_run_notes("notes-move-source") == [project_note]


def test_notes_survive_parquet_export_and_import(temp_dir):
    _logged_run("notes-parquet", "train")
    note = _add(SQLiteStorage, "notes-parquet", run_name="train")
    revised = SQLiteStorage.revise_run_note(
        "notes-parquet",
        note["note_id"],
        expected_revision=1,
        body_md="v2",
        source="cli",
    )
    SQLiteStorage._dataset_import_attempted = True
    SQLiteStorage.export_to_parquet()
    db_path = SQLiteStorage.get_project_db_path("notes-parquet")
    sidecar = db_path.with_name(f"{db_path.stem}_run_notes.parquet")
    assert sidecar.exists()
    assert sidecar in SQLiteStorage._project_parquet_paths(db_path)

    db_path.unlink()
    SQLiteStorage.import_from_parquet()

    assert SQLiteStorage.get_run_notes("notes-parquet") == [revised]
    assert (
        len(SQLiteStorage.get_run_note_history("notes-parquet", note["note_id"])) == 2
    )

    assert SQLiteStorage.delete_run("notes-parquet", "train")
    SQLiteStorage.export_to_parquet()
    assert not sidecar.exists()
    with pytest.raises(ValueError, match="reserved suffix"):
        SQLiteStorage.validate_project_name("model_run_notes")


def test_reads_of_unknown_projects_are_empty(temp_dir):
    assert SQLiteStorage.get_run_notes("never-created") == []
    assert SQLiteStorage.get_run_note_history("never-created", "n-1") == []


def test_api_local_mode_round_trips_notes(temp_dir):
    _logged_run("notes-api-local", "train")
    api = Api()

    note = api.add_run_note(
        "notes-api-local",
        scope="run",
        run_name="train",
        kind="observation",
        body_md="local",
        source="cli",
    )
    revised = api.revise_run_note(
        "notes-api-local",
        note["note_id"],
        expected_revision=1,
        body_md="local v2",
        source="cli",
    )
    with pytest.raises(RunNoteConflictError):
        api.revise_run_note(
            "notes-api-local",
            note["note_id"],
            expected_revision=1,
            body_md="stale",
            source="cli",
        )
    deleted = api.delete_run_note(
        "notes-api-local", note["note_id"], expected_revision=2, source="cli"
    )

    assert revised["revision"] == 2
    assert deleted["deleted"] is True
    assert api.run_notes("notes-api-local") == []
    assert api.run_notes("notes-api-local", include_deleted=True) == [deleted]
    assert [
        entry["revision"]
        for entry in api.run_note_history("notes-api-local", note["note_id"])
    ] == [1, 2, 3]
    assert api.capabilities()["run_notes"] is True


def test_server_round_trip_requires_write_token_for_note_writes(temp_dir):
    project = "notes-server"
    run_id = _logged_run(project, "train")
    app, url, _, full_url = trackio.show(block_thread=False, open_browser=False)
    try:
        token = parse_qs(urlparse(full_url).query)["write_token"][0]
        reader = Api(url)
        writer = Api(url, write_token=token)

        with pytest.raises(RuntimeError, match="write_token is required"):
            reader.add_run_note(
                project,
                scope="run",
                run_id=run_id,
                kind="observation",
                body_md="unauthorized",
                source="cli",
            )
        assert reader.run_notes(project) == []

        note = writer.add_run_note(
            project,
            scope="run",
            run_id=run_id,
            kind="observation",
            title="Server note",
            body_md="written over HTTP",
            source="mcp",
            note_id="server-note",
            metadata={"job": "j-1"},
        )
        assert note["run_name"] == "train"
        assert note["metadata"] == {"job": "j-1"}
        assert (
            writer.add_run_note(
                project,
                scope="run",
                run_id=run_id,
                kind="observation",
                title="Server note",
                body_md="written over HTTP",
                source="mcp",
                note_id="server-note",
            )
            == note
        )

        revised = writer.revise_run_note(
            project,
            "server-note",
            expected_revision=1,
            body_md="revised over HTTP",
            source="observatory",
        )
        assert revised["revision"] == 2

        with pytest.raises(RunNoteConflictError, match="at revision 2") as caught:
            writer.revise_run_note(
                project,
                "server-note",
                expected_revision=1,
                body_md="stale",
                source="cli",
            )
        assert caught.value.current_revision == 2
        assert caught.value.note_id == "server-note"

        with pytest.raises(RuntimeError, match="write_token is required"):
            reader.delete_run_note(
                project, "server-note", expected_revision=2, source="cli"
            )
        tombstone = writer.delete_run_note(
            project, "server-note", expected_revision=2, source="cli"
        )
        assert tombstone["deleted"] is True

        assert reader.run_notes(project) == []
        assert reader.run_notes(project, run_id=run_id, include_deleted=True) == [
            tombstone
        ]
        assert [
            entry["revision"]
            for entry in reader.run_note_history(project, "server-note")
        ] == [1, 2, 3]

        with pytest.raises(RuntimeError, match="expected_revision"):
            writer.revise_run_note(
                project,
                "server-note",
                expected_revision=0,
                body_md="bad",
                source="cli",
            )
        with pytest.raises(RuntimeError, match="does not exist"):
            writer.add_run_note(
                project,
                scope="run",
                run_name="missing",
                kind="observation",
                body_md="x",
                source="cli",
            )
    finally:
        app.close()


_DORIS_TABLES = {
    "metrics": "project_id, event_id, run_id, timestamp, run_name, step, metrics",
    "configs": "project_id, run_id, run_name, config, created_at",
    "system_metrics": "project_id, event_id, run_id, timestamp, run_name, metrics",
    "traces": "project_id, trace_id, run_id, timestamp, run_name",
    "trace_reward_components": "project_id, trace_id, run_id, projection_id, name",
    "trace_environment_metrics": "project_id, trace_id, run_id, projection_id, name, value",
    "alerts": "project_id, event_id, run_id, timestamp, run_name",
    "artifacts": "project_id, artifact_id, name",
    "artifact_versions": (
        "project_id, version_id, artifact_id, manifest, size_bytes, "
        "producer_run_id, producer_run_name"
    ),
    "artifact_aliases": "project_id, artifact_id, alias, version_id",
    "run_artifact_links": "project_id, link_id, run_id, run_name, version_id, created_at",
    "project_metadata": "project_id, metadata_key, metadata_value",
    "run_notes": "project_id, " + ", ".join(run_notes.NOTE_COLUMNS),
}


class _DorisCursor:
    """Run the Doris provider's SQL against SQLite to check its logic."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self._cursor: sqlite3.Cursor | None = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, query, params=()):
        self._cursor = self._connection.execute(
            query.replace("%s", "?"), tuple(params or ())
        )

    def executemany(self, query, rows):
        self._connection.executemany(query.replace("%s", "?"), rows)

    def fetchone(self):
        row = self._cursor.fetchone()
        return dict(row) if row is not None else None

    def fetchall(self):
        return [dict(row) for row in self._cursor.fetchall()]


@pytest.fixture
def doris(monkeypatch):
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.row_factory = sqlite3.Row
    for table, columns in _DORIS_TABLES.items():
        connection.execute(f"CREATE TABLE {table} ({columns})")
    connection.execute(
        "CREATE UNIQUE INDEX run_notes_key ON run_notes(project_id, note_id, revision)"
    )

    class Connection:
        def cursor(self):
            return _DorisCursor(connection)

    @contextmanager
    def fake_connection(*args, **kwargs):
        yield Connection()

    monkeypatch.setattr(DorisStorage, "_connection", staticmethod(fake_connection))

    def log_run(project, run_id, run_name):
        connection.execute(
            "INSERT INTO metrics VALUES (?, ?, ?, ?, ?, ?, ?)",
            (project, f"{run_id}-0", run_id, "2026-09-01T00:00:00", run_name, 0, "{}"),
        )

    log_run.connection = connection
    yield log_run
    connection.close()


def test_doris_provider_revises_conflicts_and_tombstones(doris):
    doris("proj", "run-1", "train")
    doris("other", "run-9", "train")

    note = _add(DorisStorage, "proj", run_name="train", note_id="n-1")
    assert note["run_id"] == "run-1"
    assert _add(DorisStorage, "proj", run_name="train", note_id="n-1") == note
    with pytest.raises(RunNoteConflictError):
        _add(DorisStorage, "proj", run_name="train", note_id="n-1", body_md="other")
    with pytest.raises(ValueError, match="does not exist"):
        _add(DorisStorage, "proj", run_name="missing")

    revised = DorisStorage.revise_run_note(
        "proj", "n-1", expected_revision=1, body_md="v2", source="mcp"
    )
    assert revised["revision"] == 2
    with pytest.raises(RunNoteConflictError, match="at revision 2") as caught:
        DorisStorage.revise_run_note(
            "proj", "n-1", expected_revision=1, body_md="stale", source="cli"
        )
    assert caught.value.current_revision == 2

    project_note = _add(DorisStorage, "proj", scope="project", kind="plan")
    by_run = _add(DorisStorage, "proj", run_id="run-1", kind="decision")
    assert by_run["run_name"] == "train"
    tombstone = DorisStorage.delete_run_note(
        "proj", "n-1", expected_revision=2, source="observatory"
    )
    assert tombstone["deleted"] is True
    assert {note["note_id"] for note in DorisStorage.get_run_notes("proj")} == {
        project_note["note_id"],
        by_run["note_id"],
    }
    assert DorisStorage.get_run_notes("proj", scope="project") == [project_note]
    assert DorisStorage.get_run_notes("proj", kind="decision") == [by_run]
    assert DorisStorage.get_run_notes("other") == []
    assert [
        (entry["revision"], entry["deleted"])
        for entry in DorisStorage.get_run_note_history("proj", "n-1")
    ] == [(1, False), (2, False), (3, True)]


def test_doris_run_lifecycle_updates_and_removes_notes(doris):
    doris("proj", "run-1", "train")
    doris("proj", "run-2", "eval")
    doris("proj", "run-3", "sweep")
    renamed = _add(DorisStorage, "proj", run_id="run-1")
    deleted = _add(DorisStorage, "proj", run_id="run-2")
    purged = _add(DorisStorage, "proj", run_id="run-3")
    project_note = _add(DorisStorage, "proj", scope="project")

    DorisStorage.rename_run("proj", "train", "train-renamed", run_id="run-1")
    assert DorisStorage.delete_run("proj", "eval", run_id="run-2")
    DorisStorage.purge_runs("proj", ("run-3",), ())

    remaining = {
        note["note_id"]: note
        for note in DorisStorage.get_run_notes("proj", include_deleted=True)
    }
    assert set(remaining) == {renamed["note_id"], project_note["note_id"]}
    assert remaining[renamed["note_id"]]["run_name"] == "train-renamed"
    assert DorisStorage.get_run_note_history("proj", deleted["note_id"]) == []
    assert DorisStorage.get_run_note_history("proj", purged["note_id"]) == []


def test_doris_run_deletion_removes_per_trace_reward_components_and_environment_metrics(doris):
    for run_id, name in (("run-1", "keep"), ("run-2", "delete"), ("run-3", "purge")):
        doris("proj", run_id, name)
        for table in ("trace_reward_components", "trace_environment_metrics"):
            doris.connection.execute(
                f"INSERT INTO {table} (project_id, trace_id, run_id, projection_id, name) VALUES (?, ?, ?, ?, ?)",
                ("proj", f"trace-{run_id}", run_id, "p" * 64, "metric"),
            )

    def remaining(table):
        rows = doris.connection.execute(f"SELECT run_id FROM {table} ORDER BY run_id").fetchall()
        return [row["run_id"] for row in rows]

    assert DorisStorage.delete_run("proj", "delete", run_id="run-2")
    DorisStorage.purge_runs("proj", ("run-3",), ())
    for table in ("trace_reward_components", "trace_environment_metrics"):
        assert remaining(table) == ["run-1"]
