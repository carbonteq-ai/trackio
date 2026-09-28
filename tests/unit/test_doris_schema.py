import pytest

from trackio.doris_schema import (
    MANAGED_TABLES,
    SCHEMA_VERSION,
    migration_statements,
    negotiate_schema,
    schema_statements,
)
from trackio.doris_storage import DorisStorage


def test_empty_database_is_the_only_bootstrap_state():
    assert negotiate_schema(set(), None) == "bootstrap"
    assert negotiate_schema({"unrelated_application_table"}, None) == "bootstrap"


def test_complete_current_schema_is_ready_without_bootstrap():
    assert negotiate_schema(set(MANAGED_TABLES), SCHEMA_VERSION) == "ready"


@pytest.mark.parametrize(
    ("tables", "version", "message"),
    [
        ({"metrics"}, None, "unversioned partial"),
        ({"schema_versions", "metrics"}, None, "unversioned partial"),
        (set(MANAGED_TABLES), SCHEMA_VERSION + 1, "newer"),
        (set(MANAGED_TABLES), SCHEMA_VERSION - 1, "explicit migration"),
        (
            set(MANAGED_TABLES) - {"artifact_aliases"},
            SCHEMA_VERSION,
            "artifact_aliases",
        ),
    ],
)
def test_nonempty_incompatible_schema_fails_closed(tables, version, message):
    with pytest.raises(RuntimeError, match=message):
        negotiate_schema(tables, version)


def test_schema_version_table_is_created_first_and_recorded_separately():
    statements = schema_statements()

    assert "CREATE TABLE IF NOT EXISTS schema_versions" in statements[0]
    assert len(statements) == len(MANAGED_TABLES)
    assert all(
        "INSERT INTO schema_versions" not in statement for statement in statements
    )


def test_version_two_migration_adds_trace_facts_before_recording_the_version():
    statements = migration_statements(1, 2)

    assert any("ALTER TABLE traces ADD COLUMN fact_projection_id" in statement for statement in statements)
    assert any("CREATE TABLE IF NOT EXISTS trace_reward_components" in statement for statement in statements)
    group_statements = migration_statements(2, 3)
    assert any("ADD COLUMN fact_task_id" in statement for statement in group_statements)
    assert any("ADD COLUMN fact_prompt_group_id" in statement for statement in group_statements)
    assert any("idx_trace_run_id" in statement for statement in group_statements)
    assert any("idx_trace_prompt_group_id" in statement for statement in group_statements)
    with pytest.raises(ValueError, match="unsupported"):
        migration_statements(SCHEMA_VERSION, SCHEMA_VERSION + 1)


def test_version_four_adds_the_revisioned_run_notes_table():
    assert "run_notes" in MANAGED_TABLES
    (statement,) = migration_statements(3, 4)
    normalized = " ".join(statement.split())

    assert "CREATE TABLE IF NOT EXISTS run_notes" in normalized
    assert "UNIQUE KEY(project_id, note_id, revision)" in normalized
    assert "idx_run_notes_run_id(run_id) USING INVERTED" in normalized
    for column in (
        "scope VARCHAR(16) NOT NULL",
        "run_id VARCHAR(255) NULL",
        "body_md STRING NOT NULL",
        "source VARCHAR(64) NOT NULL",
        'deleted TINYINT NOT NULL DEFAULT "0"',
        "parent_revision BIGINT NULL",
        "metadata STRING NULL",
    ):
        assert column in normalized
    bootstrap = [" ".join(item.split()) for item in schema_statements()]
    assert normalized in bootstrap


def test_version_five_adds_the_episode_ending_fact_column():
    assert SCHEMA_VERSION == 5
    assert migration_statements(4, 5) == (
        "ALTER TABLE traces ADD COLUMN fact_episode_ending VARCHAR(128) NULL",
    )
    traces = next(
        " ".join(statement.split())
        for statement in schema_statements()
        if "CREATE TABLE IF NOT EXISTS traces" in statement
    )
    assert "fact_episode_ending VARCHAR(128) NULL" in traces
    assert migration_statements(3, 5)[-1] == migration_statements(4, 5)[0]


def test_version_five_migration_skips_an_existing_column_and_records_the_version(
    monkeypatch, tmp_path
):
    from trackio import doris_schema_migration

    class _Cursor:
        def __init__(self, columns):
            self.columns = columns
            self.executed = []
            self.result = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, query, params=None):
            self.executed.append(query)
            if query.startswith("SELECT version FROM schema_versions"):
                self.result = [{"version": 4}]
            elif query == "DESCRIBE traces":
                self.result = [{"Field": column} for column in self.columns]
            elif query.startswith("SELECT TABLE_NAME"):
                self.result = [{"table_name": table} for table in MANAGED_TABLES]
            else:
                self.result = []

        def fetchone(self):
            return self.result[0] if self.result else None

        def fetchall(self):
            return self.result

    class _Connection:
        def __init__(self, cursor):
            self._cursor = cursor

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def cursor(self):
            return self._cursor

    receipt = tmp_path / "backup-receipt.json"
    receipt.write_text("{}")
    for columns, altered in (({"trace_id"}, True), ({"trace_id", "fact_episode_ending"}, False)):
        cursor = _Cursor(columns)
        monkeypatch.setattr(
            doris_schema_migration.DorisStorage,
            "_connection",
            staticmethod(lambda initialize=True, cursor=cursor: _Connection(cursor)),
        )
        result = doris_schema_migration.apply(5, receipt)
        assert result["current_version"] == 4 and result["target_version"] == 5
        assert any(query.startswith("ALTER TABLE traces ADD COLUMN fact_episode_ending") for query in cursor.executed) is altered
        assert any(query.lstrip().startswith("INSERT INTO schema_versions") for query in cursor.executed)


def test_multi_version_migration_is_the_ordered_single_steps():
    assert migration_statements(2, 4) == (
        *migration_statements(2, 3),
        *migration_statements(3, 4),
    )
    assert migration_statements(1, 4)[-1] == migration_statements(3, 4)[0]
    assert migration_statements(3, 5) == (
        *migration_statements(3, 4),
        *migration_statements(4, 5),
    )
    with pytest.raises(ValueError, match="unsupported"):
        migration_statements(4, 3)
    with pytest.raises(ValueError, match="unsupported"):
        migration_statements(0, 1)


class _SchemaCursor:
    def __init__(self, tables, version):
        self.tables = tables
        self.version = version
        self.executed = []
        self.current = ""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def execute(self, query, params=None):
        self.current = " ".join(query.split())
        self.executed.append((self.current, params))

    def fetchall(self):
        if "information_schema.tables" in self.current:
            return [{"table_name": table} for table in self.tables]
        return []

    def fetchone(self):
        if "SELECT version FROM schema_versions" in self.current:
            return {"version": self.version} if self.version is not None else None
        return None


class _SchemaConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.closed = False

    def cursor(self):
        return self._cursor

    def close(self):
        self.closed = True


def _install_schema_connection(monkeypatch, tables, version):
    cursor = _SchemaCursor(tables, version)
    connection = _SchemaConnection(cursor)
    settings = {
        "host": "doris.internal",
        "port": 9030,
        "user": "trackio",
        "password": "",
        "database": "trackio_test",
    }
    monkeypatch.setattr(DorisStorage, "_schema_ready", False)
    monkeypatch.setattr(DorisStorage, "_schema_target", None)
    monkeypatch.setattr(
        DorisStorage, "_settings", classmethod(lambda cls: settings.copy())
    )
    monkeypatch.setattr(
        "trackio.doris_storage.pymysql.connect",
        lambda **kwargs: connection,
    )
    return cursor, connection


def test_current_schema_negotiation_executes_no_ddl(monkeypatch):
    cursor, connection = _install_schema_connection(
        monkeypatch,
        set(MANAGED_TABLES),
        SCHEMA_VERSION,
    )

    DorisStorage._ensure_schema()

    statements = [query for query, _ in cursor.executed]
    assert not any(query.startswith("CREATE ") for query in statements)
    assert not any(query.startswith("INSERT ") for query in statements)
    assert connection.closed is True


def test_newer_schema_fails_before_any_write(monkeypatch):
    cursor, _ = _install_schema_connection(
        monkeypatch,
        set(MANAGED_TABLES),
        SCHEMA_VERSION + 1,
    )

    with pytest.raises(RuntimeError, match="newer"):
        DorisStorage._ensure_schema()

    statements = [query for query, _ in cursor.executed]
    assert not any(
        query.startswith(("CREATE ", "INSERT ", "UPDATE ", "DELETE "))
        for query in statements
    )


def test_empty_database_records_version_only_after_all_tables(monkeypatch):
    cursor, _ = _install_schema_connection(monkeypatch, set(), None)

    DorisStorage._ensure_schema()

    statements = [query for query, _ in cursor.executed]
    version_write = next(
        index
        for index, query in enumerate(statements)
        if query.startswith("INSERT INTO schema_versions")
    )
    table_writes = [
        index
        for index, query in enumerate(statements)
        if query.startswith("CREATE TABLE")
    ]
    assert len(table_writes) == len(MANAGED_TABLES)
    assert version_write > max(table_writes)
