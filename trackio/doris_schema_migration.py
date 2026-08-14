"""Explicit, operator-invoked Trackio Doris schema migrations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from trackio.doris_schema import SCHEMA_VERSION, migration_statements
from trackio.doris_storage import DorisStorage


def _current_version(cursor: Any) -> int:
    cursor.execute(
        "SELECT version FROM schema_versions WHERE component = %s LIMIT 1",
        ("trackio",),
    )
    row = cursor.fetchone()
    if row is None:
        raise RuntimeError("Apache Doris has no recorded Trackio schema version")
    return int(row["version"])


def preview(target: int) -> dict[str, object]:
    """Inspect the only supported migration without changing Doris."""

    with DorisStorage._connection(initialize=False) as connection, connection.cursor() as cursor:
        current = _current_version(cursor)
    return {
        "current_version": current,
        "target_version": target,
        "statements": list(migration_statements(current, target)),
    }


def apply(target: int, backup_receipt: Path) -> dict[str, object]:
    """Apply and verify the operator-approved migration after a retained backup."""

    if target != SCHEMA_VERSION:
        raise ValueError(f"target must be the runtime schema version {SCHEMA_VERSION}")
    if not backup_receipt.is_file() or backup_receipt.stat().st_size == 0:
        raise ValueError("--backup-receipt must name a non-empty verified backup receipt")
    with DorisStorage._connection(initialize=False) as connection, connection.cursor() as cursor:
        current = _current_version(cursor)
        statements = migration_statements(current, target)
        for statement in statements:
            if statement.lstrip().upper().startswith("ALTER TABLE TRACES ADD COLUMN"):
                column = statement.split()[5]
                cursor.execute("DESCRIBE traces")
                existing = {str(row["Field"]) for row in cursor.fetchall()}
                if column in existing:
                    continue
            cursor.execute(statement)
        cursor.execute("SELECT TABLE_NAME AS table_name FROM information_schema.tables WHERE table_schema = DATABASE()")
        tables = {str(row["table_name"]) for row in cursor.fetchall()}
        if "trace_reward_components" not in tables:
            raise RuntimeError("Doris trace-fact component table was not created")
        cursor.execute(
            """INSERT INTO schema_versions (component, version, applied_at)
               VALUES (%s, %s, UTC_TIMESTAMP())""",
            ("trackio", target),
        )
    DorisStorage._schema_ready = False
    return {
        "current_version": current,
        "target_version": target,
        "backup_receipt": str(backup_receipt),
        "statements_applied": len(statements),
    }
