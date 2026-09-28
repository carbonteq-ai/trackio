"""Read-only SQL over one project's data, for Doris and SQLite storage.

A statement is written in Doris SQL against logical tables that look the same
on both engines:

- ``metric_rows(run_id, run_name, step, timestamp, metrics)``: one row per
  logged batch; ``metrics`` is the JSON object of logged values.
- ``run_configs(run_id, run_name, config, created_at)``: ``config`` is JSON.
- ``traces(run_id, run_name, step, timestamp, trace_type, external_id,
  metadata, fact_*)``: trace facts are columns.
- ``run_notes(...)``: every revision of every note.

The statement must be one query (``SELECT``, optionally with ``WITH``) whose
table references are these names or its own CTEs, never qualified names or
table functions. It is wrapped in CTEs that read the project's rows from the
base tables, so it cannot see another project. On Doris it runs with a query
timeout; on SQLite storage it is translated to SQLite and runs on a read-only
connection under an authorizer, with a few Doris functions provided
(``json_extract_*``, ``max_by``, ``min_by``, ``stddev_samp``, ``stddev``,
``percentile``); anything else Doris-specific is refused there.
"""

from __future__ import annotations

import datetime as _datetime
import decimal
import json
import math
import re
import time
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Any

import sqlglot
from sqlglot import exp

FACT_COLUMNS = (
    "fact_state",
    "fact_calculator_version",
    "fact_model",
    "fact_task_type",
    "fact_task_id",
    "fact_prompt_group_id",
    "fact_episode_ending",
    "fact_rollout_step",
    "fact_is_truncated",
    "fact_has_error",
    "fact_model_input_tokens",
    "fact_model_output_tokens",
    "fact_thinking_tokens",
    "fact_tool_calls",
    "fact_model_calls",
    "fact_trace_latency_ms",
    "fact_task_reward",
    "fact_algorithm_reward",
)
LOGICAL_TABLES: dict[str, tuple[str, tuple[str, ...]]] = {
    "metric_rows": ("metrics", ("run_id", "run_name", "step", "timestamp", "metrics")),
    "run_configs": ("configs", ("run_id", "run_name", "config", "created_at")),
    "traces": (
        "traces",
        (
            "run_id",
            "run_name",
            "step",
            "timestamp",
            "trace_type",
            "external_id",
            "metadata",
            *FACT_COLUMNS,
        ),
    ),
    "run_notes": (
        "run_notes",
        (
            "note_id",
            "revision",
            "scope",
            "run_id",
            "run_name",
            "kind",
            "title",
            "body_md",
            "source",
            "created_at",
            "revised_at",
            "deleted",
        ),
    ),
}
DEFAULT_MAX_ROWS = 10_000
DEFAULT_TIMEOUT_SECONDS = 10.0
MAX_TIMEOUT_SECONDS = 120.0
SQLITE_FUNCTIONS = (
    "json_extract_double",
    "json_extract_bigint",
    "json_extract_int",
    "json_extract_string",
    "json_extract_bool",
    "max_by",
    "min_by",
    "stddev_samp",
    "stddev",
    "percentile",
    "unix_timestamp",
)


class ProjectSqlError(ValueError):
    """A statement that cannot run, with a reason the author can act on."""


def prepare(sql: str, *, engine: str, project: str, database: str | None = None) -> str:
    """Validate a Doris SQL statement and return it scoped to the project, in the engine's dialect."""

    if not sql or not sql.strip():
        raise ProjectSqlError("the query is empty")
    try:
        statements = [
            statement
            for statement in sqlglot.parse(sql, read="doris")
            if statement is not None
        ]
    except sqlglot.errors.SqlglotError as error:
        raise ProjectSqlError(f"cannot parse the query: {error}") from error
    if len(statements) != 1:
        raise ProjectSqlError("give exactly one statement")
    statement = statements[0]
    if not isinstance(statement, exp.Query):
        raise ProjectSqlError(
            "only a read-only SELECT (optionally with WITH) is allowed"
        )
    own = {cte.alias_or_name for cte in statement.find_all(exp.CTE)}
    shadowed = sorted(own & set(LOGICAL_TABLES))
    if shadowed:
        raise ProjectSqlError(f"a CTE cannot reuse the table name {shadowed[0]!r}")
    used: list[str] = []
    for table in statement.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            raise ProjectSqlError(
                "table functions are not allowed; read the project's tables"
            )
        if table.args.get("db") or table.args.get("catalog"):
            raise ProjectSqlError(
                f"use unqualified table names; {table.sql(dialect='doris')!r} is not allowed"
            )
        if table.name in own:
            continue
        if table.name not in LOGICAL_TABLES:
            raise ProjectSqlError(
                f"unknown table {table.name!r}; tables: {', '.join(LOGICAL_TABLES)}"
            )
        if table.name not in used:
            used.append(table.name)
    if any(isinstance(node, exp.Into) for node in statement.find_all(exp.Into)):
        raise ProjectSqlError("SELECT ... INTO is not allowed")
    scoped = [
        _scope(name, engine=engine, project=project, database=database) for name in used
    ]
    with_ = statement.args.get("with_")
    if with_ is None:
        statement.set("with_", exp.With(expressions=scoped))
    else:
        with_.set("expressions", [*scoped, *with_.expressions])
    return statement.sql(dialect="doris" if engine == "doris" else "sqlite")


def _scope(name: str, *, engine: str, project: str, database: str | None) -> exp.CTE:
    base, columns = LOGICAL_TABLES[name]
    select = exp.select(*(exp.column(column) for column in columns))
    if engine == "doris":
        if not database:
            raise ProjectSqlError("Doris queries need the database name")
        select = select.from_(exp.table_(base, db=database)).where(
            exp.EQ(
                this=exp.column("project_id"), expression=exp.Literal.string(project)
            )
        )
    else:
        # Qualified, so a logical table named like its base table (traces) is not a circular CTE.
        select = select.from_(exp.table_(base, db="main"))
    return exp.CTE(this=select, alias=exp.TableAlias(this=exp.to_identifier(name)))


def normalize_value(value: Any) -> Any:
    """A JSON-safe cell value."""

    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, _datetime.datetime | _datetime.date):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run_doris(
    connection: Any, prepared: str, *, max_rows: int, timeout_seconds: float
) -> tuple[list[str], list[list[Any]], bool]:
    """Run a prepared statement on a Doris connection with a query timeout."""

    with connection.cursor() as cursor:
        cursor.execute("SELECT @@query_timeout AS value")
        previous = (cursor.fetchone() or {}).get("value")
        cursor.execute("SET query_timeout = %s", (max(1, math.ceil(timeout_seconds)),))
        try:
            cursor.execute(prepared)
            columns = [description[0] for description in cursor.description or ()]
            fetched = cursor.fetchmany(max_rows + 1)
        finally:
            if previous is not None:
                cursor.execute("SET query_timeout = %s", (previous,))
    rows = [
        [
            normalize_value(_cell(row, column, index))
            for index, column in enumerate(columns)
        ]
        for row in fetched
    ]
    return columns, rows[:max_rows], len(rows) > max_rows


def _cell(row: Any, column: str, index: int) -> Any:
    return row[column] if isinstance(row, dict) else row[index]


def run_sqlite(
    connection: Any,
    prepared: str,
    *,
    max_rows: int,
    timeout_seconds: float,
    authorizer: Callable[..., int],
) -> tuple[list[str], list[list[Any]], bool]:
    """Run a prepared statement on a read-only SQLite connection."""

    register_doris_functions(connection)
    deadline = time.monotonic() + timeout_seconds
    connection.set_authorizer(authorizer)
    connection.set_progress_handler(
        lambda: 1 if time.monotonic() > deadline else 0, 10_000
    )
    try:
        cursor = connection.execute(prepared)
        columns = [description[0] for description in cursor.description or ()]
        fetched = cursor.fetchmany(max_rows + 1)
    except Exception as error:
        message = str(error)
        if "interrupted" in message:
            raise ProjectSqlError(
                f"the query ran longer than {timeout_seconds:g} seconds"
            ) from error
        missing = re.match(r"no such function: (\w+)", message)
        if missing:
            raise ProjectSqlError(
                f"{missing.group(1).lower()} is not available on SQLite storage; Doris functions provided there: "
                f"{', '.join(SQLITE_FUNCTIONS)}"
            ) from error
        raise ProjectSqlError(message) from error
    finally:
        connection.set_authorizer(None)
        connection.set_progress_handler(None, 0)
    rows = [[normalize_value(value) for value in tuple(row)] for row in fetched]
    return columns, rows[:max_rows], len(rows) > max_rows


# ---------------------------------------------------------------- SQLite stand-ins for Doris functions

_PATH = re.compile(r'\.(?:"((?:[^"\\]|\\.)*)"|([A-Za-z_][A-Za-z0-9_]*))|\[(\d+)\]')


@lru_cache(maxsize=256)
def _json(text: str) -> Any:
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=1024)
def _path(path: str) -> tuple[str | int, ...] | None:
    if not path.startswith("$"):
        return None
    parts: list[str | int] = []
    position = 1
    while position < len(path):
        match = _PATH.match(path, position)
        if match is None:
            return None
        quoted, plain, index = match.groups()
        parts.append(
            int(index)
            if index is not None
            else (quoted if quoted is not None else plain)
        )
        position = match.end()
    return tuple(parts)


def json_value(document: Any, path: Any) -> Any:
    # The turso engine stores JSON text columns as bytes.
    if isinstance(document, bytes | bytearray | memoryview):
        document = bytes(document).decode("utf-8", errors="replace")
    if not isinstance(document, str) or not isinstance(path, str):
        return None
    value = _json(document)
    parts = _path(path)
    if parts is None:
        return None
    for part in parts:
        if isinstance(part, int):
            value = (
                value[part] if isinstance(value, list) and part < len(value) else None
            )
        else:
            value = value.get(part) if isinstance(value, dict) else None
        if value is None:
            return None
    return value


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    return float(value) if isinstance(value, int | float) else None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return int(value) if isinstance(value, float) and value.is_integer() else None


def _as_string(value: Any) -> str | None:
    if value is None:
        return None
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True)


class _By:
    def __init__(self, later: bool) -> None:
        self.later = later
        self.best: tuple[Any, Any] | None = None

    def step(self, value: Any, order: Any) -> None:
        if order is None:
            return
        if self.best is None or (
            (order > self.best[1]) if self.later else (order < self.best[1])
        ):
            self.best = (value, order)

    def finalize(self) -> Any:
        return None if self.best is None else self.best[0]


class _MaxBy(_By):
    def __init__(self) -> None:
        super().__init__(later=True)


class _MinBy(_By):
    def __init__(self) -> None:
        super().__init__(later=False)


class _StddevSample:
    def __init__(self) -> None:
        self.values: list[float] = []

    def step(self, value: Any) -> None:
        number = _as_float(value)
        if number is not None:
            self.values.append(number)

    def finalize(self) -> float | None:
        if len(self.values) < 2:
            return None if not self.values else 0.0
        mean = math.fsum(self.values) / len(self.values)
        return math.sqrt(
            math.fsum((value - mean) ** 2 for value in self.values)
            / (len(self.values) - 1)
        )


class _StddevPopulation(_StddevSample):
    def finalize(self) -> float | None:
        if not self.values:
            return None
        mean = math.fsum(self.values) / len(self.values)
        return math.sqrt(
            math.fsum((value - mean) ** 2 for value in self.values) / len(self.values)
        )


class _Percentile:
    def __init__(self) -> None:
        self.values: list[float] = []
        self.fraction: float | None = None

    def step(self, value: Any, fraction: Any) -> None:
        number = _as_float(value)
        if number is not None:
            self.values.append(number)
        if self.fraction is None and isinstance(fraction, int | float):
            self.fraction = float(fraction)

    def finalize(self) -> float | None:
        if not self.values or self.fraction is None or not 0.0 <= self.fraction <= 1.0:
            return None
        ordered = sorted(self.values)
        position = self.fraction * (len(ordered) - 1)
        lower = math.floor(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _unix_timestamp(value: Any = None) -> int | None:
    if value is None:
        return int(time.time())
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if not isinstance(value, str):
        return None
    try:
        parsed = _datetime.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_datetime.timezone.utc)
    return int(parsed.timestamp())


def register_doris_functions(connection: Any) -> None:
    """Provide the Doris functions semantic queries rely on to a SQLite connection."""

    connection.create_function("unix_timestamp", 1, _unix_timestamp)
    connection.create_function("unix_timestamp", 0, _unix_timestamp)

    connection.create_function(
        "json_extract_double",
        2,
        lambda d, p: _as_float(json_value(d, p)),
        deterministic=True,
    )
    connection.create_function(
        "json_extract_bigint",
        2,
        lambda d, p: _as_int(json_value(d, p)),
        deterministic=True,
    )
    connection.create_function(
        "json_extract_int",
        2,
        lambda d, p: _as_int(json_value(d, p)),
        deterministic=True,
    )
    connection.create_function(
        "json_extract_string",
        2,
        lambda d, p: _as_string(json_value(d, p)),
        deterministic=True,
    )
    connection.create_function(
        "json_extract_bool",
        2,
        lambda d, p: (lambda v: int(v) if isinstance(v, bool) else None)(
            json_value(d, p)
        ),
        deterministic=True,
    )
    # sqlglot writes Doris max_by/min_by as arg_max/arg_min for SQLite.
    for name, aggregate in (
        ("max_by", _MaxBy),
        ("arg_max", _MaxBy),
        ("min_by", _MinBy),
        ("arg_min", _MinBy),
    ):
        connection.create_aggregate(name, 2, aggregate)
    connection.create_aggregate("stddev_samp", 1, _StddevSample)
    connection.create_aggregate("stddev", 1, _StddevPopulation)
    connection.create_aggregate("percentile", 2, _Percentile)


def clamp(max_rows: int | None, timeout_seconds: float | None) -> tuple[int, float]:
    rows = DEFAULT_MAX_ROWS if max_rows is None else int(max_rows)
    if not 1 <= rows <= 100_000:
        raise ProjectSqlError("max_rows must be between 1 and 100000")
    seconds = (
        DEFAULT_TIMEOUT_SECONDS if timeout_seconds is None else float(timeout_seconds)
    )
    if not 0 < seconds <= MAX_TIMEOUT_SECONDS:
        raise ProjectSqlError(
            f"timeout_seconds must be above 0 and at most {MAX_TIMEOUT_SECONDS:g}"
        )
    return rows, seconds


def result(
    engine: str, columns: Sequence[str], rows: list[list[Any]], truncated: bool
) -> dict[str, Any]:
    return {
        "engine": engine,
        "columns": list(columns),
        "rows": rows,
        "truncated": truncated,
    }


__all__ = [
    "DEFAULT_MAX_ROWS",
    "DEFAULT_TIMEOUT_SECONDS",
    "FACT_COLUMNS",
    "LOGICAL_TABLES",
    "ProjectSqlError",
    "SQLITE_FUNCTIONS",
    "clamp",
    "json_value",
    "normalize_value",
    "prepare",
    "register_doris_functions",
    "result",
    "run_doris",
    "run_sqlite",
]
