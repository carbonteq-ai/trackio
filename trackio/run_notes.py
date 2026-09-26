"""Revisioned Markdown notes attached to a Trackio run or project.

A note is identified by a stable ``note_id``. Every edit appends a revision, so
older revisions stay readable. Deleting appends a tombstone revision that hides
the note from default listings while keeping its history. There is no author
concept: ``source`` records only how a revision was written.

This module holds the storage-engine-neutral rules. ``SQLiteStorage`` and
``DorisStorage`` read a note's revisions, ask this module which row to append,
and persist it.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

import orjson

from trackio.exceptions import TrackioAPIError, TrackioConflictError

NOTE_SCOPES = ("run", "project")
MAX_NOTE_ID_LENGTH = 128
MAX_KIND_BYTES = 128
MAX_SOURCE_BYTES = 64
MAX_TITLE_BYTES = 1024
MAX_BODY_BYTES = 1024 * 1024
MAX_METADATA_BYTES = 64 * 1024
NOTE_COLUMNS = (
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
    "parent_revision",
    "metadata",
)
_NOTE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")
KEEP: Any = object()


class RunNoteConflictError(TrackioConflictError):
    """A note write was based on a revision that is no longer current."""

    def __init__(
        self,
        message: str,
        *,
        note_id: str | None = None,
        current_revision: int | None = None,
    ) -> None:
        super().__init__(
            message,
            detail={
                "type": "run_note",
                "note_id": note_id,
                "current_revision": current_revision,
            },
        )
        self.note_id = note_id
        self.current_revision = current_revision

    @classmethod
    def from_conflict(cls, error: TrackioConflictError) -> RunNoteConflictError:
        detail = error.detail
        current = detail.get("current_revision")
        return cls(
            str(error),
            note_id=detail.get("note_id"),
            current_revision=current if isinstance(current, int) else None,
        )


class RunNoteNotFoundError(TrackioAPIError, LookupError):
    """The requested note does not exist in the project."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_note_id() -> str:
    return uuid.uuid4().hex


def _text(value: Any, name: str, max_bytes: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be text")
    if not value.strip():
        raise ValueError(f"{name} must be non-empty")
    if len(value.encode("utf-8")) > max_bytes:
        raise ValueError(f"{name} cannot exceed {max_bytes} bytes")
    return value


def validate_note_id(note_id: Any) -> str:
    if not isinstance(note_id, str) or not _NOTE_ID_RE.fullmatch(note_id):
        raise ValueError(
            "note_id must start with a letter or digit and contain only letters, "
            "digits, '.', '_', ':', '/', or '-'"
        )
    if len(note_id) > MAX_NOTE_ID_LENGTH:
        raise ValueError(f"note_id cannot exceed {MAX_NOTE_ID_LENGTH} characters")
    return note_id


def validate_kind(kind: Any) -> str:
    return _text(kind, "kind", MAX_KIND_BYTES).strip()


def validate_source(source: Any) -> str:
    return _text(source, "source", MAX_SOURCE_BYTES).strip()


def validate_body(body_md: Any) -> str:
    return _text(body_md, "body_md", MAX_BODY_BYTES)


def validate_title(title: Any) -> str | None:
    if title is None:
        return None
    if not isinstance(title, str):
        raise ValueError("title must be text or null")
    title = title.strip()
    if not title:
        return None
    if len(title.encode("utf-8")) > MAX_TITLE_BYTES:
        raise ValueError(f"title cannot exceed {MAX_TITLE_BYTES} bytes")
    return title


def validate_scope(scope: Any) -> str:
    if scope not in NOTE_SCOPES:
        raise ValueError(f"scope must be one of {', '.join(NOTE_SCOPES)}")
    return scope


def validate_expected_revision(expected_revision: Any) -> int:
    if (
        isinstance(expected_revision, bool)
        or not isinstance(expected_revision, int)
        or expected_revision < 1
    ):
        raise ValueError("expected_revision must be a positive integer")
    return expected_revision


def encode_metadata(metadata: Any) -> str | None:
    if metadata is None:
        return None
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be an object or null")
    try:
        encoded = orjson.dumps(dict(metadata), option=orjson.OPT_SORT_KEYS)
    except TypeError as error:
        raise ValueError(f"metadata must be JSON-serializable: {error}") from error
    if len(encoded) > MAX_METADATA_BYTES:
        raise ValueError(f"metadata cannot exceed {MAX_METADATA_BYTES} bytes")
    return encoded.decode("utf-8")


def decode_metadata(value: Any) -> dict | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        value = bytes(value)
    decoded = orjson.loads(value)
    return decoded if isinstance(decoded, dict) else None


def validate_run_target(
    scope: str, run_id: Any, run_name: Any
) -> tuple[str | None, str | None]:
    """Check the run identity shape for a scope before any storage lookup."""

    for name, value in (("run_id", run_id), ("run_name", run_name)):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{name} must be non-empty text or null")
    if scope == "project":
        if run_id is not None or run_name is not None:
            raise ValueError("a project-scope note cannot name a run")
        return None, None
    if run_id is None and run_name is None:
        raise ValueError("a run-scope note requires run_id or run_name")
    return run_id, run_name


def row_to_note(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the public representation of one stored revision."""

    parent = row.get("parent_revision")
    return {
        "note_id": str(row["note_id"]),
        "revision": int(row["revision"]),
        "scope": str(row["scope"]),
        "run_id": None if row.get("run_id") is None else str(row["run_id"]),
        "run_name": None if row.get("run_name") is None else str(row["run_name"]),
        "kind": str(row["kind"]),
        "title": row.get("title"),
        "body_md": str(row["body_md"]),
        "source": str(row["source"]),
        "created_at": str(row["created_at"]),
        "revised_at": str(row["revised_at"]),
        "deleted": bool(int(row.get("deleted") or 0)),
        "parent_revision": None if parent is None else int(parent),
        "metadata": decode_metadata(row.get("metadata")),
    }


def note_to_row(note: Mapping[str, Any]) -> dict[str, Any]:
    """Return the storage representation of one public revision."""

    return {
        **{column: note.get(column) for column in NOTE_COLUMNS},
        "deleted": 1 if note.get("deleted") else 0,
        "metadata": encode_metadata(note.get("metadata")),
    }


def plan_add(
    history: list[dict[str, Any]],
    *,
    note_id: str,
    scope: str,
    run_id: str | None,
    run_name: str | None,
    kind: str,
    title: str | None,
    body_md: str,
    source: str,
    metadata: str | None,
    now: str,
) -> dict[str, Any] | None:
    """Return the first revision to store, or ``None`` for an identical retry.

    ``history`` holds the note's existing public revisions in ascending order.
    """

    if history:
        first = history[0]
        identical = (
            first["scope"] == scope
            and first["run_id"] == run_id
            and first["kind"] == kind
            and first["title"] == title
            and first["body_md"] == body_md
        )
        if identical:
            return None
        current = history[-1]["revision"]
        raise RunNoteConflictError(
            f"Run note {note_id!r} already exists at revision {current} with "
            "different content; revise it instead of adding it again.",
            note_id=note_id,
            current_revision=current,
        )
    return {
        "note_id": note_id,
        "revision": 1,
        "scope": scope,
        "run_id": run_id,
        "run_name": run_name,
        "kind": kind,
        "title": title,
        "body_md": body_md,
        "source": source,
        "created_at": now,
        "revised_at": now,
        "deleted": 0,
        "parent_revision": None,
        "metadata": metadata,
    }


def plan_revision(
    history: list[dict[str, Any]],
    *,
    project: str,
    note_id: str,
    expected_revision: int,
    source: str,
    now: str,
    body_md: str | None = None,
    kind: str | None = None,
    title: Any = KEEP,
    metadata: Any = KEEP,
    deleted: bool = False,
) -> dict[str, Any]:
    """Return the next revision of a note after optimistic-concurrency checks."""

    if not history:
        raise RunNoteNotFoundError(
            f"Run note {note_id!r} does not exist in project {project!r}"
        )
    latest = history[-1]
    current = latest["revision"]
    if latest["deleted"]:
        raise RunNoteConflictError(
            f"Run note {note_id!r} was deleted at revision {current}; "
            "a deleted note cannot be changed.",
            note_id=note_id,
            current_revision=current,
        )
    if current != expected_revision:
        raise RunNoteConflictError(
            f"Run note {note_id!r} is at revision {current}, not the expected "
            f"revision {expected_revision}; read the current revision and retry.",
            note_id=note_id,
            current_revision=current,
        )
    return {
        "note_id": note_id,
        "revision": current + 1,
        "scope": latest["scope"],
        "run_id": latest["run_id"],
        "run_name": latest["run_name"],
        "kind": latest["kind"] if kind is None else kind,
        "title": latest["title"] if title is KEEP else title,
        "body_md": latest["body_md"] if body_md is None else body_md,
        "source": source,
        "created_at": latest["created_at"],
        "revised_at": now,
        "deleted": 1 if deleted else 0,
        "parent_revision": current,
        "metadata": encode_metadata(latest["metadata"])
        if metadata is KEEP
        else metadata,
    }


def latest_notes(
    rows: Iterable[Mapping[str, Any]],
    *,
    kind: str | None = None,
    include_deleted: bool = False,
) -> list[dict[str, Any]]:
    """Reduce stored revisions to each note's latest revision, newest first."""

    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        note = row_to_note(row)
        previous = latest.get(note["note_id"])
        if previous is None or note["revision"] > previous["revision"]:
            latest[note["note_id"]] = note
    notes = [
        note
        for note in latest.values()
        if (include_deleted or not note["deleted"])
        and (kind is None or note["kind"] == kind)
    ]
    notes.sort(key=lambda note: (note["revised_at"], note["note_id"]), reverse=True)
    return notes


def history_from_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted((row_to_note(row) for row in rows), key=lambda n: n["revision"])
