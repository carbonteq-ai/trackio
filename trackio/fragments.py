"""Append-only JSONL fragments used as a durable inbox for Trackio writes.

Fragments are immutable JSONL files, one writer (process) per subdirectory, so
concurrent training processes never contend on a shared file. They are written
either to a Hugging Face Bucket inbox (when a Space is unreachable) or to a
local inbox directory (when SQLite is unsafe, e.g. on network filesystems), and
are later imported into the project SQLite database by the process that owns it
(the Space or the dashboard server). Records carry the same ``log_id``/
``alert_id`` UUIDs as the HTTP logging endpoints, and imports use
``INSERT OR IGNORE``, so importing a fragment is idempotent: fragments are
deleted only after a successful import, and re-importing after a crash is
harmless.

A claimed batch that fails is retried one fragment at a time so a single
unimportable fragment cannot hold healthy fragments hostage. Each failure is
classified: storage outages keep the fragment pending with backoff; data the
backend will never accept is moved, never deleted, to a dead-letter directory
beside the inbox; other failures (for example a trace fact whose parent trace
has not arrived) are retried with backoff until
``TRACKIO_INBOX_RETRY_MAX_AGE`` seconds after their first failure and are then
dead-lettered. Retry state is kept in a ``<fragment>.retry.json`` sidecar so it
survives server restarts.
"""

import logging
import os
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import huggingface_hub
import orjson
import pymysql

from trackio import utils
from trackio.typehints import AlertEntry, LogEntry, SystemLogEntry

FRAGMENT_VERSION = 1
INBOX_DIR_NAME = "inbox"
DEAD_LETTER_DIR_NAME = "inbox-dead-letter"
RETRY_STATE_SUFFIX = ".retry.json"
DEAD_LETTER_ERROR_SUFFIX = ".error.json"
PROCESSING_SUFFIX = ".processing"
BUCKET_INBOX_PREFIX = "trackio/inbox"
BUCKET_MEDIA_PREFIX = "trackio/media"

METRIC_KIND = "metric"
SYSTEM_METRIC_KIND = "system_metric"
ALERT_KIND = "alert"
TRACE_FACT_KIND = "trace_fact"
KINDS = {METRIC_KIND, SYSTEM_METRIC_KIND, ALERT_KIND, TRACE_FACT_KIND}


logger = logging.getLogger("trackio")

TRANSIENT = "transient"
PERMANENT = "permanent"
RETRYABLE = "retryable"

_RETRY_BASE_SECONDS = 5.0
_ERROR_MESSAGE_LIMIT = 2000
_TRANSIENT_ERROR_CODES = {1040, 1047, 1205, 1213, 2003, 2006, 2013}
_PERMANENT_MESSAGE_MARKERS = (
    "insert has filtered data",
    "strict mode",
    "too long than",
    "limit length",
    "data too long",
)
_TRANSIENT_MESSAGE_MARKERS = (
    "timeout",
    "timed out",
    "too many versions",
    "not alive",
    "try again",
    "temporarily unavailable",
    "connection refused",
    "lost connection",
)


def local_inbox_dir() -> Path:
    return utils.TRACKIO_DIR / INBOX_DIR_NAME


def dead_letter_dir(inbox_dir: Path | None = None) -> Path:
    """Return the directory that holds fragments the importer gave up on."""

    inbox = inbox_dir or local_inbox_dir()
    return inbox.parent / DEAD_LETTER_DIR_NAME


def _float_setting(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError:
        return default
    return max(value, 0.0)


def retry_max_age_seconds() -> float:
    return _float_setting("TRACKIO_INBOX_RETRY_MAX_AGE", 86400.0)


def retry_max_backoff_seconds() -> float:
    return max(_float_setting("TRACKIO_INBOX_RETRY_MAX_BACKOFF", 300.0), 1.0)


def metric_record(entry: LogEntry | dict) -> dict:
    return {
        "v": FRAGMENT_VERSION,
        "kind": METRIC_KIND,
        "project": entry["project"],
        "run": entry["run"],
        "run_id": entry.get("run_id"),
        "metrics": utils.serialize_values(entry.get("metrics") or {}),
        "step": entry.get("step"),
        "timestamp": entry.get("timestamp"),
        "config": utils.serialize_values(entry.get("config"))
        if entry.get("config")
        else None,
        "log_id": entry.get("log_id"),
    }


def system_metric_record(entry: SystemLogEntry | dict) -> dict:
    return {
        "v": FRAGMENT_VERSION,
        "kind": SYSTEM_METRIC_KIND,
        "project": entry["project"],
        "run": entry["run"],
        "run_id": entry.get("run_id"),
        "metrics": utils.serialize_values(entry.get("metrics") or {}),
        "timestamp": entry.get("timestamp"),
        "log_id": entry.get("log_id"),
    }


def alert_record(entry: AlertEntry | dict) -> dict:
    return {
        "v": FRAGMENT_VERSION,
        "kind": ALERT_KIND,
        "project": entry["project"],
        "run": entry["run"],
        "run_id": entry.get("run_id"),
        "title": entry["title"],
        "text": entry.get("text"),
        "level": entry.get("level"),
        "step": entry.get("step"),
        "timestamp": entry.get("timestamp"),
        "alert_id": entry.get("alert_id"),
    }


def trace_fact_record(entry: dict) -> dict:
    """Return one idempotent trace-fact update for the durable inbox.

    The update is deliberately separate from the native trace payload.  A
    trainer can calculate its algorithm reward later, while the importer keeps
    retrying this record until the trace it references is visible.
    """

    return {
        "v": FRAGMENT_VERSION,
        "kind": TRACE_FACT_KIND,
        "project": entry["project"],
        "run": entry["run"],
        "run_id": entry.get("run_id"),
        "update": utils.serialize_values(entry["update"]),
    }


def bucket_media_path(
    project: str,
    run: str | None,
    step: int | None,
    relative_path: str | None,
    filename: str,
) -> str:
    parts = [BUCKET_MEDIA_PREFIX, utils.canonical_project_name(project)]
    if run:
        parts.append(run)
        if step is not None:
            parts.append(str(step))
    else:
        parts.append("files")
        if relative_path:
            parts.append(str(relative_path))
    parts.append(filename)
    return "/".join(parts)


class FragmentWriter:
    """Writes immutable JSONL fragments for a single writer (process)."""

    def __init__(self, writer_id: str | None = None):
        self.writer_id = writer_id or uuid.uuid4().hex[:16]
        self._seq = 0
        self._lock = threading.Lock()

    def _next_fragment_name(self) -> str:
        with self._lock:
            name = f"{self._seq:08d}.jsonl"
            self._seq += 1
        return name

    @staticmethod
    def serialize_records(records: list[dict]) -> bytes:
        return b"".join(orjson.dumps(record) + b"\n" for record in records)

    def write_local(
        self, records: list[dict], inbox_dir: Path | None = None
    ) -> Path | None:
        if not records:
            return None
        inbox = inbox_dir or local_inbox_dir()
        writer_dir = inbox / self.writer_id
        fragment_path = writer_dir / self._next_fragment_name()
        data = self.serialize_records(records)
        for attempt in range(3):
            writer_dir.mkdir(parents=True, exist_ok=True)
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb", dir=writer_dir, suffix=".tmp", delete=False
                ) as tmp:
                    tmp.write(data)
                    tmp_path = Path(tmp.name)
                tmp_path.replace(fragment_path)
                return fragment_path
            except FileNotFoundError:
                if attempt == 2 or writer_dir.exists():
                    raise

    def write_to_bucket(self, records: list[dict], bucket_id: str) -> str | None:
        if not records:
            return None
        remote_path = (
            f"{BUCKET_INBOX_PREFIX}/{self.writer_id}/{self._next_fragment_name()}"
        )
        huggingface_hub.batch_bucket_files(
            bucket_id,
            add=[(self.serialize_records(records), remote_path)],
            token=huggingface_hub.utils.get_token(),
        )
        return remote_path


def parse_fragment_bytes(data: bytes) -> list[dict]:
    records = []
    for line in data.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = orjson.loads(line)
        except orjson.JSONDecodeError:
            continue
        if isinstance(record, dict) and record.get("kind") in KINDS:
            records.append(record)
    return records


def _group_by_run(records: list[dict]) -> dict[tuple, list[dict]]:
    grouped: dict[tuple, list[dict]] = {}
    for record in records:
        key = (record.get("project"), record.get("run"), record.get("run_id"))
        grouped.setdefault(key, []).append(record)
    return grouped


@dataclass(frozen=True)
class ClaimedFragment:
    """A fragment atomically claimed by the background importer."""

    path: Path
    records: list[dict]
    contains_trace: bool
    inbox: Path | None = None

    @property
    def pending_path(self) -> Path:
        return _pending_path(self.path)

    @property
    def only_trace_facts(self) -> bool:
        return all(record.get("kind") == TRACE_FACT_KIND for record in self.records)


def _record_contains_trace(record: dict) -> bool:
    if record.get("kind") == TRACE_FACT_KIND:
        return True
    if record.get("kind") != METRIC_KIND:
        return False
    metrics = record.get("metrics") or {}
    return any(str(key).startswith("traces/") for key in metrics)


def import_records(records: list[dict]) -> int:
    from trackio.storage import Storage  # noqa: PLC0415
    from trackio.trace_facts import TraceFactUpdate  # noqa: PLC0415

    metric_records = [r for r in records if r.get("kind") == METRIC_KIND]
    system_records = [r for r in records if r.get("kind") == SYSTEM_METRIC_KIND]
    alert_records = [r for r in records if r.get("kind") == ALERT_KIND]
    trace_fact_records = [r for r in records if r.get("kind") == TRACE_FACT_KIND]
    imported = 0

    for (project, run, run_id), group in _group_by_run(metric_records).items():
        if not project or not run:
            continue
        config = next((r["config"] for r in group if r.get("config")), None)
        has_timestamps = all(r.get("timestamp") for r in group)
        Storage.bulk_log(
            project=project,
            run=run,
            run_id=run_id,
            metrics_list=[r.get("metrics") or {} for r in group],
            steps=[r.get("step") for r in group],
            timestamps=[r["timestamp"] for r in group] if has_timestamps else None,
            config=config,
            log_ids=[r.get("log_id") for r in group],
        )
        imported += len(group)

    for (project, run, run_id), group in _group_by_run(system_records).items():
        if not project or not run:
            continue
        has_timestamps = all(r.get("timestamp") for r in group)
        Storage.bulk_log_system(
            project=project,
            run=run,
            run_id=run_id,
            metrics_list=[r.get("metrics") or {} for r in group],
            timestamps=[r["timestamp"] for r in group] if has_timestamps else None,
            log_ids=[r.get("log_id") for r in group],
        )
        imported += len(group)

    # Import source metric records before fact records.  A fact fragment that
    # arrives in an earlier scanner batch is left intact on a missing-parent
    # error and retried after the source trace is imported.
    for (project, run, run_id), group in _group_by_run(trace_fact_records).items():
        if not project or not run:
            continue
        updates = [TraceFactUpdate.from_payload(record["update"]) for record in group]
        Storage.upsert_trace_facts_batch(
            project=project,
            run=run,
            run_id=run_id,
            updates=updates,
        )
        imported += len(group)

    for (project, run, run_id), group in _group_by_run(alert_records).items():
        if not project or not run:
            continue
        has_timestamps = all(r.get("timestamp") for r in group)
        Storage.bulk_alert(
            project=project,
            run=run,
            run_id=run_id,
            titles=[r.get("title") or "" for r in group],
            texts=[r.get("text") for r in group],
            levels=[r.get("level") or "WARN" for r in group],
            steps=[r.get("step") for r in group],
            timestamps=[r["timestamp"] for r in group] if has_timestamps else None,
            alert_ids=[r.get("alert_id") for r in group],
        )
        imported += len(group)

    return imported


def _pending_path(path: Path) -> Path:
    if path.name.endswith(PROCESSING_SUFFIX):
        return path.with_name(path.name[: -len(PROCESSING_SUFFIX)])
    return path


def retry_state_path(fragment_path: Path) -> Path:
    """Return the sidecar that durably records a fragment's failed attempts."""

    pending = _pending_path(fragment_path)
    return pending.with_name(pending.name + RETRY_STATE_SUFFIX)


def read_retry_state(fragment_path: Path) -> dict[str, Any] | None:
    try:
        state = orjson.loads(retry_state_path(fragment_path).read_bytes())
    except (OSError, orjson.JSONDecodeError):
        return None
    return state if isinstance(state, dict) else None


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, suffix=".tmp", delete=False
    ) as tmp:
        tmp.write(orjson.dumps(value, option=orjson.OPT_INDENT_2))
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _is_backing_off(fragment_path: Path, now: float) -> bool:
    if not retry_state_path(fragment_path).exists():
        return False
    state = read_retry_state(fragment_path)
    if state is None:
        return False
    try:
        return float(state.get("next_attempt_at") or 0.0) > now
    except (TypeError, ValueError):
        return False


def classify_import_error(error: BaseException) -> str:
    """Classify an import failure as ``transient``, ``permanent``, or ``retryable``.

    ``transient`` failures are storage outages that must never discard evidence.
    ``permanent`` failures are data the backend will reject on every attempt.
    ``retryable`` failures may resolve later (for example a trace fact whose
    parent trace has not been imported yet) and are retried for a bounded time.
    """

    from trackio.storage import is_retryable_storage_error  # noqa: PLC0415

    message = str(error).lower()
    if any(marker in message for marker in _PERMANENT_MESSAGE_MARKERS):
        return PERMANENT
    if is_retryable_storage_error(error):
        return TRANSIENT
    if isinstance(error, (TimeoutError, ConnectionError)):
        return TRANSIENT
    if isinstance(error, pymysql.err.OperationalError):
        code = error.args[0] if error.args else None
        if code in _TRANSIENT_ERROR_CODES or any(
            marker in message for marker in _TRANSIENT_MESSAGE_MARKERS
        ):
            return TRANSIENT
        return RETRYABLE
    if isinstance(error, (pymysql.err.DataError, pymysql.err.IntegrityError)):
        return PERMANENT
    if isinstance(error, KeyError):
        return RETRYABLE
    if isinstance(error, (ValueError, TypeError)):
        return PERMANENT
    return RETRYABLE


def _error_summary(error: BaseException) -> tuple[str, str]:
    error_class = f"{type(error).__module__}.{type(error).__qualname__}"
    message = str(error)
    if len(message) > _ERROR_MESSAGE_LIMIT:
        message = message[:_ERROR_MESSAGE_LIMIT] + "...[truncated]"
    return error_class, message


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def _restore_pending(fragment: ClaimedFragment) -> None:
    if not fragment.path.exists() or fragment.path == fragment.pending_path:
        return
    try:
        fragment.path.replace(fragment.pending_path)
    except OSError:
        pass


def _dead_letter(
    fragment: ClaimedFragment,
    error: BaseException,
    classification: str,
    state: dict[str, Any],
) -> Path | None:
    inbox = fragment.inbox or local_inbox_dir()
    pending = fragment.pending_path
    try:
        relative = pending.relative_to(inbox)
    except ValueError:
        relative = Path(pending.parent.name) / pending.name
    target = dead_letter_dir(inbox) / relative
    if target.exists():
        target = target.with_name(f"{target.stem}.{uuid.uuid4().hex[:8]}.jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        fragment.path.replace(target)
    except FileNotFoundError:
        return None
    error_class, message = _error_summary(error)
    _write_json_atomic(
        target.with_name(target.name + DEAD_LETTER_ERROR_SUFFIX),
        {
            "fragment": relative.as_posix(),
            "classification": classification,
            "error_class": error_class,
            "message": message,
            "attempts": state["attempts"],
            "first_failure_at": _iso(state["first_failure_at"]),
            "dead_lettered_at": _iso(state["last_failure_at"]),
            "records": len(fragment.records),
            "kinds": sorted({str(record.get("kind")) for record in fragment.records}),
        },
    )
    retry_state_path(pending).unlink(missing_ok=True)
    logger.warning(
        "dead-lettered inbox fragment %s after %d attempt(s): %s (%s)",
        target,
        state["attempts"],
        error_class,
        classification,
    )
    return target


def _record_failure(
    fragment: ClaimedFragment,
    error: BaseException,
    *,
    classification: str | None = None,
) -> str:
    """Persist one failed attempt and either requeue or dead-letter the fragment.

    Returns ``"pending"``, ``"dead_letter"``, or ``"missing"`` when the claim
    disappeared (for example an operator moved the file aside). If the retry
    state cannot be written the fragment is still returned to the inbox.
    """

    if not fragment.path.exists():
        return "missing"
    now = time.time()
    classification = classification or classify_import_error(error)
    previous = read_retry_state(fragment.path) or {}
    try:
        attempts = int(previous.get("attempts") or 0) + 1
        first_failure_at = float(previous.get("first_failure_at") or now)
    except (TypeError, ValueError):
        attempts, first_failure_at = 1, now
    error_class, message = _error_summary(error)
    state = {
        "attempts": attempts,
        "first_failure_at": first_failure_at,
        "last_failure_at": now,
        "classification": classification,
        "error_class": error_class,
        "message": message,
    }
    expired = now - first_failure_at >= retry_max_age_seconds()
    try:
        if classification == PERMANENT or (classification == RETRYABLE and expired):
            if _dead_letter(fragment, error, classification, state) is not None:
                return "dead_letter"
            return "missing"
        delay = min(
            _RETRY_BASE_SECONDS * 2 ** min(attempts - 1, 16),
            retry_max_backoff_seconds(),
        )
        state["next_attempt_at"] = now + delay
        _write_json_atomic(retry_state_path(fragment.path), state)
    except OSError as state_error:
        logger.warning(
            "could not record inbox fragment failure for %s: %s",
            fragment.pending_path,
            state_error,
        )
        _restore_pending(fragment)
        return "pending"
    _restore_pending(fragment)
    logger.warning(
        "inbox fragment %s failed (%s, attempt %d, retry in %.0fs): %s: %s",
        fragment.pending_path,
        classification,
        attempts,
        delay,
        error_class,
        message[:200],
    )
    return "pending"


def _import_succeeded(fragment: ClaimedFragment) -> None:
    fragment.path.unlink(missing_ok=True)
    retry_state_path(fragment.path).unlink(missing_ok=True)


def claim_inbox_batch(
    inbox_dir: Path | None = None,
    *,
    max_files: int | None = 128,
) -> list[ClaimedFragment]:
    """Claim and parse a bounded batch without touching the storage backend.

    Only the scanner should call this function in the server. The atomic rename
    keeps it safe for callers that still use ``import_inbox_dir`` concurrently.
    Invalid or empty fragments are removed exactly as the legacy importer did.
    Fragments still inside their retry backoff window are skipped.
    """

    inbox = inbox_dir or local_inbox_dir()
    if not inbox.exists() or max_files is not None and max_files <= 0:
        return []
    claimed: list[ClaimedFragment] = []
    now = time.time()
    for fragment_path in sorted(inbox.rglob("*.jsonl")):
        if max_files is not None and len(claimed) >= max_files:
            break
        if _is_backing_off(fragment_path, now):
            continue
        processing_path = fragment_path.with_suffix(
            f"{fragment_path.suffix}{PROCESSING_SUFFIX}"
        )
        try:
            fragment_path.replace(processing_path)
        except FileNotFoundError:
            continue
        try:
            records = parse_fragment_bytes(processing_path.read_bytes())
            if not records:
                processing_path.unlink()
                retry_state_path(fragment_path).unlink(missing_ok=True)
                continue
            claimed.append(
                ClaimedFragment(
                    path=processing_path,
                    records=records,
                    contains_trace=any(
                        _record_contains_trace(record) for record in records
                    ),
                    inbox=inbox,
                )
            )
        except Exception:
            try:
                processing_path.replace(fragment_path)
            except OSError:
                pass
            raise
    return claimed


def _import_batch(fragments: list[ClaimedFragment]) -> int:
    priority = [
        record
        for fragment in fragments
        if not fragment.contains_trace
        for record in fragment.records
    ]
    traces = [
        record
        for fragment in fragments
        if fragment.contains_trace
        for record in fragment.records
    ]
    imported = import_records(priority)
    imported += import_records(traces)
    return imported


def import_claimed_fragments(fragments: list[ClaimedFragment]) -> int:
    """Import a claimed batch, writing scalar records before trace records.

    The batch is first imported as one unit. If that fails with anything other
    than a storage outage, each fragment is imported on its own (scalar
    fragments, then trace metrics, then trace facts) so healthy fragments are
    committed and removed while each failing fragment is requeued with backoff
    or dead-lettered according to :func:`classify_import_error`. Claims whose
    file has disappeared are skipped rather than imported from stale memory.
    """

    live = [fragment for fragment in fragments if fragment.path.exists()]
    if not live:
        return 0
    try:
        imported = _import_batch(live)
    except Exception as error:
        classification = classify_import_error(error)
        if classification == TRANSIENT or len(live) == 1:
            for fragment in live:
                _record_failure(fragment, error, classification=classification)
            return 0
        logger.warning(
            "inbox batch of %d fragments failed (%s: %s); importing individually",
            len(live),
            type(error).__name__,
            str(error)[:200],
        )
        return _import_individually(live)
    for fragment in live:
        _import_succeeded(fragment)
    return imported


def _import_individually(fragments: list[ClaimedFragment]) -> int:
    ordered = sorted(
        fragments,
        key=lambda fragment: (fragment.only_trace_facts, fragment.contains_trace),
    )
    imported = 0
    for index, fragment in enumerate(ordered):
        if not fragment.path.exists():
            continue
        try:
            count = _import_batch([fragment])
        except Exception as error:
            classification = classify_import_error(error)
            _record_failure(fragment, error, classification=classification)
            if classification == TRANSIENT:
                for remaining in ordered[index + 1 :]:
                    _record_failure(remaining, error, classification=classification)
                break
            continue
        _import_succeeded(fragment)
        imported += count
    return imported


def _recover_processing_fragments(inbox: Path) -> None:
    """Return fragments left in the processing state after a server restart.

    Retry sidecars are keyed by the pending fragment name, so recovered
    fragments keep their attempt count and first-failure time. Sidecars whose
    fragment no longer exists are removed.
    """

    for processing_path in inbox.rglob("*.jsonl.processing"):
        target = processing_path.with_suffix("")
        try:
            if target.exists():
                processing_path.unlink()
            else:
                processing_path.replace(target)
        except OSError:
            continue
    for state_path in inbox.rglob(f"*.jsonl{RETRY_STATE_SUFFIX}"):
        pending = state_path.with_name(state_path.name[: -len(RETRY_STATE_SUFFIX)])
        processing = pending.with_name(pending.name + PROCESSING_SUFFIX)
        if not pending.exists() and not processing.exists():
            state_path.unlink(missing_ok=True)


def recover_processing_fragments(inbox_dir: Path | None = None) -> None:
    """Return claimed fragments to the pending queue after a restart."""

    inbox = inbox_dir or local_inbox_dir()
    if inbox.exists():
        _recover_processing_fragments(inbox)


def import_inbox_dir(
    inbox_dir: Path | None = None,
    *,
    max_files: int | None = None,
) -> int:
    inbox = inbox_dir or local_inbox_dir()
    if not inbox.exists():
        return 0
    if max_files is None:
        _recover_processing_fragments(inbox)
    claimed = claim_inbox_batch(
        inbox,
        max_files=max_files,
    )
    imported = import_claimed_fragments(claimed)
    if max_files is None:
        for writer_dir in inbox.glob("*"):
            if writer_dir.is_dir():
                try:
                    writer_dir.rmdir()
                except OSError:
                    pass
    return imported


def list_bucket_inbox_paths(bucket_id: str) -> list[str]:
    try:
        items = huggingface_hub.list_bucket_tree(
            bucket_id,
            prefix=BUCKET_INBOX_PREFIX,
            recursive=True,
            token=huggingface_hub.utils.get_token(),
        )
    except Exception:
        return []
    return sorted(
        item.path
        for item in items
        if getattr(item, "type", None) == "file"
        and getattr(item, "path", "").endswith(".jsonl")
    )


def import_inbox_from_bucket(bucket_id: str) -> int:
    paths = list_bucket_inbox_paths(bucket_id)
    if not paths:
        return 0
    imported = 0
    consumed: list[str] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, remote_path in enumerate(paths):
            local_path = Path(tmp_dir) / f"{i}.jsonl"
            try:
                huggingface_hub.download_bucket_files(
                    bucket_id,
                    files=[(remote_path, str(local_path))],
                    token=huggingface_hub.utils.get_token(),
                )
                records = parse_fragment_bytes(local_path.read_bytes())
            except Exception:
                continue
            if records:
                imported += import_records(records)
            consumed.append(remote_path)
    if consumed:
        try:
            huggingface_hub.batch_bucket_files(
                bucket_id,
                delete=consumed,
                token=huggingface_hub.utils.get_token(),
            )
        except Exception:
            pass
    return imported


def _add_files_to_bucket(bucket_id: str, additions: list[tuple[str, str]]) -> None:
    if additions:
        huggingface_hub.batch_bucket_files(
            bucket_id,
            add=additions,
            token=huggingface_hub.utils.get_token(),
        )


def _upload_files_to_bucket(
    bucket_id: str,
    uploads: list[dict[str, Any]],
    remote_path: Callable[[dict[str, Any], Path], str],
) -> None:
    _add_files_to_bucket(
        bucket_id,
        [
            (str(p), remote_path(upload, p))
            for upload in uploads
            if (p := Path(upload["file_path"])).exists()
        ],
    )


def upload_media_files_to_bucket(bucket_id: str, uploads: list[dict[str, Any]]) -> None:
    _upload_files_to_bucket(
        bucket_id,
        uploads,
        lambda upload, p: bucket_media_path(
            project=upload["project"],
            run=upload.get("run"),
            step=upload.get("step"),
            relative_path=upload.get("relative_path"),
            filename=p.name,
        ),
    )


def upload_artifact_blobs_to_bucket(
    bucket_id: str, uploads: list[dict[str, Any]]
) -> None:
    _upload_files_to_bucket(
        bucket_id,
        uploads,
        lambda upload, p: f"trackio/{p.relative_to(utils.TRACKIO_DIR).as_posix()}",
    )
