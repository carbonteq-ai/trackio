"""Preview and apply exact Trackio run purges.

The framework computes the cross-project dependency closure.  Trackio owns the
last-mile storage transaction: selected run rows, their links, and artifact
versions that have no retained consumer.  A digest binds the apply request to
the authenticated preview so a changed project cannot be deleted accidentally.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

from trackio import cas


def _digest(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _identity(record: Mapping[str, Any]) -> str | None:
    value = record.get("id") or record.get("name")
    return str(value) if value is not None else None


def _consumer_identity(record: Mapping[str, Any]) -> str | None:
    value = record.get("run_id") or record.get("run_name")
    return str(value) if value is not None else None


def _selected_records(storage: Any, project: str, run_ids: tuple[str, ...]) -> tuple[dict[str, Any], ...]:
    records = {
        identity: dict(record)
        for record in storage.get_run_records(project)
        if (identity := _identity(record)) is not None
    }
    selected: list[dict[str, Any]] = []
    for run_id in run_ids:
        record = records.get(run_id)
        if record is None:
            continue
        selected.append(record)
    return tuple(selected)


def run_purge_summary(
    storage: Any,
    project: str,
    run_ids: Iterable[str],
) -> dict[str, Any]:
    """Return an authenticated, deterministic preview for exact provider ids."""

    selected_ids = tuple(dict.fromkeys(str(run_id) for run_id in run_ids))
    selected_set = set(selected_ids)
    records = _selected_records(storage, project, selected_ids)
    found = {_identity(record) for record in records}
    blockers = [
        f"run {run_id!r} does not exist in Trackio project {project!r}"
        for run_id in selected_ids
        if run_id not in found
    ]
    artifacts: dict[str, dict[str, Any]] = {}
    for record in records:
        provider_id = _identity(record)
        if provider_id is None:
            continue
        links = storage.get_run_artifacts(
            project,
            run_name=record.get("name"),
            run_id=record.get("id"),
        )
        for output in links.get("output", []):
            version_id = str(output["version_id"])
            entry = artifacts.setdefault(
                version_id,
                {
                    "version_id": int(output["version_id"]),
                    "name": str(output["name"]),
                    "version": int(output["version"]),
                    "size_bytes": int(output.get("size_bytes", 0)),
                    "producer_run_ids": [],
                    "consumer_run_ids": [],
                    "delete": True,
                },
            )
            entry["producer_run_ids"].append(provider_id)
            consumers = storage.get_artifact_consumers(project, int(output["version_id"]))
            for consumer in consumers:
                consumer_id = _consumer_identity(consumer)
                if consumer_id is None:
                    blockers.append(
                        f"artifact version {version_id} has a consumer without a run identity"
                    )
                    entry["delete"] = False
                    continue
                if consumer_id not in entry["consumer_run_ids"]:
                    entry["consumer_run_ids"].append(consumer_id)
                if consumer_id not in selected_set:
                    entry["delete"] = False
                    blockers.append(
                        f"artifact version {version_id} is consumed by unselected run {consumer_id!r}"
                    )

    for entry in artifacts.values():
        entry["producer_run_ids"].sort()
        entry["consumer_run_ids"].sort()
    ordered_artifacts = [artifacts[key] for key in sorted(artifacts)]
    unique_blockers = tuple(dict.fromkeys(blockers))
    semantic = {
        "project": project,
        "run_ids": list(selected_ids),
        "artifacts": ordered_artifacts,
        "blockers": list(unique_blockers),
    }
    return {
        **semantic,
        "provider": "trackio",
        "exists": bool(records),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "digest": _digest(semantic),
    }


def purge_runs(
    storage: Any,
    project: str,
    run_ids: Iterable[str],
    *,
    plan_digest: str,
) -> dict[str, Any]:
    """Apply a previously previewed run purge after rechecking its digest."""

    summary = run_purge_summary(storage, project, run_ids)
    if summary["digest"] != plan_digest:
        raise ValueError("Trackio run purge plan is stale; obtain a new preview")
    if summary["blockers"]:
        raise ValueError("Trackio run purge is blocked: " + "; ".join(summary["blockers"]))
    artifact_version_ids = tuple(
        int(artifact["version_id"])
        for artifact in summary["artifacts"]
        if artifact["delete"]
    )
    storage.purge_runs(
        project,
        tuple(dict.fromkeys(str(run_id) for run_id in run_ids)),
        artifact_version_ids,
    )
    return {
        "provider": "trackio",
        "project": project,
        "plan_digest": plan_digest,
        "deleted_provider_run_ids": list(summary["run_ids"]),
        "deleted_artifact_version_ids": list(artifact_version_ids),
        "already_absent_provider_run_ids": [],
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def project_delete_plan(storage: Any, project: str) -> dict[str, Any]:
    """Return a deterministic digest-bound preview for one project boundary."""

    summary = dict(storage.project_delete_summary(project))
    semantic = {
        key: summary[key]
        for key in sorted(summary)
        if key not in {"deleted", "created_at", "digest"}
    }
    return {
        **summary,
        "provider": "trackio",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "digest": _digest(semantic),
    }


def delete_project(storage: Any, project: str, *, plan_digest: str) -> dict[str, Any]:
    """Apply a project deletion only when its preview digest is unchanged."""

    plan = project_delete_plan(storage, project)
    if plan["digest"] != plan_digest:
        raise ValueError("Trackio project delete plan is stale; obtain a new preview")
    result = dict(storage.delete_project(project))
    return {
        **result,
        "provider": "trackio",
        "project": project,
        "plan_digest": plan_digest,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def manifest_blob_digests(manifest: object) -> set[str]:
    """Extract local CAS digests from a Trackio artifact manifest."""

    if not isinstance(manifest, list):
        return set()
    digests: set[str] = set()
    for entry in manifest:
        if not isinstance(entry, dict):
            continue
        digest = entry.get("digest")
        if isinstance(digest, str):
            try:
                digests.add(str(cas.validate_digest(digest)))
            except ValueError:
                continue
    return digests


__all__ = [
    "delete_project",
    "manifest_blob_digests",
    "project_delete_plan",
    "purge_runs",
    "run_purge_summary",
]
