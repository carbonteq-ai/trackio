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


def _link_identity(record: Mapping[str, Any]) -> str | None:
    value = record.get("run_id") or record.get("run_name")
    return str(value) if value is not None else None


def _selected_records(
    storage: Any, project: str, run_ids: tuple[str, ...]
) -> tuple[dict[str, Any], ...]:
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
    *,
    allow_absent: bool = False,
) -> dict[str, Any]:
    """Return an authenticated, deterministic preview for exact provider ids.

    Identical artifact content logged by several runs is one shared version
    with an output link per run.  Such a version is retained (never deleted
    and never a blocker) while any unselected run still links it as an
    output; its unselected consumers keep that surviving producer.  A version
    only selected runs produced is deleted, and an unselected consumer of it
    blocks.  ``allow_absent`` reports selected runs that no longer exist in
    ``absent_run_ids`` instead of blocking, so an interrupted purge can be
    finished without widening it.
    """

    selected_ids = tuple(dict.fromkeys(str(run_id) for run_id in run_ids))
    selected_set = set(selected_ids)
    records = _selected_records(storage, project, selected_ids)
    found = {_identity(record) for record in records}
    absent = [run_id for run_id in selected_ids if run_id not in found]
    selected_names = {
        str(record["name"]) for record in records if record.get("name") is not None
    }
    blockers = (
        []
        if allow_absent
        else [
            f"run {run_id!r} does not exist in Trackio project {project!r}"
            for run_id in absent
        ]
    )

    def is_selected(link: Mapping[str, Any]) -> bool:
        if link.get("run_id") is not None:
            return str(link["run_id"]) in selected_set
        return (
            link.get("run_name") is not None and str(link["run_name"]) in selected_names
        )

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
                    "retained_by_run_ids": [],
                    "delete": True,
                },
            )
            if provider_id not in entry["producer_run_ids"]:
                entry["producer_run_ids"].append(provider_id)

    for version_id, entry in artifacts.items():
        unselected_consumers: list[str] = []
        for link in storage.get_artifact_version_links(project, int(version_id)):
            identity = _link_identity(link)
            if identity is None:
                blockers.append(
                    f"artifact version {version_id} has a link without a run identity"
                )
                entry["delete"] = False
                continue
            if link.get("direction") == "input":
                if identity not in entry["consumer_run_ids"]:
                    entry["consumer_run_ids"].append(identity)
                if not is_selected(link):
                    unselected_consumers.append(identity)
            elif not is_selected(link) and identity not in entry["retained_by_run_ids"]:
                entry["retained_by_run_ids"].append(identity)
        if entry["retained_by_run_ids"]:
            entry["delete"] = False
            continue
        for consumer_id in dict.fromkeys(unselected_consumers):
            entry["delete"] = False
            blockers.append(
                f"artifact version {version_id} is consumed by unselected run {consumer_id!r}"
            )

    for entry in artifacts.values():
        entry["producer_run_ids"].sort()
        entry["consumer_run_ids"].sort()
        entry["retained_by_run_ids"].sort()
    ordered_artifacts = [artifacts[key] for key in sorted(artifacts)]
    unique_blockers = tuple(dict.fromkeys(blockers))
    semantic = {
        "project": project,
        "run_ids": list(selected_ids),
        "absent_run_ids": absent,
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
    allow_absent: bool = False,
) -> dict[str, Any]:
    """Apply a previously previewed run purge after rechecking its digest."""

    summary = run_purge_summary(storage, project, run_ids, allow_absent=allow_absent)
    if summary["digest"] != plan_digest:
        raise ValueError("Trackio run purge plan is stale; obtain a new preview")
    if summary["blockers"]:
        raise ValueError(
            "Trackio run purge is blocked: " + "; ".join(summary["blockers"])
        )
    artifact_version_ids = tuple(
        int(artifact["version_id"])
        for artifact in summary["artifacts"]
        if artifact["delete"]
    )
    absent = set(summary["absent_run_ids"])
    present = tuple(run_id for run_id in summary["run_ids"] if run_id not in absent)
    if present or artifact_version_ids:
        storage.purge_runs(project, present, artifact_version_ids)
    return {
        "provider": "trackio",
        "project": project,
        "plan_digest": plan_digest,
        "deleted_provider_run_ids": list(present),
        "deleted_artifact_version_ids": list(artifact_version_ids),
        "already_absent_provider_run_ids": list(summary["absent_run_ids"]),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def _metadata_selector(metadata: object) -> dict[str, str]:
    if not isinstance(metadata, Mapping) or not metadata:
        raise ValueError("artifact metadata selector must be a non-empty object")
    selector: dict[str, str] = {}
    for key, value in metadata.items():
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
        ):
            raise ValueError(
                "artifact metadata selector keys and values must be non-empty strings"
            )
        selector[key] = value
    return dict(sorted(selector.items()))


def unreferenced_artifact_purge_summary(
    storage: Any,
    project: str,
    metadata: Mapping[str, str],
) -> dict[str, Any]:
    """Preview deletion of unreferenced artifact versions matching metadata.

    An unreferenced version has no run link (input or output) and no
    producer run that still exists.  Nothing in the project can reach it
    through lineage, so deleting it cannot change any run's evidence.  The
    caller names the versions it owns by exact metadata values (for example
    the framework run id it recorded when logging them).
    """

    selector = _metadata_selector(metadata)
    existing = {
        identity
        for record in storage.get_run_records(project)
        if (identity := _identity(record)) is not None
    }
    artifacts: list[dict[str, Any]] = []
    for version in storage.get_unreferenced_artifact_versions(project):
        recorded = version.get("metadata")
        if not isinstance(recorded, Mapping):
            continue
        if any(recorded.get(key) != value for key, value in selector.items()):
            continue
        producer = version.get("producer_run_id")
        if producer is not None and str(producer) in existing:
            continue
        artifacts.append(
            {
                "version_id": int(version["version_id"]),
                "name": str(version["name"]),
                "version": int(version["version"]),
                "size_bytes": int(version.get("size_bytes") or 0),
            }
        )
    artifacts.sort(key=lambda artifact: artifact["version_id"])
    semantic = {"project": project, "metadata": selector, "artifacts": artifacts}
    return {
        **semantic,
        "provider": "trackio",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "digest": _digest(semantic),
    }


def purge_unreferenced_artifacts(
    storage: Any,
    project: str,
    metadata: Mapping[str, str],
    *,
    plan_digest: str,
) -> dict[str, Any]:
    """Delete exactly the previewed unreferenced artifact versions."""

    summary = unreferenced_artifact_purge_summary(storage, project, metadata)
    if summary["digest"] != plan_digest:
        raise ValueError("Trackio artifact purge plan is stale; obtain a new preview")
    version_ids = tuple(artifact["version_id"] for artifact in summary["artifacts"])
    if version_ids:
        storage.purge_artifact_versions(project, version_ids)
    return {
        "provider": "trackio",
        "project": project,
        "plan_digest": plan_digest,
        "deleted_artifact_version_ids": list(version_ids),
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
    "purge_unreferenced_artifacts",
    "run_purge_summary",
    "unreferenced_artifact_purge_summary",
]
