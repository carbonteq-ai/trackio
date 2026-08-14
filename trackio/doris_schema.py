"""Versioned Apache Doris schema for Trackio's logical evidence store."""

from __future__ import annotations

SCHEMA_VERSION = 2
MANAGED_TABLES = (
    "schema_versions",
    "metrics",
    "configs",
    "system_metrics",
    "traces",
    "trace_reward_components",
    "alerts",
    "project_metadata",
    "artifacts",
    "artifact_versions",
    "artifact_aliases",
    "run_artifact_links",
)


def negotiate_schema(
    tables: set[str],
    recorded_version: int | None,
) -> str:
    """Return ``bootstrap`` or ``ready`` without changing Doris.

    A partially-created or unversioned Trackio schema is never repaired
    implicitly. That makes a failed bootstrap visible instead of allowing a
    later process to silently claim that incomplete tables are current.
    """

    managed = tables.intersection(MANAGED_TABLES)
    if not managed:
        return "bootstrap"
    if "schema_versions" not in managed or recorded_version is None:
        raise RuntimeError(
            "Apache Doris contains an unversioned partial Trackio schema; "
            "recover or remove it before startup"
        )
    if recorded_version > SCHEMA_VERSION:
        raise RuntimeError(
            "Apache Doris Trackio schema is newer than this Trackio runtime "
            f"({recorded_version} > {SCHEMA_VERSION})"
        )
    if recorded_version < SCHEMA_VERSION:
        raise RuntimeError(
            "Apache Doris Trackio schema requires an explicit migration "
            f"({recorded_version} < {SCHEMA_VERSION})"
        )
    missing = set(MANAGED_TABLES).difference(tables)
    if missing:
        raise RuntimeError(
            "Apache Doris Trackio schema is incomplete at version "
            f"{recorded_version}; missing: {', '.join(sorted(missing))}"
        )
    return "ready"


_TRACE_FACT_COLUMNS = (
    "fact_namespace VARCHAR(128) NULL",
    "fact_calculator_version VARCHAR(256) NULL",
    "fact_projection_id VARCHAR(64) NULL",
    "fact_state VARCHAR(32) NULL",
    "fact_calculated_at VARCHAR(64) NULL",
    "fact_dimensions STRING NULL",
    "fact_provenance STRING NULL",
    "fact_model VARCHAR(512) NULL",
    "fact_task_type VARCHAR(512) NULL",
    "fact_rollout_step BIGINT NULL",
    "fact_is_truncated BOOLEAN NULL",
    "fact_has_error BOOLEAN NULL",
    "fact_model_input_tokens DOUBLE NULL",
    "fact_model_output_tokens DOUBLE NULL",
    "fact_thinking_tokens DOUBLE NULL",
    "fact_tool_calls DOUBLE NULL",
    "fact_model_calls DOUBLE NULL",
    "fact_trace_latency_ms DOUBLE NULL",
    "fact_task_reward DOUBLE NULL",
    "fact_algorithm_reward DOUBLE NULL",
    "fact_algorithm_projection_id VARCHAR(64) NULL",
    "fact_algorithm_calculator_version VARCHAR(256) NULL",
    "fact_algorithm_calculated_at VARCHAR(64) NULL",
)


def migration_statements(from_version: int, to_version: int, replication_num: int = 1) -> tuple[str, ...]:
    """Return the explicit supported Doris database transition.

    Startup deliberately refuses to apply this itself. Operators inspect and
    apply the ordered statements as part of the coordinated Trackio upgrade.
    """

    if (from_version, to_version) != (1, 2):
        raise ValueError(f"unsupported Trackio Doris migration {from_version} -> {to_version}")
    properties = f'PROPERTIES ("replication_num" = "{replication_num}")'
    component_table = f"""
        CREATE TABLE IF NOT EXISTS trace_reward_components (
            project_id VARCHAR(255) NOT NULL,
            trace_id VARCHAR(768) NOT NULL,
            projection_id VARCHAR(64) NOT NULL,
            name VARCHAR(256) NOT NULL,
            run_id VARCHAR(255) NOT NULL,
            contribution DOUBLE NULL,
            score DOUBLE NULL,
            weight DOUBLE NULL,
            source_kind VARCHAR(32) NOT NULL,
            source_id VARCHAR(256) NULL
        )
        UNIQUE KEY(project_id, trace_id, projection_id, name)
        DISTRIBUTED BY HASH(project_id, trace_id) BUCKETS 1
        {properties}
    """
    return (
        *(f"ALTER TABLE traces ADD COLUMN {column}" for column in _TRACE_FACT_COLUMNS),
        component_table,
    )


def schema_statements(replication_num: int = 1) -> tuple[str, ...]:
    properties = f'PROPERTIES ("replication_num" = "{replication_num}")'
    return (
        f"""
        CREATE TABLE IF NOT EXISTS schema_versions (
            component VARCHAR(64) NOT NULL,
            version INT NOT NULL,
            applied_at VARCHAR(64) NOT NULL
        )
        UNIQUE KEY(component)
        DISTRIBUTED BY HASH(component) BUCKETS 1
        {properties}
        """,
        f"""
        CREATE TABLE IF NOT EXISTS metrics (
            project_id VARCHAR(255) NOT NULL,
            event_id VARCHAR(64) NOT NULL,
            run_id VARCHAR(255) NOT NULL,
            timestamp VARCHAR(64) NOT NULL,
            run_name VARCHAR(255) NOT NULL,
            step BIGINT NOT NULL,
            metrics STRING NOT NULL,
            log_id VARCHAR(255) NULL,
            space_id VARCHAR(255) NULL
        )
        UNIQUE KEY(project_id, event_id)
        DISTRIBUTED BY HASH(project_id, event_id) BUCKETS 1
        {properties}
        """,
        f"""
        CREATE TABLE IF NOT EXISTS configs (
            project_id VARCHAR(255) NOT NULL,
            run_id VARCHAR(255) NOT NULL,
            run_name VARCHAR(255) NOT NULL,
            config STRING NOT NULL,
            created_at VARCHAR(64) NOT NULL
        )
        UNIQUE KEY(project_id, run_id)
        DISTRIBUTED BY HASH(project_id, run_id) BUCKETS 1
        {properties}
        """,
        f"""
        CREATE TABLE IF NOT EXISTS system_metrics (
            project_id VARCHAR(255) NOT NULL,
            event_id VARCHAR(64) NOT NULL,
            run_id VARCHAR(255) NOT NULL,
            timestamp VARCHAR(64) NOT NULL,
            run_name VARCHAR(255) NOT NULL,
            metrics STRING NOT NULL,
            log_id VARCHAR(255) NULL,
            space_id VARCHAR(255) NULL
        )
        UNIQUE KEY(project_id, event_id)
        DISTRIBUTED BY HASH(project_id, event_id) BUCKETS 1
        {properties}
        """,
        f"""
        CREATE TABLE IF NOT EXISTS traces (
            project_id VARCHAR(255) NOT NULL,
            trace_id VARCHAR(768) NOT NULL,
            run_id VARCHAR(255) NOT NULL,
            timestamp VARCHAR(64) NOT NULL,
            run_name VARCHAR(255) NOT NULL,
            step BIGINT NOT NULL,
            metric_key VARCHAR(512) NOT NULL,
            trace_index INT NULL,
            messages STRING NOT NULL,
            metadata STRING NOT NULL,
            search_text STRING NOT NULL,
            log_id VARCHAR(255) NULL,
            space_id VARCHAR(255) NULL,
            trace_type VARCHAR(64) NOT NULL,
            external_id VARCHAR(768) NULL,
            schema_version INT NULL,
            payload STRING NULL,
            fact_namespace VARCHAR(128) NULL,
            fact_calculator_version VARCHAR(256) NULL,
            fact_projection_id VARCHAR(64) NULL,
            fact_state VARCHAR(32) NULL,
            fact_calculated_at VARCHAR(64) NULL,
            fact_dimensions STRING NULL,
            fact_provenance STRING NULL,
            fact_model VARCHAR(512) NULL,
            fact_task_type VARCHAR(512) NULL,
            fact_rollout_step BIGINT NULL,
            fact_is_truncated BOOLEAN NULL,
            fact_has_error BOOLEAN NULL,
            fact_model_input_tokens DOUBLE NULL,
            fact_model_output_tokens DOUBLE NULL,
            fact_thinking_tokens DOUBLE NULL,
            fact_tool_calls DOUBLE NULL,
            fact_model_calls DOUBLE NULL,
            fact_trace_latency_ms DOUBLE NULL,
            fact_task_reward DOUBLE NULL,
            fact_algorithm_reward DOUBLE NULL,
            fact_algorithm_projection_id VARCHAR(64) NULL,
            fact_algorithm_calculator_version VARCHAR(256) NULL,
            fact_algorithm_calculated_at VARCHAR(64) NULL
        )
        UNIQUE KEY(project_id, trace_id)
        DISTRIBUTED BY HASH(project_id, trace_id) BUCKETS 1
        {properties}
        """,
        f"""
        CREATE TABLE IF NOT EXISTS trace_reward_components (
            project_id VARCHAR(255) NOT NULL,
            trace_id VARCHAR(768) NOT NULL,
            projection_id VARCHAR(64) NOT NULL,
            name VARCHAR(256) NOT NULL,
            run_id VARCHAR(255) NOT NULL,
            contribution DOUBLE NULL,
            score DOUBLE NULL,
            weight DOUBLE NULL,
            source_kind VARCHAR(32) NOT NULL,
            source_id VARCHAR(256) NULL
        )
        UNIQUE KEY(project_id, trace_id, projection_id, name)
        DISTRIBUTED BY HASH(project_id, trace_id) BUCKETS 1
        {properties}
        """,
        f"""
        CREATE TABLE IF NOT EXISTS alerts (
            project_id VARCHAR(255) NOT NULL,
            event_id VARCHAR(64) NOT NULL,
            run_id VARCHAR(255) NOT NULL,
            timestamp VARCHAR(64) NOT NULL,
            run_name VARCHAR(255) NOT NULL,
            title STRING NOT NULL,
            text STRING NULL,
            level VARCHAR(32) NOT NULL,
            step BIGINT NULL,
            alert_id VARCHAR(255) NULL
        )
        UNIQUE KEY(project_id, event_id)
        DISTRIBUTED BY HASH(project_id, event_id) BUCKETS 1
        {properties}
        """,
        f"""
        CREATE TABLE IF NOT EXISTS project_metadata (
            project_id VARCHAR(255) NOT NULL,
            metadata_key VARCHAR(255) NOT NULL,
            metadata_value STRING NOT NULL
        )
        UNIQUE KEY(project_id, metadata_key)
        DISTRIBUTED BY HASH(project_id, metadata_key) BUCKETS 1
        {properties}
        """,
        f"""
        CREATE TABLE IF NOT EXISTS artifacts (
            project_id VARCHAR(255) NOT NULL,
            artifact_id BIGINT NOT NULL,
            name VARCHAR(512) NOT NULL,
            artifact_type VARCHAR(255) NOT NULL,
            description STRING NULL,
            created_at VARCHAR(64) NOT NULL
        )
        UNIQUE KEY(project_id, artifact_id)
        DISTRIBUTED BY HASH(project_id, artifact_id) BUCKETS 1
        {properties}
        """,
        f"""
        CREATE TABLE IF NOT EXISTS artifact_versions (
            project_id VARCHAR(255) NOT NULL,
            version_id BIGINT NOT NULL,
            artifact_id BIGINT NOT NULL,
            version_number BIGINT NOT NULL,
            manifest_digest VARCHAR(64) NOT NULL,
            manifest STRING NOT NULL,
            metadata STRING NULL,
            size_bytes BIGINT NOT NULL,
            producer_run_id VARCHAR(255) NULL,
            producer_run_name VARCHAR(255) NULL,
            created_at VARCHAR(64) NOT NULL
        )
        UNIQUE KEY(project_id, version_id)
        DISTRIBUTED BY HASH(project_id, version_id) BUCKETS 1
        {properties}
        """,
        f"""
        CREATE TABLE IF NOT EXISTS artifact_aliases (
            project_id VARCHAR(255) NOT NULL,
            artifact_id BIGINT NOT NULL,
            alias VARCHAR(255) NOT NULL,
            version_id BIGINT NOT NULL,
            version_number BIGINT NOT NULL,
            updated_at VARCHAR(64) NOT NULL
        )
        UNIQUE KEY(project_id, artifact_id, alias)
        DISTRIBUTED BY HASH(project_id, artifact_id) BUCKETS 1
        {properties}
        """,
        f"""
        CREATE TABLE IF NOT EXISTS run_artifact_links (
            project_id VARCHAR(255) NOT NULL,
            link_id BIGINT NOT NULL,
            run_id VARCHAR(255) NULL,
            run_name VARCHAR(255) NULL,
            version_id BIGINT NOT NULL,
            direction VARCHAR(16) NOT NULL,
            created_at VARCHAR(64) NOT NULL
        )
        UNIQUE KEY(project_id, link_id)
        DISTRIBUTED BY HASH(project_id, link_id) BUCKETS 1
        {properties}
        """,
    )
