"""Generic, payload-free facts attached to a retained Trackio trace.

This module deliberately knows nothing about Verifiers, a model family, or a
training algorithm.  Producers calculate their facts; Trackio validates their
shape, persists the current projection, and offers a restricted aggregation
surface over declared columns.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Literal

FactState = Literal["complete", "partial", "unsupported"]
SignalSourceKind = Literal[
    "llm_judge",
    "deterministic",
    "human",
    "environment",
    "group",
    "teacher",
    "composite",
    "unknown",
]
AggregateOperation = Literal["mean", "sum", "count", "min", "max"]

_SOURCE_KINDS = frozenset(SignalSourceKind.__args__)
_DIMENSION_NAMES = frozenset(
    {
        "model",
        "model_family",
        "tokenizer_revision",
        "renderer_revision",
        "template_revision",
        "trace_schema_version",
        "task_type",
        "rollout_step",
        "is_truncated",
        "has_error",
        "reward_component_name",
        "reward_component_source_kind",
    }
)
_MEASURE_NAMES = frozenset(
    {
        "model_input_tokens",
        "model_output_tokens",
        "thinking_tokens",
        "tool_calls",
        "model_calls",
        "trace_latency_ms",
        "task_reward",
        "algorithm_reward",
        "reward_component_contribution",
        "reward_component_score",
        "reward_component_weight",
    }
)

_COMPONENT_MEASURE_FIELDS = {
    "reward_component_contribution": "contribution",
    "reward_component_score": "score",
    "reward_component_weight": "weight",
}
_COMPONENT_DIMENSION_NAMES = frozenset(
    {"reward_component_name", "reward_component_source_kind"}
)


def _text(value: object, name: str, *, maximum: int = 768) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{name} must be non-empty bounded text")
    return value


def _number(value: object, name: str) -> int | float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number or null")
    return value


@dataclass(frozen=True, slots=True)
class TraceRewardComponent:
    name: str
    contribution: int | float | None
    score: int | float | None = None
    weight: int | float | None = None
    source_kind: SignalSourceKind = "unknown"
    source_id: str | None = None

    def __post_init__(self) -> None:
        _text(self.name, "reward component name", maximum=256)
        contribution = _number(self.contribution, "reward component contribution")
        score = _number(self.score, "reward component score")
        weight = _number(self.weight, "reward component weight")
        if self.source_kind not in _SOURCE_KINDS:
            raise ValueError(
                f"unknown reward component source kind {self.source_kind!r}"
            )
        if self.source_id is not None:
            _text(self.source_id, "reward component source id", maximum=256)
        if (
            contribution is not None
            and score is not None
            and weight is not None
            and not math.isclose(
                float(contribution),
                float(score) * float(weight),
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("reward component contribution must equal score * weight")


@dataclass(frozen=True, slots=True)
class TraceFactUpdate:
    """One idempotent projection for an existing trace identity."""

    trace_type: str
    external_id: str
    namespace: str
    calculator_version: str
    projection_id: str
    dimensions: Mapping[str, str | int | float | bool | None] = field(
        default_factory=dict
    )
    measures: Mapping[str, int | float | None] = field(default_factory=dict)
    reward_components: tuple[TraceRewardComponent, ...] = ()
    provenance: Mapping[str, str] = field(default_factory=dict)
    state: FactState = "complete"
    replace_reward_components: bool = False
    calculated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        _text(self.trace_type, "trace type", maximum=64)
        _text(self.external_id, "external id")
        _text(self.namespace, "fact namespace", maximum=128)
        _text(self.calculator_version, "calculator version", maximum=256)
        if len(self.projection_id) != 64 or any(
            char not in "0123456789abcdef" for char in self.projection_id
        ):
            raise ValueError("projection id must be a lowercase SHA-256 digest")
        if self.state not in {"complete", "partial", "unsupported"}:
            raise ValueError("trace fact state is invalid")
        if self.calculated_at.tzinfo is None:
            raise ValueError("calculated_at must be timezone-aware")
        for name, value in self.dimensions.items():
            if name not in _DIMENSION_NAMES:
                raise ValueError(f"unsupported trace fact dimension {name!r}")
            if value is not None and isinstance(value, (list, tuple, Mapping)):
                raise ValueError(f"trace fact dimension {name!r} must be scalar")
        for name, value in self.measures.items():
            if name not in _MEASURE_NAMES:
                raise ValueError(f"unsupported trace fact measure {name!r}")
            _number(value, f"trace fact measure {name!r}")
        names = [component.name for component in self.reward_components]
        if len(names) != len(set(names)):
            raise ValueError("reward component names must be unique")
        if any(
            not isinstance(component, TraceRewardComponent)
            for component in self.reward_components
        ):
            raise ValueError("reward components must be TraceRewardComponent values")
        if self.replace_reward_components:
            if self.namespace != "verifiers.trace":
                raise ValueError(
                    "only a Verifiers source projection may replace reward components"
                )
        elif (
            self.reward_components
            or self.dimensions
            or set(self.measures) != {"algorithm_reward"}
        ):
            raise ValueError("a trace-fact enrichment may only supply algorithm_reward")
        if self.projection_id != projection_id(
            {
                "namespace": self.namespace,
                "calculator_version": self.calculator_version,
                "dimensions": dict(sorted(self.dimensions.items())),
                "measures": dict(sorted(self.measures.items())),
                "reward_components": [
                    {
                        "name": component.name,
                        "contribution": component.contribution,
                        "score": component.score,
                        "weight": component.weight,
                        "source": {
                            "kind": component.source_kind,
                            "id": component.source_id,
                        },
                    }
                    for component in sorted(
                        self.reward_components, key=lambda item: item.name
                    )
                ],
                "provenance": dict(sorted(self.provenance.items())),
                "state": self.state,
            }
        ):
            raise ValueError(
                "projection id does not match the immutable trace fact payload"
            )
        object.__setattr__(self, "dimensions", MappingProxyType(dict(self.dimensions)))
        object.__setattr__(self, "measures", MappingProxyType(dict(self.measures)))
        object.__setattr__(
            self,
            "reward_components",
            tuple(sorted(self.reward_components, key=lambda item: item.name)),
        )
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> TraceFactUpdate:
        components = tuple(
            TraceRewardComponent(**dict(item))
            for item in payload.get("reward_components", ())
            if isinstance(item, Mapping)
        )
        calculated_at = payload.get("calculated_at")
        return cls(
            trace_type=payload.get("trace_type"),  # type: ignore[arg-type]
            external_id=payload.get("external_id"),  # type: ignore[arg-type]
            namespace=payload.get("namespace"),  # type: ignore[arg-type]
            calculator_version=payload.get("calculator_version"),  # type: ignore[arg-type]
            projection_id=payload.get("projection_id"),  # type: ignore[arg-type]
            dimensions=payload.get("dimensions", {}),  # type: ignore[arg-type]
            measures=payload.get("measures", {}),  # type: ignore[arg-type]
            reward_components=components,
            provenance=payload.get("provenance", {}),  # type: ignore[arg-type]
            state=payload.get("state", "complete"),  # type: ignore[arg-type]
            replace_reward_components=payload.get("replace_reward_components", False),  # type: ignore[arg-type]
            calculated_at=(
                datetime.fromisoformat(calculated_at)
                if isinstance(calculated_at, str)
                else datetime.now(UTC)
            ),
        )

    def payload(self) -> dict[str, object]:
        return {
            "trace_type": self.trace_type,
            "external_id": self.external_id,
            "namespace": self.namespace,
            "calculator_version": self.calculator_version,
            "projection_id": self.projection_id,
            "dimensions": dict(self.dimensions),
            "measures": dict(self.measures),
            "reward_components": [
                {
                    "name": component.name,
                    "contribution": component.contribution,
                    "score": component.score,
                    "weight": component.weight,
                    "source_kind": component.source_kind,
                    "source_id": component.source_id,
                }
                for component in self.reward_components
            ],
            "provenance": dict(self.provenance),
            "state": self.state,
            "replace_reward_components": self.replace_reward_components,
            "calculated_at": self.calculated_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class TraceFactWriteReceipt:
    trace_id: str
    projection_id: str
    applied: bool


@dataclass(frozen=True, slots=True)
class TraceAggregate:
    measure: str
    operation: AggregateOperation = "mean"
    component_name: str | None = None

    def __post_init__(self) -> None:
        if self.measure not in _MEASURE_NAMES:
            raise ValueError(f"unsupported trace aggregate measure {self.measure!r}")
        if self.operation not in {"mean", "sum", "count", "min", "max"}:
            raise ValueError(
                f"unsupported trace aggregate operation {self.operation!r}"
            )
        if self.component_name is not None:
            _text(self.component_name, "trace aggregate component name", maximum=256)
            if self.measure not in _COMPONENT_MEASURE_FIELDS:
                raise ValueError(
                    "component_name is only valid for reward-component aggregates"
                )

    @property
    def key(self) -> str:
        """The stable output key for this aggregate request."""

        return f"{self.operation}_{self.measure}"

    @property
    def component_field(self) -> str | None:
        return _COMPONENT_MEASURE_FIELDS.get(self.measure)


@dataclass(frozen=True, slots=True)
class TraceFactsQuery:
    trace_type: str = "verifiers"
    group_by: tuple[str, ...] = ()
    aggregates: tuple[TraceAggregate, ...] = ()
    dimensions: Mapping[str, str | int | float | bool | None] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if any(name not in _DIMENSION_NAMES for name in self.group_by):
            raise ValueError("trace fact grouping includes an unsupported dimension")
        if len(self.group_by) != len(set(self.group_by)):
            raise ValueError("trace fact grouping dimensions must be unique")
        if any(name not in _DIMENSION_NAMES for name in self.dimensions):
            raise ValueError("trace fact filter includes an unsupported dimension")
        if not self.aggregates:
            raise ValueError("at least one trace aggregate is required")
        aggregate_keys = [aggregate.key for aggregate in self.aggregates]
        if len(aggregate_keys) != len(set(aggregate_keys)):
            raise ValueError("trace fact aggregate output keys must be unique")
        component_dimensions = set(self.group_by) | set(self.dimensions)
        has_component_dimensions = bool(
            component_dimensions & _COMPONENT_DIMENSION_NAMES
        )
        has_component_aggregates = any(
            aggregate.component_field is not None for aggregate in self.aggregates
        )
        if has_component_dimensions and not has_component_aggregates:
            raise ValueError(
                "reward-component dimensions require reward-component aggregates"
            )
        if has_component_aggregates and any(
            aggregate.component_field is None for aggregate in self.aggregates
        ):
            raise ValueError(
                "scalar and reward-component aggregates must be queried separately"
            )
        object.__setattr__(self, "dimensions", MappingProxyType(dict(self.dimensions)))


@dataclass(frozen=True, slots=True)
class TraceAggregateBucket:
    dimensions: Mapping[str, str | int | float | bool | None]
    trace_count: int
    values: Mapping[str, float | int | None]
    coverage: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class TraceAggregateResult:
    buckets: tuple[TraceAggregateBucket, ...]


def projection_id(payload: Mapping[str, object]) -> str:
    """Return the contract's deterministic projection digest for a raw payload."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
