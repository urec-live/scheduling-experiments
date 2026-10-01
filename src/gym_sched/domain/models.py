"""Versioned input/result schemas and separate, immutable policy-visible types.

Policies must never receive Scenario, Member or Visit: these include latent truth.
All times and durations are seconds. Coordinates are synthetic meters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Equipment(StrictModel):
    id: str
    kind: str = "generic"
    x: float = 0
    y: float = 0
    outages: list[tuple[float, float]] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_outages(self):
        intervals = sorted(self.outages)
        if any(a < 0 or b <= a for a, b in intervals):
            raise ValueError("outage intervals must be positive and ordered")
        if any(b > c for (_, b), (c, _) in pairwise(intervals)):
            raise ValueError("overlapping outages are not supported")
        return self


class Visit(StrictModel):
    id: str
    exercise: str = "exercise"
    equipment_ids: list[str] = Field(min_length=1)
    actual_seconds: dict[str, float]
    predicted_seconds: dict[str, float]
    uncertainty_seconds: dict[str, float] = Field(default_factory=dict)
    preferred_equipment_ids: list[str] = Field(default_factory=list)
    deviation_by_equipment: dict[str, float] = Field(default_factory=dict)
    alternative_exercise_by_equipment: dict[str, str] = Field(default_factory=dict)
    rest_after_seconds: float = Field(default=60, ge=0)
    predecessors: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_durations(self):
        keys = set(self.equipment_ids)
        if len(keys) != len(self.equipment_ids):
            raise ValueError("duplicate compatible equipment")
        if set(self.actual_seconds) != keys or set(self.predicted_seconds) != keys:
            raise ValueError("each compatible unit needs actual and predicted duration")
        if any(v <= 0 for v in [*self.actual_seconds.values(), *self.predicted_seconds.values()]):
            raise ValueError("visit durations must be positive")
        if not set(self.preferred_equipment_ids) <= keys:
            raise ValueError("preferred units must be compatible")
        for mapping in (
            self.uncertainty_seconds,
            self.deviation_by_equipment,
            self.alternative_exercise_by_equipment,
        ):
            if not set(mapping) <= keys:
                raise ValueError("alternative metadata must reference compatible equipment")
        if any(
            v < 0
            for v in [*self.uncertainty_seconds.values(), *self.deviation_by_equipment.values()]
        ):
            raise ValueError("uncertainty and deviation cannot be negative")
        return self


class Member(StrictModel):
    id: str
    arrival_seconds: float = Field(ge=0)
    visits: list[Visit] = Field(min_length=1)
    participates: bool = True
    deadline_seconds: float | None = Field(default=None, ge=0)
    # Latent behavior: never copied to policy observations.
    independent_keep_plan_order: bool = False
    acceptance_probability: float = Field(default=1, ge=0, le=1)
    nonpreferred_acceptance_multiplier: float = Field(default=0.7, ge=0, le=1)
    ignore_probability: float = Field(default=0, ge=0, le=1)
    patience_wait_seconds: float | None = Field(default=None, gt=0)
    preference_change_seconds: float | None = Field(default=None, ge=0)
    changed_acceptance_probability: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def valid_member(self):
        if self.deadline_seconds is not None and self.deadline_seconds < self.arrival_seconds:
            raise ValueError("deadline precedes arrival")
        seen = set()
        for visit in self.visits:
            if visit.id in seen or not set(visit.predecessors) <= seen:
                raise ValueError("visit IDs must be unique and predecessors earlier in the list")
            seen.add(visit.id)
        return self


class Scenario(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    id: str
    seed: int = 42
    arrival_window_seconds: float = Field(default=7200, gt=0)
    equipment: list[Equipment] = Field(min_length=1)
    members: list[Member] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_references(self):
        units = {e.id for e in self.equipment}
        if len(units) != len(self.equipment):
            raise ValueError("equipment IDs must be unique")
        if len({m.id for m in self.members}) != len(self.members):
            raise ValueError("member IDs must be unique")
        for m in self.members:
            if m.arrival_seconds >= self.arrival_window_seconds:
                raise ValueError("arrival must be within the arrival window")
            for v in m.visits:
                if not set(v.equipment_ids) <= units:
                    raise ValueError("unknown compatible equipment")
        return self


class SimulationConfig(StrictModel):
    response_delay_seconds: float = Field(default=0, ge=0)
    walking_seconds: float = Field(default=0, ge=0)
    observation_delay_seconds: float = Field(default=0, ge=0)
    observation_dropout: float = Field(default=0, ge=0, le=1)
    observation_interval_seconds: float = Field(default=30, gt=0)
    recommendation_expiry_seconds: float = Field(default=30, gt=0)
    decision_delay_seconds: float = Field(default=0, ge=0)
    decision_delay_trace: list[float] = Field(default_factory=list)
    allow_reordering: bool = False
    allow_substitutions: bool = False
    max_time_seconds: float = Field(default=86400, gt=0)
    max_events: int = Field(default=200000, gt=0)
    max_same_time_rounds: int = Field(default=100, gt=0)

    @model_validator(mode="after")
    def valid_delays(self):
        if any(x < 0 for x in self.decision_delay_trace):
            raise ValueError("decision delay trace must be nonnegative")
        return self


@dataclass(frozen=True)
class VisitView:
    id: str
    exercise: str
    equipment_ids: tuple[str, ...]
    predicted_seconds: dict[str, float]
    uncertainty_seconds: dict[str, float]
    preferred_equipment_ids: tuple[str, ...]
    deviation_by_equipment: dict[str, float]
    rest_after_seconds: float
    predecessors: tuple[str, ...]


@dataclass(frozen=True)
class MemberView:
    id: str
    state: str
    ready_since: float | None
    deadline_seconds: float | None
    remaining_visits: tuple[VisitView, ...]
    completed_visit_ids: tuple[str, ...]
    participates: bool
    available_at: float
    active_equipment_id: str | None = None
    previous_recommendation: tuple[str, str] | None = None
    acceptance_counts: tuple[int, int] = (0, 0)
    active_visit_id: str | None = None


@dataclass(frozen=True)
class EquipmentView:
    id: str
    kind: str
    observed_available: bool
    observed_at: float
    estimated_release_seconds: float | None


@dataclass(frozen=True)
class Observation:
    time: float
    equipment: tuple[EquipmentView, ...]
    members: tuple[MemberView, ...]
    allow_reordering: bool = False
    allow_substitutions: bool = False
    schema_version: str = "1.0"


@dataclass(frozen=True)
class Candidate:
    member_id: str
    visit_id: str
    equipment_id: str
    ready_since: float
    predicted_seconds: float
    uncertainty_seconds: float = 0
    observation_age_seconds: float = 0
    preferred: bool = True
    deviation: float = 0
    is_change: bool = False
    acceptance_estimate: float = 0.5
    ready_wait_seconds: float | None = None


@dataclass(frozen=True)
class Recommendation:
    member_id: str
    visit_id: str
    equipment_id: str


@dataclass(frozen=True)
class Decision:
    recommendations: tuple[Recommendation, ...] = ()
    defer: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyContext:
    seed: int = 0
    # Never include a scenario, RNG for exogenous events, or simulator reference.


class RunResult(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    scenario_id: str
    scenario_hash: str
    policy: str
    seed: int
    status: Literal["completed", "stalled", "truncated"]
    end_time: float
    config: dict[str, Any]
    events: list[dict[str, Any]]
    member_summaries: list[dict[str, Any]]
    equipment_summaries: list[dict[str, Any]]
    decisions: list[dict[str, Any]]
    metrics: dict[str, Any]
