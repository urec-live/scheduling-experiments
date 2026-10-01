"""Shared observable workload and feasible serial decoder for search policies.

Forecasts assume instantaneous compliance and use predicted durations. They do
not reserve equipment, predict future arrivals, control nonparticipants, or drop
late work to improve the objective. The objective is remaining ready waiting:
time between successive blocks excluding their processing and prescribed rest.
"""

from __future__ import annotations

from dataclasses import dataclass

from gym_sched.domain.models import Candidate, Observation, Recommendation

OperationKey = tuple[str, str]


@dataclass(frozen=True)
class ForecastOperation:
    member_id: str
    visit_id: str
    equipment_ids: tuple[str, ...]
    durations: tuple[float, ...]
    deviations: tuple[float, ...]
    rest_after: float
    predecessors: tuple[OperationKey, ...]

    @property
    def key(self) -> OperationKey:
        return self.member_id, self.visit_id


@dataclass(frozen=True)
class ForecastProblem:
    now: float
    operations: tuple[ForecastOperation, ...]
    member_available: dict[str, float]
    equipment_available: dict[str, float]
    deadlines: dict[str, float]


@dataclass(frozen=True)
class PlannedOperation:
    member_id: str
    visit_id: str
    equipment_id: str
    start: float
    finish: float
    ready: float
    rest_after: float
    deviation: float = 0.0

    @property
    def key(self) -> OperationKey:
        return self.member_id, self.visit_id


@dataclass(frozen=True)
class ForecastSchedule:
    operations: tuple[PlannedOperation, ...]
    waiting_seconds: float
    member_waiting: dict[str, float]
    tardiness_seconds: float
    deviation: float


def build_problem(observation: Observation, unknown_busy_seconds: float = 300.0) -> ForecastProblem:
    equipment_available = {
        equipment.id: observation.time
        if equipment.observed_available
        else max(
            observation.time,
            equipment.estimated_release_seconds
            if equipment.estimated_release_seconds is not None
            else observation.time + unknown_busy_seconds,
        )
        for equipment in observation.equipment
    }
    operations = []
    member_available = {}
    deadlines = {}
    for member in sorted(observation.members, key=lambda m: m.id):
        if not member.participates or not member.remaining_visits:
            continue
        member_available[member.id] = max(observation.time, member.available_at)
        if member.deadline_seconds is not None:
            deadlines[member.id] = member.deadline_seconds
        remaining_ids = {visit.id for visit in member.remaining_visits}
        previous = None
        for visit in member.remaining_visits:
            units = tuple(
                sorted(
                    unit
                    for unit in visit.equipment_ids
                    if unit in equipment_available
                    and (
                        observation.allow_substitutions
                        or visit.deviation_by_equipment.get(unit, 0.0) <= 0
                    )
                )
            )
            if not units:
                raise ValueError(f"No observable permitted equipment for {member.id}/{visit.id}")
            predecessors = {
                (member.id, pred) for pred in visit.predecessors if pred in remaining_ids
            }
            if previous is not None and not observation.allow_reordering:
                predecessors.add((member.id, previous))
            operations.append(
                ForecastOperation(
                    member.id,
                    visit.id,
                    units,
                    tuple(float(visit.predicted_seconds[unit]) for unit in units),
                    tuple(float(visit.deviation_by_equipment.get(unit, 0.0)) for unit in units),
                    visit.rest_after_seconds,
                    tuple(sorted(predecessors)),
                )
            )
            previous = visit.id
    return ForecastProblem(
        observation.time, tuple(operations), member_available, equipment_available, deadlines
    )


def summarize_schedule(
    problem: ForecastProblem, operations: list[PlannedOperation]
) -> ForecastSchedule:
    waiting = dict.fromkeys(problem.member_available, 0.0)
    finish = dict(problem.member_available)
    for operation in operations:
        waiting[operation.member_id] += max(0.0, operation.start - operation.ready)
        finish[operation.member_id] = max(finish[operation.member_id], operation.finish)
    return ForecastSchedule(
        tuple(sorted(operations, key=lambda op: (op.start, op.member_id, op.visit_id))),
        float(sum(waiting.values())),
        waiting,
        float(sum(max(0.0, finish[m] - deadline) for m, deadline in problem.deadlines.items())),
        float(sum(op.deviation for op in operations)),
    )


def decode(problem: ForecastProblem, genes: tuple[tuple[float, float], ...]) -> ForecastSchedule:
    """Feasible serial schedule: precedence-aware priorities + machine-choice keys.

    This decoder appends to equipment timelines; it does not claim to enumerate
    every feasible active schedule. NSGA-II and GA deliberately share it.
    """
    if len(genes) != len(problem.operations):
        raise ValueError("A priority and equipment key are required for every operation")
    remaining = dict(enumerate(problem.operations))
    completed: set[OperationKey] = set()
    member_available = dict(problem.member_available)
    equipment_available = dict(problem.equipment_available)
    planned = []
    while remaining:
        eligible = [index for index, op in remaining.items() if set(op.predecessors) <= completed]
        if not eligible:
            raise ValueError("Forecast contains cyclic or unresolved precedence")
        index = min(eligible, key=lambda i: (genes[i][0], remaining[i].key))
        operation = remaining.pop(index)
        choice = min(
            len(operation.equipment_ids) - 1,
            max(0, int(genes[index][1] * len(operation.equipment_ids))),
        )
        equipment = operation.equipment_ids[choice]
        ready = member_available[operation.member_id]
        start = max(ready, equipment_available[equipment])
        finish = start + operation.durations[choice]
        planned.append(
            PlannedOperation(
                operation.member_id,
                operation.visit_id,
                equipment,
                start,
                finish,
                ready,
                operation.rest_after,
                operation.deviations[choice],
            )
        )
        member_available[operation.member_id] = finish + operation.rest_after
        equipment_available[equipment] = finish
        completed.add(operation.key)
    return summarize_schedule(problem, planned)


def initial_genes(problem: ForecastProblem) -> tuple[tuple[float, float], ...]:
    """Deterministic shortest-predicted-block seed; GA adds random and warm seeds."""
    minimum = [min(op.durations) for op in problem.operations]
    scale = max(minimum, default=1.0)
    return tuple(
        (
            minimum[i] / scale,
            (
                min(range(len(op.durations)), key=lambda j: (op.durations[j], op.equipment_ids[j]))
                + 0.5
            )
            / len(op.durations),
        )
        for i, op in enumerate(problem.operations)
    )


def immediate_recommendations(
    problem: ForecastProblem,
    schedule: ForecastSchedule,
    candidates: tuple[Candidate, ...],
    tolerance: float = 1e-6,
) -> tuple[Recommendation, ...]:
    candidate_keys = {(c.member_id, c.visit_id, c.equipment_id) for c in candidates}
    return tuple(
        Recommendation(op.member_id, op.visit_id, op.equipment_id)
        for op in schedule.operations
        if op.start <= problem.now + tolerance
        and (op.member_id, op.visit_id, op.equipment_id) in candidate_keys
    )
