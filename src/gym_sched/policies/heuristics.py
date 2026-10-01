"""Transparent dispatching and maximum-cardinality minimum-cost matching."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np
from scipy.optimize import linear_sum_assignment

from gym_sched.domain.models import Candidate, Decision, Observation, Recommendation
from gym_sched.policies.base import BasePolicy, candidate_key


def recommendations(selected: list[Candidate]) -> tuple[Recommendation, ...]:
    return tuple(Recommendation(*candidate_key(c)) for c in sorted(selected, key=candidate_key))


def ready_wait(observation: Observation, candidate: Candidate) -> float:
    """Do not give walking time an aging bonus; keep fixture compatibility."""
    return (
        candidate.ready_wait_seconds
        if candidate.ready_wait_seconds is not None
        else observation.time - candidate.ready_since
    )


def greedy(candidates: tuple[Candidate, ...], key: Callable) -> list[Candidate]:
    members: set[str] = set()
    equipment: set[str] = set()
    selected = []
    for candidate in sorted(candidates, key=lambda c: (*key(c), *candidate_key(c))):
        if candidate.member_id not in members and candidate.equipment_id not in equipment:
            selected.append(candidate)
            members.add(candidate.member_id)
            equipment.add(candidate.equipment_id)
    return selected


def match_candidates(
    candidates: tuple[Candidate, ...], cost: Callable[[Candidate], float]
) -> list[Candidate]:
    """Lexicographic cardinality then cost, including rectangular/incomplete graphs.

    One dummy per member guarantees feasibility. Real costs are affinely scaled to
    [0, 1]; a dummy cost > number of rows makes any extra real match preferable
    to all possible real-cost savings. Parallel visit edges retain the cheapest.
    """
    if not candidates:
        return []
    members = sorted({c.member_id for c in candidates})
    equipment = sorted({c.equipment_id for c in candidates})
    edges: dict[tuple[str, str], tuple[float, Candidate]] = {}
    for candidate in sorted(candidates, key=candidate_key):
        value = float(cost(candidate))
        if not np.isfinite(value):
            raise ValueError("matching edge cost must be finite")
        pair = candidate.member_id, candidate.equipment_id
        if pair not in edges or value < edges[pair][0]:
            edges[pair] = value, candidate
    values = [value for value, _ in edges.values()]
    low, high = min(values), max(values)
    span = high - low or 1.0
    dummy_cost = float(len(members) + 1)
    costs = np.full((len(members), len(equipment) + len(members)), dummy_cost)
    costs[:, : len(equipment)] = np.inf
    rows = {member: i for i, member in enumerate(members)}
    columns = {unit: i for i, unit in enumerate(equipment)}
    for (member, unit), (value, _) in edges.items():
        costs[rows[member], columns[unit]] = (value - low) / span
    assigned_rows, assigned_columns = linear_sum_assignment(costs)
    return [
        edges[members[int(i)], equipment[int(j)]][1]
        for i, j in zip(assigned_rows, assigned_columns, strict=True)
        if j < len(equipment)
    ]


class SelfDirectedPolicy(BasePolicy):
    self_directed = True

    def decide(self, observation: Observation, candidates: tuple[Candidate, ...]) -> Decision:
        return Decision(defer=True, diagnostics={"self_directed": True})


class FCFSPolicy(BasePolicy):
    def decide(self, observation: Observation, candidates: tuple[Candidate, ...]) -> Decision:
        chosen = greedy(candidates, lambda c: (c.ready_since,))
        return Decision(recommendations(chosen), defer=not chosen)


class SPTPolicy(BasePolicy):
    def decide(self, observation: Observation, candidates: tuple[Candidate, ...]) -> Decision:
        chosen = greedy(candidates, lambda c: (c.predicted_seconds, c.ready_since))
        return Decision(recommendations(chosen), defer=not chosen)


class AgingPolicy(BasePolicy):
    defaults: ClassVar[dict[str, Any]] = {"age_weight": 1.0}

    def __init__(self, name, options=None):
        super().__init__(name, options)
        if self.options["age_weight"] <= 0:
            raise ValueError("age_weight must be positive")

    def decide(self, observation: Observation, candidates: tuple[Candidate, ...]) -> Decision:
        chosen = greedy(
            candidates,
            lambda c: (
                c.predicted_seconds - self.options["age_weight"] * ready_wait(observation, c),
                c.ready_since,
            ),
        )
        return Decision(recommendations(chosen), defer=not chosen)


class MatchingPolicy(BasePolicy):
    defaults: ClassVar[dict[str, Any]] = {"age_weight": 1.0, "deviation_penalty_seconds": 60.0}

    def __init__(self, name, options=None):
        super().__init__(name, options)
        if any(float(v) < 0 for v in self.options.values() if isinstance(v, (int, float))):
            raise ValueError("matching costs must be nonnegative")

    def edge_cost(self, observation: Observation, candidate: Candidate) -> float:
        return (
            candidate.predicted_seconds
            - self.options["age_weight"] * ready_wait(observation, candidate)
            + self.options["deviation_penalty_seconds"] * candidate.deviation
        )

    def decide(self, observation: Observation, candidates: tuple[Candidate, ...]) -> Decision:
        chosen = match_candidates(candidates, lambda c: self.edge_cost(observation, c))
        return Decision(
            recommendations(chosen), defer=not chosen, diagnostics={"matched_count": len(chosen)}
        )


class RobustMatchingPolicy(MatchingPolicy):
    """Observable acceptance estimates and conservative changed-advice gating.

    This is an experimental surrogate, not a claim of expected-utility optimality.
    With all three factors disabled it is exactly the ordinary matching policy.
    The stability factor compares previous *unfulfilled* advice; pending or active
    members must not occur among the candidates supplied by the simulator.
    """

    defaults: ClassVar[dict[str, Any]] = {
        **MatchingPolicy.defaults,
        "acceptance": True,
        "uncertainty": True,
        "stability": True,
        # Optional planning penalty, never a physical delay. Zero unless an
        # explicitly justified refusal burden is supplied by an experiment.
        "refusal_cost_seconds": 0.0,
        "uncertainty_weight": 1.0,
        "observation_age_weight": 0.25,
        "change_penalty_seconds": 15.0,
        "minimum_change_benefit_seconds": 0.0,
        "unknown_busy_seconds": 300.0,
    }

    def risk(self, candidate: Candidate) -> float:
        if not self.options["uncertainty"]:
            return 0.0
        return (
            self.options["uncertainty_weight"] * candidate.uncertainty_seconds
            + self.options["observation_age_weight"] * candidate.observation_age_seconds
        )

    def probability(self, candidate: Candidate) -> float:
        return (
            max(0.0, min(1.0, candidate.acceptance_estimate)) if self.options["acceptance"] else 1.0
        )

    def edge_cost(self, observation: Observation, candidate: Candidate) -> float:
        return (
            super().edge_cost(observation, candidate)
            + (1.0 - self.probability(candidate)) * self.options["refusal_cost_seconds"]
            + self.risk(candidate)
            + (
                self.options["change_penalty_seconds"]
                if self.options["stability"] and candidate.is_change
                else 0.0
            )
        )

    def change_benefit(self, observation: Observation, candidate: Candidate) -> float | None:
        if not candidate.is_change or not any(
            self.options[k] for k in ("acceptance", "uncertainty", "stability")
        ):
            return None
        member = next((m for m in observation.members if m.id == candidate.member_id), None)
        if member is None or member.previous_recommendation is None:
            return None
        previous_visit, previous_unit = member.previous_recommendation
        visit = next((v for v in member.remaining_visits if v.id == previous_visit), None)
        equipment = next((e for e in observation.equipment if e.id == previous_unit), None)
        if visit is None or equipment is None or previous_unit not in visit.predicted_seconds:
            return None
        previous_ready = observation.time
        if not equipment.observed_available:
            previous_ready = max(
                observation.time,
                equipment.estimated_release_seconds
                if equipment.estimated_release_seconds is not None
                else observation.time + self.options["unknown_busy_seconds"],
            )
        old_finish = previous_ready + visit.predicted_seconds[previous_unit]
        new_finish = observation.time + candidate.predicted_seconds
        probability = self.probability(candidate)
        return (
            probability * (old_finish - new_finish)
            - (1.0 - probability) * self.options["refusal_cost_seconds"]
            - self.risk(candidate)
            - (self.options["change_penalty_seconds"] if self.options["stability"] else 0.0)
        )

    def decide(self, observation: Observation, candidates: tuple[Candidate, ...]) -> Decision:
        admissible = []
        gated = 0
        for candidate in candidates:
            benefit = self.change_benefit(observation, candidate)
            if benefit is not None and benefit <= self.options["minimum_change_benefit_seconds"]:
                gated += 1
            else:
                admissible.append(candidate)
        chosen = match_candidates(tuple(admissible), lambda c: self.edge_cost(observation, c))
        return Decision(
            recommendations(chosen),
            defer=not chosen,
            diagnostics={
                "matched_count": len(chosen),
                "gated_changes": gated,
                "acceptance_enabled": bool(self.options["acceptance"]),
                "uncertainty_enabled": bool(self.options["uncertainty"]),
                "stability_enabled": bool(self.options["stability"]),
            },
        )
