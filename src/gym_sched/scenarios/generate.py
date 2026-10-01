"""Synthetic scenarios. Every number here is a modeling assumption, not gym evidence."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from gym_sched.domain import Equipment, Member, Scenario, StrictModel, Visit

from .randomness import content_hash, keyed_rng, lognormal_factor, uniform


class GeneratorConfig(StrictModel):
    id: str = "demo"
    equipment_types: int = Field(default=4, ge=1)
    units_per_type: int = Field(default=2, ge=1)
    member_count: int = Field(default=20, ge=1)
    visits_per_member: int = Field(default=4, ge=1)
    arrival_window_seconds: float = Field(default=7200, gt=0)
    arrival_pattern: Literal["even", "poisson", "burst", "poisson_batches"] = "even"
    arrival_batch_size: int = Field(default=4, ge=1)
    member_pace_factors: list[float] = Field(default_factory=lambda: [1.25, 0.75, 1.0])
    duration_seconds: list[float] = Field(default_factory=lambda: [240, 360, 480, 600])
    pace_cv: float = Field(default=0, ge=0)
    duration_cv: float = Field(default=0, ge=0)
    prediction_bias: float = Field(default=0, gt=-1)
    prediction_noise_cv: float = Field(default=0, ge=0)
    adoption_rate: float = Field(default=1, ge=0, le=1)
    acceptance_probability: float = Field(default=1, ge=0, le=1)
    acceptance_model: Literal["per_recommendation", "member_cohort"] = "per_recommendation"
    acceptor_fraction: float | None = Field(default=None, ge=0, le=1)
    ignore_probability: float = Field(default=0, ge=0, le=1)
    deadline_after_seconds: float | None = Field(default=None, gt=0)
    patience_wait_seconds: float | None = Field(default=None, gt=0)
    preference_change_seconds: float | None = Field(default=None, ge=0)
    changed_acceptance_probability: float | None = Field(default=None, ge=0, le=1)
    rest_seconds: float = Field(default=60, ge=0)
    compatibility_units: int | None = Field(default=None, ge=1)
    allow_substitutions: bool = False
    flexible_precedence: bool = False
    outages: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_parameters(self):
        if not self.duration_seconds or any(x <= 0 for x in self.duration_seconds):
            raise ValueError("duration_seconds must contain positive durations")
        if not self.member_pace_factors or any(x <= 0 for x in self.member_pace_factors):
            raise ValueError("member_pace_factors must contain positive factors")
        if self.compatibility_units and self.compatibility_units > self.units_per_type:
            raise ValueError("compatibility_units exceeds units_per_type")
        if self.acceptance_model == "member_cohort":
            if self.acceptor_fraction is None:
                raise ValueError("member_cohort requires acceptor_fraction")
            if (
                self.acceptance_probability != 1
                or self.ignore_probability != 0
                or self.preference_change_seconds is not None
                or self.changed_acceptance_probability is not None
            ):
                raise ValueError(
                    "member_cohort cannot combine probabilistic/ignored/changing responses"
                )
        elif self.acceptor_fraction is not None:
            raise ValueError("acceptor_fraction requires member_cohort")
        return self


def generate_scenario(config: dict | GeneratorConfig, seed: int = 42) -> Scenario:
    raw = config.get("scenario", config) if isinstance(config, dict) else config
    c = raw if isinstance(raw, GeneratorConfig) else GeneratorConfig.model_validate(raw)
    equipment = [
        Equipment(id=f"t{t}-u{u}", kind=f"type-{t}", x=t * 5, y=u * 3)
        for t in range(c.equipment_types)
        for u in range(c.units_per_type)
    ]
    index = {e.id: e for e in equipment}
    for outage in c.outages:
        if set(outage) != {"equipment_id", "start", "end"} or outage["equipment_id"] not in index:
            raise ValueError("outage needs equipment_id, start, end referencing a known unit")
        index[outage["equipment_id"]].outages.append((outage["start"], outage["end"]))
    # Conditional on N arrivals, a homogeneous Poisson process has uniform ordered times.
    # This generator controls N; it does not claim to sample an unconditioned Poisson count.
    if c.arrival_pattern == "poisson":
        arrivals = sorted(
            keyed_rng(seed, "arrivals")
            .uniform(0, c.arrival_window_seconds, c.member_count)
            .tolist()
        )
    elif c.arrival_pattern == "poisson_batches":
        # Conditional on the batch count, sample batch arrival times. Members
        # within a batch arrive together; the final batch may be smaller.
        groups = (c.member_count + c.arrival_batch_size - 1) // c.arrival_batch_size
        batch_times = sorted(
            keyed_rng(seed, "batch_arrivals").uniform(0, c.arrival_window_seconds, groups)
        )
        arrivals = [float(batch_times[i // c.arrival_batch_size]) for i in range(c.member_count)]
    elif c.arrival_pattern == "burst":
        arrivals = sorted(
            min(
                c.arrival_window_seconds - 1e-6,
                (i % 3) * c.arrival_window_seconds / 3
                + uniform(seed, "arrivals", i) * min(120, c.arrival_window_seconds / 6),
            )
            for i in range(c.member_count)
        )
    else:
        # Four-person batches compete for two default units; FCFS and SPT
        # therefore have an explainable deterministic difference in the demo.
        groups = (c.member_count + c.arrival_batch_size - 1) // c.arrival_batch_size
        arrivals = [
            (i // c.arrival_batch_size) * c.arrival_window_seconds / groups
            for i in range(c.member_count)
        ]
    members = []
    for i, arrival in enumerate(arrivals):
        mid = f"m{i:03d}"
        deterministic_pace = c.member_pace_factors[i % len(c.member_pace_factors)]
        pace = deterministic_pace * lognormal_factor(seed, "pace", c.pace_cv, mid)
        visits = []
        for j in range(c.visits_per_member):
            vid = f"v{j:02d}"
            t = j % c.equipment_types
            eligible = [f"t{t}-u{u}" for u in range(c.compatibility_units or c.units_per_type)]
            preferred = list(eligible)
            alternates = {}
            if c.allow_substitutions and c.equipment_types > 1:
                alt_type = (t + 1) % c.equipment_types
                for u in range(c.units_per_type):
                    eid = f"t{alt_type}-u{u}"
                    eligible.append(eid)
                    alternates[eid] = f"approved-alternative-{t}-on-{alt_type}"
            expected = c.duration_seconds[t % len(c.duration_seconds)] * deterministic_pace
            actual, predicted, uncertainty = {}, {}, {}
            for eid in eligible:
                multiplier = 1.15 if eid in alternates else 1.0
                actual[eid] = round(
                    max(
                        1.0,
                        c.duration_seconds[t % len(c.duration_seconds)]
                        * pace
                        * multiplier
                        * lognormal_factor(seed, "durations", c.duration_cv, mid, vid, eid),
                    ),
                    6,
                )
                predicted[eid] = round(
                    max(
                        1.0,
                        expected
                        * multiplier
                        * (1 + c.prediction_bias)
                        * lognormal_factor(seed, "forecasts", c.prediction_noise_cv, mid, vid, eid),
                    ),
                    6,
                )
                uncertainty[eid] = expected * multiplier * (c.duration_cv**2 + c.pace_cv**2) ** 0.5
            visits.append(
                Visit(
                    id=vid,
                    exercise=f"exercise-{t}",
                    equipment_ids=eligible,
                    actual_seconds=actual,
                    predicted_seconds=predicted,
                    uncertainty_seconds=uncertainty,
                    preferred_equipment_ids=preferred,
                    deviation_by_equipment={eid: 1 for eid in alternates},
                    alternative_exercise_by_equipment=alternates,
                    rest_after_seconds=c.rest_seconds if j < c.visits_per_member - 1 else 0,
                    predecessors=[] if c.flexible_precedence or j == 0 else [f"v{j - 1:02d}"],
                )
            )
        members.append(
            Member(
                id=mid,
                arrival_seconds=arrival,
                visits=visits,
                participates=uniform(seed, "adoption", mid) < c.adoption_rate,
                acceptance_probability=c.acceptance_probability,
                ignore_probability=c.ignore_probability,
                deadline_seconds=None
                if c.deadline_after_seconds is None
                else arrival + c.deadline_after_seconds,
                patience_wait_seconds=c.patience_wait_seconds,
                preference_change_seconds=c.preference_change_seconds,
                changed_acceptance_probability=c.changed_acceptance_probability,
            )
        )
    if c.acceptance_model == "member_cohort":
        # Exact count among advice-eligible members, rounded half up. The order
        # is stable across fractions and independent of workloads/policy draws.
        participants = sorted(
            (m for m in members if m.participates),
            key=lambda m: (uniform(seed, "acceptance_cohort", m.id), m.id),
        )
        acceptors = int(len(participants) * c.acceptor_fraction + 0.5)
        for i, member in enumerate(participants):
            member.acceptance_probability = float(i < acceptors)
            member.nonpreferred_acceptance_multiplier = 1.0
    offered = sum(sum(min(v.predicted_seconds.values()) for v in m.visits) for m in members)
    return Scenario(
        id=f"{c.id}-{seed}",
        seed=seed,
        arrival_window_seconds=c.arrival_window_seconds,
        equipment=equipment,
        members=members,
        metadata={
            **c.metadata,
            "synthetic": True,
            "generator": c.model_dump(mode="json"),
            "generator_config_hash": content_hash(c),
            "nominal_offered_load": offered / (len(equipment) * c.arrival_window_seconds),
            "arrival_count_conditioned": True,
        },
    )
