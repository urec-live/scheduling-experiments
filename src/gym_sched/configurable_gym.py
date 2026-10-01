"""UI-configurable synthetic workloads using the existing simulation engine."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gym_sched.domain.models import Equipment, Member, Scenario, SimulationConfig, Visit
from gym_sched.evaluation.metrics import validate_result
from gym_sched.policies import create_policy
from gym_sched.scenarios.randomness import uniform
from gym_sched.simulation.engine import run_episode


class GymSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    leg_press_count: int = Field(default=2, ge=0, le=10)
    cable_row_count: int = Field(default=2, ge=0, le=10)
    member_count: int = Field(default=8, ge=1, le=50)
    independent_count: int = Field(default=2, ge=0, le=50)
    arrival_pattern: str = "interval"
    arrival_seconds: float = Field(default=30, ge=0, le=3600)
    seed: int = Field(default=42, ge=0, le=2147483647)
    exercise_seconds: float = Field(default=120, gt=0, le=3600)
    recovery_seconds: float = Field(default=60, ge=0, le=3600)
    refusal_percent: float = Field(default=20, ge=0, le=100)

    @model_validator(mode="after")
    def valid_settings(self):
        if self.leg_press_count + self.cable_row_count == 0:
            raise ValueError("Provide at least one machine")
        if self.independent_count > self.member_count:
            raise ValueError("Independent members cannot exceed total members")
        if self.arrival_pattern not in {"simultaneous", "interval", "uniform"}:
            raise ValueError("Unknown arrival pattern")
        return self


def generate_gym(settings: GymSettings) -> dict:
    units = {
        "Leg press": [f"A{i + 1}" for i in range(settings.leg_press_count)],
        "Cable row": [f"B{i + 1}" for i in range(settings.cable_row_count)],
    }
    equipment = [Equipment(id=uid, kind=kind) for kind, ids in units.items() for uid in ids]
    members = []
    for index in range(settings.member_count):
        mid = f"member-{index + 1:02d}"
        arrival = 0
        if settings.arrival_pattern == "interval":
            arrival = index * settings.arrival_seconds
        elif settings.arrival_pattern == "uniform":
            arrival = round(uniform(settings.seed, "ui-arrival", mid) * settings.arrival_seconds, 3)
        kinds = list(units) if index % 2 == 0 else list(reversed(units))
        visits = [
            Visit(
                id=f"v{number + 1}",
                exercise=kind,
                equipment_ids=units[kind],
                actual_seconds=dict.fromkeys(units[kind], settings.exercise_seconds),
                predicted_seconds=dict.fromkeys(units[kind], settings.exercise_seconds),
                rest_after_seconds=settings.recovery_seconds,
            )
            for number, kind in enumerate(kinds)
            if units[kind]
        ]
        members.append(
            Member(
                id=mid,
                arrival_seconds=arrival,
                visits=visits,
                participates=index >= settings.independent_count,
                acceptance_probability=1 - settings.refusal_percent / 100,
            )
        )
    scenario = Scenario(
        id="configurable-gym",
        seed=settings.seed,
        arrival_window_seconds=max(m.arrival_seconds for m in members) + 1,
        equipment=equipment,
        members=members,
        metadata={"synthetic": True, "generator": settings.model_dump()},
    )
    return {
        "scenario": scenario.model_dump(mode="json"),
        "config": SimulationConfig(allow_reordering=True).model_dump(mode="json"),
    }


def simulate_gym(scenario_data: dict, config_data: dict) -> dict:
    scenario = Scenario.model_validate(scenario_data)
    config = SimulationConfig.model_validate(config_data)
    if len(scenario.members) > 50 or len(scenario.equipment) > 20:
        raise ValueError("This UI supports up to 50 members and 20 machines")
    if sum(len(m.visits) for m in scenario.members) > 300:
        raise ValueError("This UI supports up to 300 exercise visits")
    if config.max_events > 200000 or config.max_time_seconds > 86400:
        raise ValueError("Run limits must not exceed 200000 events or 86400 seconds")
    result = run_episode(scenario, create_policy("fcfs"), config)
    return {
        "scenario": scenario.model_dump(mode="json"),
        "config": config.model_dump(mode="json"),
        "result": result.model_dump(mode="json"),
        "findings": validate_result(result),
    }
