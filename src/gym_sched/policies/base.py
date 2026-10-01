"""Policy contracts: only immutable, observable views cross this boundary."""

from __future__ import annotations

import math
from typing import Any, ClassVar, Protocol

from gym_sched.domain.models import Candidate, Decision, Observation, PolicyContext


class Policy(Protocol):
    name: str
    options: dict[str, Any]
    self_directed: bool

    def reset(self, context: PolicyContext) -> None: ...

    def decide(self, observation: Observation, candidates: tuple[Candidate, ...]) -> Decision: ...


class BasePolicy:
    self_directed = False
    defaults: ClassVar[dict[str, Any]] = {}

    def __init__(self, name: str, options: dict[str, Any] | None = None):
        supplied = dict(options or {})
        defaults = {"seed": 0, **self.defaults}
        unknown = set(supplied) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown {name} options: {', '.join(sorted(unknown))}")
        self.name = name
        self.options = {**defaults, **supplied}
        for key, value in self.options.items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
        seed = self.options["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.context = PolicyContext(seed=seed)

    @property
    def config(self) -> dict[str, Any]:
        return dict(self.options)

    def reset(self, context: PolicyContext) -> None:
        self.context = context


def candidate_key(candidate: Candidate) -> tuple[str, str, str]:
    return candidate.member_id, candidate.visit_id, candidate.equipment_id
