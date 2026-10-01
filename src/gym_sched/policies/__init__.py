"""Policy factory. Candidate sets contain observable eligibility, never truth."""

from __future__ import annotations

import re
from typing import Any

from gym_sched.policies.base import Policy
from gym_sched.policies.heuristics import (
    AgingPolicy,
    FCFSPolicy,
    MatchingPolicy,
    RobustMatchingPolicy,
    SelfDirectedPolicy,
    SPTPolicy,
)
from gym_sched.policies.optimization import CPSATPolicy, GAPolicy

_POLICIES = {
    "self_directed": SelfDirectedPolicy,
    "fcfs": FCFSPolicy,
    "spt": SPTPolicy,
    "aging": AgingPolicy,
    "matching": MatchingPolicy,
    "robust": RobustMatchingPolicy,
    "cpsat": CPSATPolicy,
    "ga": GAPolicy,
}


def available_policies() -> tuple[str, ...]:
    return (
        *_POLICIES,
        *(f"robust_a{a}u{u}s{s}" for a in range(2) for u in range(2) for s in range(2)),
    )


def create_policy(name: str, options: dict[str, Any] | None = None) -> Policy:
    supplied = dict(options or {})
    ablation = re.fullmatch(r"robust_a([01])u([01])s([01])", name)
    if ablation:
        factors = dict(
            zip(
                ("acceptance", "uncertainty", "stability"),
                (bool(int(value)) for value in ablation.groups()),
                strict=True,
            )
        )
        for key, value in factors.items():
            if key in supplied and supplied[key] != value:
                raise ValueError(f"{key} contradicts the named ablation {name}")
        return RobustMatchingPolicy(name, {**supplied, **factors})
    try:
        policy_class = _POLICIES[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown policy {name!r}; choose from {', '.join(available_policies())}"
        ) from error
    return policy_class(name, supplied)


__all__ = ["Policy", "available_policies", "create_policy"]
