"""Behavioral cases for dispatch rules, advisory gates and forecast optimizers."""

import json
from dataclasses import replace
from itertools import pairwise

import pytest

from gym_sched.domain.models import (
    Candidate,
    EquipmentView,
    MemberView,
    Observation,
    PolicyContext,
    VisitView,
)
from gym_sched.policies import available_policies, create_policy
from gym_sched.policies.forecast import build_problem, decode, initial_genes


def visit(name="v1", durations=None, rest=0.0, predecessors=(), deviations=None):
    durations = durations or {"e1": 10.0}
    return VisitView(
        name,
        name,
        tuple(durations),
        durations,
        {},
        tuple(durations),
        deviations or {},
        rest,
        predecessors,
    )


def member(name, visits=None, ready=0.0, available=0.0, state="READY", **kwargs):
    return MemberView(
        name, state, ready, None, tuple(visits or [visit()]), (), True, available, **kwargs
    )


def observation(members, equipment=None, now=0.0, reordering=False, substitutions=False):
    equipment = equipment or [EquipmentView("e1", "bench", True, now, None)]
    return Observation(now, tuple(equipment), tuple(members), reordering, substitutions)


def candidates(obs):
    available = {e.id for e in obs.equipment if e.observed_available}
    result = []
    for person in obs.members:
        if person.state != "READY" or not person.participates:
            continue
        pending = {v.id for v in person.remaining_visits}
        for index, block in enumerate(person.remaining_visits):
            if (not obs.allow_reordering and index > 0) or set(block.predecessors) & pending:
                continue
            for unit in block.equipment_ids:
                if unit in available and (
                    obs.allow_substitutions or block.deviation_by_equipment.get(unit, 0) <= 0
                ):
                    result.append(
                        Candidate(
                            person.id,
                            block.id,
                            unit,
                            person.ready_since or 0.0,
                            block.predicted_seconds[unit],
                        )
                    )
    return tuple(result)


def pairs(decision):
    return {(r.member_id, r.visit_id, r.equipment_id) for r in decision.recommendations}


def test_factory_exact_variant_names_and_rejects_contradictory_options():
    assert len(available_policies()) == 16
    for name in available_policies():
        policy = create_policy(name)
        assert policy.name == name
        json.dumps(policy.options, allow_nan=False)
    assert create_policy("self_directed").self_directed
    with pytest.raises(ValueError, match="contradicts"):
        create_policy("robust_a0u0s0", {"acceptance": True})
    with pytest.raises(ValueError, match="Unknown"):
        create_policy("ga", {"population": 10})
    seeded = create_policy("ga", {"seed": 17})
    assert seeded.options["seed"] == seeded.context.seed == 17
    with pytest.raises(ValueError, match="seed"):
        create_policy("fcfs", {"seed": -1})


def test_fcfs_uses_request_age_and_spt_uses_predicted_duration():
    obs = observation(
        [
            member("older", [visit(durations={"e1": 30})], ready=1),
            member("newer", [visit(durations={"e1": 3})], ready=9),
        ],
        now=10,
    )
    choices = candidates(obs)
    assert next(iter(pairs(create_policy("fcfs").decide(obs, choices))))[0] == "older"
    assert next(iter(pairs(create_policy("spt").decide(obs, choices))))[0] == "newer"
    assert (
        next(iter(pairs(create_policy("aging", {"age_weight": 5}).decide(obs, choices))))[0]
        == "older"
    )


def test_matching_avoids_stranding_constrained_member():
    obs = observation(
        [
            member("a-flexible", [visit(durations={"e1": 1, "e2": 100000})]),
            member("b-constrained", [visit(durations={"e1": 2})]),
        ],
        [EquipmentView("e1", "x", True, 0, None), EquipmentView("e2", "x", True, 0, None)],
    )
    choices = candidates(obs)
    assert len(create_policy("fcfs").decide(obs, choices).recommendations) == 1
    assert pairs(create_policy("matching").decide(obs, choices)) == {
        ("a-flexible", "v1", "e2"),
        ("b-constrained", "v1", "e1"),
    }


def test_matching_minimizes_cost_within_maximum_cardinality():
    obs = observation(
        [
            member("a", [visit(durations={"e1": 1, "e2": 9})]),
            member("b", [visit(durations={"e1": 10, "e2": 1})]),
        ],
        [EquipmentView("e1", "x", True, 0, None), EquipmentView("e2", "x", True, 0, None)],
    )
    decision = create_policy("matching").decide(obs, candidates(obs))
    assert pairs(decision) == {("a", "v1", "e1"), ("b", "v1", "e2")}
    assert pairs(create_policy("robust_a0u0s0").decide(obs, candidates(obs))) == pairs(decision)


def test_robust_acceptance_and_uncertainty_terms_can_be_ablated():
    obs = observation([member("a"), member("b")])
    choices = (
        Candidate("a", "v1", "e1", 0, 10, acceptance_estimate=0.1),
        Candidate("b", "v1", "e1", 0, 10, acceptance_estimate=0.9),
    )
    assert create_policy("robust").options["refusal_cost_seconds"] == 0
    assert pairs(
        create_policy("robust_a1u0s0", {"refusal_cost_seconds": 30}).decide(obs, choices)
    ) == {("b", "v1", "e1")}
    choices = (
        replace(choices[0], uncertainty_seconds=0),
        replace(choices[1], uncertainty_seconds=100),
    )
    assert pairs(create_policy("robust_a0u1s0").decide(obs, choices)) == {("a", "v1", "e1")}


def test_stability_gate_rejects_marginal_changes_but_accepts_material_ones():
    person = member(
        "m", [visit(durations={"e1": 100, "e2": 95})], previous_recommendation=("v1", "e1")
    )
    obs = observation(
        [person], [EquipmentView("e1", "x", True, 0, None), EquipmentView("e2", "x", True, 0, None)]
    )
    changed = Candidate("m", "v1", "e2", 0, 95, is_change=True, acceptance_estimate=1)
    policy = create_policy("robust_a0u0s1")
    decision = policy.decide(obs, (changed,))
    assert decision.defer and decision.diagnostics["gated_changes"] == 1
    assert pairs(policy.decide(obs, (replace(changed, predicted_seconds=50),))) == {
        ("m", "v1", "e2")
    }
    assert pairs(create_policy("robust_a0u0s0").decide(obs, (changed,))) == {("m", "v1", "e2")}


def test_forecast_excludes_processing_rest_and_active_resource_holds():
    active = member(
        "active",
        [visit("next", {"e2": 4})],
        state="EXERCISING",
        available=15,
        active_equipment_id="e1",
    )
    waiting = member("waiting", [visit("first", {"e1": 3}, rest=7), visit("second", {"e2": 2})])
    obs = observation(
        [active, waiting],
        [EquipmentView("e1", "x", False, 0, 10), EquipmentView("e2", "x", True, 0, None)],
    )
    problem = build_problem(obs)
    schedule = decode(problem, initial_genes(problem))
    records = {op.key: op for op in schedule.operations}
    assert records["active", "next"].start >= 15
    assert records["waiting", "first"].start >= 10
    assert records["waiting", "second"].start >= records["waiting", "first"].finish + 7
    assert schedule.waiting_seconds == pytest.approx(
        sum(op.start - op.ready for op in schedule.operations)
    )


def test_forecast_does_not_drop_deadline_misses_or_control_nonparticipants():
    late = replace(member("late", [visit(durations={"e1": 20})]), deadline_seconds=1)
    autonomous = replace(member("autonomous"), participates=False)
    problem = build_problem(observation([late, autonomous]))
    assert len(problem.operations) == 1
    schedule = decode(problem, initial_genes(problem))
    assert schedule.tardiness_seconds == 19
    assert schedule.waiting_seconds == 0


def test_reordering_preserves_explicit_predecessors():
    obs = observation(
        [
            member(
                "m",
                [
                    visit("first", {"e1": 10}, rest=3),
                    visit("second", {"e1": 1}, predecessors=("first",)),
                ],
            )
        ],
        reordering=True,
    )
    problem = build_problem(obs)
    schedule = decode(problem, ((0.9, 0.0), (0.0, 0.0)))
    assert [op.visit_id for op in schedule.operations] == ["first", "second"]
    assert schedule.operations[1].start == 13


def test_fixed_order_vs_reordering_is_an_explicit_forecast_capability():
    person = member("m", [visit("first", {"e1": 10}), visit("second", {"e2": 2})])
    units = [EquipmentView("e1", "x", False, 0, 50), EquipmentView("e2", "x", True, 0, None)]
    fixed = build_problem(observation([person], units))
    flexible = build_problem(observation([person], units, reordering=True))
    genes = ((0.9, 0.0), (0.0, 0.0))
    assert decode(fixed, genes).operations[0].visit_id == "first"
    assert decode(flexible, genes).operations[0].visit_id == "second"


def test_forecast_respects_substitution_permission():
    person = member("m", [visit(durations={"e1": 10, "e2": 1}, deviations={"e2": 1})])
    units = [EquipmentView("e1", "x", True, 0, None), EquipmentView("e2", "x", True, 0, None)]
    assert build_problem(observation([person], units)).operations[0].equipment_ids == ("e1",)
    assert build_problem(observation([person], units, substitutions=True)).operations[
        0
    ].equipment_ids == ("e1", "e2")


def test_cpsat_minimizes_waiting_not_processing_or_makespan():
    obs = observation(
        [
            member("long", [visit(durations={"e1": 10})]),
            member("short", [visit(durations={"e1": 2})]),
        ]
    )
    policy = create_policy("cpsat", {"time_limit_seconds": 2.0, "deterministic_time_limit": 1.0})
    decision = policy.decide(obs, candidates(obs))
    assert decision.diagnostics["solver_status"] == "OPTIMAL"
    assert decision.diagnostics["objective_seconds"] == pytest.approx(2)
    assert decision.diagnostics["best_bound_seconds"] == pytest.approx(2)
    assert pairs(decision) == {("short", "v1", "e1")}
    assert len(policy.last_schedule.operations) == 2
    json.dumps(decision.diagnostics, allow_nan=False)


def test_cpsat_plans_all_remaining_visits_and_prescribed_rests():
    obs = observation(
        [
            member("a", [visit("one", {"e1": 2}, rest=5), visit("two", {"e1": 3})]),
            member("b", [visit("one", {"e1": 4})]),
        ]
    )
    policy = create_policy("cpsat", {"time_limit_seconds": 2.0, "deterministic_time_limit": 1.0})
    result = policy.decide(obs, candidates(obs))
    assert result.diagnostics["solver_status"] == "OPTIMAL"
    schedule = policy.last_schedule
    assert len(schedule.operations) == 3
    ops = {op.key: op for op in schedule.operations}
    assert ops["a", "two"].start >= ops["a", "one"].finish + 5
    assert result.diagnostics["objective_seconds"] == pytest.approx(schedule.waiting_seconds)
    assert schedule.waiting_seconds == pytest.approx(2)


def test_cpsat_reorder_path_waiting_and_explicit_precedence():
    units = [EquipmentView("e1", "x", False, 0, 10), EquipmentView("e2", "x", True, 0, None)]
    visits = [visit("blocked", {"e1": 2}), visit("free", {"e2": 8}, rest=1)]
    obs = observation([member("m", visits)], units, reordering=True)
    policy = create_policy("cpsat", {"time_limit_seconds": 2.0, "deterministic_time_limit": 1.0})
    result = policy.decide(obs, candidates(obs))
    assert result.diagnostics["solver_status"] == "OPTIMAL"
    assert pairs(result) == {("m", "free", "e2")}
    assert result.diagnostics["objective_seconds"] == pytest.approx(1)
    constrained = replace(visits[1], predecessors=("blocked",))
    constrained_obs = observation(
        [member("m", [visits[0], constrained]), member("dummy", [visit("v", {"e2": 1})])],
        units,
        reordering=True,
    )
    policy.decide(constrained_obs, candidates(constrained_obs))
    member_ops = [op for op in policy.last_schedule.operations if op.member_id == "m"]
    assert [op.visit_id for op in member_ops] == ["blocked", "free"]


def test_cpsat_timeout_falls_back_to_matching_and_reports_status():
    obs = observation([member("a"), member("b")])
    result = create_policy("cpsat", {"time_limit_seconds": 0.0}).decide(obs, candidates(obs))
    assert result.diagnostics["fallback"]
    assert result.diagnostics["solver_status"] == "UNKNOWN"
    assert pairs(result) == pairs(create_policy("matching").decide(obs, candidates(obs)))


def test_ga_reproduces_budgeted_runs_and_keeps_feasible_warm_start():
    obs = observation(
        [
            member("a", [visit("one", {"e1": 2}, rest=4), visit("two", {"e1": 1})]),
            member("b", [visit(durations={"e1": 3})]),
        ]
    )
    policy = create_policy("ga", {"max_evaluations": 65})
    policy.reset(PolicyContext(seed=12))
    first = policy.decide(obs, candidates(obs))
    first_schedule = policy.last_schedule
    assert first.diagnostics["evaluations"] == 65
    assert first.diagnostics["warm_start_operations"] == 0
    second = policy.decide(obs, candidates(obs))
    assert second.diagnostics["warm_start_operations"] == 3
    policy.reset(PolicyContext(seed=12))
    replay = policy.decide(obs, candidates(obs))
    assert first.recommendations == replay.recommendations
    assert first_schedule == policy.last_schedule
    assert len(first_schedule.operations) == 3
    assert replay.diagnostics["evaluations"] == 65
    ordered = sorted(first_schedule.operations, key=lambda op: op.start)
    assert all(a.finish <= b.start for a, b in pairwise(ordered))
    assert replay.diagnostics["objective_seconds"] == pytest.approx(first_schedule.waiting_seconds)
    json.dumps(replay.diagnostics, allow_nan=False)


def test_ga_zero_wall_budget_still_returns_a_feasible_incumbent():
    obs = observation([member("a"), member("b")])
    policy = create_policy("ga", {"budget_mode": "wall", "time_limit_seconds": 0.0})
    result = policy.decide(obs, candidates(obs))
    assert result.diagnostics["evaluations"] == 1
    assert result.diagnostics["incumbent_only"]
    assert result.diagnostics["fallback"]
    assert (
        result.diagnostics["budget_overrun_seconds"] >= result.diagnostics["initial_decode_seconds"]
    )
    assert len(policy.last_schedule.operations) == 2
    assert result.recommendations


def test_optional_nsga2_returns_feasible_nondominated_forecast_schedules():
    pytest.importorskip("pymoo")
    from gym_sched.policies.pareto import nsga2_frontier

    units = [EquipmentView("e1", "x", True, 0, None), EquipmentView("e2", "x", True, 0, None)]
    obs = observation(
        [
            member("a", [visit(durations={"e1": 10, "e2": 2}, deviations={"e2": 1})]),
            member("b", [visit(durations={"e1": 4})]),
        ],
        units,
        substitutions=True,
    )
    points = nsga2_frontier(obs, seed=7, population_size=12, generations=5)
    assert points
    assert points == nsga2_frontier(obs, seed=7, population_size=12, generations=5)
    for point in points:
        assert len(point.schedule.operations) == 2
        for unit in ("e1", "e2"):
            operations = sorted(
                (op for op in point.schedule.operations if op.equipment_id == unit),
                key=lambda op: op.start,
            )
            assert all(a.finish <= b.start for a, b in pairwise(operations))
        assert point.waiting_seconds == pytest.approx(point.schedule.waiting_seconds)
        assert not any(
            all(x <= y for x, y in zip(other.objectives, point.objectives, strict=True))
            and any(x < y for x, y in zip(other.objectives, point.objectives, strict=True))
            for other in points
        )


@pytest.mark.parametrize("name", available_policies())
def test_every_policy_handles_empty_eligibility_without_an_assignment(name):
    obs = observation([member("m")])
    decision = create_policy(name).decide(obs, ())
    assert decision.recommendations == ()
    assert decision.defer
