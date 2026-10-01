"""Workbook E01 contract and event-level evidence for its mechanisms."""

from collections import Counter
from pathlib import Path

import pytest

from gym_sched.domain import Equipment, Member, Scenario, Visit
from gym_sched.evaluation.experiments import _expanded_config
from gym_sched.evaluation.metrics import validate_result
from gym_sched.policies import create_policy
from gym_sched.scenarios import generate_scenario

CONFIG = Path(__file__).parents[1] / "configs/experiments/e01.yaml"


@pytest.mark.parametrize("fraction", [0, 0.25, 0.5, 0.75, 1])
def test_exact_response_cohorts_do_not_change_workload_or_leak_to_policy(fraction):
    from dataclasses import asdict

    from gym_sched.simulation import Simulator, run_episode

    config = {
        "member_count": 4,
        "equipment_types": 1,
        "units_per_type": 4,
        "visits_per_member": 2,
        "acceptance_model": "member_cohort",
        "acceptor_fraction": fraction,
    }
    scenario = generate_scenario(config, 42)
    control = generate_scenario({**config, "acceptor_fraction": 1}, 42)
    assert sum(m.acceptance_probability for m in scenario.members) == 4 * fraction
    assert [(m.arrival_seconds, m.visits) for m in scenario.members] == [
        (m.arrival_seconds, m.visits) for m in control.members
    ]
    result = run_episode(scenario, create_policy("fcfs"))
    expected = {m.id: m.acceptance_probability for m in scenario.members}
    for m in result.member_summaries:
        assert m["accepted"] == (2 if expected[m["member_id"]] else 0)
        assert m["refused"] == (0 if expected[m["member_id"]] else 2)
    assert result.metrics["mean_wait_seconds"] == 0
    sim = Simulator(scenario, create_policy("fcfs"))
    assert "acceptance_probability" not in str(asdict(sim.observation()))
    assert validate_result(result) == []


def test_cohort_fraction_applies_only_to_participants_with_stable_rounding():
    config = {
        "member_count": 21,
        "adoption_rate": 0.5,
        "acceptance_model": "member_cohort",
        "acceptor_fraction": 0.5,
    }
    scenario = generate_scenario(config, 42)
    assert scenario == generate_scenario(config, 42)
    participants = [m for m in scenario.members if m.participates]
    assert sum(m.acceptance_probability for m in participants) == int(len(participants) * 0.5 + 0.5)
    more = generate_scenario({**config, "acceptor_fraction": 0.75}, 42)
    acceptors = {m.id for m in participants if m.acceptance_probability == 1}
    assert acceptors <= {
        m.id for m in more.members if m.participates and m.acceptance_probability == 1
    }


@pytest.mark.parametrize(
    "options",
    [
        {"acceptance_model": "member_cohort"},
        {"acceptor_fraction": 0.5},
        {"acceptance_model": "member_cohort", "acceptor_fraction": 1.1},
        {
            "acceptance_model": "member_cohort",
            "acceptor_fraction": 0.5,
            "acceptance_probability": 0.5,
        },
        {"acceptance_model": "member_cohort", "acceptor_fraction": 0.5, "ignore_probability": 0.1},
    ],
)
def test_reject_ambiguous_cohort_config(options):
    with pytest.raises(ValueError):
        generate_scenario(options)


def test_e01a_quiet_hand_calculated_timeline():
    from gym_sched.simulation import run_episode

    path = Path(__file__).parents[1] / "data/fixtures/e01a_quiet.json"
    scenario = Scenario.model_validate_json(path.read_text())
    for name in ("self_directed", "fcfs"):
        result = run_episode(scenario, create_policy(name))
        starts = [
            (e["member_id"], e["equipment_id"], e["time"])
            for e in result.events
            if e["event"] == "exercise_started"
        ]
        assert starts == [("A", "machine-1", 0), ("B", "machine-2", 60), ("C", "machine-1", 360)]
        assert result.end_time == 480
        assert all(m["waiting_seconds"] == 0 for m in result.member_summaries)
        assert result.metrics["completion_rate"] == 1
        assert result.metrics["failed_claims"] == 0
        assert result.metrics["recommendation_count"] == (3 if name == "fcfs" else 0)
        assert sum(m["accepted"] for m in result.member_summaries) == (3 if name == "fcfs" else 0)
        assert validate_result(result) == []


def test_e01_panel_contract_and_independent_workloads():
    config = _expanded_config(CONFIG)
    assert config["policies"] == ["self_directed", "fcfs", "spt", "matching", "robust"]
    assert len(config["scenario_configs"]) == 10
    assert len(config["splits"]["train"]) == 30
    assert "test" not in config["splits"]
    for source in config["scenario_configs"].values():
        a = generate_scenario(source, 42)
        b = generate_scenario(source, 43)
        assert a == generate_scenario(source, 42)
        # Compare physical workloads, not hashes differing only in seed/ID.
        assert [m.arrival_seconds for m in a.members] != [m.arrival_seconds for m in b.members]
        assert all(v.actual_seconds == v.predicted_seconds for m in a.members for v in m.visits)


@pytest.mark.parametrize("size", [2, 4, 8])
def test_synchronized_batches_and_partial_last_batch(size):
    scenario = generate_scenario(
        {
            "arrival_pattern": "poisson_batches",
            "arrival_batch_size": size,
            "member_count": size * 2 + 1,
            "member_pace_factors": [1.0],
        },
        42,
    )
    assert sorted(Counter(m.arrival_seconds for m in scenario.members).values()) == [1, size, size]
    assert len({m.visits[0].actual_seconds["t0-u0"] for m in scenario.members}) == 1


@pytest.mark.parametrize(
    "options",
    [
        {"arrival_batch_size": 0},
        {"member_pace_factors": []},
        {"member_pace_factors": [0]},
        {"member_pace_factors": [-1]},
        {"member_pace_factors": [float("nan")]},
    ],
)
def test_invalid_batch_and_pace_settings(options):
    with pytest.raises(ValueError):
        generate_scenario(options)


def test_default_even_workload_is_unchanged():
    scenario = generate_scenario({"member_count": 5})
    assert [m.arrival_seconds for m in scenario.members] == [0, 0, 0, 0, 3600]
    assert [m.visits[0].actual_seconds["t0-u0"] for m in scenario.members] == [
        300,
        180,
        240,
        300,
        180,
    ]


def test_s17_fresh_advice_loses_access_to_independent_walker():
    from gym_sched.simulation import run_episode

    scenario = Scenario(
        id="s17-walk-race",
        equipment=[Equipment(id="e")],
        members=[
            Member(
                id=mid,
                arrival_seconds=arrival,
                participates=guided,
                visits=[
                    Visit(
                        id="v",
                        equipment_ids=["e"],
                        actual_seconds={"e": 120},
                        predicted_seconds={"e": 120},
                        rest_after_seconds=0,
                    )
                ],
            )
            for mid, arrival, guided in [("independent", 0, False), ("guided", 1, True)]
        ],
    )
    result = run_episode(scenario, create_policy("fcfs"), {"walking_seconds": 30})
    issued = next(e for e in result.events if e["event"] == "recommendation_issued")
    started = next(e for e in result.events if e["event"] == "exercise_started")
    failed = next(e for e in result.events if e["event"] == "claim_failed")
    assert issued["time"] == 1 and started["time"] == 30 and failed["time"] == 31
    assert started["member_id"] == "independent"
    assert failed["recommendation_id"] == issued["recommendation_id"]
    assert result.metrics["completion_rate"] == 1
    assert validate_result(result) == []


@pytest.mark.parametrize("family", list(_expanded_config(CONFIG)["scenario_configs"]))
def test_e01_regimes_complete_and_audit_all_policies(family):
    from gym_sched.simulation import run_episode

    config = _expanded_config(CONFIG)
    source = config["scenario_configs"][family]
    scenario = generate_scenario(source, 42)
    for name in config["policies"]:
        result = run_episode(
            scenario, create_policy(name, config["policy_config"][name]), source["simulation"]
        )
        assert result.metrics["completion_rate"] == 1
        assert validate_result(result) == []
        if family.startswith("S18") and name != "self_directed":
            finishes = Counter(
                e["time"] for e in result.events if e["event"] == "exercise_completed"
            )
            assert max(finishes.values(), default=0) >= 2
