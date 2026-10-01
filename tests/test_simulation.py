from dataclasses import asdict

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from gym_sched.domain import (
    Decision,
    Equipment,
    Member,
    Recommendation,
    Scenario,
    SimulationConfig,
    Visit,
)
from gym_sched.evaluation.metrics import validate_result
from gym_sched.scenarios import generate_scenario
from gym_sched.scenarios.randomness import content_hash, uniform
from gym_sched.simulation import Simulator, run_episode


class SimplePolicy:
    name = "test_fcfs"

    def __init__(self, shortest=False):
        self.options = {}
        self.shortest = shortest
        self.observations = []

    def reset(self, context):
        self.observations = []

    def decide(self, observation, candidates):
        self.observations.append(observation)
        chosen, mids, eids = [], set(), set()
        ordered = sorted(
            candidates,
            key=lambda c: (
                c.predicted_seconds if self.shortest else c.ready_since,
                c.member_id,
                c.visit_id,
                c.equipment_id,
            ),
        )
        for c in ordered:
            if c.member_id not in mids and c.equipment_id not in eids:
                chosen.append(Recommendation(c.member_id, c.visit_id, c.equipment_id))
                mids.add(c.member_id)
                eids.add(c.equipment_id)
        return Decision(tuple(chosen))


def simple(durations=(360, 120, 240), **member_options):
    return Scenario(
        id="hand",
        equipment=[Equipment(id="e")],
        members=[
            Member(
                id=chr(97 + i),
                arrival_seconds=0,
                visits=[
                    Visit(
                        id="v",
                        equipment_ids=["e"],
                        actual_seconds={"e": d},
                        predicted_seconds={"e": d},
                        rest_after_seconds=0,
                    )
                ],
                **member_options,
            )
            for i, d in enumerate(durations)
        ],
    )


def test_hand_computed_fcfs_spt():
    fcfs = run_episode(simple(), SimplePolicy())
    spt = run_episode(simple(), SimplePolicy(shortest=True))
    assert fcfs.metrics["mean_wait_seconds"] == pytest.approx(280)
    assert spt.metrics["mean_wait_seconds"] == pytest.approx(160)
    assert fcfs.metrics["completion_rate"] == spt.metrics["completion_rate"] == 1
    assert validate_result(fcfs) == validate_result(spt) == []


def test_identical_logs_and_scenario_not_mutated():
    scenario = generate_scenario({"member_count": 8, "duration_cv": 0.3}, 12)
    before = content_hash(scenario)
    one = run_episode(scenario, SimplePolicy())
    two = run_episode(scenario, SimplePolicy())
    assert one.events == two.events
    assert one.member_summaries == two.member_summaries
    assert before == content_hash(scenario)


def test_no_hidden_fields_or_future_members_and_no_truth_dependent_mask():
    scenario = simple((360, 120))
    scenario.members[1].arrival_seconds = 30
    policy = SimplePolicy()
    run_episode(scenario, policy)
    first = asdict(policy.observations[0])
    assert [m["id"] for m in first["members"]] == ["a"]
    assert "actual_seconds" not in str(first)
    assert "acceptance_probability" not in str(first)
    sim = Simulator(simple((360, 120)), SimplePolicy())
    sim.members["a"].state = "ready"
    sim.members["a"].ready_since = 0
    sim.observed["e"] = (True, 0, None)
    before = sim.candidates()
    sim.occupants["e"] = "b"  # hidden occupancy must not affect eligibility
    assert sim.candidates() == before


def test_unobserved_actual_duration_does_not_change_initial_policy_input():
    a, b = simple(), simple()
    b.members[0].visits[0].actual_seconds["e"] = 3000
    pa, pb = SimplePolicy(), SimplePolicy()
    run_episode(a, pa)
    run_episode(b, pb)
    assert pa.observations[0] == pb.observations[0]


def test_response_wait_and_walking_time_accounting():
    result = run_episode(
        simple((120,)),
        SimplePolicy(),
        SimulationConfig(response_delay_seconds=5, walking_seconds=7),
    )
    m = result.member_summaries[0]
    assert m["waiting_seconds"] == m["response_wait_seconds"] == 5
    assert m["walking_seconds"] == 7
    assert m["exercise_seconds"] == 120
    assert m["completion_time_seconds"] == 132


def test_refusal_is_immediate_and_ignored_waits_only_for_expiry():
    refused = run_episode(simple((60,), acceptance_probability=0), SimplePolicy())
    ignored = run_episode(simple((60,), ignore_probability=1), SimplePolicy())
    assert refused.member_summaries[0]["refused"] == 1
    assert refused.member_summaries[0]["waiting_seconds"] == 0
    assert ignored.member_summaries[0]["ignored"] == 1
    assert ignored.member_summaries[0]["waiting_seconds"] == 30
    assert refused.status == ignored.status == "completed"


def test_independent_members_do_not_depend_on_app_sensors_or_retry_timers():
    scenario = simple((60, 60), participates=False)
    result = run_episode(scenario, SimplePolicy(), {"observation_dropout": 1})
    starts = [e["time"] for e in result.events if e["event"] == "exercise_started"]
    assert starts == [0, 60]
    assert result.metrics["recommendation_count"] == 0
    assert result.status == "completed"
    assert validate_result(result) == []


def test_failed_access_can_choose_other_free_unit_at_same_time():
    scenario = simple((60, 60), participates=False)
    scenario.equipment.append(Equipment(id="other"))
    for m in scenario.members:
        v = m.visits[0]
        v.equipment_ids.append("other")
        v.actual_seconds["other"] = v.predicted_seconds["other"] = 60
    result = run_episode(scenario, SimplePolicy(), {"observation_dropout": 1})
    starts = [e for e in result.events if e["event"] == "exercise_started"]
    assert len(starts) == 2 and {e["time"] for e in starts} == {0}
    assert {e["equipment_id"] for e in starts} == {"e", "other"}
    assert result.metrics["mean_wait_seconds"] == 0
    assert result.metrics["failed_claims"] == 1
    assert validate_result(result) == []


def test_refusal_after_machine_taken_waits_for_release_without_repeat_advice():
    scenario = simple((60, 60), acceptance_probability=0)
    scenario.members[0].participates = False
    scenario.members[0].arrival_seconds = 1
    # Advice at 0; independent member starts at 1; refusal at 5; release at 61.
    result = run_episode(scenario, SimplePolicy(), {"response_delay_seconds": 5})
    assert result.status == "completed"
    guided = result.member_summaries[1]
    assert guided["refused"] == guided["recommendations"] == 1
    assert guided["waiting_seconds"] == 61
    assert guided["completed"]
    assert validate_result(result) == []


def test_response_at_expiry_is_ignored_and_late_response_has_no_effect():
    result = run_episode(simple((60,)), SimplePolicy(), {"response_delay_seconds": 30})
    assert result.member_summaries[0]["accepted"] == 0
    assert result.member_summaries[0]["ignored"] == 1
    assert len([e for e in result.events if e["event"] == "exercise_started"]) == 1


def test_stale_advice_cannot_double_book_and_preserves_wait():
    scenario = simple((120, 120))
    scenario.members[0].participates = False
    result = run_episode(
        scenario,
        SimplePolicy(),
        {"observation_delay_seconds": 5, "walking_seconds": 2, "response_delay_seconds": 1},
    )
    assert result.metrics["failed_claims"] > 0
    assert validate_result(result) == []
    assert result.member_summaries[1]["waiting_seconds"] > 0


def test_deadline_exact_completion_and_no_new_start_at_deadline():
    scenario = simple((60, 120), deadline_seconds=60)
    result = run_episode(scenario, SimplePolicy())
    assert result.member_summaries[0]["on_time"]
    assert not result.member_summaries[1]["completed"]
    assert result.member_summaries[1]["waiting_seconds"] == 60
    assert validate_result(result) == []


def test_deadline_does_not_preempt_and_overtime_is_counted():
    result = run_episode(simple((90,), deadline_seconds=60), SimplePolicy())
    m = result.member_summaries[0]
    assert m["completed"] and not m["on_time"]
    assert m["overtime_seconds"] == 30
    assert validate_result(result) == []


def test_patience_reacts_to_wait_not_episode_clock():
    result = run_episode(simple((90, 90), patience_wait_seconds=20), SimplePolicy())
    a, b = result.member_summaries
    assert a["completed"]
    assert b["waiting_seconds"] == 20 and not b["completed"]
    assert result.metrics["mean_wait_seconds"] == 10
    assert result.metrics["completion_rate"] == 0.5


def test_rest_and_fixed_order():
    scenario = simple((60,))
    scenario.members[0].visits[0].rest_after_seconds = 10
    scenario.members[0].visits.append(
        Visit(
            id="v2",
            equipment_ids=["e"],
            actual_seconds={"e": 30},
            predicted_seconds={"e": 30},
            predecessors=["v"],
            rest_after_seconds=0,
        )
    )
    result = run_episode(scenario, SimplePolicy())
    starts = [e for e in result.events if e["event"] == "exercise_started"]
    assert [(e["visit_id"], e["time"]) for e in starts] == [("v", 0), ("v2", 70)]
    assert result.member_summaries[0]["rest_seconds"] == 10
    assert result.member_summaries[0]["waiting_seconds"] == 0


def test_outage_blocks_new_starts_without_preemption():
    scenario = simple((100, 60))
    scenario.equipment[0].outages = [(20, 120)]
    result = run_episode(scenario, SimplePolicy())
    starts = [e["time"] for e in result.events if e["event"] == "exercise_started"]
    assert starts == [0, 120]
    assert validate_result(result) == []


def test_stalled_and_truncated_not_silently_successful():
    class NoOp(SimplePolicy):
        def decide(self, observation, candidates):
            return Decision(defer=True)

    stalled = run_episode(simple((60,)), NoOp())
    assert stalled.status == "stalled"
    capped = run_episode(simple((60,)), SimplePolicy(), {"max_time_seconds": 20})
    assert capped.status == "truncated"
    assert capped.member_summaries[0]["exercise_seconds"] == 20
    assert validate_result(capped)


def test_invalid_policy_batch_is_rejected():
    class Invalid(SimplePolicy):
        def decide(self, observation, candidates):
            return Decision((Recommendation("a", "v", "e"), Recommendation("b", "v", "e")))

    with pytest.raises(ValueError, match="conflicting"):
        run_episode(simple(), Invalid())


def test_delayed_decision_revalidated_after_departure():
    result = run_episode(
        simple((60,), deadline_seconds=2), SimplePolicy(), {"decision_delay_seconds": 5}
    )
    assert not result.member_summaries[0]["completed"]
    assert result.member_summaries[0]["waiting_seconds"] == 2
    assert not [e for e in result.events if e["event"] == "exercise_started"]


@given(seed=st.integers(0, 10000), count=st.integers(1, 12), cv=st.sampled_from([0, 0.3, 0.6]))
@settings(max_examples=20, deadline=None)
def test_generated_episode_invariants(seed, count, cv):
    scenario = generate_scenario(
        {
            "member_count": count,
            "duration_cv": cv,
            "arrival_pattern": "burst",
            "adoption_rate": 0.5,
        },
        seed,
    )
    result = run_episode(scenario, SimplePolicy(), {"observation_delay_seconds": 5})
    assert result.status == "completed"
    assert validate_result(result) == []
    for m in result.member_summaries:
        assert m["departure"] - m["arrival"] == pytest.approx(
            m["waiting_seconds"] + m["rest_seconds"] + m["walking_seconds"] + m["exercise_seconds"]
        )


def test_keyed_randomness_does_not_depend_on_consumption_order():
    before = uniform(42, "duration", "a", "v", "e")
    uniform(42, "duration", "other", "other", "other")
    assert uniform(42, "duration", "a", "v", "e") == before
    assert uniform(42, "arrivals", "a", "v", "e") != before


def test_config_and_scenario_validation():
    with pytest.raises(ValueError):
        generate_scenario({"unknown_parameter": 1})
    with pytest.raises(ValueError):
        generate_scenario({"compatibility_units": 3, "units_per_type": 2})
    with pytest.raises(ValueError):
        SimulationConfig(retry_cooldown_seconds=30)  # Removed, not silently ignored.


def test_truncation_does_not_count_future_members_as_zero_wait_entrants():
    scenario = simple((60, 60))
    scenario.members[1].arrival_seconds = 100
    result = run_episode(scenario, SimplePolicy(), {"max_time_seconds": 20})
    assert result.status == "truncated"
    assert result.metrics["member_count"] == 1
    assert [m["member_id"] for m in result.member_summaries] == ["a"]


def test_acceptance_prior_uses_declared_preferences_not_latent_parameters():
    scenario = generate_scenario({"member_count": 1, "allow_substitutions": True}, 42)
    sim = Simulator(scenario, SimplePolicy(), {"allow_substitutions": True})
    m = sim.members["m000"]
    m.state, m.ready_since = "ready", 0
    for eid in sim.observed:
        sim.observed[eid] = (True, 0, None)
    first = sim.candidates()
    assert {c.acceptance_estimate for c in first if c.preferred} == {0.5}
    assert {c.acceptance_estimate for c in first if not c.preferred} == {1 / 3}
    m.spec.acceptance_probability = 0
    m.spec.nonpreferred_acceptance_multiplier = 0
    assert sim.candidates() == first
    m.response_counts["preferred"] = [2, 0]
    assert {c.acceptance_estimate for c in sim.candidates() if c.preferred} == {0.75}


def test_aging_wait_excludes_walking_from_request_age():
    sim = Simulator(simple((60,)), SimplePolicy())
    m = sim.members["a"]
    sim._ready(m)
    sim._advance(5)
    m.state = "walking"
    sim._advance(15)
    sim._ready(m, preserve_age=True)
    sim.observed["e"] = (True, 15, None)
    candidate = sim.candidates()[0]
    assert candidate.ready_since == 0
    assert candidate.ready_wait_seconds == 5


def test_reordering_is_opt_in_and_reported_separately():
    scenario = simple((60,))
    scenario.members[0].visits.append(
        Visit(
            id="short",
            equipment_ids=["e"],
            actual_seconds={"e": 5},
            predicted_seconds={"e": 5},
            rest_after_seconds=0,
        )
    )
    fixed = run_episode(scenario, SimplePolicy(shortest=True))
    flexible = run_episode(scenario, SimplePolicy(shortest=True), {"allow_reordering": True})
    assert fixed.member_summaries[0]["reordered_visits"] == 0
    assert flexible.member_summaries[0]["reordered_visits"] == 1
    assert flexible.metrics["substitutions_per_member"] == 0
