"""Hand-checkable demonstration, with shared workload and real engine outputs."""

import json
import re

from gym_sched.visual_demo import build_payload, export_demo


def test_reordering_demo_matches_hand_calculation():
    payload = build_payload()
    fixed, flexible = (payload["runs"][key] for key in ("fixed", "flexible"))
    assert fixed["scenario_hash"] == flexible["scenario_hash"]
    for result, waiting, elapsed, order in [
        (fixed, 180, 480, ["A", "B"]),
        (flexible, 0, 300, ["B", "A"]),
    ]:
        casey = next(m for m in result["member_summaries"] if m["member_id"] == "casey")
        assert casey["waiting_seconds"] == waiting
        assert casey["departure"] - casey["arrival"] == elapsed
        assert casey["completed_visits"] == 2
        assert casey["rest_seconds"] == 60
        assert result["status"] == "completed"
        starts = [
            e["equipment_id"]
            for e in result["events"]
            if e["event"] == "exercise_started" and e["member_id"] == "casey"
        ]
        assert starts == order


def test_export_contains_replayable_engine_events(tmp_path):
    path = export_demo(tmp_path / "gym.html")
    html = path.read_text()
    match = re.search(r'<script id="simulation-data" type="application/json">(.*?)</script>', html)
    assert match
    payload = json.loads(match[1])
    assert payload["runs"]["fixed"]["events"]
    assert "/*__PAYLOAD__*/" not in html


def test_saved_run_reproduces_selected_permission(tmp_path):
    from gym_sched.visual_server import save_run

    payload = save_run(tmp_path, "flexible")
    directory = tmp_path / "runs" / payload["saved_run"]
    assert json.loads((directory / "scenario.json").read_text()) == payload["scenario"]
    assert json.loads((directory / "config.json").read_text())["allow_reordering"] is True
    assert json.loads((directory / "result.json").read_text()) == payload["runs"]["flexible"]
    assert json.loads((directory / "manifest.json").read_text())["order"] == "flexible"


def test_invalid_order_does_not_save_run(tmp_path):
    import pytest

    from gym_sched.visual_server import save_run

    with pytest.raises(ValueError):
        save_run(tmp_path, "random")
    assert not (tmp_path / "runs").exists()


def test_rejection_keeps_machine_free_and_waits_for_original_order():
    payload = build_payload("response", 100)
    accepted, declined = (payload["runs"][key] for key in ("fixed", "flexible"))
    for result, waiting, elapsed, order, refused in [
        (accepted, 2, 302, ["B", "A"], 0),
        (declined, 181, 481, ["A", "B"], 2),
    ]:
        casey = next(m for m in result["member_summaries"] if m["member_id"] == "casey")
        assert casey["waiting_seconds"] == waiting
        assert casey["completion_time_seconds"] == elapsed
        assert casey["refused"] == refused
        assert casey["completed_visits"] == 2
        starts = [e for e in result["events"] if e["event"] == "exercise_started"]
        assert [e["equipment_id"] for e in starts if e["member_id"] == "casey"] == order
    refusals = [e for e in declined["events"] if e["event"] == "refused"]
    assert refusals[0]["time"] == 2
    assert not any(
        e["event"] == "exercise_started" and e["equipment_id"] == "B" and e["time"] < 362
        for e in declined["events"]
    )
    assert (
        len(
            [
                e
                for e in declined["events"]
                if e["event"] == "recommendation_issued" and e["time"] < 181
            ]
        )
        == 1
    )


def test_decline_run_saves_its_actual_latent_configuration(tmp_path):
    from gym_sched.visual_server import save_run

    payload = save_run(tmp_path, "flexible", "response", 100)
    directory = tmp_path / "runs" / payload["saved_run"]
    scenario = json.loads((directory / "scenario.json").read_text())
    casey = next(m for m in scenario["members"] if m["id"] == "casey")
    assert casey["acceptance_probability"] == 0
    assert casey["independent_keep_plan_order"] is True
    assert json.loads((directory / "config.json").read_text())["allow_reordering"] is True


def test_probability_is_saved_and_reproducible():
    payload = build_payload("response", 35)
    casey = next(m for m in payload["scenarios"]["flexible"]["members"] if m["id"] == "casey")
    assert casey["acceptance_probability"] == 0.65
    assert casey["changed_acceptance_probability"] is None
    assert (
        payload["runs"]["flexible"]["events"]
        == build_payload("response", 35)["runs"]["flexible"]["events"]
    )
    zero = build_payload("response", 0)
    assert zero["runs"]["fixed"]["events"] == zero["runs"]["flexible"]["events"]


def test_invalid_refusal_percent():
    import pytest

    for percent in [-1, 101, float("nan"), "50", True]:
        with pytest.raises(ValueError):
            build_payload("response", percent)


def test_competing_member_uses_actual_availability_after_refusal():
    payload = build_payload("competition", 100)
    for key, offer_time, start_time, waiting in [("fixed", 122, 123, 120), ("flexible", 3, 4, 1)]:
        result = payload["runs"][key]
        sam = next(m for m in result["member_summaries"] if m["member_id"] == "sam")
        assert sam["waiting_seconds"] == waiting
        assert sam["completed_visits"] == 1
        assert sam["refused"] == 0
        assert result["status"] == "completed"
        events = [e for e in result["events"] if e.get("member_id") == "sam"]
        assert (
            next(e for e in events if e["event"] == "recommendation_issued")["time"] == offer_time
        )
        assert next(e for e in events if e["event"] == "exercise_started")["time"] == start_time
        occupied = {}
        for event in result["events"]:
            if event["event"] == "exercise_started":
                assert event["equipment_id"] not in occupied
                occupied[event["equipment_id"]] = event["member_id"]
            elif event["event"] == "exercise_completed":
                assert occupied.pop(event["equipment_id"]) == event["member_id"]
        assert not occupied
    # Introducing competition does not change Casey's original behavior model.
    pair = build_payload("response", 100)
    for key in ["fixed", "flexible"]:
        assert payload["scenarios"][key]["members"][:2] == pair["scenarios"][key]["members"]


def test_competition_saved_scenario_includes_sam(tmp_path):
    from gym_sched.visual_server import save_run

    payload = save_run(tmp_path, "flexible", "competition", 100)
    directory = tmp_path / "runs" / payload["saved_run"]
    scenario = json.loads((directory / "scenario.json").read_text())
    assert [m["id"] for m in scenario["members"]] == ["alex", "casey", "sam"]
    assert json.loads((directory / "manifest.json").read_text())["experiment"] == "competition"


def test_walking_conflict_does_not_reserve_or_double_allocate():
    payload = build_payload("race")
    assert payload["scenarios"]["fixed"] == payload["scenarios"]["flexible"]
    baseline, walking = (payload["runs"][k] for k in ["fixed", "flexible"])
    sam = next(m for m in walking["member_summaries"] if m["member_id"] == "sam")
    assert sam["failed_claims"] == 1
    assert sam["walking_seconds"] == 20
    assert sam["completed_visits"] == 1
    assert (
        next(m for m in baseline["member_summaries"] if m["member_id"] == "sam")["failed_claims"]
        == 0
    )
    offers = [e for e in walking["events"] if e["event"] == "recommendation_issued"]
    assert [(e["member_id"], e["time"], e["equipment_id"]) for e in offers] == [
        ("casey", 1, "B"),
        ("sam", 2, "B"),
    ]
    failures = [e for e in walking["events"] if e["event"] == "claim_failed"]
    assert failures[0]["member_id"] == "sam" and failures[0]["time"] == 13
    assert any(
        e["event"] == "exercise_started" and e["member_id"] == "sam" and e["time"] == 142
        for e in walking["events"]
    )
    occupied = {}
    for e in walking["events"]:
        if e["event"] == "exercise_started":
            assert e["equipment_id"] not in occupied
            occupied[e["equipment_id"]] = e["member_id"]
        elif e["event"] == "exercise_completed":
            assert occupied.pop(e["equipment_id"]) == e["member_id"]
    assert not occupied
    assert all(m["completed"] and m["refused"] == 0 for m in walking["member_summaries"])
