"""Generated arrivals/capacity and editable profiles reach the real simulator."""

import json

import pytest

from gym_sched.configurable_gym import GymSettings, generate_gym, simulate_gym
from gym_sched.visual_server import save_custom_run


def test_equipment_counts_and_compatible_visits():
    draft = generate_gym(GymSettings(leg_press_count=3, cable_row_count=1, member_count=5))
    assert len(draft["scenario"]["equipment"]) == 4
    assert len(draft["scenario"]["members"]) == 5
    first = draft["scenario"]["members"][0]
    assert first["visits"][0]["equipment_ids"] == ["A1", "A2", "A3"]
    result = simulate_gym(scenario_data=draft["scenario"], config_data=draft["config"])
    assert result["result"]["status"] == "completed"
    assert not result["findings"]


def test_arrival_patterns_and_seed_repeatability():
    regular = generate_gym(GymSettings(member_count=4, arrival_seconds=7))
    assert [m["arrival_seconds"] for m in regular["scenario"]["members"]] == [0, 7, 14, 21]
    simultaneous = generate_gym(GymSettings(arrival_pattern="simultaneous"))
    assert all(m["arrival_seconds"] == 0 for m in simultaneous["scenario"]["members"])
    settings = GymSettings(arrival_pattern="uniform", arrival_seconds=100)
    random = generate_gym(settings)
    assert random == generate_gym(settings)
    assert all(0 <= m["arrival_seconds"] <= 100 for m in random["scenario"]["members"])
    assert random != generate_gym(settings.model_copy(update={"seed": 43}))


def test_single_kind_and_capacity_changes_waiting():
    settings = GymSettings(
        leg_press_count=0,
        cable_row_count=1,
        member_count=8,
        independent_count=0,
        arrival_pattern="simultaneous",
        refusal_percent=0,
    )
    one = generate_gym(settings)
    many = generate_gym(settings.model_copy(update={"cable_row_count": 8}))
    a = simulate_gym(one["scenario"], one["config"])
    b = simulate_gym(many["scenario"], many["config"])
    assert not a["findings"] and not b["findings"]
    assert sum(m["waiting_seconds"] for m in a["result"]["member_summaries"]) > 0
    assert sum(m["waiting_seconds"] for m in b["result"]["member_summaries"]) == 0


def test_custom_profile_and_config_saved_and_reproducible(tmp_path):
    draft = generate_gym(GymSettings(member_count=2, independent_count=0))
    draft["scenario"]["members"][0]["arrival_seconds"] = 15
    draft["scenario"]["members"][0]["acceptance_probability"] = 0
    draft["config"]["walking_seconds"] = 4
    payload = save_custom_run(tmp_path, draft["scenario"], draft["config"])
    directory = tmp_path / "runs" / payload["saved_run"]
    assert json.loads((directory / "scenario.json").read_text()) == payload["scenario"]
    assert json.loads((directory / "config.json").read_text()) == payload["config"]
    assert json.loads((tmp_path / "custom_latest.json").read_text()) == payload
    assert "/*CUSTOM_DATA*/" not in (directory / "replay.html").read_text()
    replay = simulate_gym(payload["scenario"], payload["config"])
    assert replay["result"]["events"] == payload["result"]["events"]
    assert any(e["event"] == "refused" for e in payload["result"]["events"])


def test_invalid_settings_and_run_limits():
    for settings in [
        {"member_count": 51},
        {"leg_press_count": 0, "cable_row_count": 0},
        {"member_count": 1, "independent_count": 2},
        {"arrival_pattern": "unknown"},
    ]:
        with pytest.raises(ValueError):
            GymSettings(**settings)
    draft = generate_gym(GymSettings())
    draft["config"]["max_events"] = 200001
    with pytest.raises(ValueError):
        simulate_gym(draft["scenario"], draft["config"])


def test_truncated_run_retains_audit_findings(tmp_path):
    draft = generate_gym(GymSettings())
    draft["config"]["max_time_seconds"] = 1
    payload = save_custom_run(tmp_path, draft["scenario"], draft["config"])
    assert payload["result"]["status"] != "completed"
    assert payload["findings"]
    assert (tmp_path / "runs" / payload["saved_run"] / "result.json").exists()
