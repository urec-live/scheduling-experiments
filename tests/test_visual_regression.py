"""The regression runner must detect faults and preserve inspectable evidence."""

import json

from gym_sched.visual_demo import build_payload
from gym_sched.visual_regression import EXPECTATIONS, check_run, run_suite


def test_suite_saves_all_conditions_and_replays(tmp_path):
    report = run_suite(tmp_path)
    assert report["passed"]
    assert len(report["checks"]) == 8
    assert json.loads((tmp_path / "report.json").read_text()) == report
    for experiment in ["order", "response", "competition", "race"]:
        assert (tmp_path / experiment / "payload.json").exists()
        assert (tmp_path / experiment / "replay.html").exists()
    assert (tmp_path / "expectations.json").exists()


def test_checker_detects_wrong_outcome_and_double_occupancy():
    expected = json.loads(EXPECTATIONS.read_text())["race"]["conditions"]["flexible"]
    result = build_payload("race")["runs"]["flexible"]
    result["member_summaries"][0]["waiting_seconds"] = 999
    start = next(e for e in result["events"] if e["event"] == "exercise_started")
    result["events"].insert(result["events"].index(start) + 1, dict(start))
    findings = check_run(result, expected)
    assert any("expected" in p for p in findings)
    assert len(findings) > 1


def test_suite_records_build_failure_and_continues(tmp_path, monkeypatch):
    from gym_sched import visual_regression

    original = visual_regression.build_payload

    def build(experiment, probability):
        if experiment == "response":
            raise ValueError("Deliberate regression")
        return original(experiment, probability)

    monkeypatch.setattr(visual_regression, "build_payload", build)
    report = run_suite(tmp_path)
    assert not report["passed"]
    assert any(c["experiment"] == "race" and c["passed"] for c in report["checks"])
    assert "Deliberate regression" in (tmp_path / "report.md").read_text()
