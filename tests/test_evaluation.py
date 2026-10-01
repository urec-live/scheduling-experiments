import json
from collections import defaultdict
from pathlib import Path

import pytest
import yaml

from gym_sched.cli import main
from gym_sched.domain import Scenario
from gym_sched.evaluation.artifacts import environment_manifest, read_json, write_json
from gym_sched.evaluation.experiments import (
    freeze_protocol,
    pilot_count,
    run_comparison,
    run_experiment,
    select_validation,
)
from gym_sched.evaluation.metrics import validate_result
from gym_sched.evaluation.reports import build_comparisons, load_runs, render_report
from gym_sched.evaluation.statistics import (
    assess_guardrails,
    clustered_p95_bootstrap,
    paired_bootstrap,
    pilot_sample_size,
)


@pytest.fixture
def hand():
    return Scenario.model_validate_json(
        (Path(__file__).parents[1] / "data/fixtures/three_members.json").read_text()
    )


def panel(tmp_path, **overrides):
    config = {
        "id": "unit-panel",
        "baseline": "fcfs",
        "bootstrap_resamples": 20,
        "scenario_configs": {
            "small": {
                "scenario": {
                    "member_count": 3,
                    "equipment_types": 1,
                    "units_per_type": 1,
                    "visits_per_member": 1,
                    "duration_seconds": [60],
                    "arrival_window_seconds": 120,
                    "arrival_pattern": "burst",
                }
            }
        },
        "policies": ["fcfs", "spt"],
        "splits": {"train": [1, 2], "validation": [101, 102], "test": [201, 202]},
        **overrides,
    }
    path = tmp_path / "panel.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_paired_bootstrap_exact_constant_difference_and_singleton():
    result = paired_bootstrap([10, 20, 30], [5, 15, 25], resamples=100)
    assert result["improvement"] == result["ci_low"] == result["ci_high"] == 5
    assert result["relative_improvement"] == 0.25
    assert paired_bootstrap([1], [0])["ci_low"] is None
    assert paired_bootstrap([0, 0], [0, 0])["relative_improvement"] is None
    with pytest.raises(ValueError):
        paired_bootstrap([1], [1, 2])
    with pytest.raises(ValueError):
        paired_bootstrap([float("nan")], [0])


def test_p95_resamples_clusters_not_members():
    b = [[1, 2, 3], [100, 101], [5, 20, 25, 30]]
    c = [[value - 1 for value in row] for row in b]
    result = clustered_p95_bootstrap(b, c, resamples=100)
    assert result["n_clusters"] == 3
    assert result["improvement"] == pytest.approx(1)
    assert result["ci_low"] == pytest.approx(1)
    assert result["ci_high"] == pytest.approx(1)


def test_precision_sample_size_bounds_and_cap():
    small = pilot_sample_size([1, 2, 3], 30)
    assert small["recommended_n"] == 100
    large = pilot_sample_size([-10000, 10000], 1)
    assert large["recommended_n"] == 500 and large["exceeds_budget"]
    assert "power" not in large
    assert pilot_sample_size([2], 30)["recommended_n"] is None


def test_guardrails_include_primary_effect_and_never_pass_failed_or_undefined():
    b = {
        "mean_wait_seconds": 100,
        "p95_wait_seconds": 200,
        "completion_rate": 1,
        "decision_latency_p95_seconds": 0.01,
    }
    c = {**b, "mean_wait_seconds": 94, "completion_rate": 0.99}
    assert assess_guardrails(b, c)["passes"]
    assert not assess_guardrails(b, {**c, "mean_wait_seconds": 96})["passes"]
    assert not assess_guardrails(b, c, failed=True)["passes"]
    assert not assess_guardrails({**b, "p95_wait_seconds": 0}, c)["passes"]
    assert not assess_guardrails(b, {**c, "decision_latency_p95_seconds": 0.2})["passes"]


def test_artifact_roundtrip_hashes_and_paired_report(tmp_path, hand):
    out = tmp_path / "comparison"
    manifest = run_comparison(hand, ["fcfs", "spt"], out, render=False)
    assert len(manifest["runs"]) == 2
    runs = load_runs(out)
    assert all(validate_result(r) == [] for _, r in runs)
    comparison = build_comparisons(runs)[0]
    assert comparison["metrics"]["mean_wait_seconds"]["improvement"] == pytest.approx(120)
    assert comparison["metrics"]["mean_wait_seconds"]["ci_low"] is None
    assert (out / "scenario.json").exists()
    assert (out / "policies/fcfs/events.jsonl").exists()
    assert (
        read_json(out / "policies/fcfs/manifest.json")["scenario_hash"] == runs[0][1].scenario_hash
    )
    report = render_report(out)
    assert report["failed_episodes"] == 0
    assert "280.000" in (out / "report.md").read_text()
    assert (out / "figures/timeline_00.svg").exists()
    with pytest.raises(ValueError, match="already"):
        run_comparison(hand, ["fcfs"], out, render=False)


def test_failed_runs_remain_in_comparison(tmp_path, hand):
    out = tmp_path / "failed"
    run_comparison(
        hand, ["fcfs", "spt"], out, simulation_config={"max_time_seconds": 5}, render=False
    )
    result = build_comparisons(load_runs(out))[0]
    assert result["n_pairs"] == result["failed_pairs"] == 1
    assert not result["guardrails"]["passes"]


def test_panel_freezes_all_scenarios_and_keeps_disjoint_splits(tmp_path):
    config = panel(tmp_path)
    output = tmp_path / "dev"
    manifest = run_experiment(config, split="validation", output=output, render=False)
    assert len(manifest["cases"]) == 2 and len(manifest["runs"]) == 4
    assert {case["seed"] for case in manifest["cases"]} == {101, 102}
    case_hashes = {case["case_id"]: case["scenario_hash"] for case in manifest["cases"]}
    policy_hashes = defaultdict(set)
    for run in manifest["runs"]:
        policy_hashes[run["case_id"]].add(run["scenario_hash"])
    assert policy_hashes == {
        case_id: {scenario_hash} for case_id, scenario_hash in case_hashes.items()
    }
    for case in manifest["cases"]:
        assert read_json(output / case["scenario_path"])["metadata"]["split"] == "validation"
    selection = select_validation(output)
    assert selection["selected_policy"] in {"fcfs", "spt"}
    counts = pilot_count(output, baseline="fcfs", candidate="spt", split="validation")
    assert counts["families"]["small"]["recommended_n"] == 100
    with pytest.raises(ValueError, match="requires --protocol"):
        run_experiment(config, split="test", output=tmp_path / "blocked", render=False)
    protocol_file = tmp_path / "protocol.json"
    freeze_protocol(config, protocol_file)
    test = run_experiment(
        config, split="test", output=tmp_path / "heldout", protocol_path=protocol_file, render=False
    )
    assert test["protocol_hash"]
    with pytest.raises(ValueError, match="without test"):
        select_validation(tmp_path / "heldout")
    changed = yaml.safe_load(config.read_text())
    changed["scenario_configs"]["small"]["scenario"]["member_count"] = 4
    config.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match="changed since"):
        run_experiment(
            config,
            split="test",
            output=tmp_path / "mutated",
            protocol_path=protocol_file,
            render=False,
        )


def test_panel_grid_accepts_simulation_parameters(tmp_path):
    config = panel(tmp_path, parameter_grid={"simulation.observation_delay_seconds": [0, 5]})
    manifest = run_experiment(config, split="train", output=tmp_path / "grid", render=False)
    assert len(manifest["cases"]) == 4
    assert {x["simulation"]["observation_delay_seconds"] for x in manifest["cases"]} == {0, 5}


def test_duplicate_split_seeds_and_tampered_protocol_rejected(tmp_path):
    config = panel(tmp_path, splits={"train": [1], "test": [1]})
    with pytest.raises(ValueError, match="disjoint"):
        freeze_protocol(config, tmp_path / "bad.json")
    config = panel(tmp_path)
    frozen = tmp_path / "good.json"
    freeze_protocol(config, frozen)
    data = read_json(frozen)
    data["primary_endpoint"] = "changed after freeze"
    write_json(frozen, data)
    with pytest.raises(ValueError, match="integrity"):
        run_experiment(
            config, split="test", output=tmp_path / "bad-run", protocol_path=frozen, render=False
        )


def test_cli_generate_compare_and_manifest(tmp_path, capsys):
    path = tmp_path / "input.yaml"
    path.write_text("scenario:\n  member_count: 2\n  visits_per_member: 1\n")
    scenario = tmp_path / "scenario.json"
    assert main(["generate", "--config", str(path), "--seed", "12", "--output", str(scenario)]) == 0
    out = tmp_path / "cli"
    assert (
        main(
            ["compare", "--scenario", str(scenario), "--policies", "fcfs,spt", "--output", str(out)]
        )
        == 0
    )
    assert read_json(out / "manifest.json")["status"] == "complete"
    with pytest.raises(SystemExit):
        main(["generate", "--config", str(path), "--seed", "12", "--output", str(scenario)])
    assert "already exists" in capsys.readouterr().err


def test_environment_manifest_has_source_fingerprint_without_credentials():
    manifest = environment_manifest()
    assert len(manifest["source_sha256"]) == 64
    assert "dependencies" in manifest
    assert "environment_variables" not in manifest
    json.dumps(manifest, allow_nan=False)
