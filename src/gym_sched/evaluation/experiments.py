"""Reproducible split-aware panels, prospective protocol freezes, and validation-only selection."""

from __future__ import annotations

import re
from copy import deepcopy
from itertools import product
from pathlib import Path

import numpy as np
import yaml

from gym_sched.domain.models import Scenario, SimulationConfig

from .artifacts import content_hash, environment_manifest, read_json, save_run, write_json
from .metrics import validate_result
from .reports import load_runs, render_report
from .statistics import pilot_sample_size


def load_yaml(path: Path | str) -> dict:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise TypeError(f"Expected a YAML mapping: {path}")
    return data


def resolve_mapping(value, base: Path) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return deepcopy(value)
    return load_yaml((base / str(value)).resolve())


def safe_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not result:
        raise ValueError("artifact names cannot be empty")
    return result


def _expanded_config(config_path: Path | str):
    config_path = Path(config_path).resolve()
    config = load_yaml(config_path)
    base = config_path.parent
    families = config.get("scenario_configs")
    if not isinstance(families, dict) or not families:
        raise ValueError("experiment needs a nonempty scenario_configs mapping")
    resolved = deepcopy(config)
    resolved["scenario_configs"] = {
        name: resolve_mapping(source, base) for name, source in families.items()
    }
    resolved["policy_config"] = resolve_mapping(config.get("policy_config"), base)
    resolved["simulation_config"] = resolve_mapping(config.get("simulation_config"), base)
    splits = resolved.get("splits", {})
    if not splits or set(splits) - {"train", "validation", "test"}:
        raise ValueError("splits must use train, validation, and/or test")
    seen = set()
    for split, seeds in splits.items():
        if not seeds or not isinstance(seeds, list) or any(type(seed) is not int for seed in seeds):
            raise ValueError(f"{split} must contain integer seeds")
        if len(set(seeds)) != len(seeds) or seen.intersection(seeds):
            raise ValueError("Seeds must be unique within and disjoint across splits")
        seen.update(seeds)
    policies = resolved.get("policies", [])
    if not policies or len(set(policies)) != len(policies):
        raise ValueError("policies must be a nonempty list of unique names")
    if any(safe_name(name) != name for name in policies):
        raise ValueError("Policy names must be safe artifact identifiers")
    for key, choices in resolved.get("parameter_grid", {}).items():
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"parameter_grid.{key} needs a nonempty list")
    return resolved


def freeze_protocol(config_path, output, *, selection_path=None) -> dict:
    output = Path(output)
    if output.exists():
        raise ValueError("A frozen protocol cannot be overwritten; use a new protocol name")
    config = _expanded_config(config_path)
    selection = read_json(selection_path) if selection_path else None
    if selection is not None and (
        selection.get("split") != "validation" or not selection.get("selected_policy")
    ):
        raise ValueError("Freeze requires a successful validation-only selection")
    if selection and config.get("baseline", "fcfs") != selection["selected_policy"]:
        raise ValueError("Set the experiment baseline to selected_policy before freezing")
    protocol = {
        "schema_version": "1.0",
        "kind": "frozen_protocol",
        "resolved_experiment": config,
        "experiment_hash": content_hash(config),
        "primary_endpoint": "paired scenario-level mean waiting_seconds (all arrivals)",
        "secondary_endpoints": [
            "p95_wait_seconds",
            "completion_rate",
            "on_time_rate",
            "changes_per_member",
        ],
        "bootstrap_seed": 731,
        "bootstrap_resamples": config.get("bootstrap_resamples", 2000),
        "guardrails": {
            "min_wait_reduction_fraction": 0.05,
            "max_tail_increase_fraction": 0.05,
            "max_completion_drop": 0.01,
            "latency_budget_seconds": config.get("latency_budget_seconds", 0.1),
        },
        "selection": selection,
        "environment": environment_manifest(),
        "disclosure": "Synthetic evaluation. No claim of human gym benefit or algorithm novelty.",
    }
    protocol["protocol_hash"] = content_hash(protocol)
    write_json(output, protocol)
    return protocol


def verify_protocol(protocol_path, config):
    protocol = read_json(protocol_path)
    claimed_hash = protocol.pop("protocol_hash", None)
    if not claimed_hash or content_hash(protocol) != claimed_hash:
        raise ValueError("Protocol integrity check failed")
    if protocol.get("experiment_hash") != content_hash(config):
        raise ValueError("Experiment/configs changed since protocol freeze")
    if (
        protocol.get("environment", {}).get("source_sha256")
        != environment_manifest()["source_sha256"]
    ):
        raise ValueError("Source changed since protocol freeze; create a new prospective protocol")
    return claimed_hash


def _variants(config):
    grid = config.get("parameter_grid", {})
    names = sorted(grid)
    combinations = product(*(grid[key] for key in names)) if names else [()]
    for values in combinations:
        overrides = dict(zip(names, values))
        tag = "base" if not names else content_hash(overrides)[:8]
        yield tag, overrides


def _set_dotted(target, key, value):
    parts = key.split(".")
    current = target
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            raise ValueError(f"Unknown grid config path: {key}")
        current = current[part]
    if parts[-1] not in current:
        raise ValueError(f"Unknown grid config key: {key}")
    current[parts[-1]] = value


def run_comparison(
    scenario: Scenario,
    policy_names,
    directory,
    *,
    simulation_config=None,
    policy_options=None,
    baseline="fcfs",
    render=True,
):
    from gym_sched.policies import create_policy
    from gym_sched.simulation.engine import run_episode

    directory = Path(directory)
    if (directory / "manifest.json").exists():
        raise ValueError(f"Output already contains a run: {directory}")
    if len(set(policy_names)) != len(policy_names):
        raise ValueError("Policy names must be unique")
    directory.mkdir(parents=True, exist_ok=True)
    config = SimulationConfig.model_validate(simulation_config or {})
    if baseline not in policy_names:
        raise ValueError("Comparison baseline must be included in --policies")
    # Freeze exact exogenous scenario before invoking even the first policy.
    write_json(directory / "scenario.json", scenario)
    manifest = {
        **environment_manifest(),
        "kind": "comparison",
        "status": "running",
        "scenario_hash": content_hash(scenario),
        "config_hash": content_hash(config),
        "policy_config": policy_options or {},
        "baseline": baseline,
        "runs": [],
    }
    write_json(directory / "manifest.json", manifest)
    for name in policy_names:
        if safe_name(name) != name:
            raise ValueError("Unsafe policy name")
        policy = create_policy(name, (policy_options or {}).get(name, {}))
        result = run_episode(scenario, policy, config)
        path = Path("policies") / name
        save_run(result, directory / path, policy_options=getattr(policy, "options", {}))
        manifest["runs"].append(
            {
                "path": str(path),
                "policy": name,
                "split": "exploratory",
                "family": "single",
                "scenario_id": scenario.id,
            }
        )
        write_json(directory / "manifest.json", manifest)
    manifest["status"] = "complete"
    write_json(directory / "manifest.json", manifest)
    if render:
        render_report(directory, baseline=baseline)
    return manifest


def run_experiment(config_path, *, split=None, output=None, protocol_path=None, render=True):
    from gym_sched.policies import create_policy
    from gym_sched.scenarios.generate import generate_scenario
    from gym_sched.simulation.engine import run_episode

    config = _expanded_config(config_path)
    selected_splits = [split] if split else [s for s in config["splits"] if s != "test"]
    if not selected_splits or any(s not in config["splits"] for s in selected_splits):
        raise ValueError("Requested split is not configured")
    if "test" in selected_splits and not protocol_path:
        raise ValueError("Test split requires --protocol from freeze-protocol")
    protocol_hash = verify_protocol(protocol_path, config) if protocol_path else None
    directory = Path(
        output
        or config.get("output", f"outputs/experiments/{safe_name(config.get('id', 'panel'))}")
    )
    if (directory / "manifest.json").exists():
        raise ValueError(f"Output already contains an experiment: {directory}; choose --output")
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "resolved_config.json", config)
    cases = []
    variants = list(_variants(config))
    for selected_split in selected_splits:
        for family, source in config["scenario_configs"].items():
            for tag, overrides in variants:
                scenario_config = deepcopy(source.get("scenario", source))
                simulation = config["simulation_config"] or source.get("simulation", {})
                simulation = simulation.get("simulation", simulation)
                from gym_sched.scenarios.generate import GeneratorConfig

                combined = {
                    "scenario": GeneratorConfig.model_validate(scenario_config).model_dump(),
                    "simulation": SimulationConfig.model_validate(simulation).model_dump(),
                }
                for key, value in overrides.items():
                    _set_dotted(
                        combined,
                        key if key.startswith(("scenario.", "simulation.")) else f"scenario.{key}",
                        value,
                    )
                scenario_config = combined["scenario"]
                scenario_config["metadata"] = {
                    **scenario_config.get("metadata", {}),
                    "split": selected_split,
                    "family": family,
                }
                simulation = SimulationConfig.model_validate(combined["simulation"])
                for seed in config["splits"][selected_split]:
                    scenario = generate_scenario(scenario_config, seed)
                    case_id = f"{selected_split}_{safe_name(family)}_{tag}_{seed}"
                    path = Path("scenarios") / f"{case_id}.json"
                    write_json(directory / path, scenario)
                    cases.append(
                        {
                            "case_id": case_id,
                            "split": selected_split,
                            "family": family if tag == "base" else f"{family}/{tag}",
                            "scenario_path": str(path),
                            "scenario_hash": content_hash(scenario),
                            "seed": seed,
                            "overrides": overrides,
                            "simulation": simulation.model_dump(),
                        }
                    )
    manifest = {
        **environment_manifest(),
        "kind": "experiment",
        "status": "running",
        "experiment_id": config.get("id", "panel"),
        "experiment_hash": content_hash(config),
        "protocol_hash": protocol_hash,
        "planned_episodes": len(cases) * len(config["policies"]),
        "cases": cases,
        "runs": [],
        "baseline": config.get("baseline", "fcfs"),
        "bootstrap_resamples": config.get("bootstrap_resamples", 2000),
        "latency_budget_seconds": config.get("latency_budget_seconds", 0.1),
    }
    write_json(directory / "manifest.json", manifest)
    # All scenarios above are saved before any policy executes. Never regenerate per policy.
    for case in cases:
        scenario = Scenario.model_validate(read_json(directory / case["scenario_path"]))
        for name in config["policies"]:
            policy = create_policy(name, config["policy_config"].get(name, {}))
            result = run_episode(
                scenario, policy, SimulationConfig.model_validate(case["simulation"])
            )
            path = Path("runs") / case["case_id"] / name
            save_run(result, directory / path, policy_options=getattr(policy, "options", {}))
            manifest["runs"].append(
                {
                    "path": str(path),
                    "policy": name,
                    "split": case["split"],
                    "family": case["family"],
                    "scenario_id": scenario.id,
                    "scenario_hash": case["scenario_hash"],
                    "seed": case["seed"],
                    "case_id": case["case_id"],
                }
            )
            write_json(directory / "manifest.json", manifest)
    manifest["status"] = "complete"
    write_json(directory / "manifest.json", manifest)
    if render:
        render_report(
            directory,
            baseline=config.get("baseline", "fcfs"),
            bootstrap_resamples=config.get("bootstrap_resamples", 2000),
        )
    return manifest


def select_validation(directory, *, baseline="fcfs", candidates=None):
    runs = load_runs(directory)
    if any(e.get("split") == "test" for e, _ in runs):
        raise ValueError("Selection must use an artifact directory without test outcomes")
    runs = [(e, r) for e, r in runs if e.get("split") == "validation"]
    if not runs:
        raise ValueError("No validation episodes found")
    from .reports import build_comparisons

    comparisons = build_comparisons(runs, baseline, bootstrap_resamples=1000)
    available = {r.policy for _, r in runs}
    # Select a reference, not the proposed candidate method. FCFS can legitimately
    # remain strongest; improvement over it is not a requirement for selection.
    candidates = candidates or sorted(available & {"fcfs", "spt", "aging", "matching", "cpsat"})
    if not set(candidates) <= available:
        raise ValueError("Requested validation candidate was not evaluated")
    scores = []
    families = {e["family"] for e, _ in runs}
    for name in candidates:
        panels = [x for x in comparisons if x["candidate"] == name]
        selected_runs = [(e, r) for e, r in runs if r.policy == name]
        family_values = {
            family: [
                r.metrics["mean_wait_seconds"] for e, r in selected_runs if e["family"] == family
            ]
            for family in families
        }
        expected = {(e["family"], r.scenario_hash) for e, r in runs if r.policy == baseline}
        actual = {(e["family"], r.scenario_hash) for e, r in selected_runs}
        eligible = (
            bool(expected)
            and actual == expected
            and not any(validate_result(r) for _, r in selected_runs)
            and all(family_values.values())
        )
        latency_budget = read_json(Path(directory) / "manifest.json").get(
            "latency_budget_seconds", 0.1
        )
        for family in families:
            b_rows = [r for e, r in runs if e["family"] == family and r.policy == baseline]
            c_rows = [r for e, r in selected_runs if e["family"] == family]
            if not b_rows or not c_rows:
                eligible = False
                continue
            for metric, tolerance, lower in (
                ("completion_rate", 0.01, True),
                ("p95_wait_seconds", 0.05, False),
            ):
                b = float(np.mean([r.metrics[metric] for r in b_rows]))
                c = float(np.mean([r.metrics[metric] for r in c_rows]))
                eligible &= c >= b - tolerance if lower else c <= b * (1 + tolerance)
            latencies = [r.metrics.get("decision_latency_p95_seconds") for r in c_rows]
            eligible &= (
                all(x is not None for x in latencies)
                and float(np.mean(latencies)) <= latency_budget
            )
        cost = float(np.mean([np.mean(v) for v in family_values.values()])) if eligible else None
        improvement = (
            float(np.mean([x["metrics"]["mean_wait_seconds"]["improvement"] for x in panels]))
            if panels
            else 0.0
            if name == baseline
            else None
        )
        scores.append(
            {
                "policy": name,
                "eligible": eligible,
                "mean_family_improvement_seconds": improvement,
                "mean_family_wait_seconds": cost,
                "failed_pairs": sum(x["failed_pairs"] for x in panels),
            }
        )
    eligible = sorted(
        [s for s in scores if s["eligible"]],
        key=lambda s: (s["mean_family_wait_seconds"], s["policy"]),
    )
    selection = {
        "split": "validation",
        "baseline": baseline,
        "selected_policy": eligible[0]["policy"] if eligible else None,
        "criterion": "Minimum equal-family mean waiting among paired valid references preserving completion/tail/latency margins; ties lexical",
        "scores": scores,
        "validation_manifest_hash": content_hash(read_json(Path(directory) / "manifest.json")),
        "warning": "Selection is exploratory; a frozen held-out test is required for confirmatory evaluation.",
    }
    write_json(Path(directory) / "selection.json", selection)
    return selection


def pilot_count(directory, *, baseline, candidate, half_width_seconds=None, split="train"):
    if half_width_seconds is not None and half_width_seconds <= 0:
        raise ValueError("half-width must be positive")
    paired = {}
    for entry, result in load_runs(directory):
        if entry.get("split") == split and result.policy in (baseline, candidate):
            paired.setdefault((entry.get("family"), result.scenario_hash), {})[result.policy] = (
                result
            )
    families = {}
    for (family, _), policies in paired.items():
        rows = families.setdefault(family, {"differences": [], "baseline": [], "failures": 0})
        if baseline in policies and candidate in policies:
            b, c = policies[baseline], policies[candidate]
            if validate_result(b) or validate_result(c):
                rows["failures"] += 1
            else:
                rows["differences"].append(
                    b.metrics["mean_wait_seconds"] - c.metrics["mean_wait_seconds"]
                )
                rows["baseline"].append(b.metrics["mean_wait_seconds"])
    results = {}
    for family, rows in families.items():
        if rows["failures"]:
            results[family] = {
                "recommended_n": None,
                "failed_pairs": rows["failures"],
                "reason": "Resolve failed pilot episodes first.",
            }
        else:
            h = (
                half_width_seconds
                if half_width_seconds is not None
                else (max(30, 0.05 * float(np.mean(rows["baseline"]))) if rows["baseline"] else 30)
            )
            results[family] = pilot_sample_size(rows["differences"], h)
    return {"split": split, "baseline": baseline, "candidate": candidate, "families": results}
