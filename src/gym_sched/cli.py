"""Command-line research workflow. Scenarios are always saved before policy evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from gym_sched.domain.models import Scenario
from gym_sched.evaluation.artifacts import canonical_json, read_json, write_json
from gym_sched.evaluation.experiments import (
    freeze_protocol,
    load_yaml,
    pilot_count,
    run_comparison,
    run_experiment,
    select_validation,
)
from gym_sched.evaluation.reports import render_report


def main(argv=None):
    parser = argparse.ArgumentParser(prog="gym-sched", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="Freeze one synthetic exogenous scenario")
    generate.add_argument("--config", required=True)
    generate.add_argument("--seed", required=True, type=int)
    generate.add_argument("--output")
    compare = commands.add_parser("compare", help="Run paired policies on one saved scenario")
    compare.add_argument("--scenario", required=True)
    compare.add_argument("--policies", default="self_directed,fcfs,spt,aging,matching,robust")
    compare.add_argument("--config", help="Simulation YAML, optionally nested under simulation:")
    compare.add_argument("--policy-config", help="YAML mapping policy name to options")
    compare.add_argument("--output")
    compare.add_argument("--baseline", default="fcfs")
    report = commands.add_parser("report", help="Rebuild reports from saved results")
    report.add_argument("--run", required=True)
    report.add_argument("--baseline")
    experiment = commands.add_parser("experiment", help="Run a split-aware scenario panel")
    experiment.add_argument("--config", required=True)
    experiment.add_argument("--split", choices=["train", "validation", "test"])
    experiment.add_argument("--output")
    experiment.add_argument("--protocol", help="Frozen protocol required for test split")
    freeze = commands.add_parser(
        "freeze-protocol", help="Prospectively freeze configs, seeds and source fingerprint"
    )
    freeze.add_argument("--config", required=True)
    freeze.add_argument("--output", required=True)
    freeze.add_argument("--selection", help="Optional validation selection.json")
    select = commands.add_parser(
        "select-validation", help="Choose a candidate using validation episodes only"
    )
    select.add_argument("--run", required=True)
    select.add_argument("--baseline", default="fcfs")
    select.add_argument("--candidates", help="Comma-separated policy names")
    pilot = commands.add_parser(
        "pilot-count", help="Estimate independent scenario count from pilot paired variance"
    )
    pilot.add_argument("--run", required=True)
    pilot.add_argument("--baseline", default="fcfs")
    pilot.add_argument("--candidate", required=True)
    pilot.add_argument(
        "--half-width-seconds",
        type=float,
        help="Target 95%% interval half-width; default max(30s, 5%% baseline mean) per family",
    )
    pilot.add_argument("--split", default="train", choices=["train", "validation"])
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            from gym_sched.scenarios.generate import generate_scenario

            config = load_yaml(args.config)
            scenario = generate_scenario(config.get("scenario", config), args.seed)
            from gym_sched.evaluation.experiments import safe_name

            path = Path(args.output or f"outputs/scenarios/{safe_name(scenario.id)}.json")
            if path.exists():
                raise ValueError(f"Scenario already exists: {path}; choose another --output")
            write_json(path, scenario)
            output = {
                "scenario": str(path),
                "scenario_id": scenario.id,
                "members": len(scenario.members),
            }
        elif args.command == "compare":
            scenario = Scenario.model_validate(read_json(args.scenario))
            config = load_yaml(args.config) if args.config else {}
            options = load_yaml(args.policy_config) if args.policy_config else {}
            from gym_sched.evaluation.experiments import safe_name

            directory = args.output or f"outputs/runs/{safe_name(scenario.id)}"
            result = run_comparison(
                scenario,
                [x.strip() for x in args.policies.split(",") if x.strip()],
                directory,
                simulation_config=config.get("simulation", config),
                policy_options=options,
                baseline=args.baseline,
            )
            output = {"output": directory, "episodes": len(result["runs"])}
        elif args.command == "report":
            baseline = args.baseline or read_json(Path(args.run) / "manifest.json").get(
                "baseline", "fcfs"
            )
            output = render_report(args.run, baseline=baseline)
        elif args.command == "experiment":
            manifest = run_experiment(
                args.config, split=args.split, output=args.output, protocol_path=args.protocol
            )
            output = {
                "experiment_id": manifest["experiment_id"],
                "episodes": len(manifest["runs"]),
                "status": manifest["status"],
            }
        elif args.command == "freeze-protocol":
            protocol = freeze_protocol(args.config, args.output, selection_path=args.selection)
            output = {"protocol": args.output, "protocol_hash": protocol["protocol_hash"]}
        elif args.command == "select-validation":
            output = select_validation(
                args.run,
                baseline=args.baseline,
                candidates=args.candidates.split(",") if args.candidates else None,
            )
        else:
            output = pilot_count(
                args.run,
                baseline=args.baseline,
                candidate=args.candidate,
                half_width_seconds=args.half_width_seconds,
                split=args.split,
            )
        print(canonical_json(output))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
