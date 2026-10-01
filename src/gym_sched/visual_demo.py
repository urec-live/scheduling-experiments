"""Export a self-contained, event-driven replay of a tiny scheduling experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gym_sched.domain.models import Scenario, SimulationConfig
from gym_sched.evaluation.metrics import validate_result
from gym_sched.policies import create_policy
from gym_sched.simulation.engine import run_episode


def demo_scenario(competition: bool = False) -> Scenario:
    """One fixed workload; only the run's reordering permission changes."""
    filename = "visual_three_members.json" if competition else "visual_two_members.json"
    path = Path(__file__).resolve().parents[2] / "data/fixtures" / filename
    return Scenario.model_validate_json(path.read_text())


def build_payload(experiment: str = "order", refusal_percent: float = 50) -> dict:
    if experiment not in {"order", "response", "competition", "race"}:
        raise ValueError("Unknown experiment")
    if (
        isinstance(refusal_percent, bool)
        or not isinstance(refusal_percent, (int, float))
        or not 0 <= refusal_percent <= 100
    ):
        raise ValueError("Refusal percentage must be a number from 0 to 100")
    base = demo_scenario(experiment == "competition")
    if experiment == "race":
        path = (
            Path(__file__).resolve().parents[2] / "data/fixtures/visual_simultaneous_arrivals.json"
        )
        base = Scenario.model_validate_json(path.read_text())
    scenarios = {}
    runs = {}
    for key, reorder in [("fixed", False), ("flexible", True)]:
        scenario = base.model_copy(deep=True)
        config = SimulationConfig(allow_reordering=reorder)
        if experiment in {"response", "competition"}:
            casey = next(m for m in scenario.members if m.id == "casey")
            casey.independent_keep_plan_order = True
            casey.acceptance_probability = 1 if key == "fixed" else 1 - refusal_percent / 100
            casey.preference_change_seconds = None
            casey.changed_acceptance_probability = None
            config = SimulationConfig(allow_reordering=True, response_delay_seconds=1)
        if experiment == "race":
            config = SimulationConfig(
                allow_reordering=True,
                response_delay_seconds=1,
                walking_seconds=0 if key == "fixed" else 10,
            )
        scenarios[key] = scenario.model_dump(mode="json")
        result = run_episode(
            scenario,
            create_policy("fcfs"),
            config,
        )
        problems = validate_result(result)
        if problems:
            raise ValueError(f"Invalid visual demo: {problems}")
        runs[key] = result.model_dump(mode="json")
    return {
        "scenario": scenarios["fixed"],
        "scenarios": scenarios,
        "runs": runs,
        "experiment": experiment,
        "refusal_percent": refusal_percent,
    }


def export_demo(output: Path, payload: dict | None = None) -> Path:
    template = Path(__file__).with_name("visual_demo.html").read_text()
    payload = json.dumps(build_payload() if payload is None else payload, allow_nan=False).replace(
        "<", "\\u003c"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace("/*__PAYLOAD__*/", payload))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/visual-gym/index.html"))
    args = parser.parse_args()
    print(export_demo(args.output).resolve())


if __name__ == "__main__":
    main()
