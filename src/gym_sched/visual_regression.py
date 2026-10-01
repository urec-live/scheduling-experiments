"""Run the four visual mechanism checks without watching the browser."""

import argparse
import json
from pathlib import Path

from gym_sched.domain.models import RunResult
from gym_sched.evaluation.metrics import validate_result
from gym_sched.visual_demo import build_payload, export_demo

EXPECTATIONS = Path(__file__).resolve().parents[2] / "data/fixtures/visual_regression.json"


def check_run(result: dict, expected: dict) -> list[str]:
    problems = validate_result(RunResult.model_validate(result))
    members = {m["member_id"]: m for m in result["member_summaries"]}
    if set(members) != set(expected):
        problems.append("Member roster differs from expected roster")
    for member_id, fields in expected.items():
        member = members.get(member_id, {})
        for field, value in fields.items():
            actual = member.get(field)
            if actual != value:
                problems.append(f"{member_id}.{field}: expected {value!r}, got {actual!r}")
        if not member.get("completed"):
            problems.append(f"{member_id}: unfinished planned work")
    return problems


def run_suite(output: Path) -> dict:
    specifications = json.loads(EXPECTATIONS.read_text())
    output.mkdir(parents=True, exist_ok=True)
    # Copy the expected outcomes so each report retains its evaluation contract.
    (output / "expectations.json").write_text(json.dumps(specifications, indent=2) + "\n")
    checks = []
    for experiment, specification in specifications.items():
        directory = output / experiment
        directory.mkdir(exist_ok=True)
        try:
            payload = build_payload(experiment, specification["refusal_percent"])
            (directory / "payload.json").write_text(json.dumps(payload, indent=2) + "\n")
            export_demo(directory / "replay.html", payload)
            for condition, expected in specification["conditions"].items():
                problems = check_run(payload["runs"][condition], expected)
                checks.append(
                    {
                        "experiment": experiment,
                        "condition": condition,
                        "passed": not problems,
                        "problems": problems,
                        "replay": f"{experiment}/replay.html",
                    }
                )
        except (ValueError, RuntimeError, KeyError, TypeError, OSError, AssertionError) as error:
            # Retain failed experiments in the report and continue the other checks.
            checks.append(
                {
                    "experiment": experiment,
                    "condition": "build",
                    "passed": False,
                    "problems": [f"{type(error).__name__}: {error}"],
                }
            )
    report = {
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
        "scope": "Deterministic synthetic regression checks; not real-gym efficacy evidence.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Visual experiment regression report",
        "",
        report["scope"],
        "",
        "| Experiment | Condition | Result | Replay |",
        "|---|---|---|---|",
    ]
    for check in checks:
        replay = f"[Open]({check['replay']})" if "replay" in check else "Unavailable"
        lines.append(
            f"| {check['experiment']} | {check['condition']} | "
            f"{'PASS' if check['passed'] else 'FAIL'} | {replay} |"
        )
    for check in checks:
        if check["problems"]:
            lines.extend(["", f"## {check['experiment']} / {check['condition']}", ""])
            lines.extend(f"- {problem}" for problem in check["problems"])
    lines.extend(
        [
            "",
            (
                "Checks cover expected member outcomes, completed work, time accounting, "
                "capacity, and the existing event audit. Detailed mechanism tests remain in pytest."
            ),
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/visual-regression"))
    args = parser.parse_args()
    report = run_suite(args.output)
    passed = sum(c["passed"] for c in report["checks"])
    print(
        f"{passed}/{len(report['checks'])} checks passed. Report: {(args.output / 'report.md').resolve()}"
    )
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
