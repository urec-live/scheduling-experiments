"""Local interactive experiment runner; never connects to production."""

import argparse
import json
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import uuid4

from gym_sched.configurable_gym import GymSettings, generate_gym, simulate_gym
from gym_sched.visual_demo import build_payload, export_demo
from gym_sched.visual_regression import run_suite


def save_run(
    root: Path, order: str, experiment: str = "order", refusal_percent: float = 50
) -> dict:
    if order not in {"fixed", "flexible"}:
        raise ValueError("Order must be fixed or flexible")
    payload = build_payload(experiment, refusal_percent)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    directory = root / "runs" / run_id
    directory.mkdir(parents=True)
    result = payload["runs"][order]
    for name, value in {
        "scenario.json": payload["scenarios"][order],
        "config.json": result["config"],
        "result.json": result,
        "manifest.json": {
            "order": order,
            "experiment": experiment,
            "refusal_percent": refusal_percent,
            "policy": "fcfs",
            "run_id": run_id,
        },
    }.items():
        (directory / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    payload.update(selected_order=order, saved_run=run_id)
    return payload


def save_custom_run(root: Path, scenario: dict, config: dict) -> dict:
    payload = simulate_gym(scenario, config)
    run_id = "custom-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    directory = root / "runs" / run_id
    directory.mkdir(parents=True)
    payload["saved_run"] = run_id
    for name, value in {
        "scenario": payload["scenario"],
        "config": payload["config"],
        "result": payload["result"],
        "payload": payload,
    }.items():
        (directory / f"{name}.json").write_text(json.dumps(value, indent=2, allow_nan=False))
    template = Path(__file__).with_name("configurable_gym.html").read_text()
    embedded = json.dumps(payload, allow_nan=False).replace("<", "\\u003c")
    (directory / "replay.html").write_text(template.replace("/*CUSTOM_DATA*/", embedded))
    temporary = root / f"{run_id}.tmp"
    temporary.write_text(json.dumps(payload, allow_nan=False))
    temporary.replace(root / "custom_latest.json")
    return payload


def make_handler(root: Path):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, body, content_type="application/json"):
            raw = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path in {"/", "/index.html"}:
                self.reply(200, (root / "index.html").read_text(), "text/html; charset=utf-8")
            elif self.path == "/configure":
                template = Path(__file__).with_name("configurable_gym.html").read_text()
                self.reply(
                    200, template.replace("/*CUSTOM_DATA*/", "null"), "text/html; charset=utf-8"
                )
            elif self.path in {
                f"/regression/{experiment}/replay.html"
                for experiment in ["order", "response", "competition", "race"]
            }:
                path = root / self.path.lstrip("/")
                if path.exists():
                    self.reply(200, path.read_text(), "text/html; charset=utf-8")
                else:
                    self.reply(404, json.dumps({"error": "Run preset checks first"}))
            elif self.path == "/api/custom/latest" and (root / "custom_latest.json").exists():
                self.reply(200, (root / "custom_latest.json").read_text())
            else:
                self.reply(404, json.dumps({"error": "Not found"}))

        def do_POST(self):
            if self.path not in {
                "/api/run",
                "/api/custom/generate",
                "/api/custom/run",
                "/api/regression",
            }:
                self.reply(404, json.dumps({"error": "Not found"}))
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1000000:
                    raise ValueError("Invalid request size")
                request = json.loads(self.rfile.read(length))
                if self.path == "/api/regression":
                    if request != {}:
                        raise ValueError("Preset regression checks use their fixed configurations")
                    self.reply(200, json.dumps(run_suite(root / "regression")))
                    return
                if self.path == "/api/custom/generate":
                    payload = generate_gym(GymSettings.model_validate(request))
                    self.reply(200, json.dumps(payload, allow_nan=False))
                    return
                if self.path == "/api/custom/run":
                    if not isinstance(request, dict) or set(request) != {"scenario", "config"}:
                        raise ValueError("Provide scenario and config")
                    payload = save_custom_run(root, request["scenario"], request["config"])
                    self.reply(200, json.dumps(payload, allow_nan=False))
                    return
                if not isinstance(request, dict) or set(request) not in (
                    {"order"},
                    {"order", "experiment"},
                    {"order", "experiment", "refusal_percent"},
                ):
                    raise ValueError("Provide the condition and optional experiment")
                payload = save_run(
                    root,
                    request["order"],
                    request.get("experiment", "order"),
                    request.get("refusal_percent", 50),
                )
            except (ValueError, TypeError) as error:
                self.reply(400, json.dumps({"error": str(error)}))
                return
            self.reply(200, json.dumps(payload, allow_nan=False))

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output", type=Path, default=Path("outputs/visual-gym"))
    args = parser.parse_args()
    export_demo(args.output / "index.html")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args.output))
    print(f"Interactive gym: http://127.0.0.1:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
