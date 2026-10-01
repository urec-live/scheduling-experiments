"""Portable, inspectable run artifacts. No environment variables or credentials are collected."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .metrics import compute_metrics, validate_result


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def canonical_json(value) -> str:
    return json.dumps(
        value, default=json_default, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def content_hash(value) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def write_json(path: Path | str, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, default=json_default, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def read_json(path: Path | str):
    return json.loads(Path(path).read_text())


def _git(root: Path, *args):
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(root), *args], stderr=subprocess.DEVNULL, text=True, timeout=5
            ).strip()
            or None
        )
    except (subprocess.SubprocessError, OSError):
        return None


def environment_manifest(root: Path | None = None) -> dict:
    root = root or Path(__file__).resolve().parents[3]
    packages = {}
    for name in (
        "urec-scheduling-experiments",
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "simpy",
        "pydantic",
        "pyyaml",
        "ortools",
        "pymoo",
        "gymnasium",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    digest = hashlib.sha256()
    for path in sorted((root / "src").rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    for name in ("pyproject.toml", "uv.lock"):
        path = root / name
        if path.exists():
            digest.update(name.encode())
            digest.update(path.read_bytes())
    git_status = _git(root, "status", "--porcelain")
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "dependencies": packages,
        "git_revision": _git(root, "rev-parse", "HEAD"),
        "git_dirty": bool(git_status),
        "source_sha256": digest.hexdigest(),
        "data_origin": "synthetic",
        "schema_version": "1.0",
    }


def _write_csv(path, rows):
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: canonical_json(v) if isinstance(v, (dict, list)) else v for k, v in row.items()}
            )


def _write_jsonl(path, rows):
    path.write_text("".join(canonical_json(row) + "\n" for row in rows))


def save_run(result, directory: Path | str, *, scenario=None, policy_options=None) -> dict:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if not result.metrics:
        result.metrics = compute_metrics(result)
    findings = validate_result(result)
    write_json(directory / "result.json", result)
    _write_jsonl(directory / "events.jsonl", result.events)
    _write_jsonl(directory / "decisions.jsonl", result.decisions)
    _write_csv(directory / "members.csv", result.member_summaries)
    _write_csv(directory / "equipment.csv", result.equipment_summaries)
    options = policy_options or {}
    manifest = {
        **environment_manifest(),
        "kind": "episode",
        "scenario_id": result.scenario_id,
        "scenario_hash": result.scenario_hash,
        "policy": result.policy,
        "policy_options": options,
        "policy_options_hash": content_hash(options),
        "seed": result.seed,
        "simulation_config": result.config,
        "config_hash": content_hash(result.config),
        "status": result.status,
        "audit_findings": findings,
        "result_hash": content_hash(result),
    }
    if scenario is not None:
        write_json(directory / "scenario.json", scenario)
    write_json(directory / "manifest.json", manifest)
    return manifest
