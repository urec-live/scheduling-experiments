"""Paired scenario-cluster inference. Positive improvements mean lower candidate cost."""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil

import numpy as np
from scipy.stats import norm


def paired_bootstrap(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    seed=731,
    resamples=2000,
    confidence=0.95,
) -> dict:
    b, c = np.asarray(baseline, dtype=float), np.asarray(candidate, dtype=float)
    if len(b) != len(c):
        raise ValueError("paired samples need equal lengths")
    if not 0 < confidence < 1 or resamples < 1:
        raise ValueError("invalid bootstrap configuration")
    if not (np.isfinite(b).all() and np.isfinite(c).all()):
        raise ValueError("bootstrap inputs must be finite")
    delta = b - c
    n = len(delta)
    out = {
        "n_pairs": n,
        "improvement": float(np.mean(delta)) if n else None,
        "ci_low": None,
        "ci_high": None,
        "confidence": confidence,
        "exploratory": n < 30,
        "relative_improvement": float(np.mean(delta) / np.mean(b))
        if n and np.mean(b) > 0
        else None,
    }
    if n >= 2:
        rng = np.random.default_rng(seed)
        draws = np.mean(delta[rng.integers(0, n, size=(resamples, n))], axis=1)
        alpha = (1 - confidence) / 2
        out["ci_low"], out["ci_high"] = map(float, np.quantile(draws, [alpha, 1 - alpha]))
    return out


def clustered_p95_bootstrap(
    baseline: Sequence[Sequence[float]],
    candidate: Sequence[Sequence[float]],
    *,
    seed=732,
    resamples=2000,
) -> dict:
    """Resample paired scenarios, then recompute pooled member p95 (never iid member rows)."""
    if len(baseline) != len(candidate):
        raise ValueError("cluster lists must be paired")
    clusters = [
        (np.asarray(b, dtype=float), np.asarray(c, dtype=float))
        for b, c in zip(baseline, candidate)
    ]
    if any(not len(b) or not len(c) for b, c in clusters):
        raise ValueError("p95 clusters must contain members")
    if any(not np.isfinite(b).all() or not np.isfinite(c).all() for b, c in clusters):
        raise ValueError("p95 clusters must be finite")
    n = len(clusters)
    if not n:
        return {
            "n_clusters": 0,
            "improvement": None,
            "ci_low": None,
            "ci_high": None,
            "exploratory": True,
        }

    def contrast(indices):
        return float(
            np.quantile(np.concatenate([clusters[i][0] for i in indices]), 0.95)
            - np.quantile(np.concatenate([clusters[i][1] for i in indices]), 0.95)
        )

    out = {
        "n_clusters": n,
        "improvement": contrast(range(n)),
        "ci_low": None,
        "ci_high": None,
        "exploratory": n < 30,
    }
    if n >= 2:
        rng = np.random.default_rng(seed)
        draws = [contrast(rng.integers(0, n, n)) for _ in range(resamples)]
        out["ci_low"], out["ci_high"] = map(float, np.quantile(draws, [0.025, 0.975]))
    return out


def pilot_sample_size(
    paired_differences: Sequence[float],
    target_half_width_seconds: float,
    *,
    alpha=0.05,
    minimum=100,
    maximum=500,
) -> dict:
    """Plan mean-contrast precision, not statistical power or tail-endpoint precision."""
    if target_half_width_seconds <= 0 or not 0 < alpha < 1:
        raise ValueError("half-width must be positive; alpha must be in (0, 1)")
    if minimum < 2 or maximum < minimum:
        raise ValueError("invalid sample size bounds")
    values = np.asarray(paired_differences, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("pilot inputs must be finite")
    if len(values) < 2:
        return {
            "pilot_n": len(values),
            "recommended_n": None,
            "reason": "At least two independent paired scenarios are needed.",
        }
    sd = float(np.std(values, ddof=1))
    raw = ceil((norm.ppf(1 - alpha / 2) * sd / target_half_width_seconds) ** 2)
    return {
        "pilot_n": len(values),
        "paired_sd_seconds": sd,
        "target_half_width_seconds": target_half_width_seconds,
        "alpha": alpha,
        "unclipped_n": raw,
        "exploratory_pilot": len(values) < 30,
        "recommended_n": min(max(raw, minimum), maximum),
        "exceeds_budget": raw > maximum,
        "warning": "Approximate mean-contrast planning only; does not power tail/subgroup endpoints.",
    }


def assess_guardrails(
    baseline: dict,
    candidate: dict,
    *,
    failed=False,
    max_tail_increase_fraction=0.05,
    max_completion_drop=0.01,
    min_wait_reduction_fraction=0.05,
    latency_budget_seconds=0.1,
) -> dict:
    """Descriptive checks, not a significance claim. Undefined denominators are never passes."""
    specs = {
        "mean_wait": (
            baseline.get("mean_wait_seconds"),
            candidate.get("mean_wait_seconds"),
            "reduction",
            min_wait_reduction_fraction,
        ),
        "tail_wait": (
            baseline.get("p95_wait_seconds"),
            candidate.get("p95_wait_seconds"),
            "relative",
            max_tail_increase_fraction,
        ),
        "completion_rate": (
            baseline.get("completion_rate"),
            candidate.get("completion_rate"),
            "lower",
            max_completion_drop,
        ),
        "decision_latency": (
            0,
            candidate.get("decision_latency_p95_seconds"),
            "absolute",
            latency_budget_seconds,
        ),
    }
    checks = {}
    for name, (b, c, mode, tolerance) in specs.items():
        if failed:
            checks[name] = "failed_episode"
        elif b is None or c is None or (mode in {"relative", "reduction"} and b <= 0):
            checks[name] = "undefined"
        else:
            if mode == "lower":
                passed = c >= b - tolerance
            elif mode == "relative":
                passed = c <= b * (1 + tolerance)
            elif mode == "reduction":
                passed = c <= b * (1 - tolerance)
            else:
                passed = c <= b + tolerance
            checks[name] = "pass" if passed else "fail"
    return {
        "checks": checks,
        "passes": all(v == "pass" for v in checks.values()),
        "interpretation": "Descriptive guardrails; require paired uncertainty and a frozen protocol.",
    }
