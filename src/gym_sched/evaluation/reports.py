"""Readable reports and deterministic plots from saved artifacts, without rerunning policies."""

from __future__ import annotations

import html
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from gym_sched.domain.models import RunResult

from .artifacts import content_hash, read_json, write_json
from .metrics import validate_result
from .statistics import assess_guardrails, clustered_p95_bootstrap, paired_bootstrap


def load_runs(directory: Path | str) -> list[tuple[dict, RunResult]]:
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    entries = (
        [{"split": "exploratory", "family": "single", "path": "."}]
        if manifest.get("kind") == "episode"
        else manifest.get("runs", [])
    )
    runs = []
    for entry in entries:
        path = directory / entry["path"]
        episode_manifest = read_json(path / "manifest.json")
        result = RunResult.model_validate(read_json(path / "result.json"))
        if episode_manifest.get("result_hash") != content_hash(result):
            raise ValueError(f"Saved result integrity check failed: {path}")
        runs.append((entry, result))
    return runs


def build_comparisons(runs, baseline="fcfs", bootstrap_resamples=2000, latency_budget_seconds=0.1):
    grouped = defaultdict(dict)
    for entry, result in runs:
        key = (
            entry.get("split", "exploratory"),
            entry.get("family", "single"),
            result.scenario_hash,
        )
        if result.policy in grouped[key]:
            raise ValueError(f"Duplicate policy/scenario pair: {key}, {result.policy}")
        grouped[key][result.policy] = result
    panels = defaultdict(list)
    for (split, family, _), policies in grouped.items():
        if baseline not in policies:
            continue
        for policy, candidate in policies.items():
            if policy != baseline:
                panels[(split, family, policy)].append((policies[baseline], candidate))
    summaries = []
    for (split, family, policy), pairs in sorted(panels.items()):
        failures = sum(bool(validate_result(b) or validate_result(c)) for b, c in pairs)
        item = {
            "split": split,
            "family": family,
            "baseline": baseline,
            "candidate": policy,
            "n_pairs": len(pairs),
            "failed_pairs": failures,
            "adequate_replications_for_descriptive_summary": failures == 0 and len(pairs) >= 30,
            "metrics": {},
        }
        for metric in (
            "mean_wait_seconds",
            "p95_wait_seconds",
            "completion_rate",
            "on_time_rate",
            "changes_per_member",
            "mean_deviation",
            "decision_latency_p95_seconds",
        ):
            valid = [
                (b.metrics.get(metric), c.metrics.get(metric))
                for b, c in pairs
                if b.metrics.get(metric) is not None and c.metrics.get(metric) is not None
            ]
            statistics = paired_bootstrap(
                [b for b, _ in valid], [c for _, c in valid], resamples=bootstrap_resamples
            )
            statistics["undefined_pairs"] = len(pairs) - len(valid)
            statistics["direction"] = (
                "positive favors baseline"
                if metric.endswith("rate")
                else "positive favors candidate"
            )
            item["metrics"][metric] = statistics
        item["pooled_member_p95_cluster_bootstrap"] = clustered_p95_bootstrap(
            [[m.get("waiting_seconds", 0) for m in b.member_summaries] for b, _ in pairs],
            [[m.get("waiting_seconds", 0) for m in c.member_summaries] for _, c in pairs],
            resamples=bootstrap_resamples,
        )
        means = []
        for side in (0, 1):
            keys = (
                "mean_wait_seconds",
                "p95_wait_seconds",
                "completion_rate",
                "changes_per_member",
                "decision_latency_p95_seconds",
            )
            means.append(
                {
                    key: float(np.mean([pair[side].metrics[key] for pair in pairs]))
                    if all(pair[side].metrics.get(key) is not None for pair in pairs)
                    else None
                    for key in keys
                }
            )
        item["guardrails"] = assess_guardrails(
            *means, failed=failures > 0, latency_budget_seconds=latency_budget_seconds
        )
        summaries.append(item)
    return summaries


def _fmt(value):
    return (
        "undefined" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)
    )


def _plots(directory, runs, comparisons):
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="gym-sched-mpl-"))
    os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="gym-sched-cache-"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rc_context

    assets = directory / "figures"
    assets.mkdir(exist_ok=True)
    paths = []

    def save(fig, name):
        for ext in ("png", "svg"):
            fig.savefig(
                assets / f"{name}.{ext}",
                dpi=140,
                bbox_inches="tight",
                metadata={"Date": None} if ext == "svg" else {"Software": "gym-sched"},
            )
        plt.close(fig)
        paths.append(f"figures/{name}.png")

    with rc_context({"svg.hashsalt": "gym-sched-v1", "font.size": 9}):
        # One reproducible scenario only: avoid mixing populations or splits in a histogram.
        if runs:
            chosen_hash = runs[0][1].scenario_hash
            first = [(e, r) for e, r in runs if r.scenario_hash == chosen_hash]
            fig, ax = plt.subplots(figsize=(9, 4))
            bins = np.histogram_bin_edges(
                [
                    m.get("waiting_seconds", 0) / 60
                    for _, result in first
                    for m in result.member_summaries
                ],
                bins=12,
            )
            for _, result in first:
                ax.hist(
                    [m.get("waiting_seconds", 0) / 60 for m in result.member_summaries],
                    bins=bins,
                    histtype="step",
                    linewidth=1.5,
                    label=result.policy,
                )
            ax.set(
                xlabel="Accumulated waiting (minutes)",
                ylabel="Members",
                title=f"Synthetic wait distribution — {first[0][1].scenario_id}",
            )
            ax.legend(fontsize=7)
            save(fig, "wait_distribution")
            for index, (_, result) in enumerate(first):
                units = sorted(e["equipment_id"] for e in result.equipment_summaries)
                fig, ax = plt.subplots(figsize=(11, max(2.4, 0.48 * len(units) + 1.8)))
                members = sorted(m["member_id"] for m in result.member_summaries)
                palette = plt.get_cmap("tab20")
                colors = {mid: palette(i % 20) for i, mid in enumerate(members)}
                slots = {unit: i for i, unit in enumerate(units)}
                for event in result.events:
                    if event.get("event") == "exercise_started":
                        duration = min(
                            event.get("duration_seconds", 0),
                            max(0, result.end_time - event["time"]),
                        )
                        ax.broken_barh(
                            [(event["time"] / 60, duration / 60)],
                            (slots[event["equipment_id"]] - 0.35, 0.7),
                            facecolors=colors[event["member_id"]],
                            edgecolors="white",
                            linewidth=0.7,
                        )
                        if duration >= max(60, result.end_time / 60):
                            ax.text(
                                (event["time"] + duration / 2) / 60,
                                slots[event["equipment_id"]],
                                event["member_id"],
                                ha="center",
                                va="center",
                                fontsize=6,
                                bbox={
                                    "facecolor": "white",
                                    "alpha": 0.8,
                                    "edgecolor": "none",
                                    "pad": 1,
                                },
                            )
                ax.set(
                    yticks=list(slots.values()),
                    yticklabels=units,
                    xlabel="Simulation time (minutes)",
                    title=f"Synthetic equipment occupancy — {result.policy} ({result.status})",
                )
                ax.set_ylim(-0.65, max(0.65, len(units) - 0.35))
                ax.set_xlim(0, max(1, result.end_time / 60))
                ax.grid(axis="x", alpha=0.15)
                save(fig, f"timeline_{index:02d}")
        if comparisons:
            fig, ax = plt.subplots(figsize=(10, max(3, min(20, 0.4 * len(comparisons)))))
            for i, row in enumerate(comparisons):
                m = row["metrics"]["mean_wait_seconds"]
                if m["improvement"] is None:
                    continue
                lo, hi = m["ci_low"], m["ci_high"]
                if lo is not None:
                    ax.hlines(i, lo, hi, color="#286a8d")
                ax.plot(
                    m["improvement"], i, "o", color="#b45b26" if row["failed_pairs"] else "#286a8d"
                )
            labels = [
                f"{r['split']}/{r['family']}: {r['candidate']} (n={r['n_pairs']})"
                for r in comparisons
            ]
            ax.set(
                yticks=range(len(labels)),
                yticklabels=labels,
                xlabel="Baseline minus candidate mean wait (seconds); paired 95% interval",
                title="Synthetic scenario-level paired differences",
            )
            ax.axvline(0, color="#777777", linewidth=0.8)
            save(fig, "paired_mean_wait")
    return paths


def render_report(directory: Path | str, *, baseline="fcfs", bootstrap_resamples=None) -> dict:
    directory = Path(directory)
    runs = load_runs(directory)
    manifest = read_json(directory / "manifest.json")
    comparisons = build_comparisons(
        runs,
        baseline,
        bootstrap_resamples or manifest.get("bootstrap_resamples", 2000),
        manifest.get("latency_budget_seconds", 0.1),
    )
    write_json(directory / "comparisons.json", comparisons)
    failures = sum(bool(validate_result(r)) for _, r in runs)
    lines = [
        "# Synthetic gym scheduling experiment report",
        "",
        "All outcomes are simulated; they are not empirical gym results or proof of product benefit.",
        "",
        f"Episodes: {len(runs)}. Failed or audit-flagged episodes: {failures}. Baseline: `{baseline}`.",
        "",
        (
            "Panels with fewer than 30 independent scenarios are exploratory. Failed/stalled/truncated episodes "
            "remain visible; their waiting measures may be censored and cannot pass guardrails. "
            "Bootstrap intervals resample paired scenarios, not individual members. "
            "Intervals are unadjusted for multiple comparisons. A held-out test claim additionally requires a frozen protocol."
        ),
        "",
        "## Episode results",
        "",
        "| Split / family | Scenario | Policy | Status | Mean wait s | p95 wait s | Completion | Utilization |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for entry, r in runs:
        m = r.metrics
        lines.append(
            f"| {entry.get('split', 'exploratory')} / {entry.get('family', 'single')} | {r.scenario_id} | {r.policy} | {r.status} | {_fmt(m.get('mean_wait_seconds'))} | {_fmt(m.get('p95_wait_seconds'))} | {_fmt(m.get('completion_rate'))} | {_fmt(m.get('utilization'))} |"
        )
    lines += [
        "",
        "## Paired comparisons",
        "",
        (
            "Positive wait differences favor the candidate. Completion-rate differences use baseline minus candidate, "
            "so negative values favor the candidate. Zero-denominator relative improvements are undefined."
        ),
        "",
        "| Split / family | Candidate | Pairs | Failed pairs | Mean wait improvement s | 95% CI | Guardrails |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in comparisons:
        m = row["metrics"]["mean_wait_seconds"]
        lines.append(
            f"| {row['split']} / {row['family']} | {row['candidate']} | {row['n_pairs']} | {row['failed_pairs']} | {_fmt(m['improvement'])} | [{_fmt(m['ci_low'])}, {_fmt(m['ci_high'])}] | {'descriptive pass' if row['guardrails']['passes'] else 'not passed'} |"
        )
    lines += [
        "",
        "## Metric definitions and limitations",
        "",
        "- Mean and p95 wait include every arriving member, including unfinished members; failed episodes are censored.",
        "- Completion and on-time rates use all arrivals. Completion-time summaries explicitly include completers only.",
        "- Utilization uses the common arrival observation window, excluding unavailable capacity; drain utilization is separate.",
        "- Throughput is completed exercise visits per hour through episode end. Recommendation acceptance uses issued recommendations.",
        "- Normalized wait divides by minimum predicted required work. Jain's index measures equality of burden, not satisfaction; all-zero burden is undefined.",
        "- Policy compute latency is measured wall time; simulated decision delay is a separate configured input. Hardware and package versions are in manifests.",
        "- Per-scenario p95 contrasts and pooled-member p95 cluster bootstrap answer different questions; both are saved in comparisons.json.",
        "- Subgroup counts, waiting and completion are in each result.json. Small or empty subgroups do not support fairness claims.",
    ]
    if failures:
        lines += ["", "## Audit findings", ""]
        for _, r in runs:
            for finding in validate_result(r):
                lines.append(f"- {r.scenario_id}/{r.policy}: {finding}")
    paths = _plots(directory, runs, comparisons)
    lines += ["", "## Figures", ""]
    for path in paths:
        lines.append(f"![{Path(path).stem}]({path})")
    markdown = "\n".join(lines) + "\n"
    (directory / "report.md").write_text(markdown)
    # Keep generated HTML dependency-free, escaped, readable, and useful offline.
    body = f"<pre>{html.escape(markdown.split('## Figures')[0])}</pre>"
    body += "".join(
        f'<figure><img src="{html.escape(path)}" alt="{html.escape(Path(path).stem)}"></figure>'
        for path in paths
    )
    (directory / "report.html").write_text(
        "<!doctype html><html><meta charset='utf-8'><title>Gym scheduling research report</title>"
        "<style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 24px;color:#172c38}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.6 ui-monospace,monospace}img{max-width:100%}figure{margin:32px 0}</style>"
        f"<body>{body}</body></html>"
    )
    return {
        "episodes": len(runs),
        "failed_episodes": failures,
        "comparisons": len(comparisons),
        "report": str(directory / "report.html"),
    }
