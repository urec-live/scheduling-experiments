"""Episode metrics with explicit denominators; unfinished members remain in the cohort."""

from __future__ import annotations

from typing import Any

import numpy as np


def ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator > 0 else None


def _mean(values):
    return float(np.mean(values)) if values else None


def _p95(values):
    return float(np.quantile(values, 0.95)) if values else None


def compute_metrics(result) -> dict[str, Any]:
    members = result.member_summaries
    equipment = result.equipment_summaries
    n = len(members)
    waits = [float(m.get("waiting_seconds", 0)) for m in members]
    completed = [m for m in members if m.get("completed", False)]
    completed_times = [
        float(m["completion_time_seconds"])
        for m in completed
        if m.get("completion_time_seconds") is not None
    ]
    latencies = [
        float(d["latency_seconds"])
        for d in result.decisions
        if d.get("latency_seconds") is not None
    ]
    completed_visits = sum(m.get("completed_visits", 0) for m in members)
    total_visits = sum(m.get("total_visits", 0) for m in members)
    occupied = sum(e.get("occupied_window_seconds", 0) for e in equipment)
    available = sum(e.get("available_window_seconds", 0) for e in equipment)
    recs = sum(m.get("recommendations", 0) for m in members)
    normalized = [
        m.get("waiting_seconds", 0) / m["required_work_seconds"]
        for m in members
        if m.get("required_work_seconds", 0) > 0
    ]
    # Jain's fairness index on normalized waiting burden, not service quality.
    fairness = ratio(sum(normalized) ** 2, len(normalized) * sum(x * x for x in normalized))
    out = {
        "member_count": n,
        "completed_members": len(completed),
        "completed_visits": completed_visits,
        "total_visits": total_visits,
        "mean_wait_seconds": _mean(waits),
        "p95_wait_seconds": _p95(waits),
        "max_wait_seconds": max(waits, default=None),
        "completion_rate": ratio(len(completed), n),
        "abandonment_rate": ratio(
            sum(m.get("departure") is not None and not m.get("completed") for m in members), n
        ),
        "visit_completion_rate": ratio(completed_visits, total_visits),
        "on_time_rate": ratio(sum(bool(m.get("on_time", False)) for m in members), n),
        "mean_completion_seconds_completed_only": _mean(completed_times),
        "throughput_per_hour": ratio(completed_visits, result.end_time / 3600),
        "utilization": ratio(occupied, available),
        "drain_utilization": ratio(
            sum(e.get("drain_occupied_seconds", 0) for e in equipment),
            sum(e.get("drain_available_seconds", 0) for e in equipment),
        ),
        "mean_rest_seconds": _mean([m.get("rest_seconds", 0) for m in members]),
        "mean_walking_seconds": _mean([m.get("walking_seconds", 0) for m in members]),
        "mean_response_wait_seconds": _mean([m.get("response_wait_seconds", 0) for m in members]),
        "mean_overtime_seconds": _mean([m.get("overtime_seconds", 0) for m in members]),
        "mean_deviation": ratio(sum(m.get("deviation", 0) for m in members), completed_visits),
        "changes_per_member": ratio(sum(m.get("changes", 0) for m in members), n),
        "reorders_per_member": ratio(sum(m.get("reordered_visits", 0) for m in members), n),
        "substitutions_per_member": ratio(sum(m.get("substituted_visits", 0) for m in members), n),
        "recommendation_count": recs,
        "acceptance_rate": ratio(sum(m.get("accepted", 0) for m in members), recs),
        "refusal_rate": ratio(sum(m.get("refused", 0) for m in members), recs),
        "ignore_rate": ratio(sum(m.get("ignored", 0) for m in members), recs),
        "failed_claims": sum(m.get("failed_claims", 0) for m in members),
        "mean_normalized_wait": _mean(normalized),
        "jain_normalized_wait": fairness,
        "decision_count": len(result.decisions),
        "decision_latency_mean_seconds": _mean(latencies),
        "decision_latency_p95_seconds": _p95(latencies),
        "decision_latency_max_seconds": max(latencies, default=None),
        "decision_latency_p50_seconds": float(np.median(latencies)) if latencies else None,
        "decision_over_100ms_rate": ratio(sum(t > 0.1 for t in latencies), len(latencies)),
        "fallback_rate": ratio(
            sum(bool(d.get("diagnostics", {}).get("fallback", False)) for d in result.decisions),
            len(result.decisions),
        ),
        "episode_failed": result.status != "completed",
    }
    groups = {}
    for label, cohort in {
        "participants": [m for m in members if m.get("participates", True)],
        "nonparticipants": [m for m in members if not m.get("participates", True)],
        "single_equipment_option": [m for m in members if m.get("equipment_flexibility", 0) <= 1],
        "multiple_equipment_options": [m for m in members if m.get("equipment_flexibility", 0) > 1],
    }.items():
        groups[label] = {
            "n": len(cohort),
            "mean_wait_seconds": _mean([m.get("waiting_seconds", 0) for m in cohort]),
            "p95_wait_seconds": _p95([m.get("waiting_seconds", 0) for m in cohort]),
            "completion_rate": ratio(sum(bool(m.get("completed")) for m in cohort), len(cohort)),
        }
    out["subgroups"] = groups
    return out


def validate_result(result) -> list[str]:
    """Return audit findings; a failed episode can never pass evaluation guardrails."""
    problems = []
    if result.status != "completed":
        problems.append(f"Episode {result.status}; waiting times can be censored.")
    ids = [m["member_id"] for m in result.member_summaries]
    if len(set(ids)) != len(ids):
        problems.append("Duplicate member summaries.")
    for m in result.member_summaries:
        for key in (
            "waiting_seconds",
            "rest_seconds",
            "walking_seconds",
            "response_wait_seconds",
            "completed_visits",
            "total_visits",
            "deviation",
        ):
            if m.get(key, 0) < -1e-8:
                problems.append(f"Negative {key}: {m['member_id']}.")
        if m.get("completed_visits", 0) > m.get("total_visits", 0):
            problems.append(f"Completed visits exceed plan: {m['member_id']}.")
        if m.get("completed") and m.get("completed_visits") != m.get("total_visits"):
            problems.append(f"Completion flag disagrees with visits: {m['member_id']}.")
        if m.get("departure") is not None:
            accounted = sum(
                m.get(k, 0)
                for k in ("waiting_seconds", "rest_seconds", "walking_seconds", "exercise_seconds")
            )
            elapsed = m["departure"] - m["arrival"]
            if abs(accounted - elapsed) > 1e-6 * max(1, elapsed):
                problems.append(f"Incomplete elapsed-time accounting: {m['member_id']}.")
    for e in result.equipment_summaries:
        if e.get("occupied_window_seconds", 0) > e.get("available_window_seconds", 0) + 1e-7:
            problems.append(f"Utilization exceeds capacity: {e['equipment_id']}.")
    active_equipment, active_members = {}, {}
    for event in result.events:
        kind = event.get("event")
        unit, member = event.get("equipment_id"), event.get("member_id")
        if kind == "exercise_started":
            if unit in active_equipment:
                problems.append(f"Overlapping occupancy: {unit} at {event.get('time')}.")
            if member in active_members:
                problems.append(f"Concurrent visits: {member} at {event.get('time')}.")
            active_equipment[unit] = member
            active_members[member] = unit
        elif kind == "exercise_completed":
            if active_equipment.get(unit) != member:
                problems.append(f"Unmatched exercise completion: {member}/{unit}.")
            active_equipment.pop(unit, None)
            active_members.pop(member, None)
    if result.status == "completed" and active_equipment:
        problems.append("Completed episode still has occupied equipment.")
    return problems
