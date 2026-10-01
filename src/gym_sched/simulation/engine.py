"""Explicit phased event calendar driven by a SimPy process.

SimPy advances the clock; the calendar batches equal-time domain events so
construction order cannot accidentally decide allocation priority. Advice never
claims a resource. Only phase-three physical start attempts can do that.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from time import perf_counter

import simpy

from gym_sched.domain import (
    Candidate,
    Decision,
    EquipmentView,
    Member,
    MemberView,
    Observation,
    PolicyContext,
    Recommendation,
    RunResult,
    Scenario,
    SimulationConfig,
    Visit,
    VisitView,
)
from gym_sched.scenarios.randomness import content_hash, uniform

WAITING = {"ready", "pending", "deciding"}
PRESENT = WAITING | {"rest", "walking", "exercising"}


@dataclass
class MemberState:
    spec: Member
    state: str = "not_arrived"
    ready_since: float | None = None
    done: list[str] = field(default_factory=list)
    active: tuple[str, str] | None = None
    started_at: float | None = None
    available_at: float = 0
    pending: dict | None = None
    previous: tuple[str, str] | None = None
    fallback: bool = False
    departure: float | None = None
    must_leave: bool = False
    waiting: float = 0
    resting: float = 0
    walking: float = 0
    exercising: float = 0
    response_wait: float = 0
    recommendations: int = 0
    changes: int = 0
    accepted: int = 0
    refused: int = 0
    ignored: int = 0
    failed_claims: int = 0
    deviation: float = 0
    reordered_visits: int = 0
    substituted_visits: int = 0
    attempts: dict[str, int] = field(default_factory=dict)
    response_counts: dict[str, list[int]] = field(default_factory=dict)
    waiting_at_ready: float = 0


class Simulator:
    def __init__(self, scenario: Scenario, policy, config: SimulationConfig | dict | None = None):
        # Revalidate/copy so policies and caller mutations cannot alter running truth.
        self.scenario = Scenario.model_validate(scenario.model_dump())
        self.config = (
            config
            if isinstance(config, SimulationConfig)
            else SimulationConfig.model_validate(config or {})
        )
        self.policy = policy
        self.env = simpy.Environment()
        self.time = 0.0
        self.calendar: list[tuple] = []
        self.sequence = 0
        self.processed = 0
        self.members = {m.id: MemberState(m) for m in self.scenario.members}
        self.equipment = {e.id: e for e in self.scenario.equipment}
        self.occupants: dict[str, str] = {}
        self.offline: set[str] = set()
        self.observed = {eid: (False, -1.0, None) for eid in self.equipment}
        self.events: list[dict] = []
        self.decisions: list[dict] = []
        self.intervals: list[dict] = []
        self.status = "completed"
        self._decision_pending = False
        self._last_decision_signature: str | None = None
        self._setup()

    def _setup(self):
        for mid, m in self.members.items():
            self._schedule(m.spec.arrival_seconds, 2, "arrival", member_id=mid)
            if m.spec.deadline_seconds is not None:
                self._schedule(m.spec.deadline_seconds, 1, "deadline", member_id=mid)
        for eid, e in self.equipment.items():
            for start, end in e.outages:
                self._schedule(start, 1, "outage_start", equipment_id=eid)
                self._schedule(end, 1, "outage_end", equipment_id=eid)
        self._schedule(0, 4, "sensor_tick")

    def _schedule(self, time: float, phase: int, kind: str, **payload):
        if time < self.time - 1e-7:
            raise ValueError("cannot schedule an event in the past")
        self.sequence += 1
        # Completions win deadline ties; members win claim ties in stable ID order.
        kind_order = {"finish": 0, "deadline": 1, "patience": 2}.get(kind, 3)
        key = (kind_order, payload.get("member_id", ""), payload.get("equipment_id", ""), kind)
        heapq.heappush(self.calendar, (float(time), phase, key, self.sequence, kind, payload))

    def _log(self, event: str, **payload):
        self.events.append({"time": self.time, "event": event, **payload})

    def _advance(self, time: float):
        elapsed = time - self.time
        for m in self.members.values():
            if m.state in WAITING:
                m.waiting += elapsed
                if m.state == "pending":
                    m.response_wait += elapsed
            elif m.state == "rest":
                m.resting += elapsed
            elif m.state == "walking":
                m.walking += elapsed
            elif m.state == "exercising":
                m.exercising += elapsed
        self.time = float(time)

    def _set_state(self, m: MemberState, state: str):
        m.state = state
        if state in WAITING and m.spec.patience_wait_seconds is not None:
            remaining = max(0.0, m.spec.patience_wait_seconds - m.waiting)
            self._schedule(self.time + remaining, 1, "patience", member_id=m.spec.id)

    def _ready(self, m: MemberState, preserve_age: bool = False):
        if m.state == "departed":
            return
        if m.must_leave or (
            m.spec.deadline_seconds is not None and self.time >= m.spec.deadline_seconds
        ):
            self._depart(m, "deadline")
            return
        if not preserve_age or m.ready_since is None:
            m.ready_since = self.time
            m.waiting_at_ready = m.waiting
        self._set_state(m, "ready")
        m.available_at = self.time
        self._log("ready", member_id=m.spec.id, ready_since=m.ready_since)

    def _depart(self, m: MemberState, reason: str):
        if m.state == "departed":
            return
        if m.state == "exercising":
            m.must_leave = True
            return
        if m.pending:
            self._log(
                "recommendation_cancelled",
                member_id=m.spec.id,
                recommendation_id=m.pending["id"],
                reason=reason,
            )
            m.pending = None
        m.state = "departed"
        m.departure = self.time
        self._log(
            "departure",
            member_id=m.spec.id,
            reason=reason,
            completed=len(m.done) == len(m.spec.visits),
        )

    def _visit(self, m: MemberState, vid: str) -> Visit:
        return next(v for v in m.spec.visits if v.id == vid)

    def _eligible_visits(self, m: MemberState) -> list[Visit]:
        remaining = [v for v in m.spec.visits if v.id not in m.done]
        if not self.config.allow_reordering:
            remaining = remaining[:1]
        return [v for v in remaining if set(v.predecessors) <= set(m.done)]

    def _units(self, visit: Visit) -> list[str]:
        return [
            eid
            for eid in visit.equipment_ids
            if self.config.allow_substitutions or visit.deviation_by_equipment.get(eid, 0) == 0
        ]

    def _capture(self, eid: str):
        if (
            uniform(self.scenario.seed, "sensor-dropout", eid, self.time)
            < self.config.observation_dropout
        ):
            return
        available = eid not in self.occupants and eid not in self.offline
        release = None
        if not available:
            release = self.time + 300.0
            mid = self.occupants.get(eid)
            if mid and self.members[mid].spec.participates:
                m = self.members[mid]
                visit = self._visit(m, m.active[0])
                release = max(self.time, m.started_at + visit.predicted_seconds[eid])
        self._schedule(
            self.time + self.config.observation_delay_seconds,
            4,
            "observation",
            equipment_id=eid,
            observed_available=available,
            observed_at=self.time,
            estimated_release_seconds=release,
        )

    def observation(self) -> Observation:
        units = []
        for eid in sorted(self.equipment):
            available, timestamp, release = self.observed[eid]
            if not available and release is not None and release <= self.time:
                # Forecast expired, but no release has been observed. No truth lookup.
                release = self.time + self.config.observation_interval_seconds
            units.append(
                EquipmentView(eid, self.equipment[eid].kind, available, timestamp, release)
            )
        visible = []
        for mid in sorted(self.members):
            m = self.members[mid]
            # Non-app users have no declared workout or immediate state telemetry.
            if m.state not in PRESENT or not m.spec.participates:
                continue
            remaining = []
            for v in m.spec.visits:
                if v.id in m.done or (m.active and v.id == m.active[0]):
                    continue
                remaining.append(
                    VisitView(
                        v.id,
                        v.exercise,
                        tuple(self._units(v)),
                        {eid: v.predicted_seconds[eid] for eid in self._units(v)},
                        {eid: v.uncertainty_seconds.get(eid, 0) for eid in self._units(v)},
                        tuple(v.preferred_equipment_ids),
                        dict(v.deviation_by_equipment),
                        v.rest_after_seconds,
                        tuple(v.predecessors),
                    )
                )
            available_at = m.available_at
            if m.active:
                v = self._visit(m, m.active[0])
                available_at = max(self.time, m.started_at + v.predicted_seconds[m.active[1]])
                available_at += v.rest_after_seconds
            visible.append(
                MemberView(
                    mid,
                    m.state,
                    m.ready_since,
                    m.spec.deadline_seconds,
                    tuple(remaining),
                    tuple(m.done),
                    True,
                    available_at,
                    m.active[1] if m.active else None,
                    m.previous,
                    (m.accepted, m.refused + m.ignored),
                    m.active[0] if m.active else None,
                )
            )
        return Observation(
            self.time,
            tuple(units),
            tuple(visible),
            self.config.allow_reordering,
            self.config.allow_substitutions,
        )

    def candidates(self) -> tuple[Candidate, ...]:
        result = []
        for mid in sorted(self.members):
            m = self.members[mid]
            if m.state != "ready" or not m.spec.participates or m.fallback:
                continue
            for v in self._eligible_visits(m):
                for eid in sorted(self._units(v)):
                    available, timestamp, _ = self.observed[eid]
                    if not available:
                        continue
                    previous_same_visit = m.previous and m.previous[0] == v.id
                    preferred = not v.preferred_equipment_ids or eid in v.preferred_equipment_ids
                    successes, failures = m.response_counts.get(
                        "preferred" if preferred else "alternative", [0, 0]
                    )
                    # Transparent observable priors: Beta(1,1) preferred and
                    # Beta(1,2) alternatives; no access to latent willingness.
                    estimate = (successes + 1) / (successes + failures + (2 if preferred else 3))
                    result.append(
                        Candidate(
                            mid,
                            v.id,
                            eid,
                            m.ready_since,
                            v.predicted_seconds[eid],
                            v.uncertainty_seconds.get(eid, 0),
                            max(0, self.time - timestamp),
                            preferred,
                            v.deviation_by_equipment.get(eid, 0),
                            bool(previous_same_visit and m.previous[1] != eid),
                            estimate,
                            max(0, m.waiting - m.waiting_at_ready),
                        )
                    )
        return tuple(result)

    def _self_choices(self):
        for mid in sorted(self.members):
            m = self.members[mid]
            if m.state != "ready" or not (
                m.fallback
                or not m.spec.participates
                or getattr(self.policy, "self_directed", False)
            ):
                continue
            eligible = self._eligible_visits(m)
            if m.spec.independent_keep_plan_order:
                first = next((v for v in m.spec.visits if v.id not in m.done), None)
                eligible = [v for v in eligible if v == first]
            choices = [
                (v, eid)
                for v in eligible
                for eid in self._units(v)
                # Autonomous members look at equipment locally, independently
                # of app sensors. This current-availability assumption belongs
                # to member behavior only; policy candidates still use sensors.
                if eid not in self.occupants and eid not in self.offline
            ]
            if not choices:
                # Stay autonomous for this ready request. Releases/outage ends
                # trigger another choice; no timer or repeated advice is needed.
                continue
            # Members independently choose; duplicate attempts are resolved physically.
            v, eid = min(
                choices,
                key=lambda pair: (
                    pair[0].deviation_by_equipment.get(pair[1], 0),
                    bool(
                        pair[0].preferred_equipment_ids
                        and pair[1] not in pair[0].preferred_equipment_ids
                    ),
                    pair[0].id,
                    pair[1],
                ),
            )
            self._walk(m, v.id, eid, None)

    def _walk(self, m: MemberState, vid: str, eid: str, rid: str | None):
        m.state = "walking"
        m.available_at = self.time + self.config.walking_seconds
        self._log(
            "walking_started",
            member_id=m.spec.id,
            visit_id=vid,
            equipment_id=eid,
            recommendation_id=rid,
            duration_seconds=self.config.walking_seconds,
        )
        self._schedule(
            m.available_at,
            3,
            "claim",
            member_id=m.spec.id,
            visit_id=vid,
            equipment_id=eid,
            recommendation_id=rid,
        )

    def _choose_independently(self, m: MemberState):
        m.fallback = True
        self._ready(m, preserve_age=True)

    def _issue(self, rec: Recommendation):
        m = self.members[rec.member_id]
        v = self._visit(m, rec.visit_id)
        m.recommendations += 1
        if m.previous and m.previous[0] == rec.visit_id and m.previous[1] != rec.equipment_id:
            m.changes += 1
        m.previous = (rec.visit_id, rec.equipment_id)
        rid = f"{m.spec.id}:{m.recommendations}"
        ordinal = m.attempts.get(v.id, 0)
        m.attempts[v.id] = ordinal + 1
        m.pending = {
            "id": rid,
            "visit_id": v.id,
            "equipment_id": rec.equipment_id,
            "ordinal": ordinal,
            "expires": self.time + self.config.recommendation_expiry_seconds,
        }
        self._set_state(m, "pending")
        self._log(
            "recommendation_issued",
            member_id=m.spec.id,
            visit_id=v.id,
            equipment_id=rec.equipment_id,
            recommendation_id=rid,
            expires_at=m.pending["expires"],
        )
        self._schedule(
            m.pending["expires"], 2, "expire", member_id=m.spec.id, recommendation_id=rid
        )
        ignored = (
            uniform(self.scenario.seed, "ignore", m.spec.id, v.id, ordinal)
            < m.spec.ignore_probability
        )
        if ignored:
            return
        self._schedule(
            self.time + self.config.response_delay_seconds,
            2,
            "response",
            member_id=m.spec.id,
            recommendation_id=rid,
        )

    def _handle(self, kind: str, p: dict):
        m = self.members.get(p.get("member_id"))
        if kind == "arrival":
            self._log("arrival", member_id=m.spec.id)
            self._ready(m)
        elif kind in {"deadline", "patience"}:
            if m.state not in PRESENT:
                return
            if kind == "patience" and (
                m.state not in WAITING or m.waiting + 1e-7 < m.spec.patience_wait_seconds
            ):
                return
            self._depart(m, kind)
        elif kind == "ready":
            if m.state != "rest":
                return
            self._ready(m)
        elif kind == "response":
            if not m.pending or m.pending["id"] != p["recommendation_id"] or m.state != "pending":
                return
            advice = m.pending
            if self.time >= advice["expires"]:
                return
            v = self._visit(m, advice["visit_id"])
            probability = m.spec.acceptance_probability
            if (
                m.spec.preference_change_seconds is not None
                and self.time >= m.spec.preference_change_seconds
                and m.spec.changed_acceptance_probability is not None
            ):
                probability = m.spec.changed_acceptance_probability
            if (
                v.preferred_equipment_ids
                and advice["equipment_id"] not in v.preferred_equipment_ids
            ):
                probability *= m.spec.nonpreferred_acceptance_multiplier
            accepted = (
                uniform(self.scenario.seed, "acceptance", m.spec.id, v.id, advice["ordinal"])
                < probability
            )
            m.pending = None
            response_class = (
                "preferred"
                if (
                    not v.preferred_equipment_ids
                    or advice["equipment_id"] in v.preferred_equipment_ids
                )
                else "alternative"
            )
            counts = m.response_counts.setdefault(response_class, [0, 0])
            counts[0 if accepted else 1] += 1
            if accepted:
                m.accepted += 1
                self._log(
                    "accepted",
                    member_id=m.spec.id,
                    recommendation_id=advice["id"],
                    visit_id=v.id,
                    equipment_id=advice["equipment_id"],
                )
                self._walk(m, v.id, advice["equipment_id"], advice["id"])
            else:
                m.refused += 1
                self._log(
                    "refused",
                    member_id=m.spec.id,
                    recommendation_id=advice["id"],
                    visit_id=v.id,
                    equipment_id=advice["equipment_id"],
                )
                self._choose_independently(m)
        elif kind == "expire":
            if m.pending and m.pending["id"] == p["recommendation_id"]:
                v = self._visit(m, m.pending["visit_id"])
                response_class = (
                    "preferred"
                    if (
                        not v.preferred_equipment_ids
                        or m.pending["equipment_id"] in v.preferred_equipment_ids
                    )
                    else "alternative"
                )
                m.response_counts.setdefault(response_class, [0, 0])[1] += 1
                m.ignored += 1
                self._log("expired", member_id=m.spec.id, recommendation_id=p["recommendation_id"])
                self._log("ignored", member_id=m.spec.id, recommendation_id=p["recommendation_id"])
                m.pending = None
                self._choose_independently(m)
        elif kind == "claim":
            if m.state != "walking":
                return
            eid, vid = p["equipment_id"], p["visit_id"]
            if m.must_leave or (
                m.spec.deadline_seconds is not None and self.time >= m.spec.deadline_seconds
            ):
                self._depart(m, "deadline")
                return
            v = self._visit(m, vid)
            if eid not in self._units(v) or v not in self._eligible_visits(m):
                raise AssertionError("incompatible or out-of-order physical start")
            if eid in self.occupants or eid in self.offline:
                m.failed_claims += 1
                self._log("claim_failed", **p)
                self._choose_independently(m)
                return
            assert m.active is None
            self.occupants[eid] = m.spec.id
            m.active = (vid, eid)
            m.started_at = self.time
            m.state = "exercising"
            m.ready_since = None
            m.fallback = False
            duration = v.actual_seconds[eid]
            m.deviation += v.deviation_by_equipment.get(eid, 0)
            expected_visit = next(block.id for block in m.spec.visits if block.id not in m.done)
            m.reordered_visits += int(vid != expected_visit)
            m.substituted_visits += int(
                eid in v.alternative_exercise_by_equipment
                or v.deviation_by_equipment.get(eid, 0) > 0
            )
            self._log(
                "exercise_started",
                **p,
                duration_seconds=duration,
                predicted_seconds=v.predicted_seconds[eid],
                exercise=v.alternative_exercise_by_equipment.get(eid, v.exercise),
                deviation=v.deviation_by_equipment.get(eid, 0),
            )
            self._schedule(
                self.time + duration,
                0,
                "finish",
                member_id=m.spec.id,
                visit_id=vid,
                equipment_id=eid,
            )
            self._capture(eid)
        elif kind == "finish":
            eid, vid = p["equipment_id"], p["visit_id"]
            assert m.active == (vid, eid) and self.occupants[eid] == m.spec.id
            self.intervals.append(
                {
                    "member_id": m.spec.id,
                    "visit_id": vid,
                    "equipment_id": eid,
                    "start": m.started_at,
                    "end": self.time,
                }
            )
            del self.occupants[eid]
            m.done.append(vid)
            m.active = None
            m.state = "ready"  # enables departure without interrupting completed work
            self._log("exercise_completed", **p)
            self._capture(eid)
            if len(m.done) == len(m.spec.visits):
                self._depart(m, "completed")
            elif m.must_leave or (
                m.spec.deadline_seconds is not None and self.time >= m.spec.deadline_seconds
            ):
                self._depart(m, "deadline")
            else:
                rest = self._visit(m, vid).rest_after_seconds
                m.state = "rest"
                m.available_at = self.time + rest
                self._log("rest_started", member_id=m.spec.id, duration_seconds=rest)
                self._schedule(m.available_at, 2, "ready", member_id=m.spec.id)
        elif kind == "sensor_tick":
            for eid in sorted(self.equipment):
                self._capture(eid)
            if self.config.observation_delay_seconds or self.config.observation_dropout:
                self._schedule(
                    self.time + self.config.observation_interval_seconds, 4, "sensor_tick"
                )
        elif kind == "observation":
            eid = p["equipment_id"]
            if p["observed_at"] >= self.observed[eid][1]:
                self.observed[eid] = (
                    p["observed_available"],
                    p["observed_at"],
                    p["estimated_release_seconds"],
                )
                self._log("observation", **p)
        elif kind in {"outage_start", "outage_end"}:
            eid = p["equipment_id"]
            if kind == "outage_start":
                self.offline.add(eid)
            else:
                self.offline.discard(eid)
            # Outages block new visits; they never preempt a visit already in progress.
            self._log(
                "outage_started" if kind == "outage_start" else "outage_ended", equipment_id=eid
            )
            self._capture(eid)
        elif kind == "deliver_decision":
            self._decision_pending = False
            valid = {(c.member_id, c.visit_id, c.equipment_id) for c in self.candidates()}
            for rec in p["recommendations"]:
                if (rec.member_id, rec.visit_id, rec.equipment_id) in valid:
                    self._issue(rec)
                else:
                    self._log(
                        "stale_decision",
                        member_id=rec.member_id,
                        visit_id=rec.visit_id,
                        equipment_id=rec.equipment_id,
                    )
        else:
            raise AssertionError(f"unhandled event {kind}")

    def _dispatch(self):
        self._self_choices()
        if self._decision_pending or getattr(self.policy, "self_directed", False):
            return
        started = perf_counter()
        obs = self.observation()
        candidates = self.candidates()
        if not candidates:
            return
        # A unchanged no-op is not retried merely because an irrelevant timer fired.
        signature = content_hash(
            [
                [
                    (
                        c.member_id,
                        c.visit_id,
                        c.equipment_id,
                        c.ready_since,
                        c.observation_age_seconds,
                    )
                    for c in candidates
                ],
                self.time,
            ]
        )
        if signature == self._last_decision_signature:
            return
        self._last_decision_signature = signature
        decision: Decision = self.policy.decide(obs, candidates)
        elapsed = perf_counter() - started
        allowed = {(c.member_id, c.visit_id, c.equipment_id) for c in candidates}
        selected = [(r.member_id, r.visit_id, r.equipment_id) for r in decision.recommendations]
        if (
            any(a not in allowed for a in selected)
            or len({a[0] for a in selected}) != len(selected)
            or len({a[2] for a in selected}) != len(selected)
            or (decision.defer and selected)
        ):
            raise ValueError("policy returned an invalid or conflicting recommendation batch")
        i = len(self.decisions)
        delay = (
            self.config.decision_delay_trace[i % len(self.config.decision_delay_trace)]
            if self.config.decision_delay_trace
            else self.config.decision_delay_seconds
        )
        self.decisions.append(
            {
                "time": self.time,
                "decision_index": i,
                "latency_seconds": elapsed,
                "delivery_delay_seconds": delay,
                "candidate_count": len(candidates),
                "recommendation_count": len(selected),
                "diagnostics": decision.diagnostics,
            }
        )
        if not selected:
            self._log("decision_deferred")
            return
        if delay:
            self._decision_pending = True
            self._schedule(
                self.time + delay, 5, "deliver_decision", recommendations=decision.recommendations
            )
        else:
            for rec in decision.recommendations:
                self._issue(rec)

    def _all_done(self) -> bool:
        return all(m.state == "departed" for m in self.members.values())

    def _process(self):
        rounds = 0
        last_time = -1
        while not self._all_done():
            if not self.calendar:
                self.status = "stalled"
                self._log("stalled", reason="unresolved members with no future events")
                break
            next_time = self.calendar[0][0]
            if next_time > self.config.max_time_seconds or self.processed >= self.config.max_events:
                stop = min(next_time, self.config.max_time_seconds)
                self._advance(stop)
                self.status = "truncated"
                self._log("truncated", reason="simulation watchdog")
                break
            if next_time > self.time:
                yield self.env.timeout(next_time - self.time)
                self._advance(float(self.env.now))
            rounds = rounds + 1 if self.time == last_time else 1
            last_time = self.time
            if rounds > self.config.max_same_time_rounds:
                self.status = "stalled"
                self._log("stalled", reason="same-time progress guard")
                break
            # Drain all direct effects at the timestamp before a policy call.
            # Newly created lower-phase events are a microphase; no recursive dispatch.
            while self.calendar and self.calendar[0][0] <= self.time:
                _, _, _, _, kind, payload = heapq.heappop(self.calendar)
                self._handle(kind, payload)
                self.processed += 1
                if self.processed >= self.config.max_events:
                    break
            if not self._all_done():
                self._dispatch()

    def run(self) -> RunResult:
        self.policy.reset(
            PolicyContext(seed=int(getattr(self.policy, "options", {}).get("seed", 0)))
        )
        self.env.process(self._process())
        self.env.run()
        return self.result()

    def result(self) -> RunResult:
        member_rows = []
        for mid, m in sorted(self.members.items()):
            if m.state == "not_arrived":
                continue  # Future planned arrivals are not entrants in a truncated episode.
            completed = len(m.done) == len(m.spec.visits)
            final_time = m.departure if m.departure is not None else self.time
            ontime = completed and (
                m.spec.deadline_seconds is None or final_time <= m.spec.deadline_seconds
            )
            member_rows.append(
                {
                    "member_id": mid,
                    "arrival": m.spec.arrival_seconds,
                    "departure": m.departure,
                    "state": m.state,
                    "completed": completed,
                    "completed_visits": len(m.done),
                    "total_visits": len(m.spec.visits),
                    "waiting_seconds": m.waiting,
                    "rest_seconds": m.resting,
                    "walking_seconds": m.walking,
                    "exercise_seconds": m.exercising,
                    "response_wait_seconds": m.response_wait,
                    "completion_time_seconds": final_time - m.spec.arrival_seconds
                    if completed
                    else None,
                    "on_time": ontime,
                    "overtime_seconds": max(0, final_time - m.spec.deadline_seconds)
                    if m.spec.deadline_seconds is not None
                    else 0,
                    "participates": m.spec.participates,
                    "required_work_seconds": sum(
                        min(v.predicted_seconds.values()) for v in m.spec.visits
                    ),
                    "equipment_flexibility": sum(len(self._units(v)) for v in m.spec.visits)
                    / len(m.spec.visits),
                    "recommendations": m.recommendations,
                    "changes": m.changes,
                    "accepted": m.accepted,
                    "refused": m.refused,
                    "ignored": m.ignored,
                    "failed_claims": m.failed_claims,
                    "deviation": m.deviation,
                    "reordered_visits": m.reordered_visits,
                    "substituted_visits": m.substituted_visits,
                }
            )
        intervals = list(self.intervals)
        for mid, m in self.members.items():
            if m.active:
                intervals.append(
                    {
                        "member_id": mid,
                        "visit_id": m.active[0],
                        "equipment_id": m.active[1],
                        "start": m.started_at,
                        "end": self.time,
                    }
                )
        equipment_rows = []
        window = self.scenario.arrival_window_seconds

        def overlap(a, b, c, d):
            return max(0, min(b, d) - max(a, c))

        for eid, e in sorted(self.equipment.items()):
            spans = [x for x in intervals if x["equipment_id"] == eid]
            # A visit active at outage start is allowed to finish. Such occupied time
            # remains usable capacity; the outage removes only otherwise idle time.
            unavailable = sum(overlap(a, b, 0, window) for a, b in e.outages)
            unavailable -= sum(
                overlap(max(a, x["start"]), min(b, x["end"]), 0, window)
                for a, b in e.outages
                for x in spans
            )
            available = window - unavailable
            occupied = sum(overlap(x["start"], x["end"], 0, window) for x in spans)
            drain_duration = max(0, self.time - window)
            drain_unavailable = sum(overlap(a, b, window, self.time) for a, b in e.outages)
            drain_unavailable -= sum(
                overlap(max(a, x["start"]), min(b, x["end"]), window, self.time)
                for a, b in e.outages
                for x in spans
            )
            equipment_rows.append(
                {
                    "equipment_id": eid,
                    "occupied_seconds": sum(x["end"] - x["start"] for x in spans),
                    "occupied_window_seconds": occupied,
                    "available_window_seconds": available,
                    "utilization": occupied / available if available > 0 else None,
                    "drain_occupied_seconds": sum(
                        overlap(x["start"], x["end"], window, self.time) for x in spans
                    ),
                    "drain_available_seconds": max(0, drain_duration - drain_unavailable),
                }
            )
        result = RunResult(
            scenario_id=self.scenario.id,
            scenario_hash=content_hash(self.scenario),
            policy=self.policy.name,
            seed=self.scenario.seed,
            status=self.status,
            end_time=self.time,
            config=self.config.model_dump(mode="json"),
            events=self.events,
            member_summaries=member_rows,
            equipment_summaries=equipment_rows,
            decisions=self.decisions,
            metrics={},
        )
        from gym_sched.evaluation.metrics import compute_metrics

        result.metrics = compute_metrics(result)
        return result


def run_episode(
    scenario: Scenario, policy, config: SimulationConfig | dict | None = None
) -> RunResult:
    return Simulator(scenario, policy, config).run()
