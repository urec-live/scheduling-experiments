"""Rolling CP-SAT and CPU genetic search over the same observed workload."""

from __future__ import annotations

import math
import random
import time
from typing import Any, ClassVar

from gym_sched.domain.models import Candidate, Decision, Observation, PolicyContext
from gym_sched.policies.base import BasePolicy
from gym_sched.policies.forecast import (
    ForecastSchedule,
    PlannedOperation,
    build_problem,
    decode,
    immediate_recommendations,
    initial_genes,
    summarize_schedule,
)
from gym_sched.policies.heuristics import MatchingPolicy


class CPSATPolicy(BasePolicy):
    defaults: ClassVar[dict[str, Any]] = {
        "time_limit_seconds": 0.1,
        "deterministic_time_limit": 0.05,
        "time_resolution_seconds": 0.001,
        "unknown_busy_seconds": 300.0,
    }

    def __init__(self, name="cpsat", options=None):
        super().__init__(name, options)
        if any(float(v) < 0 for v in self.options.values()):
            raise ValueError("CP-SAT limits must be nonnegative")
        if self.options["time_resolution_seconds"] <= 0:
            raise ValueError("time_resolution_seconds must be positive")
        self.fallback = MatchingPolicy("matching")
        self.last_schedule: ForecastSchedule | None = None

    def reset(self, context: PolicyContext) -> None:
        super().reset(context)
        self.fallback.reset(context)
        self.last_schedule = None

    def _fallback(self, observation, candidates, diagnostics):
        decision = self.fallback.decide(observation, candidates)
        return Decision(
            decision.recommendations,
            decision.defer,
            {**diagnostics, "fallback": True, "fallback_policy": "matching"},
        )

    def decide(self, observation: Observation, candidates: tuple[Candidate, ...]) -> Decision:
        started = time.perf_counter()
        self.last_schedule = None
        if not candidates:
            return Decision(
                defer=True, diagnostics={"solver_status": "NO_CANDIDATES", "fallback": False}
            )
        try:
            from ortools.sat.python import cp_model
        except ImportError:
            return self._fallback(observation, candidates, {"solver_status": "UNAVAILABLE"})
        problem = build_problem(observation, self.options["unknown_busy_seconds"])
        if not problem.operations:
            return Decision(defer=True, diagnostics={"solver_status": "EMPTY", "fallback": False})
        resolution = float(self.options["time_resolution_seconds"])

        def ticks(value):
            return max(0, math.ceil(value / resolution - 1e-9))

        model = cp_model.CpModel()
        base_release = max(
            [*problem.member_available.values(), *problem.equipment_available.values()]
        )
        horizon = (
            ticks(base_release - problem.now)
            + sum(ticks(max(op.durations)) + ticks(op.rest_after) for op in problem.operations)
            + len(problem.operations)
            + 1
        )
        starts, ends, durations, choices = {}, {}, {}, {}
        equipment_intervals: dict[str, list[Any]] = {
            unit: [] for unit in problem.equipment_available
        }
        for operation in problem.operations:
            key = operation.key
            starts[key] = model.new_int_var(0, horizon, f"start_{key}")
            ends[key] = model.new_int_var(0, horizon, f"end_{key}")
            durations[key] = model.new_int_var(1, horizon, f"duration_{key}")
            model.add(ends[key] == starts[key] + durations[key])
            model.add(
                starts[key] >= ticks(problem.member_available[operation.member_id] - problem.now)
            )
            selectors = []
            for index, unit in enumerate(operation.equipment_ids):
                selected = model.new_bool_var(f"equipment_{key}_{unit}")
                choices[key, unit] = selected
                selectors.append(selected)
                duration = max(1, ticks(operation.durations[index]))
                interval = model.new_optional_interval_var(
                    starts[key], duration, ends[key], selected, f"interval_{key}_{unit}"
                )
                equipment_intervals[unit].append(interval)
                model.add(durations[key] == duration).only_enforce_if(selected)
                model.add(
                    starts[key] >= ticks(problem.equipment_available[unit] - problem.now)
                ).only_enforce_if(selected)
            model.add_exactly_one(selectors)
        for intervals in equipment_intervals.values():
            if intervals:
                model.add_no_overlap(intervals)

        operation_map = {op.key: op for op in problem.operations}
        waits = []
        for operation in problem.operations:
            for predecessor in operation.predecessors:
                model.add(
                    starts[operation.key]
                    >= ends[predecessor] + ticks(operation_map[predecessor].rest_after)
                )
        for member, available in problem.member_available.items():
            member_ops = [op for op in problem.operations if op.member_id == member]
            if not observation.allow_reordering:
                previous = None
                for operation in member_ops:
                    ready = (
                        ticks(available - problem.now)
                        if previous is None
                        else (ends[previous.key] + ticks(previous.rest_after))
                    )
                    wait = model.new_int_var(0, horizon, f"wait_{operation.key}")
                    model.add(wait == starts[operation.key] - ready)
                    waits.append(wait)
                    previous = operation
            else:
                # A Hamiltonian path through a depot specifies the actual next
                # block and therefore its waiting, including predecessor rest.
                arcs = []
                for j, operation in enumerate(member_ops, start=1):
                    wait = model.new_int_var(0, horizon, f"wait_{operation.key}")
                    waits.append(wait)
                    first = model.new_bool_var(f"first_{operation.key}")
                    last = model.new_bool_var(f"last_{operation.key}")
                    arcs.extend([(0, j, first), (j, 0, last)])
                    model.add(
                        wait == starts[operation.key] - ticks(available - problem.now)
                    ).only_enforce_if(first)
                    for i, predecessor in enumerate(member_ops, start=1):
                        if i == j:
                            continue
                        follows = model.new_bool_var(f"follows_{predecessor.key}_{operation.key}")
                        arcs.append((i, j, follows))
                        model.add(
                            wait
                            == starts[operation.key]
                            - ends[predecessor.key]
                            - ticks(predecessor.rest_after)
                        ).only_enforce_if(follows)
                model.add_circuit(arcs)
        model.minimize(sum(waits))

        # A feasible observable heuristic helps the tightly budgeted solver.
        hint = decode(problem, initial_genes(problem))
        for operation in hint.operations:
            model.add_hint(starts[operation.key], ticks(operation.start - problem.now))
            for unit in operation_map[operation.key].equipment_ids:
                model.add_hint(choices[operation.key, unit], int(unit == operation.equipment_id))
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = int(self.context.seed % (2**31 - 1))
        solver.parameters.max_time_in_seconds = float(self.options["time_limit_seconds"])
        solver.parameters.max_deterministic_time = float(self.options["deterministic_time_limit"])
        status = solver.solve(model)
        diagnostics = {
            "solver_status": str(solver.status_name(status)),
            "solver_status_code": int(status),
            "solver_wall_seconds": float(solver.wall_time),
            "planner_wall_seconds": float(time.perf_counter() - started),
            "forecast_operations": len(problem.operations),
            "fallback": False,
        }
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return self._fallback(observation, candidates, diagnostics)
        planned = []
        for member in problem.member_available:
            member_ops = sorted(
                (op for op in problem.operations if op.member_id == member),
                key=lambda op: (solver.value(starts[op.key]), op.key),
            )
            ready = problem.now + ticks(problem.member_available[member] - problem.now) * resolution
            for operation in member_ops:
                unit = next(
                    unit
                    for unit in operation.equipment_ids
                    if solver.value(choices[operation.key, unit])
                )
                start = problem.now + int(solver.value(starts[operation.key])) * resolution
                finish = problem.now + int(solver.value(ends[operation.key])) * resolution
                planned.append(
                    PlannedOperation(
                        member,
                        operation.visit_id,
                        unit,
                        start,
                        finish,
                        ready,
                        ticks(operation.rest_after) * resolution,
                        operation.deviations[operation.equipment_ids.index(unit)],
                    )
                )
                ready = finish + ticks(operation.rest_after) * resolution
        schedule = summarize_schedule(problem, planned)
        self.last_schedule = schedule
        objective = float(solver.objective_value * resolution)
        bound = float(solver.best_objective_bound * resolution)
        diagnostics.update(
            {
                "objective_seconds": objective,
                "best_bound_seconds": bound,
                "relative_gap": float(
                    max(0.0, objective - bound) / max(abs(objective), resolution)
                ),
                "forecast_tardiness_seconds": float(schedule.tardiness_seconds),
                "planner_wall_seconds": float(time.perf_counter() - started),
            }
        )
        recommendations = immediate_recommendations(problem, schedule, candidates, resolution / 2)
        return Decision(recommendations, defer=not recommendations, diagnostics=diagnostics)


def order_crossover(parent, other, rng):
    """Preserve one parent's priority segment and the other's remaining order."""
    n = len(parent)
    if n < 2:
        return parent
    first = sorted(range(n), key=lambda i: (parent[i][0], i))
    second = sorted(range(n), key=lambda i: (other[i][0], i))
    left, right = sorted(rng.sample(range(n + 1), 2))
    segment = first[left:right]
    remaining = iter(i for i in second if i not in set(segment))
    order = [first[i] if left <= i < right else next(remaining) for i in range(n)]
    ranks = {index: (rank + 0.5) / n for rank, index in enumerate(order)}
    return tuple((ranks[i], parent[i][1] if rng.random() < 0.5 else other[i][1]) for i in range(n))


class GAPolicy(BasePolicy):
    defaults: ClassVar[dict[str, Any]] = {
        "population_size": 32,
        "tournament_size": 3,
        "elites": 2,
        "crossover_probability": 0.8,
        "mutation_probability": 0.2,
        "budget_mode": "evaluations",
        "max_evaluations": 256,
        "time_limit_seconds": 0.1,
        "unknown_busy_seconds": 300.0,
    }

    def __init__(self, name="ga", options=None):
        super().__init__(name, options)
        for key in ("population_size", "tournament_size", "elites", "max_evaluations"):
            value = self.options[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if (
            self.options["population_size"] < 2
            or self.options["elites"] >= self.options["population_size"]
        ):
            raise ValueError("population must have at least two members and exceed elites")
        for key in ("crossover_probability", "mutation_probability"):
            if not 0 <= self.options[key] <= 1:
                raise ValueError(f"{key} must be in [0, 1]")
        if self.options["budget_mode"] not in ("evaluations", "wall"):
            raise ValueError("budget_mode must be 'evaluations' or 'wall'")
        if self.options["time_limit_seconds"] < 0 or self.options["unknown_busy_seconds"] < 0:
            raise ValueError("time limits must be nonnegative")
        self.random = random.Random(self.context.seed)
        self.warm_genes: dict[tuple[str, str], tuple[float, str]] = {}
        self.last_schedule: ForecastSchedule | None = None

    def reset(self, context: PolicyContext) -> None:
        super().reset(context)
        self.random = random.Random(context.seed)
        self.warm_genes = {}
        self.last_schedule = None

    def decide(self, observation: Observation, candidates: tuple[Candidate, ...]) -> Decision:
        started = time.perf_counter()
        self.last_schedule = None
        if not candidates:
            return Decision(defer=True, diagnostics={"evaluations": 0, "fallback": False})
        problem = build_problem(observation, self.options["unknown_busy_seconds"])
        if not problem.operations:
            return Decision(defer=True, diagnostics={"evaluations": 0, "fallback": False})
        default = initial_genes(problem)
        warm = []
        warm_used = 0
        for index, operation in enumerate(problem.operations):
            previous = self.warm_genes.get(operation.key)
            if previous and previous[1] in operation.equipment_ids:
                warm.append(
                    (
                        previous[0],
                        (operation.equipment_ids.index(previous[1]) + 0.5)
                        / len(operation.equipment_ids),
                    )
                )
                warm_used += 1
            else:
                warm.append(default[index])
        evaluations = 0
        population = []
        best = None

        def evaluate(genes):
            nonlocal evaluations, best
            schedule = decode(problem, genes)
            item = (float(schedule.waiting_seconds), genes, schedule)
            evaluations += 1
            if best is None or item[:2] < best[:2]:
                best = item
            return item

        def budget_available():
            if self.options["budget_mode"] == "evaluations":
                return evaluations < self.options["max_evaluations"]
            return time.perf_counter() - started < self.options["time_limit_seconds"]

        # Always return a feasible incumbent, even under a zero wall-time budget.
        initial_decode_start = time.perf_counter()
        population.append(evaluate(default))
        initial_decode_seconds = time.perf_counter() - initial_decode_start
        if warm_used and budget_available():
            population.append(evaluate(tuple(warm)))
        while len(population) < self.options["population_size"] and budget_available():
            genes = tuple((self.random.random(), self.random.random()) for _ in problem.operations)
            population.append(evaluate(genes))
        generations = 0
        while budget_available():
            population.sort(key=lambda item: item[:2])
            next_population = population[: min(self.options["elites"], len(population))]

            def tournament(pool=population):
                entrants = [
                    self.random.choice(pool) for _ in range(self.options["tournament_size"])
                ]
                return min(entrants, key=lambda item: item[:2])[1]

            while len(next_population) < self.options["population_size"] and budget_available():
                parent = tournament()
                other = tournament()
                crossover = self.random.random() < self.options["crossover_probability"]
                combined = order_crossover(parent, other, self.random) if crossover else parent
                child = []
                for priority, equipment in combined:
                    if self.random.random() < self.options["mutation_probability"]:
                        priority = self.random.random()
                    if self.random.random() < self.options["mutation_probability"]:
                        equipment = self.random.random()
                    child.append((priority, equipment))
                next_population.append(evaluate(tuple(child)))
            population = next_population
            generations += 1
        assert best is not None
        _, genes, schedule = best
        self.last_schedule = schedule
        self.warm_genes = {
            operation.key: (
                genes[index][0],
                operation.equipment_ids[
                    min(
                        len(operation.equipment_ids) - 1,
                        int(genes[index][1] * len(operation.equipment_ids)),
                    )
                ],
            )
            for index, operation in enumerate(problem.operations)
        }
        advice = immediate_recommendations(problem, schedule, candidates)
        elapsed = time.perf_counter() - started
        incumbent_only = self.options["budget_mode"] == "wall" and evaluations == 1
        return Decision(
            advice,
            defer=not advice,
            diagnostics={
                "objective_seconds": float(schedule.waiting_seconds),
                "forecast_tardiness_seconds": float(schedule.tardiness_seconds),
                "forecast_operations": len(problem.operations),
                "evaluations": int(evaluations),
                "generations": int(generations),
                "warm_start_operations": int(warm_used),
                "fallback": bool(incumbent_only),
                "incumbent_only": bool(incumbent_only),
                "initial_decode_seconds": float(initial_decode_seconds),
                "budget_overrun_seconds": float(
                    max(0.0, elapsed - self.options["time_limit_seconds"])
                )
                if self.options["budget_mode"] == "wall"
                else 0.0,
                "budget_mode": str(self.options["budget_mode"]),
                "planner_wall_seconds": float(elapsed),
            },
        )
