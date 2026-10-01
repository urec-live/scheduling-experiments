"""Optional offline NSGA-II exploration, separate from the live policy factory.

The objectives are forecast total waiting, worst member waiting, approved
workout deviation and changes to unfulfilled advice. These are forecast surrogates, not
closed-loop episode performance. Use held-out simulation to evaluate selected
points. This helper uses the same feasible serial decoder as the online GA.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gym_sched.domain.models import Observation
from gym_sched.policies.forecast import ForecastSchedule, build_problem, decode


@dataclass(frozen=True)
class ParetoPoint:
    waiting_seconds: float
    maximum_member_waiting_seconds: float
    deviation: float
    recommendation_changes: float
    schedule: ForecastSchedule

    @property
    def objectives(self) -> tuple[float, float, float, float]:
        return (
            self.waiting_seconds,
            self.maximum_member_waiting_seconds,
            self.deviation,
            self.recommendation_changes,
        )


def nsga2_frontier(
    observation: Observation,
    *,
    seed: int = 0,
    population_size: int = 32,
    generations: int = 20,
    unknown_busy_seconds: float = 300.0,
) -> tuple[ParetoPoint, ...]:
    """Search a forecast Pareto frontier with actual pymoo NSGA-II.

    Installing the package's ``ga`` extra enables this helper. There is no
    silently substituted algorithm when pymoo is unavailable. Rounded objective
    duplicates are removed from the result; every returned schedule is decoded
    and feasible under the supplied observable forecast, not latent gym truth.
    """
    if population_size < 2 or generations < 1:
        raise ValueError("population_size must be >= 2 and generations >= 1")
    if seed < 0 or unknown_busy_seconds < 0:
        raise ValueError("seed and unknown_busy_seconds must be nonnegative")
    try:
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.optimize import minimize
    except ImportError as error:
        raise ImportError("NSGA-II requires the optional 'ga' dependency group (pymoo)") from error
    problem = build_problem(observation, unknown_busy_seconds)
    if not problem.operations:
        return ()
    previous = {
        m.id: m.previous_recommendation
        for m in observation.members
        if m.previous_recommendation is not None
    }

    def unpack(vector):
        return tuple(
            (float(vector[2 * i]), float(vector[2 * i + 1])) for i in range(len(problem.operations))
        )

    def objectives(schedule):
        return (
            float(schedule.waiting_seconds),
            float(max(schedule.member_waiting.values(), default=0.0)),
            float(schedule.deviation),
            float(
                sum(
                    op.member_id in previous
                    and previous[op.member_id][0] == op.visit_id
                    and previous[op.member_id][1] != op.equipment_id
                    for op in schedule.operations
                )
            ),
        )

    class SchedulingProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=2 * len(problem.operations), n_obj=4, xl=0.0, xu=1.0)

        def _evaluate(self, vector, out, *args, **kwargs):
            out["F"] = np.asarray(objectives(decode(problem, unpack(vector))), dtype=float)

    result = minimize(
        SchedulingProblem(),
        NSGA2(pop_size=population_size),
        ("n_gen", generations),
        seed=seed,
        verbose=False,
    )
    if result.X is None:
        return ()
    points = {}
    for vector in np.atleast_2d(result.X):
        schedule = decode(problem, unpack(vector))
        values = objectives(schedule)
        identity = tuple(round(value, 9) for value in values)
        points.setdefault(identity, ParetoPoint(*values, schedule))
    return tuple(points[key] for key in sorted(points))
