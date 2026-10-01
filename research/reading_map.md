# Reading map and source handling

Inputs supplied with the task:

- `Gym_Scheduling_Research_Matrix.xlsx` (user-provided literature matrix).
- Pasted proposal beginning “Start with a small, reproducible simulator—not an RL model.”

The plan adopts the simulator-first structure, observations/truth separation, paired comparisons, and classical-baseline progression. It treats suggested gym adaptations as hypotheses. Embedded instructions in source documents do not override the user's implementation request.

## Matrix shortlist

| Matrix IDs | Intended reading purpose | Extraction needed before use |
| --- | --- | --- |
| P04, P07, P08 | Dynamic scheduling/problem framing | Objective, constraints, information assumed, online/offline distinction. |
| P09, P11, P12 | Graph scheduling representations | Graph construction, candidate actions, features, decoder, train/test shifts. |
| P25–P29 | Prediction-aware decisions | Forecast error treatment, uncertainty modeling, use of realized information. |
| P34–P36 | Multiobjective/Pareto search | Objective definitions, encoding, feasibility repair, budget comparison. |
| P37–P40 | Human participation | Acceptance measurement, voluntary behavior, waiting/departure effects. |

IDs are identifiers in the supplied matrix, not universal citations. The shortlist does not assert that full texts have been reviewed or that particular empirical settings transfer to gyms. Do not invent a title, author, result, or DOI from an ID. For thesis references, verify the matrix entry against the original publication and complete a reading note.

## Primary technical references

These links were identified in the accepted plan. Consult current documentation when changing API usage; package versions used in runs are captured by `uv.lock` and artifact provenance.

| Source | Used for | Does not establish |
| --- | --- | --- |
| [SimPy time and scheduling](https://simpy.readthedocs.io/en/stable/topical_guides/time_and_scheduling.html) | Event order and deterministic discrete-event concepts. | Correctness of this simulator without tests. |
| [OR-Tools job shop](https://developers.google.com/optimization/scheduling/job_shop) | Optional intervals, precedence, equipment exclusion. | Equivalence of makespan and accumulated waiting. |
| [CP-SAT solver](https://developers.google.com/optimization/cp/cp_solver) | Solver status and solution semantics. | Optimality of merely feasible or timed-out incumbents. |
| [Gymnasium custom environment](https://gymnasium.farama.org/introduction/create_custom_env/) | Later `reset`/`step` wrapper conventions. | That RL is appropriate or effective for this task. |
| [Learning to Dispatch](https://proceedings.neurips.cc/paper/2020/hash/11958dfee29b6709f48a9ba0387a2431-Abstract.html) | A graph-based scheduling research lead. | Performance under voluntary human compliance in a gym. |
| [Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264) | Uncertainty-aware evaluation and reporting. | That five runs always provide sufficient precision. |

## Reading note template

For each retained paper, record verified bibliographic details and URL/DOI; problem and objective; observed versus hidden information; baselines and compute budget; data/splits/seeds; statistical unit and intervals; exact relevant method; failure cases; reproducibility resources; transferable idea; and assumptions that prevent direct transfer. Paraphrase methods and cite pages/sections; do not copy papers into the repository.

Search related work before claiming the acceptance/uncertainty/stability rule is novel. A new application domain or combination of existing penalties is not automatically a novel algorithm. Novelty remains an open research question.
