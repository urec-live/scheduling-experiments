# UREC Live scheduling experiments

A standalone Python laboratory for comparing advisory gym schedulers on the **same saved synthetic workloads**. It separates physical gym state, information available to a scheduler, and member responses. Production UREC Live code and services are not required.

Start with the deterministic example, inspect its timeline, then introduce uncertainty and human behavior. The software provides experiment infrastructure; a smoke run is not a completed thesis study or evidence of real-gym effectiveness.

## Setup and first experiment

For a small step-by-step introduction, start with
[E01-A: three members and two machines](research/e01a-walkthrough.md).
It compares independent choice with fully accepted FCFS guidance in just two runs.

Next is [E01-B: accept, refuse, or a fixed member split](research/e01b-responses.md).
Members now choose independently immediately after refusal or failed access.
The previous E01 pilot used a removed 30-second pause and remains historical evidence.

For the scenario spreadsheet's **E01 baseline study**, follow the
[E01 runbook](research/e01.md). Its executable configuration is
`configs/experiments/e01.yaml`: ten regimes, five policies, and thirty paired
development episodes per regime (1,500 runs). Start with `--split train` and
review the pilot before validation or final-test selection.

From this directory, with `uv` installed:

```bash
uv sync --python 3.12
uv run pytest
uv run gym-sched generate --config configs/scenarios/demo.yaml --seed 42 --output outputs/scenarios/demo-42.json
uv run gym-sched compare --scenario outputs/scenarios/demo-42.json --policies fcfs,spt --config configs/scenarios/demo.yaml --output outputs/demo
uv run gym-sched report --run outputs/demo
```

The `generate` command also works without `--output` and prints the saved path. `compare` runs each policy against that saved scenario; `report` reads artifacts without rerunning policies. Use a new output directory for a new comparison to keep provenance distinct.

Verify the hand-calculated example:

```bash
uv run gym-sched compare --scenario data/fixtures/three_members.json --policies fcfs,spt --output outputs/hand-example
uv run gym-sched report --run outputs/hand-example
```

For three simultaneous six-, two-, and four-minute visits on one unit, FCFS mean waiting is **280 seconds** and SPT is **160 seconds**. Both finish at 720 seconds. These values check simulation and waiting semantics, not a general claim that SPT is best.

## Scenarios and policies

| Configuration | Purpose |
| --- | --- |
| `configs/scenarios/demo.yaml` | Eight units, twenty members, four visits, fixed heterogeneous durations, perfect compliance. |
| `configs/scenarios/stochastic.yaml` | Independent duration variability and forecast noise, conditional Poisson arrivals. |
| `configs/scenarios/advisory.yaml` | Bursts, partial adoption, refusals, observation delay/dropout, deadlines, patience. |
| `configs/scenarios/flexibility.yaml` | Separate order/substitution experiment with both workload and action permissions changed. |

Scenario YAML contains `scenario:` generator settings and `simulation:` dynamics. Generation saves the workload; pass `--config` to comparison to apply its simulation settings. Default dynamics are deterministic/perfect-observation settings, so omitting an advisory config changes the intended experiment.

All units are seconds unless stated otherwise. The generator's `poisson` option fixes the requested member count and samples the conditional arrival times; it does not draw an unconditioned Poisson member count. Values and human behavior are uncalibrated assumptions.

| Policy name | Behavior |
| --- | --- |
| `self_directed` | Independent member choice with physical contention resolution. |
| `fcfs` | Oldest ready request first. |
| `spt` | Shortest predicted visit first. |
| `aging` | Predicted duration minus accumulated waiting. |
| `matching` | Maximum-cardinality minimum-cost batch matching. |
| `robust` | Matching with acceptance, uncertainty, and stability mechanisms. |
| `robust_a0u0s0` … `robust_a1u1s1` | All eight mechanism combinations on the same planner. |
| `cpsat` | Rolling-horizon constrained scheduling from observable forecasts. |
| `ga` | Genetic priorities/equipment choices with feasible decoding. |

Example advisory comparison:

```bash
uv run gym-sched generate --config configs/scenarios/advisory.yaml --seed 42 --output outputs/scenarios/advisory-42.json
uv run gym-sched compare --scenario outputs/scenarios/advisory-42.json --policies self_directed,fcfs,spt,aging,matching,robust,cpsat,ga --config configs/scenarios/advisory.yaml --policy-config configs/policies/defaults.yaml --output outputs/advisory
uv run gym-sched report --run outputs/advisory
```

`configs/policies/defaults.yaml` maps policy names to options. GA defaults to fixed evaluations for reproducible comparisons; latency mode is a distinct experiment. Total measured decision time includes more than solver time. Inspect timeouts, fallbacks, and overruns rather than assuming an internal time limit guarantees the total budget.

## Repeated experiments and thesis workflow

```bash
uv run gym-sched experiment --config configs/experiments/smoke.yaml --split train
uv run gym-sched experiment --config configs/experiments/pilot.yaml --split train
```

The smoke experiment is small. The pilot template has thirty scenarios per family; it may take substantially longer. `train` means development data for these untrained policies. Use separate validation to select settings and the reference; freeze the protocol before opening final test results. Run `uv run gym-sched --help` and subcommand `--help` for output overrides and protocol/selection tools.

The supplied panel configurations are editable research templates, not a finished final-study specification. Read [the protocol](research/protocol.md) before interpreting repeated results. A saved protocol snapshot does not replace independent preregistration or make a test set inaccessible.

Each comparison/experiment saves a manifest and per-policy run artifacts, including:

- Scenario/configuration snapshots and hashes; source revision/hash, dependency versions, hardware, and seeds.
- `result.json`, `events.jsonl`, `decisions.jsonl`, `members.csv`, and `equipment.csv`.
- Metrics, failed-run status, audit findings, solver/search diagnostics, and measured latency.
- Reports and exported equipment timelines, waiting distributions, and paired-comparison figures.

Generated artifacts belong under `outputs/` and are ignored by Git. Commit source/configuration changes before a final study; uncommitted source hashes improve traceability but do not replace an archived revision. Never omit failed episodes or infer confidence from a single scenario.

## Model semantics and limitations

Waiting is time present and ready, excluding active visits, prescribed rest, and walking. Pending-response delay counts as waiting when the member is ready. Abandoning members remain in the waiting denominator. Report completion and tail waits alongside means.

Advice does not reserve equipment. A member may reject it or arrive after a unit becomes busy. Observed availability drives recommendations; actual capacity governs starts. Pending advice and accepted walks remain in place until their outcome. Later advice can incur a stability penalty relative to previous suggestions; the initial protocol does not redirect members while walking.

Same-time phases release equipment, process lifecycle events, resolve physical attempts, deliver observations, then decide. Hidden durations, future arrivals, latent willingness/patience, and true release times never belong in policy input. A deadline blocks new starts but permits an ongoing visit to finish. Final active work can therefore create overtime.

The engine reports completed, stalled, and truncated episodes distinctly. State/behavior models are simplified and synthetic. Fixed workloads and fixed evaluation budgets can reproduce decisions; wall-clock optimization may vary with machine load. A solver's `FEASIBLE` status is not proof of optimality. Historical replay and synthetic results do not establish counterfactual benefits to real gym members.

## Research documents and extension path

- [Full research plan](RESEARCH_PLAN.md): questions, architecture, algorithms, validation, and sixteen-week roadmap.
- [Protocol](research/protocol.md): splits, precision targets, paired inference, guardrails, and reporting.
- [Assumptions](research/assumptions.md): model choices and information boundary.
- [Reading map](research/reading_map.md): supplied paper IDs and primary technical references.
- [Claims ledger](research/claims.md): what is supported and what remains a hypothesis.
- [App integration](research/app_integration.md): future read-only calibration and missing telemetry.

Keep notebooks analytical; core rules live in tested modules. Use the policy interface to add methods without changing the simulator. An optional offline NSGA-II helper is implemented in `gym_sched.policies.pareto.nsga2_frontier(observation)`. It uses the GA decoder and returns forecast trade-offs for total waiting, worst-member waiting, workout deviation, and changes to unfulfilled advice. These forecast points must subsequently be tested in closed-loop episodes; they are not claims of a globally optimal Pareto frontier.

The Gymnasium wrapper contract is documented in `src/gym_sched/adapters/README.md`. A live wrapper, trained RL/GNN policies, and a learned GA hybrid remain gated extensions. The gate is a frozen, validated classical benchmark, followed by nongraph learning, a controlled graph comparison, and one GA hybrid at matched computation budget.

Optional dependencies, when those branches require them:

```bash
uv sync --extra ga --extra learning
```

Installing these packages does not train a model or establish an experimental result.

To exercise the implemented NSGA-II helper's tests, use `uv run --extra ga pytest`.

## Protocol commands

```bash
uv run gym-sched experiment --config configs/experiments/smoke.yaml --split validation --output outputs/validation
uv run gym-sched select-validation --run outputs/validation
uv run gym-sched pilot-count --run outputs/validation --split validation --baseline fcfs --candidate spt
```

Reference selection compares classical policies using equal-family mean waiting,
with completion, tail, and latency checks. Set the experiment YAML's `baseline`
to the selected name before freezing with `--selection`. Then:

```bash
uv run gym-sched freeze-protocol --config configs/experiments/smoke.yaml --selection outputs/validation/selection.json --output outputs/protocol.json
uv run gym-sched experiment --config configs/experiments/smoke.yaml --split test --protocol outputs/protocol.json --output outputs/heldout
```

The smoke test split has only two seeds and remains exploratory even with a frozen
protocol. Final-study configurations need the pilot-derived 100–500 seeds per
principal regime. Source/config changes invalidate a freeze. Saved-result hashes
are checked before reporting. Pilot precision is calculated independently per family;
`--half-width-seconds` overrides the default `max(30 seconds, 5% of reference mean)`.

Other panel templates are `congestion.yaml`, `observation.yaml`, and `ablations.yaml`.
Grid keys can target `scenario.*` or `simulation.*`. Observation delay, duration
uncertainty, adoption, and willingness should be varied separately before combined
stress tests. A policy's `seed` option controls its search RNG independently of
scenario seeds.
# Visual gym demo

Run `uv run python -m gym_sched.visual_demo` and open
`outputs/visual-gym/index.html` in a browser. The exported page is self-contained
and works offline. To serve it locally, run
`uv run python -m http.server 8765 --bind 127.0.0.1 --directory outputs/visual-gym`
and visit `http://127.0.0.1:8765`.

The first demo implements the small S05/S43 comparison: two machines, one
independent occupant, and one guided member completing two exercises. Switch
between fixed order and permitted reordering, play/pause, step through event
times, scrub the timeline, and inspect either member. Simultaneous events are
applied together so the floor shows a consistent state at each timestamp.
The completed-run comparison is explicitly a result, not a future prediction.

Select a member and use **View complete profile** for all declared plan fields,
per-exercise equipment/alternative choices, current order permissions, time
settings, latent response assumptions and shared run conditions. A complete JSON
disclosure preserves every underlying member/configuration field. Future profile
fields are explicitly unspecified or unmodeled; they do not change behavior.
These are intended-system example profiles, not profiles from production accounts.

Both replays are generated by the existing simulator and FCFS policy from the
same scenario. No browser-side scheduler or production connection is involved.
The page shows actual activity to the experiment observer; its optional sensor
labels use logged observations. These views agree with perfect sensing here.
The initial version does not implement the rest of the spreadsheet, arbitrary
user behavior, walking animation, false sensing, or runtime plan editing.
Add these as separate controlled experiments after validating this baseline.


Interactive order-permission experiment: run `uv run python -m gym_sched.visual_server`
then open http://127.0.0.1:8765/. Casey’s profile allows editing only exercise order;
Run experiment & save invokes the existing simulator, reloads both comparison runs,
and selects the requested condition. Other member and sensor settings remain fixed.
The source workload is `data/fixtures/visual_two_members.json`. Each invocation saves
`scenario.json`, `config.json`, `result.json`, and `manifest.json` under
`outputs/visual-gym/runs/<run-id>/`. The global simulation order flag affects Casey
in this tiny experiment; Alex has one exercise. This does not implement general
per-member order permissions. Top order buttons preview existing results; they
do not edit or save the profile setting. Offline exports support replay only.
These deterministic synthetic results demonstrate the mechanism, not real-gym benefit.


Experiment 2: refusal/deviation likelihood. Casey’s profile offers a percentage
from 0 to 100 (default 50). This applies to every recommendation: acceptance
probability is 1 minus the refusal fraction, with existing seeded per-response
draws. No timed willingness change or guaranteed first refusal remains.
Deviation here means not following advice and choosing independently; changing
course after acceptance is not modeled. Independent choice keeps original plan
order through the latent `independent_keep_plan_order` member field (default
false outside this experiment). It never enters policy observations/candidates.
Both response runs allow reordering and use one-second responses. The comparison
is 0% refusal versus the configured percentage, with identical work and seed 42.
A saved run records the actual probability in scenario.json and the requested
percentage in manifest.json. Repeating a configuration reproduces the events;
a single seeded run is not an empirical refusal-rate estimate. No reservations
or forced refusal cooldowns are added. Tests cover 0%, 100%, intermediate values,
invalid input, repeatability, start order, and condition-specific saved truth.
At 100%, Casey refuses both recommendations, waits for A, and later independently
uses B. Expected waiting is 181 seconds and elapsed 481 seconds (including response
time); baseline waiting is 2 seconds and elapsed 302. No third member is included.


Experiment 3: competing member. Select “3 · Competing member” in Casey’s
profile and run with the same refusal percentage. The readable workload is
`data/fixtures/visual_three_members.json`: Alex and Casey are unchanged; Sam
arrives at simulation second 3, has one 120-second cable-row exercise, and always
accepts. This arrival is after Casey’s first response at second 2, deliberately
isolating real availability from simultaneous claims. Response delay remains
one second for both participants. The same existing simulator/FCFS policy runs
0% refusal versus the configured Casey probability. Advice is not a reservation.
At 0%, Casey starts B at 2, releases it at 122, and Sam receives advice at 122,
starts at 123, and finishes at 243 (120 seconds waiting including response).
At 100%, Casey refuses at 2; Sam gets advice at 3, starts at 4, and finishes
at 124 (one second waiting). Casey’s wait/order remains unchanged. The replay
includes Sam’s marker, full profile, actual state, advice, and outcome rows.
Saved runs include all three members. Tests verify advice/start times, completed
work, unchanged Casey configuration, and no overlap in equipment occupancy.
This does not test simultaneous recommendations, stale sensing, or travel races
and remains a synthetic mechanism check, not evidence of real-gym effectiveness.


Experiment 4: walking conflict. Select “4 · Walking and competing advice” in
Casey’s profile. The fixture `data/fixtures/visual_simultaneous_arrivals.json`
keeps Alex on A; Casey and Sam arrive together at second 1, each with one
120-second B exercise. Both always accept. Casey’s two-exercise plan is reduced
to B only to isolate one shared resource. FCFS runs a zero-walking baseline and
a fixed ten-second walk for all members (including Alex), with one-second
responses, immediate sensing, and no substitutions. Neither advice nor travel
reserves equipment. The existing physical start check enforces single occupancy.
With walking: Casey gets advice at 1, accepts/walks at 2; Sam gets advice at 2,
accepts/walks at 3. Casey starts B at 12. Sam fails at 13, immediately returns
to independent choice, waits for release at 132, walks again, and starts at 142.
There is no forced cooldown or repeated advice. Equal arrivals do not require
simultaneous advice: FCFS avoids duplicate equipment within one decision batch,
but later decisions may recommend it while the first member is in transit.
The zero-walk baseline has no failed start; Sam is offered B after release.
The UI animates walking, logs failed access, and displays Sam’s failed starts,
waiting, and total visit duration. Tests verify shared workload, competing offers
before occupancy, one failure, eventual completion, and no occupancy overlap.
Member-ID ordering determines this synthetic winner; this is not a fairness
guarantee. No sensor delay or actual hardware is introduced in this experiment.


Visual regression suite: `uv run python -m gym_sched.visual_regression`.
This runs four experiments with eight conditions using the same engine and
payload builder as the UI. Expected member outcomes are explicitly specified
in `data/fixtures/visual_regression.json`; they are not regenerated from current
simulator results. Order and walking conditions share the same workload; refusal
and competition compare 0% against 100% refusal with fixed seed 42.
Each condition checks expected waiting, elapsed time, recovery, walking, completed
visits, refusals and failed starts, plus the existing validator’s capacity, event,
and time-accounting audits. All members must complete their work. Detailed
advice/start-order assertions remain in pytest.
The output `outputs/visual-regression/` contains report.md, report.json, a copy
of the expectations, and per-experiment payload.json plus offline replay.html.
A failed build is recorded and remaining experiments continue; any failure
returns a nonzero exit status. Re-running replaces this report directory’s files.
Offline replays support inspection; editing/running requires the interactive
server. Tests deliberately corrupt outcomes/events and inject a build failure
to verify fault reporting. These are small deterministic regression guards,
not larger-gym scheduling benchmarks or measured real-gym benefit. Gym size,
arrivals, hardware modeling and policies remain unchanged in this step.


Configurable gym UI: run `uv run python -m gym_sched.visual_server`, then open
http://127.0.0.1:8765/configure. The existing four experiments remain unchanged.
The new generator exposes machine counts (two initial types), member count,
independent-member count, simultaneous/regular/uniform-window arrivals, seed,
exercise/recovery durations and refusal likelihood. Zero copies of either kind
are allowed, but at least one machine is required. Each member has one exercise
per available kind; alternating plan order is a declared synthetic assumption.
All copies of a kind are compatible equivalents. Every prediction initially
equals the actual duration. Uniform arrivals use keyed seeded draws.
Structured controls expose ordering/substitutions, walking/response/decision
delays, and sensor delay/dropout/interval/expiry. The member inspector edits
arrival, refusal, participation and independent order; full member JSON exposes
all exercise settings and remaining latent fields. Full scenario/config editors
provide every existing model field, including new equipment kinds, outages,
prerequisites, deadlines and alternative-equipment metadata. Generated workloads
are starting assumptions, not calibrated real-gym behavior. Tracking-only is
not a separate member mode. Physical room size and routes are not modeled.
A profile edit must be applied before running; profile arrival edits extend the
scenario arrival window. Whole-scenario JSON must declare a valid arrival window.
The replay uses the last executed scenario while drafts remain independently
editable. The dynamic floor reflects actual machine/member counts, observations
and events; it is an illustrative layout, not path geometry.
Runs still use the existing FCFS engine. Server validation supports at most
50 members, 20 machines, 300 visits, 200000 events and 86400 simulation seconds.
Truncated runs are saved with audit findings and are not reported as successes.
Each run saves scenario/config/result/payload JSON plus an offline replay under
outputs/visual-gym/runs/custom-*/. The latest executed configuration/result is
restored on refresh; draft download/upload enables explicit reuse. Generated
settings stored in metadata describe the original generator inputs, not later
manual edits. Tests verify counts, compatible units, arrival patterns, seeds,
capacity-related waiting, profile/config propagation, saved repeatability, input
limits and preserved truncated results. These are simulation checks only.

The configurable UI also provides “Run preset regression checks” with pass/fail
results and replay links; no terminal command is needed for those eight checks.
