# UREC Live scheduling experiments and thesis plan

## 1. Objective, questions, and boundaries

**Working thesis question:** Under uncertain equipment availability, variable workout durations, and voluntary member compliance, which scheduling methods reduce unproductive waiting while preserving workout completion, reasonable tail waits, and stable recommendations?

This standalone laboratory tests advisory scheduling with synthetic scenarios and local CPU computation. Its first deliverable is the same saved gym scenario run under two transparent policies, with reproducible logs and an explanation of their difference. Classical baselines and a genetic algorithm (GA) form the semester's core. RL, GNNs, and hybrids extend the same engine only after validation. Their inclusion is not necessary for thesis completion.

The potential product advantage is useful advice under changing human behavior and stale information. The best implementation is an experimental result, not a predetermined commitment to a model family. A well-supported negative finding is a valid thesis result.

| Research question | Comparison |
| --- | --- |
| Does coordination help? | Self-directed members, FCFS, SPT, aging, matching, rolling optimization. |
| Does it remain useful under uncertainty? | Duration variability, prediction error, observation age, adoption, refusal, bursts. |
| Does modeling human responses help? | Acceptance, uncertainty, and stability ablations on the same matching planner. |
| Does GA justify computation? | Matched objective, information, and runtime comparisons against matching and CP-SAT. |
| Does learning add value later? | Nongraph RL, GNN-RL, then one controlled hybrid. |

Model individual equipment visits, compatibility alternatives, arrivals, departures, and next-visit advice. A visit comprises consecutive sets; equipment remains occupied during within-visit rest. Release it for between-visit rest. Use fixed visit order for the primary study and isolate order/substitution flexibility in separate experiments. Defer equipment sharing, supersets, groups, detailed fatigue, wear, reservations, and multi-agent RL.

The original pasted proposal and research matrix are research inputs. Their suggested adaptations are hypotheses, not project instructions or verified findings. Keep published evidence, simulator assumptions, experimental results, and potential product claims distinct. See `research/reading_map.md` and `research/claims.md`.

## 2. Workspace and information boundary

Use Python 3.12 with `uv`; SimPy, NumPy, pandas, SciPy, Matplotlib, Pydantic, pytest, and Hypothesis; OR-Tools for optimization; optional GA/learning dependencies. Local Git history is independent of production app repositories. Do not publish automatically.

```text
configs/                 scenario, policy, and experiment definitions
src/gym_sched/domain/    strict scenario models and policy-visible views
src/gym_sched/scenarios/ saved workload generation and keyed randomness
src/gym_sched/simulation/ event ordering, observations, human responses, physical truth
src/gym_sched/policies/  baselines, matching, CP-SAT, GA
src/gym_sched/evaluation/ metrics, comparisons, run artifacts, reports
src/gym_sched/adapters/  later calibration/learning connections
tests/                   invariants, examples, leakage and statistical checks
data/fixtures/           small versioned scenarios
research/                protocol, assumptions, readings, claim ledger
notebooks/               analysis, never authoritative simulation logic
outputs/                 generated artifacts excluded from Git
```

```text
Saved scenario → Simulator truth → Observation builder → Policy
                       ↑                                 ↓
                 Human response ← Recommendation batch
                       ↓
               Physical start attempt → Events → Metrics → Report
```

Maintain three separate layers:

1. **Truth:** actual equipment occupancy, sampled durations, latent preferences/patience, and future events.
2. **Information:** reported occupancy and timestamps, declared visits/preferences, estimated durations, observed responses, and stated deadlines.
3. **Decision:** recommendations using only those observations and observed-feasible candidates.

Policies must not receive scenarios, latent member records, true finish times, or the future-event queue. Eligibility uses observed availability; physical starts check actual capacity. Filtering recommendations using hidden truth would invalidate the stale-information experiment.

```python
generate_scenario(config, seed) -> Scenario
policy.reset(context)
policy.decide(observation, eligible_recommendations) -> Decision
run_episode(scenario, policy, config) -> RunResult
evaluate(run_results, protocol) -> ExperimentReport
```

Use stable IDs, seconds internally, schema versions, and distinct scenario/policy seeds. A decision is a batch of member–visit–unit advice or explicit defer. A batch cannot duplicate a member or a unit. A policy may plan future visits internally but communicates only immediate advice. Deferring advances to the next event; no future event with unresolved members means `stalled`, not success. Time/event limits mean `truncated` and remain visible in reports.

## 3. Simulation semantics

### Deterministic starting case

Eight units across four types; twenty arrivals during 7,200 seconds; four visits/member; explicit compatibility; one member/unit; heterogeneous fixed durations; fixed visit order; sixty seconds of between-visit rest; zero walking; immediate, perfect compliance. Stop arrivals and drain workouts. These are debugging assumptions, not measured gym parameters.

The one-unit verification fixture has simultaneous arrivals with durations six, two, and four minutes. In stable member order, FCFS mean waiting is 14/3 minutes and SPT mean waiting is 8/3 minutes. The demonstration workload must contain enough contention for scheduling differences to appear.

### Equal-time events

Apply phases explicitly, rather than relying on implicit resource queues:

1. Complete visits and release equipment.
2. Apply departures, arrivals, readiness, and due human responses.
3. Resolve physical start attempts using a stable policy-independent tie rule.
4. Deliver due observations.
5. Decide and communicate the next recommendations.

Zero-delay responses generated by phase 5 enter a subsequent same-time round. Enforce a round limit and permit at most one behavioral response sample for a given decision opportunity. SimPy's same-time processing is sequential; event insertion order alone is not a scientific specification. [SimPy scheduling](https://simpy.readthedocs.io/en/stable/topical_guides/time_and_scheduling.html)

### Advice and human behavior

- One pending recommendation per member, with ID, issue/expiry times, and lifecycle outcome. Default expiry is thirty seconds. No member retry cooldown is imposed.
- Response delay, acceptance, refusal, ignored advice, and nonparticipation are distinct. Receiving or accepting advice never reserves equipment.
- After acceptance, walking precedes an authoritative start attempt. Stale advice may fail. Preserve readiness age across failures and refusals.
- Refusal/expiry invokes immediate independent choice across policy families until the current request starts. Failed access also returns immediately to independent choice. Nonparticipants act independently using idealized local availability; no forced pause is imposed. App policies still use sensor observations only.
- The initial implementation holds pending recommendations and accepted walks until their outcome. Its stability rule compares a later suggestion against previously communicated advice; it does not interrupt someone mid-walk. A future proactive replacement protocol needs separate semantics and experiments.
- Acceptance depends on the suggestion and latent willingness; deployed policies estimate it from observable response counts and declared preferences, never true probabilities.
- Draw duration variability separately from forecast error. Correlated member pace persists across visits. Key stochastic samples so policy call order does not change the latent workload.
- Introduce bursts, stale/missing observations, response behavior, deadlines, waiting-dependent abandonment, preference shifts, and temporary equipment outages in that order.
- At a stated deadline, prevent new starts but let an ongoing visit finish. Record overtime. Completion exactly at the deadline is on time.

Outages prevent new starts; an ongoing equipment visit is allowed to finish. Report resulting occupied time during an outage separately from ordinary available capacity when interpreting utilization. Full interruption/recovery of an ongoing visit is outside this model.

## 4. Policy progression and proposed contribution

| Stage | Policy | Role |
| --- | --- | --- |
| B0 | Self-directed | Uncoordinated member choices, not centralized FCFS. |
| B1 | FCFS | Current request readiness age, with deterministic ties. |
| B2 | SPT | Shortest predicted processing time. |
| B3 | Aging | Predicted seconds minus accumulated ready-wait seconds. |
| B4 | Matching | Maximize compatible assignment count, then minimize aging cost. |
| B5 | Rolling CP-SAT | Strong planning comparator with the same observations. |
| A1 | GA | Feasible schedule search with fixed evaluations or latency limits. |
| A2 | Offline NSGA-II | Expose waiting, deviation, and stability trade-offs. |
| L1–L3 | RL, GNN-RL, hybrid | Later, after freezing the benchmark. |

Tie-break classical policies by readiness time, member ID, visit ID, equipment ID. Earliest predicted finish is not a separate contribution when it is identical to SPT; distinguish it only where equipment-dependent start opportunities or durations change rankings.

### Rolling-horizon CP-SAT

Plan remaining declared visits of currently known members using predicted durations and release estimates. Enforce compatibility, capacity, precedence, rest, and ongoing visits. Do not use hidden finish times or future arrivals. Optimize predicted total ready waiting in the fixed-workload study. Communicate immediate advice, then replan after events. On no usable incumbent, fall back to matching.

Record solver status, objective, lower bound, runtime, and fallback. `FEASIBLE` does not mean `OPTIMAL`; an objective bound applies only to the exact formulation and available information. Small exact instances must optimize the same waiting objective as the comparison, not makespan from a tutorial. A clairvoyant reference must be labeled separately. [OR-Tools job shop](https://developers.google.com/optimization/scheduling/job_shop), [CP-SAT statuses](https://developers.google.com/optimization/cp/cp_solver)

### Candidate: conservative changes to recommendations

On the same matching planner, independently switch:

1. Acceptance awareness: empirical response estimates and declared preferences.
2. Uncertainty: duration uncertainty and observation age.
3. Stability: a cost for changing previously communicated advice.

Issue changed advice only when forecast benefit, including refusal consequences, exceeds uncertainty and change costs. Start with an empirical estimator, not simultaneous complex learning of durations, preferences, and scheduling. Run all eight `robust_a{0,1}u{0,1}s{0,1}` combinations; planner-family comparisons are separate. Freeze estimator and penalty parameters on validation data. A latent-probability oracle, if later added, must be explicitly diagnostic.

### Genetic search

Represent operation priorities and compatible equipment choices. Decode through a common feasibility-preserving schedule builder. Preserve member order in the main study; relax only declared precedence in flexibility experiments. Seed with the previous valid incumbent, heuristic schedules, and random feasible schedules. Use tournament selection, elitism, order-preserving crossover, and priority/equipment mutations. Warm-start when the operation set permits it.

Development defaults: population 32, tournament size 3, two elites, crossover 0.8, mutation 0.2. Tune only a predefined small validation grid. Keep the forecast objective aligned with CP-SAT. Fixed-evaluation mode is reproducible; wall-clock mode returns the best valid incumbent before a deadline or falls back to matching.

Compare 20, 100, and 500 ms budgets; 100 ms is primary. Measure total decision latency including observations, candidate construction, planning, and decoding. Planner internal time limits cannot by themselves guarantee that total latency meets budget. Record actual latency and overrun frequency. Offline NSGA-II follows the single-objective study; utilization balancing is exploratory and must not be interpreted as wear reduction.

## 5. Experimental protocol

Use successive targeted panels, not a huge Cartesian product:

| Panel | Factors |
| --- | --- |
| Correctness | Hand examples, simultaneous events, scarce compatibility. |
| Congestion | Low/medium/overloaded demand; report workload/capacity, not member count alone. |
| Durations | Fixed, CV 0.3, CV 0.6; correlated member pace. |
| Forecasts | Independently vary prediction bias/noise. |
| Observations | Fresh, 5 s and 30 s delay, missing updates. |
| Human response | High/medium/low willingness, refusal, ignoring, response delay. |
| Adoption | 100%, 50%, 25%; separate from acceptance willingness. |
| Flexibility | Fixed order, permitted reorder, approved substitutes. |
| Disruption | Bursts, unavailable units, deadlines, abandonment. |
| Generalization | Held-out sizes, layouts, demand mixes, compatibility distributions. |

Use a few prespecified joint stress cases after interpreting individual mechanisms. Expensive GA/CP-SAT initially run in representative regimes. Synthetic willingness numbers refer to reference actions; realized acceptance remains action-dependent.

### Outcomes

Waiting is time present and ready, excluding active visits, prescribed rest, and walking. Include pending-response and failed-attempt delay when ready. Keep waiting for members who later abandon.

| Outcome | Interpretation |
| --- | --- |
| **Mean accumulated waiting per entrant** | Primary outcome; entrant is denominator. |
| P95 and maximum member waiting | Tail burden. |
| Completion, on-time completion | Guardrails against apparent improvements caused by departures. |
| Required workload completed | Count/normalized workload, with definition recorded. |
| Arrival-to-completion | Completers only, always beside completion rate. |
| Acceptance/refusal/ignored | Recommendation usefulness with explicit denominators. |
| Changes/member, failed attempts | Advice stability and stale-information failures. |
| Order/substitution deviation | Difference from declared workout. |
| Utilization | Occupied/available seconds over the common arrival window. |
| Decision latency, fallback, overrun | Feasibility of online decisions. |

Report walking, rest, overtime, and response overhead separately. Stratify by workload length, compatibility flexibility, and participation. Use the fixed two-hour arrival window for primary utilization and report drain-period metrics separately. Do not silently combine these windows.

### Randomness, splits, and inference

- Save each scenario before any policy sees it. Save configuration snapshots, hashes, revision, dependency versions, hardware, and all relevant seeds.
- Keep exogenous randomness, policy search randomness, and future training randomness separate. A shared seed with policy-dependent draw order is insufficient.
- Key durations by member/visit/unit, retain member pace correlation, and key behavioral shocks by stable member/decision opportunities. Share latent patience, not realized abandonment time.
- Use development, validation, and untouched test seeds. `train` in the CLI denotes the development split for untrained policies. Also hold out distributions; new seeds alone do not test structural generalization.
- Run thirty pilot scenarios per principal regime. Estimate variance of paired episode differences. Choose final sample count for a CI half-width of max(30 seconds, 5% of baseline waiting), at least 100 and at most 500; disclose precision missed at the cap.
- Independent replication is the complete gym episode, not each competing member. Resample paired episode differences for 95% bootstrap intervals. Never manufacture sample size by pooling members across episodes.
- Choose the strongest reference using validation outcomes, freeze its parameters and the primary comparison, then run final tests. Label extra comparisons exploratory or correct multiplicity for confirmatory claims.
- Final learning studies use at least five independently trained models and separate training-seed from scenario variation. [RL evaluation guidance](https://arxiv.org/abs/2108.13264)

Initial, provisional engineering targets: at least 5% less mean waiting; completion loss no more than one percentage point; P95 wait no more than 5% worse; zero physical violations; practical runtime. Freeze targets after pilot work and before test access. These are engineering assumptions, not validated member preferences. A zero-wait reference makes percentage improvement undefined; report absolute differences and no claimed relative win.

Use a common configured delivery delay to compare solution quality. Separately replay saved per-policy latency traces to study advice aging. This avoids silently making workload reproducibility depend on machine speed.

## 6. Validation and artifacts

Required checks cover no unit/member overlap, compatibility, precedence/rest, ongoing-visit protection, nonnegative waiting, complete time accounting, deterministic replay, hidden-information isolation, advice expiry/stale response/refusal/retry, deadline boundaries, stall/truncation, identical physics for identical actions, equivalent degenerate dispatch rules, feasible GA offspring decoding, and solver fallback/status reporting.

Property tests should vary small workloads and stochastic cases in addition to hand examples. Inspect equipment timelines to explain resource contention. Exhaustive or exact small references can check objective quality; beating FCFS alone cannot establish near-optimality.

Each run stores the scenario/configuration, hash and provenance, events, recommendation lifecycle, member/unit summaries, metrics, latency, and failure status. Reports include timelines, waiting distributions, paired comparisons where there are repeated episodes, and interpretation limits. A single-scenario difference is a debugging result, not a population estimate.

Keep failed episodes in the manifest and report their frequency. A failed episode is not zero waiting. Do not silently drop infeasible incumbents, omit timeouts, or tune against artifacts labeled final test.

## 7. Semester deliverables and learning gate

| Weeks | Deliverable | Exit criterion |
| --- | --- | --- |
| 1–2 | Formal model, environment, assumptions, source shortlist | Actions, observations, waiting, and stopping rules explicit. |
| 3–4 | Deterministic simulator and baselines | Hand examples and physical invariants pass. |
| 5–6 | Stochastic scenarios, matching, paired evaluation | Reproducible repeated comparisons; no future leakage. |
| 7–8 | Advice, stale information, deadlines, candidate rule | All eight mechanism ablations run on identical semantics. |
| 9–10 | CP-SAT and small exact references | Reference, protocol, and holdouts frozen. |
| 11–12 | GA and runtime comparisons | Feasibility and quality–latency trade-offs measured. |
| 13–14 | Final study and failure/generalization analysis | Every result rebuildable from saved artifacts. |
| 15–16 | Thesis methods/results/figures and reproduction package | Every claim is supported or labeled future work. |

For a twelve-week deadline, reduce secondary panels and offline Pareto exploration. Preserve validation, strong baselines, core ablations, and reporting. Implementing research tooling does not complete these scientific milestones; pilot calibration, reading, protocol freeze, final analysis, and thesis writing remain research activities.

After the core benchmark is frozen, wrap this engine in Gymnasium rather than building a second simulator. Start with nongraph observations, then a GNN encoder over members/visits/units/precedence/compatibility, then one hybrid that initializes GA priorities from learned scores. Compare ordinary GA at the same runtime. Keep candidate actions/decoding fixed when attributing benefit to graphs. A GNN is an encoder, not an optimizer. [Gymnasium](https://gymnasium.farama.org/introduction/create_custom_env/), [Learning to Dispatch](https://proceedings.neurips.cc/paper/2020/hash/11958dfee29b6709f48a9ba0387a2431-Abstract.html)

Charge RL reward by elapsed simulated waiting, not policy calls. Penalize remaining work at departures to avoid rewarding abandonment. Distinguish termination from truncation; use independently trained models and held-out settings. Reassess compute only then. The initial workspace is not evidence that RL or GNN is effective.

## 8. Connection to UREC Live and evidence claims

Keep simulation independent of live backend and database credentials. Existing equipment–exercise relationships and session timestamps offer a future mapping. Weekly plans with muscle-group targets are not fully specified visits. Historical timestamps do not establish waiting, advice acceptance, or reservations.

A later read-only calibration adapter needs equipment/exercise exports, pseudonymous duration records, occurrence and receipt timestamps, recommendation/response/start-attempt/departure telemetry, and observation source/freshness/confidence. Delayed uploads make receipt time different from occurrence time. Calibration and factual replay do not identify counterfactual outcomes under an alternative scheduler.

Deployment progression is calibration → shadow advice → opt-in pilot. A reservation mechanism needs a separate ownership/expiry design. Do not change production APIs as part of the laboratory.

The thesis package must include assumptions, formal model, validated simulator, protocol, classical/GA results, controlled ablations, failures, and a claim ledger distinguishing synthetic findings, calibration needs, product implications, and novelty questions. Reference matrix groups P04/P07/P08, P09/P11/P12, P25–P29, P34–P36, and P37–P40 as reading leads. Read original methods before adopting study settings or asserting novelty.

Success means establishing which method works, under which assumptions, and where it fails, with a reusable laboratory prepared for later learning experiments.
