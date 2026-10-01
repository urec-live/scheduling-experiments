# Experimental protocol and decision record

Status: **development protocol; not a frozen or preregistered final study**. This document defines the intended process. A generated protocol snapshot records a particular experiment configuration; it does not establish independent preregistration or prevent a researcher from viewing held-out data.

## Behavior revision: no member cooldown

Following the user's revised model, refusal and failed access allow immediate
independent choice. Independent members use idealized local availability and
never depend on the app's sensor feed. Waiting for occupied equipment remains
waiting; there is no added retry delay. The source revision/fingerprint must
separate these outcomes from previous runs. Old E01 pilot artifacts retain their
historical results and must not be pooled with or cited as evidence for this model.
Older scenario YAML with `retry_cooldown_seconds` must remove that obsolete field;
the schema rejects it rather than silently changing its meaning.

Start response experiments with `configs/experiments/e01b-responses.yaml`: all
accept, all refuse, and an exact 50/50 member split, each paired against independent
choice. This six-episode teaching example uses one seed and supports no inference.
All four members are guidance-eligible; the independent policy suppresses advice.
Future percentage sweeps must state whether percentages describe members or
per-advice response probabilities. Do not treat always-refusers as independent
nonusers: they receive advice, even when they decline it.

## Shared gym environment contract

Every policy in a comparison must face the same saved scenario for a given
regime and seed. The scenario defines members, declared visits, compatibility,
capacity, arrivals, sampled durations, rest, latent response settings, walking
settings, observation dynamics, outages, deadlines, and patience. The experiment
runner must save those exogenous conditions before invoking any policy and then
reuse the saved scenario for each policy.

Each policy still runs in its own clean simulator copy. That is intentional:
different decisions create different queues, refusals, walking conflicts,
occupancy, and sensor histories. Those downstream differences are outcomes, not
different starting worlds. A custom policy may use richer observable features
than FCFS, but FCFS must still experience the same realistic gym conditions and
hard constraints.

## Questions and comparisons

Primary candidate: conservative matching with acceptance, uncertainty, and stability enabled. Select the strongest noncandidate reference using validation scenarios, then lock both its settings and the comparison. Primary outcome: mean accumulated ready waiting per entrant. Evaluate policy-family comparisons separately from the eight matching-mechanism ablations.

Record before opening final tests:

| Item | Decision to preserve |
| --- | --- |
| Source revision and environment | Git commit, lockfile hash, Python and dependency versions, CPU/OS. |
| Workload definition | Generator and simulation config hashes, saved scenario hashes. |
| Split | Development/validation/test seed lists plus held-out distribution definitions. |
| Reference and candidate | Names, complete parameter values, validation selection rationale. |
| Primary contrast | Candidate minus reference; lower waiting is favorable. |
| Regimes | Prespecified principal regimes and secondary stress/generalization cases. |
| Sample size | Pilot variance, target half-width, minimum/maximum, cap disclosure. |
| Guardrails | Completion, P95, feasibility, latency, treatment of failed episodes. |
| Inference | Paired episode bootstrap, seed, repetitions, alpha, multiplicity policy. |

`train` in CLI configurations is development data; no policy training is implied. Do not select a reference separately after observing each test regime. If regime-specific dispatch is a proposed product feature, define its selection rule from observable state during validation and evaluate that rule as a policy.

## Sample size and paired inference

Use thirty paired pilot episodes per principal regime. Let `d_i` be candidate waiting minus reference waiting for the same saved scenario. Estimate the standard deviation `s_d`; a starting normal-approximation count is `ceil((1.96 * s_d / h)^2)` with `h = max(30, 0.05 * baseline_mean)` seconds. Clamp to 100–500. This is a precision calculation, not a power claim, and skewed/heavy-tailed waiting may require more episodes. Report the achieved bootstrap interval and whether it met `h`.

Resample episodes with replacement and recompute the mean paired difference to obtain a 95% percentile bootstrap interval. Preserve pairing. A member is not an independent replicate because members share equipment. With one episode, report descriptive differences only; a zero-width resampling interval is not evidence of certainty.

Multiple policy seeds per scenario require aggregation within scenario or a declared hierarchical analysis; they do not create independent workload episodes. Future RL training runs add another variation level and must not be pooled as independent members. Start with at least five independent trained models.

Do not call a solver incumbent optimal unless the recorded status certifies it. Do not interpret an optimization gap on an estimated deterministic planning snapshot as a bound on real-world uncertain advisory performance.

## Primary success rules

Freeze after pilot work:

- At least 5% lower mean waiting than the selected reference.
- At most 0.01 lower completion rate.
- At most 5% higher P95 waiting.
- Zero physical capacity, compatibility, or precedence violations.
- Total measured decision latency appropriate for the chosen 100 ms primary budget; also report 20/500 ms sensitivity, overruns, and fallbacks.

Report estimates and intervals alongside thresholds. If baseline mean or P95 equals zero, relative change is undefined; use absolute differences without claiming a percentage improvement. These are provisional engineering targets and do not establish user utility. Negative or mixed results remain findings.

Final-test failures are not discardable. Retain `stalled`/`truncated` episodes and failures in the manifest, report their rate, and avoid treating partial waiting as completed-episode performance. A protocol may define a penalized composite sensitivity analysis, but must not invent that penalty after seeing which policy fails.

## Panels and validation grid

The workbook-driven E01 panel is specified in [e01.md](e01.md) and
`configs/experiments/e01.yaml`. It covers S01/S04/S17/S18 with ten separate
regimes and thirty development seeds each. Fixed visit durations are paired
across policies while random arrivals provide independent workload replications.
The batch generator preserves within-batch synchronization. Keep regime-level
inference separate because regimes reuse seed streams. FCFS is provisional;
choose among FCFS/SPT/matching on validation only, weighting the ten regimes
equally. E01 has no final test split until precision and protocol review.

`configs/experiments/pilot.yaml` is a modest pilot template. Its counts do not constitute a final powered study. Additional panel files vary one principal mechanism at a time; avoid assuming a Cartesian grid is automatically an interpretable study. Final generalization must include structural changes (size, demand, compatibility), not only unfamiliar seeds.

GA development grid: population 16/32/64; fixed evaluations 64/128/256; mutation 0.1/0.2/0.3. Vary one setting at a time initially. Preserve tournament size 3, two elites, and crossover 0.8 unless a separately declared validation grid changes them. Compare the chosen fixed-evaluation variant with deadline mode at matched online budgets.

Candidate development grid: vary uncertainty/stability penalties and acceptance prior only on validation; retain all eight ablations with the same underlying settings. Store the final values and their rationale. Do not describe a favorable ablation chosen after inspection as confirmatory.

## Interpretation and dissemination

Report pooled and regime-specific performance, waiting tails, completion, workload completed, member subgroups, advice outcomes, latency, and failure counts. Show at least one failure timeline. Preserve the exact artifact-to-figure lineage. Secondary analyses are exploratory unless explicitly adjusted for multiple comparisons.

Synthetic evaluation supports conditional claims about this model. Live member benefits require calibration and prospective testing. A reading matrix is a discovery aid, not proof that the proposed mechanism is novel. Update the claim ledger before writing thesis or product language.
# Visual demo validation

The two-member visual demo is a deterministic mechanism check for matrix S05
and S43. Verify the shared scenario hash, required work, recovery, capacity and
hand-calculated waits before considering extensions. Browser playback tests
cover order switching, same-time event stepping, seeking, member selection and
responsive layout. The displayed completed-run outcomes are not confidence
intervals or live release-time predictions. Do not use repeated replay or seeds
of this fixed scenario as evidence of general scheduling benefit.


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
