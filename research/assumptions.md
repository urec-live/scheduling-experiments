# Simulator assumptions and sensitivity register

Every numeric setting is synthetic until calibrated. Version assumptions with the protocol rather than silently replacing defaults after looking at results.

| Assumption | Initial value/model | Sensitivity or validation |
| --- | --- | --- |
| Arrival period | 7,200 s | Fixed common measurement window; drain separately. |
| Capacity | One member per equipment unit | Sharing requires a new model. |
| Visit granularity | Consecutive sets on a unit | Within-visit rest occupies equipment. |
| Between-visit rest | 60 s, equipment released | Shorter/longer values; do not count rest as waiting. |
| Duration | Type-based mean with member heterogeneity | CV 0, 0.3, 0.6; independent forecast bias/noise. |
| Pace correlation | One shared member factor | Compare independent versus correlated durations. |
| Advice | Optional, no reservation | Acceptance differs from successful equipment start. |
| Advice expiry | 30 s | Ignored advice expires; expiry is not a member retry delay. |
| Refusal or failed access | Immediate independent choice until this request starts | No forced pause and no repeated advice for the same unresolved request. |
| Autonomous availability | Current local view of free equipment, independent of app sensors | Idealized local perception; no knowledge of future releases or durations. |
| Response cohorts | Optional exact fraction of guidance-eligible members always accept; remainder always refuse | Seeded assignment, count rounded half up; independent users remain separate. |
| Response | Configured delay and latent willingness | Action preferences, refusals, ignored advice. |
| Adoption | Participant membership | 1, 0.5, 0.25 independently of acceptance. |
| Observations | Timestamped status updates | 0/5/30 s delay; lost updates and refresh interval. |
| Pending advice | One per member; accepted walking action retained | Proactive interruption is a later protocol. |
| Walking | Configured time, zero in demo | Distances/layouts are future calibration; time excluded from wait. |
| Deadlines | Declared; block new starts, finish ongoing visit | On-time iff all visits complete by deadline. |
| Patience | Latent accumulated-wait threshold | Share threshold across policies, not departure timestamps. |
| Preferences | Declared preferred units plus latent response behavior | Time-dependent willingness as separate stress case. |
| Acceptance estimator | Per-member preferred/alternative response counts, Beta(1,1)/Beta(1,2) priors | Priors are assumptions; ignored advice counts as nonacceptance. No latent willingness is exposed. |
| Outages | No new start during outage; no interrupted active visit | Availability denominators must account for outage periods. |
| Fitness adequacy | Visit completion proxy | No physiological benefit, injury, fatigue, or wear claims. |
| Substitutions | Explicitly approved alternatives | Count deviation; never infer medical equivalence. |

## Information available to each policy

Policy views may contain observed status and observation age; estimated ongoing release; currently known members; declared remaining visits, compatibility, precedence, rest, preferences, and deadlines; prior advice; empirical response counts. Views do not contain sampled actual durations, exact future finishes, future arrivals, latent patience, true acceptance parameters, or the event queue.

Ground truth must not leak through policy candidate generation, diagnostics, solver fallback selection, or oracle-trained forecasts. Check these surfaces in addition to the main observation object. Autonomous member behavior uses current physical availability as an idealized local view; that information is not copied to the scheduler's sensor observations or candidates. It contains no future information.

## Key scientific limitations

The simulator represents equipment access, not complete training quality. It assumes the declared visits are meaningful and approved. Simplified constant response/walking delays, synthetic latent preferences, retry rules, and outage behavior may affect rankings. Matching, CP-SAT, and GA objectives are estimated surrogates; favorable surrogate quality does not guarantee favorable realized advice outcomes.

The demonstration, fixture, and smoke run validate software pathways. They do not calibrate a human model, demonstrate real-gym effectiveness, establish novelty, or complete a thesis experiment. Fixed wall-clock budgets also depend on hardware and cannot guarantee deterministic schedules; use fixed evaluations for reproducibility and replay recorded latency for delivery-delay stress tests.

The deterministic `even` arrival generator defaults to four-person batches spread evenly
through the arrival window, creating visible competition for two units per type.
Repeated seeds of this fully deterministic scenario repeat the same workload;
they are correctness checks rather than uncertainty evidence.

`arrival_batch_size` controls batch size for `even` and `poisson_batches` only.
`poisson_batches` fixes the number of batches at ceil(member_count / batch_size)
and draws ordered uniform batch times within the arrival window, conditional on
that count. It does not sample an unconditioned Poisson count. All members in a
batch arrive simultaneously, with a potentially smaller final batch. Its random
stream is keyed independently of individual arrivals and policy decisions.
`member_pace_factors` is a positive repeating sequence, defaulting to the original
1.25/0.75/1.0 pattern. E01 S18 uses [1.0] with zero duration/pace CV to exercise
synchronized finishes. These settings change workload generation only and expose
no additional physical truth to policies. See [E01](e01.md) for workbook coverage
and the explicit limitations of participation-based subgroup reporting.

Queue age for FCFS preserves the timestamp at which the current request became
ready. Aging costs use accumulated waiting for that request, excluding walking.
After refusal, expiry, or failed access, a member immediately becomes ready for
independent choice, retaining request age and accumulated waiting. This lasts
until a visit starts; a later visit may receive advice again. If no eligible
machine is locally free, the member stays ready and chooses again when events
change availability. There is no retry timer. Simultaneous choices still compete
at physical start, and a failed attempt can try a different free machine at the
same time. Positive visit durations and finite capacity resolve these attempts;
the same-time watchdog remains an error guard, not a behavioral delay.

`acceptance_model: member_cohort` requires `acceptor_fraction` in [0, 1]. The
generator assigns round-half-up(fraction × guidance-eligible member count)
acceptors, using an independent seeded ranking. Other eligible members refuse.
Assignment remains fixed for the episode and is hidden from policies. Approved
nonpreferred suggestions are also always accepted by acceptors. Ignored advice,
time-varying acceptance and per-recommendation probabilities cannot be combined
with this model. Existing `per_recommendation` behavior is still available for
separately labeled experiments. A member fraction does not guarantee the same
fraction of advice events when members receive different numbers of suggestions.

The robust planner's optional `refusal_cost_seconds` is now zero by default and
in bundled policy settings. It is an estimated objective penalty, not a delay
enforced on a member. Nonzero values require a separately justified assumption;
the old 30-second penalty no longer matches current dynamics.

CP-SAT and GA forecast immediate compliance, no unseen future arrivals, and no
control of nonparticipants. Ongoing visits use predicted residual duration, not
sampled finish time. CP-SAT discretizes to milliseconds; GA's feasible decoder
appends operations to equipment timelines. Different search spaces and internal
solver overhead remain explicit limitations. A wall deadline can be exceeded by
model construction or a single decode; full latency and overruns are measured.

Configured decision-delay traces repeat when exhausted. This is an injected
latency stress model, not a guarantee that the same decisions occur after delays.
Started visits crossing an outage finish normally; occupied time during that
grace period stays in usable-capacity denominators. A watchdog-truncated run counts
only members who have actually arrived and cannot pass research guardrails.
# Visual demonstration scope

The `gym_sched.visual_demo` export is a playback interface, not a new simulator
or scheduling policy. S05/S43 use one identical workload and FCFS in both runs;
only `SimulationConfig.allow_reordering` changes. Casey's visit predecessor lists
are empty because either order is permitted in the flexible condition. Fixed
order is enforced by the existing configuration. Neither visit is substituted.

Alex arrives at simulation second 0 and occupies A for 181 seconds. Casey
arrives at second 1; the replay clock starts there, with 180 seconds of Alex's
occupancy remaining. Casey has two 120-second visits and 60 seconds of recovery
between them. There is no walking, observation delay, response delay or refusal.
Expected Casey waiting is 180/0 seconds and elapsed time is 480/300 seconds for
fixed/flexible order. These are constructed correctness examples, not empirical
benefit estimates. Actual events displayed to the human viewer do not cross the
simulator's policy observation boundary. Alex's display name is not identity
telemetry supplied to the scheduler.

The detailed profile inspector is observer-only. It separates declared plan
fields from latent behavior and shared experiment conditions. Unspecified sets,
reps, weights, devices and future preference fields remain explicitly unset;
displaying them does not implement their effects. Order permission reflects the
selected global demonstration condition, not a newly added per-member schema.


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
