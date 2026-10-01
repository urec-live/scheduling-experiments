# Later connection to UREC Live

This package is independent of the React Native app, Angular admin, Spring backend, and Neon database. It does not read production credentials or issue live scheduling recommendations.

## Mapping and missing measurements

| Existing concept | Research mapping | Limitation |
| --- | --- | --- |
| Equipment and equipment–exercise links | Unit IDs, type, compatible visit alternatives. | Status alone does not indicate ownership or a reservation. |
| Workout session start/end | Observed visit duration candidates. | Delayed uploads, missing starts, or member editing require validation. |
| Workout sets | Within-visit workload descriptors. | Sets do not fully determine occupancy time. |
| Weekly muscle-group targets | Source for a future visit-plan builder. | Not an explicit ordered list of scheduling operations. |
| Floor plans/positions | Potential walking estimates. | Coordinate units and actual pathways require calibration. |
| WebSocket status changes | Availability observations. | Occurrence time, receipt time, source, and confidence are needed. |

A future read-only import contract should use pseudonymous stable member IDs; unit/exercise compatibility; duration observations and their quality flags; timestamps for actual occurrence and system receipt; recommendation issue/expiry/response; attempted and actual starts; willingness/declared preferences; explicit ready/rest/walking state where available; departures; observation source/freshness/confidence.

Do not infer waiting from gaps between workout sessions: the gap can contain rest, walking, unrelated activities, or offline upload delay. Do not infer acceptance from a subsequent session without linking a recommendation and response. Import adapters should validate units, references, timestamp order, and missing data, then create a versioned snapshot with provenance.

## Product progression

1. **Calibration:** fit plausible arrival/duration/behavior distributions using appropriately obtained observations. Preserve a holdout for model checks.
2. **Shadow advice:** generate recommendations without changing member behavior; evaluate freshness, feasibility, timing, and monitoring. Shadow data cannot establish counterfactual waiting reductions.
3. **Opt-in pilot:** define observable outcomes and experiment allocation; evaluate benefits and guardrails prospectively.

Monitor data age, advice acceptance/expiry, failed starts, completion, tail wait, fallback and latency, subgroup outcomes, and policy version. An equipment-reservation product would additionally require authoritative ownership, expiry, conflict handling, and client/backend changes; it is outside the advisory experiment.

Historical replay is factual replay. It cannot reveal how other people would have reacted to a recommendation that was never issued.
