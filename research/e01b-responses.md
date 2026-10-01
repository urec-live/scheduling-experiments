# E01-B: accepting and refusing advice

The member cooldown has been removed. Declining advice, or finding a machine
occupied on arrival, immediately allows independent choice. If another approved
machine is free, the member can head there immediately. If none is free, they
wait for equipment rather than a retry timer. Physical capacity still governs
every start.

Independent users receive no advice and do not consult app sensors. Their current
local perception of availability is assumed accurate; this is a simplification
of real gym behavior. They still follow their declared exercises, approved
equipment, and recovery requirements. They do not know future release times.
Scheduler observations and candidates remain sensor-based.

After refusal/failed access, independent choice continues until the current visit
starts. This avoids repeatedly offering advice for the same unresolved request.
Guidance can be offered for the next visit. Refusal means declining guidance,
not declaring a machine unacceptable; independent choice may select that machine.
The response time and ignored-message expiry are separate existing mechanisms.
In this example responses are immediate and nobody ignores advice.

## The three samples

Each sample has four advice-eligible members arriving together, two equivalent
machines, and one 60-second visit per member. Walking time is zero.

| Sample | Always accept | Always refuse |
| --- | ---: | ---: |
| All accept | 4 | 0 |
| All refuse | 0 | 4 |
| Half accept | 2 | 2 |

Each is replayed with independent choice and FCFS guidance, making six runs.
The same arrivals, durations and equipment are used throughout. Cohort assignment
is seeded and hidden from the scheduler. Under the independent comparison,
everyone acts independently and no advice is issued, including to acceptors.
Refusers are distinct from independent nonusers: refusers are offered advice.

For other percentages, set `acceptance_model: member_cohort` and
`acceptor_fraction: 0.25`, for example. The fraction applies to guidance-eligible
members only. Counts are rounded to the nearest whole member, with halves rounded
up. With four eligible members, 25% means one acceptor. Saved member
`acceptance_probability` values are 1 or 0. They are latent behavioral settings,
not facts given to the policy. Report actual cohort counts when rounding matters.
Members may receive different numbers of recommendations, so a member split is
not necessarily the same split of advice events.

## Observed result

| Guided sample | Accepted | Refused | Mean waiting | Completed |
| --- | ---: | ---: | ---: | ---: |
| All accept | 4 | 0 | 30 s | 4 of 4 |
| All refuse | 0 | 4 | 30 s | 4 of 4 |
| Half accept | 2 | 2 | 30 s | 4 of 4 |

All three independent comparisons also averaged 30 seconds waiting and completed
all four visits, with no recommendations. Two members start at time 0 and finish
at 60 seconds. The other two start at 60 and finish at 120 seconds. The mean wait
is (0 + 0 + 60 + 60) / 4 = 30 seconds. No refusal delay is added. All six runs
completed without audit findings. This is one deterministic correctness example,
not evidence that acceptance never affects outcomes in larger or varied gyms.

## Files and replay

Configuration: [e01b-responses.yaml](../configs/experiments/e01b-responses.yaml).
Executed artifacts: [report.html](../outputs/e01b-responses-no-cooldown/report.html).

From `scheduling-experiments/`:

```bash
uv run gym-sched experiment --config configs/experiments/e01b-responses.yaml --split train --output outputs/e01b-responses
```

Use a new output name on each replay. If the local `uv` launcher fails, substitute
`.venv/bin/python -m gym_sched.cli` for `uv run gym-sched`.

The old broad E01 panel remains a separately labeled per-recommendation probability
experiment. Its saved 1,500-run pilot used the old cooldown and is historical.
Do not compare those outcomes to these as if cooldown were the only change:
autonomous local perception and the default refusal objective penalty also changed.
The robust planner now has zero default refusal penalty. An explicit nonzero
planning penalty is still possible, but it never imposes a member delay.
