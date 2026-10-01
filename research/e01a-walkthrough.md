# E01-A: one quiet-gym example

Question: when a suitable machine is free at every arrival, does anyone wait?

This is a small teaching and correctness example for S01 in the scenario
spreadsheet. It is separate from the earlier 30-workload pilot and is not an
additional statistical study.

## 1. The gym and members

There are two equivalent machines. Every member can use either one and needs
only one visit. Times below are minutes and seconds since the example begins.

| Member | Arrives | Exercise time |
| --- | --- | --- |
| A | 00:00 | 4 minutes |
| B | 01:00 | 4 minutes |
| C | 06:00 | 2 minutes |

Assumptions: machines start free, occupancy information is accurate and immediate,
and exercise durations are predicted correctly. Walking and response time are
zero. There are no refusals, deadlines, substitutions, or recovery intervals
between visits because each member has just one visit. These are synthetic
control conditions, not measurements of actual members.

## 2. Replay the exact same workload twice

**Independent run:** `self_directed` lets everyone choose independently and issues
no advice. The existing independent-choice rule picks the first eligible free
machine by ID when preferences are equal.

**Guided run:** `fcfs` advises the oldest ready request, and every member accepts.
FCFS is sufficient here because requests never have to queue. Other algorithms
are unnecessary for understanding this example.

The saved fixture marks everyone eligible for guidance so FCFS can advise them.
The `self_directed` policy overrides that eligibility and makes everyone act
independently without changing the workload. Consequently, saved subgroup labels
still say “participants” in both runs; those labels describe eligibility, not
actual guidance received. This example does not implement tracking-focused or
time-focused identity. Use advice events to distinguish the two conditions.

## 3. What happened in both runs

| Time | Event | Why nobody waits |
| --- | --- | --- |
| 00:00 | A starts machine 1. | Both machines were free. |
| 01:00 | B starts machine 2. | A is still exercising, but machine 2 is free. |
| 04:00 | A finishes and releases machine 1. | No member is waiting. |
| 05:00 | B finishes and releases machine 2. | No member is waiting. |
| 06:00 | C starts machine 1. | Both machines are free again. |
| 08:00 | C finishes. | All three members have completed their visits. |

## 4. Observations

| Outcome | Independent | Guided with FCFS |
| --- | ---: | ---: |
| Mean waiting per member | 0 seconds | 0 seconds |
| Maximum waiting | 0 seconds | 0 seconds |
| Members completed | 3 of 3 | 3 of 3 |
| Advice issued / accepted | 0 / 0 | 3 / 3 |
| Failed equipment access attempts | 0 | 0 |
| Last member finishes | 08:00 | 08:00 |

Guidance changes how the machine is chosen, but it provides no waiting benefit
in this example: everyone already has immediate access. That is the expected
control result. The model now has no member cooldown. The earlier quiet pilot's
roughly 1.7-second mean wait used random arrivals, which occasionally overlapped;
this deliberately spaced example has no queue at all.

## 5. Files and replay

The input is [e01a_quiet.json](../data/fixtures/e01a_quiet.json). From
`scheduling-experiments/`, run:

```bash
uv run gym-sched compare --scenario data/fixtures/e01a_quiet.json --policies self_directed,fcfs --output outputs/e01a-quiet-walkthrough
```

That output directory already holds the executed example. For another run, use a
new output name. If the local `uv` launcher crashes, use
`.venv/bin/python -m gym_sched.cli` in place of `uv run gym-sched`.

Open the [saved report](../outputs/e01a-quiet-walkthrough/report.html).
Its `policies/self_directed/events.jsonl` and `policies/fcfs/events.jsonl` contain
the actual event sequences. The regression test
`test_e01a_quiet_hand_calculated_timeline` checks starts, waiting, completion,
advice counts and access failures against the hand-calculated expectations.

Next, review [E01-B response cases](e01b-responses.md), which now use immediate
independent choice after refusal. The saved E01-A run predates that revision,
but this no-refusal timeline is unchanged and checked by the regression test.
