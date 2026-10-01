# Hand-calculated verification

For the quiet-gym control, use `e01a_quiet.json`: three members, two equivalent
machines, and spaced arrivals. Independent choice and FCFS both produce zero
waiting. See the [step-by-step walkthrough](../../research/e01a-walkthrough.md).

`three_members.json` contains one equipment unit and three members arriving at time zero. Each needs one visit lasting six, two, or four minutes. There are no deadlines, walking, rest, response delay, or forecasting error.

| Policy | Member order | Waiting seconds | Mean waiting |
| --- | --- | --- | --- |
| FCFS | 001, 002, 003 | 0, 360, 480 | 280 s = 14/3 min |
| SPT | 002, 003, 001 | 0, 120, 360 | 160 s = 8/3 min |

Both policies finish at 720 seconds and occupy the unit for 720 seconds. Utilization in the common 7,200-second arrival window is 10%; utilization over their completed 720-second trajectory is 100%. Always state the measurement window.

Run from the workspace root:

```bash
uv run gym-sched compare --scenario data/fixtures/three_members.json --policies fcfs,spt --output outputs/hand-example
uv run gym-sched report --run outputs/hand-example
```

Fixture values are synthetic and carry no empirical claim about gyms.
