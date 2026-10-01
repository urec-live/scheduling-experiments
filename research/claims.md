# Evidence and claims ledger

Use this ledger when turning experiments into thesis or product statements. Update rows with artifact hashes and source details; an empty evidence cell means the claim is not established.

| Proposed statement | Current evidence class | What is still needed |
| --- | --- | --- |
| The simulator reproduces the 6/2/4-minute example. | Software verification, after tests pass. | Preserve test result and fixture revision. |
| A policy reduces waiting in the demonstration. | One synthetic scenario. | Paired, repeated held-out episodes and guardrails. |
| Acceptance awareness helps when willingness changes. | Hypothesis. | Eight-condition controlled ablation on fixed planner. |
| GA improves the quality–latency trade-off. | Hypothesis. | Matched budget/objective comparisons and latency records. |
| Recommendations remain useful with stale observations. | Hypothesis. | Delay/dropout panels, failed-attempt and completion outcomes. |
| Gym members experience less waiting. | Not established by simulation. | Calibrated model and prospective opt-in evaluation. |
| A result is near-optimal. | Not established by beating FCFS. | Matched exact objective solution or valid bound; stated information setting. |
| The method is original thesis IP. | Novelty unverified. | Full-text related-work analysis and institutional IP guidance if relevant. |
| Wear or injury risk is reduced. | Out of scope. | Separate measurement, domain model, and appropriate study. |

Negative findings should be retained. Report when a simple heuristic matches or beats a more complex approach, when a subgroup fares worse, or when a runtime budget fails. Do not use aggregate mean waiting to hide lower completion or worse tail waits.

When publishing a figure or table, record scenario/config hashes, code revision, split, policy settings, analysis definition, and run path. Use “in these synthetic scenarios” until stronger evidence exists.
