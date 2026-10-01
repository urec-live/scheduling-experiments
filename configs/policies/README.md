# Policy parameter sets

`defaults.yaml` is a mapping of policy name to constructor options, consumed by `compare --policy-config` and experiment `policy_config`. Omitted settings use the policy's recorded defaults. The manifest records resolved options; preserve it with each result.

GA `budget_mode: evaluations` uses a fixed evaluation count and keyed policy RNG for repeatable search. `budget_mode: wall` is a wall-clock comparison and can return different incumbents on different machines. The 20/100/500 ms files apply equal **planner** wall limits to CP-SAT and GA. Record total pipeline time and overhead separately; internal planner budgets do not guarantee end-to-end deadlines.

CP-SAT also has a deterministic-work cap and uses one worker. Whichever configured cap is reached first can stop search. Its 1 ms planning resolution conservatively discretizes durations; compare approximate objectives within that resolution. GA and CP-SAT optimize a deterministic forecast with instantaneous compliance, not the full stochastic advisory outcome. The simulator supplies the same human behavior to both.

Named `robust_a0u0s0` through `robust_a1u1s1` select the acceptance/uncertainty/stability switches. The shared YAML anchor gives all eight the same remaining settings. Changing penalties for just one ablation would confound the mechanism comparison. The all-off variant should match ordinary matching in the same fixed-workload/action setting.

Do not use final test scenarios to choose population size, evaluation count, penalty weights, priors, or solver settings. See the validation grid in `research/protocol.md`.
