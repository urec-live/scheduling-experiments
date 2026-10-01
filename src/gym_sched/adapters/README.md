# Integration and learning boundary

Production adapters and trained models are gated research extensions, not active
services. No credentials or production network access belong in this package.

A future read-only import must translate pseudonymous equipment/visit exports to
the versioned domain models, preserving source IDs, occurrence and receipt times,
missingness, exclusions, and calibration provenance. Never label real imports as
synthetic; the current generator and artifact workflow only support synthetic data.

A future Gymnasium wrapper should pause the existing `Simulator` immediately
before `_dispatch`, accept a `Decision`, and resume to the next decision boundary.
Do not replace its physical event engine. Expose only `Observation` and candidate
features, with masks derived from observed availability. Integrate waiting over
elapsed simulation time for reward; penalize incomplete required workload when
departures occur. Map natural drain/departure to `terminated`, watchdog limits to
`truncated`, and test bootstrapping semantics. Pending/accepted advice stays locked.

First compare a nongraph candidate scorer with a graph encoder under the same
action decoder. Train on development instances and evaluate independent training
seeds on frozen tests. GPU access is optional and no training is implemented or
claimed by installing the `learning` extra.
