# Scheduling experiment laboratory

This folder is an independent Python 3.12 project and local Git repository.
Do not connect to production services or reuse app/backend/CV environments.

- Install: `uv sync --python 3.12`; optional Pareto tests: `uv sync --extra ga`.
- Check: `uv run --extra ga pytest`; `uv run ruff check src tests`; `uv run ruff format --check src tests`.
- Quick workflow: see README.md. Generated artifacts stay under ignored `outputs/`.
- Preserve scenario truth / observation / decision separation. No actual sampled
  durations, future arrivals, latent behavior, or physical occupancy may leak into
  policy candidates. Physical start checks must still enforce truth.
- Advice never reserves equipment. Preserve same-time phases. No forced member
  cooldown: refusal/failed access returns to independent choice immediately.
  Prevent repeated attempts through event-driven availability, not artificial
  delay; audit all capacity, precedence, and member-time accounting.
- Keep paired scenario randomness independent of policy draw order. Do not tune
  on final tests or omit stalled/truncated episodes from reports.
- Any schema, metric, objective, or behavior change requires corresponding tests
  and an assumptions/protocol update. Never label synthetic smoke evidence as
  demonstrated real-gym benefit or established novelty.
- RL/GNN training and production adapters are gated future work, not stubs to
  present as finished algorithms. Extend the existing simulator when the gate is met.
