# ADR-014: Offline Agent-Policy Experiments

**Status:** Accepted — implements `TL-68`
**Owner:** Shaiv Pandya
**Date:** 2026-06-12
**Revisit trigger:** Any proposal to run a learned process policy in production shadow/canary mode, add sequence-level offline RL, or use new reward sources beyond process/eval/human-reviewed signals.

## Context

`TL-68` introduces offline RL and contextual-bandit experiments for agent decisions only. The owned-model program already has trajectory capture, feedback labels, training datasets, preference optimization, replay environments, TalismanBench, and rollout controls. It does not yet have a governed way to compare learned policies for agent-process choices before live tests.

Intent routing is the first suitable domain because it already records route context, logged route decisions, shadow comparisons, supervised candidates, and optional human labels. This gives a narrow first slice while preserving the issue gate: no online RL, no learned execution policy, and no override of deterministic gates.

## Decision

Create a dedicated offline policy experiment module in `decision_quality/agent_policy_experiments.py`.

The first implementation:

1. Defines versioned contracts for logged decision examples, action candidates, propensities, rewards, manifests, and reports.
2. Converts durable intent-router training rows into contextual-bandit examples.
3. Requires propensity metadata for counterfactual policy comparisons and records missing propensities as exclusions.
4. Allows rewards from process checks, eval scores, human review, bounded outcome labels, or synthetic fixtures.
5. Rejects future-leaking and P&L-only reward fields.
6. Emits a JSON report with exclusion counts, inverse-propensity reward confidence intervals, `compare_reports`-style deterministic regressions, gate-boundary violations, and known biases.
7. Keeps production rollout code, gateway policy, and route selection unchanged.

## Alternatives Considered

| Alternative | Pros | Cons |
|-------------|------|------|
| Reuse `intent_router_training.py` directly | Minimal new surface | That module trains supervised classifiers, not counterfactual policy reports or reward/propensity contracts |
| Extend TalismanBench as a fourth corpus immediately | Centralized release report | TalismanBench gates model candidates; TL-68 needs policy-choice rows and propensity exclusions first |
| Use replay environments as the first policy domain | Strong process rewards | Replay trajectories are not production logged choices and currently lack propensities |
| Build sequence-level offline RL first | Closer to long-term RL goal | Too broad for the first governed slice and harder to audit for leakage and gate boundaries |
| Add production shadow policy routing now | Faster feedback | Violates TL-68 gate requiring offline reports before live policy tests |

## Risks

- Intent-router rows do not yet provide broad multi-arm propensities, so early reports may have many `missing_propensity` or `counterfactual_action_unobserved` exclusions.
- Inverse-propensity estimates can have high variance when action probabilities are small.
- Human labels and outcome labels can be sparse or biased; reports must show source counts and known biases.
- Replay process rewards remain useful validation evidence but are not a substitute for logged production propensity data.

## References

- `decision_quality/agent_policy_experiments.py`
- `api/intent_router_training_store.py`
- `decision_quality/intent_router.py`
- `decision_quality/agent_replay_environments.py`
- `decision_quality/eval_corpus.py`
- `docs/talisman_offline_policy_experiments.md`
- Linear `TL-68`, `TL-84`, `TL-92`, `TL-94`, `TL-97`
- [Program architecture guide](../talisman_owned_agent_model_program.md)
