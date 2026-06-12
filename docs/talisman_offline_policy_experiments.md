# Talisman Offline Policy Experiments

`TL-68` adds an offline contextual-bandit and ranking report layer for agent-process choices. The first implementation evaluates intent-router choices because that path already captures route context, logged actions, shadow candidates, optional human labels, and durable training rows.

## Scope

Offline policy experiments can compare process choices such as:

- intent route and workflow selection
- tool-pack or required-tool selection
- whether to request missing inputs, stop, or defer
- candidate prioritization where logged propensities are available

They must not optimize direct trading execution, live order behavior, or realized P&L as a sole reward. Source, policy, approval, and DecisionQuality gates remain deterministic controls outside the learned policy.

## Contracts

Implementation lives in `decision_quality/agent_policy_experiments.py`.

| Contract | Version | Purpose |
| --- | --- | --- |
| `LoggedDecisionExample` | `1` | One context, logged action, candidate action set, propensity, reward, split group, and provenance row |
| `ActionCandidate` | `1` | A route/tool/process action available to a candidate policy |
| `PropensityMetadata` | `1` | Logging policy and action probabilities required for counterfactual comparison |
| `RewardComponent` | `1` | Auditable bounded reward evidence from process checks, eval scores, human review, or bounded outcome labels |
| `ExperimentManifest` | `1` | Reproducible report configuration |
| `experiment_report.json` | `1` | Baseline-vs-candidate report, exclusions, confidence intervals, and known biases |

## Inputs

The first supported input is intent-router telemetry from `api/intent_router_training_store.py`. JSONL input can contain either already-normalized `LoggedDecisionExample` objects or raw intent-router training rows. Raw rows are converted with `logged_example_from_intent_router_row()`.

Useful source fields:

- `regex_baseline`, `llm_candidate`, `supervised_candidate`, and `applied_route`
- `shadow_comparison`
- `label_intent_class` and related reviewer labels
- optional `propensity` metadata

Rows without valid propensities are excluded from propensity-required counterfactual comparisons with explicit reasons. They can still be inspected in report exclusions.

## Rewards

Allowed reward sources:

- `process_reward`
- `eval_score`
- `human_review`
- `outcome_label`
- `synthetic`

Outcome labels are allowed only when mapped to process categories such as routing, tool selection, source quality, gate compliance, or stopping/defer discipline. Reward components reject future-leaking or direct return fields such as `forward_return_pct`, `benchmark_return_pct`, `realized_pnl`, `end_price`, and related P&L-only signals.

## Report CLI

Dry-run from persisted intent-router rows:

```bash
python -m decision_quality.agent_policy_experiments report --dry-run
```

Run from a JSONL fixture:

```bash
python -m decision_quality.agent_policy_experiments report \
  --input-jsonl docs/agent_policy_experiments/examples/router_rows.jsonl \
  --output-dir outputs/agent_policy_experiments
```

Allow inspection without valid propensities:

```bash
python -m decision_quality.agent_policy_experiments report \
  --input-jsonl /path/to/router_rows.jsonl \
  --allow-missing-propensity \
  --dry-run
```

The report records:

- baseline and candidate policy names
- row and evaluated counts
- exclusion reasons such as `missing_propensity`, `counterfactual_action_unobserved`, and `gate_boundary_violation`
- logged reward and candidate inverse-propensity reward confidence intervals
- `compare_reports`-style deterministic regression summary
- known offline-evaluation biases

## Offline-Only Boundary

This layer does not:

- alter `api/owned_model_rollout.py`
- change gateway policy or model provider routing
- run live canaries
- override source, policy, approval, or DecisionQuality gates
- persist a learned process policy for production use

Any future shadow or online policy test needs a separate approval issue and must consume these reports as pre-live evidence.

## Verification

```bash
pytest tests/test_agent_policy_experiments.py -q
```

Adjacent confidence checks:

```bash
pytest tests/test_agent_replay_environments.py tests/test_agent_trajectories.py -q
pytest tests/test_agent_training_datasets.py tests/test_agent_preference_training.py -q
pytest tests/test_owned_model_rollout.py tests/test_agent_owned_model_rollout.py -q
```

## Related Docs

- `docs/adr/014-offline-agent-policy-experiments.md`
- `docs/intent_router_rollout.md`
- `docs/talisman_agent_replay_environments.md`
- `docs/talisman_training_datasets.md`
- `docs/talisman_owned_model_rollout.md`
