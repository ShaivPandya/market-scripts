# Agent Replay Environments

`TL-94` adds replayable agent environments and process-level reward functions for offline evaluation and future RL experiments.

## Purpose

The replay harness lets agent candidates execute representative Talisman tasks in a deterministic, mock-backed environment without live market dependencies or production routing changes. Each episode produces:

- a versioned observation from `reset`
- an executable chat-eval backend run with `mock_tools`
- decomposable process rewards mapped from existing deterministic checks
- an exportable trajectory compatible with the TL-87 step vocabulary

## Contract Versions

| Field | Value |
| --- | --- |
| `environment_schema_version` | `1` |
| `reward_schema_version` | `1` |
| First backend | `chat_eval` |

Environment cases live under `docs/agent_replay_environments/cases/`. Approved chat eval cases can also be included when `--include-chat-eval-cases` is set.

## Reward Components

Process rewards decompose existing deterministic checks into auditable categories:

| Category | Example checks |
| --- | --- |
| `tool_selection` | `expected_tool_coverage`, `routing_required_tool_names` |
| `argument_validity` | `workflow_tool_metadata` |
| `source_quality` | `tool_quality_*`, stale/blocked source probes |
| `structured_output` | `dimension_*`, `required_point_*`, `no_raw_json` |
| `gate_compliance` | `gate_action_consistency`, scout/skeptic/sizer checks |
| `missing_input_recognition` | missing-input and blocker language checks |
| `efficiency` | `max_tool_calls`, latency (when configured) |
| `stopping_defer` | stance, forbidden actionable language, nonempty answer |

## Anti-Gaming Probes

Probe fixtures under `docs/agent_replay_environments/cases/` cover:

| Probe | Intent |
| --- | --- |
| `shortcut` | Skipping required tools |
| `fabricated_source` | Claiming unavailable source evidence |
| `excessive_tool` | Calling more tools than allowed |
| `policy_boundary` | Actionable language when gates block |
| `premature_stop` | Empty or incomplete answers |

## Usage

Dry-run inventory:

```bash
python -m decision_quality.agent_replay_environments --approved-only --dry-run
```

Run approved probe fixtures (requires configured auth/LLM for live agent execution):

```bash
python -m decision_quality.agent_replay_environments --approved-only
```

Parallel smoke:

```bash
python -m decision_quality.agent_replay_environments --approved-only --parallel --max-workers 4
```

Offline unit tests score crafted runs without live model calls:

```bash
pytest tests/test_agent_replay_environments.py -q
```

## Non-Production Boundary

Replay execution:

- patches tools through the chat eval runner (`mocked_tool_executor`)
- disables governance audit by default
- does **not** change production `LLM_PROVIDER`, gateway policy, or TL-92 rollout controls
- exports trajectories with `training_eligible=false` and `exclusion_reasons=["replay_environment_not_production_capture"]`

Production shadow/canary routing remains owned by `TL-92`. Offline RL experiments remain owned by `TL-68` and require this environment layer plus stable rollout gates.

## Offline Policy Experiments (`TL-68`)

`TL-68` consumes replay reward categories as validation evidence for process rewards, but it does not treat replay-only trajectories as production logged bandit rows. Contextual-bandit comparisons require logged action and propensity metadata, currently starting from intent-router training rows in `api/intent_router_training_store.py`.

See `docs/talisman_offline_policy_experiments.md` and `docs/adr/014-offline-agent-policy-experiments.md` for the offline-only report contract.

## Related Docs

- [talisman_trajectories.md](talisman_trajectories.md) — trajectory step vocabulary
- [talisman_bench/README.md](talisman_bench/README.md) — release gate and benchmark inventory
- [decision_quality_chat_evals/README.md](decision_quality_chat_evals/README.md) — mock tool replay cases
- [talisman_agent_model_training.md](talisman_agent_model_training.md) — candidate registry and training lifecycle
- [talisman_offline_policy_experiments.md](talisman_offline_policy_experiments.md) — contextual-bandit reports for agent-process choices

## References

- Linear `TL-84`, `TL-87`, `TL-89`, `TL-90`, `TL-91`, `TL-94`, `TL-68`, `TL-92`, `TL-97`
- Implementation: `decision_quality/agent_replay_environments.py`
