# Owned-model rollout controls (TL-92)

`TL-92` introduces governed production routing for approved Talisman-owned models through shadow burn-in, bounded canary allocation, explicit fallback reasons, lifecycle enforcement, and immediate rollback without redeploying application code.

## Control plane

Rollout policy is persisted in gateway settings under `gateway_policy.owned_model_rollout` and validated through `/api/settings/llm`.

Emergency break-glass overrides are env-only:

| Variable | Default | Purpose |
| --- | --- | --- |
| `AGENT_OWNED_MODEL_ROLLOUT_KILL_SWITCH` | `false` | Disable all owned-model shadow and canary routing immediately |
| `AGENT_OWNED_MODEL_FORCE_BASELINE` | `false` | Force frontier baseline responses even if rollout is enabled |
| `AGENT_OWNED_MODEL_SHADOW_MODE` | unset | Optional override for persisted `shadow_enabled` |
| `AGENT_OWNED_MODEL_CANARY_ENABLED` | unset | Optional override for persisted `canary_enabled` |

GCP deploy defaults are in `infra/gcp/lib.sh` `common_env_vars()`.

## Persisted rollout policy

Example gateway policy fragment:

```json
{
  "owned_model_rollout": {
    "enabled": false,
    "shadow_enabled": true,
    "canary_enabled": false,
    "canary_percent": 0,
    "min_confidence": 0.70,
    "approved_task_classes": [
      "agent_turn",
      "synthesis",
      "routing",
      "routing_tool_use",
      "tool_use",
      "structured_output"
    ],
    "approved_candidate_id": null,
    "approved_model_ids": [],
    "candidate_provider": "talisman",
    "rule_version": "owned_model_rollout_v1"
  }
}
```

Update through AI Settings or API with a required `gateway_note` audit entry.

## Prerequisites

| Requirement | Source |
| --- | --- |
| Approved registry candidate | `data/agent_model_candidates/registry.json` via `TL-91` |
| Governed inference endpoint | `docs/talisman_inference_service.md` via `TL-95` |
| Provider adapter | `talisman_openai_compat.py` via `TL-86` |
| Release evidence | `outputs/talisman_bench/<timestamp>/release_report.json` via `TL-89` |

Do not enable rollout until:

1. `approved_candidate_id` points to an `approved` registry entry.
2. Talisman provider lifecycle is not `disabled`.
3. `TALISMAN_BASE_URL` and model aliases point at the governed inference service.
4. TalismanBench release gate passed for the candidate.

## Phase A — shadow burn-in

1. Keep `LLM_PROVIDER` on the frontier baseline.
2. Set `gateway_policy.owned_model_rollout.enabled=true` and `shadow_enabled=true`.
3. Set `approved_candidate_id` to the promoted candidate.
4. Leave `canary_enabled=false` and `canary_percent=0`.
5. Monitor agent SSE events:
   - `owned_model_rollout`
   - `done.owned_model_rollout.reporting`
   - `done.owned_model_rollout.telemetry.shadow_comparison`
6. Burn-in gate (Balanced):
   - No deterministic policy, source-quality, approval, or decision-quality gate regressions
   - Comparable candidate/baseline trajectories recorded without changing user-visible output
   - Review fallback reasons and mismatch clusters before enabling canary

## Phase B — bounded canary

1. After shadow burn-in passes, set `canary_enabled=true`.
2. Increase `canary_percent` gradually (for example `1`, `5`, `10`, `25`).
3. Restrict `approved_task_classes` to the first production task class you are graduating.
4. Optionally pin `approved_model_ids` to the served alias from the deployment manifest.
5. Monitor:
   - fallback rate and explicit fallback reasons
   - latency and estimated cost by provider in `done.timings.models`
   - gate outcomes and tool-call validity in trajectories

## Phase C — rollback and kill switch

Instant rollback options:

1. Set `AGENT_OWNED_MODEL_ROLLOUT_KILL_SWITCH=true` on Cloud Run services handling agent chat.
2. Set `AGENT_OWNED_MODEL_FORCE_BASELINE=true` to keep rollout telemetry enabled but force baseline responses.
3. Set gateway `owned_model_rollout.enabled=false` with a `gateway_note`.
4. Set gateway `provider_lifecycle.talisman=disabled` or model lifecycle to `disabled` for the served alias.

Rollback drill:

1. Enable canary on a non-production environment at low allocation.
2. Trigger a controlled candidate failure (endpoint outage or lifecycle disablement).
3. Verify frontier baseline restoration, explicit fallback reason emission, and unchanged deterministic gates.
4. Restore candidate lifecycle and confirm shadow telemetry resumes.

## Fallback taxonomy

Every fallback is recorded with a stable reason from `api/owned_model_rollout.py`:

- `rollout_disabled`
- `kill_switch_active`
- `force_baseline_active`
- `task_class_not_eligible`
- `candidate_not_approved`
- `candidate_lifecycle_disabled`
- `candidate_unavailable`
- `provider_lifecycle_disabled`
- `model_lifecycle_disabled`
- `confidence_below_threshold`
- `unsupported_capability`
- `endpoint_failure`
- `endpoint_timeout`
- `malformed_output`
- `schema_failure`
- `policy_denied`
- `gate_failure`
- `canary_not_selected`

## Reporting surfaces

There is no dedicated rollout dashboard in this slice. Use:

- SSE `owned_model_rollout` and `done.owned_model_rollout.reporting`
- egress manifests on `egress_recorded`
- agent trajectories (`raw_payload.owned_model_rollout`, model/tool steps)
- audit events for gateway policy changes

Aggregate offline from trajectory exports or log pipelines when building operator dashboards.

## Offline Policy Experiments (`TL-68`)

`TL-68` can use rollout telemetry and gate patterns as evidence, but it does not change owned-model rollout behavior. Offline contextual-bandit reports live in `decision_quality/agent_policy_experiments.py` and currently start from intent-router logged choices, not provider-level canary routing.

Do not enable canary allocation, change gateway policy, or route user-visible traffic based on a TL-68 experiment report without a later explicit approval issue. Source, policy, approval, and DecisionQuality gates remain non-overridable for both rollout and offline policy experiments.

## Release operations integration (TL-96)

Before increasing canary allocation or promoting a new registry candidate:

1. Run `python -m decision_quality.agent_model_release_ops dry-run --candidate-id <candidate_id>`.
2. Record a human `rollout_approved` decision with approver, bench report, and rollback target.
3. Review drift alerts from trajectory fallback summaries and feedback failure clusters.

Rollback and retirement decisions should also be recorded through `record-decision` or `retire` so lineage from production failure → feedback → dataset → candidate → bench → rollout remains auditable. See `docs/talisman_model_release_operations.md`.

Partial gateway settings updates must preserve an existing `owned_model_rollout` block when rollout fields are omitted.

## Graduation thresholds

Production graduation requires the thresholds defined by `TL-84` and `TL-89`:

- No regression in deterministic policy, source-quality, approval, or decision-quality gate failures
- Tool selection and argument validity meet or exceed the approved mid-tier frontier baseline on TalismanBench
- Held-out synthesis quality meets the approved benchmark threshold
- P95 latency and cost improve for routed task classes
- Frontier fallback succeeds for endpoint failures, low confidence, and unsupported tasks

Run TalismanBench against the managed endpoint before increasing canary allocation:

```bash
python -m decision_quality.talisman_bench \
  --candidate-base-url "$TALISMAN_BASE_URL" \
  --candidate-api-key "$TALISMAN_API_KEY" \
  --candidate-model "$TALISMAN_MODEL_MID"
```

## Verification

```bash
pytest tests/test_owned_model_rollout.py tests/test_agent_owned_model_rollout.py -q
pytest tests/test_llm_settings.py tests/test_agent_governance.py -q
pytest tests/test_agent_policy_experiments.py -q
pytest tests/test_agent_model_release_ops.py -q
```

Failure-path checks to exercise manually or in staging:

- endpoint outage → `endpoint_failure` fallback to baseline
- timeout → `endpoint_timeout`
- malformed tool call/output → `malformed_output`
- schema/gate failure → `schema_failure` or `gate_failure`
- lifecycle disablement → `provider_lifecycle_disabled` / `model_lifecycle_disabled`
- kill switch → immediate baseline-only traffic

## Related docs

- Program architecture guide: [talisman_owned_agent_model_program.md](talisman_owned_agent_model_program.md) (`TL-97`)
- Release operations: [talisman_model_release_operations.md](talisman_model_release_operations.md) (`TL-96`)
- Inference service: [talisman_inference_service.md](talisman_inference_service.md) (`TL-95`)
- Training/registry: [talisman_agent_model_training.md](talisman_agent_model_training.md) (`TL-91`)
