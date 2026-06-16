# Talisman Model Release Operations (TL-96)

`TL-96` operationalizes governed refresh, monitoring, approval, rollback, and retirement for Talisman-owned agent models. It composes existing training, bench, inference, and rollout primitives into one backend-only operating process without autonomous production promotion.

## Prerequisites

| Requirement | Source |
| --- | --- |
| Governed dataset export | [talisman_training_datasets.md](talisman_training_datasets.md) via `TL-90` |
| Candidate training and registry | [talisman_agent_model_training.md](talisman_agent_model_training.md) via `TL-91` |
| Release gate evidence | [talisman_bench/README.md](talisman_bench/README.md) via `TL-89` |
| Inference deployment | [talisman_inference_service.md](talisman_inference_service.md) via `TL-95` |
| Shadow/canary rollout | [talisman_owned_model_rollout.md](talisman_owned_model_rollout.md) via `TL-92` |

## Release workflow

```text
Stage 1  Dataset export (TL-90)
Stage 2  Train + register candidate (TL-91)
Stage 3  TalismanBench release report (TL-89)
Stage 4  Human approval + release record (TL-96)
Stage 5  Promote registry candidate (TL-91)
Stage 6  Build inference manifest + deploy (TL-95)
Stage 7  Shadow burn-in, then bounded canary (TL-92)
Stage 8  Ongoing monitoring + refresh review (TL-96)
```

Every stage below `Stage 5` can be validated with `--dry-run` without mutating production state.

## Refresh triggers

`decision_quality/agent_model_release_ops.py` evaluates:

| Trigger | Signal | Default action |
| --- | --- | --- |
| `new_reviewed_data` | Recent human-reviewed feedback count | Open refresh review |
| `failure_clusters` | Failure-tag totals from feedback | Open refresh review |
| `rollout_fallback_drift` | Owned-model fallback rate in trajectories | Rollback or refresh review |
| `gate_regression` | Deterministic gate failures in recent trajectories | Rollback review |
| `scheduled_review` | Scheduler/admin dry-run job | Open refresh review |
| `missing_active_candidate` | Registry has no active approved candidate | Block rollout |

Threshold env vars:

| Variable | Default |
| --- | --- |
| `AGENT_MODEL_RELEASE_FALLBACK_RATE_THRESHOLD` | `0.15` |
| `AGENT_MODEL_RELEASE_GATE_FAILURE_THRESHOLD` | `3` |
| `AGENT_MODEL_RELEASE_REVIEWED_FEEDBACK_THRESHOLD` | `5` |

## Dry-run release workflow

```bash
python -m decision_quality.agent_model_release_ops dry-run \
  --registry data/agent_model_candidates/registry.json \
  --candidate-id <candidate_id> \
  --output-dir outputs/model_release_ops
```

Outputs `outputs/model_release_ops/release_dry_run_<timestamp>.json` with:

- candidate summaries and promotion/deployment validation errors
- refresh triggers and drift alerts keyed by task class / candidate / fallback reason
- lineage pointers for dataset, bench report, model card, and rollback target
- `ready_for_promotion` and `ready_for_rollout` flags

Scheduled/admin dry-run:

```bash
curl -X POST "$API_URL/api/admin/jobs/enqueue-agent-model-release-refresh" \
  -H "X-Scheduler-Secret: $SCHEDULER_SECRET"
```

Enable weekly scheduler wiring with `SCHEDULE_AGENT_MODEL_RELEASE_REFRESH=1` in `infra/gcp/setup-scheduler.sh`. The job is **disabled by default** (`:-0`).

## Human approval and release records

Promotion and rollout still require explicit human approval. Record immutable decisions without changing registry state:

```bash
python -m decision_quality.agent_model_release_ops record-decision \
  --candidate-id <candidate_id> \
  --decision-type rollout_approved \
  --approver operator@example.com \
  --approval-note "Shadow burn-in complete; enable 5% canary for synthesis." \
  --bench-report outputs/talisman_bench/<timestamp>/release_report.json \
  --rollback-candidate-id <prior_active_candidate_id>
```

Decision types:

- `promotion_approved`
- `rollout_approved`
- `rollback`
- `retirement`
- `refresh_review`

Records are written to `outputs/model_release_ops/release_records/`.

## Monitoring and drift alerts

There is no dedicated dashboard in this slice. Monitoring uses:

- trajectory `raw_payload.owned_model_rollout` metadata
- human feedback failure tags
- registry lifecycle and bench evidence
- optional scheduled dry-run reports

Drift alerts identify affected `candidate_id`, task class, fallback reason, and recommend `monitor`, `refresh_review`, or `rollback_review`.

## Rollback drill

1. Set `AGENT_OWNED_MODEL_ROLLOUT_KILL_SWITCH=true` or gateway `owned_model_rollout.enabled=false` with `gateway_note`.
2. Disable the candidate in the registry if needed.
3. Set `provider_lifecycle.talisman=disabled` or model lifecycle to `disabled`.
4. Record a `rollback` decision with the prior active candidate as rollback target.
5. Verify frontier baseline restoration and unchanged deterministic gates.

See [talisman_owned_model_rollout.md](talisman_owned_model_rollout.md) for rollout-specific steps.

## Retirement

Retirement disables routing but preserves audit lineage:

```bash
python -m decision_quality.agent_model_release_ops retire \
  --candidate-id <candidate_id> \
  --approver operator@example.com \
  --retirement-note "Retire after rollback drill; keep artifacts for audit."
```

The command:

1. Sets registry lifecycle to `disabled` and clears `active_candidate_id`
2. Writes `outputs/model_release_ops/retirement_records/retirement_<candidate_id>_<timestamp>.json`
3. Emits a serving cleanup checklist for inference scale-down and secret rotation

Artifact directories, model cards, bench reports, and release records remain addressable for audit.

## Disabled, deprecated, and candidate behavior

| State | Routing | Audit retention |
| --- | --- | --- |
| `candidate` | Not routed; trained artifact with manifest | Full |
| `approved` | Eligible when referenced by rollout policy | Full |
| `deprecated` | Not routed; superseded | Full |
| `disabled` | Blocked from promotion, deploy, and rollout | Full |

## Verification

```bash
pytest tests/test_agent_model_release_ops.py -q
pytest tests/test_async_jobs.py tests/test_admin_jobs_security.py -q
pytest tests/test_llm_settings.py tests/test_owned_model_rollout.py tests/test_agent_owned_model_rollout.py -q
```

Failure-path checks:

- promotion/rollout decision without bench evidence is rejected
- disabled candidate retirement preserves lineage but clears active alias
- partial gateway settings updates preserve existing `owned_model_rollout`
- scheduled dry-run does not mutate registry or gateway policy

## Related docs

- Training/registry: [talisman_agent_model_training.md](talisman_agent_model_training.md)
- Rollout controls: [talisman_owned_model_rollout.md](talisman_owned_model_rollout.md)
- Inference service: [talisman_inference_service.md](talisman_inference_service.md)
- ADR: [adr/015-model-release-operations.md](adr/015-model-release-operations.md)
- Program architecture guide: [talisman_owned_agent_model_program.md](talisman_owned_agent_model_program.md) (`TL-97`)
