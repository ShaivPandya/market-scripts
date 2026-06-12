# ADR-015: Model Release Operations

**Status:** Accepted — implements `TL-96`  
**Owner:** Shaiv Pandya  
**Date:** 2026-06-12  
**Revisit trigger:** Need for DB-backed release workflow concurrency, external alert delivery, or autonomous promotion policy.

## Context

The owned-model program already has governed dataset export (`TL-90`), training/registry (`TL-91`), TalismanBench (`TL-89`), inference deployment (`TL-95`), and shadow/canary rollout (`TL-92`). Operators still run these steps manually across separate CLIs and runbooks.

`TL-96` must turn one-off delivery into a controlled operating process with refresh triggers, human approval, monitoring summaries, rollback/retirement records, and dry-run refresh without production mutation.

## Decision

1. **Add `decision_quality/agent_model_release_ops.py`** as a file-backed release operations layer.
2. **Persist dry-run reports, release decisions, and retirement records** under `outputs/model_release_ops/`.
3. **Expose CLI commands** `dry-run`, `record-decision`, and `retire`; do not add autonomous promotion.
4. **Add scheduled/admin dry-run job** `agent_model_release_refresh` via existing maintenance job infrastructure.
5. **Use trajectory and feedback summaries** for backend-only drift/regression alerts; defer dashboards and external paging.
6. **Preserve existing `owned_model_rollout` policy** when gateway settings updates omit rollout fields.

## Alternatives Considered

| Alternative | Pros | Cons |
| --- | --- | --- |
| File-backed ops layer + CLI (selected) | Matches TL-91 registry pattern; easy audit; no new schema | Manual GPU training remains operator-run |
| Postgres release workflow tables | Strong concurrency and API integration | Overkill before ops process is proven |
| Frontend operator dashboard | Better visibility | Out of scope for TL-96; telemetry already exists in SSE/trajectories |
| External alert delivery | Faster incident response | Requires paging integration not present in repo |

## Risks

- Dry-run monitoring depends on trajectory/feedback volume; low traffic yields sparse alerts.
- File-backed release records require operator discipline to avoid editing artifacts after write.
- Retirement removes routing but does not automatically delete cloud serving artifacts or secrets.

## References

- [agent_model_release_ops.py](../../decision_quality/agent_model_release_ops.py)
- [talisman_model_release_operations.md](../talisman_model_release_operations.md)
- [talisman_agent_model_training.md](../talisman_agent_model_training.md)
- [talisman_owned_model_rollout.md](../talisman_owned_model_rollout.md)
- [ADR-011](011-agent-model-training-registry.md)
- [ADR-013](013-owned-model-rollout-controls.md)
- Linear `TL-84`, `TL-89`, `TL-91`, `TL-92`, `TL-95`, `TL-96`, `TL-97`
- [Program architecture guide](../talisman_owned_agent_model_program.md)
