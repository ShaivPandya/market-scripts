# ADR 013: Owned-model shadow, canary, and fallback controls

## Status

Accepted — implements `TL-92`.

## Context

Talisman can serve an approved owned model through the first-party `talisman` provider (`TL-86`, `TL-95`) after passing TalismanBench release gates (`TL-89`). Production still needs governed routing that:

- preserves frontier baseline responses during shadow burn-in
- allows bounded canary allocation by task class and model version
- records explicit fallback reasons
- supports immediate rollback without redeploying application code

Prior rollout patterns exist for intent routing and synthesis supervised overlays, but owned-model routing requires provider-level orchestration tied to registry lifecycle and gateway policy.

## Decision

1. Persist rollout controls in `gateway_policy.owned_model_rollout` alongside existing provider/model lifecycle settings.
2. Implement rollout decisions in `api/owned_model_rollout.py` with deterministic canary bucketing keyed by session and turn id.
3. Wire agent chat to:
   - keep baseline output in shadow mode while recording candidate comparisons
   - route eligible canary traffic to `talisman` with automatic baseline fallback
   - emit rollout telemetry through SSE and trajectory metadata
4. Provide env emergency overrides:
   - `AGENT_OWNED_MODEL_ROLLOUT_KILL_SWITCH`
   - `AGENT_OWNED_MODEL_FORCE_BASELINE`

Deterministic policy gates in `api/agent_governance.py` remain authoritative for both candidate and fallback providers.

## Consequences

- Operators can change rollout posture through gateway settings with audited notes, plus env kill switches for incidents.
- Reporting is backend/event-based in this slice; dashboards can consume SSE, trajectories, and audit exports.
- Canary eligibility requires an approved registry candidate and non-disabled lifecycle state for the served model alias.
- Future UI work can expose the persisted policy without changing the rollout contract.

## References

- `docs/talisman_owned_model_rollout.md`
- `docs/talisman_owned_agent_model_program.md`
- `api/owned_model_rollout.py`
- `docs/talisman_inference_service.md`
- `docs/talisman_bench/README.md`
