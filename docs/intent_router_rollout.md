# Intent router rollout checklist (TL-53)

## MVP defaults
- Confidence fallback cutoff: `0.70` (`AGENT_INTENT_ROUTER_CONFIDENCE_THRESHOLD`)
- Shadow burn-in: 3–7 days **or** 200 turns (whichever comes first)
- Enablement gate (Balanced):
  - No policy-gate regressions
  - No critical misroutes
  - Fallback rate `< 25%`
  - Top mismatch causes reviewed

## Environment flags
Set these on **Cloud Run** services that handle agent chat (`talisman-api` and `talisman-agent-worker`). They are included in `infra/gcp/lib.sh` `common_env_vars()` for deploy scripts.

| Variable | Deploy default | Purpose |
| --- | --- | --- |
| `AGENT_INTENT_ROUTER_ENABLED` | `true` | Master switch for LLM router calls |
| `AGENT_INTENT_ROUTER_SHADOW_MODE` | `true` | Log router-vs-regex diffs without changing behavior |
| `AGENT_INTENT_ROUTER_CONFIDENCE_THRESHOLD` | `0.70` | Minimum confidence before applying LLM routing |

Override at deploy time, e.g. `AGENT_INTENT_ROUTER_SHADOW_MODE=false ./infra/gcp/deploy-api.sh`, or edit env vars in the GCP console (Cloud Run → service → Edit → Variables).

## Phase A — shadow-only burn-in
1. Set `AGENT_INTENT_ROUTER_ENABLED=true` and `AGENT_INTENT_ROUTER_SHADOW_MODE=true`.
2. Collect `done.intent_router.telemetry.shadow_comparison` rows until burn-in window completes.
3. Run offline routing evals:
   ```bash
   python -m decision_quality.chat_eval_runner --routing-only
   ```
4. Review mismatch clusters (`tool_only_in_candidate`, hidden DQ disagreements, workflow disagreements).

## Phase B — controlled enablement
1. Set `AGENT_INTENT_ROUTER_SHADOW_MODE=false` after Balanced gate passes.
2. Keep instant rollback by setting `AGENT_INTENT_ROUTER_ENABLED=false`.
3. Monitor fallback rate and policy-gate regressions in agent logs.

## Phase C — tuning
1. Adjust threshold only after reviewing shadow telemetry.
2. Feed labeled rows into TL-55 supervised training loop:
   ```bash
   python -m decision_quality.intent_router_training export
   python -m decision_quality.intent_router_training train
   ```
3. Enable supervised rollout behind feature flags:
   - `AGENT_INTENT_ROUTER_SUPERVISED_ENABLED=true`
   - `AGENT_INTENT_ROUTER_SUPERVISED_MODEL_PATH=/path/to/model.joblib`
4. Persist production shadow rows with:
   - `AGENT_INTENT_ROUTER_TRAINING_CAPTURE_ENABLED=true`
   - optional `AGENT_INTENT_ROUTER_TRAINING_CAPTURE_MISMATCH_ONLY=true`

## Regression safety
- High-risk trade prompts must still route through hidden decision quality when regex baseline requires it.
- Policy-gated tool blocks remain unchanged; router only selects context/tools/flows.
