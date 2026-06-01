# Supervised synthesis rollout (TL-67)

## MVP defaults
- Confidence fallback cutoff: `0.70` (`AGENT_SYNTHESIS_SUPERVISED_CONFIDENCE_THRESHOLD`)
- Shadow burn-in: run shadow mode until offline eval gates pass
- Enablement gate:
  - Triage accuracy `>= 0.70` on holdout
  - Stance accuracy `>= 0.65` on holdout
  - No increase in deterministic gate failures vs prompt-only baseline

## Environment flags

| Variable | Deploy default | Purpose |
| --- | --- | --- |
| `AGENT_SYNTHESIS_SUPERVISED_ENABLED` | `false` | Master switch for applying supervised triage overlay |
| `AGENT_SYNTHESIS_SUPERVISED_SHADOW_MODE` | `true` | Log supervised-vs-LLM diffs without changing behavior |
| `AGENT_SYNTHESIS_SUPERVISED_CONFIDENCE_THRESHOLD` | `0.70` | Minimum confidence before applying supervised triage |
| `AGENT_SYNTHESIS_SUPERVISED_MODEL_PATH` | unset | Path to joblib artifact from training |

## Phase A — export and train
```bash
python -m decision_quality.synthesis_supervised_training export
python -m decision_quality.synthesis_supervised_training train
python -m decision_quality.synthesis_supervised_training eval \
  --dataset outputs/synthesis_supervised_training/<version>/dataset.jsonl \
  --model data/synthesis_supervised_models/<version>/model.joblib
```

## Phase B — shadow burn-in
1. Set `AGENT_SYNTHESIS_SUPERVISED_SHADOW_MODE=true` (default).
2. Optionally set `AGENT_SYNTHESIS_SUPERVISED_MODEL_PATH` to the trained artifact.
3. Review `done.opportunity_candidate_preflight.supervised_triage.shadow_comparison` telemetry.
4. Run offline eval with rollout gates:
   ```bash
   python -m decision_quality.eval_runner --approved-only --dry-run \
     --supervised-model data/synthesis_supervised_models/<version>/model.joblib
   python -m decision_quality.chat_eval_runner --approved-only --dry-run \
     --supervised-model data/synthesis_supervised_models/<version>/model.joblib
   ```

## Phase C — controlled enablement
1. Set `AGENT_SYNTHESIS_SUPERVISED_ENABLED=true` and `AGENT_SYNTHESIS_SUPERVISED_SHADOW_MODE=false`.
2. Keep instant rollback via `AGENT_SYNTHESIS_SUPERVISED_ENABLED=false`.
3. Deterministic DQ/source/policy gates remain authoritative; supervised only adjusts pre-gate triage stance.

## Dataset sources
- Structured DQ eval cases (`docs/decision_quality_evals/cases/`)
- Chat eval cases with opportunity preflight or expected stance (`docs/decision_quality_chat_evals/cases/`)
- OpportunityCandidate eval fixtures (`docs/opportunity_candidate_evals/cases/`)

Only `review` and `approved` cases export to training data. Splits use deterministic hashing with leakage checks on `split_group`.
