# ADR-012: Preference Optimization Training From Reviewed Response Pairs

**Status:** Proposed
**Owner:** Shaiv Pandya
**Date:** 2026-06-08
**Revisit trigger:** Preference algorithm changes, new reward-source categories, or promotion evidence requirements that alter parent-lineage or ablation reporting.

## Context

`TL-93` must improve synthesis, calibration, and tool-use preferences using governed chosen/rejected pairs after the supervised candidate is stable (`TL-91`). The repo already exports `preference.jsonl` from `TL-90` and registers SFT candidates from `TL-91`, but preference optimization training, parent lineage, and SFT-parent evaluation were not yet implemented.

## Decision

**Extend the existing file-backed trainer/registry in** `decision_quality/agent_model_training.py` **with an isolated preference training mode** rather than creating a parallel registry.

Key properties:

1. **Config isolation** — `training_method=preference` uses separate validation (`dpo_trainable_count >= 1`, approved `parent_candidate_id`) from SFT configs.
2. **Complete pairs only** — reject-only rows remain exportable evidence but are excluded from DPO training via `dpo_trainable_count`.
3. **Parent lineage** — preference candidates record `parent_candidate_id`, parent artifact path, and optional SFT-parent TalismanBench comparison.
4. **Reward-source accounting** — dataset manifests and model cards record human, synthetic, and judge-assisted counts for ablation.
5. **Promotion evidence** — preference promotion requires passing frontier release gate plus non-regression vs SFT parent bench report.
6. **Smoke-first CI** — `preference_algorithm=smoke` produces deterministic artifacts without GPU dependencies; TRL DPO remains operator-run via `requirements-training.txt`.

## Alternatives Considered

| Alternative | Pros | Cons |
| --- | --- | --- |
| Extend existing trainer/registry (selected) | Reuses TL-91 patterns, one lifecycle model | Larger single module |
| Separate preference registry | Clear separation | Duplicates promotion/rollback logic |
| Train reject-only rows with implicit chosen | More data | Weak supervision, unclear DPO target |
| Online preference learning from production | Faster iteration | Violates TL-93 non-goals and governance |

## Risks

- Smoke preference artifacts are not inference-ready; operators must not promote without real bench evidence.
- Reject-only feedback remains useful for review but cannot train DPO until a chosen alternative exists.
- Parent regression checks depend on operators attaching both frontier and parent bench reports.

## References

- [agent_model_training.py](../../decision_quality/agent_model_training.py)
- [agent_training_datasets.py](../../decision_quality/agent_training_datasets.py)
- [talisman_agent_model_training.md](../talisman_agent_model_training.md)
- [talisman_training_datasets.md](../talisman_training_datasets.md)
- [ADR-011](011-agent-model-training-registry.md)
- Linear `TL-84`, `TL-89`, `TL-90`, `TL-91`, `TL-93`, `TL-97`
