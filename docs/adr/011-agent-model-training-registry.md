# ADR-011: Agent Model Training And Candidate Registry

**Status:** Proposed
**Owner:** Shaiv Pandya
**Date:** 2026-06-07
**Revisit trigger:** Trainer backend changes, new base-model generation, or promotion gate policy updates that alter reproducibility or release evidence requirements.

## Context

`TL-91` must train the first Talisman-owned generative agent candidates from governed datasets (`TL-90`), evaluate them through TalismanBench (`TL-89`), and register immutable artifacts with promotion gates before any production routing (`TL-92`).

The repo already has lightweight sklearn registries for routing and synthesis overlays. Generative SFT/LoRA candidates require stronger lineage: dataset content hashes, trainer config hashes, artifact digests, model cards, and explicit lifecycle states.

## Decision

**Adopt a file-backed candidate registry** under `data/agent_model_candidates/` with CLI orchestration in `decision_quality/agent_model_training.py`.

Key properties:

1. **Immutable candidate identity** — `candidate_id` and `artifact_digest` derive from trainer config hash plus dataset content hashes, not mutable registry aliases.
2. **Offline-first smoke path** — `trainer_backend=smoke` produces deterministic artifacts for CI without GPU dependencies.
3. **Optional heavyweight deps** — TRL/PEFT packages live in `requirements-training.txt`, not core `requirements.txt`.
4. **Promotion gates** — `approved` state requires model card completeness, dataset leakage pass, matching artifact digest, and TalismanBench `release_gate.passed=true`.
5. **Rollback** — `deprecate` and `disable` clear active aliases without deleting artifact directories.

## Alternatives Considered

| Alternative | Pros | Cons |
| --- | --- | --- |
| File-backed registry + CLI (selected) | Matches existing supervised model patterns, easy audit, no new DB schema | Manual operator steps for GPU training |
| Postgres registry table | Strong concurrency and API integration | Overkill before `TL-92` rollout wiring |
| Inline training in API admin jobs | Convenient triggering | Couples GPU workloads to API runtime and complicates reproducibility |
| Core `requirements.txt` ML deps | Simpler install story | Slows CI and API environments that never train |

## Risks

- Smoke artifacts are not inference-ready; operators must not promote them without real bench evidence on served weights.
- File-backed registry requires discipline to avoid editing artifacts after registration.
- TRL/PEFT training remains environment-dependent and is not fully automated in this issue.

## References

- [agent_model_training.py](../../decision_quality/agent_model_training.py)
- [talisman_agent_model_training.md](../talisman_agent_model_training.md)
- [talisman_training_datasets.md](../talisman_training_datasets.md)
- [talisman_bench/README.md](../talisman_bench/README.md)
- [ADR-010](010-open-weight-base-model-and-inference-host.md)
- [ADR-012](012-preference-optimization-training.md)
- Linear `TL-84`, `TL-89`, `TL-90`, `TL-91`, `TL-92`, `TL-93`, `TL-97`
