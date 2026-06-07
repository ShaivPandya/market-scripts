# Talisman Agent Model Training And Registry

`TL-91` trains the first Talisman-owned generative agent candidates from governed SFT datasets and registers them with reproducible evidence, model cards, and promotion gates.

## Inputs

| Input | Source | Requirement |
| --- | --- | --- |
| Governed SFT dataset | `outputs/agent_training_datasets/<version>/` from `TL-90` | `manifest.json` with `leakage_check_passed=true` and `sft_count >= 1` |
| Base model selection | `docs/talisman_bench/candidate_matrix.json` and ADR-010 | Default primary: `Qwen/Qwen2.5-7B-Instruct` via `qwen-local-vllm` |
| Release evidence | `outputs/talisman_bench/<timestamp>/release_report.json` | Required for registry promotion |

## Trainer config

Trainer configs are versioned JSON objects validated by `decision_quality/agent_model_training.py`.

Required fields:

- `base_model_id` and optional `base_model_revision`
- `dataset_manifest_path`
- `chat_template`
- `lora` hyperparameters (`rank`, `alpha`, `dropout`, `target_modules`, `use_qlora`)
- `training` hyperparameters (`epochs`, batch sizes, `learning_rate`, `max_seq_length`, `seed`)
- `trainer_backend`: `smoke` for CI and pipeline validation; `trl` or `peft` for operator-run GPU training
- optional `serve` metadata (`served_model_name`, `combination_id`)

Initialize a default config:

```bash
python -m decision_quality.agent_model_training init-config \
  --dataset-manifest outputs/agent_training_datasets/<version>/manifest.json \
  --output configs/agent_sft_smoke.json
```

Validate a config:

```bash
python -m decision_quality.agent_model_training validate-config \
  --config configs/agent_sft_smoke.json
```

## Smoke training (CI-safe)

Smoke training produces deterministic artifact directories without GPU execution. Use a pinned `--run-version` to reproduce candidate ids and artifact digests.

```bash
python -m decision_quality.agent_model_training smoke-train \
  --config configs/agent_sft_smoke.json \
  --run-version pinned_smoke_v1 \
  --output-dir outputs/agent_model_training
```

Artifact layout:

```text
outputs/agent_model_training/<run_version>/
  adapter_config.json
  metrics.json
  model_card.json
  training_manifest.json
```

## Register, promote, rollback

Register a trained artifact:

```bash
python -m decision_quality.agent_model_training register-candidate \
  --artifact-dir outputs/agent_model_training/<run_version> \
  --config configs/agent_sft_smoke.json \
  --bench-report outputs/talisman_bench/<timestamp>/release_report.json
```

Promote to approved (requires passing TalismanBench release gate):

```bash
python -m decision_quality.agent_model_training promote \
  --candidate-id <candidate_id> \
  --bench-report outputs/talisman_bench/<timestamp>/release_report.json
```

Rollback / retirement:

```bash
python -m decision_quality.agent_model_training deprecate --candidate-id <candidate_id>
python -m decision_quality.agent_model_training disable --candidate-id <candidate_id>
```

Promotion refuses candidates when:

- dataset leakage checks failed
- model card is missing required fields
- artifact digest no longer matches immutable manifest
- TalismanBench `release_gate.passed` is false

Deprecating or disabling the active approved candidate clears the registry alias without deleting artifacts.

## Operator-run GPU training

Real SFT/LoRA training uses optional dependencies from `requirements-training.txt` and remains operator-run outside normal API installs.

Suggested workflow:

1. Export governed dataset (`TL-90`).
2. Initialize and validate trainer config.
3. Run TRL/PEFT training against `sft.jsonl` using the pinned base model from ADR-010.
4. Serve merged weights through local vLLM OpenAI-compatible endpoint.
5. Run TalismanBench manual release check from `docs/talisman_bench/README.md`.
6. Register artifact directory, attach release report, and promote if gates pass.

Example vLLM serve command (operator environment):

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --max-model-len 32768
```

Point TalismanBench candidate env vars at the served endpoint before registration.

## Registry

Committed registry path: `data/agent_model_candidates/registry.json`

Lifecycle states:

- `candidate`: trained artifact with manifest and model card
- `approved`: passed promotion gates; may be referenced by registry alias
- `deprecated`: superseded but retained for audit/replay
- `disabled`: blocked from promotion or routing

## Model card

Each candidate publishes `model_card.json` with:

- dataset lineage and content hashes
- intended task classes
- limitations and known failures
- license reference
- training metrics and optional bench summary

## Related docs

- Dataset curation: `docs/talisman_training_datasets.md`
- Release gate: `docs/talisman_bench/README.md`
- Base model/host ADR: `docs/adr/010-open-weight-base-model-and-inference-host.md`
- Trainer/registry ADR: `docs/adr/011-agent-model-training-registry.md`
- Cross-cutting architecture audit: `TL-97`
