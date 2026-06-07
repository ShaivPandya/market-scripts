# Talisman Agent Training Datasets

`TL-90` converts eligible trajectories, reviewed feedback, training-designated eval fixtures, and explicitly marked synthetic/teacher seeds into governed SFT and preference datasets.

## Sources

| Source | Eligibility | Output |
| --- | --- | --- |
| Sanitized trajectories + approved human feedback | Trajectory promoted for training; feedback `training_eligible=true`; response version matches | SFT example |
| Sanitized trajectories + reject/correct feedback | Same guards as above; never reads raw trajectory payloads | Preference example |
| Eval fixtures in `review` status | Explicitly training-designated; never includes approved TalismanBench release-gate cases | SFT example |
| Seed JSONL under `docs/agent_training_datasets/seeds/` | Must declare `source_type` of `synthetic` or `teacher` | SFT example |

Approved TalismanBench inventory cases are strictly excluded from training exports even when their hash split would otherwise be train or validation.

## Schemas

- SFT schema version: `1`
- Preference schema version: `1`
- Manifest version: `1`
- Transformation version: `agent_training_datasets_v1`

Every exported example includes:

- immutable `example_id`, `source_type`, and governed `source_id`
- task class, messages, and target or preference labels
- deterministic `split_group` and assigned `split`
- reviewer/teacher provenance and redaction manifest references
- `content_hash` for deduplication and rebuild verification

Teacher-generated examples are visibly marked through `signal_source=teacher` and never count as human-approved labels.

## CLI Export

```bash
python -m decision_quality.agent_training_datasets export --dry-run

python -m decision_quality.agent_training_datasets export \
  --export-version 20260607_fixed \
  --output-dir outputs/agent_training_datasets
```

Output layout:

```text
outputs/agent_training_datasets/<version>/
  sft.jsonl
  preference.jsonl
  manifest.json
```

The manifest records source counts, exclusion reason codes, split statistics, release-gate exclusion counts, leakage results, and content hashes. Rebuilding with the same source snapshot, configuration, and `--export-version` reproduces manifest hashes.

## Admin Export API

Admin-only endpoint:

```http
POST /api/admin/agent/training-datasets/export?dry_run=false
```

Query parameters:

- `limit` (default `5000`)
- `dry_run` (default `false`)
- `include_eval_fixtures` (default `true`)
- `include_seeds` (default `true`)

The endpoint returns manifest metadata only. It does not expose raw trajectory payloads. Successful exports emit an `agent_training_dataset_export` audit event in the `agent_learning` category with counts, exclusion stats, and content hashes.

## Exclusion And Leakage Rules

Exports exclude or fail rows when:

- a source matches an approved TalismanBench release-gate case
- a non-trajectory row reuses a release-gate split group
- feedback lacks a matching exportable trajectory or response version
- duplicate `content_hash` values appear in the same export
- split-group leakage appears across train/validation/holdout assignments

Release-gate contamination is a hard export failure. Other ineligible rows are recorded in `manifest.exclusions` with explicit reason codes.

## Review Workflow

Human feedback promotion remains owned by `TL-88`:

- `approve` + explicit training opt-in promotes trajectories
- `reject` and `correct` can export preference labels without auto-promoting trajectories

Synthetic and teacher seed rows may carry `review_status` of `pending`, `released`, or `rejected`. Only released rows should be included in downstream training jobs unless an operator explicitly overrides that policy in a future release workflow.

## Downstream Training

Governed exports feed the SFT/LoRA trainer and candidate registry owned by `TL-91`:

- Trainer/registry CLI: `decision_quality/agent_model_training.py`
- Operating docs: `docs/talisman_agent_model_training.md`

Training jobs must reference the exported `manifest.json` path and refuse datasets with `leakage_check_passed=false`.

## Related Docs

- Trajectory contract: `docs/talisman_trajectories.md`
- TalismanBench held-out rules: `docs/talisman_bench/README.md`
- Agent model training/registry: `docs/talisman_agent_model_training.md`
- Cross-cutting architecture audit: `TL-97`
