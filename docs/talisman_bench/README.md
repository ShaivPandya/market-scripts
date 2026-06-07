# TalismanBench

TalismanBench is the release gate for owned Talisman agent model candidates. It orchestrates the existing structured DecisionQuality, chat, and OpportunityCandidate eval corpora, enforces split-group leakage checks, and produces candidate-versus-baseline release reports.

Production first-party provider wiring remains owned by `TL-86`. TalismanBench includes a benchmark-only OpenAI-compatible client for manual release checks against candidate endpoints.

## Manifest

The committed manifest lives at `docs/talisman_bench/manifest.json`. It defines:

- included corpora and baseline references
- benchmark dimensions per corpus
- split-group policy for leakage checks
- hard blockers that fail the release gate
- scored metrics reported alongside blockers
- graduation thresholds and baseline-regression tolerances
- baseline and candidate model configuration

## CI gate (offline, no LLM)

The `eval-gates` job runs TalismanBench structural and leakage gates on every PR:

```bash
pytest tests/test_talisman_bench.py -q --tb=short

python -m decision_quality.talisman_bench \
  --manifest docs/talisman_bench/manifest.json \
  --approved-only \
  --dry-run
```

Offline gates validate:

- manifest shape and referenced paths
- approved-case metadata and input-ref integrity
- committed baseline inventory sync
- split-group leakage across the benchmark inventory
- dry-run execution across all three eval runners

## Manual release check (LLM-backed)

Manual release checks compare one external baseline against one OpenAI-compatible candidate. They require provider API credentials for the baseline and a reachable candidate endpoint.

```bash
export TALISMAN_BENCH_CANDIDATE_BASE_URL="http://localhost:8000/v1"
export TALISMAN_BENCH_CANDIDATE_API_KEY="local-key"
export TALISMAN_BENCH_CANDIDATE_MODEL="talisman-owned-v1"

python -m decision_quality.talisman_bench \
  --manifest docs/talisman_bench/manifest.json \
  --approved-only \
  --baseline-model mid \
  --candidate-openai-base-url "$TALISMAN_BENCH_CANDIDATE_BASE_URL" \
  --candidate-api-key-env TALISMAN_BENCH_CANDIDATE_API_KEY \
  --candidate-model "$TALISMAN_BENCH_CANDIDATE_MODEL"
```

Reports are written to `outputs/talisman_bench/<timestamp>/release_report.json`.

## Release report contract

`release_report.json` separates:

- **Hard blockers**: manifest errors, leakage, structural failures, missing baseline inventory, deterministic failures, and baseline regressions. Any failed hard blocker rejects release.
- **Scored metrics**: deterministic pass rate, judge totals, latency, token use, and estimated cost when available. Scored metrics inform review but cannot override hard blockers.

## Graduation thresholds

Current manifest defaults:

| Threshold | Value |
| --- | --- |
| Minimum deterministic pass rate | 95% |
| Maximum new deterministic failures vs baseline | 0 |
| Minimum judge total mean | 14.0 |
| Maximum latency P95 regression | 15% |
| Maximum cost regression | 20% |

Adjust thresholds in the manifest only after an intentional benchmark policy change.

## Held-out data rules

- Approved benchmark cases are the held-out release inventory.
- Split groups are assigned deterministically from case metadata.
- A split-group collision across train/eval/holdout partitions fails the run.
- Do not train on held-out TalismanBench cases.

## Out of scope

- Production `talisman` provider integration (`TL-86`)
- Training dataset curation (`TL-90`) and SFT/LoRA registry (`TL-91`)
- Shadow/canary rollout controls (`TL-92`)

## Documentation impact

This issue adds TalismanBench operating docs and manifest references. Cross-cutting architecture guidance and the final documentation audit remain owned by `TL-97`.
