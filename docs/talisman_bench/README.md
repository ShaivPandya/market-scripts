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
- candidate matrix path and selection protocol defaults

## Candidate matrix (TL-85)

The open-weight model and inference-host matrix lives at `docs/talisman_bench/candidate_matrix.json`. It records:

- at least three credible open-weight model candidates
- at least two hosting approaches
- combination metadata for model + host pairings
- smoke-case subsets for fast viability checks
- provisional primary/fallback selection and revisit triggers

TalismanBench validates the matrix offline and annotates release reports with matrix version, selection metadata, and combination details.

## CI gate (offline, no LLM)

The `eval-gates` job runs TalismanBench structural and leakage gates on every PR:

```bash
pytest tests/test_talisman_bench.py tests/test_bench_openai_client.py -q --tb=short

python -m decision_quality.talisman_bench \
  --manifest docs/talisman_bench/manifest.json \
  --approved-only \
  --dry-run
```

Offline gates validate:

- manifest shape and referenced paths
- candidate matrix shape and combination references
- approved-case metadata and input-ref integrity
- committed baseline inventory sync
- split-group leakage across the benchmark inventory
- dry-run execution across all three eval runners

## Model selection protocol (TL-85)

The selection protocol compares open-weight candidates against the frontier baseline using fixed settings:

| Setting | Value |
| --- | --- |
| Temperature | 0.0 |
| Max tokens | 4096 |
| Baseline provider/tier | `openai` / `mid` |
| Candidate protocol | OpenAI-compatible `chat.completions` |
| Latency tolerance | 15% P95 regression |
| Cost tolerance | 20% regression |
| Repeat-run tolerance | 5% scored-metric drift |

### Benchmark dimensions

- tool-call name accuracy and argument-schema validity
- structured-output schema validity
- deterministic gate compliance across structured, chat, and opportunity corpora
- latency P50/P95 from eval `elapsed_ms`
- token totals and estimated cost when combination pricing is configured
- failure behavior for malformed tool calls, schema failures, context overflow, timeout, and endpoint unavailability

### Smoke and holdout scopes

- `--smoke-only` runs the three-case smoke subset from the selected matrix combination.
- `--holdout-only` runs only inventory rows assigned to the `holdout` split.
- Full release checks run all 43 approved cases when neither flag is set.

### Chat/tool candidate routing

When TalismanBench evaluates a candidate against the chat corpus, it enables benchmark-only agent mode (`TALISMAN_BENCH_AGENT_MODE=1`). This routes agent streaming through the OpenAI-compatible candidate endpoint without changing production provider settings.

## Preference candidate evaluation (TL-93)

Preference-trained candidates should be evaluated against both:

1. the frontier baseline (existing release check)
2. the approved SFT parent candidate bench report

Attach both reports when registering or promoting a preference candidate:

```bash
python -m decision_quality.agent_model_training register-candidate \
  --artifact-dir outputs/agent_model_training/<pref_run_version> \
  --config configs/agent_preference_smoke.json \
  --bench-report outputs/talisman_bench/<candidate_timestamp>/release_report.json \
  --parent-bench-report outputs/talisman_bench/<parent_timestamp>/release_report.json
```

Promotion refuses preference candidates that introduce new deterministic failures relative to the SFT parent report. Model cards store the parent comparison summary and reward-source counts for ablation review.

## Manual release check (LLM-backed)

Manual release checks compare one external baseline against one OpenAI-compatible candidate. They require provider API credentials for the baseline and a reachable candidate endpoint.

```bash
export TALISMAN_BENCH_CANDIDATE_BASE_URL="http://localhost:8000/v1"
export TALISMAN_BENCH_CANDIDATE_API_KEY="local-key"
export TALISMAN_BENCH_CANDIDATE_MODEL="qwen2.5-7b-instruct"

python -m decision_quality.talisman_bench \
  --manifest docs/talisman_bench/manifest.json \
  --approved-only \
  --baseline-model mid \
  --combination-id qwen-local-vllm \
  --candidate-openai-base-url "$TALISMAN_BENCH_CANDIDATE_BASE_URL" \
  --candidate-api-key-env TALISMAN_BENCH_CANDIDATE_API_KEY \
  --candidate-model "$TALISMAN_BENCH_CANDIDATE_MODEL"
```

Smoke-only viability check:

```bash
python -m decision_quality.talisman_bench \
  --manifest docs/talisman_bench/manifest.json \
  --approved-only \
  --combination-id qwen-local-vllm \
  --smoke-only \
  --candidate-openai-base-url "$TALISMAN_BENCH_CANDIDATE_BASE_URL" \
  --candidate-model "$TALISMAN_BENCH_CANDIDATE_MODEL"
```

Reports are written to `outputs/talisman_bench/<timestamp>/release_report.json`.

## Release report contract

`release_report.json` separates:

- **Hard blockers**: manifest errors, leakage, structural failures, missing baseline inventory, deterministic failures, and baseline regressions. Any failed hard blocker rejects release.
- **Scored metrics**: deterministic pass rate, judge totals, latency, token use, and estimated cost when available. Scored metrics inform review but cannot override hard blockers.
- **Selection metadata**: smoke/holdout scope, combination id, and candidate matrix version.

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
- `TL-90` strictly excludes all approved release-gate cases from governed training exports regardless of hash split assignment. See `docs/talisman_training_datasets.md`.

## Candidate registration

After a manual release check, attach `release_report.json` to the candidate registry workflow documented in `docs/talisman_agent_model_training.md`. Registry promotion requires `release_gate.passed=true`.

## Managed endpoint validation (TL-95)

After the governed inference service is provisioned, point benchmark candidate env vars at the private managed endpoint and record capacity evidence:

```bash
export TALISMAN_BENCH_CANDIDATE_BASE_URL="${TALISMAN_BASE_URL}"
export TALISMAN_BENCH_CANDIDATE_API_KEY="${TALISMAN_API_KEY}"
export TALISMAN_BENCH_CANDIDATE_MODEL="${TALISMAN_MODEL_MID}"

python -m decision_quality.talisman_bench \
  --manifest docs/talisman_bench/manifest.json \
  --combination-id qwen-managed-gpu \
  --smoke-only
```

Store the resulting `release_report.json` with P50/P95 latency, throughput, and cost evidence before production rollout (`TL-92`). See `docs/talisman_inference_service.md`.

## Out of scope

- Production `talisman` provider integration (`TL-86`)
- Governed inference service provisioning (`TL-95`)
- Shadow/canary rollout controls — see `docs/talisman_owned_model_rollout.md` (`TL-92`)
- Replayable agent environments and process rewards (`TL-94`) — see `docs/talisman_agent_replay_environments.md`

## Documentation impact

This issue adds TalismanBench operating docs, the candidate matrix, and ADR-010 for the initial open-weight model/host selection. SFT/LoRA training and candidate registry are owned by `TL-91` (`docs/talisman_agent_model_training.md`). Program architecture guidance and the final documentation audit are owned by `TL-97` (`docs/talisman_owned_agent_model_program.md`).
