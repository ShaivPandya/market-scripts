# Talisman Governed Inference Service

`TL-95` provisions the first-party OpenAI-compatible inference service as a governed production dependency independent of the Talisman application deployment.

## Service contract

The inference service exposes the capabilities required by `TL-86`:

- `/v1/chat/completions` with text, streaming, tool calling, and structured JSON output
- model identity on every governed startup/readiness response: `candidate_id`, `artifact_digest`, `served_model_name`, `combination_id`, `runtime_version`
- refusal to start when registry lifecycle is not `approved` or artifact digest no longer matches

Application code consumes the service only through `TALISMAN_*` environment variables. It does not depend on host-specific control APIs.

## Prerequisites

| Requirement | Source |
| --- | --- |
| Approved registry candidate | `data/agent_model_candidates/registry.json` via `TL-91` |
| Candidate matrix serving params | `docs/talisman_bench/candidate_matrix.json` via `TL-85` |
| Provider adapter | `talisman_openai_compat.py` via `TL-86` |
| Release evidence | `outputs/talisman_bench/<timestamp>/release_report.json` via `TL-89` |

Smoke-trained artifacts are not inference-ready. Promote only candidates with real served weights and passing TalismanBench evidence.

## Deployment manifest

Build and validate a deployment manifest before provisioning:

```bash
python -m decision_quality.agent_inference_deployment validate \
  --candidate-id <approved_candidate_id>

python -m decision_quality.agent_inference_deployment build-manifest \
  --candidate-id <approved_candidate_id> \
  --environment nonprod \
  --combination-id qwen-managed-gpu
```

Manifests are written to `outputs/inference_deployments/<environment>/<candidate_id>.json`.

Required manifest fields:

- `candidate_id`, `artifact_digest`, `artifact_path`
- `base_model_id`, `served_model_name`, `combination_id`
- `endpoint_protocol`, `runtime_version`, `serving`
- `model_tier_aliases` for `TALISMAN_MODEL_LOW|MID|HIGH`

## GCP non-production provisioning

1. Copy and fill `infra/gcp/config.sh` from `config.example.sh`.
2. Create inference secrets:

```bash
./infra/gcp/setup-secrets.sh
```

3. Build and push the inference image:

```bash
docker build -f infra/gcp/Dockerfile.inference \
  -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REPO}/inference:${IMAGE_TAG} .
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REPO}/inference:${IMAGE_TAG}
```

4. Deploy the private GPU service:

```bash
CANDIDATE_ID=<approved_candidate_id> ./infra/gcp/deploy-inference-service.sh
```

The deploy script:

- validates registry lifecycle and artifact digest
- uploads the deployment manifest to `gs://${GCS_STATE_BUCKET}/inference/deployments/...`
- deploys a private Cloud Run GPU service (`--no-allow-unauthenticated`)
- records the OpenAI-compatible base URL for `TALISMAN_BASE_URL` rotation

## Startup and readiness

Governed startup check:

```bash
python -m decision_quality.inference_readiness startup-check \
  --deployment-manifest outputs/inference_deployments/nonprod/<candidate_id>.json \
  --registry data/agent_model_candidates/registry.json
```

Readiness assessment distinguishes:

- **healthy / not_ready** — governance passed, model not yet loaded
- **healthy / ready** — governance passed and served model alias is exposed
- **refused** — disabled, unapproved, or digest-mismatch candidate

vLLM serve command generation:

```bash
python -m decision_quality.inference_readiness serve \
  --deployment-manifest outputs/inference_deployments/nonprod/<candidate_id>.json \
  --registry data/agent_model_candidates/registry.json
```

`INFERENCE_ALLOW_SERVE=1` is required to start vLLM through the governed entrypoint.

## Application wiring

Set Talisman provider env vars to the private endpoint:

```bash
TALISMAN_BASE_URL=https://<inference-service-url>/v1
TALISMAN_API_KEY=<secret>
TALISMAN_MODEL_LOW=qwen2.5-7b-instruct
TALISMAN_MODEL_MID=qwen2.5-7b-instruct
TALISMAN_MODEL_HIGH=qwen2.5-7b-instruct
TALISMAN_TIMEOUT_S=120
```

GCP deploys bind `TALISMAN_BASE_URL` and `TALISMAN_API_KEY` from Secret Manager through `API_SECRETS` and `WORKER_SECRETS` in `infra/gcp/config.example.sh`.

Production routing, shadow mode, canary allocation, and frontier fallback are implemented by `TL-92`. See `docs/talisman_owned_model_rollout.md`. Keep `LLM_PROVIDER` on the frontier baseline until rollout controls are enabled and shadow burn-in completes.

## Contract smoke

Offline tests:

```bash
pytest tests/test_agent_inference_deployment.py tests/test_inference_readiness.py \
  tests/test_inference_deploy_scripts.py -q
```

Live non-production smoke (env-gated):

```bash
export TALISMAN_INFERENCE_SMOKE=1
export TALISMAN_BASE_URL=https://<inference-service-url>/v1
export TALISMAN_API_KEY=<secret>
export TALISMAN_MODEL_MID=qwen2.5-7b-instruct
pytest tests/test_inference_endpoint_smoke.py -q
```

## Capacity and release evidence

Before production rollout (`TL-92`), run TalismanBench against the managed endpoint:

```bash
export TALISMAN_BENCH_CANDIDATE_BASE_URL="${TALISMAN_BASE_URL}"
export TALISMAN_BENCH_CANDIDATE_API_KEY="${TALISMAN_API_KEY}"
export TALISMAN_BENCH_CANDIDATE_MODEL="${TALISMAN_MODEL_MID}"

python -m decision_quality.talisman_bench \
  --manifest docs/talisman_bench/manifest.json \
  --combination-id qwen-managed-gpu \
  --smoke-only
```

Record P50/P95 latency, throughput, saturation, error rate, and estimated cost in the release report under `outputs/talisman_bench/`.

## Monitoring

```bash
./infra/gcp/setup-inference-monitoring.sh
```

Log-based metrics and alerts cover:

- startup refusals from registry/digest gates
- request error spikes
- P95 generation latency saturation

## Rollback

1. Disable the candidate in the registry:

```bash
python -m decision_quality.agent_model_training disable --candidate-id <candidate_id>
```

2. Redeploy the prior manifest artifact or scale the inference service to zero.
3. Rotate `TALISMAN_BASE_URL` back to the prior secret version or remove Talisman secrets from API/worker deploys.
4. Verify readiness returns `refused` for the disabled candidate.

## Documentation impact

| Area | Update |
| --- | --- |
| ADR-010 | Managed-host capacity evidence and non-prod provisioning status |
| `docs/talisman_agent_model_training.md` | Link to inference deploy workflow |
| `docs/talisman_bench/README.md` | Managed endpoint validation protocol |
| `infra/gcp/README.md` | Inference deploy and monitoring scripts |
| `.env.example` | Production `TALISMAN_*` references |
| `TL-97` | Cross-cutting architecture guide input |

## Related docs

- Provider adapter: `talisman_openai_compat.py`
- Training/registry: `docs/talisman_agent_model_training.md`
- Release gate: `docs/talisman_bench/README.md`
- Base model/host ADR: `docs/adr/010-open-weight-base-model-and-inference-host.md`
