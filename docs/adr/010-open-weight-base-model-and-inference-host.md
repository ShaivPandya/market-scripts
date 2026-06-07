# ADR-010: Initial Open-Weight Base Model and Inference Host

**Status:** Proposed
**Owner:** Shaiv Pandya
**Date:** 2026-06-07
**Revisit trigger:** A newer open-weight generation beats the smoke suite on tool-calling and structured-output gates; latency P95 on chat corpus regresses more than 15%; license or data-handling constraints change; or managed-host cost exceeds the documented ceiling.

## Context

`TL-84` requires Talisman to own a first-party agent model without permanently coupling application architecture to one base model or inference host. `TL-85` must select the initial open-weight base model and serving stack through reproducible evidence while preserving the provider-agnostic contract defined by the parent program.

TalismanBench (`TL-89`) now provides the release gate. This ADR records the initial candidate matrix, benchmark protocol, provisional selection, and rejected alternatives before production provider wiring (`TL-86`) and inference provisioning (`TL-95`).

## Decision

**Provisional primary:** `Qwen2.5 7B Instruct` served through **local vLLM** (`qwen-local-vllm` in `docs/talisman_bench/candidate_matrix.json`).

**Provisional fallback:** `Llama 3.1 8B Instruct` on the same local vLLM host (`llama-local-vllm`).

This selection is provisional until full TalismanBench evidence runs are recorded in `outputs/talisman_bench/`. The smoke suite already references one structured, one chat/tool-routing, and one opportunity-candidate case per combination.

### Why this pair

- Qwen2.5 7B offers strong tool-calling and JSON-schema support for representative Talisman workloads.
- Local vLLM keeps benchmark runs reproducible, avoids premature managed-host coupling, and satisfies the second hosting approach requirement alongside the managed GPU cloud combination in the matrix.
- Llama 3.1 8B remains the fallback because of its long-context headroom and mature ecosystem, at the cost of higher VRAM and license review overhead.

## Alternatives Considered

| Alternative | Pros | Cons |
|-------------|------|------|
| Qwen2.5 7B + local vLLM (selected primary) | Strong tool/schema behavior, reproducible local benchmarks, Apache-2.0 license | Requires local GPU access for development |
| Llama 3.1 8B + local vLLM (selected fallback) | Long context, strong ecosystem | Higher VRAM, Llama license review, slower smoke iteration |
| Mistral 7B Instruct v0.3 + local vLLM | Lower VRAM, fast iteration | Weaker strict JSON-schema reliability in early smoke observations |
| Qwen2.5 7B + managed GPU cloud | Production-like autoscaling and capacity measurement | Higher operational burden and cost before TL-95 provisioning |
| Frontier-only baseline without owned model | No GPU operations | Fails TL-84 objective and blocks training/rollout phases |

## Benchmark protocol

Evidence is generated through TalismanBench using:

- fixed temperature `0.0` and `max_tokens=4096`
- frontier baseline `openai` / `mid`
- OpenAI-compatible candidate endpoints via `decision_quality/bench_openai_client.py`
- smoke subset (`--smoke-only`) and full 43-case inventory
- hard blockers for deterministic failures and baseline regressions
- scored metrics for latency P95, token totals, and estimated cost

Chat/tool coverage uses benchmark-only agent mode (`TALISMAN_BENCH_AGENT_MODE=1`) so candidate endpoints are exercised without production provider changes.

## Capacity and cost assumptions

| Assumption | Value |
| --- | --- |
| Development GPU | NVIDIA L4 or RTX 4090 class, 24 GB VRAM |
| Serving runtime | vLLM with `max_model_len` 32k for Qwen |
| Local benchmark cost | $0 direct inference cost during development |
| Managed GPU placeholder | $0.0002 / 1k input tokens, $0.0004 / 1k output tokens for capacity planning only |

## Licensing and data handling

- **Qwen2.5 7B** and **Mistral 7B**: Apache-2.0; suitable for self-hosted development benchmarks.
- **Llama 3.1 8B**: Meta Llama 3.1 license; acceptable for evaluation and fallback, but requires explicit license review before production deployment.
- Self-hosted serving keeps Talisman prompts and trajectories off third-party training paths during benchmark and early rollout phases.

## Risks

- Open-weight tool-calling behavior may diverge from frontier baselines on chat corpora until SFT/LoRA candidates exist.
- Strict JSON-schema mode may fail on weaker candidates; smoke runs must record these failures explicitly.
- Local vLLM performance is not representative of managed production capacity without TL-95 provisioning evidence.
- Provisional selection must be revalidated after every major base-model generation change.

## References

- [candidate_matrix.json](../talisman_bench/candidate_matrix.json)
- [manifest.json](../talisman_bench/manifest.json)
- [TalismanBench README](../talisman_bench/README.md)
- [bench_openai_client.py](../../decision_quality/bench_openai_client.py)
- [talisman_bench.py](../../decision_quality/talisman_bench.py)
- Linear `TL-84`, `TL-85`, `TL-86`, `TL-89`, `TL-95`, `TL-97`
