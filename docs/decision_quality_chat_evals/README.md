# Decision-Quality Chat Evals

These cases test Stan's live chat behavior for serious investment idea questions. They complement `docs/decision_quality_evals/`, which tests the structured `DecisionQuality` JSON contract directly.

The chat runner sends each case through `/api/agent/chat`, consumes SSE, records the final answer and tool trace, and then scores the answer with deterministic checks. An optional judge grades conversational quality after deterministic checks pass.

## Run

### CI gate (offline, no LLM)

The `eval-gates` job in `.github/workflows/ci.yml` runs the approved-corpus offline checks on every PR. To reproduce locally:

```bash
pytest tests/test_decision_quality_model.py \
  tests/test_decision_quality_eval_runner.py \
  tests/test_decision_quality_eval_corpus.py \
  tests/test_decision_quality_chat_eval_runner.py \
  tests/test_agent_decision_quality_chat.py \
  tests/test_opportunity_candidate_model.py \
  -q --tb=short

python -m decision_quality.eval_runner --approved-only --dry-run --no-judge
python -m decision_quality.chat_eval_runner --approved-only --dry-run
```

This gate validates gold outputs, input-ref hashes, committed baseline inventory sync, and prompt/input leakage checks. It does **not** call an LLM or compare live model output against baselines.

### Offline tests

Offline tests use mocked providers and mocked tools:

```bash
pytest tests/test_decision_quality_chat_eval_runner.py tests/test_agent_decision_quality_chat.py
```

Workflow artifact and proposal boundary checks are deterministic and local:

```bash
pytest tests/test_decision_quality_chat_eval_runner.py tests/test_generated_review_approval_suppression.py
```

### Manual release check (LLM-backed)

Manual model-backed runs (requires LLM API keys; not run in PR CI):

```bash
python -m decision_quality.chat_eval_runner --judge
```

The live chat hidden pass is enabled by default and can be disabled with:

```bash
AGENT_DECISION_QUALITY_CHAT_ENABLED=false
```

Before changing `auto_report/prompts/agent_system.md`, `auto_report/prompts/decision_quality.md`, model routing, or `auto_report/prompts/decision_quality_chat_synthesis.md`, run both:

```bash
python -m decision_quality.eval_runner --judge
python -m decision_quality.chat_eval_runner --judge
```

Reports are written to `outputs/decision_quality_chat_evals/`.

## Case Rules

- Cases are as-of tests. Put only information available on `as_of_date` in `inputs/`.
- `input_refs` must include SHA-256 hashes for every local input artifact.
- `mock_tools` should make replay deterministic where possible.
- `required_points` should be concrete claims or risks that a good Stan answer must surface.
- `forbidden_patterns` should catch lazy generic phrasing.
- `corpus_tags`, `failure_type`, `tool_pack`, and `required_dq_dimensions` are required before promotion to `approved`.
- `workflow_expectations` is optional. Use it for workflow cases that must include a `workflow_run_id`,
  emit parseable fenced `artifacts` JSON with expected keys, and describe generated actions as proposals or pending approvals.
- Workflow/proposal boundary cases remain `draft` or `review` until TL-46 lands.
- LLM judge scores are secondary; deterministic checks are the pass/fail gate.

## Promotion Workflow

1. Capture or author a case as `draft`.
2. Fill `mock_tools`, `required_points`, routing/tool expectations, and hashed `input_refs`.
3. Add regression metadata: `corpus_tags`, `failure_type`, `tool_pack`, `required_dq_dimensions`.
4. Move to `review` and run offline chat eval tests.
5. Promote to `approved` once deterministic expectations are stable.
6. Refresh `baselines/approved_corpus_baseline.json` with `--update-baseline`.

Example approved-corpus commands (manual release check; requires LLM API keys):

```bash
python -m decision_quality.chat_eval_runner --approved-only --dry-run
python -m decision_quality.chat_eval_runner --approved-only --baseline docs/decision_quality_chat_evals/baselines/approved_corpus_baseline.json
python -m decision_quality.chat_eval_runner --approved-only --corpus-tag routing_tool_use --no-judge
python -m decision_quality.chat_eval_runner --approved-only --update-baseline
```

## Active-Learning Failure Loop (TL-65)

Captured chat failures move through a file-backed review queue in `cases/`:

1. **Capture** a real failure as `draft` with standardized `failure_tags`.
2. **Review** by filling `mock_tools`, `required_points`, hashed `input_refs`, and `routing_expectations`.
3. **Promote** to `approved` once deterministic checks are stable.
4. **Export** reviewed/approved router-labeled rows for supervised training.
5. **Refresh** the approved corpus baseline after prompt/model/router changes.

Training export gate: rows only leave `review` or `approved` cases with human-reviewed
`routing_expectations.intent_class`. Draft captures are never exported.

### Failure tags

Tags are grouped in `decision_quality.eval_corpus.FAILURE_TAG_CATEGORIES`:

- `routing`: `wrong_routing`, `wrong_tools`
- `hidden_dq`: `missed_hidden_dq`, `missing_invalidation`, `missing_mispricing`, `missing_catalyst`, `generic_answer`
- `source_quality`: `source_freshness`, `price_confirmation`, `stale_data`
- `opportunity_identification`: `missing_mispricing`, `weak_opportunity_id`
- `synthesis_quality`: `generic_answer`, `bad_synthesis`, `process_regression`
- `policy_action_gating`: `sizing_discipline`, `workflow_boundary_violation`, `overconfident_actionability`

Common aliases: `generic` → `generic_answer`, `stale_data` → `source_freshness`.

### Capturing failures

Turn a bad real chat response into a draft case:

```bash
python -m decision_quality.capture_chat_eval \
  --session-id SESSION \
  --turn-index 0 \
  --failure-tags generic,missing_invalidation
```

The exporter reads `memory_db`, redacts obvious secrets, seeds `observed_tool_calls` into
`expected_tool_names`, and writes a draft case for human completion.

### Router training export

Export reviewed/approved chat eval cases with router labels and failure metadata:

```bash
python -m decision_quality.intent_router_training export --active-learning --no-db
```

Output lands in `outputs/intent_router_training/{version}/active_learning_router.jsonl` with fields such as
`failure_tags`, `failure_type`, `corpus_tags`, `source_session_id`, and `eval_status`.

Standard router fixture export (telemetry + `routing_*` fixtures) remains:

```bash
python -m decision_quality.intent_router_training export
```
