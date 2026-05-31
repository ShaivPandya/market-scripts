# Decision-Quality Chat Evals

These cases test Stan's live chat behavior for serious investment idea questions. They complement `docs/decision_quality_evals/`, which tests the structured `DecisionQuality` JSON contract directly.

The chat runner sends each case through `/api/agent/chat`, consumes SSE, records the final answer and tool trace, and then scores the answer with deterministic checks. An optional judge grades conversational quality after deterministic checks pass.

## Run

Offline tests use mocked providers and mocked tools:

```bash
pytest tests/test_decision_quality_chat_eval_runner.py tests/test_agent_decision_quality_chat.py
```

Workflow artifact and proposal boundary checks are deterministic and local:

```bash
pytest tests/test_decision_quality_chat_eval_runner.py tests/test_generated_review_approval_suppression.py
```

Manual model-backed runs:

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
- `workflow_expectations` is optional. Use it for workflow cases that must include a `workflow_run_id`,
  emit parseable fenced `artifacts` JSON with expected keys, and describe generated actions as proposals or pending approvals.
- LLM judge scores are secondary; deterministic checks are the pass/fail gate.

## Capturing Failures

Turn a bad real chat response into a draft case:

```bash
python -m decision_quality.capture_chat_eval --session-id SESSION --turn-index 0 --failure-tags generic,missing_invalidation
```

The exporter reads `memory_db`, redacts obvious secrets, and writes a draft case for human completion.
