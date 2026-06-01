# Decision Quality Evals

This directory is an authoring kit for investment decision-quality eval cases.

The goal is not to grade whether a trade later made money. The goal is to test whether the model followed a good investment process:

- clear thesis
- clear mispricing
- clear catalyst or reason-now
- concrete invalidation
- honest disconfirming evidence
- disciplined actionability
- calibrated confidence
- sizing/risk awareness when relevant

## Directory Layout

- `schema_draft.json` - draft JSON schema for one eval case.
- `case_template.json` - blank case template to copy for new cases.
- `rubric.md` - human grading rubric for process quality.
- `cases/` - draft and approved eval cases.

## Case Lifecycle

Use the `status` field to track authoring state:

- `draft` - input refs or gold output are incomplete.
- `review` - ready for human review.
- `approved` - stable enough to use in prompt/model comparisons.
- `archived` - no longer part of the active eval set.

Approved cases must also include regression metadata:

- `corpus_tags` - one or more of `structured_dq`, `chat_behavior`, `routing_tool_use`, `opportunity_identification`, `workflow_boundary`
- `failure_type` - normalized failure taxonomy when the case guards a known regression class
- `tool_pack` - optional routing/tool-pack label
- `required_dq_dimensions` - rubric dimensions this case is meant to protect

Workflow/proposal boundary cases stay `draft` or `review` until TL-46 lands, even if they already carry `workflow_boundary` tags.

## Authoring Workflow

1. Pick a historical decision or thesis.
2. Add the exact input refs the model should see.
3. Decide the user question the model must answer.
4. Fill the `gold_output` with the ideal structured decision object.
5. Score the gold output against `rubric.md`.
6. Add `corpus_tags`, `failure_type`, `tool_pack`, and `required_dq_dimensions`.
7. Mark the case `approved` only after the inputs, metadata, and expected output are stable.
8. Refresh the approved baseline snapshot after promotion.

For approved cases, prefer stable source snapshots or hashes so future repo edits do not silently change the eval.

### Promotion Checklist

Before moving a case from `review` to `approved`:

- input refs exist and SHA-256 hashes match the checked-in artifacts
- deterministic expectations are stable and documented in the case JSON
- `corpus_tags` and `required_dq_dimensions` are filled
- the case passes offline validation in `tests/test_decision_quality_model.py`
- the approved baseline under `baselines/approved_corpus_baseline.json` is updated intentionally

After a prompt, router, or model change:

1. Run the approved corpus only.
2. Compare against the committed baseline.
3. Treat new deterministic failures as release blockers; judge deltas are review-only.

## Running Evals

Use offline tests first. They validate that every case has a strict `DecisionQuality` gold output, clean input refs, and passing gate behavior without calling an LLM:

```bash
.venv/bin/python -m pytest tests/test_decision_quality_model.py tests/test_decision_quality_eval_runner.py
```

Run the approved regression corpus before and after prompt or model changes:

```bash
.venv/bin/python -m decision_quality.eval_runner --approved-only --no-judge
.venv/bin/python -m decision_quality.eval_runner --approved-only --judge --baseline docs/decision_quality_evals/baselines/approved_corpus_baseline.json
```

Filter by corpus tag when debugging a subset:

```bash
.venv/bin/python -m decision_quality.eval_runner --approved-only --corpus-tag structured_dq --no-judge
```

Refresh the committed baseline after an intentional prompt/model improvement:

```bash
.venv/bin/python -m decision_quality.eval_runner --approved-only --no-judge --update-baseline
```

Run the full judged model eval before and after prompt or model changes:

```bash
.venv/bin/python -m decision_quality.eval_runner --judge
```

Run one case when debugging a specific failure:

```bash
.venv/bin/python -m decision_quality.eval_runner --case nvda_ai_platform_long_2026 --no-judge
```

Inspect sanitized prompts without calling an LLM:

```bash
.venv/bin/python -m decision_quality.eval_runner --dry-run --no-judge
```

Reports are written to `outputs/decision_quality_evals/` unless `--output` is supplied. The runner excludes gold outputs, human notes, and future outcome context from solver prompts.

Interpret the report summary as follows:

- `deterministic_failures` - cases where schema parsing, gates, action, conviction, missing-input alignment, or required decision fields failed.
- `judge_failures` - cases where the optional judge score fell below threshold, leakage was detected, or the judge reported fatal issues.
- `leakage_detected` - whether the candidate appears to use future outcomes or answer-key facts not present in the sanitized inputs.
- Judge totals are scored `0-20`; use `18+` as strong, `14-17` as review, and below `14` as failed by default.

For a baseline workflow, run `--judge`, save the report path, make the prompt/model change, run `--judge` again, and compare failures plus judge-score deltas. The committed baseline lives in `baselines/approved_corpus_baseline.json`; `--baseline` writes a delta report and fails on new deterministic regressions.
