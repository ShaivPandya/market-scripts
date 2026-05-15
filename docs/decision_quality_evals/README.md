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

## Authoring Workflow

1. Pick a historical decision or thesis.
2. Add the exact input refs the model should see.
3. Decide the user question the model must answer.
4. Fill the `gold_output` with the ideal structured decision object.
5. Score the gold output against `rubric.md`.
6. Mark the case `approved` only after the inputs and expected output are stable.

For approved cases, prefer stable source snapshots or hashes so future repo edits do not silently change the eval.

## Running Evals

Offline tests validate that every case has a strict `DecisionQuality` gold output, clean input refs, and passing gate behavior:

```bash
.venv/bin/python -m pytest tests/test_decision_quality_model.py tests/test_decision_quality_eval_runner.py
```

Manual model evals run the solver against sanitized case inputs and write a report under `outputs/decision_quality_evals/`:

```bash
.venv/bin/python -m decision_quality.eval_runner --judge
```

Use `--dry-run --no-judge` to inspect sanitized prompts without calling an LLM. The runner excludes gold outputs, human notes, and future outcome context from solver prompts.
