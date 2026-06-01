# OpportunityCandidate Evals

This directory is an authoring kit for OpportunityCandidate triage eval cases.

The goal is to test whether the model follows good opportunity-triage process:

- clear trigger and why-now
- honest missing-input surfacing
- disciplined non-actionable next_action values
- correct graduation versus research/watch/avoid behavior

## Directory Layout

- `case_template.json` - blank case template to copy for new cases.
- `rubric.md` - human grading rubric for triage quality.
- `cases/` - draft, review, and approved eval cases.
- `baselines/approved_corpus_baseline.json` - committed approved-corpus regression snapshot.

## Case Lifecycle

Use the `status` field to track authoring state:

- `draft` - input refs or gold output are incomplete.
- `review` - ready for human review.
- `approved` - stable enough to use in prompt/model comparisons.
- `archived` - no longer part of the active eval set.

Approved cases must also include regression metadata:

- `corpus_tags` - include `opportunity_identification` for this corpus
- `failure_type` - normalized failure taxonomy when the case guards a known regression class
- `failure_tags` - richer active-learning tags
- `required_oc_dimensions` - rubric dimensions this case is meant to protect
- `expected_graduation`, `expected_final_action`, and `expected_gate_status`

Optional scout/skeptic expectation fields:

- `expected_scout_status`
- `expected_skeptic_status`
- `expected_skeptic_block_reasons`

## Authoring Workflow

1. Pick a triage scenario: graduation, research/watch, avoid, proactive scan, or scout/skeptic block.
2. Add the exact input refs the model should see.
3. Decide the user question the model must answer.
4. Fill the `gold_output` with the ideal structured OpportunityCandidate object.
5. Score the gold output against `rubric.md`.
6. Add corpus metadata and deterministic expectations.
7. Mark the case `approved` only after inputs, metadata, and expected output are stable.
8. Refresh the approved baseline snapshot after promotion.

### Promotion Checklist

Before moving a case from `review` to `approved`:

- deterministic expectations are stable and documented in the case JSON
- `corpus_tags`, `failure_type`, and `required_oc_dimensions` are filled
- the case passes offline validation in `tests/test_opportunity_candidate_model.py`
- the case passes dry-run validation in `tests/test_opportunity_candidate_eval_runner.py`
- the approved baseline under `baselines/approved_corpus_baseline.json` is updated intentionally

After a prompt, router, or model change:

1. Run the approved corpus only.
2. Compare against the committed baseline.
3. Treat new deterministic failures as release blockers; judge deltas are review-only.

## Running Evals

### CI gate (offline, no LLM)

The `eval-gates` job in `.github/workflows/ci.yml` runs the approved-corpus offline checks on every PR. To reproduce locally:

```bash
pytest tests/test_opportunity_candidate_model.py \
  tests/test_opportunity_candidate_eval_runner.py \
  -q --tb=short

python -m decision_quality.opportunity_candidate_eval_runner --approved-only --dry-run --no-judge
```

This gate validates gold outputs, approved metadata, gold-at-rest deterministic expectations, and committed baseline inventory sync. It does **not** call an LLM or compare live model output against baselines.

### Manual release check (LLM-backed)

Run the approved regression corpus before and after prompt or model changes (requires LLM API keys; not run in PR CI):

```bash
python -m decision_quality.opportunity_candidate_eval_runner --approved-only --no-judge
python -m decision_quality.opportunity_candidate_eval_runner --approved-only --judge --baseline docs/opportunity_candidate_evals/baselines/approved_corpus_baseline.json
```

Filter by corpus tag when debugging a subset:

```bash
python -m decision_quality.opportunity_candidate_eval_runner --approved-only --corpus-tag opportunity_identification --no-judge
```

Refresh the committed baseline after an intentional prompt/model improvement (local only; requires LLM runs):

```bash
python -m decision_quality.opportunity_candidate_eval_runner --approved-only --no-judge --update-baseline
```

Run one case:

```bash
python -m decision_quality.opportunity_candidate_eval_runner --case opportunity_candidate_graduate_nvda_2026 --no-judge
```

Dry-run all approved cases:

```bash
python -m decision_quality.opportunity_candidate_eval_runner --approved-only --dry-run --no-judge
```
