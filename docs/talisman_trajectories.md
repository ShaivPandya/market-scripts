# Talisman Agent Trajectories

`TL-87` introduces the trajectory contract for complete Stan agent turns. `TL-88` adds explicit human feedback and labeling workflows on top of that contract without changing the SSE client protocol.

## Storage Contract

The canonical production table is `agent_trajectories`, created by `migrations/versions/20260607_0001_agent_trajectories.py`. Production uses Postgres through Alembic. Local and test runs use the SQLite initialization in `api/agent_trajectories.py` only when SQLite state is explicitly allowed.

Each row stores:

- `trajectory_id`, `schema_version`, `session_id`, `client_turn_id`, `captured_at`, and `completed_at`.
- `final_disposition`, `provider`, `model`, `prompt_version`, and optional `code_version`.
- Sensitivity, consent state, training eligibility, exclusion reasons, retention class, redaction policy, and dataset split group.
- `raw_payload_json` for restricted audit/replay use.
- `sanitized_payload_json` plus `redaction_manifest_json` for future training export.
- Provenance and source-provenance references, including agent turn event IDs when available.

Schema version `1` requires ordered step IDs. The current agent hook records route, model-call, tool-call, gate, and final steps from the existing SSE/provenance timing data without changing the client SSE contract.

## Trust Boundaries

The raw record is restricted operational data. It may contain message text, route metadata, gate output, tool status, and other turn evidence needed to audit or replay behavior. It is not a training dataset.

The sanitized training view is derived at insert time. It redacts provider secrets, token-like values, sensitive key names, and converts raw/tool-result/output payload fields to hash-only summaries. The sanitized view is the only payload returned by `export_sanitized_trajectories()`.

## Eligibility And Export

New agent-captured trajectories default to `training_eligible=false` because explicit training consent is not part of `TL-87`. Eligibility is still represented in the schema so `TL-88` and `TL-90` can promote reviewed records later without changing the trajectory contract.

Export rejects a row when:

- The schema version is unknown.
- The row is tombstoned.
- `training_eligible` is false.
- The redaction manifest is missing or does not match `agent_trajectory_training_v1`.
- The sanitized payload is missing messages or ordered steps.
- A second redaction pass still finds restricted fields.

Dataset split groups are deterministic from session and client turn identifiers, or from message hash when a session is unavailable. This keeps future train/eval leakage checks stable.

## Retention And Deletion

Raw trajectories use `agent_trajectory_365d`. Sanitized training views use `agent_training_view_365d`.

Deletion uses `tombstone_trajectory()`. The row remains as audit lineage, but `training_eligible` is set false and the sanitized view is updated so future exports exclude it. This avoids silently orphaning downstream references while satisfying the training-data deletion requirement.

## Backfill Guidance

Backfill must be conservative. Only records that can satisfy schema version `1`, ordered step construction, source provenance, redaction manifest generation, and deterministic export checks should be inserted. Legacy session transcripts without reliable tool/gate/model provenance should remain excluded rather than patched with guessed fields.

Backfilled rows should record their source in `source_provenance`, include explicit exclusion reasons when they are not exportable, and never mark training eligibility true unless reviewed consent and redaction checks are present.

## Operational Notes

Trajectory capture is best-effort and non-blocking inside `api/routers/agent.py`. Insert failures are logged as `agent_trajectory_capture_failed` and do not change the user-visible stream.

The current implementation does not change ADR-006 model egress assumptions. Model prompts still follow the existing gateway policy; trajectory export is a separate training-data boundary enforced after the turn completes.

## Human Feedback (`TL-88`)

Human-reviewed labels live in `agent_response_feedback`, created by `migrations/versions/20260607_0002_agent_response_feedback.py`. Local and test runs initialize the same contract in `api/agent_response_feedback.py`.

Each label stores:

- `feedback_id`, `trajectory_id`, `session_id`, `client_turn_id`, and immutable `response_version`.
- Reviewer actor, reviewed timestamp, decision (`approve`, `reject`, `correct`), optional corrected response, failure tags, and notes.
- `training_eligible` only when the reviewer explicitly opts in.
- `signal_source = human_reviewed`, distinct from inferred behavioral signals.

### API

Authenticated routes under `/api/agent/feedback`:

- `POST /api/agent/feedback` — create or update one label for the current reviewer and response version.
- `GET /api/agent/feedback` — list labels, fetch labels for one turn, or return the unlabeled review queue.
- `GET /api/agent/feedback/export` — export eligible human-reviewed labels for preference dataset builders.

Lookup accepts either `trajectory_id` or `session_id` + `client_turn_id`. Feedback is idempotent per `(trajectory_id, reviewer_actor_id, response_version)`.

### Promotion Rules

- New trajectories still default to `training_eligible=false`.
- `approve` plus explicit `eligible_for_training=true` can promote the parent trajectory through `promote_trajectory_for_training()`.
- `reject` and `correct` labels can be exportable for preference datasets, but they never auto-promote trajectories.
- Promotion clears `missing_training_consent`, refreshes the sanitized payload, and leaves all other export guards in place.

### Deletion And Export

`tombstone_trajectory()` cascades to feedback rows for the same trajectory. Tombstoned feedback is excluded from `export_human_reviewed_feedback()`.

Training exports remain conservative:

- Trajectory export still uses `export_sanitized_trajectories()`.
- Human-reviewed labels export separately through `export_human_reviewed_feedback()`.
- Only labels with `training_eligible=true` and `signal_source=human_reviewed` are returned.

### UI Disclosure

Completed Stan responses expose approve, reject, and correct actions in the agent chat UI. The interface explains that feedback is stored with the trajectory and model version and may be used for evaluation review or optional governed training datasets.
