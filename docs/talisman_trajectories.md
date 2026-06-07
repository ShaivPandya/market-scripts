# Talisman Agent Trajectories

`TL-87` introduces a backend-only trajectory contract for complete Stan agent turns. The record is the source of truth for future training-data curation, replay, and production-learning analysis, but it does not add reviewer UI or feedback workflows.

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
