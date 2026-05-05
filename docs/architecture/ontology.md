# Authoritative Bitemporal Ontology Boundary

## Summary

The ontology is the authoritative operational object model for current Talisman domains. Portfolio, thesis, process, source, decision, and derived snapshot state is represented as temporal ontology object and relation versions in Postgres.

This is a breaking Postgres-only architecture. SQLite-backed state paths are legacy migration inputs, not canonical stores for the temporal ontology.

## Ownership

`ontology/object_service.py` is the write boundary for operational objects and relations. Domain routes, workflow handlers, approval application, ingestion, and agent tools should call the object service instead of mutating `portfolio_db.py`, `thesis_db.py`, or `core_db.py` as canonical state.

The supported domain scope is the existing Talisman domain model:

- positions and hedge positions
- theses, thesis status, evaluations, catalysts, kill conditions, and thesis claims
- workflow runs, report runs, workflow artifacts, action items, watch triggers, research notes, recommendations, scenarios, risk metrics, policy gate results, trade proposals, action runs, executed actions, audit events, and action events
- approvals and approval-linked workflow/report artifacts
- source records and computed snapshots that support those objects

Orders, broker orders, fills, OMS objects, broker execution, and counterparties are intentionally out of scope. `TradeProposal` and `ExecutedAction` are decision-support governance records only; they do not represent broker execution or fills.

## Temporal Model

Every authoritative row has valid-time and transaction-time semantics.

- `valid_from` and `valid_to` describe when the represented fact is true in the business domain.
- `tx_from` and `tx_to` describe when Talisman knew or believed that version.
- `tx_to IS NULL` means the version is current in transaction time.
- A current read means `tx_to IS NULL` and the valid interval contains the requested `as_of`, defaulting to now.
- A historical read with `tx_as_of` returns what Talisman knew at that transaction time.

Corrections must not mutate closed audit rows. They close the affected transaction version and insert a replacement version with `supersedes_version_id`.

## Tables

The authoritative tables are:

- `ontology_object_versions`
- `ontology_relation_versions`
- `source_record_versions`
- `computed_snapshot_versions`

Legacy ontology runs and snapshot tables may exist during migration, but query and mutation paths should move to temporal object/relation/source/snapshot tables. Old snapshot tables are migration input and compatibility scaffolding only.

Postgres `btree_gist` is required for exclusion constraints that reject overlapping current valid intervals for the same object or relation UID.

## Source Records

Source adapters write `source_record_versions` before object materialization. Each record carries vendor/source metadata, as-of/load-time metadata, a record key hash, payload hash, status, quality, provenance event ID, and bitemporal bounds.

Refreshes are idempotent when the normalized payload and temporal metadata are unchanged. Late-arriving records can have old valid time and new transaction time.

Raw secret values must never be persisted in payloads or provenance summaries. Existing redaction and audit summary rules still apply.

## Decision Objects

Decision-support artifacts are first-class ontology objects:

- `Recommendation`, `Scenario`, `RiskMetric`, `PolicyGateResult`, `InvestmentPolicy`, and `SourceRecord` capture generated analysis, constraints, and evidence.
- `TradeProposal` captures a staged or approval-pending financial decision derived from a recommendation. It is not an order.
- `Approval` records proposal resolution separately from application state.
- `ActionRun` records attempted application, and `ExecutedAction` records the immutable post-approval summary of applied ontology/domain mutations.
- `ObjectVersionRef` points to exact temporal object versions produced or mutated by a governed action.
- `AuditEvent` records redacted, retention-classed observations over decision and action activity.

Recommended lifecycle values are:

- `Recommendation.decision_state`: `generated`, `proposed`, `under_review`, `approved`, `rejected`, `acted`, `closed`, `superseded`.
- `TradeProposal.decision_state`: `staged`, `policy_checked`, `pending_approval`, `approved`, `rejected`, `executed_action_recorded`, `expired`, `superseded`.
- `Approval.resolution_state`: `pending`, `approved`, `rejected`, `expired`.
- `Approval.application_state`: `pending`, `applying`, `applied`, `failed`, `not_applicable`.
- `ActionRun.execution_state`: `running`, `succeeded`, `failed`, `rolled_back`, `denied`.
- `WorkflowArtifact.state`: `extracted`, `ignored`, `auto_recorded`, `proposed`, `approved`, `rejected`, `failed`.

The expected lineage path is:

`ReportRun/WorkflowRun -> SourceRecord/WorkflowArtifact -> Recommendation -> RiskMetric/Scenario/PolicyGateResult/InvestmentPolicy -> TradeProposal -> Approval -> ActionRun -> ExecutedAction -> ObjectVersionRef -> AuditEvent`.

## Object And Relation Writes

Object writes provide:

- object type
- business key
- normalized properties
- valid interval
- actor/provenance metadata
- optional source record, action run, approval, and input hash links

Relation writes provide:

- source object UID
- target object UID
- relation type
- normalized relation properties
- valid interval
- actor/provenance metadata

The object service derives stable object and relation UIDs, validates schema metadata, preserves provenance links, and creates new transaction versions when facts change.

Agent proposals cannot directly mutate authoritative objects. They create workflow artifacts or pending approvals. Approval application writes through the object service and links approval/action provenance to the produced version rows.

`ontology/decision_writeback.py` is the migration facade for report, workflow artifact, and approval-application writeback. It records safe automatic artifacts directly (`ReportRun`, `WorkflowRun`, raw `WorkflowArtifact`, `SourceRecord`, `RiskMetric`, `Scenario`, `PolicyGateResult`, `AuditEvent`, `ActionEvent`, and non-applied generated recommendations) and creates approval-linked decision artifacts for governed mutations.

Approval is required for financial recommendation actions (`buy`, `sell`, `reduce`, `exit`, `rebalance`, `hedge`), every `TradeProposal`, and workflow/report artifacts that would create or alter user-visible research, process, portfolio, or thesis state.

## Reads

Ontology and business read routes accept:

- `as_of`
- `tx_as_of`
- `include_history`

Object-bearing responses should include temporal metadata under `_meta.temporal`, including object UID, version ID, valid interval, transaction interval, and temporal confidence.

Default reads return the current transaction/current valid-time view. Historical reads must use object, relation, source record, and computed snapshot versions consistently.

## Migration And Cutover

The migration sequence is:

1. Enable Postgres migrations, including `btree_gist`.
2. Create the temporal ontology tables and constraints.
3. Freeze writes with `WRITE_FREEZE=true`.
4. Run the one-time operational and temporal backfills from legacy stores and ontology snapshots into temporal versions.
5. Mark reconstructed rows with `temporal_confidence='backfilled'`.
6. Switch routes, workflow handlers, action application, ingestion, and agent tools to object-service writes.
7. Update UI surfaces to show or carry temporal metadata where relevant.
8. Remove or hard-disable legacy SQLite-backed state paths after tests pass.
9. After production verification gates pass, delete the one-time ontology backfill utilities and their dedicated tests. Preserve Alembic migrations and schema history permanently.

Backfilled historical precision is limited by timestamps already present in the legacy data. Unknown or reconstructed timestamps should use the cutover time and explicit backfilled confidence.
