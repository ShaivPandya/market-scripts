# Ontology Architecture

This package owns Talisman's ontology boundary: typed operational and decision
objects, relations, source records, materialized risk snapshots, and the
action/tool metadata that lets workflows and agents interact with those objects
safely.

The runtime architecture is Postgres-backed bitemporal ontology. Snapshot rows
remain for explicit `run_id` inspection, but operational writes and semantic
reads use temporal objects, relations, source records, and read models.

## Boundary

`ontology/` is responsible for:

- Defining ontology node, edge, object, relation, and action schemas.
- Normalizing source payloads into versioned ontology object/relation schemas.
- Ingesting portfolio, market, macro, liquidity, sector, sentiment, positioning,
  thesis, and process data into ontology snapshots.
- Writing temporal object, relation, source record, and computed snapshot
  versions.
- Serving ontology reads through policy-aware service methods.
- Describing the safe tool/action surface exposed to agents and workflows.

`ontology/` is not responsible for:

- Order management, broker execution, fills, or counterparties.
- Raw market-data vendor clients outside the adapter outputs consumed here.
- UI presentation beyond shaping API/tool responses.
- Arbitrary direct mutation of portfolio/thesis/process state outside
  `OntologyCommandService`.
- OMS, broker orders, fills, broker execution, or counterparties. Decision
  records such as `TradeProposal` and `ExecutedAction` stop at governed
  decision support and do not model execution.

## Source-Of-Truth Stores

Current runtime state has one write authority: the temporal ontology.

| Domain | Current write authority | Ontology representation |
| --- | --- | --- |
| Portfolio and hedge positions | `OntologyCommandService` | Materialized as `Position`, `Asset`, `Sector`, and risk edges during ingestion. |
| Thesis status and evaluations | `OntologyCommandService` plus thesis content helpers | Materialized as `Thesis`, `Evaluation`, and optional thesis relations during ingestion. |
| Catalysts, kill conditions, thesis claims, action items, watch triggers, approvals, action runs, audit/provenance | `OntologyCommandService` and `OntologyObjectService` | Linked through action/provenance metadata and selected snapshot entities. |
| Source adapter records | `source_record_versions` through `TemporalOntologyRepository` when Postgres state is enabled | Authoritative temporal source record history for normalized adapter outputs. |
| Ontology objects and relations | `ontology_object_versions` and `ontology_relation_versions` through `OntologyObjectService` in the target architecture | Bitemporal object/relation versions with actor, approval, action, source, input hash, and provenance links. |
| Query snapshots | `ontology_runs`, `ontology_snapshot_nodes`, `ontology_snapshot_edges` through `OntologyRepository` | Materialized read model for semantic ontology queries and historical snapshot comparison. |

The runtime authority is `OntologyObjectService` backed by
`TemporalOntologyRepository`. Governed actions, approvals, proposals, and
workflow artifacts write directly to temporal ontology objects and relations.

## Module Map

- `models.py` defines the in-memory graph types:
  `OntologyNode`, `OntologyEdge`, and `InterpretedQuery`.
- `schemas/objects.py` defines canonical object payload schemas such as
  `Position`, `Asset`, `Signal`, `Thesis`, `Evaluation`, and `Catalyst`.
- `schemas/relations.py` defines allowed relation names, source/target types,
  cardinality, and required relation properties.
- `schemas/registry.py` normalizes nodes/edges, checks
  canonical IDs, and validates relation cardinality/core graph constraints.
- `schema_definitions.py` stores schema definitions and per-run schema bindings.
- `sources/` contains adapter wrappers that normalize raw module payloads into
  DTOs with status, quality, lineage, drift, and coverage metadata.
- `ingestion.py` runs adapters, computes risk components, builds the snapshot
  graph, writes temporal source/object/relation versions, saves a snapshot run,
  and prunes old runs.
- `repository.py` stores and queries live/snapshot graph rows. Runtime use is
  Postgres-backed through `PostgresStateConnection`; explicit SQLite `db_path`
  usage remains compatibility/test scaffolding for snapshot tables.
- `temporal_repository.py` is the authoritative Postgres repository for
  bitemporal object, relation, source record, and computed snapshot versions.
- `object_service.py` is the typed write/read boundary above the temporal
  repository.
- `read_model.py` refreshes and queries indexed temporal Postgres read models
  for semantic ontology query paths.
- `decision_writeback.py` records report outputs, workflow artifacts, approval
  proposals, executed actions, and exact object-version mutation references
  through `OntologyObjectService`.
- `service.py` is the policy-aware semantic query service used by API routes and
  agent tools.
- `action_registry.py` is the canonical domain action and agent tool metadata
  registry. Mutations are applied by `OntologyCommandService`.
- `policy.py` defines ontology actions, actors, object/edge resources,
  redaction, graph filtering, and the default admin/system policy.

## Snapshot Graph

Semantic ontology queries read indexed temporal Postgres read models. Snapshot
graph rows remain available for explicit `run_id` migration/debug queries.

The graph shape is:

- `Position -> references_asset -> Asset`
- `Asset -> belongs_to_sector -> Sector`
- `Position -> has_thesis -> Thesis`
- `Thesis -> evaluated_by -> Evaluation`
- `Thesis -> has_catalyst -> Catalyst`
- `MacroIndicator -> emits_signal -> Signal`
- `Sector -> affected_by -> MacroIndicator`
- `Position -> exposed_to_signal -> Signal`

Snapshot ingestion flow:

1. `OntologyQueryService.query()` decides whether to reuse the latest run or
   refresh. A run can be reused for up to `SNAPSHOT_REUSE_MAX_AGE` when required
   modules are healthy and positions exist.
2. `ingest_into_repository()` creates an ontology run provenance event and runs
   required adapters: `portfolio`, `market_breadth`, `top50_breadth`,
   `vix_term_structure`, `sector_metrics`, and `liquidity`.
3. Optional adapters add `sentiment`, `positioning_summary`, `economic_growth`,
   and `labor_market`. The deep adapter bucket is currently empty.
4. Adapter results become source status metadata and `source_record_versions`.
5. Ingestion builds graph nodes and edges, computes volatility, breadth, sector,
   and macro risk components, and attaches top signal evidence to each position.
6. `normalize_graph()` enforces canonical IDs, validates relation types, and
   records optional thesis warnings as partial source status.
7. `_write_temporal_graph_versions()` writes the normalized nodes and edges
   through `OntologyObjectService`.
8. Temporal read models are refreshed after successful temporal writes.
9. `OntologyRepository.save_snapshot()` writes `ontology_runs`,
   `ontology_snapshot_nodes`, `ontology_snapshot_edges`, schema bindings, audit,
   and provenance links.
10. Runs older than `SNAPSHOT_RETENTION_DAYS` are pruned.

`schema_mode="upgraded"` is the semantic-query mode. Repository helpers can
load stored snapshot payloads, but service-level semantic queries intentionally
require current schemas.

## Temporal Model

Temporal ontology state is Postgres-only and lives behind
`TemporalOntologyRepository`.

Authoritative tables:

- `ontology_object_versions`
- `ontology_relation_versions`
- `source_record_versions`
- `computed_snapshot_versions`

Each version has valid time and transaction time:

- `valid_from` / `valid_to` describe when the fact is true in the business
  domain.
- `tx_from` / `tx_to` describe when Talisman knew that version.
- `tx_to IS NULL` means the transaction version is current.
- Corrections close the current transaction version and insert a replacement
  linked by `supersedes_version_id`.

`OntologyObjectService` derives stable object/relation UIDs, normalizes payloads
against the schema registry, preserves actor/provenance/action/approval/source
metadata, and returns `_meta.temporal` envelopes.

The `/ontology/objects`, `/ontology/relations`, and `/ontology/source-records`
routes read temporal tables. `/ontology/query` returns
`mode="temporal_read_model"` and object-bearing rows include mandatory
`_meta.temporal` fields. Snapshot rows remain available for explicit `run_id`
debug/migration queries; `refresh_snapshot` is deprecated unless paired with
`run_id`.

The one-time ontology backfill utilities are cutover-only scaffolding. After
production verification gates pass, remove those utilities and their dedicated
tests; keep Alembic migrations and schema history permanently.

## Identity And Schemas

Canonical identity functions live in `schemas/identity.py`. Call those helpers
instead of constructing IDs ad hoc when adding new object writers.

Important conventions:

- Position IDs are ticker-scoped, for example `position:MU`.
- Asset IDs are ticker-scoped, for example `asset:MU`.
- Sector IDs are slugged sector names.
- Thesis, evaluation, catalyst, macro indicator, and signal IDs are derived from
  stable business keys.
- Object schema names generally match object types, versioned from `1`.
- Relation schema names match relation types, versioned from `1`.
- Edge property schema names are `Relation` or `PositionSignalExposure`.
- Snapshot payloads use the current object schema contract at read/write
  boundaries.
- Decision schemas include `Recommendation`, `Scenario`, `TradeProposal`,
  `Approval`, `ActionRun`, `ExecutedAction`, `AuditEvent`, `SourceRecord`,
  `ObjectVersionRef`, `RiskMetric`, `InvestmentPolicy`, and
  `PolicyGateResult`.

When adding a node or relation type, update the Pydantic schema, identity
expectations, relation registry, schema definitions, ingestion/query logic, and
tests together. Relation changes should include source type, target type,
cardinality, required properties, and whether the relation is optional.

## Action Lifecycle

`action_registry.py` owns two related registries:

- `DomainAction`: typed validation, authorization, approval, output, and effect
  metadata for state-changing operations.
- `ToolExposure`: the agent/workflow-safe callable surface derived from action
  bindings, input schemas, access modes, aliases, and ontology policy specs.

Runtime mutation flow is ontology-primary:

1. APIs, agent proposal tools, reports, and workflow artifacts call
   `OntologyCommandService`.
2. Proposal paths write `Approval`, `PolicyGateResult` where applicable,
   provenance, and audit objects through `OntologyObjectService`.
3. Approval application writes new temporal object versions plus
   `ActionRun`/`ExecutedDecisionRecord` lineage.
4. `action_registry` exposes action/tool metadata; mutation execution is
   fail-closed and delegated to `OntologyCommandService`.

## Decision Writeback

Report sync, workflow artifacts, and approval application should use
`decision_writeback.py` or `OntologyObjectService` for ontology-backed decision
artifacts.

Safe automatic records:

- `ReportRun`, `WorkflowRun`, raw `WorkflowArtifact`, `SourceRecord`,
  `RiskMetric`, `Scenario`, `PolicyGateResult`, `AuditEvent`, and `ActionEvent`.
- Generated recommendations in non-applied states such as `generated` or
  closed audit states.

Approval-backed records:

- Financial recommendation actions: `buy`, `sell`, `reduce`, `exit`,
  `rebalance`, and `hedge`.
- Every `TradeProposal` before it is represented by an `ExecutedAction`.
- Workflow/report artifacts that create or alter user-visible research,
  process, portfolio, thesis, watch, action item, or note state.

The target decision lineage is:

`ReportRun/WorkflowRun -> SourceRecord/WorkflowArtifact -> Recommendation -> RiskMetric/Scenario/PolicyGateResult/InvestmentPolicy -> TradeProposal -> Approval -> ActionRun -> ExecutedAction -> ObjectVersionRef -> AuditEvent`.

Cutover runtime invariants:

- Governed writes use ontology objects/relations directly.
- Ontology writeback failures fail the caller.
- Query routes use temporal read models by default.

## Agent Safety Model

Agents do not receive arbitrary write access to the backing stores.

- Agent-visible tools are described by `ToolExposure` and can be listed with
  `iter_tool_exposures(agent_exposed_only=True)`.
- Tool access modes are `read`, `compute`, `proposal`, and `execute`. Proposal
  tools create pending approvals instead of applying mutations.
- Proposal tools bind to domain actions through `_PROPOSAL_TOOL_BINDINGS`, so
  tool input is adapted into the same typed action schemas used by direct
  execution.
- `query_ontology` is a read tool with dynamic policy checks. Requesting
  `include_graph` requires `graph.read`; requesting `refresh_snapshot` requires
  `snapshot.refresh`.
- `DefaultOntologyPolicy` currently allows internal system actors and actors
  with the `admin` role. Other actors are denied and get no fields.
- Query responses and graphs are filtered through `filter_graph()`,
  `redact_properties()`, object checks, relation checks, and authorization stats.
- Async ontology jobs carry the serialized actor in the job payload; job reads
  are limited to the same actor, admin, or system.

The safe default for new agent capabilities is a proposal tool with an approval
spec. Only add direct execution when the action is explicitly allowed for the
intended actor types and has audit, validation, rollback/postcondition behavior,
and tests.

## Provenance And Audit

Ontology ingestion, snapshot saves, semantic reads, action execution, proposals,
and workflow artifacts all emit audit/provenance records where possible.

Common links:

- Ontology run provenance links source adapter runs, source records, snapshot
  nodes/edges, and temporal object/relation versions.
- Action run provenance links the domain action, source workflow/agent/user,
  approval, resulting entities, and audit events.
- Source records store redacted payload summaries and stable payload hashes; raw
  secret values should not be persisted in source records or provenance summaries.

Audit failures are intentionally best-effort around the business operation, but
action run status should still be completed on validation, authorization, and
runtime errors.

## Query Surfaces

API routes in `api/routers/ontology.py` expose:

- `GET /ontology/runs`
- `GET /ontology/runs/{run_id}`
- `POST /ontology/query`
- `POST /ontology/query/async`
- `GET /ontology/query/async/{job_id}`
- `GET /ontology/objects`
- `GET /ontology/objects/{object_uid}`
- `GET /ontology/relations`
- `GET /ontology/source-records`

Agent tools call the same service path through `api/agent_tools.py`. `query()`
supports pagination, filters, optional graph output, explicit snapshot run IDs,
and snapshot comparison intent. `compare_snapshots()` diffs two materialized
runs by position set, risk score changes, signal transitions, and component
scores.

## Change Guidelines

When touching this package:

- Decide whether the change belongs to the temporal authority, the snapshot
  compatibility graph, or the agent/action safety layer before editing.
- Do not make non-ontology stores and temporal tables both authoritative for the
  same fact.
- Use `OntologyObjectService` for new temporal object/relation writes.
- Use `OntologyRepository` only for compatibility graph snapshots and snapshot
  queries.
- Normalize with `schemas/registry.py` before persisting graph nodes/edges.
- Add schema-definition coverage when introducing or versioning schemas.
- Preserve actor, approval, action run, input hash, source record, and provenance
  fields on any write path.
- Keep agent mutation paths proposal-first unless there is a specific, reviewed
  reason for direct execution.
- Add or update tests in `tests/test_ontology_repository.py`,
  `tests/test_ontology_source_adapters.py`, and action/API tests when changing
  snapshot shape, schemas, policy, or tool exposure behavior.
