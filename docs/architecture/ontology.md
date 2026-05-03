# Ontology Materialization Boundary

## Summary

The `ontology/` package provides a materialized semantic/risk graph for portfolio risk analysis, agent tools, and UI surfaces. It links portfolio positions to assets, sectors, theses, evaluations, catalysts, macro indicators, and market signals so the app can query risk exposure with evidence.

The ontology graph is not the canonical operational object store today. It is a typed snapshot built from canonical backing stores and market data modules, then persisted as ontology runs for read/query workflows.

## Canonical Stores

Operational state is owned by backing stores outside the ontology graph:

- `portfolio/portfolio_db.py` is the source of truth for editable portfolio positions and hedge positions.
- `portfolio/thesis_db.py` owns thesis metadata, thesis status history, and thesis evaluations. Thesis markdown content is stored through the thesis content/state storage path.
- `portfolio/core_db.py` owns Investing OS process state: catalysts, kill conditions, thesis claims, workflow runs, action items, watch triggers, research notes, pending approvals, action runs/events, recommendations, report runs, and audit events.
- Market, macro, sector, liquidity, positioning, sentiment, and technical modules own their own computed source payloads. Ontology ingestion reads their outputs through source adapters.

These stores define the operational schemas and write semantics. Ontology schemas validate graph snapshots; they do not replace the backing-store schemas.

## Materialization Flow

Ontology snapshots are built on demand by `ontology/ingestion.py`.

1. `OntologyQueryService` resolves whether to reuse a recent run, read a requested historical `run_id`, or refresh a snapshot.
2. `ontology.sources.registry.build_adapter_registry()` selects required and optional source adapters.
3. Adapters normalize source module outputs into DTOs and source status metadata.
4. `ingest_into_repository()` constructs typed `OntologyNode` and `OntologyEdge` objects for positions, assets, sectors, theses, evaluations, catalysts, macro indicators, signals, and position-signal exposures.
5. `ontology/schemas/*` normalizes and validates graph object identities, relation types, relation cardinality, and edge properties.
6. `OntologyRepository.save_snapshot()` persists the run in `ontology_runs`, `snapshot_nodes`, `snapshot_edges`, and schema binding tables.
7. Old runs are pruned by retention policy.

The snapshot tables are optimized for historical replay, comparison, and query joins. They are expected to lag backing-store writes until a new snapshot is built or an existing run is explicitly selected.

## Writes And Approvals

Application writes do not mutate ontology graph nodes as authoritative state.

- Direct user routes execute typed domain actions against the backing stores. For example, portfolio position edits go through `update_portfolio_positions`, and thesis status changes go through `change_thesis_status`.
- Agent and workflow proposal tools create pending approvals in `portfolio/core_db.py`.
- Approval resolution executes the registered domain action with `approval_apply` context, then mutates the corresponding backing store.
- Refreshed ontology snapshots later materialize the updated backing-store state into graph nodes and edges.

This keeps the ontology graph read-oriented for risk semantics while backing stores continue to own operational consistency, validation, audit records, and action side effects.

## Read Paths

The main read path is:

1. Agent tools, `/api/v1/ontology/*` routes, Ontology Workbench, and the Position Dossier risk tab call `OntologyQueryService`.
2. The service parses the query, enforces ontology read policy, chooses or refreshes a snapshot run, and queries `OntologyRepository`.
3. Repository methods read `snapshot_*` tables to return position risk rows, graph fragments, evidence drivers, aggregate buckets, and snapshot comparisons.

Agent tools such as `query_ontology` and `get_ontology_diff` should be treated as read/query surfaces over materialized graph snapshots, not as CRUD APIs for portfolio, thesis, or process entities.

## What The Graph Is Not Responsible For Today

The ontology graph does not currently own:

- Portfolio or hedge position writes.
- Thesis markdown writes, thesis status changes, or evaluation persistence.
- Catalyst, kill condition, action item, watch trigger, research note, recommendation, workflow, approval, or audit lifecycles.
- Immediate consistency after backing-store writes.
- Direct user or agent mutation of graph nodes as durable operational objects.
- A full replacement for `portfolio_db`, `thesis_db`, `core_db`, or market module schemas.

Keep public API names and existing `ontology` route/tool names stable unless there is a separate compatibility plan.

## If The Ontology Becomes Authoritative Later

Making the ontology graph authoritative would be a separate architecture migration. It would need:

- Explicit ownership decisions for each entity type and relation.
- Durable graph write APIs with validation, conflict handling, idempotency, and transaction boundaries.
- A versioning and migration plan for graph schemas and backing-store data.
- Backfill and reconciliation jobs from current SQLite/Postgres and file-backed state.
- Eventing or cache invalidation so graph writes and derived views stay consistent.
- Authorization and approval semantics at graph-object and graph-relation granularity.
- A compatibility plan for existing routers, agent tools, UI calls, tests, and stored data.

Until that migration exists, contributors should treat ontology runs as materialized semantic/risk snapshots over canonical operational state.
