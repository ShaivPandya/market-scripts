# Authoritative Bitemporal Ontology Boundary

## Summary

The ontology is the authoritative operational object model for current Talisman domains. Portfolio, thesis, process, source, and derived snapshot state is represented as temporal ontology object and relation versions in Postgres.

This is a breaking Postgres-only architecture. SQLite-backed state paths are legacy migration inputs, not canonical stores for the temporal ontology.

## Ownership

`ontology/object_service.py` is the write boundary for operational objects and relations. Domain routes, workflow handlers, approval application, ingestion, and agent tools should call the object service instead of mutating `portfolio_db.py`, `thesis_db.py`, or `core_db.py` as canonical state.

The supported domain scope is the existing Talisman domain model:

- positions and hedge positions
- theses, thesis status, evaluations, catalysts, kill conditions, and thesis claims
- workflow runs, action items, watch triggers, research notes, recommendations, action runs, and action events
- approvals and approval-linked workflow artifacts
- source records and computed snapshots that support those objects

Orders, trades, fills, OMS objects, broker execution, mandates, and counterparties are intentionally out of scope.

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
4. Backfill current state from legacy stores and ontology snapshots into temporal versions.
5. Mark reconstructed rows with `temporal_confidence='backfilled'`.
6. Switch routes, workflow handlers, action application, ingestion, and agent tools to object-service writes.
7. Update UI surfaces to show or carry temporal metadata where relevant.
8. Remove or hard-disable legacy SQLite-backed state paths after tests pass.

Backfilled historical precision is limited by timestamps already present in the legacy data. Unknown or reconstructed timestamps should use the cutover time and explicit backfilled confidence.
