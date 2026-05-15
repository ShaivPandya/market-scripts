# ADR-008: Disaster Recovery Targets

**Status:** Accepted
**Owner:** Shaiv Pandya
**Date:** 2026-05-14
**Revisit trigger:** The platform manages time-sensitive capital allocation decisions where downtime has direct financial impact; or a data loss event occurs.

## Context

Talisman runs on Google Cloud (Cloud Run, Cloud SQL, Cloud Storage, Firebase Hosting). The production database contains portfolio state, ontology history, audit/provenance records, agent session history, and configuration. Research documents and generated reports are stored in Cloud Storage.

## Decision

**Best-effort recovery with daily Cloud SQL backups**.

| Target | Value |
|--------|-------|
| RPO (Recovery Point Objective) | ≤ 24 hours (Cloud SQL automated daily backups) |
| RTO (Recovery Time Objective) | ≤ 4 hours (redeploy from git + restore backup) |

No automated failover, no multi-region, no continuous WAL archival. The system is a personal tool — a few hours of downtime or up to a day of data loss is acceptable.

## Alternatives Considered

| Alternative | Pros | Cons |
|-------------|------|------|
| Multi-region active-active | Near-zero RPO/RTO | Extreme cost and complexity for a personal tool |
| Continuous WAL archival + point-in-time recovery | RPO < 5 min | Moderate cost, operational complexity |
| Daily automated backups (current) | Simple, GCP-managed, low cost | Up to 24h data loss |

## Risks

- A Cloud SQL failure between backups loses up to 24h of state changes.
- No tested restore procedure (SHA-36 would address this).
- Git contains code but not runtime state — portfolio positions, theses, and approvals are only in the database.

## References

- SHA-36 (disaster recovery restore drill) would validate the restore procedure.
- `infra/gcp/README.md` documents deployment and Cloud SQL setup.
