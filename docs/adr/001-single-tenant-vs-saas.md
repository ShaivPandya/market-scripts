# ADR-001: Single-Tenant Personal Tool vs Multi-Tenant SaaS

**Status:** Accepted
**Owner:** Shaiv Pandya
**Date:** 2026-05-14
**Revisit trigger:** Decision to onboard external users, investors, or team members beyond the owner.

## Context

Talisman began as a personal investment research and portfolio monitoring tool. The codebase assumes a single admin account for authentication, portfolio ownership, ontology writes, and approval authority. Scaling to multi-tenant SaaS would require per-tenant data isolation, identity federation, billing, and compliance boundaries.

## Decision

Remain a **single-tenant personal tool**. All infrastructure, state, and data belong to a single owner-operator. Authentication uses a single master password or Cloudflare Access identity. Multi-tenancy is not a near-term goal.

## Alternatives Considered

| Alternative | Pros | Cons |
|-------------|------|------|
| Multi-tenant SaaS | Revenue potential, broader user base | Major auth/isolation/compliance rework, billing infrastructure, support burden |
| Multi-user single-tenant | Team collaboration within one org | Requires RBAC, scoped portfolios, audit per-user — moderate rework |

## Risks

- If team members need access, the single-admin model becomes a bottleneck.
- No revenue path without tenancy or licensing model.

## References

- [README — Authentication And State](../../../README.md#authentication-and-state)
- SHA-10 (OIDC spike), SHA-11 (scope schema) are future options if this decision changes.
