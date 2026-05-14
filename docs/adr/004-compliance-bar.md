# ADR-004: Compliance and Regulatory Classification

**Status:** Accepted
**Owner:** Shaiv Pandya
**Date:** 2026-05-14
**Revisit trigger:** The platform is used by a regulated entity, manages external capital, or provides investment advice to third parties.

## Context

Talisman is a personal tool for the owner's own investment research. It does not manage external capital, provide investment advice to others, or operate as a registered investment adviser (RIA), broker-dealer, or fund administrator.

## Decision

The compliance bar is **personal/unregulated**. The system is not designed to meet SEC, FINRA, MiFID, or other regulatory requirements for investment advisers or fund managers. Audit trails, approvals, and provenance exist for the owner's own decision quality — not for regulatory reporting.

## Alternatives Considered

| Alternative | Pros | Cons |
|-------------|------|------|
| RIA-grade compliance (SEC Rule 206) | Enables managing external capital | Books & records, custody, advertising, compliance officer, annual audit |
| Fund-grade compliance (3(c)(1)/3(c)(7)) | LP capital, performance fees | Full fund admin, audit, custody, regulatory filings |
| Personal use (current) | No regulatory overhead | Cannot manage external capital or provide advice |

## Risks

- If the owner begins managing external capital (even informally), the system's audit trail may not meet regulatory standards.
- Provenance and approval records are designed for decision quality, not regulatory defensibility.

## References

- SHA-22 (approval revocation/rollback metadata) and SHA-20 (approval policy matrix) would move toward regulatory-grade audit if needed.
