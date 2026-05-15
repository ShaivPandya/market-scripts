# ADR-003: Authoritative Data Vendors and Licensing

**Status:** Accepted
**Owner:** Shaiv Pandya
**Date:** 2026-05-14
**Revisit trigger:** A new asset class, geography, or data quality requirement demands a vendor change; or a current vendor changes licensing terms.

## Context

Talisman ingests market, macro, fundamental, and news data from multiple sources. The current adapter set includes FRED, Yahoo Finance, EDGAR/SEC, EIA, Eurostat, SODA, and web scraping for select data. There is no formal registry of which source is authoritative for each data domain.

## Decision

Use **publicly available and freely licensed data sources** as the primary data layer. FRED is authoritative for US macro and rates. Yahoo Finance is the default for equity prices and fundamentals. EDGAR is authoritative for SEC filings. No paid terminal or enterprise data vendor (Bloomberg, Refinitiv, FactSet) is currently integrated.

## Alternatives Considered

| Alternative | Pros | Cons |
|-------------|------|------|
| Bloomberg Terminal API | Gold-standard market data, full asset coverage | Expensive, licensing restrictions on storage/redistribution, vendor lock-in |
| Refinitiv / LSEG | Broad coverage, good fixed income | Cost, complex onboarding |
| FactSet | Strong fundamentals and analytics | Cost, API complexity |
| Multiple free sources (current) | No cost, no licensing risk, full control | Gaps in coverage, freshness, and reliability; no SLA |

## Risks

- Free sources may have delayed, incomplete, or unreliable data.
- Yahoo Finance has no SLA and occasionally changes its API surface.
- Lack of a vendor SLA means data freshness targets (ADR-005) depend on best-effort scraping.

## References

- SHA-6 (source/vendor registry) would formalize the adapter list into an explicit registry.
- [README — Configure environment](../../../README.md#configure-environment) lists current API keys.
