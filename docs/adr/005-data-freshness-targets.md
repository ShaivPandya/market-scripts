# ADR-005: Data Freshness Targets

**Status:** Accepted
**Owner:** Shaiv Pandya
**Date:** 2026-05-14
**Revisit trigger:** A trading strategy requires intraday or real-time data; or a data source consistently fails to meet its target.

## Context

Different data domains have different freshness requirements. Market prices need to be reasonably current for portfolio valuation. Macro data updates on known schedules (monthly, quarterly). Fundamental data updates on earnings cycles. The system uses Cloud Scheduler to trigger periodic snapshot refreshes.

## Decision

Target freshness by domain:

| Domain | Target | Source |
|--------|--------|--------|
| Equity prices | End-of-day (15-20 min delayed intraday) | Yahoo Finance |
| Portfolio valuation | End-of-day | Derived from equity prices |
| US macro (GDP, CPI, employment) | Same-day after release | FRED |
| Rates / yield curve | End-of-day | FRED |
| SEC filings | Within 24h of EDGAR publication | EDGAR |
| Commodities | End-of-day | Yahoo Finance / EIA |
| FX rates | End-of-day | Yahoo Finance / FRED |
| News / sentiment | Best-effort, typically < 1h | Web sources |

These are **best-effort targets**, not SLAs. The system does not have real-time streaming data.

## Alternatives Considered

| Alternative | Pros | Cons |
|-------------|------|------|
| Real-time streaming (WebSocket feeds) | Sub-second freshness | Cost, complexity, infrastructure requirements |
| Paid delayed feeds (15-min) | Reliable, known latency | Vendor cost |
| End-of-day batch (current) | Simple, free, sufficient for position-level analysis | Not suitable for intraday trading |

## Risks

- Yahoo Finance delays or outages can cause stale portfolio valuations.
- No alerting when data falls below freshness targets (SHA-7 would address this).

## References

- SHA-7 (source freshness read model) would make freshness visible in the UI.
- Cloud Scheduler config in `infra/gcp/setup-scheduler.sh`.
