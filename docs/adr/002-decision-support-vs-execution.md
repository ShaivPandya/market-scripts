# ADR-002: Decision-Support Only vs Broker/OMS Execution

**Status:** Accepted
**Owner:** Shaiv Pandya
**Date:** 2026-05-14
**Revisit trigger:** Integration with a prime broker, OMS, or execution management system is prioritized.

## Context

Talisman provides research, analytics, portfolio monitoring, thesis tracking, and AI-assisted recommendations. It does not place trades, connect to brokers, or manage order flow. The approval and action system records proposed actions and human approvals but does not execute them against live markets.

## Decision

Talisman is a **decision-support platform only**. It recommends, analyzes, and records — but does not execute trades. All execution happens outside the system through the owner's existing broker/OMS relationships.

## Alternatives Considered

| Alternative | Pros | Cons |
|-------------|------|------|
| Broker API integration (e.g., IBKR, FIX) | End-to-end workflow, faster execution | Regulatory exposure, operational risk, broker-specific coupling, settlement/reconciliation complexity |
| Paper trading / simulation execution | Safe testing of strategies | Still requires broker integration for live; adds maintenance without revenue impact |

## Risks

- Manual execution introduces latency between recommendation and action.
- No automated reconciliation between recommended and actual positions.
- If the system evolves toward autonomous agent actions, the no-execution boundary must be actively enforced.

## References

- SHA-20 (approval policy matrix), SHA-21 (dual-control approvals) define the decision-support approval model.
