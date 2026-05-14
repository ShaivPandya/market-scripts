# ADR-009: Simulation and Backtesting Fidelity

**Status:** Accepted
**Owner:** Shaiv Pandya
**Date:** 2026-05-14
**Revisit trigger:** A strategy or model depends on simulation accuracy for capital allocation; or backtesting results diverge materially from live outcomes.

## Context

Talisman includes analytical tools (portfolio optimizer, sizer, hedging tool, DCF, signal aggregator) and research-grade backtesting (aluminum model, fundamental momentum). These tools use simplified assumptions: no transaction costs, no slippage, no market impact, no funding costs, and limited liquidity modeling.

## Decision

**Research-grade simulation fidelity**. Analytical tools are for research and decision-support, not for production trading signals. Backtests use simplified assumptions and are not calibrated for live execution.

| Aspect | Current Fidelity |
|--------|-----------------|
| Transaction costs | Not modeled |
| Slippage / market impact | Not modeled |
| Funding costs | Not modeled |
| Liquidity constraints | Not modeled |
| Rebalancing frequency | Assumed instantaneous |
| Data quality | End-of-day, delayed (ADR-005) |

## Alternatives Considered

| Alternative | Pros | Cons |
|-------------|------|------|
| Execution-grade simulation | Accurate P&L forecasting, strategy validation | Requires tick data, market microstructure modeling, significant engineering |
| Research-grade (current) | Simple, fast, sufficient for thesis validation | Results may not translate to live execution |

## Risks

- Backtesting results may overstate strategy performance due to missing costs and slippage.
- Owner may over-rely on simplified analytics for sizing decisions.

## References

- SHA-25 (scenario simulator API) would improve scenario analysis fidelity.
- Portfolio sizer and optimizer in `portfolio/`.
