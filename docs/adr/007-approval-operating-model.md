# ADR-007: Approval Operating Model

**Status:** Accepted
**Owner:** Shaiv Pandya
**Date:** 2026-05-14
**Revisit trigger:** Multiple users need independent approval authority; or the system begins executing actions that require dual-control sign-off.

## Context

Talisman tracks investment recommendations, proposed actions, and approvals through the ontology command service and workspace entities. The current model assumes a single owner who both generates and approves actions. There is no dual-control, quorum, or delegation workflow.

## Decision

**Single-owner approval model**. The owner is both the proposer and approver of all investment actions. Approvals are recorded for decision-quality tracking, not for separation-of-duties compliance. The agent can propose actions, but only the owner can approve them through the UI.

## Alternatives Considered

| Alternative | Pros | Cons |
|-------------|------|------|
| Dual-control (maker/checker) | Separation of duties, regulatory alignment | Requires multiple users, adds latency |
| Tiered auto-approval | Low-risk actions auto-approved, high-risk require manual | Risk of miscategorization, audit complexity |
| Single-owner (current) | Simple, fast, fits personal tool model | No independent check on decisions |

## Risks

- No independent check on the owner's decisions.
- If the agent proposes and the owner rubber-stamps, the approval record adds little decision quality value.

## References

- SHA-20 (configurable approval policy matrix), SHA-21 (dual-control workflows) are future options.
- [ontology/command_service](../../../ontology/command_service.py) manages the approval lifecycle.
